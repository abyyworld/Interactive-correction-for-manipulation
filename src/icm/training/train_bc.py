"""Behaviour-cloning trainer: resumable, streaming, and quiet about it.

Operational choices are shaped by the target setup — a GPU box left running at
home, driven over SSH:

* **Checkpoint on an interval, resume from one flag.** A dropped connection or a
  reboot should cost minutes, not a run.
* **Metrics as JSON Lines.** Monitoring a run from another machine is then
  ``tail -f``, with no TensorBoard install, no port forwarding and no daemon.
* **Validation split by episode, never by frame.** Frames inside one episode are
  highly correlated; splitting by frame leaks almost-identical neighbouring
  states into validation and reports a loss far better than the truth.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..policies.bc import BCPolicy, PolicyConfig
from .dataset import DatasetConfig, InterventionDataset


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    num_workers: int = 2
    val_fraction: float = 0.1
    log_every: int = 50
    eval_every: int = 1000
    checkpoint_every: int = 1000
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    compile: bool = False

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # Apple silicon
            return torch.device("mps")
        return torch.device("cpu")


class TorchFrames(Dataset):
    """Adapts the numpy dataset to PyTorch, restricted to a set of indices."""

    def __init__(self, base: InterventionDataset, indices: np.ndarray, vocab: tuple[str, ...] = ()):
        self.base = base
        self.indices = indices
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sample = self.base[int(self.indices[i])]
        out = {k: torch.as_tensor(v) for k, v in sample.items()}
        if self.vocab:
            out["instruction"] = torch.zeros(len(self.vocab))
        return out


def episode_split(
    base: InterventionDataset, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by episode so no episode contributes to both sides."""
    episodes = sorted({(ri, eid) for ri, eid, _, _ in base.samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_fraction)) if len(episodes) > 1 else 0
    val_set = set(episodes[:n_val])
    train_idx, val_idx = [], []
    for i, (ri, eid, _, _) in enumerate(base.samples):
        (val_idx if (ri, eid) in val_set else train_idx).append(i)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class JsonlLogger:
    """Append-only metrics log. Readable with tail -f from anywhere."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")


def train_bc(
    data_roots,
    out_dir: str | Path,
    policy_config: PolicyConfig | None = None,
    data_config: DatasetConfig | None = None,
    train_config: TrainConfig | None = None,
    resume: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = train_config or TrainConfig()
    dcfg = data_config or DatasetConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    base = InterventionDataset(data_roots, dcfg)
    if len(base) == 0:
        raise ValueError(
            "dataset is empty. With supervision='corrections' this usually means no "
            "episode contained an intervention."
        )
    pcfg = policy_config or PolicyConfig(chunk=dcfg.chunk)
    sample = base[0]
    pcfg.state_dim = int(sample["state"].shape[0])
    pcfg.chunk = dcfg.chunk

    train_idx, val_idx = episode_split(base, cfg.val_fraction, cfg.seed)
    train_ds = TorchFrames(base, train_idx, pcfg.vocab)
    val_ds = TorchFrames(base, val_idx, pcfg.vocab) if len(val_idx) else None

    device = cfg.resolve_device()
    # Workers each hold their own shard cache, so keep the count modest: this is
    # a memory-bound pipeline, not a CPU-bound one.
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(train_ds) > cfg.batch_size,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    policy = BCPolicy(pcfg)
    state_mean, state_std = base.state_stats()
    policy.set_normalization(state_mean, state_std)
    policy = policy.to(device)
    if cfg.compile and hasattr(torch, "compile"):
        policy = torch.compile(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger = JsonlLogger(out_dir / "metrics.jsonl")
    ckpt_path = out_dir / "checkpoint.pt"
    start_step = 0
    if resume and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(state["policy"])
        opt.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        if progress:
            print(f"[train] resumed from step {start_step}")

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "policy": {
                    **asdict(pcfg),
                    "image_keys": list(pcfg.image_keys),
                    "vocab": list(pcfg.vocab),
                },
                "data": {
                    **asdict(dcfg),
                    "credit": dcfg.credit.value,
                    "image_keys": list(dcfg.image_keys),
                },
                "train": asdict(cfg),
                "dataset": base.summary(),
            },
            indent=2,
            default=str,
        )
    )

    def batches():
        while True:
            yield from loader

    stream = batches()
    policy.train()
    t0 = time.time()
    running: list[float] = []
    best_val = float("inf")

    for step in range(start_step, cfg.steps):
        batch = {k: v.to(device, non_blocking=True) for k, v in next(stream).items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, metrics = policy.loss(batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        running.append(metrics["loss"])

        if (step + 1) % cfg.log_every == 0:
            rec = {
                "step": step + 1,
                "loss": float(np.mean(running)),
                "lr": lr_at(step, cfg),
                "elapsed_s": time.time() - t0,
                "pos_l1": metrics["pos_l1"],
                "gripper_l1": metrics["gripper_l1"],
            }
            logger.log(rec)
            if progress:
                print(
                    f"  step {step + 1:>6}  loss {rec['loss']:.4f}  lr {rec['lr']:.2e}", flush=True
                )
            running = []

        if val_loader is not None and (step + 1) % cfg.eval_every == 0:
            val = evaluate_loss(policy, val_loader, device)
            logger.log({"step": step + 1, "val_loss": val})
            best_val = min(best_val, val)
            if progress:
                print(f"  step {step + 1:>6}  val {val:.4f}", flush=True)
            policy.train()

        if (step + 1) % cfg.checkpoint_every == 0 or step + 1 == cfg.steps:
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step + 1,
                    "policy_config": asdict(pcfg),
                },
                ckpt_path,
            )

    final_val = (
        evaluate_loss(policy, val_loader, device) if val_loader is not None else float("nan")
    )
    summary = {
        "steps": cfg.steps,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "train_frames": len(train_ds),
        "val_frames": len(val_ds) if val_ds else 0,
        "dataset": base.summary(),
        "device": str(device),
        "wall_time_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


@torch.no_grad()
def evaluate_loss(policy, loader, device) -> float:
    policy.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = policy.loss(batch)
        total += float(loss) * len(batch["action"])
        n += len(batch["action"])
    policy.train()
    return total / max(n, 1)


def load_policy(checkpoint: str | Path, device: str | torch.device = "cpu") -> BCPolicy:
    """Rebuild a policy from a checkpoint, config included."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = PolicyConfig(
        **{
            **state["policy_config"],
            "image_keys": tuple(state["policy_config"]["image_keys"]),
            "vocab": tuple(state["policy_config"].get("vocab", ())),
            "hidden": tuple(state["policy_config"]["hidden"]),
        }
    )
    policy = BCPolicy(cfg)
    policy.load_state_dict(state["policy"])
    policy.to(device).eval()
    return policy
