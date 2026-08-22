"""Does misattribution actually damage the trained policy, and by how much?

The first half of the project measures *whether* supervisors attribute errors to
the wrong phase. This half measures what that costs.

Design
------
The agent under supervision is the scripted expert with an injected fault - a
stand-in for a policy with a systematic error, chosen because its failure mode is
known exactly. The supervisor intervenes, names a phase, the episode is rewound
according to the credit-assignment strategy, and a correction is demonstrated.
A behaviour-cloning policy is then trained on the corrections alone and evaluated
on fresh episodes.

Only the rewind target differs between conditions. Episode seeds, faults,
supervisor and training hyperparameters are identical, so any difference in final
success rate is attributable to where the correction was applied.

The controlled comparison
-------------------------
Rewinding further necessarily produces more corrective frames, so ORACLE has both
better-placed *and* more data. Reporting only that comparison would confound the
two. The ``equalise_frames`` condition subsamples every dataset to the size of
the smallest, which isolates *where* the correction landed from *how much* of it
there was.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..control.scripted import ExpertConfig, ScriptedExpert
from ..envs.faults import DELAYED_FAULTS, IMMEDIATE_FAULTS, FaultInjector, FaultType, sample_fault
from ..envs.pick_place import EnvConfig, PickPlaceEnv
from ..eval.metrics import evaluate_agent
from ..eval.rollout import ScriptedAgent
from ..policies.runner import PolicyAgent, RunnerConfig
from ..study.supervisor import SupervisorConfig, SyntheticSupervisor
from ..training.dagger import PHASE_NAMES, collect_round
from ..training.dataset import DatasetConfig, InterventionDataset
from ..training.train_bc import TrainConfig, load_policy
from ..training.weighting import CreditAssignment


@dataclass
class DegradationConfig:
    strategies: tuple[CreditAssignment, ...] = (
        CreditAssignment.ONSET,
        CreditAssignment.SYMPTOM,
        CreditAssignment.STATED,
        CreditAssignment.ORACLE,
    )
    collect_episodes: int = 150
    eval_episodes: int = 150
    trace_accuracy: float = 0.35
    faults: tuple[FaultType, ...] = DELAYED_FAULTS + IMMEDIATE_FAULTS
    severity_range: tuple[float, float] = (0.8, 1.0)
    equalise_frames: bool = True
    #: When > 0, collect this many fault-free expert demonstrations once and add
    #: them, unchanged, to every condition's training set.
    #:
    #: This is the control for the confound the first run exposed. Rewinding
    #: changes *where the corrective demonstration starts*, and by the time a
    #: supervisor reacts the world has usually settled back into APPROACH - so
    #: the shallowest rewind incidentally supplies the best coverage of the state
    #: distribution every evaluation episode begins in. That coverage effect
    #: swamped the attribution effect. Giving every condition an identical demo
    #: pool holds coverage constant, leaving the placement of the corrective
    #: states as the only thing that varies.
    shared_demo_episodes: int = 0
    seed: int = 0
    state_key: str = "privileged"
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            steps=4000,
            batch_size=128,
            num_workers=0,
            eval_every=1000,
            checkpoint_every=2000,
            log_every=500,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strategies"] = [s.value for s in self.strategies]
        d["faults"] = [f.value for f in self.faults]
        return d


def run_degradation_experiment(
    out_root: str | Path,
    config: DegradationConfig | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = config or DegradationConfig()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    env = PickPlaceEnv(EnvConfig(render_images=False), seed=cfg.seed)
    expert_cfg = ExpertConfig()

    shared_demos: Path | None = None
    if cfg.shared_demo_episodes > 0:
        shared_demos = out_root / "data_shared_demos"
        if not (shared_demos / "run.json").is_file():
            if progress:
                print(
                    f"[demos] collecting {cfg.shared_demo_episodes} fault-free demonstrations",
                    flush=True,
                )
            _collect_demos(env, shared_demos, cfg, expert_cfg)

    # Identical episodes across conditions: same seeds, same faults, same
    # supervisor draw. Only the rewind target changes.
    def make_agent_factory(round_seed: int):
        def make_agent(i: int):
            rng = np.random.default_rng(round_seed * 10_000 + i)
            spec = sample_fault(rng, types=cfg.faults, severity_range=cfg.severity_range)
            return ScriptedAgent(ScriptedExpert(expert_cfg), FaultInjector(spec, rng))

        return make_agent

    collected: dict[str, tuple[Path, Any]] = {}
    for strategy in cfg.strategies:
        # Reuse an existing collection rather than duplicating it. Collection is
        # the slowest stage and is fully determined by the seed, so re-running it
        # would produce identical episodes at real cost - and appending into the
        # same directory would silently double every dataset.
        existing = out_root / f"data_{strategy.value}"
        if (existing / "run.json").is_file():
            summary_path = existing / "collection.json"
            saved = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            if progress:
                print(f"[collect] {strategy.value}: reusing {existing}", flush=True)
            collected[strategy.value] = (existing, _ReusedCollection(saved))
            continue
        if progress:
            print(f"[collect] {strategy.value}", flush=True)
        supervisor = SyntheticSupervisor(
            SupervisorConfig(trace_accuracy=cfg.trace_accuracy),
            expert_cfg,
            rng=np.random.default_rng(cfg.seed + 7),
        )
        path, summary = collect_round(
            env,
            make_agent_factory(cfg.seed),
            supervisor,
            out_root / f"data_{strategy.value}",
            n_episodes=cfg.collect_episodes,
            strategy=strategy,
            seed=cfg.seed,
            record_keys=("proprio", "privileged"),
            expert_config=expert_cfg,
        )
        collected[strategy.value] = (path, summary)
        if progress:
            print(f"           {summary.to_dict()}", flush=True)

    # Frame budget shared by every condition, so coverage is compared, not volume.
    sizes = {}
    for name, (path, _) in collected.items():
        ds = InterventionDataset(
            path,
            DatasetConfig(
                supervision="corrections", credit=CreditAssignment.ONSET, state_key=cfg.state_key
            ),
        )
        sizes[name] = len(ds)
    budget = min(sizes.values()) if cfg.equalise_frames and sizes else None
    if progress:
        print(f"[frames] {sizes}  budget={budget}", flush=True)

    results: dict[str, Any] = {}
    for strategy in cfg.strategies:
        name = strategy.value
        path, coll = collected[name]
        run_dir = out_root / f"train_{name}"
        dcfg = DatasetConfig(
            supervision="corrections", credit=CreditAssignment.ONSET, state_key=cfg.state_key
        )
        # credit=ONSET here on purpose: the rewind already decided which steps
        # carry corrective actions, so the dataset must take the recording at
        # face value rather than re-deriving a span.
        if budget is not None:
            ds = InterventionDataset(path, dcfg)
            ds.subsample(budget, seed=cfg.seed)
            frames_used = len(ds)
        else:
            frames_used = sizes[name]

        if progress:
            print(
                f"[train] {name} on {frames_used} correction frames"
                + (" + shared demos" if shared_demos else ""),
                flush=True,
            )
        train_summary = _train_with_budget(
            path, run_dir, dcfg, cfg, budget, shared_demos=shared_demos
        )

        policy = load_policy(run_dir / "checkpoint.pt", device=cfg.train.resolve_device())
        agent = PolicyAgent(
            policy, RunnerConfig(state_key=cfg.state_key, device=str(cfg.train.resolve_device()))
        )
        if progress:
            print(f"[eval] {name}", flush=True)
        ev = evaluate_agent(env, agent, n_episodes=cfg.eval_episodes, seed=cfg.seed + 999)
        results[name] = {
            "collection": coll.to_dict(),
            "frames": frames_used,
            "frames_available": sizes[name],
            "train": train_summary,
            "eval": ev.to_dict(),
            "failure_breakdown": ev.failure_breakdown(),
        }
        if progress:
            print(f"        {name}: {ev}", flush=True)

    env.close()
    report = {"config": cfg.to_dict(), "results": results, "comparisons": _pairwise(results)}
    (out_root / "degradation.json").write_text(json.dumps(report, indent=2, default=float))
    return report


class _ReusedCollection:
    """Stands in for a CollectionSummary when a prior collection is reused."""

    def __init__(self, saved: dict[str, Any]):
        self._saved = saved

    def to_dict(self) -> dict[str, Any]:
        return {**self._saved, "reused": True}


def _collect_demos(env, out_dir: Path, cfg, expert_cfg) -> Path:
    """Fault-free expert demonstrations, shared unchanged by every condition."""
    from interventionkit import InterventionRecorder

    from ..eval.rollout import rollout

    recorder = InterventionRecorder(
        out_dir,
        task="pick_place",
        phase_names=PHASE_NAMES,
        config={"source": "scripted_expert", "role": "shared coverage pool"},
    )
    agent = ScriptedAgent(ScriptedExpert(expert_cfg))
    for i in range(cfg.shared_demo_episodes):
        seed = 500_000 + cfg.seed * 1000 + i
        with recorder.episode(seed=seed, instruction=env.default_instruction()) as ep:
            rollout(
                env,
                agent,
                recorder=ep,
                seed=seed,
                record_keys=("proprio", "privileged"),
                record_agent_as="expert",
            )
    return out_dir


def _train_with_budget(data_path, run_dir, dcfg, cfg, budget, shared_demos=None):
    """Train in a subprocess.

    Training deliberately runs in a process that never imports MuJoCo. Besides
    being a clean stage boundary, it avoids a reproducible segfault: on CPU-only
    setups an optimiser step crashes when MuJoCo and PyTorch's MKL kernels are
    loaded in the same process. Collection and training already communicate
    through the dataset on disk, so the split costs nothing.
    """
    import subprocess
    import sys

    # Root order matters: --cap-per-root is positional, so the corrections must
    # come first and the shared demo pool second.
    roots = [str(data_path)] + ([str(shared_demos)] if shared_demos is not None else [])
    cmd = [
        sys.executable,
        "-m",
        "icm.cli.train",
        *roots,
        "-o",
        str(run_dir),
        "--steps",
        str(cfg.train.steps),
        "--batch-size",
        str(cfg.train.batch_size),
        "--lr",
        str(cfg.train.lr),
        "--chunk",
        str(dcfg.chunk),
        "--device",
        cfg.train.device,
        "--workers",
        str(cfg.train.num_workers),
        "--seed",
        str(cfg.seed),
        "--state-key",
        cfg.state_key,
        "--supervision",
        dcfg.supervision,
        "--credit",
        dcfg.credit.value,
        "--val-fraction",
        str(cfg.train.val_fraction),
        "--log-every",
        str(cfg.train.log_every),
        "--eval-every",
        str(cfg.train.eval_every),
        "--checkpoint-every",
        str(cfg.train.checkpoint_every),
        "--quiet",
    ]
    if shared_demos is not None:
        # Cap only the corrections; the demo pool is identical everywhere by
        # design, so capping it too would reintroduce the difference being
        # controlled for.
        cmd += ["--cap-per-root", str(budget if budget is not None else -1), "-1"]
    elif budget is not None:
        cmd += ["--subsample", str(budget)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"training subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    return json.loads((Path(run_dir) / "done.json").read_text())


def _pairwise(results: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(results)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ea, eb = results[a]["eval"], results[b]["eval"]
            lo_a, hi_a = ea["ci95_low"], ea["ci95_high"]
            lo_b, hi_b = eb["ci95_low"], eb["ci95_high"]
            overlap = not (hi_a < lo_b or hi_b < lo_a)
            out.append(
                {
                    "a": a,
                    "b": b,
                    "success_a": ea["success_rate"],
                    "success_b": eb["success_rate"],
                    "difference": ea["success_rate"] - eb["success_rate"],
                    "intervals_overlap": overlap,
                    "resolved": not overlap,
                }
            )
    return out
