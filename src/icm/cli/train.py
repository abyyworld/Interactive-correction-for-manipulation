"""``icm-train`` - train a behaviour-cloning policy from recorded episodes.

Runs as its own process by design. Training imports PyTorch and nothing else;
the simulator is never loaded. Besides being cleaner, this sidesteps a real
crash: on some CPU-only setups, MuJoCo and PyTorch's MKL kernels segfault when
an optimiser steps in a process where both are loaded (observed reproducibly on
a headless x86 container with torch 2.13 CPU). Collection and training are
separate stages that communicate through the dataset on disk, so keeping them in
separate processes costs nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="icm-train", description=__doc__.split("\n")[0])
    ap.add_argument("data", nargs="+", help="one or more interventionkit run directories")
    ap.add_argument("-o", "--out", required=True, help="output directory for checkpoints and metrics")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk", type=int, default=8, help="action chunk length")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--state-key", default="proprio", choices=["proprio", "privileged"],
                    help="privileged uses ground-truth object poses: a state-based upper bound, "
                         "not a deployable policy")
    ap.add_argument("--images", nargs="*", default=[],
                    help="image observation keys, e.g. wrist_rgb scene_rgb")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "small"])
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights (needs network)")
    ap.add_argument("--image-size", type=int, default=84)
    ap.add_argument("--supervision", default="corrections", choices=["corrections", "all", "demos"])
    ap.add_argument("--credit", default="onset", choices=["onset", "symptom", "stated", "oracle"])
    ap.add_argument("--subsample", type=int, default=0,
                    help="cap training frames, for size-controlled comparisons (0 = all)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline end to end")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ..policies.bc import PolicyConfig
    from ..training.dataset import DatasetConfig
    from ..training.train_bc import TrainConfig, train_bc
    from ..training.weighting import CreditAssignment

    if args.smoke:
        args.steps = min(args.steps, 100)
        args.batch_size = min(args.batch_size, 16)
        args.workers = 0
        args.eval_every = args.checkpoint_every = args.log_every = 50

    dcfg = DatasetConfig(
        supervision=args.supervision,
        credit=CreditAssignment(args.credit),
        chunk=args.chunk,
        image_keys=tuple(args.images),
        state_key=args.state_key,
    )
    pcfg = PolicyConfig(
        chunk=args.chunk,
        image_keys=tuple(args.images),
        image_size=args.image_size,
        backbone=args.backbone,
        pretrained=args.pretrained,
    )
    tcfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, num_workers=args.workers,
        seed=args.seed, device=args.device, val_fraction=args.val_fraction,
        log_every=args.log_every, eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )

    if args.subsample > 0:
        _patch_subsample(args.subsample, args.seed)

    summary = train_bc(
        args.data, args.out, pcfg, dcfg, tcfg, resume=args.resume, progress=not args.quiet
    )
    if not args.quiet:
        print(json.dumps(summary, indent=2, default=float))
    Path(args.out, "done.json").write_text(json.dumps(summary, default=float))
    return 0


def _patch_subsample(n: int, seed: int) -> None:
    """Cap the dataset size for size-controlled comparisons."""
    import icm.training.train_bc as tb

    original = tb.InterventionDataset

    class Budgeted(original):  # type: ignore[misc,valid-type]
        def __init__(self, roots, config=None):
            super().__init__(roots, config)
            self.subsample(n, seed=seed)

    tb.InterventionDataset = Budgeted


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
