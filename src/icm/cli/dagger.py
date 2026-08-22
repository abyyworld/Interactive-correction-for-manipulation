"""``icm-dagger`` - run the credit-assignment degradation experiment."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="icm-dagger", description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--collect", type=int, default=150, help="episodes collected per strategy")
    ap.add_argument("--eval", type=int, default=150, help="evaluation episodes per policy")
    ap.add_argument("--steps", type=int, default=6000, help="training steps per policy")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--trace-accuracy", type=float, default=0.35)
    ap.add_argument("--strategies", nargs="*", default=["onset", "symptom", "stated", "oracle"])
    ap.add_argument(
        "--no-equalise", action="store_true", help="do not match dataset sizes across strategies"
    )
    ap.add_argument(
        "--shared-demos",
        type=int,
        default=0,
        help="add this many fault-free demonstrations to every condition, holding "
        "initial-state coverage constant so that only the placement of the "
        "corrective states differs between strategies",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ..study.degradation import DegradationConfig, run_degradation_experiment
    from ..training.train_bc import TrainConfig
    from ..training.weighting import CreditAssignment

    cfg = DegradationConfig(
        strategies=tuple(CreditAssignment(s) for s in args.strategies),
        collect_episodes=args.collect,
        eval_episodes=args.eval,
        trace_accuracy=args.trace_accuracy,
        equalise_frames=not args.no_equalise,
        shared_demo_episodes=args.shared_demos,
        seed=args.seed,
        train=TrainConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            num_workers=0,
            device=args.device,
            log_every=max(1, args.steps // 4),
            eval_every=max(1, args.steps // 4),
            checkpoint_every=max(1, args.steps // 2),
        ),
    )
    report = run_degradation_experiment(args.out, cfg, progress=not args.quiet)

    if not args.quiet:
        print(f"\n{'strategy':<10}{'frames':>8}{'success':>10}{'95% CI':>18}")
        for name, r in report["results"].items():
            e = r["eval"]
            print(
                f"{name:<10}{r['frames']:>8}{e['success_rate']:>10.3f}"
                f"   [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}]"
            )
        print("\nresolved differences:")
        any_resolved = False
        for c in report["comparisons"]:
            if c["resolved"]:
                any_resolved = True
                print(f"  {c['a']} vs {c['b']}: {c['difference']:+.3f}")
        if not any_resolved:
            print("  none - the intervals all overlap at this sample size")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
