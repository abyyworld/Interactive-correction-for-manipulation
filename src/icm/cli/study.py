"""``icm-study`` - run the error-attribution study and write a report."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="icm-study", description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-n", "--episodes-per-fault", type=int, default=40)
    ap.add_argument("--controls", type=int, default=40, help="fault-free episodes")
    ap.add_argument(
        "--trace-accuracy",
        type=float,
        default=0.35,
        help="probability the supervisor traces the cause correctly",
    )
    ap.add_argument(
        "--sweep",
        nargs="*",
        type=float,
        default=None,
        help="sweep trace accuracy over these values instead of a single run",
    )
    ap.add_argument("--severity", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true", help="also write an HTML report")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ..study.attribution_study import StudyConfig, run_attribution_study, sweep_trace_accuracy

    cfg = StudyConfig(
        episodes_per_fault=args.episodes_per_fault,
        control_episodes=args.controls,
        trace_accuracy=args.trace_accuracy,
        severity=args.severity,
        seed=args.seed,
    )

    if args.sweep:
        results = sweep_trace_accuracy(args.out, tuple(args.sweep), cfg, progress=not args.quiet)
        if not args.quiet:
            print(f"\n{'trace_acc':<11}{'onset':>9}{'symptom':>10}{'stated':>9}")
            for acc, rep in results.items():
                d = rep["by_lag_class"].get("delayed", {})
                print(
                    f"{acc:<11}{d.get('onset_misattribution_rate', float('nan')):>9.3f}"
                    f"{d.get('symptom_misattribution_rate', float('nan')):>10.3f}"
                    f"{d.get('stated_misattribution_rate', float('nan')):>9.3f}"
                )
        return 0

    report = run_attribution_study(args.out, cfg, progress=not args.quiet)
    if not args.quiet:
        _print_report(report)

    if args.report:
        from interventionkit.report import build_report, write_report

        from interventionkit import RunReader, analyse

        reader = RunReader(args.out)
        episodes = reader.episodes()
        summary = analyse(episodes, n_phases=4, phase_names=("approach", "grasp", "lift", "place"))
        html = build_report(
            summary, episodes, title="Error attribution study", run_stats=reader.stats()
        )
        path = write_report(Path(args.out) / "report.html", html)
        if not args.quiet:
            print(f"wrote {path}")
    return 0


def _print_report(report: dict) -> None:
    print(f"\n{'lag class':<15}{'n':>5}{'onset':>9}{'symptom':>10}{'stated':>9}{'det.lag':>9}")
    for lag, v in report["by_lag_class"].items():
        print(
            f"{lag:<15}{v['n']:>5.0f}{v['onset_misattribution_rate']:>9.3f}"
            f"{v['symptom_misattribution_rate']:>10.3f}{v['stated_misattribution_rate']:>9.3f}"
            f"{v['mean_detection_lag']:>9.1f}"
        )
    print(f"\n{'fault':<18}{'detect':>8}{'rescued':>9}{'takeover':>10}")
    for k, v in report["per_fault"].items():
        print(
            f"{k:<18}{v['detection_rate']:>8.2f}{v['success_rate_with_correction']:>9.2f}"
            f"{v['mean_takeover_step']:>10.1f}"
        )
    print(
        f"\nunnecessary intervention rate (no fault present): "
        f"{report['unnecessary_intervention_rate']:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
