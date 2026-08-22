"""Command line entry points: ``ik-report`` and ``ik-inspect``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attribution import analyse
from .report import build_markdown, build_report, dump_summary_json, write_report
from .store import RunReader


def _load(root: str) -> RunReader:
    try:
        return RunReader(root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def report_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ik-report", description="Build an intervention report from a run."
    )
    ap.add_argument("run", help="run directory containing run.json")
    ap.add_argument(
        "-o", "--out", default=None, help="output path (.html or .md); default <run>/report.html"
    )
    ap.add_argument("--json", default=None, help="also write the raw summary as JSON")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    reader = _load(args.run)
    names = tuple(reader.phase_names) or ("approach", "grasp", "lift", "place")
    episodes = reader.episodes()
    summary = analyse(episodes, n_phases=len(names), phase_names=names)
    stats = reader.stats()

    out = Path(args.out) if args.out else Path(args.run) / "report.html"
    title = args.title or f"Intervention report - {reader.meta.run_id}"
    content = (
        build_markdown(summary, stats)
        if out.suffix.lower() in (".md", ".markdown")
        else build_report(summary, episodes, title=title, run_stats=stats)
    )
    write_report(out, content)
    print(f"wrote {out}")
    if args.json:
        dump_summary_json(args.json, summary)
        print(f"wrote {args.json}")
    return 0


def inspect_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ik-inspect", description="Summarise a run on the terminal.")
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=0, help="also list the first N episodes")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    reader = _load(args.run)
    names = tuple(reader.phase_names) or ("approach", "grasp", "lift", "place")
    stats = reader.stats()
    summary = analyse(reader.episodes(), n_phases=len(names), phase_names=names)

    if args.json:
        print(json.dumps({"stats": stats, "attribution": summary.to_dict()}, indent=2))
        return 0

    print(
        f"run       : {reader.meta.run_id}  (task={reader.meta.task}, schema v{reader.meta.schema_version})"
    )
    print(f"created   : {reader.meta.created_utc}")
    for k, v in stats.items():
        print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")
    print("attribution:")
    print(f"  onset misattribution   {summary.onset_misattribution_rate:.3f}")
    print(
        f"  stated misattribution  {summary.stated_misattribution_rate:.3f} (n={summary.n_stated})"
    )
    print(f"  mean detection lag     {summary.mean_detection_lag:.1f} steps")
    print(f"  mean credit IoU        {summary.mean_credit_iou:.3f}")

    if args.episodes:
        print(f"\nfirst {args.episodes} episodes:")
        for ep in reader.episodes()[: args.episodes]:
            segs = ", ".join(
                f"{s.start}-{s.end}@{s.onset_phase_name or s.onset_phase}" for s in ep.interventions
            )
            print(
                f"  {ep.episode_id}  steps={ep.n_steps:<4} success={str(ep.success):<5} interventions=[{segs}]"
            )
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    return report_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(report_main())
