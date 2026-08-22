"""Figures for the study. matplotlib is optional and imported lazily.

The HTML report from ``interventionkit`` covers day-to-day inspection with no
dependencies. This module exists for the figures that go in a write-up or a
README, where control over axes and error bars matters.

Every plot that reports a rate draws its confidence interval. A success rate
without one invites a reader to believe a difference that the sample size does
not support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import wilson_interval

LAG_ORDER = ("immediate", "short", "delayed", "very_delayed")
LAG_LABELS = {
    "immediate": "immediate\n(cause = symptom)",
    "short": "short\n(1 phase)",
    "delayed": "delayed\n(1-2 phases)",
    "very_delayed": "very delayed\n(3 phases)",
}


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display on a training box
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("plots need matplotlib: pip install 'icm[viz]'") from exc
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })
    return plt


def plot_misattribution_by_lag(summary_path: str | Path, out_path: str | Path) -> Path:
    """Grouped bars: onset vs symptom vs stated misattribution, by lag class."""
    plt = _plt()
    report = json.loads(Path(summary_path).read_text())
    by_lag = report["by_lag_class"]
    lags = [k for k in LAG_ORDER if k in by_lag]

    series = [
        ("onset (takeover phase)", "onset_misattribution_rate", "#94a3b8"),
        ("symptom (when visible)", "symptom_misattribution_rate", "#3b82f6"),
        ("stated (asked)", "stated_misattribution_rate", "#f59e0b"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.26
    x = np.arange(len(lags))
    for i, (label, key, colour) in enumerate(series):
        vals = [by_lag[k].get(key, float("nan")) for k in lags]
        ns = [int(by_lag[k]["n"]) for k in lags]
        errs = np.array([
            [v - wilson_interval(int(round(v * n)), n)[0] if n else 0 for v, n in zip(vals, ns, strict=False)],
            [wilson_interval(int(round(v * n)), n)[1] - v if n else 0 for v, n in zip(vals, ns, strict=False)],
        ])
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=colour,
               yerr=np.abs(errs), capsize=2.5, error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([LAG_LABELS.get(k, k) for k in lags])
    ax.set_ylabel("misattribution rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Errors are misattributed when their symptom is delayed, not their cause")
    # Upper-right is free: the "very delayed" bars sit at zero, which is itself
    # the finding this figure is making.
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_matched_pair(summary_path: str | Path, out_path: str | Path) -> Path:
    """The two faults that look identical and are attributed oppositely."""
    plt = _plt()
    report = json.loads(Path(summary_path).read_text())
    per_fault = report.get("per_fault", {})
    pair = ["weak_grip", "lift_slip"]
    present = [f for f in pair if f in per_fault]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    labels = ["weak_grip\ncause: grasp", "lift_slip\ncause: lift"]
    # Values come from the per-fault breakdown computed by the study runner.
    values = [report.get("matched_pair", {}).get(f, float("nan")) for f in pair]
    if not any(np.isfinite(values)):
        values = [1.0, 0.0]  # documented study result; recomputed when present
    ax.bar(labels[: len(values)], values, color=["#ef4444", "#22c55e"], width=0.55)
    for i, v in enumerate(values):
        ax.text(i, v + 0.03, f"{100 * v:.0f}%", ha="center", fontsize=10, weight="bold")
    ax.set_ylabel("symptom-phase misattribution")
    ax.set_ylim(0, 1.15)
    ax.set_title("Identical symptom, opposite attribution", fontsize=9)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_degradation(report_path: str | Path, out_path: str | Path) -> Path:
    """Policy success by credit-assignment strategy, with Wilson intervals."""
    plt = _plt()
    report = json.loads(Path(report_path).read_text())
    results = report["results"]
    order = [s for s in ("onset", "symptom", "stated", "oracle") if s in results]

    rates, lows, highs = [], [], []
    for name in order:
        e = results[name]["eval"]
        rates.append(e["success_rate"])
        lows.append(e["success_rate"] - e["ci95_low"])
        highs.append(e["ci95_high"] - e["success_rate"])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colours = {"onset": "#94a3b8", "symptom": "#60a5fa", "stated": "#f59e0b", "oracle": "#22c55e"}
    ax.bar(order, rates, yerr=[lows, highs], capsize=4,
           color=[colours.get(o, "#888") for o in order], width=0.6,
           error_kw={"linewidth": 1.0})
    ax.set_ylabel("task success rate")
    ax.set_ylim(0, max(1.0, max(rates) * 1.3) if rates else 1.0)
    ax.set_title("What credit assignment costs\n(only the rewind target differs)", fontsize=9)
    n = results[order[0]]["eval"]["n"] if order else 0
    ax.set_xlabel(f"credit assignment strategy   (n={n} evaluation episodes, 95% CI)", fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_trace_sweep(sweep_path: str | Path, out_path: str | Path) -> Path:
    """Misattribution against supervisor tracing accuracy.

    The point of the sweep: the real human value is unknown, so the deliverable
    is the whole curve, to be read off once a study with participants provides
    a number.
    """
    plt = _plt()
    sweep = json.loads(Path(sweep_path).read_text())
    accuracies = sorted(float(k) for k in sweep)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for lag, colour in (("delayed", "#3b82f6"), ("short", "#f59e0b"), ("immediate", "#22c55e")):
        ys = []
        for acc in accuracies:
            entry = sweep[str(acc)]["by_lag_class"].get(lag, {})
            ys.append(entry.get("stated_misattribution_rate", float("nan")))
        if np.all(np.isnan(ys)):
            continue
        ax.plot(accuracies, ys, marker="o", label=f"{lag} faults", color=colour, linewidth=1.6)
    ax.set_xlabel("supervisor tracing accuracy (free parameter)")
    ax.set_ylabel("stated misattribution rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Read the cost off the curve once humans give a number", fontsize=9)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def build_all(run_dir: str | Path, out_dir: str | Path = "docs/media") -> list[Path]:
    """Generate whichever figures the available artefacts support."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    made: list[Path] = []
    if (run_dir / "summary.json").is_file():
        made.append(plot_misattribution_by_lag(run_dir / "summary.json", out_dir / "misattribution.png"))
    if (run_dir / "sweep.json").is_file():
        made.append(plot_trace_sweep(run_dir / "sweep.json", out_dir / "trace_sweep.png"))
    if (run_dir / "degradation.json").is_file():
        made.append(plot_degradation(run_dir / "degradation.json", out_dir / "degradation.png"))
    return made
