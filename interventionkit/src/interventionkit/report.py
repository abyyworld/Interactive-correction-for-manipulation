"""Self-contained HTML/Markdown reporting. No plotting dependencies.

Charts are emitted as hand-built inline SVG. That is a deliberate constraint:
the point of this package is that someone can `pip install interventionkit` and
get a report out of their own interactive-learning run without dragging in
matplotlib, pandas or a browser toolchain. A report that only renders on the
author's machine is not a tool.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from .attribution import AttributionSummary, per_phase_breakdown
from .schema import EpisodeMeta

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 900px; padding: 2rem 1.25rem; }
h1 { font-size: 1.6rem; margin-bottom: .25rem; }
h2 { font-size: 1.15rem; margin-top: 2.25rem; border-bottom: 1px solid #8883; padding-bottom: .3rem; }
.sub { opacity: .65; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #8883; }
th { font-weight: 600; opacity: .8; }
td.num, th.num { text-align: right; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0; }
.card { flex: 1 1 150px; border: 1px solid #8884; border-radius: 8px; padding: .7rem .9rem; }
.card .v { font-size: 1.5rem; font-weight: 600; }
.card .k { font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em; }
.diag { background: #22c55e22; font-weight: 600; }
.off  { background: #ef444422; }
footer { margin-top: 3rem; font-size: .8rem; opacity: .6; }
"""


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:  # NaN
            return "-"
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.1f}%"


def _bar_svg(labels: list[str], values: list[float], width: int = 820, height: int = 190) -> str:
    """Horizontal-axis bar chart as inline SVG."""
    if not values:
        return ""
    pad_l, pad_b, pad_t = 46, 28, 12
    vmax = max(values) or 1.0
    n = len(values)
    plot_w = width - pad_l - 12
    plot_h = height - pad_b - pad_t
    bw = plot_w / n * 0.62
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for gy in (0.0, 0.5, 1.0):
        y = pad_t + plot_h * (1 - gy)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - 12}" y2="{y:.1f}" stroke="#8884" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="11" fill="currentColor" opacity=".6" text-anchor="end">{gy * vmax:.2f}</text>'
        )
    for i, (lab, v) in enumerate(zip(labels, values, strict=False)):
        cx = pad_l + plot_w * (i + 0.5) / n
        h = plot_h * (v / vmax) if vmax else 0
        parts.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{pad_t + plot_h - h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#3b82f6" rx="3"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t + plot_h - h - 5:.1f}" font-size="11" fill="currentColor" text-anchor="middle">{v:.2f}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 9}" font-size="11" fill="currentColor" opacity=".75" text-anchor="middle">{html.escape(lab)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _confusion_table(matrix: np.ndarray, names: list[str], caption: str) -> str:
    if matrix.size == 0 or matrix.sum() == 0:
        return f"<p><em>{html.escape(caption)}: no data.</em></p>"
    n = matrix.shape[0]
    names = (names + [f"phase{i}" for i in range(n)])[:n]
    head = "".join(f'<th class="num">{html.escape(x)}</th>' for x in names)
    rows = []
    for i in range(n):
        total = matrix[i].sum()
        cells = []
        for j in range(n):
            cls = "diag" if i == j else ("off" if matrix[i, j] else "")
            frac = (
                f"<br><span style='opacity:.6;font-size:.8em'>{100 * matrix[i, j] / total:.0f}%</span>"
                if total
                else ""
            )
            cells.append(f'<td class="num {cls}">{matrix[i, j]}{frac}</td>')
        rows.append(f"<tr><th>{html.escape(names[i])}</th>{''.join(cells)}</tr>")
    return (
        f"<p class='sub'>{html.escape(caption)}</p>"
        f"<table><thead><tr><th>true root \\ observed</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_report(
    summary: AttributionSummary,
    episodes: list[EpisodeMeta],
    *,
    title: str = "Intervention report",
    run_stats: dict[str, Any] | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    names = list(summary.phase_names) or ["approach", "grasp", "lift", "place"]
    stats = run_stats or {}

    cards = [
        ("episodes", str(summary.n_episodes)),
        ("success rate", _pct(stats.get("success_rate", float("nan")))),
        ("intervened", _pct(stats.get("intervention_rate", float("nan")))),
        ("steps corrected", _pct(stats.get("corrected_fraction", float("nan")))),
        ("onset misattribution", _pct(summary.onset_misattribution_rate)),
        ("stated misattribution", _pct(summary.stated_misattribution_rate)),
        ("mean detection lag", f"{_fmt(summary.mean_detection_lag, 1)} steps"),
        ("credit IoU", _fmt(summary.mean_credit_iou)),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'
        for k, v in cards
    )

    breakdown = per_phase_breakdown(episodes, n_phases=len(names))
    bd_rows = "".join(
        f"<tr><td>{html.escape(names[p])}</td><td class='num'>{int(v['n'])}</td>"
        f"<td class='num'>{_pct(v['misattribution_rate'])}</td>"
        f"<td class='num'>{_fmt(v['mean_detection_lag'], 1)}</td></tr>"
        for p, v in sorted(breakdown.items())
    )
    bd_table = (
        "<table><thead><tr><th>true root phase</th><th class='num'>n</th>"
        "<th class='num'>misattributed</th><th class='num'>detection lag (steps)</th>"
        "</tr></thead><tbody>" + bd_rows + "</tbody></table>"
        if bd_rows
        else "<p><em>No ground-truth-labelled episodes.</em></p>"
    )
    bd_chart = _bar_svg(
        [names[p] for p in sorted(breakdown)],
        [breakdown[p]["misattribution_rate"] for p in sorted(breakdown)],
    )

    extras = "".join(f"<h2>{html.escape(t)}</h2>{body}" for t, body in (extra_sections or []))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">Generated by interventionkit. Rates over episodes carrying ground-truth error labels
({summary.n_with_ground_truth} of {summary.n_episodes}).</p>
<div class="cards">{card_html}</div>

<h2>Misattribution by true root-cause phase</h2>
<p class="sub">Fraction of episodes where the phase the supervisor intervened in differs from the
phase the error originated in. Errors whose consequences are delayed are expected to be
misattributed more often.</p>
{bd_chart}
{bd_table}

<h2>Attribution confusion</h2>
{_confusion_table(summary.onset_confusion, names, "Implicit attribution: where the supervisor took over.")}
{_confusion_table(summary.stated_confusion, names, "Stated attribution: where the supervisor said the error was.")}

{extras}
<footer>interventionkit &middot; schema v1 &middot; {html.escape(str(summary.n_episodes))} episodes analysed</footer>
</body></html>"""


def build_markdown(summary: AttributionSummary, run_stats: dict[str, Any] | None = None) -> str:
    stats = run_stats or {}
    names = list(summary.phase_names) or ["approach", "grasp", "lift", "place"]
    lines = [
        "# Intervention report",
        "",
        f"- episodes: **{summary.n_episodes}** ({summary.n_with_ground_truth} with ground truth)",
        f"- success rate: **{_pct(stats.get('success_rate', float('nan')))}**",
        f"- episodes with an intervention: **{_pct(stats.get('intervention_rate', float('nan')))}**",
        f"- steps under correction: **{_pct(stats.get('corrected_fraction', float('nan')))}**",
        "",
        "## Attribution",
        "",
        f"- onset (implicit) misattribution rate: **{_pct(summary.onset_misattribution_rate)}**",
        f"- stated misattribution rate: **{_pct(summary.stated_misattribution_rate)}** (n={summary.n_stated})",
        f"- mean detection lag: **{_fmt(summary.mean_detection_lag, 1)}** steps",
        f"- mean credit-window IoU: **{_fmt(summary.mean_credit_iou)}**",
        f"- interventions after the point of no return: **{_pct(summary.late_intervention_rate)}**",
        "",
        "### Confusion (rows = true root phase, cols = intervention phase)",
        "",
        "| true \\ observed | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    m = summary.onset_confusion
    for i in range(min(m.shape[0], len(names))):
        lines.append(f"| {names[i]} | " + " | ".join(str(int(v)) for v in m[i]) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, content: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def dump_summary_json(path: str | Path, summary: AttributionSummary) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary.to_dict(), indent=2))
    return p
