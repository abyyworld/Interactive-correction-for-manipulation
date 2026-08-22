"""The figures are part of the deliverable, so a broken one should fail CI.

These do not check what the plots look like — only that each builder consumes a
report of the shape the study actually writes and produces a file. That is
enough to catch the failure mode that matters here: a schema drift in the
report silently breaking figure generation long after the run finished.
"""

from __future__ import annotations

import json

import pytest

matplotlib = pytest.importorskip("matplotlib")


def _report(shared_demos: int | None) -> dict:
    strategies = ("onset", "symptom", "stated", "oracle")
    return {
        "config": {"shared_demo_episodes": shared_demos},
        "results": {
            name: {
                "frames": 100,
                "frames_available": 120,
                "collection": {"mean_rewind_offset": float(i * 10)},
                "eval": {
                    "n": 50,
                    "successes": 10 + i,
                    "success_rate": (10 + i) / 50,
                    "ci95_low": 0.05,
                    "ci95_high": 0.45,
                },
            }
            for i, name in enumerate(strategies)
        },
    }


@pytest.mark.parametrize("shared_demos", [None, 150])
def test_build_all_picks_the_figure_matching_the_experiment(tmp_path, shared_demos):
    """The controlled variant answers a different question and gets its own file."""
    from icm.eval.plots import build_all

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "degradation.json").write_text(json.dumps(_report(shared_demos)))
    # correction_start_coverage() reads recorded episodes; with no data_* dirs it
    # returns an empty mapping, which the uncontrolled plot treats as "no overlay".
    made = build_all(run_dir, tmp_path / "media")

    expected = "rewind_depth.png" if shared_demos else "degradation.png"
    assert [p.name for p in made] == [expected]
    assert made[0].stat().st_size > 0


def test_rewind_depth_plot_runs_on_a_real_report(tmp_path):
    from icm.eval.plots import plot_rewind_depth

    report = tmp_path / "degradation.json"
    report.write_text(json.dumps(_report(150)))
    out = plot_rewind_depth(report, tmp_path / "out.png")
    assert out.is_file() and out.stat().st_size > 0
