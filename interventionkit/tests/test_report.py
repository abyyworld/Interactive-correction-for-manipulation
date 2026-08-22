from interventionkit.attribution import analyse
from interventionkit.report import build_markdown, build_report, write_report
from interventionkit.schema import EpisodeMeta, InterventionSegment

NAMES = ("approach", "grasp", "lift", "place")


def _episodes(n=6):
    out = []
    for i in range(n):
        seg = InterventionSegment(start=10, end=20, onset_phase=2, attributed_phase=0 if i < 2 else 2)
        out.append(
            EpisodeMeta(
                episode_id=f"e{i}", task="t", seed=i, n_steps=30, success=i % 2 == 0,
                interventions=[seg], ground_truth={"root_phase": 0, "root_onset_step": 3},
            )
        )
    return out


def test_html_report_is_self_contained(tmp_path):
    eps = _episodes()
    s = analyse(eps, n_phases=4, phase_names=NAMES)
    html = build_report(s, eps, title="T", run_stats={"success_rate": 0.5})
    assert html.startswith("<!doctype html>")
    assert "<svg" in html  # charts inline, no external assets
    assert "http://" not in html and "cdn" not in html.lower()
    p = write_report(tmp_path / "r.html", html)
    assert p.read_text() == html


def test_markdown_report():
    eps = _episodes()
    md = build_markdown(analyse(eps, n_phases=4, phase_names=NAMES), {"success_rate": 0.5})
    assert "# Intervention report" in md
    assert "onset (implicit) misattribution rate" in md


def test_report_survives_empty_run():
    s = analyse([], n_phases=4, phase_names=NAMES)
    html = build_report(s, [], run_stats={})
    assert "<!doctype html>" in html  # NaNs render as "-" rather than crashing
