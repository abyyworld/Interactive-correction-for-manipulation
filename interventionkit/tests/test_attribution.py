import numpy as np
import pytest
from interventionkit.attribution import analyse, interval_iou, per_phase_breakdown
from interventionkit.schema import EpisodeMeta, InterventionSegment

NAMES = ("approach", "grasp", "lift", "place")


def _ep(i, onset_phase, root_phase, attributed=None, start=30, end=50, root_onset=10, pnr=None):
    seg = InterventionSegment(
        start=start, end=end, onset_phase=onset_phase, attributed_phase=attributed
    )
    gt = {"root_phase": root_phase, "root_onset_step": root_onset}
    if pnr is not None:
        gt["pnr_step"] = pnr
    return EpisodeMeta(
        episode_id=f"e{i}",
        task="t",
        seed=i,
        n_steps=60,
        success=False,
        interventions=[seg],
        ground_truth=gt,
    )


def test_interval_iou():
    assert interval_iou(0, 10, 0, 10) == 1.0
    assert interval_iou(0, 5, 5, 10) == 0.0
    assert interval_iou(0, 10, 5, 15) == pytest.approx(1 / 3)
    assert interval_iou(3, 3, 3, 3) == 0.0  # degenerate, must not divide by zero


def test_onset_misattribution_counts_phase_mismatch():
    eps = [_ep(i, onset_phase=2, root_phase=0) for i in range(8)]
    eps += [_ep(i + 8, onset_phase=0, root_phase=0) for i in range(2)]
    s = analyse(eps, n_phases=4, phase_names=NAMES)
    assert s.onset_misattribution_rate == pytest.approx(0.8)
    assert s.onset_confusion[0, 2] == 8
    assert s.onset_confusion[0, 0] == 2


def test_stated_attribution_tracked_separately():
    eps = [_ep(i, onset_phase=2, root_phase=0, attributed=0 if i < 7 else 2) for i in range(10)]
    s = analyse(eps, n_phases=4, phase_names=NAMES)
    assert s.onset_misattribution_rate == pytest.approx(1.0)  # always took over at lift
    assert s.stated_misattribution_rate == pytest.approx(0.3)  # but usually traced it back
    assert s.n_stated == 10


def test_detection_lag_and_credit_iou():
    eps = [_ep(i, onset_phase=2, root_phase=0, start=30, end=50, root_onset=10) for i in range(4)]
    s = analyse(eps, n_phases=4)
    assert s.mean_detection_lag == pytest.approx(20.0)
    # corrected [30,50) vs truly-wrong [10,60) -> 20/50
    assert s.mean_credit_iou == pytest.approx(0.4)


def test_late_intervention_rate():
    eps = [_ep(i, 2, 0, start=50, pnr=40) for i in range(3)]
    eps += [_ep(i + 3, 2, 0, start=30, pnr=40) for i in range(1)]
    s = analyse(eps, n_phases=4)
    assert s.late_intervention_rate == pytest.approx(0.75)


def test_episodes_without_ground_truth_are_skipped():
    good = _ep(0, 2, 0)
    bare = EpisodeMeta(episode_id="x", task="t", seed=1, n_steps=10, success=True)
    s = analyse([good, bare], n_phases=4)
    assert s.n_episodes == 2
    assert s.n_with_ground_truth == 1


def test_no_data_yields_nan_not_crash():
    s = analyse([], n_phases=4)
    assert s.n_episodes == 0
    assert np.isnan(s.onset_misattribution_rate)


def test_per_phase_breakdown_splits_by_root_cause():
    eps = [_ep(i, onset_phase=2, root_phase=0) for i in range(6)]  # delayed-consequence
    eps += [_ep(i + 6, onset_phase=3, root_phase=3) for i in range(4)]  # immediate
    bd = per_phase_breakdown(eps, n_phases=4)
    assert bd[0]["misattribution_rate"] == pytest.approx(1.0)
    assert bd[3]["misattribution_rate"] == pytest.approx(0.0)
    assert bd[0]["n"] == 6
