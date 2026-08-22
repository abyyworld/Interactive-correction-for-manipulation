import numpy as np
import pytest

from interventionkit import InterventionRecorder, RunReader, analyse

PHASES = ("approach", "grasp", "lift", "place")


def test_records_and_derives_segments(tmp_path):
    rec = InterventionRecorder(tmp_path, task="pick_place", phase_names=PHASES)
    with rec.episode(seed=0, instruction="pick up the red block") as ep:
        for t in range(10):
            if 4 <= t < 7:
                ep.human_step(np.zeros(7), phase=2, proprio=np.zeros(3))
            else:
                ep.policy_step(np.zeros(7), phase=min(t // 3, 3), proprio=np.zeros(3))
        ep.attribute(phase=0, confidence=0.4, notes="approach drifted")
        ep.finish(success=False, ground_truth={"root_phase": 0, "root_onset_step": 2})

    r = RunReader(tmp_path)
    (meta,) = r.episodes()
    assert meta.n_steps == 10
    assert meta.instruction == "pick up the red block"
    seg = meta.interventions[0]
    assert (seg.start, seg.end) == (4, 7)
    assert seg.onset_phase_name == "lift"
    assert seg.attributed_phase == 0 and seg.attributed_phase_name == "approach"
    assert seg.confidence == pytest.approx(0.4)


def test_multiple_segments(tmp_path):
    rec = InterventionRecorder(tmp_path, task="t", phase_names=PHASES)
    with rec.episode(seed=0) as ep:
        ep.policy_step(np.zeros(7), phase=0)
        ep.human_step(np.zeros(7), phase=1)
        ep.policy_step(np.zeros(7), phase=1)
        ep.human_step(np.zeros(7), phase=2)
        ep.human_step(np.zeros(7), phase=2)
        ep.finish(success=True)
    (meta,) = RunReader(tmp_path).episodes()
    assert [(s.start, s.end) for s in meta.interventions] == [(1, 2), (3, 5)]


def test_attribute_without_intervention_raises(tmp_path):
    rec = InterventionRecorder(tmp_path, task="t", phase_names=PHASES)
    with rec.episode(seed=0) as ep:
        ep.policy_step(np.zeros(7), phase=0)
        with pytest.raises(RuntimeError, match="no intervention"):
            ep.attribute(phase=0)
        ep.finish(success=True)


def test_end_to_end_analysis(tmp_path):
    """The full loop a user of this package would run."""
    rec = InterventionRecorder(tmp_path, task="pick_place", phase_names=PHASES)
    for i in range(10):
        with rec.episode(seed=i) as ep:
            for t in range(20):
                if t >= 12:
                    ep.human_step(np.zeros(7), phase=2)
                else:
                    ep.policy_step(np.zeros(7), phase=min(t // 5, 3))
            # 3 of 10 supervisors correctly trace the fault back to approach
            ep.attribute(phase=0 if i < 3 else 2)
            ep.finish(success=False, ground_truth={"root_phase": 0, "root_onset_step": 4})

    reader = RunReader(tmp_path)
    s = analyse(reader.episodes(), n_phases=4, phase_names=PHASES)
    assert s.n_with_ground_truth == 10
    assert s.onset_misattribution_rate == pytest.approx(1.0)
    assert s.stated_misattribution_rate == pytest.approx(0.7)
    assert s.mean_detection_lag == pytest.approx(8.0)
