import json

import numpy as np
import pytest
from interventionkit.store import RunReader, RunWriter

PHASES = ("approach", "grasp", "lift", "place")


def _make_run(tmp_path, n_episodes=5, n_steps=8):
    with RunWriter(tmp_path, task="pick_place", phase_names=PHASES) as run:
        for ep in range(n_episodes):
            w = run.episode(seed=ep, instruction="pick up the red block")
            for t in range(n_steps):
                actor = "human" if (ep == 1 and 3 <= t < 6) else "policy"
                w.record(
                    action=np.full(7, t, dtype=np.float32),
                    actor=actor,
                    phase=min(t // 2, 3),
                    proprio=np.zeros(24, dtype=np.float32),
                )
            w.finish(success=(ep % 2 == 0), ground_truth={"root_phase": 0} if ep == 1 else {})
    return RunReader(tmp_path)


def test_roundtrip_and_stats(tmp_path):
    r = _make_run(tmp_path)
    assert len(r) == 5
    stats = r.stats()
    assert stats["episodes"] == 5
    assert stats["success_rate"] == pytest.approx(3 / 5)
    assert stats["intervention_rate"] == pytest.approx(1 / 5)
    assert stats["corrected_steps"] == 3


def test_arrays_roundtrip_exactly(tmp_path):
    r = _make_run(tmp_path)
    data = r.load("ep_000000")
    assert data["action"].shape == (8, 7)
    np.testing.assert_array_equal(data["action"][3], np.full(7, 3, dtype=np.float32))
    assert list(data["actor"][:2]) == ["policy", "policy"]


def test_lazy_open_does_not_require_all_fields(tmp_path):
    r = _make_run(tmp_path)
    meta, arrays = next(r.iter_arrays(keys=("action", "phase")))
    assert set(arrays) == {"action", "phase"}
    assert meta.n_steps == 8


def test_frame_index_is_flat(tmp_path):
    r = _make_run(tmp_path, n_episodes=3, n_steps=4)
    idx = r.frame_index()
    assert len(idx) == 12
    assert idx[0] == ("ep_000000", 0)
    assert idx[-1] == ("ep_000002", 3)


def test_index_rebuild_when_missing(tmp_path):
    reader = _make_run(tmp_path, n_episodes=3)
    assert len(reader) == 3
    (tmp_path / "index.jsonl").unlink()
    assert len(RunReader(tmp_path)) == 3  # rebuilt from the per-episode sidecars


def test_ragged_fields_are_rejected(tmp_path):
    """A field appearing only on some steps would silently misalign the dataset."""
    with RunWriter(tmp_path, task="t", phase_names=PHASES) as run:
        w = run.episode(seed=0)
        w.record(action=np.zeros(7), actor="policy", phase=0)
        with pytest.raises(ValueError, match="every step must record"):
            w.record(action=np.zeros(7), actor="policy", phase=0, surprise=np.zeros(3))


def test_exception_discards_episode(tmp_path):
    with RunWriter(tmp_path, task="t", phase_names=PHASES) as run:
        try:
            with run.episode(seed=0) as w:
                w.record(action=np.zeros(7), actor="policy", phase=0)
                raise RuntimeError("collection crashed")
        except RuntimeError:
            pass
    assert len(RunReader(tmp_path)) == 0
    assert not list((tmp_path / "episodes").glob("*.npz"))


def test_not_a_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunReader(tmp_path)


def test_run_meta_written(tmp_path):
    _make_run(tmp_path, n_episodes=1)
    meta = json.loads((tmp_path / "run.json").read_text())
    assert meta["task"] == "pick_place"
    assert meta["config"]["phase_names"] == list(PHASES)
