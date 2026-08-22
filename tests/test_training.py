"""Dataset, credit assignment and policy tests.

These deliberately avoid stepping an optimiser: on CPU-only setups MuJoCo and
PyTorch's MKL kernels segfault when an optimiser runs in a process where both are
loaded. Training is exercised end to end through a subprocess instead, which is
also how the real pipeline runs it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from icm.training.weighting import CreditAssignment, corrected_span
from interventionkit import InterventionRecorder

PHASES = ("approach", "grasp", "lift", "place")

#: PyTorch is an optional extra ("make torch-cpu"), because the environment, the
#: scripted expert and the whole attribution study need nothing from it. Tests
#: that do need it skip rather than fail, so a base install reports a clean run
#: instead of five errors that look like the project is broken.
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed; run 'make torch-cpu' or 'make torch-cuda'",
)


@pytest.fixture
def run_dir(tmp_path):
    """A run where the failure is caused in APPROACH but noticed during LIFT."""
    recorder = InterventionRecorder(tmp_path, task="pick_place", phase_names=PHASES)
    for ep in range(6):
        with recorder.episode(seed=ep) as episode:
            timeline = []
            for t in range(40):
                phase = min(t // 10, 3)
                timeline.append(phase)
                if t >= 25:
                    episode.expert_step(
                        np.full(7, 0.5, dtype=np.float32),
                        phase=phase,
                        proprio=np.zeros(24, dtype=np.float32),
                        privileged=np.zeros(27, dtype=np.float32),
                    )
                else:
                    episode.policy_step(
                        np.zeros(7, dtype=np.float32),
                        phase=phase,
                        proprio=np.zeros(24, dtype=np.float32),
                        privileged=np.zeros(27, dtype=np.float32),
                    )
            episode.attribute(0 if ep < 2 else 2)
            episode.finish(
                success=False,
                ground_truth={"root_phase": 0, "root_onset_step": 4, "symptom_step": 22},
                extra={"phase_timeline": timeline},
            )
    return tmp_path


def test_credit_strategies_move_the_span_start(run_dir):
    from interventionkit import RunReader

    meta = RunReader(run_dir).episodes()[0]
    seg = meta.interventions[0]
    assert corrected_span(seg, meta, CreditAssignment.ONSET)[0] == 25
    assert corrected_span(seg, meta, CreditAssignment.SYMPTOM)[0] == 22
    assert corrected_span(seg, meta, CreditAssignment.ORACLE)[0] == 4
    # This episode's supervisor traced it back to approach, whose first step is 0.
    assert corrected_span(seg, meta, CreditAssignment.STATED)[0] == 0


def test_stated_strategy_follows_a_wrong_attribution(run_dir):
    """The whole point: a wrong attribution rewinds to states that were fine."""
    from interventionkit import RunReader

    wrong = [m for m in RunReader(run_dir).episodes() if m.interventions[0].attributed_phase == 2]
    meta = wrong[0]
    start, _ = corrected_span(meta.interventions[0], meta, CreditAssignment.STATED)
    assert start == 20  # start of LIFT, well after the real cause at step 4


def test_dataset_selects_only_corrective_frames(run_dir):
    from icm.training.dataset import DatasetConfig, InterventionDataset

    ds = InterventionDataset(
        run_dir,
        DatasetConfig(
            supervision="corrections", credit=CreditAssignment.ONSET, state_key="privileged"
        ),
    )
    assert len(ds) == 6 * 15  # steps 25..39 of each episode
    sample = ds[0]
    np.testing.assert_allclose(sample["action"][0], np.full(7, 0.5), atol=1e-6)


def test_dataset_chunk_padding_repeats_the_last_action(run_dir):
    """Zero padding would teach the policy to jerk to the origin at episode end."""
    from icm.training.dataset import DatasetConfig, InterventionDataset

    ds = InterventionDataset(
        run_dir, DatasetConfig(supervision="corrections", chunk=8, state_key="privileged")
    )
    last = ds[len(ds) - 1]
    np.testing.assert_allclose(last["action"][-1], last["action"][0], atol=1e-6)
    assert last["action_valid"].sum() == 1


def test_subsample_is_deterministic(run_dir):
    from icm.training.dataset import DatasetConfig, InterventionDataset

    a = InterventionDataset(run_dir, DatasetConfig(state_key="privileged"))
    b = InterventionDataset(run_dir, DatasetConfig(state_key="privileged"))
    a.subsample(20, seed=1)
    b.subsample(20, seed=1)
    assert len(a) == len(b) == 20
    assert a.samples == b.samples


@requires_torch
def test_episode_split_never_shares_an_episode(run_dir):
    from icm.training.dataset import DatasetConfig, InterventionDataset
    from icm.training.train_bc import episode_split

    ds = InterventionDataset(run_dir, DatasetConfig(state_key="privileged"))
    train, val = episode_split(ds, 0.34, seed=0)
    train_eps = {ds.samples[i][1] for i in train}
    val_eps = {ds.samples[i][1] for i in val}
    assert train_eps and val_eps
    assert not (train_eps & val_eps)


@requires_torch
def test_policy_shapes_and_backward():
    import torch

    from icm.policies.bc import BCPolicy, PolicyConfig

    policy = BCPolicy(PolicyConfig(state_dim=27, chunk=8))
    batch = {
        "state": torch.randn(4, 27),
        "action": torch.randn(4, 8, 7),
        "action_valid": torch.ones(4, 8),
        "weight": torch.ones(4),
    }
    assert policy(batch).shape == (4, 8, 7)
    loss, metrics = policy.loss(batch)
    loss.backward()
    assert np.isfinite(metrics["loss"])


@requires_torch
def test_action_valid_mask_excludes_padding():
    import torch

    from icm.policies.bc import BCPolicy, PolicyConfig

    policy = BCPolicy(PolicyConfig(state_dim=27, chunk=4))
    policy.eval()  # dropout is stochastic in train mode; two passes would not match
    batch = {
        "state": torch.zeros(1, 27),
        "action": torch.zeros(1, 4, 7),
        "action_valid": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "weight": torch.ones(1),
    }
    masked, _ = policy.loss(batch)
    batch["action"][0, 1:] = 1e6  # garbage in the padded region
    still_masked, _ = policy.loss(batch)
    assert torch.allclose(masked, still_masked)


@requires_torch
def test_temporal_ensemble_prefers_recent_predictions():
    from icm.policies.bc import TemporalEnsemble

    ens = TemporalEnsemble(chunk=3, action_dim=1, decay=1.0)
    ens.step(np.array([[0.0], [0.0], [0.0]]))
    out = ens.step(np.array([[1.0], [1.0], [1.0]]))
    assert 0.5 < out[0] <= 1.0  # newest prediction dominates


def test_size_estimator_scales_as_expected():
    from icm.training.dataset import estimate_size

    one = estimate_size(1000)
    ten = estimate_size(10_000)
    assert ten["raw_gb"] == pytest.approx(one["raw_gb"] * 10, rel=1e-6)
    assert estimate_size(1000, n_cameras=1, depth=False)["raw_gb"] < one["raw_gb"]


@requires_torch
def test_training_runs_end_to_end_in_a_subprocess(run_dir, tmp_path):
    out = tmp_path / "train_out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "icm.cli.train",
            str(run_dir),
            "-o",
            str(out),
            "--smoke",
            "--state-key",
            "privileged",
            "--workers",
            "0",
            "--val-fraction",
            "0.34",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (out / "checkpoint.pt").is_file()
    assert (out / "metrics.jsonl").is_file()


def test_per_root_cap_is_applied_per_source(run_dir, tmp_path):
    """Each source is capped independently.

    This is the control that separates "which states were corrected" from "how
    much data there was": one source is held fixed while the other is matched in
    size. Capping the total instead would trim both and reintroduce the
    difference being controlled for.
    """
    import shutil

    from icm.training.dataset import DatasetConfig, InterventionDataset

    # Two distinct directories: frames_per_root is keyed by directory name, so a
    # duplicate root would collapse into a single entry.
    second = tmp_path / "second_run"
    shutil.copytree(run_dir, second)

    uncapped = InterventionDataset(
        [run_dir, second],
        DatasetConfig(supervision="corrections", state_key="privileged"),
    )
    counts_uncapped = uncapped.summary()["frames_per_root"]
    assert len(counts_uncapped) == 2
    assert min(counts_uncapped.values()) > 10

    capped = InterventionDataset(
        [run_dir, second],
        DatasetConfig(
            supervision="corrections",
            state_key="privileged",
            frame_cap_per_root=(10, 10**9),
        ),
    )
    counts = capped.summary()["frames_per_root"]
    assert counts[run_dir.name] == 10, "first root should be capped"
    assert counts[second.name] == counts_uncapped[second.name], "second root untouched"


def test_shared_demo_pool_reaches_the_training_subprocess(run_dir, tmp_path, monkeypatch):
    """The shared-demo control must actually be passed to the trainer.

    Regression test: an earlier version constructed the subprocess command with a
    single data root, so --shared-demos silently had no effect and the controlled
    experiment reproduced the uncontrolled one exactly.
    """
    import icm.study.degradation as degradation

    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out = Path(cmd[cmd.index("-o") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text("{}")
        return _Result()

    # degradation imports subprocess inside the function, so patch the module
    # itself rather than an attribute that does not exist on it.
    monkeypatch.setattr(subprocess, "run", fake_run)

    from icm.training.dataset import DatasetConfig

    cfg = degradation.DegradationConfig(shared_demo_episodes=5)
    degradation._train_with_budget(
        run_dir,
        tmp_path / "out",
        DatasetConfig(state_key="privileged"),
        cfg,
        budget=123,
        shared_demos=tmp_path / "demos",
    )
    cmd = captured["cmd"]
    assert str(tmp_path / "demos") in cmd, "shared demo pool missing from the command"
    assert str(run_dir) in cmd
    # Corrections capped, demo pool uncapped, in that order.
    i = cmd.index("--cap-per-root")
    assert cmd[i + 1 : i + 3] == ["123", "-1"]
    assert cmd.index(str(run_dir)) < cmd.index(str(tmp_path / "demos"))
