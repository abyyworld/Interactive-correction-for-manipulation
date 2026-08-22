"""Rewind-and-redemonstrate protocol.

The degradation experiment's validity rests entirely on the rewind landing where
the strategy says it should, and on the recorded episode being a coherent
policy-prefix-then-correction sequence. Both are checked here rather than
assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from icm.training.dagger import first_step_of_phase, resolve_rewind_step
from icm.training.weighting import CreditAssignment as CA
from tests.conftest import requires_assets

PHASES = [0] * 20 + [1] * 10 + [2] * 10 + [3] * 20
COMMON = dict(takeover_step=45, symptom_step=38, root_onset_step=3, phases=PHASES)


def test_each_strategy_rewinds_where_it_says():
    assert resolve_rewind_step(CA.ONSET, attributed_phase=0, **COMMON) == 45
    assert resolve_rewind_step(CA.SYMPTOM, attributed_phase=0, **COMMON) == 38
    assert resolve_rewind_step(CA.ORACLE, attributed_phase=0, **COMMON) == 3
    assert resolve_rewind_step(CA.STATED, attributed_phase=0, **COMMON) == 0


def test_a_wrong_attribution_rewinds_past_the_causal_states():
    """The mechanism the whole experiment measures."""
    correct = resolve_rewind_step(CA.STATED, attributed_phase=0, **COMMON)
    wrong = resolve_rewind_step(CA.STATED, attributed_phase=2, **COMMON)
    oracle = resolve_rewind_step(CA.ORACLE, attributed_phase=0, **COMMON)
    assert correct <= oracle + 1  # blaming the right phase lands near the cause
    assert wrong == 30  # start of LIFT, long after the cause at step 3
    assert wrong > oracle


def test_missing_information_falls_back_to_the_takeover_step():
    """A run lacking ground truth must degrade to HG-DAgger, not crash."""
    bare = dict(takeover_step=45, symptom_step=None, root_onset_step=None, phases=[])
    for strategy in CA:
        assert resolve_rewind_step(strategy, attributed_phase=None, **bare) == 45


def test_first_step_of_phase():
    assert first_step_of_phase(PHASES, 0) == 0
    assert first_step_of_phase(PHASES, 2) == 30
    assert first_step_of_phase(PHASES, 9) is None


@requires_assets
def test_rewind_produces_a_coherent_episode(env, tmp_path):
    """Policy prefix then correction, contiguous, with the rewind recorded."""
    from icm.control.scripted import ScriptedExpert
    from icm.envs.faults import FaultInjector, FaultSpec, FaultType
    from icm.eval.rollout import ScriptedAgent
    from icm.study.supervisor import SyntheticSupervisor
    from icm.training.dagger import collect_interactive_episode
    from interventionkit import InterventionRecorder, RunReader

    recorder = InterventionRecorder(
        tmp_path, task="pick_place", phase_names=("approach", "grasp", "lift", "place")
    )
    supervisor = SyntheticSupervisor(rng=np.random.default_rng(0))
    injector = FaultInjector(
        FaultSpec(type=FaultType.WEAK_GRIP, severity=1.0), np.random.default_rng(0)
    )
    agent = ScriptedAgent(ScriptedExpert(), injector)

    with recorder.episode(seed=3) as ep:
        result = collect_interactive_episode(
            env,
            agent,
            supervisor,
            ep,
            seed=3,
            strategy=CA.ORACLE,
            corrective_expert=ScriptedExpert(),
            record_keys=("proprio", "privileged"),
        )

    assert result.intervened
    assert result.rewind_step is not None
    assert result.rewind_step <= result.takeover_step

    (meta,) = RunReader(tmp_path).episodes()
    assert meta.n_steps == result.n_policy_steps + result.n_correction_steps
    assert len(meta.extra["phase_timeline"]) == meta.n_steps
    assert meta.ground_truth["rewind_step"] == result.rewind_step

    data = RunReader(tmp_path).load(meta.episode_id)
    actors = list(data["actor"])
    # Exactly one transition: policy steps, then corrective steps.
    assert actors[: result.n_policy_steps] == ["policy"] * result.n_policy_steps
    assert set(actors[result.n_policy_steps :]) == {"expert"}


@requires_assets
def test_deeper_rewind_starts_the_correction_earlier(env, tmp_path):
    """ORACLE must begin its correction before ONSET does, on the same episode."""
    from icm.control.scripted import ScriptedExpert
    from icm.envs.faults import FaultInjector, FaultSpec, FaultType
    from icm.eval.rollout import ScriptedAgent
    from icm.study.supervisor import SyntheticSupervisor
    from icm.training.dagger import collect_interactive_episode
    from interventionkit import InterventionRecorder

    rewinds = {}
    for strategy in (CA.ONSET, CA.ORACLE):
        recorder = InterventionRecorder(
            tmp_path / strategy.value,
            task="pick_place",
            phase_names=("approach", "grasp", "lift", "place"),
        )
        injector = FaultInjector(
            FaultSpec(type=FaultType.GRASP_OFFSET, severity=1.0), np.random.default_rng(0)
        )
        with recorder.episode(seed=5) as ep:
            r = collect_interactive_episode(
                env,
                ScriptedAgent(ScriptedExpert(), injector),
                SyntheticSupervisor(rng=np.random.default_rng(0)),
                ep,
                seed=5,
                strategy=strategy,
                corrective_expert=ScriptedExpert(),
                record_keys=("proprio",),
            )
        rewinds[strategy.value] = r.rewind_step if r.intervened else None

    if rewinds["onset"] is None or rewinds["oracle"] is None:
        pytest.skip("no intervention on this seed")
    # grasp_offset originates at step 0, so oracle rewinds to the very beginning.
    assert rewinds["oracle"] < rewinds["onset"]


def test_failure_breakdown_never_counts_a_success():
    """DONE is a success state; it cannot appear among failures."""
    from icm.envs.phases import Phase
    from icm.eval.metrics import EvalResult

    result = EvalResult(
        n=4,
        successes=2,
        # Deliberately interleaved: a positional reconstruction would pair the
        # first two phases with success and mislabel the rest.
        final_phases=[int(Phase.APPROACH), int(Phase.DONE), int(Phase.LIFT), int(Phase.DONE)],
        outcomes=[False, True, False, True],
    )
    breakdown = result.failure_breakdown()
    assert breakdown == {"approach": 1, "lift": 1}
    assert "done" not in breakdown
