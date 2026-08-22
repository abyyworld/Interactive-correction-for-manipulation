import numpy as np
import pytest

from tests.conftest import requires_assets

pytestmark = requires_assets


@pytest.mark.parametrize(
    "fault_name",
    [
        "grasp_offset",
        "premature_close",
        "weak_grip",
        "lift_slip",
        "early_release",
        "wrong_object",
    ],
)
def test_every_fault_fires_and_causes_failure(env, fault_name):
    """A fault that silently fails to fire shows up as a supervisor missing an
    error that was never introduced, which would corrupt the study."""
    from icm.envs.faults import FaultInjector, FaultSpec, FaultType
    from icm.eval.rollout import ScriptedAgent, rollout

    fired = successes = 0
    n = 10
    for i in range(n):
        injector = FaultInjector(
            FaultSpec(type=FaultType(fault_name), severity=0.9), np.random.default_rng(i)
        )
        result = rollout(env, ScriptedAgent(injector=injector), seed=6000 + i)
        fired += int(injector.state.onset_step is not None)
        successes += int(result.success)
    assert fired == n, f"{fault_name} only fired {fired}/{n} times"
    assert successes <= 1, f"{fault_name} still succeeded {successes}/{n} times"


def test_no_fault_control_always_succeeds(env):
    from icm.envs.faults import FaultInjector, FaultSpec, FaultType
    from icm.eval.rollout import ScriptedAgent, rollout

    for i in range(10):
        injector = FaultInjector(FaultSpec(type=FaultType.NONE), np.random.default_rng(i))
        result = rollout(env, ScriptedAgent(injector=injector), seed=6000 + i)
        assert result.success
        assert injector.state.onset_step is None


def test_weak_grip_does_not_leak_into_the_next_episode(env):
    """Model mutation must be undone, or every later rollout is contaminated."""
    from icm.envs.faults import FaultInjector, FaultSpec, FaultType
    from icm.eval.rollout import ScriptedAgent, rollout

    injector = FaultInjector(
        FaultSpec(type=FaultType.WEAK_GRIP, severity=1.0), np.random.default_rng(0)
    )
    rollout(env, ScriptedAgent(injector=injector), seed=1)
    clean = rollout(env, ScriptedAgent(), seed=1)
    assert clean.success
    np.testing.assert_array_equal(
        env.model.actuator_forcerange[env.robot.gripper_actuator_id],
        env._orig_gripper_forcerange,
    )


def test_matched_pair_shares_a_symptom_but_not_a_cause():
    """weak_grip and lift_slip both drop the object during the lift."""
    from icm.envs.faults import FAULT_PHASES, FaultType
    from icm.envs.phases import Phase

    weak_root, weak_symptom, weak_lag = FAULT_PHASES[FaultType.WEAK_GRIP]
    slip_root, slip_symptom, slip_lag = FAULT_PHASES[FaultType.LIFT_SLIP]
    assert weak_symptom == slip_symptom == Phase.LIFT
    assert weak_root != slip_root
    assert weak_lag == "delayed" and slip_lag == "immediate"


def test_fault_suspends_on_takeover(env):
    """A fault models the agent's error; if it survived a takeover the human
    could not fix it and the correction data would be worthless."""
    from icm.envs.faults import FaultInjector, FaultSpec, FaultType
    from icm.eval.rollout import ScriptedAgent

    injector = FaultInjector(
        FaultSpec(type=FaultType.WEAK_GRIP, severity=1.0), np.random.default_rng(0)
    )
    agent = ScriptedAgent(injector=injector)
    env.reset(seed=0)
    agent.reset(env)
    assert env.model.actuator_forcerange[env.robot.gripper_actuator_id][1] < 1.0
    agent.on_supervisor_engage(env)
    assert injector.suspended
    np.testing.assert_array_equal(
        env.model.actuator_forcerange[env.robot.gripper_actuator_id],
        env._orig_gripper_forcerange,
    )
