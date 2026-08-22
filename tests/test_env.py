import numpy as np
import pytest

from tests.conftest import requires_assets

pytestmark = requires_assets

NOOP = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_reset_is_deterministic_given_a_seed(env):
    a = env.reset(seed=7)["proprio"].copy()
    obj_a = env.object_pos().copy()
    b = env.reset(seed=7)["proprio"].copy()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(obj_a, env.object_pos())


def test_state_restore_is_bit_exact(env):
    """The counterfactual attribution measurement depends on this exactly."""
    env.reset(seed=3)
    for _ in range(12):
        env.step(np.array([0.5, 0.2, -0.3, 0, 0, 0, 1.0]))
    snapshot = env.get_state()
    qpos = env.data.qpos.copy()

    for _ in range(15):
        env.step(np.array([-1.0, 0.4, 0.5, 0, 0, 0, -1.0]))
    env.set_state(snapshot)
    np.testing.assert_array_equal(env.data.qpos, qpos)


def test_replay_from_a_snapshot_is_deterministic(env):
    env.reset(seed=4)
    for _ in range(10):
        env.step(NOOP)
    snapshot = env.get_state()
    action = np.array([0.3, -0.2, 0.1, 0, 0, 0, 1.0])

    first = []
    for _ in range(10):
        env.step(action)
        first.append(env.robot.tcp_pos.copy())
    env.set_state(snapshot)
    for i in range(10):
        env.step(action)
        np.testing.assert_array_equal(env.robot.tcp_pos, first[i])


def test_gravity_compensation_removes_steady_state_droop(env):
    env.reset(seed=0)
    for _ in range(50):
        env.step(NOOP)
    assert np.linalg.norm(env.robot.tcp_pos - env._target_pos) < 1e-3


def test_objects_spawn_apart_and_off_the_goal(env):
    for seed in range(15):
        env.reset(seed=seed)
        positions = [env.object_pos(s.name)[:2] for s in env.object_specs]
        for i, a in enumerate(positions):
            assert np.linalg.norm(a - env.goal_pos) > env.config.goal_radius
            for b in positions[i + 1:]:
                assert np.linalg.norm(a - b) >= env.config.min_object_separation - 1e-6


def test_actions_are_clipped_and_workspace_bounded(env):
    env.reset(seed=0)
    for _ in range(60):
        env.step(np.array([10.0, 10.0, 10.0, 0, 0, 0, 5.0]))
    lo = np.array(env.config.workspace_low)
    hi = np.array(env.config.workspace_high)
    assert np.all(env._target_pos >= lo - 1e-9)
    assert np.all(env._target_pos <= hi + 1e-9)


def test_expert_solves_the_task(env):
    """Guards the whole pipeline: everything downstream assumes a competent expert."""
    from icm.eval.rollout import ScriptedAgent, rollout

    agent = ScriptedAgent()
    successes = sum(rollout(env, agent, seed=1000 + i).success for i in range(20))
    assert successes >= 19, f"expert only solved {successes}/20"


def test_grasp_detection_is_debounced(env):
    """Raw contact drops out mid-lift; unfiltered it corrupts phase labels."""
    from icm.control.scripted import ScriptedExpert

    env.reset(seed=11)
    expert = ScriptedExpert()
    expert.reset()
    raw_dropouts = 0
    latched_dropouts = 0
    was_latched = False
    for _ in range(env.config.max_episode_steps):
        _, _, term, trunc, info = env.step(expert.act(env))
        if was_latched and not env.contact_grasp():
            raw_dropouts += 1
        if was_latched and not info["grasped"]:
            latched_dropouts += 1
        was_latched = info["grasped"] or was_latched
        if term or trunc:
            break
    assert latched_dropouts <= raw_dropouts
