import numpy as np

from tests.conftest import requires_assets

pytestmark = requires_assets


def test_ik_converges_across_the_workspace(env):
    from icm.envs.panda import grasp_quat

    rng = np.random.default_rng(0)
    errors = []
    for _ in range(40):
        target = np.array(
            [rng.uniform(0.36, 0.68), rng.uniform(-0.24, 0.34), rng.uniform(0.03, 0.30)]
        )
        result = env.robot.solve_ik(
            target, grasp_quat(rng.uniform(-np.pi / 2, np.pi / 2)), max_iters=200
        )
        assert result.converged, f"IK failed for {target}"
        errors.append(result.pos_err)
    assert max(errors) < 1e-3  # sub-millimetre


def test_orientation_error_uses_the_world_frame(env):
    """Expressed locally, the solver stalls at a ~pi residual that looks unreachable."""
    from icm.envs.panda import grasp_quat

    target = np.array([0.45, 0.0, 0.22])
    result = env.robot.solve_ik(target, grasp_quat(0.3), max_iters=200)
    assert result.converged
    assert result.rot_err < 1e-2


def test_ik_does_not_disturb_simulation_state(env):
    env.reset(seed=0)
    before = env.data.qpos.copy()
    env.robot.solve_ik(np.array([0.55, 0.1, 0.15]), max_iters=50)
    np.testing.assert_array_equal(env.data.qpos, before)


def test_ik_respects_joint_limits(env):
    from icm.envs.panda import grasp_quat

    result = env.robot.solve_ik(np.array([0.5, 0.0, 0.15]), grasp_quat(0.0), max_iters=100)
    lo = env.robot.arm_joint_range[:, 0]
    hi = env.robot.arm_joint_range[:, 1]
    assert np.all(result.qpos >= lo - 1e-6)
    assert np.all(result.qpos <= hi + 1e-6)
