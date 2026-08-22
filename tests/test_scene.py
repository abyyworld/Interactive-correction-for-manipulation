import numpy as np
import pytest

from tests.conftest import requires_assets

pytestmark = requires_assets


def test_scene_compiles_with_expected_structure():
    import mujoco

    from icm.envs.assets.scene import build_model

    model, xml = build_model()
    # 7 arm joints + 2 fingers + 3 free-jointed objects * 7
    assert model.nq == 9 + 3 * 7
    assert model.nu == 8
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    assert {"wrist", "scene", "front"} <= names


def test_wrist_camera_is_attached_to_the_hand():
    """It must move with the gripper; a world-frame wrist camera is silently useless."""
    import mujoco

    from icm.envs.assets.scene import build_model

    model, _ = build_model()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[cam])
    assert body == "hand"


def test_gravity_compensation_is_active():
    """Set post-compile it silently does nothing; ngravcomp is the real check."""
    from icm.envs.assets.scene import SceneSpec, build_model

    on, _ = build_model(SceneSpec(gravity_compensation=True))
    off, _ = build_model(SceneSpec(gravity_compensation=False))
    assert on.ngravcomp > 0
    assert off.ngravcomp == 0


def test_lookat_basis_is_a_rotation():
    from icm.envs.assets.scene import lookat_basis

    basis = lookat_basis((1.0, -0.7, 0.6), (0.45, 0.0, 0.05))
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-9)
    assert np.linalg.det(basis) == pytest.approx(1.0)
    # Camera looks down its own -z.
    forward = np.array([0.45, 0.0, 0.05]) - np.array([1.0, -0.7, 0.6])
    forward /= np.linalg.norm(forward)
    np.testing.assert_allclose(basis[:, 2], -forward, atol=1e-9)
