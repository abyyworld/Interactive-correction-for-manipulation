"""Thin, typed wrapper around the Franka Panda inside a compiled MuJoCo model.

Everything above this layer (controllers, teleop, policies, the study) talks about
the robot in terms of a TCP pose and a gripper width. Keeping the index
bookkeeping — which qpos entry is joint 4, which actuator drives the fingers — in
exactly one place is what stops a whole class of silent off-by-one bugs that
manifest as "the policy learned nothing".
"""

from __future__ import annotations

import mujoco
import numpy as np

from .assets.scene import TCP_OFFSET_Z  # noqa: F401  (re-exported for convenience)

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
TCP_SITE = "tcp"
HAND_BODY = "hand"

# Menagerie remaps the finger tendon to a 0-255 command range (Franka's own
# convention) rather than metres. 255 = fully open = 8 cm between fingertips.
GRIPPER_CTRL_MAX = 255.0
GRIPPER_MAX_WIDTH = 0.08

# Upstream Menagerie "home" keyframe. Elbow up, gripper already pointing down,
# but ~0.82 m above the table: fine as a neutral pose, useless as a start state.
HOME_ARM_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Where episodes actually start: gripper open, pointing down, hovering over the
# middle of the workspace so that every object is a short move away. Solved by
# IK at reset rather than hard-coded, so it stays correct if the table moves.
READY_TCP_POS = np.array([0.45, 0.0, 0.22])


def grasp_quat(yaw: float = 0.0) -> np.ndarray:
    """Quaternion for a top-down grasp with the given yaw about the world z axis.

    The gripper's +z points out between the fingers, so a top-down grasp needs
    the TCP z axis aligned with world -z. ``yaw`` then rotates the finger opening
    direction, which matters as soon as objects stop being rotationally symmetric.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    # Columns: x, y, z of the TCP frame expressed in world coordinates.
    mat = np.array(
        [
            [s, c, 0.0],
            [c, -s, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


class PandaRobot:
    """Index bookkeeping + kinematics helpers for one compiled Panda model."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        def jid(name: str) -> int:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise KeyError(f"joint {name!r} not present in model")
            return i

        def aid(name: str) -> int:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise KeyError(f"actuator {name!r} not present in model")
            return i

        self.arm_joint_ids = np.array([jid(n) for n in ARM_JOINTS])
        self.finger_joint_ids = np.array([jid(n) for n in FINGER_JOINTS])
        self.arm_qpos_adr = np.array([model.jnt_qposadr[i] for i in self.arm_joint_ids])
        self.arm_dof_adr = np.array([model.jnt_dofadr[i] for i in self.arm_joint_ids])
        self.finger_qpos_adr = np.array([model.jnt_qposadr[i] for i in self.finger_joint_ids])

        self.arm_actuator_ids = np.array([aid(n) for n in ARM_ACTUATORS])
        self.gripper_actuator_id = aid(GRIPPER_ACTUATOR)

        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self.tcp_site_id < 0:
            raise KeyError(
                "TCP site missing - build the model via icm.envs.assets.scene.build_model"
            )
        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)

        self.arm_joint_range = model.jnt_range[self.arm_joint_ids].copy()
        # Scratch state for IK, so solving never perturbs the live simulation.
        self._ik_data = mujoco.MjData(model)

    # ------------------------------------------------------------------ state

    @property
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    @property
    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof_adr].copy()

    @property
    def gripper_width(self) -> float:
        """Distance between the fingertips, metres."""
        return float(self.data.qpos[self.finger_qpos_adr].sum())

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site_id].copy()

    @property
    def tcp_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()

    @property
    def tcp_quat(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.tcp_site_id].copy())
        return quat

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.tcp_pos, self.tcp_quat

    # ------------------------------------------------------------------ control

    def set_arm_ctrl(self, qpos_target: np.ndarray) -> None:
        """Command joint position targets (the Panda actuators are position-servo)."""
        lo = self.arm_joint_range[:, 0]
        hi = self.arm_joint_range[:, 1]
        self.data.ctrl[self.arm_actuator_ids] = np.clip(qpos_target, lo, hi)

    def set_gripper_ctrl(self, opening: float) -> None:
        """``opening`` in [0, 1]: 0 fully closed, 1 fully open."""
        self.data.ctrl[self.gripper_actuator_id] = (
            float(np.clip(opening, 0.0, 1.0)) * GRIPPER_CTRL_MAX
        )

    @property
    def gripper_ctrl(self) -> float:
        return float(self.data.ctrl[self.gripper_actuator_id] / GRIPPER_CTRL_MAX)

    def reset_arm(self, qpos: np.ndarray, gripper_opening: float = 1.0) -> None:
        """Teleport the arm and sync the controller setpoint to match.

        Setting qpos without also setting ctrl is a classic MuJoCo footgun: the
        position servos would immediately yank the arm back toward whatever the
        previous command was.
        """
        self.data.qpos[self.arm_qpos_adr] = qpos
        self.data.qvel[self.arm_dof_adr] = 0.0
        width = gripper_opening * GRIPPER_MAX_WIDTH
        self.data.qpos[self.finger_qpos_adr] = width / 2.0
        self.set_arm_ctrl(qpos)
        self.set_gripper_ctrl(gripper_opening)

    # ------------------------------------------------------------------ kinematics

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        *,
        seed_qpos: np.ndarray | None = None,
        **kwargs,
    ):
        """Solve IK from ``seed_qpos`` (default: current pose) without touching sim state.

        Retries once from the home posture if the first attempt fails to converge.
        """
        from ..control.ik import solve_ik as _solve

        kwargs.setdefault("rest_qpos", HOME_ARM_QPOS)
        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = None if target_quat is None else np.asarray(target_quat, dtype=float)

        def attempt(seed: np.ndarray | None):
            self._ik_data.qpos[:] = self.data.qpos
            self._ik_data.qvel[:] = 0.0
            if seed is not None:
                self._ik_data.qpos[self.arm_qpos_adr] = seed
            return _solve(
                self.model,
                self._ik_data,
                self.tcp_site_id,
                target_pos,
                target_quat,
                dof_indices=self.arm_dof_adr,
                qpos_indices=self.arm_qpos_adr,
                joint_indices=self.arm_joint_ids,
                **kwargs,
            )

        result = attempt(seed_qpos)
        if result.converged:
            return result

        # Retry from the home posture. Damped least squares is a local method, so
        # convergence depends on the seed: started from a degenerate configuration
        # (several joints pinned at their limits, as after a bare mj_resetData) it
        # gets stuck in a local minimum on targets it solves easily from a sane
        # posture. One deterministic restart removes that failure mode without
        # making callers responsible for choosing a good seed.
        if seed_qpos is None or not np.allclose(seed_qpos, HOME_ARM_QPOS):
            retry = attempt(HOME_ARM_QPOS)
            if retry.converged:
                return retry
            return retry if retry.pos_err < result.pos_err else result
        return result

    def ready_qpos(self, tcp_pos: np.ndarray | None = None, yaw: float = 0.0) -> np.ndarray:
        """Arm configuration for the episode start pose."""
        target = READY_TCP_POS if tcp_pos is None else np.asarray(tcp_pos, dtype=float)
        result = self.solve_ik(target, grasp_quat(yaw), seed_qpos=HOME_ARM_QPOS, max_iters=200)
        if not result.converged:
            raise RuntimeError(
                f"could not solve IK for ready pose {target} "
                f"(pos_err={result.pos_err:.4f} m, rot_err={result.rot_err:.4f} rad)"
            )
        return result.qpos
