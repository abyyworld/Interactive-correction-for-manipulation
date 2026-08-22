"""Multi-phase pick-and-place environment: approach -> grasp -> lift -> place.

Design choices worth defending
------------------------------
**Cartesian delta actions, not joint targets.** The action is a change in TCP
pose plus a gripper command. VR hand tracking produces exactly this, so the human
and the policy share one action space with no translation layer - which is what
makes an intervention directly usable as a training label.

**An internal target pose, integrated across steps.** ``step`` accumulates deltas
into ``self._target_pos/_target_quat`` rather than adding them to the *measured*
TCP pose. Feeding measurement back in would let servo lag and contact
disturbances integrate into a drifting setpoint, so a policy that outputs zeros
would slowly sink into the table. It also mirrors how real teleop works.

**Full snapshot/restore.** ``get_state``/``set_state`` capture everything needed
to resume an episode bit-exactly, including the controller setpoint. This is not
a convenience: the attribution study measures the *point of no return* by
replaying an episode from a saved state and taking over at different timesteps.
Without exact restore there is no ground truth to compare human attributions to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .assets import scene as scene_mod
from .assets.scene import GOAL_POS, WORKSPACE_X, WORKSPACE_Y, ObjectSpec, SceneSpec, build_model
from .cameras import CameraConfig, CameraRig
from .panda import PandaRobot, grasp_quat
from .phases import Phase, PhaseInputs, PhaseThresholds, PhaseTracker

#: Action layout, shared by the expert, every teleop device and the policy.
ACTION_DIM = 7
ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


@dataclass
class EnvConfig:
    """Everything tunable about the task. Serialised into every dataset shard."""

    # --- observation ---
    image_size: int = 84
    cameras: tuple[str, ...] = ("wrist", "scene")
    use_depth: bool = True
    render_images: bool = True  # off => state-only, ~40x faster for study sweeps
    #: Expose ground-truth object poses in the observation. Never used by a
    #: camera-based policy, but it makes a state-based policy learnable, which is
    #: what allows the credit-assignment experiment to run on CPU in minutes
    #: instead of requiring the full image pipeline and a GPU.
    include_privileged: bool = True

    # --- control ---
    control_hz: float = 20.0
    physics_dt: float = 0.002
    max_delta_pos: float = 0.035  # metres per control step => 0.7 m/s ceiling
    max_delta_rot: float = 0.18  # radians per control step
    lock_roll_pitch: bool = True  # keep the grasp top-down; VR mode unlocks it
    ik_iters: int = 40  # per control step; the seed is the previous solution
    gravity_compensation: bool = True

    # --- task ---
    max_episode_steps: int = 220  # 11 s at 20 Hz
    n_distractors: int = 2
    randomize_objects: bool = True
    #: Choose the target object at random each episode and phrase the
    #: instruction accordingly. With this off, "pick up the red block" is the
    #: only instruction ever seen and a language-conditioned policy can ignore
    #: the text entirely while appearing to succeed - the classic way a
    #: language-conditioned benchmark measures nothing.
    randomize_target: bool = False
    min_object_separation: float = 0.11
    goal_radius: float = 0.06
    lift_height: float = 0.10
    workspace_low: tuple[float, float, float] = (0.30, -0.30, 0.012)
    workspace_high: tuple[float, float, float] = (0.72, 0.42, 0.42)

    def substeps(self) -> int:
        n = round((1.0 / self.control_hz) / self.physics_dt)
        if n < 1:
            raise ValueError("control_hz too high for physics_dt")
        return int(n)


@dataclass
class EpisodeInfo:
    """Per-step diagnostics. Everything the recorder and the study need."""

    phase: Phase = Phase.APPROACH
    grasped: bool = False
    success: bool = False
    failed: bool = False
    tcp_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    object_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ik_converged: bool = True
    ik_pos_err: float = 0.0


class PickPlaceEnv:
    """Franka Panda pick-and-place with phase labelling and exact state restore."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        cfg = self.config

        objects = list(scene_mod.DEFAULT_OBJECTS[: 1 + cfg.n_distractors])
        self.object_specs: tuple[ObjectSpec, ...] = tuple(objects)
        scene_spec = SceneSpec(
            objects=self.object_specs,
            timestep=cfg.physics_dt,
            gravity_compensation=cfg.gravity_compensation,
        )
        self.model, self.scene_xml = build_model(scene_spec)
        self.data = mujoco.MjData(self.model)

        self.robot = PandaRobot(self.model, self.data)
        self.rig: CameraRig | None = None
        if cfg.render_images:
            self.rig = CameraRig(
                self.model,
                CameraConfig(
                    names=cfg.cameras,
                    width=cfg.image_size,
                    height=cfg.image_size,
                    depth=cfg.use_depth,
                ),
            )

        self.tracker = PhaseTracker(
            PhaseThresholds(lift_height=cfg.lift_height, goal_radius=cfg.goal_radius)
        )

        # Cached ids: name lookups inside the control loop are pure overhead.
        self._obj_body = {
            o.name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.name)
            for o in self.object_specs
        }
        self._obj_qadr = {
            o.name: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o.name}_free")
            ]
            for o in self.object_specs
        }
        self._obj_dofadr = {
            o.name: self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o.name}_free")
            ]
            for o in self.object_specs
        }
        self._left_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self._right_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.target_name = self.object_specs[0].name
        self._default_target = self.object_specs[0].name

        self.goal_pos = np.array(GOAL_POS, dtype=float)
        # Faults may weaken the gripper by lowering its force limit, which is a
        # mutation of the *model*. The env owns restoring it, so an injector can
        # never leak a weakened gripper into the next episode - a bug that would
        # silently contaminate every subsequent rollout.
        self._orig_gripper_forcerange = self.model.actuator_forcerange[
            self.robot.gripper_actuator_id
        ].copy()
        self.np_random = np.random.default_rng(seed)
        # Faults may weaken the gripper by lowering its force limit, which mutates
        # the *model*. The env owns capturing and restoring the nominal value, so
        # an injector can never leak a weakened gripper into the next episode - a
        # bug that would silently contaminate every subsequent rollout.
        self._orig_gripper_forcerange = self.model.actuator_forcerange[
            self.robot.gripper_actuator_id
        ].copy()
        self._grasp_latched = False
        self._no_contact_steps = 0
        self._substeps = cfg.substeps()
        self._step_count = 0
        self._target_pos = np.zeros(3)
        self._target_quat = grasp_quat(0.0)
        self._gripper_cmd = 1.0
        self._instruction = ""
        self._last_info = EpisodeInfo()

    # ------------------------------------------------------------------ spaces

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def dt(self) -> float:
        return 1.0 / self.config.control_hz

    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        spec: dict[str, tuple[int, ...]] = {"proprio": (self.proprio().shape[0],)}
        if self.rig is not None:
            spec.update(self.rig.observation_shapes)
        return spec

    # ------------------------------------------------------------------ reset

    def _sample_object_positions(self) -> dict[str, np.ndarray]:
        """Rejection-sample non-overlapping object positions inside the workspace.

        Objects that spawn touching each other make grasps fail for reasons that
        have nothing to do with the policy, which would pollute the failure
        analysis the whole study depends on.
        """
        cfg = self.config
        placed: list[np.ndarray] = []
        out: dict[str, np.ndarray] = {}
        goal = self.goal_pos
        for spec in self.object_specs:
            for _ in range(200):
                xy = np.array(
                    [
                        self.np_random.uniform(*WORKSPACE_X),
                        self.np_random.uniform(*WORKSPACE_Y),
                    ]
                )
                if np.linalg.norm(xy - goal) < cfg.goal_radius + 0.06:
                    continue  # do not spawn on the goal pad: the task would start solved
                if all(np.linalg.norm(xy - p) >= cfg.min_object_separation for p in placed):
                    placed.append(xy)
                    out[spec.name] = np.array([xy[0], xy[1], spec.half])
                    break
            else:  # pragma: no cover - only if the workspace is mis-configured
                raise RuntimeError(
                    f"could not place {spec.name} after 200 attempts; "
                    "workspace may be too small for min_object_separation"
                )
        return out

    def reset(
        self,
        seed: int | None = None,
        instruction: str | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        cfg = self.config

        if target is not None:
            if target not in self._obj_body:
                raise KeyError(f"unknown target object {target!r}")
            self.target_name = target
        elif cfg.randomize_target:
            self.target_name = str(self.np_random.choice([s.name for s in self.object_specs]))
        else:
            self.target_name = self._default_target

        mujoco.mj_resetData(self.model, self.data)
        self.model.actuator_forcerange[self.robot.gripper_actuator_id] = (
            self._orig_gripper_forcerange
        )
        ready = self.robot.ready_qpos()
        self.robot.reset_arm(ready, gripper_opening=1.0)

        positions = (
            self._sample_object_positions()
            if cfg.randomize_objects
            else {
                s.name: np.array([0.46 + 0.10 * i, -0.10 + 0.12 * i, s.half])
                for i, s in enumerate(self.object_specs)
            }
        )
        for spec in self.object_specs:
            adr = self._obj_qadr[spec.name]
            yaw = self.np_random.uniform(-np.pi, np.pi) if cfg.randomize_objects else 0.0
            self.data.qpos[adr : adr + 3] = positions[spec.name]
            self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            dof = self._obj_dofadr[spec.name]
            self.data.qvel[dof : dof + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._target_pos = self.robot.tcp_pos.copy()
        self._target_quat = self.robot.tcp_quat.copy()
        self._gripper_cmd = 1.0
        self._step_count = 0
        self._grasp_latched = False
        self._no_contact_steps = 0
        self.tracker.reset()
        self._instruction = instruction if instruction is not None else self.default_instruction()
        self._last_info = self._compute_info()
        return self.observation()

    @property
    def target_spec(self) -> ObjectSpec:
        return next(s for s in self.object_specs if s.name == self.target_name)

    def default_instruction(self) -> str:
        """Templated instruction naming the current target object.

        The goal is described as "the pad" rather than "the green pad" on
        purpose: one of the objects is also green, and under a bag-of-words
        encoding the colour word would then appear in every instruction and stop
        discriminating between targets.
        """
        return f"pick up the {self.target_spec.noun} and place it on the pad"

    @property
    def instruction(self) -> str:
        return self._instruction

    # ------------------------------------------------------------------ step

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        cfg = self.config
        action = np.clip(np.asarray(action, dtype=float).reshape(ACTION_DIM), -1.0, 1.0)

        self._target_pos = np.clip(
            self._target_pos + action[:3] * cfg.max_delta_pos,
            np.array(cfg.workspace_low),
            np.array(cfg.workspace_high),
        )

        if cfg.lock_roll_pitch:
            # Only yaw is integrated; roll/pitch stay at the top-down grasp.
            # Re-deriving the quaternion each step (instead of composing small
            # rotations) stops numerical drift from slowly tilting the gripper.
            self._yaw = getattr(self, "_yaw", 0.0) + float(action[5]) * cfg.max_delta_rot
            self._yaw = float(np.clip(self._yaw, -np.pi / 2, np.pi / 2))
            self._target_quat = grasp_quat(self._yaw)
        else:
            delta = np.zeros(4)
            mujoco.mju_axisAngle2Quat(
                delta,
                _safe_axis(action[3:6]),
                float(np.linalg.norm(action[3:6]) * cfg.max_delta_rot),
            )
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, delta, self._target_quat)
            mujoco.mju_normalize4(new_quat)
            self._target_quat = new_quat

        # Gripper: action in [-1, 1] maps to closed..open. A continuous command
        # (rather than a binary open/close) is what a VR trigger actually gives.
        self._gripper_cmd = float((action[6] + 1.0) / 2.0)

        result = self.robot.solve_ik(
            self._target_pos,
            self._target_quat,
            seed_qpos=self.robot.arm_qpos,
            max_iters=cfg.ik_iters,
        )
        self.robot.set_arm_ctrl(result.qpos)
        self.robot.set_gripper_ctrl(self._gripper_cmd)

        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        info = self._compute_info()
        info.ik_converged = result.converged
        info.ik_pos_err = result.pos_err
        self._last_info = info

        terminated = info.success or info.failed
        truncated = (not terminated) and self._step_count >= cfg.max_episode_steps
        reward = 1.0 if info.success else 0.0
        return self.observation(), reward, terminated, truncated, self.info_dict(info)

    # ------------------------------------------------------------------ state queries

    def object_pos(self, name: str | None = None) -> np.ndarray:
        name = name or self.target_name
        return self.data.xpos[self._obj_body[name]].copy()

    def object_quat(self, name: str | None = None) -> np.ndarray:
        name = name or self.target_name
        return self.data.xquat[self._obj_body[name]].copy()

    def object_speed(self, name: str | None = None) -> float:
        name = name or self.target_name
        dof = self._obj_dofadr[name]
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    #: Consecutive contact-free control steps before a held object counts as
    #: released. Contact detection flickers: during the acceleration of a lift a
    #: finger momentarily separates, and the raw signal reads 0 for a single step
    #: while the cube is plainly still held. Measured on a healthy episode, three
    #: such dropouts occur per lift. Unfiltered they corrupt the phase label
    #: (LIFT briefly becomes APPROACH) and make any "object dropped" detector
    #: fire on perfectly good episodes.
    GRASP_RELEASE_STEPS = 3

    def is_grasped(self, name: str | None = None) -> bool:
        """True when the object is held. Debounced for the target object.

        Contact-based rather than width-based: a width threshold cannot tell
        "holding the cube" from "closed on empty air next to the cube", and that
        distinction is the difference between a successful grasp and the most
        common failure mode.
        """
        if name is None or name == self.target_name:
            return self._grasp_latched
        return self.contact_grasp(name)

    def contact_grasp(self, name: str | None = None) -> bool:
        """Raw, unfiltered contact test: both fingers touching ``name``."""
        name = name or self.target_name
        body = self._obj_body[name]
        left = right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == body:
                other = b2
            elif b2 == body:
                other = b1
            else:
                continue
            if other == self._left_finger:
                left = True
            elif other == self._right_finger:
                right = True
            if left and right:
                return True
        return False

    def geometric_hold(self, name: str | None = None) -> bool:
        """Whether the fingers are closed *on the object* rather than on air.

        Contact detection alone proved unreliable: MuJoCo drops finger contacts
        for several consecutive steps under lift acceleration even while the cube
        is plainly held (observed with gripper width 0.042 m on a 0.042 m cube,
        object rising, contacts reporting nothing). This geometric test is the
        complement - the finger separation matches the object width and the
        object sits between the fingertips - and it cannot be fooled by the
        common failure it needs to exclude: closing on empty air drives the width
        to ~0, which does not match the object width at all.
        """
        name = name or self.target_name
        spec = next(s for s in self.object_specs if s.name == name)
        width_matches = abs(self.robot.gripper_width - 2.0 * spec.half) < 0.008
        if not width_matches:
            return False
        return float(np.linalg.norm(self.object_pos(name) - self.robot.tcp_pos)) < 0.05

    def _update_grasp_latch(self) -> bool:
        raw = self.contact_grasp() or self.geometric_hold()
        if raw:
            self._no_contact_steps = 0
        else:
            self._no_contact_steps += 1
        self._grasp_latched = raw or (
            self._grasp_latched and self._no_contact_steps < self.GRASP_RELEASE_STEPS
        )
        return self._grasp_latched

    def _compute_info(self) -> EpisodeInfo:
        obj = self.object_pos()
        grasped = self._update_grasp_latch()
        phase = self.tracker.update(
            PhaseInputs(
                tcp_pos=tuple(self.robot.tcp_pos),
                object_pos=tuple(obj),
                goal_pos=tuple(self.goal_pos),
                grasped=grasped,
                gripper_width=self.robot.gripper_width,
                object_speed=self.object_speed(),
                object_half_height=self.target_spec.half,
            )
        )
        return EpisodeInfo(
            phase=phase,
            grasped=grasped,
            success=phase == Phase.DONE,
            failed=phase == Phase.FAILED,
            tcp_pos=self.robot.tcp_pos,
            object_pos=obj,
        )

    def info_dict(self, info: EpisodeInfo | None = None) -> dict[str, Any]:
        info = info or self._last_info
        return {
            "phase": int(info.phase),
            "phase_name": info.phase.label,
            "grasped": bool(info.grasped),
            "success": bool(info.success),
            "failed": bool(info.failed),
            "step": self._step_count,
            "tcp_pos": info.tcp_pos.copy(),
            "object_pos": info.object_pos.copy(),
            "ik_converged": bool(info.ik_converged),
            "ik_pos_err": float(info.ik_pos_err),
        }

    # ------------------------------------------------------------------ observation

    def proprio(self) -> np.ndarray:
        """Low-dimensional observation: what a real robot reports about itself.

        Deliberately excludes object poses. A policy given ground-truth object
        state would learn a shortcut that no camera-only deployment can reproduce,
        and the generalisation numbers would be meaningless.
        """
        mat = self.robot.tcp_mat
        return np.concatenate(
            [
                self.robot.tcp_pos,
                mat[:, 0],  # 6D rotation representation: continuous, unlike
                mat[:, 1],  # quaternions, which have a sign ambiguity that
                [self.robot.gripper_width],  # makes regression targets discontinuous
                self.robot.arm_qpos,
                self.robot.arm_qvel,
            ]
        ).astype(np.float32)

    def privileged_state(self) -> np.ndarray:
        """Ground-truth object poses. For the scripted expert and analysis only."""
        parts = [self.robot.tcp_pos, [self.robot.gripper_width]]
        for spec in self.object_specs:
            parts.append(self.object_pos(spec.name))
            parts.append(self.object_quat(spec.name))
        parts.append(self.goal_pos)
        return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts]).astype(
            np.float32
        )

    def observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {"proprio": self.proprio()}
        if self.config.include_privileged:
            obs["privileged"] = self.privileged_state()
        if self.rig is not None:
            obs.update(self.rig.render(self.data))
        return obs

    def render_frame(
        self, camera: str = "scene", width: int = 480, height: int = 360
    ) -> np.ndarray:
        rig = self.rig
        if rig is None:
            rig = CameraRig(
                self.model, CameraConfig(names=(camera,), width=8, height=8, depth=False)
            )
        return rig.render_single(self.data, camera, width, height)

    # ------------------------------------------------------------------ snapshot / restore

    def get_state(self) -> dict[str, Any]:
        """Full snapshot sufficient to resume the episode bit-exactly."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "act": self.data.act.copy() if self.data.act.size else np.zeros(0),
            "time": float(self.data.time),
            "target_pos": self._target_pos.copy(),
            "target_quat": self._target_quat.copy(),
            "yaw": float(getattr(self, "_yaw", 0.0)),
            "gripper_cmd": float(self._gripper_cmd),
            "step_count": int(self._step_count),
            "grasp_latched": bool(self._grasp_latched),
            "no_contact_steps": int(self._no_contact_steps),
            "tracker": {
                "phase": int(self.tracker.phase),
                "ever_grasped": self.tracker.ever_grasped,
                "ever_lifted": self.tracker.ever_lifted,
            },
            "instruction": self._instruction,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]
        if self.data.act.size:
            self.data.act[:] = state["act"]
        self.data.time = state["time"]
        self._target_pos = np.array(state["target_pos"], dtype=float)
        self._target_quat = np.array(state["target_quat"], dtype=float)
        self._yaw = float(state.get("yaw", 0.0))
        self._gripper_cmd = float(state["gripper_cmd"])
        self._step_count = int(state["step_count"])
        self._grasp_latched = bool(state.get("grasp_latched", False))
        self._no_contact_steps = int(state.get("no_contact_steps", 0))
        self._instruction = state.get("instruction", self._instruction)
        tr = state["tracker"]
        self.tracker.restore(tr["phase"], tr["ever_grasped"], tr["ever_lifted"])
        mujoco.mj_forward(self.model, self.data)
        self._last_info = self._compute_info()

    def close(self) -> None:
        if self.rig is not None:
            self.rig.close()
            self.rig = None


def _safe_axis(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(v, dtype=float) / n
