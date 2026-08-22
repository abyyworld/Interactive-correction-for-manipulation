"""Scripted pick-and-place expert driven by ground-truth object pose.

Role in the project
-------------------
This is not the deliverable — it is the instrument. It provides three things the
rest of the work depends on:

1. A baseline success rate the learned policy is measured against.
2. Demonstration data for the initial behaviour-cloning policy.
3. The *corrective* action source during interactive correction. When a synthetic
   supervisor takes over, it acts through this expert, so a correction is drawn
   from the same distribution a competent human would produce.

Because injected faults are meant to be the *only* source of failure in the
attribution study, the fault-free expert has to be close to perfect. Anything it
fails at on its own becomes noise in the misattribution measurement.

It emits actions in the same normalised 7-D space as the policy and the teleop
devices, rather than commanding waypoints directly. A demonstration is then
literally a sequence of policy-compatible actions, with no conversion step where
a subtle scaling bug could silently corrupt every label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ..envs.phases import Phase


class Stage(IntEnum):
    HOVER = 0  # move to a point directly above the object
    DESCEND = 1  # lower onto the grasp pose
    CLOSE = 2  # close the fingers and let them settle
    LIFT = 3  # raise the object clear of the table
    TRANSPORT = 4  # carry it over the goal
    LOWER = 5  # lower to placing height
    RELEASE = 6  # open the fingers
    RETREAT = 7  # back off so the object is visibly free
    DONE = 8

    #: Which task phase each stage belongs to. The mapping is what lets a fault
    #: injected at a *stage* be reported as an error in a *phase*.
    @property
    def phase(self) -> Phase:
        return _STAGE_PHASE[self]


_STAGE_PHASE = {
    Stage.HOVER: Phase.APPROACH,
    Stage.DESCEND: Phase.APPROACH,
    Stage.CLOSE: Phase.GRASP,
    Stage.LIFT: Phase.LIFT,
    Stage.TRANSPORT: Phase.PLACE,
    Stage.LOWER: Phase.PLACE,
    Stage.RELEASE: Phase.PLACE,
    Stage.RETREAT: Phase.PLACE,
    Stage.DONE: Phase.DONE,
}


@dataclass
class ExpertConfig:
    hover_height: float = 0.10  # above the object before descending
    grasp_z_offset: float = 0.002  # TCP height relative to object centre when grasping
    lift_height: float = 0.18  # transport altitude above the table
    place_clearance: float = 0.012  # object bottom above the pad before release
    # Gain on the target-pose update. The controller integrates target += kp*(wp - tcp),
    # so kp is a per-step convergence rate, NOT a stiffness: kp=1 is deadbeat and
    # anything above ~1.5 overshoots the waypoint every step and never settles
    # inside the arrival tolerance. An earlier value of 6.0 made HOVER and DESCEND
    # oscillate until they hit their stage timeouts, which read as "the gripper
    # never arrives" rather than "the gain is wrong".
    pos_gain: float = 0.8
    yaw_gain: float = 0.8
    pos_tol: float = 0.006  # waypoint arrival tolerance
    xy_tol: float = 0.004  # tighter tolerance before committing to a descent
    yaw_tol: float = 0.08
    close_steps: int = 12  # dwell while the fingers close and load up
    release_steps: int = 8
    stage_timeout: int = 90  # per-stage step budget; prevents silent hangs
    noise_std: float = 0.0  # optional action noise, for dataset diversity


@dataclass
class ExpertState:
    stage: Stage = Stage.HOVER
    stage_steps: int = 0
    timed_out: bool = False
    history: list[Stage] = field(default_factory=list)


def wrap_to_quarter(angle: float) -> float:
    """Fold an angle into [-pi/4, pi/4].

    A box presents an identical grasp every 90 degrees, so the gripper should
    always take the nearest equivalent yaw instead of unwinding through a large
    rotation that risks hitting a wrist joint limit.
    """
    a = (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(a)


def yaw_of_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class ScriptedExpert:
    """Waypoint state machine emitting normalised env actions."""

    def __init__(self, config: ExpertConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or ExpertConfig()
        self.rng = rng or np.random.default_rng(0)
        self.state = ExpertState()
        #: Set by fault injection; see ``icm.envs.faults``. Offsets are applied to
        #: the grasp waypoint, in metres, in the world frame.
        self.grasp_offset = np.zeros(3)
        self.gripper_close_value = -1.0
        self.target_name: str | None = None

    def reset(self, target_name: str | None = None) -> None:
        self.state = ExpertState()
        self.grasp_offset = np.zeros(3)
        self.gripper_close_value = -1.0
        self.target_name = target_name

    # ------------------------------------------------------------------ helpers

    def _goto(self, env, waypoint: np.ndarray, yaw_target: float, gripper: float) -> np.ndarray:
        """P-controller toward a waypoint, expressed as a normalised action."""
        cfg = self.config
        err = waypoint - env.robot.tcp_pos
        delta = np.clip(cfg.pos_gain * err / env.config.max_delta_pos, -1.0, 1.0)

        yaw_err = wrap_to_quarter(yaw_target - getattr(env, "_yaw", 0.0))
        dyaw = float(np.clip(cfg.yaw_gain * yaw_err / env.config.max_delta_rot, -1.0, 1.0))

        action = np.zeros(7)
        action[:3] = delta
        action[5] = dyaw
        action[6] = gripper
        if cfg.noise_std > 0:
            action[:6] += self.rng.normal(0.0, cfg.noise_std, size=6)
        return np.clip(action, -1.0, 1.0)

    def _target_object(self, env) -> str:
        return self.target_name or env.target_name

    def grasp_yaw(self, env) -> float:
        """Wrist yaw aligning the fingers with the nearest object face."""
        theta = yaw_of_quat(env.object_quat(self._target_object(env)))
        return wrap_to_quarter(-theta)

    # ------------------------------------------------------------------ policy

    def act(self, env) -> np.ndarray:
        cfg = self.config
        st = self.state
        name = self._target_object(env)
        obj = env.object_pos(name)
        goal = env.goal_pos
        tcp = env.robot.tcp_pos
        yaw = self.grasp_yaw(env)
        half = env.object_specs[0].half

        grasp_point = obj + self.grasp_offset + np.array([0.0, 0.0, cfg.grasp_z_offset])

        st.stage_steps += 1
        if st.stage_steps > cfg.stage_timeout and st.stage not in (Stage.DONE,):
            # Never hang: advancing on timeout turns "stuck forever" into a
            # recorded failure, which is analysable. A hang is not.
            st.timed_out = True
            self._advance(Stage(min(int(st.stage) + 1, int(Stage.DONE))))

        if st.stage == Stage.HOVER:
            wp = np.array([grasp_point[0], grasp_point[1], obj[2] + cfg.hover_height])
            action = self._goto(env, wp, yaw, 1.0)
            aligned = np.linalg.norm(tcp[:2] - wp[:2]) < cfg.xy_tol
            if aligned and abs(tcp[2] - wp[2]) < cfg.pos_tol and abs(wrap_to_quarter(yaw - getattr(env, "_yaw", 0.0))) < cfg.yaw_tol:
                self._advance(Stage.DESCEND)
            return action

        if st.stage == Stage.DESCEND:
            action = self._goto(env, grasp_point, yaw, 1.0)
            if np.linalg.norm(tcp - grasp_point) < cfg.pos_tol:
                self._advance(Stage.CLOSE)
            return action

        if st.stage == Stage.CLOSE:
            action = self._goto(env, grasp_point, yaw, self.gripper_close_value)
            if st.stage_steps >= cfg.close_steps:
                self._advance(Stage.LIFT)
            return action

        if st.stage == Stage.LIFT:
            wp = np.array([grasp_point[0], grasp_point[1], cfg.lift_height])
            action = self._goto(env, wp, yaw, self.gripper_close_value)
            if tcp[2] > cfg.lift_height - cfg.pos_tol:
                self._advance(Stage.TRANSPORT)
            return action

        if st.stage == Stage.TRANSPORT:
            wp = np.array([goal[0], goal[1], cfg.lift_height])
            action = self._goto(env, wp, yaw, self.gripper_close_value)
            if np.linalg.norm(tcp[:2] - wp[:2]) < cfg.pos_tol:
                self._advance(Stage.LOWER)
            return action

        if st.stage == Stage.LOWER:
            wp = np.array([goal[0], goal[1], half + cfg.place_clearance + half])
            action = self._goto(env, wp, yaw, self.gripper_close_value)
            if tcp[2] < wp[2] + cfg.pos_tol:
                self._advance(Stage.RELEASE)
            return action

        if st.stage == Stage.RELEASE:
            action = self._goto(env, tcp, yaw, 1.0)
            if st.stage_steps >= cfg.release_steps:
                self._advance(Stage.RETREAT)
            return action

        if st.stage == Stage.RETREAT:
            wp = np.array([goal[0], goal[1], cfg.lift_height])
            action = self._goto(env, wp, yaw, 1.0)
            if tcp[2] > cfg.lift_height - 0.02:
                self._advance(Stage.DONE)
            return action

        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def _advance(self, stage: Stage) -> None:
        self.state.history.append(self.state.stage)
        self.state.stage = stage
        self.state.stage_steps = 0

    @property
    def stage(self) -> Stage:
        return self.state.stage

    @property
    def done(self) -> bool:
        return self.state.stage == Stage.DONE
