"""Controlled error injection with known root cause.

Why inject faults at all
------------------------
The research question is whether a supervisor attributes an error to the phase
it actually came from. Answering it requires knowing the true answer, and on
organically-failing rollouts nobody does — the "real" cause of a failure is a
matter of interpretation, which is precisely the thing under study.

So errors are *introduced deliberately*, at a chosen phase, with a recorded
onset step. The supervisor then reacts to the consequences without being told
what was done, and their attribution can be scored against ground truth.

The delayed-consequence faults are the point
--------------------------------------------
Faults are grouped by how far the symptom is from the cause:

* ``GRASP_OFFSET``   cause APPROACH -> symptom LIFT   (the fingers close beside
                     the cube; nothing looks wrong until the arm rises empty)
* ``WRONG_OBJECT``   cause APPROACH -> symptom PLACE  (everything executes
                     perfectly, on the wrong block)
* ``WEAK_GRIP``      cause GRASP    -> symptom LIFT
* ``PREMATURE_CLOSE``cause APPROACH -> symptom GRASP  (short delay)
* ``LIFT_SLIP``      cause LIFT     -> symptom LIFT   (immediate)
* ``EARLY_RELEASE``  cause PLACE    -> symptom PLACE  (immediate)

The last two are controls. If misattribution were an artefact of the measurement
rather than a real effect, immediate-consequence faults would be misattributed
at the same rate as delayed ones. They should not be.

``WEAK_GRIP`` and ``LIFT_SLIP`` form the sharpest pair in the set: both end with
the object falling during the lift, so the *symptom is visually identical*, but
one originates in GRASP and the other in LIFT. Any difference in how they are
attributed cannot be explained by the symptom looking different, which is the
obvious confound for this kind of study.

An earlier ``LIFT_DRIFT`` fault, which added a lateral bias to the lift action,
is deliberately gone: the expert closes the loop on measured TCP position every
step, so it simply corrected the bias away and the fault had no effect at all
(15/15 episodes still succeeded). A fault has to corrupt the controller's
*intent*, not perturb its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..control.scripted import ScriptedExpert, Stage
from .phases import Phase


class FaultType(str, Enum):
    NONE = "none"
    GRASP_OFFSET = "grasp_offset"
    PREMATURE_CLOSE = "premature_close"
    WEAK_GRIP = "weak_grip"
    LIFT_SLIP = "lift_slip"
    EARLY_RELEASE = "early_release"
    WRONG_OBJECT = "wrong_object"


#: Phase each fault is introduced in, and where its effect typically surfaces.
#: ``lag`` is the qualitative distance between them and is what the study varies.
FAULT_PHASES: dict[FaultType, tuple[Phase, Phase, str]] = {
    FaultType.NONE: (Phase.DONE, Phase.DONE, "none"),
    FaultType.GRASP_OFFSET: (Phase.APPROACH, Phase.LIFT, "delayed"),
    FaultType.PREMATURE_CLOSE: (Phase.APPROACH, Phase.GRASP, "short"),
    FaultType.WEAK_GRIP: (Phase.GRASP, Phase.LIFT, "delayed"),
    FaultType.LIFT_SLIP: (Phase.LIFT, Phase.LIFT, "immediate"),
    FaultType.EARLY_RELEASE: (Phase.PLACE, Phase.PLACE, "immediate"),
    FaultType.WRONG_OBJECT: (Phase.APPROACH, Phase.PLACE, "very_delayed"),
}

#: Faults whose consequences are separated from their cause. These carry the
#: experimental signal; the others are controls.
DELAYED_FAULTS = (FaultType.GRASP_OFFSET, FaultType.WEAK_GRIP, FaultType.WRONG_OBJECT)
IMMEDIATE_FAULTS = (FaultType.LIFT_SLIP, FaultType.EARLY_RELEASE)


@dataclass
class FaultSpec:
    """A fault to inject. ``severity`` is in [0, 1] and scales the effect."""

    type: FaultType = FaultType.NONE
    severity: float = 1.0
    #: Optional explicit step at which to trigger; otherwise the fault fires
    #: when its owning stage begins.
    trigger_step: int | None = None

    @property
    def root_phase(self) -> Phase:
        return FAULT_PHASES[self.type][0]

    @property
    def expected_symptom_phase(self) -> Phase:
        return FAULT_PHASES[self.type][1]

    @property
    def lag_class(self) -> str:
        return FAULT_PHASES[self.type][2]

    def to_dict(self) -> dict:
        return {
            "fault": self.type.value,
            "severity": float(self.severity),
            "root_phase": int(self.root_phase),
            "root_phase_name": self.root_phase.label,
            "expected_symptom_phase": int(self.expected_symptom_phase),
            "lag_class": self.lag_class,
        }


@dataclass
class FaultState:
    active: bool = False
    onset_step: int | None = None  # step at which behaviour first deviated
    applied: dict = field(default_factory=dict)


class FaultInjector:
    """Applies a :class:`FaultSpec` to a :class:`ScriptedExpert`.

    Faults are expressed as modifications to the expert's *intent* (its grasp
    waypoint, its gripper command, its chosen object) rather than as noise added
    to the emitted action. That matters: an agent with a systematically wrong
    goal is what a mis-generalising policy actually looks like, whereas additive
    noise is recoverable by the next timestep and produces no delayed
    consequence to misattribute.
    """

    #: Metres of lateral grasp offset at severity 1.0, against a 4.2 cm cube.
    #: A severity sweep showed the transition is sharp: below ~20 mm the grasp
    #: still succeeds, above ~30 mm it misses entirely, and the intermediate
    #: band that grasps-then-slips is too narrow (2-4 episodes in 20) to build
    #: an experiment on. So this fault is a clean miss, and its delay comes from
    #: *observability* rather than from physics: the fingers closing beside the
    #: cube is easy to miss, and the failure becomes obvious only when the arm
    #: lifts away empty, one phase later.
    MAX_GRASP_OFFSET = 0.034
    #: Weak grip is modelled as a reduced gripper *force limit*, not a wider
    #: finger opening. Commanding a narrower opening actually grips a cube
    #: harder, so the obvious implementation makes the grasp stronger. Reducing
    #: the force limit is also the physically faithful model of a weak grasp:
    #: the fingers close normally and then fail to hold under the acceleration
    #: of the lift. Measured on this scene, the gripper holds down to ~1.2 N and
    #: slips reliably below ~0.8 N (default limit is 100 N).
    WEAK_GRIP_FORCE_LO = 0.0040  # fraction of nominal force at severity 1.0
    WEAK_GRIP_FORCE_HI = 0.0085  # at severity 0.0
    #: Gripper command during a lift slip. The cube is 42 mm across and the
    #: gripper spans 80 mm, so any opening above ~0.53 releases it.

    def __init__(self, spec: FaultSpec | None = None, rng: np.random.Generator | None = None):
        self.spec = spec or FaultSpec()
        self.rng = rng or np.random.default_rng(0)
        self.state = FaultState()
        self._target_override: str | None = None
        self.suspended = False

    def suspend(self, env) -> None:
        """Disable the fault, called when a supervisor takes control.

        A fault models *the agent's* erroneous behaviour, not a broken robot. If
        it persisted through a takeover, the human's correction would fail too
        and the correction data would be worthless - you would be measuring an
        unfixable robot rather than a fixable policy. Action-level faults simply
        stop being applied; the weak-grip fault additionally restores the gripper
        force limit it lowered.
        """
        self.suspended = True
        if self.spec.type is FaultType.WEAK_GRIP:
            env.model.actuator_forcerange[env.robot.gripper_actuator_id] = (
                env._orig_gripper_forcerange
            )

    def reset(self, env, expert: ScriptedExpert) -> None:
        self.state = FaultState()
        self._target_override = None
        self.suspended = False
        spec = self.spec
        if spec.type is FaultType.NONE:
            return

        if spec.type is FaultType.GRASP_OFFSET:
            # Fires immediately: the expert aims at the wrong point from the
            # first approach step, so onset is step 0.
            angle = self.rng.uniform(-np.pi, np.pi)
            mag = self.MAX_GRASP_OFFSET * spec.severity
            expert.grasp_offset = np.array([mag * np.cos(angle), mag * np.sin(angle), 0.0])
            self.state.active = True
            self.state.onset_step = 0
            self.state.applied = {"offset_xy": expert.grasp_offset[:2].tolist()}

        elif spec.type is FaultType.WRONG_OBJECT:
            distractors = [s.name for s in env.object_specs if s.name != env.target_name]
            if distractors:
                self._target_override = str(self.rng.choice(distractors))
                expert.reset(target_name=self._target_override)
                self.state.active = True
                self.state.onset_step = 0
                self.state.applied = {"picked": self._target_override}

        elif spec.type is FaultType.WEAK_GRIP:
            gid = env.robot.gripper_actuator_id
            scale = self.WEAK_GRIP_FORCE_HI - (
                self.WEAK_GRIP_FORCE_HI - self.WEAK_GRIP_FORCE_LO
            ) * spec.severity
            # env.reset() has already restored the nominal range, so this scales
            # the true nominal value rather than an already-weakened one.
            env.model.actuator_forcerange[gid] = env._orig_gripper_forcerange * scale
            self.state.applied = {
                "grip_force_scale": float(scale),
                "grip_force_N": float(env._orig_gripper_forcerange[1] * scale),
            }

        elif spec.type is FaultType.LIFT_SLIP:
            self.state.applied = {"release_cmd": float(0.1 + 0.5 * spec.severity)}

    def modify(self, env, expert: ScriptedExpert, action: np.ndarray, step: int) -> np.ndarray:
        """Adjust the expert's action for stage-triggered faults."""
        spec = self.spec
        if spec.type is FaultType.NONE or self.suspended:
            return action
        action = np.array(action, dtype=float)

        if spec.type is FaultType.PREMATURE_CLOSE:
            # Close the gripper while still descending, before the object is
            # between the fingers: it gets nudged away.
            if expert.stage is Stage.DESCEND and env.robot.tcp_pos[2] > env.object_pos()[2] + 0.03:
                action[6] = -1.0
                if self.state.onset_step is None:
                    self.state.active = True
                    self.state.onset_step = step

        elif spec.type is FaultType.WEAK_GRIP:
            if expert.stage in (Stage.CLOSE, Stage.LIFT, Stage.TRANSPORT, Stage.LOWER):
                if self.state.onset_step is None:
                    self.state.active = True
                    self.state.onset_step = step

        elif spec.type is FaultType.LIFT_SLIP:
            # Release once the object is clear of the table, so the failure is
            # unambiguously a lift-phase event rather than a failed grasp.
            if expert.stage is Stage.LIFT and env.object_pos()[2] > 0.5 * env.config.lift_height:
                action[6] = float(0.1 + 0.5 * spec.severity)
                if self.state.onset_step is None:
                    self.state.active = True
                    self.state.onset_step = step

        elif spec.type is FaultType.EARLY_RELEASE:
            # Open the fingers mid-transport, well before the goal.
            if expert.stage is Stage.TRANSPORT:
                dist = float(np.linalg.norm(env.robot.tcp_pos[:2] - env.goal_pos))
                # Threshold is relative to the goal radius, not an absolute
                # distance. With a fixed 0.21 m threshold the fault silently
                # never fired when the object happened to spawn near the goal,
                # so 28% of "faulty" episodes succeeded with no fault at all -
                # which would have shown up as supervisors failing to detect an
                # error that was never introduced. Object spawning already
                # excludes a band around the goal, so this condition is
                # guaranteed true when transport begins.
                if dist > env.config.goal_radius + 0.055:
                    action[6] = 1.0
                    if self.state.onset_step is None:
                        self.state.active = True
                        self.state.onset_step = step

        return np.clip(action, -1.0, 1.0)

    # ------------------------------------------------------------------ record

    def ground_truth(self, env=None) -> dict:
        gt = self.spec.to_dict()
        gt["root_onset_step"] = self.state.onset_step
        gt["fault_active"] = self.state.active
        gt.update({f"applied_{k}": v for k, v in self.state.applied.items()})
        return gt


def sample_fault(
    rng: np.random.Generator,
    types: tuple[FaultType, ...] = DELAYED_FAULTS + IMMEDIATE_FAULTS,
    severity_range: tuple[float, float] = (0.7, 1.0),
) -> FaultSpec:
    """Draw a random fault. Severity is kept high so faults reliably cause failure.

    A fault that only sometimes causes a visible failure would make "the
    supervisor did not intervene" ambiguous between "did not notice" and
    "nothing went wrong", which contaminates the detection-lag measurement.
    """
    ftype = types[int(rng.integers(len(types)))]
    return FaultSpec(type=ftype, severity=float(rng.uniform(*severity_range)))
