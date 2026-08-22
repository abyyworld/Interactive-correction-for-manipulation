"""Task phase definitions and a state-based phase tracker.

Phases are the unit of analysis for the whole project. The research question —
"when the human intervenes at step 3, was the actual error at step 1?" — is only
answerable if there is an unambiguous, controller-agnostic way to say which phase
the robot was in at any instant.

Two properties matter more than the exact boundaries:

1. **Controller independence.** The tracker reads only physical state, never a
   script's internal stage counter. A learned policy has no stage counter, and if
   phase labels came from the expert's state machine we could not label policy
   rollouts at all — which is exactly where interventions happen.
2. **Stability.** Phase labels feed credit assignment. A label that flickers
   between LIFT and PLACE for a few frames around a boundary would smear the
   correction window, so transitions require the predicate to hold for several
   consecutive control steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Phase(IntEnum):
    """The four task stages, plus terminal states."""

    APPROACH = 0  # moving toward the object, gripper open
    GRASP = 1  # positioned over the object, closing the fingers
    LIFT = 2  # object held, rising off the table
    PLACE = 3  # object at transport height, moving to and releasing over the goal
    DONE = 4  # object resting in the goal region, gripper released
    FAILED = 5  # unrecoverable: object off the table

    @property
    def label(self) -> str:
        return self.name.lower()


#: Phases a supervisor can attribute an error to. DONE/FAILED are outcomes, not
#: places where a mistake is made, so they are excluded from attribution.
ATTRIBUTABLE_PHASES: tuple[Phase, ...] = (Phase.APPROACH, Phase.GRASP, Phase.LIFT, Phase.PLACE)


@dataclass(frozen=True)
class PhaseInputs:
    """Physical quantities the tracker needs. Deliberately plain data.

    Keeping this a pure dataclass rather than reaching into ``MjData`` means the
    phase logic is unit-testable without spinning up a simulator, and the same
    tracker can label episodes replayed from disk.
    """

    tcp_pos: tuple[float, float, float]
    object_pos: tuple[float, float, float]
    goal_pos: tuple[float, float]
    grasped: bool
    gripper_width: float
    object_speed: float
    table_z: float = 0.0
    object_half_height: float = 0.021


@dataclass
class PhaseThresholds:
    """Geometric thresholds separating the phases.

    Defaults are sized relative to the 4.2 cm cube: ``grasp_xy_tol`` is roughly
    half the cube width, so "over the object" means the fingers could actually
    close on it rather than merely being nearby.
    """

    grasp_xy_tol: float = 0.035  # horizontal distance TCP-to-object to count as "over it"
    grasp_z_tol: float = 0.055  # TCP height above object centre to count as "down at it"
    lift_height: float = 0.10  # object height above the table that ends LIFT
    goal_radius: float = 0.06  # horizontal distance to goal counting as placed
    settle_speed: float = 0.02  # m/s below which the object counts as at rest
    drop_z: float = -0.05  # object below this (relative to table) has left the table
    hysteresis_steps: int = 2  # consecutive steps a new phase must hold before it is accepted


class PhaseTracker:
    """Maps physical state to a phase, with hysteresis and monotone progress memory."""

    def __init__(self, thresholds: PhaseThresholds | None = None):
        self.th = thresholds or PhaseThresholds()
        self.reset()

    def reset(self) -> None:
        self._phase = Phase.APPROACH
        self._candidate = Phase.APPROACH
        self._candidate_count = 0
        self._ever_grasped = False
        self._ever_lifted = False

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def ever_grasped(self) -> bool:
        return self._ever_grasped

    @property
    def ever_lifted(self) -> bool:
        return self._ever_lifted

    def _raw_phase(self, s: PhaseInputs) -> Phase:
        obj_height = s.object_pos[2] - s.table_z
        if obj_height < self.th.drop_z:
            return Phase.FAILED

        dx = s.object_pos[0] - s.goal_pos[0]
        dy = s.object_pos[1] - s.goal_pos[1]
        at_goal_xy = (dx * dx + dy * dy) ** 0.5 < self.th.goal_radius
        resting = obj_height < s.object_half_height * 1.6 and s.object_speed < self.th.settle_speed
        if at_goal_xy and resting and not s.grasped:
            return Phase.DONE

        if s.grasped:
            return Phase.LIFT if obj_height < self.th.lift_height else Phase.PLACE

        # Not currently holding the object. If it was already lifted and then
        # dropped, we are back to approaching it - a recovery, which is precisely
        # the situation a human correction has to handle.
        tcp_dx = s.tcp_pos[0] - s.object_pos[0]
        tcp_dy = s.tcp_pos[1] - s.object_pos[1]
        over_object = (tcp_dx * tcp_dx + tcp_dy * tcp_dy) ** 0.5 < self.th.grasp_xy_tol
        down_at_object = (s.tcp_pos[2] - s.object_pos[2]) < self.th.grasp_z_tol
        if over_object and down_at_object:
            return Phase.GRASP
        return Phase.APPROACH

    def update(self, s: PhaseInputs) -> Phase:
        """Advance the tracker one control step and return the current phase."""
        if s.grasped:
            self._ever_grasped = True
        if s.grasped and (s.object_pos[2] - s.table_z) >= self.th.lift_height:
            self._ever_lifted = True

        raw = self._raw_phase(s)
        if raw == self._phase:
            self._candidate_count = 0
            self._candidate = raw
            return self._phase

        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1

        # Terminal states are accepted immediately: waiting out the hysteresis
        # would let an episode run past a definitive success or failure.
        if raw in (Phase.DONE, Phase.FAILED) or self._candidate_count >= self.th.hysteresis_steps:
            self._phase = raw
            self._candidate_count = 0
        return self._phase
