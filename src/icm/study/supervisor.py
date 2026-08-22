"""A simulated supervisor: detects failures, takes over, and reports a cause.

Why simulate a human at all
---------------------------
The flagship experiment puts a person in VR and measures how they attribute
errors. That study needs a headset, participants and ethics approval. This model
is what makes everything *around* it buildable and testable first: it exercises
the full intervention pipeline, produces reproducible CPU-only results, and —
most importantly — lets the downstream question be answered as a *function* of
attribution accuracy rather than at a single unknown value.

That last point is the design's whole justification. We do not know what a real
human's tracing accuracy is. But we can measure how much policy performance
degrades at every possible value, so that when the human number arrives from the
VR study it can be read straight off the curve.

What this is not
----------------
It is **not** a validated model of human behaviour and no result from it should
be reported as a human result. ``trace_accuracy`` in particular is a free
parameter, not a measurement. The honest claim is: *if* supervisors trace causes
correctly a fraction p of the time, *then* the trained policy loses this much
performance.

Detection is deliberately restricted to what a person watching the screen could
actually see. It never reads the injected fault. So a fault whose symptom is
subtle is detected late by construction, which is the effect under study rather
than something hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..control.scripted import ExpertConfig, ScriptedExpert
from ..envs.phases import ATTRIBUTABLE_PHASES, Phase
from ..teleop.base import TeleopCommand


@dataclass
class SupervisorConfig:
    """Parameters of the simulated supervisor. All are swept, none are measured."""

    #: Reaction latency between noticing a problem and taking control, in control
    #: steps. At 20 Hz, a mean of 6 steps is 300 ms - the low end of human
    #: visual choice-reaction time, appropriate for an alert operator watching
    #: for exactly one kind of event.
    reaction_mean: float = 6.0
    reaction_std: float = 3.0
    max_reaction: int = 40

    #: Probability of ever noticing the failure. Below 1.0 some episodes simply
    #: run to completion uncorrected, as they do with real operators.
    detection_prob: float = 0.95

    #: Probability the supervisor correctly traces the failure back to its true
    #: root-cause phase, rather than blaming the phase where they noticed it.
    #: THE key swept parameter. 0.0 = pure recency bias, 1.0 = perfect causal
    #: reasoning. Real humans are somewhere in between and this study is
    #: designed to find out where.
    trace_accuracy: float = 0.35

    #: Probability of reporting an attribution at all when asked.
    report_prob: float = 1.0

    #: Steps of no phase progress before a stall is *considered*. Phase duration
    #: alone is not evidence of a stall - a healthy approach legitimately runs
    #: 76 steps, and thresholding on duration alone manufactured interventions on
    #: 20% of perfectly good episodes. A stall additionally requires the robot to
    #: have stopped moving.
    stall_patience: int = 120
    #: Robot displacement below which motion counts as absent, metres.
    stall_motion_eps: float = 0.012
    stall_motion_window: int = 20

    #: Once engaged, stay engaged for the rest of the episode. Real human-gated
    #: schemes usually hand back, but holding control keeps the correction
    #: segment unambiguous, which matters for credit assignment.
    hold_until_end: bool = True
    min_takeover_steps: int = 25


@dataclass
class DetectionEvent:
    step: int
    reason: str
    phase: Phase


@dataclass
class SupervisorState:
    detected: DetectionEvent | None = None
    takeover_step: int | None = None
    engaged: bool = False
    will_detect: bool = True
    reaction_delay: int = 0
    reported: bool = False
    last_phase: Phase = Phase.APPROACH
    last_phase_change: int = 0
    ever_grasped: bool = False
    tcp_history: list[np.ndarray] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


class SyntheticSupervisor:
    """Detects observable failures, takes over via the scripted expert, attributes a cause."""

    def __init__(
        self,
        config: SupervisorConfig | None = None,
        expert_config: ExpertConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.config = config or SupervisorConfig()
        self.rng = rng or np.random.default_rng(0)
        self.expert = ScriptedExpert(expert_config or ExpertConfig(), rng=self.rng)
        self.state = SupervisorState()
        #: True root-cause phase, supplied by the study harness. Used *only* to
        #: model whether the supervisor traces correctly - never to detect.
        self.true_root_phase: Phase | None = None
        self._on_engage = None

    def reset(self, env, seed: int | None = None, true_root_phase: Phase | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.expert.rng = self.rng
        self.state = SupervisorState()
        self.state.will_detect = bool(self.rng.random() < self.config.detection_prob)
        delay = self.rng.normal(self.config.reaction_mean, self.config.reaction_std)
        self.state.reaction_delay = int(np.clip(round(delay), 1, self.config.max_reaction))
        self.true_root_phase = true_root_phase
        self.expert.reset()

    def set_engage_callback(self, fn) -> None:
        """Called once when the supervisor first takes control (used to suspend faults)."""
        self._on_engage = fn

    # ------------------------------------------------------------------ detection

    def _detect(self, env, info: dict, step: int) -> DetectionEvent | None:
        """Observable anomaly predicates. Never reads the injected fault.

        Every predicate has to survive a *healthy* episode without firing. A
        false positive is not a harmless extra correction: it records an
        intervention with no ground-truth root cause, which silently biases the
        misattribution denominator.
        """
        phase = Phase(info["phase"])
        if info["success"] or phase is Phase.DONE:
            return None
        obj = np.asarray(info["object_pos"])
        grasped = bool(info["grasped"])
        tcp = np.asarray(info["tcp_pos"])
        goal_dist = float(np.linalg.norm(obj[:2] - env.goal_pos))
        gripper_closing = env._gripper_cmd < 0.45

        if grasped:
            self.state.ever_grasped = True

        # 1. The arm is rising with the fingers shut and nothing in them. This is
        #    what a failed grasp actually looks like to an observer: the moment of
        #    closing beside the cube is easy to miss, the empty lift is not.
        if (
            gripper_closing
            and not grasped
            and tcp[2] > obj[2] + 0.07
            and obj[2] < 0.06
            and env.robot.gripper_width < 0.030
        ):
            return DetectionEvent(step, "empty_lift", phase)

        # 2. Held, then not held, and not because it was placed on the goal.
        if self.state.ever_grasped and not grasped and goal_dist > env.config.goal_radius + 0.02:
            if obj[2] < env.config.lift_height * 0.8:
                return DetectionEvent(step, "object_dropped", phase)

        # 3. Carrying the wrong object entirely.
        for spec in env.object_specs[1:]:
            if env.is_grasped(spec.name):
                return DetectionEvent(step, "wrong_object", phase)

        # 4. The object was knocked away rather than grasped. The goal-distance
        #    guard is essential: a normal release drops the object from a few
        #    centimetres up, which is indistinguishable from a knock on speed
        #    alone and fired on most healthy episodes without it.
        if (
            not grasped
            and env.object_speed() > 0.15
            and tcp[2] < obj[2] + 0.12
            and goal_dist > env.config.goal_radius + 0.03
        ):
            return DetectionEvent(step, "object_knocked", phase)

        # 5. Nothing is happening: no phase progress *and* the arm has stopped.
        self.state.tcp_history.append(tcp.copy())
        if len(self.state.tcp_history) > self.config.stall_motion_window + 1:
            self.state.tcp_history.pop(0)
        if phase != self.state.last_phase:
            self.state.last_phase = phase
            self.state.last_phase_change = step
        elif step - self.state.last_phase_change > self.config.stall_patience:
            hist = self.state.tcp_history
            if len(hist) > self.config.stall_motion_window:
                moved = float(np.linalg.norm(hist[-1] - hist[0]))
                if moved < self.config.stall_motion_eps:
                    return DetectionEvent(step, "stalled", phase)

        return None

    # ------------------------------------------------------------------ attribution

    def _attribute(self, symptom_phase: Phase) -> Phase:
        """Report a cause: the truth, or the phase where the symptom appeared."""
        if self.true_root_phase is not None and self.rng.random() < self.config.trace_accuracy:
            return self.true_root_phase
        if symptom_phase in ATTRIBUTABLE_PHASES:
            return symptom_phase
        # Symptom surfaced in a terminal state; blame the last real phase.
        return Phase.PLACE

    # ------------------------------------------------------------------ interface

    def poll(self, env, obs: dict, info: dict, step: int) -> TeleopCommand:
        st = self.state
        cfg = self.config

        if not st.engaged:
            if st.will_detect and st.detected is None:
                event = self._detect(env, info, step)
                if event is not None:
                    st.detected = event
                    st.history.append(f"detected {event.reason} at step {event.step}")
            if st.detected is not None and step >= st.detected.step + st.reaction_delay:
                st.engaged = True
                st.takeover_step = step
                # Re-plan from the current state rather than from scratch: a
                # plain reset opens the gripper, dropping anything currently
                # held and manufacturing a failure the takeover itself caused.
                self.expert.resume_from_state(env)
                if self._on_engage is not None:
                    self._on_engage()
            if not st.engaged:
                return TeleopCommand(engaged=False)

        action = self.expert.act(env)
        attribution = None
        confidence = None
        notes = ""
        if not st.reported and st.detected is not None:
            if self.rng.random() < cfg.report_prob:
                phase = self._attribute(st.detected.phase)
                attribution = int(phase)
                # Confidence is higher when blaming what they just saw, which is
                # the pattern that makes recency bias hard to filter out by
                # thresholding on confidence.
                blamed_symptom = phase == st.detected.phase
                confidence = float(
                    np.clip(self.rng.normal(0.75 if blamed_symptom else 0.55, 0.12), 0, 1)
                )
                notes = st.detected.reason
            st.reported = True

        return TeleopCommand(
            action=action,
            engaged=True,
            attribution=attribution,
            confidence=confidence,
            notes=notes,
            extra={
                "takeover_step": st.takeover_step,
                "detect_reason": st.detected.reason if st.detected else "",
                # Phase at the moment the symptom became visible. Distinct from
                # the phase at takeover: after a drop the tracker legitimately
                # reverts to APPROACH, so by the time a human has reacted the
                # robot no longer looks like it is in the phase where the failure
                # appeared.
                "detect_phase": int(st.detected.phase) if st.detected else -1,
                "detect_step": st.detected.step if st.detected else -1,
            },
        )

    def close(self) -> None:
        return None
