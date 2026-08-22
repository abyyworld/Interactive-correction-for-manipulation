"""The intervention recorder: the piece most worth reusing outside this project.

Usage sketch::

    rec = InterventionRecorder("runs/session1", task="pick_place",
                               phase_names=("approach", "grasp", "lift", "place"))
    with rec.episode(seed=0, instruction="pick up the red block") as ep:
        while not done:
            if human_has_control:
                ep.human_step(action, phase=phase, proprio=obs["proprio"])
            else:
                ep.policy_step(action, phase=phase, proprio=obs["proprio"])
        ep.attribute(phase=APPROACH, confidence=0.6, notes="grasp looked offset")
        ep.finish(success=False)

Why the actor is a method name rather than a string argument: mislabelling who
produced an action silently corrupts every downstream correction label, and a
typo in ``actor="polciy"`` would pass a string check. ``policy_step`` and
``human_step`` cannot be misspelled without raising.

Segments are tracked live rather than derived at the end, so ``attribute()`` can
annotate the intervention that is still in progress — which is when a supervisor
is actually able to say what they think went wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ACTOR_EXPERT, ACTOR_HUMAN, ACTOR_POLICY, EpisodeMeta, InterventionSegment
from .store import EpisodeWriter, RunWriter


class EpisodeRecorder:
    """Records one episode, tracking intervention segments as they happen."""

    def __init__(self, writer: EpisodeWriter, phase_names: tuple[str, ...]):
        self._w = writer
        self._phase_names = phase_names
        self._segments: list[InterventionSegment] = []
        self._open: InterventionSegment | None = None
        self._t = 0

    # ------------------------------------------------------------------ steps

    def policy_step(self, action, phase: int = -1, **arrays: Any) -> None:
        self._close_segment()
        self._w.record(action=action, actor=ACTOR_POLICY, phase=phase, **arrays)
        self._t += 1

    def human_step(self, action, phase: int = -1, trigger: str = "human_gated", **arrays: Any) -> None:
        self._open_segment(phase, trigger)
        self._w.record(action=action, actor=ACTOR_HUMAN, phase=phase, **arrays)
        self._t += 1

    def expert_step(self, action, phase: int = -1, trigger: str = "scripted", **arrays: Any) -> None:
        """A synthetic supervisor. Recorded as an intervention, flagged as not human."""
        self._open_segment(phase, trigger)
        self._w.record(action=action, actor=ACTOR_EXPERT, phase=phase, **arrays)
        self._t += 1

    def _open_segment(self, phase: int, trigger: str) -> None:
        if self._open is None:
            name = self._phase_names[phase] if 0 <= phase < len(self._phase_names) else ""
            self._open = InterventionSegment(
                start=self._t, end=self._t, onset_phase=int(phase), onset_phase_name=name, trigger=trigger
            )
            self._segments.append(self._open)
        self._open.end = self._t + 1

    def _close_segment(self) -> None:
        self._open = None

    # ------------------------------------------------------------------ annotation

    def attribute(
        self,
        phase: int,
        *,
        confidence: float | None = None,
        notes: str = "",
        segment: int = -1,
    ) -> None:
        """Record where the supervisor believes the error actually occurred.

        This is the measurement the project turns on. It is stored separately from
        ``onset_phase`` (where they intervened) precisely so the two can disagree.
        """
        if not self._segments:
            raise RuntimeError("attribute() called but no intervention has occurred")
        seg = self._segments[segment]
        seg.attributed_phase = int(phase)
        seg.attributed_phase_name = (
            self._phase_names[phase] if 0 <= phase < len(self._phase_names) else ""
        )
        if confidence is not None:
            seg.confidence = float(confidence)
        if notes:
            seg.notes = notes

    @property
    def interventions(self) -> list[InterventionSegment]:
        return list(self._segments)

    @property
    def intervened(self) -> bool:
        return bool(self._segments)

    @property
    def step_count(self) -> int:
        return self._t

    # ------------------------------------------------------------------ finish

    def finish(self, success: bool, **kwargs: Any) -> EpisodeMeta:
        return self._w.finish(success, interventions=self._segments, **kwargs)

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._w.__exit__(exc_type, exc, tb)


class InterventionRecorder:
    """Facade over a run directory. One per collection session."""

    def __init__(
        self,
        root: str | Path,
        task: str,
        *,
        phase_names: tuple[str, ...] = (),
        config: dict[str, Any] | None = None,
        compress: bool = True,
        notes: str = "",
    ):
        self.phase_names = tuple(phase_names)
        self._run = RunWriter(
            root, task, config=config, phase_names=self.phase_names, compress=compress, notes=notes
        )

    @property
    def root(self) -> Path:
        return self._run.root

    def episode(self, seed: int, instruction: str = "", episode_id: str | None = None) -> EpisodeRecorder:
        return EpisodeRecorder(self._run.episode(seed, instruction, episode_id), self.phase_names)
