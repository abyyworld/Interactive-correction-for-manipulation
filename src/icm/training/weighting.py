"""Credit assignment: which steps a correction is allowed to relabel.

This is where misattribution turns into policy damage, and it is the mechanism
the second half of the project measures.

When a supervisor takes over at step *t*, standard HG-DAgger treats the steps
from *t* onward as supervision and leaves everything before *t* alone. If the
error actually originated 40 steps earlier, those causal states are never
corrected: the policy keeps producing the behaviour that led to the failure and
only learns to clean up afterwards.

The strategies below differ only in where the corrected span *starts*:

``ONSET``     from the takeover step. What HG-DAgger does today.
``SYMPTOM``   from the step the failure first became visible - the best a
              timestamp-based scheme could do.
``STATED``    from the beginning of the phase the supervisor blamed. This is what
              a system that *asks* can implement, and it is only as good as the
              supervisor's tracing.
``ORACLE``    from the true root-cause onset. Not implementable in practice; it
              is the ceiling the others are measured against.

Comparing ORACLE against STATED at a given tracing accuracy is exactly the price
of misattribution, in units of task success.
"""

from __future__ import annotations

from enum import Enum


class CreditAssignment(str, Enum):
    ONSET = "onset"
    SYMPTOM = "symptom"
    STATED = "stated"
    ORACLE = "oracle"


def corrected_span(segment, meta, strategy: CreditAssignment) -> tuple[int, int]:
    """Return the ``[start, end)`` step range a correction is credited to.

    The end is always the end of the recorded intervention: only steps where a
    supervisor actually acted carry a corrective action. The strategies move the
    *start* earlier, which in the rewind protocol determines how far back the
    episode is replayed before re-demonstrating.
    """
    gt = meta.ground_truth or {}
    start, end = segment.start, segment.end

    if strategy is CreditAssignment.ONSET:
        return start, end

    if strategy is CreditAssignment.SYMPTOM:
        s = gt.get("symptom_step")
        return (int(s) if s is not None else start), end

    if strategy is CreditAssignment.STATED:
        phase = segment.attributed_phase
        if phase is None:
            return start, end
        s = phase_first_step(meta, int(phase))
        return (s if s is not None else start), end

    if strategy is CreditAssignment.ORACLE:
        s = gt.get("root_onset_step")
        return (int(s) if s is not None else start), end

    raise ValueError(f"unknown credit assignment strategy: {strategy}")


def phase_first_step(meta, phase: int) -> int | None:
    """First step of the given phase, from the per-episode phase timeline.

    Stored on the episode by the collector; falls back to None when a run
    predates that field so callers degrade to ONSET rather than crashing.
    """
    timeline = (meta.extra or {}).get("phase_timeline")
    if not timeline:
        return None
    for step, p in enumerate(timeline):
        if int(p) == phase:
            return step
    return None
