"""Versioned on-disk schema for intervention-annotated episodes.

This package is deliberately independent of MuJoCo, PyTorch and the rest of the
research code. It records *what a supervisor did and when*, which is the same
problem whether the agent is a robot arm, a browser agent or an LLM pipeline.
Keeping it dependency-light (numpy only) is what makes it reusable.

Schema stability
----------------
Every episode carries ``schema_version``. Readers refuse versions they do not
understand rather than silently misinterpreting fields — a dataset that loads
but means something different is far more expensive than one that fails loudly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1

#: Who produced the action at a given timestep.
ACTOR_POLICY = "policy"
ACTOR_HUMAN = "human"
ACTOR_EXPERT = "expert"
ACTORS = (ACTOR_POLICY, ACTOR_HUMAN, ACTOR_EXPERT)


@dataclass
class InterventionSegment:
    """One contiguous stretch of steps where a supervisor was in control.

    ``onset_phase`` is where the supervisor *took over*; ``attributed_phase`` is
    where they said the error actually occurred. The gap between the two is the
    quantity this whole package exists to measure — a supervisor who intervenes
    during LIFT but believes the mistake happened during APPROACH is telling you
    something the intervention timestamp alone cannot.
    """

    start: int  # inclusive step index
    end: int  # exclusive step index
    onset_phase: int
    onset_phase_name: str = ""
    attributed_phase: int | None = None
    attributed_phase_name: str | None = None
    trigger: str = "human_gated"  # human_gated | uncertainty | scripted
    confidence: float | None = None  # supervisor's self-reported confidence, if collected
    notes: str = ""

    @property
    def duration(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterventionSegment:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EpisodeMeta:
    """Sidecar metadata for one episode. Small enough to scan thousands of."""

    episode_id: str
    task: str
    seed: int
    n_steps: int
    success: bool
    instruction: str = ""
    interventions: list[InterventionSegment] = field(default_factory=list)
    actor_counts: dict[str, int] = field(default_factory=dict)
    #: Ground truth about deliberately injected errors, when the episode comes
    #: from a controlled study. Absent for organically collected data.
    ground_truth: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    created_utc: str = ""

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def intervened(self) -> bool:
        return len(self.interventions) > 0

    @property
    def n_corrected_steps(self) -> int:
        return sum(seg.duration for seg in self.interventions)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["interventions"] = [s.to_dict() for s in self.interventions]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeMeta:
        version = d.get("schema_version", 0)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"episode {d.get('episode_id')!r} uses schema version {version}, "
                f"but this interventionkit understands at most {SCHEMA_VERSION}. Upgrade the package."
            )
        d = dict(d)
        d["interventions"] = [InterventionSegment.from_dict(s) for s in d.get("interventions", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RunMeta:
    """Metadata for a whole collection run (many episodes)."""

    run_id: str
    task: str
    created_utc: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunMeta:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def segments_from_actors(
    actors: list[str] | Any,
    phases: list[int] | Any | None = None,
    trigger: str = "human_gated",
) -> list[InterventionSegment]:
    """Derive intervention segments from a per-step actor label sequence.

    Deriving segments rather than asking the caller to declare them means the
    boundaries can never disagree with the recorded actions — a class of bug that
    would silently mislabel which steps are corrections.
    """
    segments: list[InterventionSegment] = []
    start: int | None = None
    for i, actor in enumerate(list(actors) + [None]):
        is_human = actor in (ACTOR_HUMAN, ACTOR_EXPERT)
        if is_human and start is None:
            start = i
        elif not is_human and start is not None:
            onset_phase = int(phases[start]) if phases is not None else -1
            segments.append(
                InterventionSegment(start=start, end=i, onset_phase=onset_phase, trigger=trigger)
            )
            start = None
    return segments
