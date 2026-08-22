"""interventionkit - record, index and analyse human interventions in agent rollouts.

A small, dependency-light toolkit for interactive imitation learning and agent
oversight. It answers one question well: when a supervisor corrected the agent,
*where did the error actually come from*, and did the correction land on it?

    from interventionkit import InterventionRecorder, RunReader, analyse

    rec = InterventionRecorder("runs/demo", task="pick_place",
                               phase_names=("approach", "grasp", "lift", "place"))
    with rec.episode(seed=0) as ep:
        ep.policy_step(action, phase=0, proprio=obs)
        ep.human_step(correction, phase=2, proprio=obs)
        ep.attribute(phase=0, notes="approach was offset")
        ep.finish(success=False, ground_truth={"root_phase": 0, "root_onset_step": 5})

    print(analyse(RunReader("runs/demo").episodes()).onset_misattribution_rate)
"""

from .attribution import AttributionSummary, analyse, interval_iou, per_phase_breakdown
from .recorder import EpisodeRecorder, InterventionRecorder
from .report import build_markdown, build_report, write_report
from .schema import (
    ACTOR_EXPERT,
    ACTOR_HUMAN,
    ACTOR_POLICY,
    SCHEMA_VERSION,
    EpisodeMeta,
    InterventionSegment,
    RunMeta,
    segments_from_actors,
)
from .store import RunReader, RunWriter

__version__ = "0.1.0"

__all__ = [
    "ACTOR_EXPERT",
    "ACTOR_HUMAN",
    "ACTOR_POLICY",
    "SCHEMA_VERSION",
    "AttributionSummary",
    "EpisodeMeta",
    "EpisodeRecorder",
    "InterventionRecorder",
    "InterventionSegment",
    "RunMeta",
    "RunReader",
    "RunWriter",
    "analyse",
    "build_markdown",
    "build_report",
    "interval_iou",
    "per_phase_breakdown",
    "segments_from_actors",
    "write_report",
    "__version__",
]
