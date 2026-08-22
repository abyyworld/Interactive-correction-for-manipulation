"""Correction-quality and error-attribution metrics.

The question this module answers
--------------------------------
When a supervisor intervenes in a multi-step task, the timestep they take over
at is not necessarily where the agent went wrong. A grasp mis-aligned during
APPROACH only becomes visible when the object slips during LIFT. If corrections
are credited to the phase where the *symptom* appeared, the underlying cause is
never corrected, and the policy keeps making it.

Three quantities matter:

``onset_misattribution_rate``
    How often the phase a supervisor intervened in differs from the phase the
    error actually originated in. This is *implicit* attribution — what a system
    would infer if it just used the intervention timestamp, which is what
    standard HG-DAgger does.

``stated_misattribution_rate``
    How often the supervisor's explicitly reported cause differs from the truth.
    Asking is not free, so the interesting comparison is whether asking buys
    enough accuracy to be worth the interaction cost.

``credit_iou``
    Overlap between the steps a correction actually relabels and the steps that
    were genuinely wrong. This is the quantity that predicts downstream policy
    damage: a correction window that misses the causal steps trains the policy
    on states that were already fine.

Ground truth comes from ``EpisodeMeta.ground_truth``, populated by controlled
fault injection (``root_phase``, ``root_onset_step``) or by counterfactual
rollout (``pnr_step``, the last step at which a takeover still rescues the
episode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .schema import EpisodeMeta


@dataclass
class AttributionSummary:
    """Aggregate attribution statistics over a set of episodes."""

    n_episodes: int = 0
    n_with_intervention: int = 0
    n_with_ground_truth: int = 0

    onset_misattribution_rate: float = float("nan")
    stated_misattribution_rate: float = float("nan")
    n_stated: int = 0

    mean_detection_lag: float = float("nan")  # steps between true onset and takeover
    median_detection_lag: float = float("nan")
    mean_credit_iou: float = float("nan")
    late_intervention_rate: float = float("nan")  # took over after the point of no return

    #: rows = true root phase, cols = phase the supervisor intervened in
    onset_confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    #: rows = true root phase, cols = phase the supervisor reported
    stated_confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    phase_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.__dict__.items()
        }
        d["phase_names"] = list(self.phase_names)
        return d


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection-over-union of two half-open step intervals."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def analyse(
    episodes: Iterable[EpisodeMeta],
    n_phases: int = 4,
    phase_names: tuple[str, ...] = (),
) -> AttributionSummary:
    """Compute attribution statistics over episodes carrying ground truth."""
    eps = list(episodes)
    onset_conf = np.zeros((n_phases, n_phases), dtype=int)
    stated_conf = np.zeros((n_phases, n_phases), dtype=int)

    onset_wrong = onset_total = 0
    stated_wrong = stated_total = 0
    lags: list[int] = []
    ious: list[float] = []
    late = late_total = 0
    n_gt = 0

    for ep in eps:
        gt = ep.ground_truth or {}
        root_phase = gt.get("root_phase")
        if root_phase is None or not ep.interventions:
            continue
        n_gt += 1
        root_phase = int(root_phase)
        # The first intervention is the one under test: later ones are responses
        # to the state the first correction left behind, not to the original error.
        seg = ep.interventions[0]

        if 0 <= root_phase < n_phases and 0 <= seg.onset_phase < n_phases:
            onset_conf[root_phase, seg.onset_phase] += 1
        onset_total += 1
        onset_wrong += int(seg.onset_phase != root_phase)

        if seg.attributed_phase is not None:
            stated_total += 1
            stated_wrong += int(int(seg.attributed_phase) != root_phase)
            if 0 <= root_phase < n_phases and 0 <= int(seg.attributed_phase) < n_phases:
                stated_conf[root_phase, int(seg.attributed_phase)] += 1

        root_onset = gt.get("root_onset_step")
        if root_onset is not None:
            lags.append(seg.start - int(root_onset))
            # The steps that genuinely needed correcting run from the true onset
            # to the end of the episode; compare against what was relabelled.
            ious.append(interval_iou(seg.start, seg.end, int(root_onset), ep.n_steps))

        pnr = gt.get("pnr_step")
        if pnr is not None:
            late_total += 1
            late += int(seg.start > int(pnr))

    summary = AttributionSummary(
        n_episodes=len(eps),
        n_with_intervention=sum(1 for e in eps if e.intervened),
        n_with_ground_truth=n_gt,
        onset_misattribution_rate=(onset_wrong / onset_total) if onset_total else float("nan"),
        stated_misattribution_rate=(stated_wrong / stated_total) if stated_total else float("nan"),
        n_stated=stated_total,
        mean_detection_lag=float(np.mean(lags)) if lags else float("nan"),
        median_detection_lag=float(np.median(lags)) if lags else float("nan"),
        mean_credit_iou=float(np.mean(ious)) if ious else float("nan"),
        late_intervention_rate=(late / late_total) if late_total else float("nan"),
        onset_confusion=onset_conf,
        stated_confusion=stated_conf,
        phase_names=tuple(phase_names),
    )
    return summary


def per_phase_breakdown(
    episodes: Iterable[EpisodeMeta], n_phases: int = 4
) -> dict[int, dict[str, float]]:
    """Misattribution rate split by the true root-cause phase.

    Aggregate rates hide the effect that matters: errors whose consequences are
    delayed (a bad approach that only shows up at lift) should be misattributed
    far more often than errors that are visible immediately.
    """
    buckets: dict[int, list[int]] = {p: [] for p in range(n_phases)}
    lags: dict[int, list[int]] = {p: [] for p in range(n_phases)}
    for ep in episodes:
        gt = ep.ground_truth or {}
        root = gt.get("root_phase")
        if root is None or not ep.interventions:
            continue
        root = int(root)
        if root not in buckets:
            continue
        seg = ep.interventions[0]
        buckets[root].append(int(seg.onset_phase != root))
        if gt.get("root_onset_step") is not None:
            lags[root].append(seg.start - int(gt["root_onset_step"]))
    out: dict[int, dict[str, float]] = {}
    for p, vals in buckets.items():
        if not vals:
            continue
        out[p] = {
            "n": float(len(vals)),
            "misattribution_rate": float(np.mean(vals)),
            "mean_detection_lag": float(np.mean(lags[p])) if lags[p] else float("nan"),
        }
    return out
