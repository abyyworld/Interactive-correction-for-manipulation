"""Evaluation: success rate with a binomial confidence interval, and where it fails.

A bare success rate over 50 episodes is nearly uninformative on its own - the
95% interval is roughly +/-14 points - so every number this project reports
carries its interval. Two policies whose intervals overlap have not been shown
to differ, and saying so plainly is the difference between an honest results
table and a misleading one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..envs.phases import Phase


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Preferred over the normal approximation because success rates here are often
    near 0 or 1, where the normal interval famously produces bounds outside
    [0, 1] and badly wrong coverage.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class EvalResult:
    n: int = 0
    successes: int = 0
    episode_lengths: list[int] = field(default_factory=list)
    final_phases: list[int] = field(default_factory=list)
    interventions: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.n)

    def failure_breakdown(self) -> dict[str, int]:
        """Which phase failing episodes ended in - the actionable half of a result."""
        out: dict[str, int] = {}
        for p, ok in zip(self.final_phases, self._success_flags(), strict=False):
            if ok:
                continue
            name = Phase(p).label
            out[name] = out.get(name, 0) + 1
        return out

    def _success_flags(self) -> list[bool]:
        flags = [True] * self.successes + [False] * (self.n - self.successes)
        return flags[: len(self.final_phases)]

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.ci
        return {
            "n": self.n,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "mean_episode_length": float(np.mean(self.episode_lengths))
            if self.episode_lengths
            else float("nan"),
            "intervention_rate": self.interventions / self.n if self.n else float("nan"),
        }

    def __str__(self) -> str:
        lo, hi = self.ci
        return f"{self.successes}/{self.n} = {100 * self.success_rate:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def evaluate_agent(env, agent, n_episodes: int = 50, seed: int = 0, supervisor=None) -> EvalResult:
    """Roll out an agent and summarise. Deliberately thin - the rollout does the work."""
    from .rollout import rollout

    result = EvalResult()
    successes = 0
    for i in range(n_episodes):
        r = rollout(env, agent, supervisor=supervisor, seed=seed * 1000 + i)
        result.n += 1
        successes += int(r.success)
        result.episode_lengths.append(r.steps)
        result.final_phases.append(int(r.final_phase))
        result.interventions += int(r.intervened)
    result.successes = successes
    return result


def compare(a: EvalResult, b: EvalResult, label_a: str = "A", label_b: str = "B") -> dict[str, Any]:
    """Compare two policies, and say explicitly when the difference is not resolved."""
    lo_a, hi_a = a.ci
    lo_b, hi_b = b.ci
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    return {
        label_a: a.to_dict(),
        label_b: b.to_dict(),
        "difference": a.success_rate - b.success_rate,
        "intervals_overlap": overlap,
        "verdict": (
            f"{label_a} and {label_b} are not distinguishable at this sample size"
            if overlap
            else f"{label_a} {'>' if a.success_rate > b.success_rate else '<'} {label_b}"
        ),
    }
