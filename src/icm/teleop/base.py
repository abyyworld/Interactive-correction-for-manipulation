"""Common interface for anything that can take control from the policy.

A keyboard, a VR headset and a synthetic supervisor all answer the same three
questions each control step: *are you taking over right now*, *what action do you
want*, and *where do you think the error was*. Expressing that as one protocol
means the DAgger loop is written once and every input device — including the
simulated one used for reproducible CPU-only studies — drops straight in.

The action is in the environment's normalised 7-D space, so no device-specific
rescaling happens in the training loop, where a scaling bug would silently
corrupt every recorded correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class TeleopCommand:
    """One control step's worth of supervisor input."""

    #: Normalised 7-D action. Ignored unless ``engaged``.
    action: np.ndarray = field(default_factory=lambda: np.zeros(7))
    #: True while the supervisor is driving. The transition False->True marks the
    #: start of an intervention segment.
    engaged: bool = False
    #: Phase index the supervisor blames for the failure, reported once when
    #: they have an opinion. ``None`` means "not asked" or "no opinion".
    attribution: int | None = None
    #: Self-reported confidence in that attribution, in [0, 1].
    confidence: float | None = None
    #: Free-text rationale, collected in the human study.
    notes: str = ""
    #: Request to abandon the episode (a real operator's emergency stop).
    abort: bool = False
    #: Device-specific extras, recorded but not interpreted.
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TeleopSource(Protocol):
    """Anything that can supervise a rollout."""

    def reset(self, env: Any, seed: int | None = None) -> None:
        """Prepare for a new episode."""

    def poll(self, env: Any, obs: dict, info: dict, step: int) -> TeleopCommand:
        """Return this step's command. Must not block for longer than the control period."""

    def close(self) -> None:
        """Release any device resources."""


class NullTeleop:
    """Never intervenes. The baseline arm of every experiment."""

    def reset(self, env: Any, seed: int | None = None) -> None:
        return None

    def poll(self, env: Any, obs: dict, info: dict, step: int) -> TeleopCommand:
        return TeleopCommand(engaged=False)

    def close(self) -> None:
        return None
