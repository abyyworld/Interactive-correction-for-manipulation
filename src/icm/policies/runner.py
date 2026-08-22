"""Wraps a trained policy so it can be dropped into the rollout loop as an agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .bc import BCPolicy, TemporalEnsemble


@dataclass
class RunnerConfig:
    state_key: str = "privileged"
    device: str = "cpu"
    temporal_ensemble: bool = True
    #: Higher decay weights the newest prediction more. Measured on this task:
    #: no ensembling 0/30, decay 0.35 -> 3/30, decay 1.5 -> 5/30. Blending helps,
    #: but heavy smoothing lags the closed-loop controller it is imitating.
    ensemble_decay: float = 1.5
    #: Gaussian exploration noise. Useful during DAgger collection: a perfectly
    #: deterministic policy visits a narrow state distribution and the supervisor
    #: never sees the failure modes worth correcting.
    action_noise: float = 0.0


class PolicyAgent:
    """Adapts :class:`BCPolicy` to the ``Agent`` protocol used by rollouts."""

    def __init__(
        self,
        policy: BCPolicy,
        config: RunnerConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.policy = policy
        self.config = config or RunnerConfig()
        self.rng = rng or np.random.default_rng(0)
        self.device = torch.device(self.config.device)
        self.policy.to(self.device).eval()
        self.ensemble = TemporalEnsemble(
            policy.config.chunk, policy.config.action_dim, self.config.ensemble_decay
        )
        self._vocab = policy.config.vocab

    def reset(self, env, seed: int | None = None) -> None:
        self.ensemble.reset()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _batch(self, obs: dict) -> dict[str, torch.Tensor]:
        cfg = self.policy.config
        batch = {
            "state": torch.as_tensor(obs[self.config.state_key], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        }
        for key in cfg.image_keys:
            batch[key] = torch.as_tensor(obs[key]).unsqueeze(0).to(self.device)
        if self._vocab:
            batch["instruction"] = torch.zeros(1, len(self._vocab), device=self.device)
        return batch

    @torch.no_grad()
    def act(self, env, obs: dict, info: dict, step: int) -> np.ndarray:
        chunk = self.policy(self._batch(obs))[0].float().cpu().numpy()
        action = self.ensemble.step(chunk) if self.config.temporal_ensemble else chunk[0]
        if self.config.action_noise > 0:
            action = action + self.rng.normal(0.0, self.config.action_noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)
