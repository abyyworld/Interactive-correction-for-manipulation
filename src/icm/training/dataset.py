"""Streaming dataset over interventionkit runs.

Memory constraint drives every decision here. The target machine has 16 GB of
system RAM and an 8 GB GPU, and a 10k-episode image dataset is tens of
gigabytes, so nothing may assume the corpus fits in memory:

* The index is built from the JSONL sidecars, touching no array data.
* Episodes are opened lazily and cached in a small LRU, so a batch of random
  frames touches a handful of shards rather than the whole run.
* Images stay ``uint8`` on disk and until the moment they are normalised, which
  is a 4x memory saving over float32 across the whole pipeline.

``estimate_size`` exists so that a mis-specified collection run fails at the
planning stage rather than after filling the disk.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from interventionkit import RunReader

from .weighting import CreditAssignment, corrected_span


@dataclass
class DatasetConfig:
    """What to feed the policy and which frames count as supervision."""

    #: Frames to train on. "corrections" is HG-DAgger's rule: only supervisor
    #: actions are supervision, because the policy's own actions are exactly the
    #: behaviour we are trying to change.
    supervision: str = "corrections"  # corrections | all | demos
    credit: CreditAssignment = CreditAssignment.ONSET
    #: Number of future actions predicted per observation. Chunking is the single
    #: biggest win in behaviour cloning for manipulation: it removes most of the
    #: compounding error from single-step prediction.
    chunk: int = 8
    image_keys: tuple[str, ...] = ()
    state_key: str = "proprio"
    #: Cap on simultaneously open shards. Sized for image episodes (~5 MB each),
    #: where 32 is already 160 MB. State-only episodes are ~100 kB, so the cache
    #: is widened automatically when no image keys are requested - with 200
    #: episodes and a 32-shard cache, random sampling reloads a shard on most
    #: accesses and the loader becomes the training bottleneck.
    cache_size: int = 32
    state_only_cache_size: int = 512
    correction_weight: float = 1.0
    demo_weight: float = 1.0


class EpisodeCache:
    """Small LRU over open ``.npz`` handles."""

    def __init__(self, reader: RunReader, size: int = 32):
        self.reader = reader
        self.size = size
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, episode_id: str) -> dict[str, np.ndarray]:
        if episode_id in self._cache:
            self._cache.move_to_end(episode_id)
            return self._cache[episode_id]
        data = self.reader.load(episode_id)
        self._cache[episode_id] = data
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return data

    def clear(self) -> None:
        self._cache.clear()


class InterventionDataset:
    """Frame-level dataset with credit-assignment-aware supervision masks.

    Deliberately framework-agnostic (returns numpy). ``to_torch`` wraps it for
    PyTorch so the core indexing logic stays testable without importing torch.
    """

    def __init__(self, roots: str | Path | list, config: DatasetConfig | None = None):
        self.config = config or DatasetConfig()
        roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
        self.readers = [RunReader(r) for r in roots]
        cache_size = (
            self.config.cache_size
            if self.config.image_keys
            else max(self.config.cache_size, self.config.state_only_cache_size)
        )
        self.caches = [EpisodeCache(r, cache_size) for r in self.readers]

        self.samples: list[tuple[int, str, int, float]] = []  # (reader, episode, t, weight)
        self._build_index()

    def _build_index(self) -> None:
        cfg = self.config
        for ri, reader in enumerate(self.readers):
            for meta in reader.episodes():
                spans = []
                for seg in meta.interventions:
                    start, end = corrected_span(seg, meta, cfg.credit)
                    spans.append((start, end))

                for t in range(meta.n_steps):
                    in_correction = any(s <= t < e for s, e in spans)
                    if cfg.supervision == "corrections":
                        if not in_correction:
                            continue
                        weight = cfg.correction_weight
                    elif cfg.supervision == "demos":
                        # Successful uncorrected episodes only: plain BC data.
                        if meta.intervened or not meta.success:
                            continue
                        weight = cfg.demo_weight
                    else:  # "all"
                        weight = cfg.correction_weight if in_correction else cfg.demo_weight
                    self.samples.append((ri, meta.episode_id, t, weight))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        cfg = self.config
        ri, episode_id, t, weight = self.samples[idx]
        data = self.caches[ri].get(episode_id)

        actions = data["action"]
        n = len(actions)
        # Pad the chunk by repeating the final action: at the end of an episode
        # the correct behaviour is to hold still, which is what a repeated last
        # action encodes. Zero-padding would instead teach a jerk to the origin.
        idxs = np.minimum(np.arange(t, t + cfg.chunk), n - 1)
        chunk = actions[idxs].astype(np.float32)
        valid = (np.arange(t, t + cfg.chunk) < n).astype(np.float32)

        out: dict[str, np.ndarray] = {
            "action": chunk,
            "action_valid": valid,
            "weight": np.float32(weight),
        }
        if cfg.state_key in data:
            out["state"] = data[cfg.state_key][t].astype(np.float32)
        for key in cfg.image_keys:
            if key in data:
                out[key] = data[key][t]  # uint8; normalised on device
        return out

    def subsample(self, n: int, seed: int = 0) -> None:
        """Randomly restrict to ``n`` frames, in place.

        Used to control for dataset size when comparing credit-assignment
        strategies. Rewinding further necessarily yields more corrective frames,
        so an uncontrolled comparison cannot separate "corrected the right
        states" from "simply had more data".
        """
        if n >= len(self.samples):
            return
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(self.samples), size=n, replace=False)
        self.samples = [self.samples[i] for i in sorted(keep)]

    # ------------------------------------------------------------------ stats

    def action_stats(self, max_samples: int = 20000) -> tuple[np.ndarray, np.ndarray]:
        """Mean/std of actions for normalisation, over a bounded sample."""
        step = max(1, len(self) // max_samples)
        acc = [self[i]["action"][0] for i in range(0, len(self), step)]
        arr = np.stack(acc) if acc else np.zeros((1, 7), dtype=np.float32)
        return arr.mean(0), arr.std(0) + 1e-6

    def state_stats(self, max_samples: int = 20000) -> tuple[np.ndarray, np.ndarray]:
        step = max(1, len(self) // max_samples)
        acc = [self[i]["state"] for i in range(0, len(self), step) if "state" in self[i]]
        if not acc:
            return np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)
        arr = np.stack(acc)
        return arr.mean(0), arr.std(0) + 1e-6

    def summary(self) -> dict[str, Any]:
        eps = {(ri, e) for ri, e, _, _ in self.samples}
        return {
            "frames": len(self.samples),
            "episodes_contributing": len(eps),
            "runs": len(self.readers),
            "credit": self.config.credit.value,
            "supervision": self.config.supervision,
        }


def estimate_size(
    n_episodes: int,
    steps_per_episode: int = 120,
    image_size: int = 84,
    n_cameras: int = 2,
    depth: bool = True,
    compression: float = 0.35,
) -> dict[str, float]:
    """Predict dataset size before generating it.

    Generating a dataset that does not fit on disk is a several-hour mistake,
    and the arithmetic is easy to get wrong by an order of magnitude. At 84 px,
    two cameras with depth, 10k episodes of 120 steps is roughly 150 GB raw.
    """
    per_frame = n_cameras * image_size * image_size * 3
    if depth:
        per_frame += n_cameras * image_size * image_size * 2
    per_frame += 24 * 4 + 7 * 4  # proprio + action
    raw = per_frame * steps_per_episode * n_episodes
    return {
        "bytes_per_frame": float(per_frame),
        "raw_gb": raw / 1e9,
        "compressed_gb": raw * compression / 1e9,
        "frames": float(steps_per_episode * n_episodes),
    }
