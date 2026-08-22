"""Sharded, streaming episode storage.

Memory model
------------
One ``.npz`` per episode plus one ``.json`` sidecar, indexed by a top-level
``index.jsonl``. This layout is chosen for a hard constraint: datasets must be
usable on a 16 GB machine, so nothing may require holding the whole corpus in
RAM.

* ``index.jsonl`` is one small JSON object per episode. Scanning 50k episodes to
  compute statistics touches a few megabytes, never the image data.
* ``numpy.load`` on an ``.npz`` returns a lazy archive: members decompress only
  when accessed, so reading just the actions of an episode never materialises its
  images.
* Random access for training goes through a flat (episode, step) index built from
  the sidecars, so a sampler can address any frame without a global load.

The alternative — one giant HDF5 or a single array file — is faster to read but
makes appending during interactive collection awkward and turns a crash
mid-session into a corrupted dataset rather than one lost episode.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .schema import SCHEMA_VERSION, EpisodeMeta, InterventionSegment, RunMeta, segments_from_actors

INDEX_NAME = "index.jsonl"
RUN_META_NAME = "run.json"
EPISODE_DIR = "episodes"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class EpisodeWriter:
    """Accumulates one episode in memory, then flushes it to a shard.

    An episode is bounded (a few hundred steps), so buffering one is fine; it is
    the *corpus* that must never be resident. Buffering also means a crashed
    episode simply never appears, instead of leaving a truncated shard the reader
    would have to defend against.
    """

    def __init__(self, run: RunWriter, episode_id: str, seed: int, instruction: str = ""):
        self.run = run
        self.episode_id = episode_id
        self.seed = seed
        self.instruction = instruction
        self._columns: dict[str, list[Any]] = {}
        self._actors: list[str] = []
        self._phases: list[int] = []
        self._n = 0
        self._finished = False
        self.ground_truth: dict[str, Any] = {}
        self.extra: dict[str, Any] = {}

    def record(
        self,
        *,
        action,
        actor: str,
        phase: int = -1,
        **arrays: Any,
    ) -> None:
        """Append one timestep.

        ``actor`` is the load-bearing field: intervention segments are derived
        from it, so it must reflect who *actually* produced ``action``.
        """
        if self._finished:
            raise RuntimeError("cannot record into a finished episode")
        self._actors.append(actor)
        self._phases.append(int(phase))
        self._append("action", np.asarray(action))
        for key, value in arrays.items():
            self._append(key, np.asarray(value))
        self._n += 1

    def _append(self, key: str, value: np.ndarray) -> None:
        col = self._columns.setdefault(key, [])
        if len(col) != self._n:
            raise ValueError(
                f"field {key!r} appeared at step {self._n} but has {len(col)} prior entries; "
                "every step must record the same set of fields"
            )
        col.append(value)

    def finish(
        self,
        success: bool,
        *,
        interventions: list[InterventionSegment] | None = None,
        ground_truth: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        compress: bool | None = None,
    ) -> EpisodeMeta:
        if self._finished:
            raise RuntimeError("episode already finished")
        self._finished = True

        segs = (
            interventions
            if interventions is not None
            else segments_from_actors(self._actors, self._phases)
        )
        for seg in segs:
            if not seg.onset_phase_name and 0 <= seg.onset_phase < len(self.run.phase_names):
                seg.onset_phase_name = self.run.phase_names[seg.onset_phase]
            if (
                seg.attributed_phase is not None
                and not seg.attributed_phase_name
                and 0 <= seg.attributed_phase < len(self.run.phase_names)
            ):
                seg.attributed_phase_name = self.run.phase_names[seg.attributed_phase]

        counts: dict[str, int] = {}
        for a in self._actors:
            counts[a] = counts.get(a, 0) + 1

        meta = EpisodeMeta(
            episode_id=self.episode_id,
            task=self.run.meta.task,
            seed=self.seed,
            n_steps=self._n,
            success=bool(success),
            instruction=self.instruction,
            interventions=segs,
            actor_counts=counts,
            ground_truth={**self.ground_truth, **(ground_truth or {})},
            extra={**self.extra, **(extra or {})},
        )

        payload = {k: np.asarray(v) for k, v in self._columns.items()}
        payload["actor"] = np.array(self._actors, dtype="U16")
        payload["phase"] = np.asarray(self._phases, dtype=np.int16)

        shard = self.run.episode_dir / f"{self.episode_id}.npz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        use_compress = self.run.compress if compress is None else compress
        saver = np.savez_compressed if use_compress else np.savez
        fd, tmp = tempfile.mkstemp(dir=str(shard.parent), suffix=".npz.tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                saver(f, **payload)
            os.replace(tmp, shard)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        _atomic_write_text(self.run.episode_dir / f"{self.episode_id}.json", meta.to_json())
        self.run._append_index(meta)
        return meta

    # Context-manager sugar so a crash mid-episode cannot silently write a
    # half-episode: __exit__ without an explicit finish() discards it.
    def __enter__(self) -> EpisodeWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._finished = True  # discard
        elif not self._finished:
            self.finish(success=False)


class RunWriter:
    """Creates a run directory and writes episodes into it."""

    def __init__(
        self,
        root: str | Path,
        task: str,
        *,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        phase_names: tuple[str, ...] = (),
        compress: bool = True,
        notes: str = "",
    ):
        self.root = Path(root)
        self.episode_dir = self.root / EPISODE_DIR
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.phase_names = tuple(phase_names)
        self.compress = compress
        self.meta = RunMeta(
            run_id=run_id or self.root.name,
            task=task,
            config={**(config or {}), "phase_names": list(self.phase_names)},
            notes=notes,
        )
        _atomic_write_text(self.root / RUN_META_NAME, json.dumps(self.meta.to_dict(), indent=2))
        self._index_path = self.root / INDEX_NAME
        self._counter = 0

    def episode(
        self, seed: int, instruction: str = "", episode_id: str | None = None
    ) -> EpisodeWriter:
        if episode_id is None:
            episode_id = f"ep_{self._counter:06d}"
        self._counter += 1
        return EpisodeWriter(self, episode_id, seed, instruction)

    def _append_index(self, meta: EpisodeMeta) -> None:
        # Append-only: concurrent collectors can share a run, and a crash costs at
        # most the line being written rather than the whole index.
        with open(self._index_path, "a") as f:
            f.write(meta.to_json() + "\n")

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc) -> None:
        return None


class RunReader:
    """Streaming reader over a run directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / RUN_META_NAME).is_file():
            raise FileNotFoundError(
                f"{self.root} is not an interventionkit run (no {RUN_META_NAME})"
            )
        self.meta = RunMeta.from_dict(json.loads((self.root / RUN_META_NAME).read_text()))
        if self.meta.schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"run uses schema version {self.meta.schema_version}; this package supports {SCHEMA_VERSION}"
            )
        self.episode_dir = self.root / EPISODE_DIR
        self._episodes: list[EpisodeMeta] | None = None

    @property
    def phase_names(self) -> list[str]:
        return list(self.meta.config.get("phase_names", []))

    def episodes(self) -> list[EpisodeMeta]:
        """All episode metadata. Sidecars only - never touches array data."""
        if self._episodes is None:
            index = self.root / INDEX_NAME
            metas: list[EpisodeMeta] = []
            if index.is_file():
                for line in index.read_text().splitlines():
                    line = line.strip()
                    if line:
                        metas.append(EpisodeMeta.from_dict(json.loads(line)))
            else:  # index lost: rebuild from sidecars
                for path in sorted(self.episode_dir.glob("*.json")):
                    metas.append(EpisodeMeta.from_dict(json.loads(path.read_text())))
            self._episodes = metas
        return self._episodes

    def __len__(self) -> int:
        return len(self.episodes())

    def __iter__(self) -> Iterator[EpisodeMeta]:
        return iter(self.episodes())

    def load(self, episode_id: str) -> dict[str, np.ndarray]:
        """Eagerly load one episode's arrays."""
        with np.load(self.episode_dir / f"{episode_id}.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def open(self, episode_id: str):
        """Lazy handle: members decompress only on access. Close it when done."""
        return np.load(self.episode_dir / f"{episode_id}.npz", allow_pickle=False)

    def iter_arrays(
        self, keys: tuple[str, ...] | None = None
    ) -> Iterator[tuple[EpisodeMeta, dict]]:
        """Stream episodes one at a time, optionally only selected fields."""
        for meta in self.episodes():
            with self.open(meta.episode_id) as z:
                names = keys or tuple(z.files)
                yield meta, {k: z[k] for k in names if k in z.files}

    def frame_index(self) -> list[tuple[str, int]]:
        """Flat ``(episode_id, step)`` index for random-access sampling."""
        return [(m.episode_id, t) for m in self.episodes() for t in range(m.n_steps)]

    def stats(self) -> dict[str, Any]:
        eps = self.episodes()
        n = len(eps)
        if n == 0:
            return {"episodes": 0}
        n_succ = sum(e.success for e in eps)
        n_int = sum(e.intervened for e in eps)
        steps = sum(e.n_steps for e in eps)
        corrected = sum(e.n_corrected_steps for e in eps)
        return {
            "episodes": n,
            "success_rate": n_succ / n,
            "intervened_episodes": n_int,
            "intervention_rate": n_int / n,
            "total_steps": steps,
            "corrected_steps": corrected,
            "corrected_fraction": corrected / steps if steps else 0.0,
            "mean_episode_length": steps / n,
        }
