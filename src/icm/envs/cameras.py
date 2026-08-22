"""Offscreen RGB-D rendering for the task cameras.

One hard-won detail drives the design: **MuJoCo renderers must be created once
and reused**. Constructing a ``mujoco.Renderer`` per camera or per frame creates
a fresh GL context each time; under software rendering (OSMesa, which is what a
headless box without a GPU falls back to) every context after the first returns
black frames. That failure is silent - you get a dataset of black images and a
policy that "mysteriously fails to learn".

So the rig allocates exactly one RGB renderer and one depth renderer per distinct
resolution and switches cameras on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class CameraConfig:
    """Which cameras to render, at what size, and whether depth is captured."""

    names: tuple[str, ...] = ("wrist", "scene")
    width: int = 84
    height: int = 84
    depth: bool = True
    #: Depth beyond this (metres) is clipped. The whole workspace sits within
    #: ~1.5 m of every camera; clipping lets depth be stored as uint16 millimetres
    #: at 1 mm resolution, halving dataset size versus float32 with no real loss.
    max_depth: float = 2.0


@dataclass
class RenderStats:
    frames: int = 0
    seconds: float = 0.0
    _t: float = field(default=0.0, repr=False)


class CameraRig:
    """Renders a fixed set of cameras to RGB (uint8) and depth (uint16 mm)."""

    def __init__(self, model: mujoco.MjModel, config: CameraConfig | None = None):
        self.model = model
        self.config = config or CameraConfig()
        for name in self.config.names:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise KeyError(f"camera {name!r} not in model")

        self._rgb = mujoco.Renderer(model, self.config.height, self.config.width)
        self._depth = None
        if self.config.depth:
            self._depth = mujoco.Renderer(model, self.config.height, self.config.width)
            self._depth.enable_depth_rendering()
        self.stats = RenderStats()

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Return ``{"<cam>_rgb": uint8[H,W,3], "<cam>_depth": uint16[H,W]}``."""
        out: dict[str, np.ndarray] = {}
        scale = 1000.0  # metres -> millimetres
        max_mm = int(self.config.max_depth * scale)
        for name in self.config.names:
            self._rgb.update_scene(data, camera=name)
            out[f"{name}_rgb"] = self._rgb.render().copy()
            if self._depth is not None:
                self._depth.update_scene(data, camera=name)
                d = self._depth.render()
                out[f"{name}_depth"] = np.clip(d * scale, 0, max_mm).astype(np.uint16)
        self.stats.frames += 1
        return out

    def render_single(self, data: mujoco.MjData, camera: str, width: int, height: int) -> np.ndarray:
        """One-off high-resolution RGB frame, for GIFs and figures.

        Allocates a renderer per call, so it is only for the handful of frames
        that go into media - never inside a data-collection loop.
        """
        r = mujoco.Renderer(self.model, height, width)
        try:
            r.update_scene(data, camera=camera)
            return r.render().copy()
        finally:
            r.close()

    def close(self) -> None:
        self._rgb.close()
        if self._depth is not None:
            self._depth.close()

    @property
    def observation_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for name in self.config.names:
            shapes[f"{name}_rgb"] = (self.config.height, self.config.width, 3)
            if self.config.depth:
                shapes[f"{name}_depth"] = (self.config.height, self.config.width)
        return shapes
