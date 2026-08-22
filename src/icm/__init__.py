"""Interactive correction for manipulation.

Selecting a MuJoCo rendering backend
------------------------------------
MuJoCo picks its offscreen GL backend from the ``MUJOCO_GL`` environment
variable, and the right value differs per platform. Hard-coding one (an earlier
Makefile exported ``osmesa`` unconditionally) makes the project Linux-only:
macOS renders through CGL and Windows through WGL, and neither accepts
``osmesa``.

So the choice is made here, at import, before any MuJoCo object exists — the
backend is bound at first renderer construction, so setting it later silently
has no effect. An explicit ``MUJOCO_GL`` set by the user always wins.
"""

from __future__ import annotations

import ctypes.util
import os
import pathlib
import sys

__version__ = "0.1.0"


def _has_gpu() -> bool:
    """Whether a real GPU is present, as opposed to a software GL stack.

    Mesa provides an EGL implementation even with no GPU, so the presence of
    libEGL proves nothing. Choosing EGL on such a machine renders correctly and
    then raises EGLError from its destructor on every renderer teardown, which
    looks like a crash to anyone running the project for the first time.
    """
    if any(pathlib.Path("/dev/dri").glob("render*")):
        return True
    return bool(ctypes.util.find_library("nvidia-ml") or ctypes.util.find_library("cuda"))


def default_gl_backend() -> str | None:
    """Best offscreen GL backend for this machine, or None to let MuJoCo decide.

    On Linux with a GPU, EGL renders on the hardware and is roughly fifty times
    faster than OSMesa's software rasteriser - the difference between generating
    a 10k-episode image dataset in hours versus days. Without a GPU, OSMesa is
    both the only thing that works cleanly and what CI uses.

    macOS and Windows select CGL and WGL natively; setting MUJOCO_GL there breaks
    rendering rather than helping it.
    """
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return None
    if _has_gpu() and ctypes.util.find_library("EGL"):
        return "egl"
    if ctypes.util.find_library("OSMesa"):
        return "osmesa"
    if ctypes.util.find_library("EGL"):
        return "egl"
    return None


def configure_gl(force: bool = False) -> str | None:
    """Set ``MUJOCO_GL`` if it is not already set. Returns the effective value."""
    if not force and os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    backend = default_gl_backend()
    if backend:
        os.environ["MUJOCO_GL"] = backend
    return os.environ.get("MUJOCO_GL")


configure_gl()
