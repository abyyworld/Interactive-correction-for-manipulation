"""Fetch the Franka Panda MJCF + meshes from MuJoCo Menagerie at a pinned commit.

Why fetch instead of vendor
---------------------------
The Panda visual/collision meshes are ~33 MB. Committing them would dominate the
repository and make every clone slow, for assets that are already published and
Apache-2.0 licensed upstream. Instead we pin an exact commit SHA so every machine
resolves byte-identical geometry, and fall back to a primitive-geom robot when the
meshes are absent (see ``icm.envs.panda.SceneSpec``) so that CI, offline clones and
``pytest`` never require the network.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Pinned 2026-08. Update deliberately: robot kinematics changing under you is a
# silent source of "my policy stopped working" bugs.
MENAGERIE_SHA = "da76818e269b82289eba39808e2fb91d679d6994"
ROBOT_SUBDIR = "franka_emika_panda"

ASSETS_DIR = Path(__file__).resolve().parent
TARGET_DIR = ASSETS_DIR / "menagerie" / ROBOT_SUBDIR

# Files we actually need. Menagerie ships MJX variants and PNGs we never load.
KEEP_FILES = ("panda.xml", "hand.xml", "LICENSE")


def is_available() -> bool:
    """True when the pinned Panda meshes are present on this machine."""
    return (TARGET_DIR / "panda.xml").is_file() and (TARGET_DIR / "assets").is_dir()


def fetch(force: bool = False, quiet: bool = False) -> Path:
    """Clone the pinned Menagerie commit and copy the Panda folder into the package."""
    if is_available() and not force:
        if not quiet:
            print(f"[assets] already present at {TARGET_DIR}")
        return TARGET_DIR

    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch robot assets but was not found on PATH")

    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "menagerie"
        if not quiet:
            print(f"[assets] cloning {MENAGERIE_URL} @ {MENAGERIE_SHA[:12]} ...")
        # Shallow-fetch just the pinned commit: ~1 GB of history is not worth downloading.
        subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "remote", "add", "origin", MENAGERIE_URL], check=True
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "fetch", "--quiet", "--depth", "1", "origin", MENAGERIE_SHA],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "checkout", "--quiet", "FETCH_HEAD"], check=True
        )

        src = tmp_path / ROBOT_SUBDIR
        if not src.is_dir():
            raise RuntimeError(f"pinned commit does not contain {ROBOT_SUBDIR}/")

        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)
        TARGET_DIR.mkdir(parents=True)
        shutil.copytree(src / "assets", TARGET_DIR / "assets")
        for name in KEEP_FILES:
            if (src / name).is_file():
                shutil.copy2(src / name, TARGET_DIR / name)

    (TARGET_DIR / "PINNED_SHA").write_text(MENAGERIE_SHA + "\n")
    if not quiet:
        n = len(list((TARGET_DIR / "assets").iterdir()))
        print(f"[assets] installed {n} mesh files -> {TARGET_DIR}")
    return TARGET_DIR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch pinned Franka Panda assets.")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        fetch(force=args.force, quiet=args.quiet)
    except Exception as exc:  # pragma: no cover - network/CLI failure path
        print(f"[assets] FAILED: {exc}", file=sys.stderr)
        print("[assets] the simulator will fall back to primitive-geom robot visuals.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
