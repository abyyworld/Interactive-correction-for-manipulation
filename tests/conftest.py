import os

import pytest

# Software rendering: these tests must run on a CI box with no GPU.
os.environ.setdefault("MUJOCO_GL", "osmesa")


def _assets_available() -> bool:
    try:
        from icm.envs.assets import fetch
    except ImportError:
        return False
    return fetch.is_available()


requires_assets = pytest.mark.skipif(
    not _assets_available(),
    reason="Franka Panda assets not installed; run `make assets`",
)


@pytest.fixture(scope="session")
def env():
    """One environment shared across the session.

    Building the model compiles MJCF and 67 meshes, which takes about a second;
    reset() fully restores state, so sharing one instance is safe and keeps the
    suite fast.
    """
    from icm.envs.pick_place import EnvConfig, PickPlaceEnv

    e = PickPlaceEnv(EnvConfig(render_images=False), seed=0)
    e.reset(seed=0)
    yield e
    e.close()
