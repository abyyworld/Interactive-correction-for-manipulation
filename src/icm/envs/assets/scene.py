"""Programmatic MJCF construction for the pick-and-place task.

The scene is generated in Python rather than kept as a static ``.xml`` because
almost everything about it is an experimental variable: how many distractor
objects there are, their colours and shapes (which the language-conditioned
policy keys off), camera resolution, and table extents. Hand-editing XML for
every ablation is how scenes silently drift out of sync with the code.

Geometry conventions
--------------------
* The robot base sits at the world origin, bolted to the table surface, which
  is the plane ``z = 0``. This mirrors a real Franka mounted on a workbench and
  keeps every object z-coordinate a readable "height above the table".
* The floor is at ``z = TABLE_HEIGHT`` below the table top.
* Objects rest on the table at ``z = half_size``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fetch

# --- Table / workspace geometry (metres) ------------------------------------
TABLE_HEIGHT = 0.75  # table top above the floor
TABLE_HALF = (0.55, 0.50, 0.025)  # half-extents of the table slab
TABLE_CENTER_X = 0.35  # slab centre; robot base at x=0 sits near the back edge

# Region in which objects may be spawned. Chosen to sit comfortably inside the
# Panda's 0.855 m reach while leaving clearance from the base and table edges.
WORKSPACE_X = (0.38, 0.66)
WORKSPACE_Y = (-0.22, 0.22)

GOAL_POS = (0.52, 0.30)
GOAL_RADIUS = 0.075

CUBE_HALF = 0.021  # 4.2 cm cube: fits the 8 cm Panda gripper stroke with margin

# Distance from the Panda hand flange (the ``hand`` body origin) to the point
# midway between the closed fingertips. This is the frame every controller,
# teleop device and policy action is expressed in; putting it between the
# fingertips rather than at the flange is what makes "move 1 cm toward the cube"
# mean what it says.
TCP_OFFSET_Z = 0.1034


@dataclass(frozen=True)
class ObjectSpec:
    """One manipulable object on the table."""

    name: str
    rgba: tuple[float, float, float, float]
    shape: str = "box"  # "box" | "cylinder"
    half: float = CUBE_HALF
    mass: float = 0.05
    color_word: str = ""  # used by language conditioning, e.g. "red"
    shape_word: str = ""  # e.g. "block"

    def geom_size(self) -> str:
        if self.shape == "box":
            return f"{self.half} {self.half} {self.half}"
        # cylinder: radius, half-height
        return f"{self.half} {self.half}"

    @property
    def noun(self) -> str:
        return f"{self.color_word} {self.shape_word}".strip()


DEFAULT_OBJECTS: tuple[ObjectSpec, ...] = (
    ObjectSpec("target", (0.85, 0.15, 0.15, 1.0), "box", color_word="red", shape_word="block"),
    ObjectSpec("distractor_a", (0.15, 0.35, 0.85, 1.0), "box", color_word="blue", shape_word="block"),
    ObjectSpec("distractor_b", (0.15, 0.65, 0.25, 1.0), "cylinder", color_word="green", shape_word="cylinder"),
)


@dataclass
class SceneSpec:
    """Everything needed to emit a scene MJCF."""

    objects: tuple[ObjectSpec, ...] = DEFAULT_OBJECTS
    timestep: float = 0.002
    camera_fovy: float = 58.0
    # Third-person camera framing the whole workspace, used for GIFs and as the
    # optional second policy input.
    scene_cam_pos: tuple[float, float, float] = (1.05, -0.75, 0.65)
    scene_cam_target: tuple[float, float, float] = (0.45, 0.0, 0.05)
    front_cam_pos: tuple[float, float, float] = (1.25, 0.0, 0.50)
    front_cam_target: tuple[float, float, float] = (0.40, 0.0, 0.05)
    # Eye-in-hand camera, injected into the Panda ``hand`` body post-parse.
    # Expressed in the hand frame, whose +z points out between the fingers.
    wrist_cam_pos: tuple[float, float, float] = (0.0, -0.050, 0.012)
    wrist_cam_target: tuple[float, float, float] = (0.0, 0.0, 0.13)
    wrist_cam_up: tuple[float, float, float] = (0.0, -1.0, 0.0)
    extra_worldbody: str = ""
    include_goal_pad: bool = True
    #: Compensate gravity on the robot links (not the objects). A real Franka
    #: does this inside its own controller, so enabling it matches the hardware
    #: rather than making the task easier.
    gravity_compensation: bool = True
    _panda_xml: Path | None = field(default=None, repr=False)


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """Return a MuJoCo ``xyaxes`` string orienting a camera at ``pos`` toward ``target``.

    MuJoCo cameras look down their local -z axis with +x right and +y up. Solving
    for that basis numerically is far less error-prone than hand-writing
    quaternions, and it means moving a camera is a one-line change.
    """
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("camera position and target coincide")
    forward /= n
    cam_z = -forward
    up = np.asarray(up, dtype=float)
    x_axis = np.cross(up, cam_z)
    if np.linalg.norm(x_axis) < 1e-6:  # looking straight up/down: pick another up
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), cam_z)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(cam_z, x_axis)
    return " ".join(f"{v:.6f}" for v in (*x_axis, *y_axis))


def lookat_basis(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Return the 3x3 camera rotation (columns = x, y, z axes) looking at ``target``."""
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("camera position and target coincide")
    forward /= n
    cam_z = -forward
    up = np.asarray(up, dtype=float)
    x_axis = np.cross(up, cam_z)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), cam_z)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(cam_z, x_axis)
    return np.column_stack([x_axis, y_axis, cam_z])


def lookat_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (w, x, y, z) orienting a camera at ``pos`` toward ``target``."""
    import mujoco

    basis = lookat_basis(pos, target, up)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, basis.flatten())
    return quat


def _panda_include_path() -> Path:
    """Path to a keyframe-stripped copy of the Menagerie ``panda.xml``.

    The upstream file ends with a ``<keyframe>`` whose ``qpos`` has exactly 9
    entries (7 arm joints + 2 fingers). Adding free-jointed objects raises ``nq``
    to 9 + 7*n_objects, and MuJoCo rejects a keyframe whose size no longer
    matches. We cache a stripped copy rather than mutating the fetched assets so
    that re-running the fetcher stays idempotent, and we set the home pose from
    Python instead (see ``icm.envs.panda.HOME_QPOS``).
    """
    if not fetch.is_available():
        raise FileNotFoundError(
            "Franka Panda assets are not installed.\n"
            "Run:  make assets       (or)  python -m icm.envs.assets.fetch\n"
            f"Expected them at: {fetch.TARGET_DIR}"
        )
    src = fetch.TARGET_DIR / "panda.xml"
    dst = fetch.TARGET_DIR / "_panda_nokeyframe.xml"
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        text = src.read_text()
        text = re.sub(r"<keyframe>.*?</keyframe>", "", text, flags=re.DOTALL)
        dst.write_text(text)
    return dst


def _table_xml() -> str:
    hx, hy, hz = TABLE_HALF
    top_z = -hz  # slab centre so that its top surface is exactly z = 0
    leg_h = (TABLE_HEIGHT - 2 * hz) / 2.0
    leg_z = -2 * hz - leg_h
    legs = []
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lx = TABLE_CENTER_X + sx * (hx - 0.06)
        ly = sy * (hy - 0.06)
        legs.append(
            f'      <geom name="table_leg{i}" type="box" size="0.03 0.03 {leg_h:.4f}" '
            f'pos="{lx:.4f} {ly:.4f} {leg_z:.4f}" material="table_leg" contype="0" conaffinity="0"/>'
        )
    legs_xml = "\n".join(legs)
    return f"""    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="{hx} {hy} {hz}" pos="{TABLE_CENTER_X} 0 {top_z:.4f}"
            material="table" friction="1.0 0.02 0.001" solimp="0.98 0.99 0.001" solref="0.005 1"/>
{legs_xml}
    </body>"""


def _goal_xml() -> str:
    gx, gy = GOAL_POS
    return f"""    <body name="goal" pos="{gx} {gy} 0">
      <geom name="goal_pad" type="cylinder" size="{GOAL_RADIUS} 0.001" pos="0 0 0.001"
            material="goal" contype="0" conaffinity="0" group="1"/>
      <site name="goal_site" type="sphere" size="0.008" pos="0 0 0.001" rgba="0.1 0.9 0.4 0.0"/>
    </body>"""


def _object_xml(spec: ObjectSpec, index: int) -> str:
    # Objects spawn stacked out of the way; ``PickPlaceEnv.reset`` writes their
    # real pose into qpos. Free joints mean we can teleport them without any
    # solver fight, which is what randomised placement needs.
    z = spec.half
    x = 0.5
    y = -0.4 - 0.1 * index
    inertia = f'<inertial pos="0 0 0" mass="{spec.mass}" diaginertia="1e-4 1e-4 1e-4"/>'
    return f"""    <body name="{spec.name}" pos="{x} {y} {z}">
      <freejoint name="{spec.name}_free"/>
      {inertia}
      <geom name="{spec.name}_geom" type="{spec.shape}" size="{spec.geom_size()}"
            rgba="{' '.join(str(c) for c in spec.rgba)}" friction="1.0 0.02 0.001"
            solimp="0.98 0.99 0.001" solref="0.005 1" condim="4" priority="1"/>
    </body>"""


def build_scene_xml(spec: SceneSpec | None = None) -> str:
    """Emit the complete task MJCF as a string."""
    spec = spec or SceneSpec()
    panda_xml = spec._panda_xml or _panda_include_path()
    meshdir = (panda_xml.parent / "assets").as_posix()

    objects_xml = "\n".join(_object_xml(o, i) for i, o in enumerate(spec.objects))
    goal_xml = _goal_xml() if spec.include_goal_pad else ""

    scene_axes = lookat_xyaxes(spec.scene_cam_pos, spec.scene_cam_target)
    front_axes = lookat_xyaxes(spec.front_cam_pos, spec.front_cam_target)

    return f"""<mujoco model="icm_pick_place">
  <option timestep="{spec.timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.01" zfar="6"/>
    <global azimuth="140" elevation="-25"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.58" rgb2="0.08 0.10 0.14"
             width="512" height="1024"/>
    <texture type="2d" name="floor_tex" builtin="checker" mark="edge"
             rgb1="0.22 0.24 0.27" rgb2="0.16 0.18 0.21" markrgb="0.4 0.4 0.4"
             width="300" height="300"/>
    <material name="floor" texture="floor_tex" texuniform="true" texrepeat="4 4" reflectance="0.05"/>
    <texture type="2d" name="table_tex" builtin="flat" rgb1="0.82 0.76 0.66" rgb2="0.82 0.76 0.66"
             width="64" height="64"/>
    <material name="table" texture="table_tex" texuniform="true" reflectance="0.02" shininess="0.1"/>
    <material name="table_leg" rgba="0.25 0.26 0.28 1"/>
    <material name="goal" rgba="0.15 0.75 0.35 0.35"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.6 -0.4 1.6" dir="-0.3 0.25 -1" directional="true"
           diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <light name="fill" pos="0.2 0.6 1.4" dir="0.15 -0.4 -1" directional="true" diffuse="0.28 0.28 0.3"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 {-TABLE_HEIGHT}" material="floor"/>

    <camera name="scene" pos="{' '.join(str(v) for v in spec.scene_cam_pos)}"
            xyaxes="{scene_axes}" fovy="{spec.camera_fovy}" mode="fixed"/>
    <camera name="front" pos="{' '.join(str(v) for v in spec.front_cam_pos)}"
            xyaxes="{front_axes}" fovy="{spec.camera_fovy}" mode="fixed"/>

{_table_xml()}
{goal_xml}
{objects_xml}
{spec.extra_worldbody}
  </worldbody>

  <include file="{panda_xml.as_posix()}"/>

  <!-- MuJoCo applies the LAST <compiler> it parses, so this must follow the
       include: it redirects mesh lookup to the fetched Menagerie assets, whose
       own relative meshdir does not survive textual inclusion. -->
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
</mujoco>"""


def build_model(spec: SceneSpec | None = None):
    """Compile the scene and return ``(mujoco.MjModel, xml_string)``.

    The wrist camera cannot be written into the MJCF text: it must be a child of
    the Panda's ``hand`` body, which only exists after the Menagerie include is
    parsed. We therefore parse to an ``MjSpec``, attach the camera to the real
    body, and compile from there. That keeps camera placement a Python-level
    parameter instead of file surgery on fetched third-party assets.
    """
    import mujoco

    spec = spec or SceneSpec()
    xml = build_scene_xml(spec)
    try:
        mj_spec = mujoco.MjSpec.from_string(xml)
    except ValueError as exc:  # surface the offending XML, MuJoCo errors are terse
        raise ValueError(f"failed to parse scene MJCF: {exc}") from exc

    hand = mj_spec.find_body("hand")
    if hand is None:
        raise RuntimeError(
            "Panda 'hand' body not found - the pinned Menagerie assets may have changed shape."
        )
    cam = hand.add_camera()
    cam.name = "wrist"
    cam.pos = np.asarray(spec.wrist_cam_pos, dtype=float)
    cam.quat = lookat_quat(spec.wrist_cam_pos, spec.wrist_cam_target, spec.wrist_cam_up)
    cam.fovy = spec.camera_fovy

    # Grasp frame. Menagerie ships no site on the hand, so every downstream
    # component would otherwise re-derive the fingertip offset independently.
    tcp = hand.add_site()
    tcp.name = "tcp"
    tcp.pos = np.array([0.0, 0.0, TCP_OFFSET_Z])
    tcp.size = np.array([0.005, 0.005, 0.005])
    tcp.rgba = np.array([1.0, 0.2, 0.2, 0.0])  # invisible; used for kinematics only

    if spec.gravity_compensation:
        # Must happen before compile: MuJoCo counts gravity-compensated bodies at
        # compile time (``ngravcomp``) and skips the computation entirely when
        # that count is zero. Writing ``model.body_gravcomp`` on an already
        # compiled model silently does nothing - the field changes, the force
        # never appears.
        for body in _iter_bodies(mj_spec.worldbody):
            if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
                body.gravcomp = 1.0

    model = mj_spec.compile()
    return model, xml


def _iter_bodies(body):
    """Depth-first walk over an MjSpec body tree."""
    for child in body.bodies:
        yield child
        yield from _iter_bodies(child)
