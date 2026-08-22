"""Wire format between a VR client and the simulator.

The headset runs its own runtime (Unity, WebXR, OpenXR) in a separate process,
often on a separate machine. Rather than embed a VR SDK in the training code,
the boundary is a tiny UDP protocol: the client sends hand poses, the simulator
sends back nothing. That keeps this repository installable and testable with no
headset attached, and means swapping Quest for Index for a phone AR app is a
client-side change only.

UDP, not TCP, on purpose. Teleoperation wants the *latest* pose, not every pose.
A dropped packet should be forgotten, not retransmitted while newer poses queue
behind it — head-of-line blocking is exactly the failure mode that makes teleop
feel laggy and unusable.

Message format (one JSON object per datagram, UTF-8)::

    {
      "seq": 1042,                  # monotonic; receiver drops out-of-order packets
      "t": 1712345678.912,          # client clock, seconds
      "pos": [x, y, z],             # hand position, metres, client frame
      "quat": [w, x, y, z],         # hand orientation
      "trigger": 0.0,               # 0 open .. 1 closed
      "engaged": true,              # clutch: is the operator driving?
      "attribution": 1,             # optional: phase index they blame
      "confidence": 0.7,            # optional
      "abort": false
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

PROTOCOL_VERSION = 1
DEFAULT_PORT = 5555
MAX_PACKET_BYTES = 2048


@dataclass
class VRPacket:
    seq: int = 0
    t: float = 0.0
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    trigger: float = 0.0
    engaged: bool = False
    attribution: int | None = None
    confidence: float | None = None
    abort: bool = False
    version: int = PROTOCOL_VERSION
    extra: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        payload = asdict(self)
        payload["pos"] = list(self.pos)
        payload["quat"] = list(self.quat)
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> VRPacket:
        d = json.loads(raw.decode("utf-8"))
        version = int(d.get("version", 0))
        if version > PROTOCOL_VERSION:
            raise ValueError(
                f"VR client speaks protocol v{version}, this receiver understands v{PROTOCOL_VERSION}"
            )
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
        if "pos" in d:
            d["pos"] = tuple(float(x) for x in d["pos"])
        if "quat" in d:
            d["quat"] = tuple(float(x) for x in d["quat"])
        return cls(**d)

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.pos, dtype=float)
