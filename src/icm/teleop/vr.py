"""VR hand-tracking teleoperation over UDP.

Mapping hands to a robot
------------------------
Absolute pose mapping (hand position *is* TCP position) fails immediately in
practice: the human workspace and the robot workspace are different sizes and in
different places, and the operator's arm gets tired holding an absolute frame.

So this uses **clutched relative control**, the standard solution. While the
operator holds the trigger, the *change* in hand pose since they engaged is
applied to the TCP pose it had at that moment. Releasing and re-gripping
re-centres, exactly like lifting and repositioning a mouse. Scaling then lets a
20 cm hand movement span the robot's 60 cm workspace.

Safety and honesty
------------------
* Packets are timestamped and stale input disengages control rather than
  replaying the last command; a frozen hand pose commanding a moving robot is
  the worst possible failure.
* Out-of-order datagrams are dropped by sequence number.
* The receiver never blocks the control loop: it drains the socket
  non-blockingly and uses the newest packet available.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

import numpy as np

from .base import TeleopCommand
from .protocol import DEFAULT_PORT, MAX_PACKET_BYTES, VRPacket


@dataclass
class VRConfig:
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    #: Robot metres per metre of hand movement. Above 1 amplifies, which reduces
    #: fatigue at the cost of precision.
    position_scale: float = 2.5
    #: Seconds after which input is considered stale and control is dropped.
    timeout: float = 0.25
    #: Axis remap from the client frame to the robot frame. Defaults assume a
    #: y-up client (Unity/WebXR convention) and a z-up robot.
    axis_map: tuple[int, int, int] = (0, 2, 1)
    axis_sign: tuple[float, float, float] = (1.0, -1.0, 1.0)
    lock_roll_pitch: bool = True
    max_delta: float = 1.0


class VRTeleop:
    """Receives hand poses and converts them to environment actions."""

    def __init__(self, config: VRConfig | None = None, sock: socket.socket | None = None):
        self.config = config or VRConfig()
        self.sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if sock is None:
            self.sock.bind((self.config.host, self.config.port))
        self.sock.setblocking(False)

        self._last: VRPacket | None = None
        self._last_seq = -1
        self._anchor_hand: np.ndarray | None = None
        self._anchor_tcp: np.ndarray | None = None
        self._engaged = False
        self._reported = False

    # ------------------------------------------------------------------ io

    def drain(self) -> VRPacket | None:
        """Read every pending datagram and keep the newest valid one."""
        newest = None
        while True:
            try:
                raw, _ = self.sock.recvfrom(MAX_PACKET_BYTES)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                packet = VRPacket.from_bytes(raw)
            except (ValueError, UnicodeDecodeError):
                continue  # a malformed packet must not kill the control loop
            if packet.seq < self._last_seq:
                continue  # out of order; UDP makes no ordering promise
            self._last_seq = packet.seq
            newest = packet
        if newest is not None:
            self._last = newest
        return newest

    # ------------------------------------------------------------------ mapping

    def _to_robot_frame(self, v: np.ndarray) -> np.ndarray:
        m, s = self.config.axis_map, self.config.axis_sign
        return np.array([v[m[0]] * s[0], v[m[1]] * s[1], v[m[2]] * s[2]])

    def reset(self, env, seed: int | None = None) -> None:
        self._anchor_hand = None
        self._anchor_tcp = None
        self._engaged = False
        self._reported = False
        self.drain()  # discard input queued while the previous episode ran

    def poll(self, env, obs: dict, info: dict, step: int) -> TeleopCommand:
        import time

        self.drain()
        packet = self._last
        if packet is None:
            return TeleopCommand(engaged=False)

        # A stale packet means the client stopped sending. Disengage rather than
        # repeating the last pose: a frozen hand driving a live robot is worse
        # than no input at all.
        if self.config.timeout > 0 and packet.t > 0:
            if abs(time.time() - packet.t) > self.config.timeout:
                self._engaged = False
                self._anchor_hand = None
                return TeleopCommand(engaged=False, extra={"stale": True})

        if not packet.engaged:
            self._engaged = False
            self._anchor_hand = None
            return TeleopCommand(engaged=False, abort=packet.abort)

        hand = self._to_robot_frame(packet.position)
        if self._anchor_hand is None:
            # Clutch engaged: anchor here so the robot does not jump.
            self._anchor_hand = hand
            self._anchor_tcp = env.robot.tcp_pos.copy()
            self._engaged = True

        target = self._anchor_tcp + (hand - self._anchor_hand) * self.config.position_scale
        delta = (target - env.robot.tcp_pos) / env.config.max_delta_pos
        action = np.zeros(7)
        action[:3] = np.clip(delta, -self.config.max_delta, self.config.max_delta)
        action[6] = 1.0 - 2.0 * float(np.clip(packet.trigger, 0.0, 1.0))

        attribution = None
        confidence = None
        if packet.attribution is not None and not self._reported:
            attribution = int(packet.attribution)
            confidence = packet.confidence
            self._reported = True

        return TeleopCommand(
            action=action,
            engaged=True,
            attribution=attribution,
            confidence=confidence,
            abort=packet.abort,
            extra={"seq": packet.seq, "trigger": packet.trigger},
        )

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class MockVRClient:
    """Sends synthetic packets. Lets the whole VR path be tested with no headset."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, pos, trigger: float = 0.0, engaged: bool = True, **kwargs) -> VRPacket:
        import time

        self.seq += 1
        packet = VRPacket(
            seq=self.seq,
            t=time.time(),
            pos=tuple(float(x) for x in pos),
            trigger=trigger,
            engaged=engaged,
            **kwargs,
        )
        self.sock.sendto(packet.to_bytes(), self.addr)
        return packet

    def close(self) -> None:
        self.sock.close()
