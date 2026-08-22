"""Keyboard teleoperation, for developing and demoing without a headset.

Not a serious input device for manipulation - discrete keys map badly to a
continuous 6-DoF task - but it makes the intervention pipeline usable by anyone
who clones the repository, including reviewers with no VR hardware. It also
makes the human-in-the-loop path debuggable without putting a headset on.

Requires ``pygame`` (``pip install 'icm[teleop]'``) because it needs a focused
window to read key state; a terminal cannot report key *release*, which
continuous control needs.

Controls::

    W/S   +x / -x        SPACE  hold to take control (clutch)
    A/D   +y / -y        G      toggle gripper
    Q/E   +z / -z        1-4    attribute the error to approach/grasp/lift/place
    Z/C   yaw            ESC    abort the episode
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import TeleopCommand


@dataclass
class KeyboardConfig:
    speed: float = 1.0
    yaw_speed: float = 0.6
    window_size: tuple[int, int] = (480, 200)
    caption: str = "icm teleop - hold SPACE to take control"


class KeyboardTeleop:
    def __init__(self, config: KeyboardConfig | None = None):
        self.config = config or KeyboardConfig()
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "keyboard teleop needs pygame. Install with: pip install 'icm[teleop]'"
            ) from exc
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.config.window_size)
        pygame.display.set_caption(self.config.caption)
        self.font = pygame.font.SysFont("monospace", 14)
        self._gripper_open = True
        self._attribution: int | None = None
        self._reported = False
        self._abort = False

    def reset(self, env, seed: int | None = None) -> None:
        self._gripper_open = True
        self._attribution = None
        self._reported = False
        self._abort = False

    def poll(self, env, obs: dict, info: dict, step: int) -> TeleopCommand:
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._abort = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_g:
                    self._gripper_open = not self._gripper_open
                elif event.key == pg.K_ESCAPE:
                    self._abort = True
                elif pg.K_1 <= event.key <= pg.K_4:
                    self._attribution = event.key - pg.K_1

        keys = pg.key.get_pressed()
        engaged = bool(keys[pg.K_SPACE])
        s = self.config.speed
        action = np.zeros(7)
        action[0] = s * (keys[pg.K_w] - keys[pg.K_s])
        action[1] = s * (keys[pg.K_a] - keys[pg.K_d])
        action[2] = s * (keys[pg.K_q] - keys[pg.K_e])
        action[5] = self.config.yaw_speed * (keys[pg.K_z] - keys[pg.K_c])
        action[6] = 1.0 if self._gripper_open else -1.0

        self._draw(info, engaged)

        attribution = None
        if engaged and self._attribution is not None and not self._reported:
            attribution = self._attribution
            self._reported = True

        return TeleopCommand(action=action, engaged=engaged, attribution=attribution,
                             abort=self._abort)

    def _draw(self, info: dict, engaged: bool) -> None:
        pg = self.pygame
        self.screen.fill((18, 18, 22))
        lines = [
            f"phase   : {info.get('phase_name', '?')}",
            f"grasped : {info.get('grasped', False)}",
            f"control : {'HUMAN (space held)' if engaged else 'policy'}",
            f"gripper : {'open' if self._gripper_open else 'closed'}",
            f"blame   : {self._attribution if self._attribution is not None else '- (press 1-4)'}",
        ]
        for i, line in enumerate(lines):
            colour = (250, 180, 90) if (i == 2 and engaged) else (210, 210, 215)
            self.screen.blit(self.font.render(line, True, colour), (14, 14 + 26 * i))
        pg.display.flip()

    def close(self) -> None:
        self.pygame.quit()
