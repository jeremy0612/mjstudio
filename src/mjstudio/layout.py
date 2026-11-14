"""Viewport math: split one framebuffer into main view + camera strip + side panel.

MuJoCo/OpenGL rects are (left, bottom, width, height) with origin at the bottom-left.
Layout: panel on the right (full height), camera strip along the bottom of the remaining
column, main interactive view filling the rest above the strip.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass
class Rects:
    main: "mujoco.MjrRect"
    thumbs: list  # list[mujoco.MjrRect], one per camera
    panel: "mujoco.MjrRect"


def compute(fb_w: int, fb_h: int, n_cams: int,
            panel_w: int = 300, strip_h: int = 190, pad: int = 4) -> Rects:
    panel_w = min(panel_w, max(0, fb_w // 3))
    content_w = max(1, fb_w - panel_w)
    strip_h = min(strip_h, fb_h // 3) if n_cams > 0 else 0

    panel = mujoco.MjrRect(fb_w - panel_w, 0, panel_w, fb_h)
    main = mujoco.MjrRect(0, strip_h, content_w, max(1, fb_h - strip_h))

    thumbs: list = []
    if n_cams > 0:
        tw = content_w // n_cams
        for i in range(n_cams):
            thumbs.append(mujoco.MjrRect(i * tw, 0, max(1, tw - pad), max(1, strip_h - pad)))
    return Rects(main=main, thumbs=thumbs, panel=panel)
