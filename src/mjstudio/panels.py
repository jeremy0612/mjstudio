"""Draw the info + log side panel (background rectangle + text overlay)."""
from __future__ import annotations

import mujoco

_FONT = mujoco.mjtFont.mjFONT_NORMAL
_TOPLEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT


def draw_panel(con, rect, info: dict, logs: list, title: str = "mjstudio",
               max_log_lines: int = 26) -> None:
    """Dark panel with a title, key/value info lines, then the tail of the log."""
    mujoco.mjr_rectangle(rect, 0.11, 0.11, 0.13, 1.0)

    lines = [title, "-" * 28]
    for k, v in info.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("logs:")
    lines.extend(logs[-max_log_lines:])

    mujoco.mjr_overlay(_FONT, _TOPLEFT, rect, "\n".join(lines), "", con)
