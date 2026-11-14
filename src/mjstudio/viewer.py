"""PanelViewer — wraps MuJoCo's own passive viewer and adds *draggable* panels.

This does NOT reimplement the 3D view. It launches the real `mujoco.viewer`
(`launch_passive`) — all its camera controls, perturbations, keybinds and the native
left/right UI panels come for free — and overlays a camera-thumbnail strip + an info/log
panel via the viewer's `set_images`/`set_texts` hooks.

The panels are mouse-draggable: grab a panel edge and pull to resize. That needs the
cursor position, which the stock passive viewer never exposed to Python. mjstudio reads
it from a tiny patch to MuJoCo's `_simulate` that adds `Simulate.mouse`
(see mjstudio/third_party + mjstudio/patches). On unpatched MuJoCo the panels still
render — they just can't be dragged (a one-time log says so).

    viewer = PanelViewer(model, data, cameras=["cam_top", "cam_back"]).open()
    while viewer.is_running():
        step()                         # you own the model/data and the clock
        viewer.info(step=i, fps=f)
        viewer.log("event")
        viewer.update()                # render thumbnails + overlays + viewer.sync()

Coordinates: `Simulate.mouse` and the overlay rects are both framebuffer pixels,
bottom-left origin — the same frame as `handle.viewport` — so no conversion is needed.

macOS: the passive viewer needs `mjpython` (that is MuJoCo's requirement, not ours).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

_GRAB = 10              # px hit-zone around a draggable edge
_MIN_FRAC = 0.05        # smallest panel size as a fraction of the viewport


def _nn_resize(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Nearest-neighbour resize of an HWC image. No cv2/PIL — thumbnails don't need it."""
    h, w = img.shape[:2]
    if tw <= 0 or th <= 0 or h == 0 or w == 0:
        return img
    yi = (np.arange(th) * h // th).clip(0, h - 1)
    xi = (np.arange(tw) * w // tw).clip(0, w - 1)
    return img[yi][:, xi]


class PanelViewer:
    def __init__(self, model, data, cameras=None, title: str = "mjstudio",
                 log_maxlen: int = 400, flip_images: bool = False,
                 strip_frac: float = 0.26, info_w_frac: float = 0.30,
                 info_h_frac: float = 0.55):
        self._m = model
        self._d = data
        self._cameras = list(cameras or [])
        self._flip = flip_images
        self._handle = None
        self._renderer = None  # lazy offscreen renderer (only if update() isn't given frames)
        self._info: dict = {}
        self._logs: deque[str] = deque(maxlen=log_maxlen)
        self._t0 = time.time()
        self.title = title
        # panel sizes, as fractions of the 3D viewport (responsive to window/panel resize)
        self._strip_frac = float(strip_frac)
        self._info_w_frac = float(info_w_frac)
        self._info_h_frac = float(info_h_frac)
        # drag state
        self._prev_left = 0
        self._drag = None          # None | 'strip' | 'info_w' | 'info_h'
        self._mouse_ok = None      # None=unknown, True/False after first probe
        # optional PIL text rendering for a draggable info panel
        try:
            from PIL import ImageFont
            self._pil = True
            try:
                self._font = ImageFont.truetype("Menlo.ttc", 13)
            except Exception:  # noqa: BLE001
                self._font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            self._pil = False
            self._font = None

    # ---- lifecycle ----
    def open(self, show_left_ui: bool = True, show_right_ui: bool = True) -> "PanelViewer":
        # Keep MuJoCo's native left/right panels (the whole point of embedding the
        # original viewer). Defaults match launch_passive's own defaults.
        import mujoco.viewer
        self._handle = mujoco.viewer.launch_passive(
            self._m, self._d, show_left_ui=show_left_ui, show_right_ui=show_right_ui)
        return self

    def is_running(self) -> bool:
        return self._handle is not None and self._handle.is_running()

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._handle = None

    # ---- panel content ----
    def info(self, **kw) -> None:
        self._info.update(kw)

    def log(self, msg: str) -> None:
        self._logs.append(f"{time.time() - self._t0:7.2f}s  {msg}")

    # ---- the one call per frame ----
    def update(self, frames: dict | None = None) -> None:
        """Handle drags, overlay the panels, then sync the real viewer."""
        if not self.is_running():
            return
        import mujoco

        vp = self._handle.viewport  # mjrRect of the 3D view (excludes native panels)
        self._handle_drag(vp)

        images = []
        # --- camera strip along the bottom of the 3D viewport ---
        frames = self._render_thumbs() if frames is None else frames
        strip_h = max(1, int(self._strip_frac * vp.height))
        if frames and self._cameras:
            x = vp.left
            for name in self._cameras:
                fr = frames.get(name)
                if fr is None:
                    continue
                h, w = fr.shape[:2]
                tw = max(1, int(strip_h * w / h))
                thumb = _nn_resize(fr, tw, strip_h)
                if self._flip:
                    thumb = thumb[::-1]
                if x + tw > vp.left + vp.width:   # clip the strip to the viewport width
                    break
                images.append((mujoco.MjrRect(x, vp.bottom, tw, strip_h), np.ascontiguousarray(thumb)))
                x += tw

        # --- info/log panel, top-right of the viewport ---
        lines = [self.title, "-" * 30]
        lines += [f"{k}: {v}" for k, v in self._info.items()]
        lines += ["", "logs:"] + list(self._logs)[-40:]
        iw = max(1, int(self._info_w_frac * vp.width))
        ih = max(1, int(self._info_h_frac * vp.height))
        info_img = self._render_info_image(lines, iw, ih)
        if info_img is not None:
            ix = vp.left + vp.width - iw
            iy = vp.bottom + vp.height - ih
            img = info_img[::-1] if self._flip else info_img
            images.append((mujoco.MjrRect(ix, iy, iw, ih), np.ascontiguousarray(img)))
        else:
            # no PIL: fall back to anchored (non-draggable) overlay text
            self._handle.set_texts([(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "\n".join(lines[:60]), "",
            )])

        if images:
            self._handle.set_images(images)
        self._handle.sync()

    # ---- drag handling ----
    def _mouse(self):
        """(x, y, left) in framebuffer px, bottom-left origin — or None if unavailable."""
        sim = self._handle._get_sim()
        if sim is None:
            return None
        try:
            mx, my, dx, dy, sx, sy, left, right, button = sim.mouse
        except AttributeError:
            if self._mouse_ok is None:
                self.log("mujoco not patched: Simulate.mouse missing — panels fixed (no drag)")
            self._mouse_ok = False
            return None
        self._mouse_ok = True
        return mx, my, left

    def _handle_drag(self, vp) -> None:
        m = self._mouse()
        if m is None:
            return
        mx, my, left = m
        strip_top = vp.bottom + self._strip_frac * vp.height
        info_left = vp.left + vp.width - self._info_w_frac * vp.width
        info_bot = vp.bottom + vp.height - self._info_h_frac * vp.height

        if left and not self._prev_left:            # button just pressed → maybe grab an edge
            if abs(my - strip_top) <= _GRAB and vp.left <= mx <= vp.left + vp.width * 0.6:
                self._drag = "strip"
            elif abs(mx - info_left) <= _GRAB and my >= info_bot - _GRAB:
                self._drag = "info_w"
            elif abs(my - info_bot) <= _GRAB and mx >= info_left - _GRAB:
                self._drag = "info_h"
        elif not left:
            self._drag = None

        if self._drag == "strip" and vp.height:
            self._strip_frac = min(0.8, max(_MIN_FRAC, (my - vp.bottom) / vp.height))
        elif self._drag == "info_w" and vp.width:
            self._info_w_frac = min(0.8, max(_MIN_FRAC, (vp.left + vp.width - mx) / vp.width))
        elif self._drag == "info_h" and vp.height:
            self._info_h_frac = min(0.95, max(_MIN_FRAC, (vp.bottom + vp.height - my) / vp.height))

        self._prev_left = left

    # ---- rendering helpers ----
    def _render_thumbs(self) -> dict:
        if not self._cameras:
            return {}
        import mujoco
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._m, 240, 320)
        out = {}
        for name in self._cameras:
            self._renderer.update_scene(self._d, camera=name)
            out[name] = self._renderer.render()
        return out

    def _render_info_image(self, lines, w, h):
        if not self._pil:
            return None
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (w, h), (18, 20, 26))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w - 1, h - 1], outline=(70, 80, 100))
        y, lh = 6, 16
        for i, line in enumerate(lines):
            if y > h - lh:
                break
            color = (120, 200, 255) if i == 0 else (210, 215, 225)
            draw.text((8, y), str(line)[:200], fill=color, font=self._font)
            y += lh
        return np.asarray(img, dtype=np.uint8)
