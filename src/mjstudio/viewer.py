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

import threading
import time
from collections import deque

import numpy as np

_GRAB = 22              # px hit-zone around a draggable edge (framebuffer px; retina ~2x)
_BAR = 6                # thickness of a visible drag-handle bar
_MIN_FRAC = 0.05        # smallest panel size as a fraction of the viewport


def _nn_resize(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Nearest-neighbour resize of an HWC image. No cv2/PIL — thumbnails don't need it."""
    h, w = img.shape[:2]
    if tw <= 0 or th <= 0 or h == 0 or w == 0:
        return img
    yi = (np.arange(th) * h // th).clip(0, h - 1)
    xi = (np.arange(tw) * w // tw).clip(0, w - 1)
    return img[yi][:, xi]


def _fit(img: np.ndarray, tw: int, th: int, bg=(18, 20, 26)) -> np.ndarray:
    """Fit img inside a tw×th window keeping its aspect ratio (letterbox), centred on a
    bg canvas. Resizing the window changes the window size, not the frame's shape."""
    tw, th = max(1, int(tw)), max(1, int(th))
    h, w = img.shape[:2]
    canvas = np.empty((th, tw, 3), np.uint8)
    canvas[:] = bg
    if w == 0 or h == 0:
        return canvas
    s = min(tw / w, th / h)
    rw, rh = max(1, min(tw, int(w * s))), max(1, min(th, int(h * s)))
    small = _nn_resize(img, rw, rh)
    y0, x0 = (th - rh) // 2, (tw - rw) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = small[:rh, :rw]
    return canvas


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
        self._cam_snap = None      # saved free-camera pose while a panel is being dragged
        # background pump: decouples overlay render + mouse/drag from the app's call
        # rate, so panels stay live and draggable even while the app thread blocks
        # (e.g. a multi-GB policy load, or a slow inference step).
        self._lock = threading.Lock()
        self._frames = None        # latest frames handed in by the app (may be None)
        self._thread = None
        self._auto = True
        self._stop = False
        self._pump_err = None
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
    def open(self, show_left_ui: bool = True, show_right_ui: bool = True,
             auto: bool = True) -> "PanelViewer":
        # Keep MuJoCo's native left/right panels (the whole point of embedding the
        # original viewer). Defaults match launch_passive's own defaults.
        # auto=True (default): a background thread pumps the overlays + drag at ~60 Hz,
        # so the app only has to feed data via info()/log()/update(frames=...). Set
        # auto=False to drive everything yourself from update() (single-thread apps).
        import mujoco.viewer
        self._handle = mujoco.viewer.launch_passive(
            self._m, self._d, show_left_ui=show_left_ui, show_right_ui=show_right_ui)
        self._auto = auto
        if auto:
            self._stop = False
            self._thread = threading.Thread(target=self._pump, name="mjstudio-pump", daemon=True)
            self._thread.start()
        return self

    def is_running(self) -> bool:
        return self._handle is not None and self._handle.is_running()

    def lock(self):
        """Context manager for the viewer's sim mutex. Hold it around ANY mutation of the
        model/data you passed in (``mj_step``/``mj_forward``/``mjv_updateScene``) so your
        writes don't race the background pump's ``sync()`` — otherwise MuJoCo raises
        "attempting to copy mjData while stack is in use". No-op once the window closes."""
        import contextlib
        if self._handle is None:
            return contextlib.nullcontext()
        return self._handle.lock()

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            # The pump owns the offscreen renderer and frees it on exit (see _pump).
            self._thread.join(timeout=1.0)
            self._thread = None
        elif self._renderer is not None:  # auto=False: the caller's thread rendered
            self._close_renderer()
        # ponytail: if the pump didn't exit in time, leak its renderer — closing it from
        # here would block forever on the GL context lock; process exit frees it.
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._handle = None

    # ---- panel content (thread-safe; the app thread writes, the pump reads) ----
    def info(self, **kw) -> None:
        with self._lock:
            self._info.update(kw)

    def log(self, msg: str) -> None:
        with self._lock:
            self._logs.append(f"{time.time() - self._t0:7.2f}s  {msg}")

    # ---- feed / drive ----
    def update(self, frames: dict | None = None) -> None:
        """Hand the latest frames to the viewer.

        auto=True (default): just stores them; the background pump renders + handles
        drag at ~60 Hz. auto=False: renders synchronously on the caller's thread.
        """
        if self._auto:
            with self._lock:
                self._frames = frames
            return
        self._draw(frames)

    def _pump(self) -> None:
        """Background render/drag loop — runs until close() or the window is shut."""
        while not self._stop and self.is_running():
            with self._lock:
                frames = self._frames
            try:
                self._draw(frames)
            except Exception as e:  # noqa: BLE001 — a render loop must survive transients
                if self._pump_err is None:
                    self._pump_err = repr(e)
                    print(f"[mjstudio] pump error (will keep running): {e!r}")
            time.sleep(1 / 60)
        # Free the renderer HERE, on the thread that created/used it: mujoco's CGL
        # as_current() locks the context (CGLLockContext) and never unlocks, so closing it
        # from another thread (close() on the main thread) deadlocks in make_current().
        self._close_renderer()

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None

    def _draw(self, frames: dict | None = None) -> None:
        """Handle drags, overlay the panels, then sync the real viewer."""
        if not self.is_running():
            return
        import mujoco

        vp = self._handle.viewport  # mjrRect of the 3D view (excludes native panels)
        frames = self._render_thumbs() if frames is None else frames
        cams = [c for c in self._cameras if frames and frames.get(c) is not None]

        self._handle_drag(vp, self._geom(vp, len(cams)))
        g = self._geom(vp, len(cams))   # geometry after any drag

        images = []
        # --- camera strip: all N thumbs fit across the viewport width ---
        tw, th = int(g["tw"]), int(g["th"])
        for i, name in enumerate(cams):
            thumb = _fit(frames[name], tw, th)   # keep aspect, letterbox — don't distort
            if self._flip:
                thumb = thumb[::-1]
            x = int(vp.left + i * tw)
            images.append((mujoco.MjrRect(x, vp.bottom, tw, th), np.ascontiguousarray(thumb)))
        if cams:
            images.append(self._hbar(vp.left, g["strip_y"], len(cams) * tw, self._drag == "strip"))

        # --- info/log panel, top-right of the viewport ---
        with self._lock:   # snapshot shared state quickly, render without the lock
            info_items = list(self._info.items())
            logs = list(self._logs)[-40:]
        lines = [self.title, "-" * 30]
        lines += [f"{k}: {v}" for k, v in info_items]
        lines += ["", "logs:"] + logs
        iw, ih = int(g["iw"]), int(g["ih"])
        info_img = self._render_info_image(lines, iw, ih)
        if info_img is not None:
            img = info_img[::-1] if self._flip else info_img
            images.append((mujoco.MjrRect(int(g["ix"]), int(g["iy"]), iw, ih), np.ascontiguousarray(img)))
            images.append(self._vbar(g["ix"], g["iy"], ih, self._drag == "info_w"))   # left edge
            images.append(self._hbar(g["ix"], g["iy"], iw, self._drag == "info_h"))   # bottom edge
        else:
            # no PIL: fall back to anchored (non-draggable) overlay text
            self._handle.set_texts([(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "\n".join(lines[:60]), "",
            )])

        if images:
            self._handle.set_images(images)
        self._handle.sync()

    def _geom(self, vp, n):
        """Panel geometry in framebuffer px (same frame as the mouse).

        Camera strip: thumbs fill the full viewport width (each = vp.width/n); dragging
        the strip's top edge changes only their height (vertical). The info/log panel
        resizes on both axes. ponytail: thumbs are stretched to fill (aspect not kept) —
        that's the requested "fill the width" behaviour.
        """
        th = (self._strip_frac * vp.height) if n else 0.0
        tw = (vp.width / n) if n else 0.0
        iw = self._info_w_frac * vp.width
        ih = self._info_h_frac * vp.height
        return {
            "th": th, "tw": tw, "strip_w": vp.width if n else 0.0, "strip_y": vp.bottom + th,
            "iw": iw, "ih": ih,
            "ix": vp.left + vp.width - iw, "iy": vp.bottom + vp.height - ih,
        }

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

    def _handle_drag(self, vp, g) -> None:
        m = self._mouse()
        if m is None:
            return
        mx, my, left = m
        if left and not self._prev_left:            # button just pressed → maybe grab an edge
            if abs(my - g["strip_y"]) <= _GRAB and vp.left - _GRAB <= mx <= vp.left + g["strip_w"] + _GRAB:
                self._drag = "strip"
            elif abs(mx - g["ix"]) <= _GRAB and my >= g["iy"] - _GRAB:
                self._drag = "info_w"
            elif abs(my - g["iy"]) <= _GRAB and mx >= g["ix"] - _GRAB:
                self._drag = "info_h"
        elif not left:
            self._drag = None

        if self._drag == "strip" and vp.height:
            self._strip_frac = min(0.8, max(_MIN_FRAC, (my - vp.bottom) / vp.height))
        elif self._drag == "info_w" and vp.width:
            self._info_w_frac = min(0.8, max(_MIN_FRAC, (vp.left + vp.width - mx) / vp.width))
        elif self._drag == "info_h" and vp.height:
            self._info_h_frac = min(0.95, max(_MIN_FRAC, (vp.bottom + vp.height - my) / vp.height))

        self._freeze_camera()
        self._prev_left = left

    def _freeze_camera(self) -> None:
        """Keep the 3D scene perfectly still while a panel is being dragged.

        MuJoCo orbits its free camera on any left-drag in the 3D view, and the passive
        viewer gives Python no way to veto that input. So on grab we switch the viewer
        camera to type USER — MuJoCo then renders the last pose and ignores the orbit, so
        there's no per-frame fight and no shake — and on release we restore the free camera
        and its saved pose, so the view returns exactly where it was. State change happens
        only on grab/release (not every frame), so nothing jitters."""
        cam = getattr(self._handle, "cam", None)
        if cam is None:
            return
        import mujoco
        if self._drag:
            if self._cam_snap is None:   # just grabbed → freeze the view
                self._cam_snap = (int(cam.type), cam.azimuth, cam.elevation,
                                  cam.distance, tuple(cam.lookat))
                cam.type = mujoco.mjtCamera.mjCAMERA_USER
        elif self._cam_snap is not None:  # just released → restore the free camera
            t, az, el, di, lk = self._cam_snap
            cam.azimuth, cam.elevation, cam.distance = az, el, di
            cam.lookat[:] = lk
            cam.type = t
            self._cam_snap = None

    # ---- rendering helpers ----
    def _render_thumbs(self) -> dict:
        if not self._cameras:
            return {}
        import mujoco
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._m, 240, 320)
        out = {}
        with self.lock():   # update_scene reads the shared data; serialize with sync()
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

    def _hbar(self, x, y, w, active):
        """Horizontal drag-handle bar centred on edge y -> (MjrRect, img)."""
        import mujoco
        w = max(1, int(w))
        img = np.empty((_BAR, w, 3), np.uint8)
        img[:] = (80, 220, 255) if active else (95, 105, 125)
        return (mujoco.MjrRect(int(x), int(y - _BAR // 2), w, _BAR), img)

    def _vbar(self, x, y, h, active):
        """Vertical drag-handle bar centred on edge x -> (MjrRect, img)."""
        import mujoco
        h = max(1, int(h))
        img = np.empty((h, _BAR, 3), np.uint8)
        img[:] = (80, 220, 255) if active else (95, 105, 125)
        return (mujoco.MjrRect(int(x - _BAR // 2), int(y), _BAR, h), img)
