"""PanelViewer: one GLFW window = interactive MuJoCo 3D + camera strip + info/log panel.

Runs its render loop on the MAIN thread (GLFW needs it on macOS); the caller's work
(e.g. an inference loop) runs on a worker thread and pushes state through ``self.feed``
(or the ``set_qpos`` / ``info`` / ``log`` shortcuts). The window opens as soon as you
call ``open()`` — before any heavy work the worker does.
"""
from __future__ import annotations

import threading
import time

import glfw
import mujoco

from . import layout as _layout
from .feed import Feed
from .panels import draw_panel

_FONT = mujoco.mjtFont.mjFONT_NORMAL


class PanelViewer:
    def __init__(self, model, cameras=None, title: str = "mjstudio",
                 maxgeom: int = 10000, fps: float = 30.0, width: int = 1200, height: int = 800):
        self._model = model
        self._cameras = list(cameras or [])
        self._title = title
        self._maxgeom = maxgeom
        self._fps = fps
        self._w, self._h = width, height

        self.feed = Feed()
        self._data = mujoco.MjData(model)
        self._window = None
        self._scn = None
        self._con = None
        self._cam = mujoco.MjvCamera()          # the free (orbit) camera
        self._opt = mujoco.MjvOption()
        self._cam_ids: list[int] = []
        self._running = False
        self._follow = -1                        # -1 = free cam; else index into cameras
        self._last = (0.0, 0.0)
        self._btn = {"left": False, "right": False, "middle": False}

    # ---- feed shortcuts (callable from the worker thread) ----
    def set_qpos(self, qpos): self.feed.set_qpos(qpos)
    def set_frames(self, frames): self.feed.set_frames(frames)
    def info(self, **kw): self.feed.info(**kw)
    def log(self, msg): self.feed.log(msg)

    # ---- lifecycle ----
    def open(self) -> "PanelViewer":
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)
        self._window = glfw.create_window(self._w, self._h, self._title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed (no display?)")
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        self._cam_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in self._cameras
        ]
        mujoco.mjv_defaultOption(self._opt)
        mujoco.mjv_defaultFreeCamera(self._model, self._cam)
        self._scn = mujoco.MjvScene(self._model, maxgeom=self._maxgeom)
        self._con = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150)

        glfw.set_cursor_pos_callback(self._window, self._on_cursor)
        glfw.set_mouse_button_callback(self._window, self._on_button)
        glfw.set_scroll_callback(self._window, self._on_scroll)
        glfw.set_key_callback(self._window, self._on_key)
        self._running = True
        return self

    def is_running(self) -> bool:
        return (self._running and self._window is not None
                and not glfw.window_should_close(self._window))

    def render(self) -> None:
        """Draw one frame from the latest fed state. Call on the main thread."""
        if not self.is_running():
            return
        info, logs = self._apply_feed()
        fb_w, fb_h = glfw.get_framebuffer_size(self._window)
        rects = _layout.compute(fb_w, fb_h, len(self._cam_ids))

        # main interactive view (free cam, or following a fixed camera)
        maincam = self._fixed_cam(self._cam_ids[self._follow]) if self._follow >= 0 else self._cam
        label = self._cameras[self._follow] if self._follow >= 0 else "free-cam  (Tab: cycle)"
        mujoco.mjv_updateScene(self._model, self._data, self._opt, None, maincam,
                               mujoco.mjtCatBit.mjCAT_ALL, self._scn)
        mujoco.mjr_render(rects.main, self._scn, self._con)
        mujoco.mjr_overlay(_FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, rects.main, label, "", self._con)

        # camera thumbnails
        for i, cid in enumerate(self._cam_ids):
            if cid < 0 or i >= len(rects.thumbs):
                continue
            mujoco.mjv_updateScene(self._model, self._data, self._opt, None,
                                   self._fixed_cam(cid), mujoco.mjtCatBit.mjCAT_ALL, self._scn)
            mujoco.mjr_render(rects.thumbs[i], self._scn, self._con)
            mujoco.mjr_overlay(_FONT, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               rects.thumbs[i], self._cameras[i], "", self._con)

        draw_panel(self._con, rects.panel, info, logs, title=self._title)
        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def run(self, worker=None) -> None:
        """Blocking main-thread loop. If ``worker`` is given it runs on a side thread
        as ``worker(self)``; the loop ends when the window closes or the worker returns."""
        if not self._running:
            self.open()
        if worker is not None:
            threading.Thread(target=self._worker_wrap, args=(worker,), daemon=True).start()
        dt = 1.0 / self._fps
        try:
            while self.is_running():
                t0 = time.time()
                self.render()
                nap = dt - (time.time() - t0)
                if nap > 0:
                    time.sleep(nap)
        finally:
            self.close()

    def close(self) -> None:
        self._running = False
        if self._con is not None:
            try:
                self._con.free()
            except Exception:  # noqa: BLE001
                pass
            self._con = None
        if self._window is not None:
            try:
                glfw.destroy_window(self._window)
            except Exception:  # noqa: BLE001
                pass
            self._window = None
        try:
            glfw.terminate()
        except Exception:  # noqa: BLE001
            pass

    # ---- internals ----
    def _fixed_cam(self, cam_id: int):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FIXED
        c.fixedcamid = cam_id
        return c

    def _apply_feed(self):
        q, info, logs, _frames = self.feed.snapshot()
        if q is not None and q.shape[0] == self._data.qpos.shape[0]:
            self._data.qpos[:] = q
            mujoco.mj_forward(self._model, self._data)
        return info, logs

    def _worker_wrap(self, worker):
        try:
            worker(self)
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the window
            self.log(f"worker error: {exc}")
        if self._window is not None:
            glfw.set_window_should_close(self._window, True)

    # ---- input callbacks ----
    def _on_cursor(self, window, xpos, ypos):
        dx, dy = xpos - self._last[0], ypos - self._last[1]
        self._last = (xpos, ypos)
        if self._follow >= 0 or not any(self._btn.values()):
            return
        h = max(1, glfw.get_window_size(window)[1])
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                 or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._btn["left"]:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._btn["right"]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self._model, action, dx / h, dy / h, self._scn, self._cam)

    def _on_button(self, window, button, act, mods):
        down = act == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._btn["left"] = down
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._btn["right"] = down
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._btn["middle"] = down
        self._last = glfw.get_cursor_pos(window)

    def _on_scroll(self, window, xoff, yoff):
        if self._follow >= 0:
            return
        mujoco.mjv_moveCamera(self._model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoff, self._scn, self._cam)

    def _on_key(self, window, key, scancode, act, mods):
        if act != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_TAB:
            self._follow += 1
            if self._follow >= len(self._cam_ids):
                self._follow = -1
