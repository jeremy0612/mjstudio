"""PanelViewer — wraps MuJoCo's own passive viewer and adds panels around it.

This does NOT reimplement the 3D view. It launches the real `mujoco.viewer`
(`launch_passive`) — all its camera controls, perturbations, keybinds and UI come for
free — and uses that viewer's official overlay hooks (`set_images`, `set_texts`) to add
a camera-thumbnail strip and an info/log panel on top. "Embed the original, wrap it."

    viewer = PanelViewer(model, data, cameras=["cam_top", "cam_back"]).open()
    while viewer.is_running():
        step()                         # you own the model/data and the clock
        viewer.info(step=i, fps=f)
        viewer.log("event")
        viewer.update()                # render thumbnails + overlays + viewer.sync()

macOS: the passive viewer needs `mjpython` (that is MuJoCo's requirement, not ours).
"""
from __future__ import annotations

import time
from collections import deque


class PanelViewer:
    def __init__(self, model, data, cameras=None, title: str = "mjstudio",
                 thumb=(320, 240), log_maxlen: int = 400, flip_images: bool = False):
        self._m = model
        self._d = data
        self._cameras = list(cameras or [])
        self._tw, self._th = thumb
        self._flip = flip_images
        self._handle = None
        self._renderer = None  # lazy; only used when update() must render its own thumbnails
        self._info: dict = {}
        self._logs: deque[str] = deque(maxlen=log_maxlen)
        self._t0 = time.time()
        self.title = title

    # ---- lifecycle ----
    def open(self, show_left_ui: bool = False, show_right_ui: bool = False) -> "PanelViewer":
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
        """Overlay the camera thumbnails + info/log panel, then sync the real viewer.

        Pass ``frames`` ({cam_name: HWC uint8}) to reuse renders you already have
        (avoids a second offscreen renderer); omit it and PanelViewer renders the
        cameras itself.
        """
        if not self.is_running():
            return
        import mujoco

        frames = self._render_thumbs() if frames is None else frames
        if frames:
            imgs, x = [], 0
            for name in self._cameras:
                fr = frames.get(name)
                if fr is None:
                    continue
                if self._flip:
                    fr = fr[::-1]
                h, w = fr.shape[:2]
                imgs.append((mujoco.MjrRect(x, 0, w, h), fr))  # strip along the bottom
                x += w
            if imgs:
                self._handle.set_images(imgs)

        lines = [self.title, "-" * 26]
        lines += [f"{k}: {v}" for k, v in self._info.items()]
        lines += ["", "logs:"] + list(self._logs)[-18:]
        self._handle.set_texts([(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "\n".join(lines), "",
        )])
        self._handle.sync()

    def _render_thumbs(self) -> dict:
        if not self._cameras:
            return {}
        import mujoco
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._m, self._th, self._tw)
        out = {}
        for name in self._cameras:
            self._renderer.update_scene(self._d, camera=name)
            out[name] = self._renderer.render()
        return out
