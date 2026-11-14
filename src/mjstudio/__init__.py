"""mjstudio — a reusable native MuJoCo paneled viewer.

One GLFW window: the interactive MuJoCo 3D view + a camera-thumbnail strip + an
info/log panel. Runs its loop on the main thread; feed it state from a worker thread.

    from mjstudio import PanelViewer

    viewer = PanelViewer(model, cameras=["cam_top", "cam_back"])
    def work(v):
        for i in range(1000):
            v.set_qpos(data.qpos); v.info(step=i); v.log("...")
    viewer.run(worker=work)   # blocks until the window closes
"""
from .feed import Feed
from .viewer import PanelViewer

__all__ = ["PanelViewer", "Feed"]
__version__ = "0.1.0"
