"""mjstudio — wrap MuJoCo's own viewer and add panels around it.

Launches the real `mujoco.viewer` (full interactive 3D for free) and uses its official
overlay hooks to add a camera-thumbnail strip + an info/log panel.

    from mjstudio import PanelViewer

    viewer = PanelViewer(model, data, cameras=["cam_top", "cam_back"]).open()
    while viewer.is_running():
        step()
        viewer.info(step=i); viewer.log("...")
        viewer.update()        # thumbnails + panel + viewer.sync()
"""
from .viewer import PanelViewer

__all__ = ["PanelViewer"]
__version__ = "0.2.0"
