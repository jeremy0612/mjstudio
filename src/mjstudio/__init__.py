"""mjstudio — wrap MuJoCo's own viewer and add mouse-draggable panels around it.

Launches the real `mujoco.viewer` (full interactive 3D + native left/right UI for free)
and uses its official overlay hooks to add a camera-thumbnail strip + an info/log panel.
Grab a panel edge and drag to resize — that needs the cursor position, which a tiny patch
to MuJoCo's `_simulate` exposes as `Simulate.mouse` (see the repo's patches/ + scripts/).
On unpatched MuJoCo the panels still render; they just can't be dragged.

    from mjstudio import PanelViewer

    viewer = PanelViewer(model, data, cameras=["cam_top", "cam_back"]).open()
    while viewer.is_running():
        step()
        viewer.info(step=i); viewer.log("...")
        viewer.update()        # thumbnails + panel + viewer.sync()
"""
from .viewer import PanelViewer

__all__ = ["PanelViewer"]
__version__ = "0.3.0"
