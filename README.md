# mjstudio

A thin wrapper around **MuJoCo's own viewer** that adds panels around it — without
reimplementing the 3D view.

It launches the real `mujoco.viewer` (`launch_passive`), so all the interactive 3D
(orbit, zoom, perturbations, camera cycling, keybinds, built-in UI) comes straight from
MuJoCo. On top of it, mjstudio uses the passive viewer's official overlay hooks to add:

- a **camera-thumbnail strip** (one live render per named fixed camera) via `set_images`,
- an **info + log panel** (key/value info and a timestamped log) via `set_texts`.

Built for watching and profiling policies in sim, and reusable across MuJoCo projects.
Deps: `mujoco` + `numpy` only.

## Install

```bash
pip install -e .
```

## Use

```python
from mjstudio import PanelViewer

viewer = PanelViewer(model, data, cameras=["cam_left_wrist", "cam_top"]).open()
while viewer.is_running():
    step()                         # you own model/data and the clock
    viewer.info(step=i, fps=f)     # key/value lines in the panel
    viewer.log("event happened")   # timestamped log tail
    viewer.update()                # thumbnails + panel + viewer.sync()
viewer.close()
```

`viewer.update(frames=...)` takes `{cam_name: HWC uint8}` to reuse renders you already
have (e.g. a policy's observation frames) instead of rendering the thumbnails a second
time; omit it and mjstudio renders the cameras itself with its own offscreen renderer.

### Controls
All from MuJoCo's own viewer (unchanged): drag to orbit, right-drag to pan, scroll to
zoom, double-click + Ctrl-drag to perturb, `Tab`/`[`/`]` to cycle cameras, etc.

## Demo

```bash
python   mjstudio/examples/pendulum.py      # Linux / Windows
mjpython mjstudio/examples/pendulum.py      # macOS
```

## Notes
- **macOS**: the passive viewer requires `mjpython` (MuJoCo's own requirement).
- Overlay hooks (`set_images`/`set_texts`) need a MuJoCo with passive-viewer overlay
  support (3.x+). If thumbnails appear upside down, pass `flip_images=True`.
- The viewer shows the `model`/`data` you pass in; mutate them in your loop and call
  `update()` — mjstudio never steps the sim itself.

MIT licensed.
