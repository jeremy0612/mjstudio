# mjstudio

A small, reusable **native MuJoCo paneled viewer**. One GLFW window holding:

- the real interactive MuJoCo 3D view (orbit / pan / zoom, same `mjv_*`/`mjr_*` stack as
  the stock viewer),
- a **camera-thumbnail strip** (one live render per named fixed camera),
- an **info + log panel** (key/value info and a timestamped, scrolling log).

Built for watching and **profiling** policies in simulation, and for reuse across MuJoCo
projects. Pure `mujoco` + `glfw` + `numpy` — no web stack, no imgui.

## Why it exists

MuJoCo's stock passive viewer can't host custom panels, and on macOS it needs `mjpython`
(which runs your code off the main thread). `mjstudio` instead **is** the main-thread loop
in a plain `python` process, so the window opens the moment you build the model — before
any heavy work — and your sim/inference runs on a worker thread that feeds it state.

## Install

```bash
pip install -e .          # from this directory
# deps: mujoco>=3.2, glfw>=2.6, numpy
```

## Use

```python
from mjstudio import PanelViewer

viewer = PanelViewer(model, cameras=["cam_left_wrist", "cam_top"], title="my sim")
viewer.open()                         # window up now

def work(v):                          # runs on a worker thread
    while v.is_running():
        step()
        v.set_qpos(data.qpos)         # drives the 3D view + all thumbnails
        v.info(step=i, fps=f)         # key/value lines in the side panel
        v.log("event happened")       # timestamped log tail

viewer.run(worker=work)               # blocking main-thread loop; ends on window close
```

Or drive it manually: `viewer.open()` then `while viewer.is_running(): viewer.render()`.

### Controls
- **Left-drag** orbit · **Right-drag** pan · **Scroll** zoom (hold **Shift** for the
  horizontal variants).
- **Tab** cycles the main view: free camera → each fixed camera → back.
- **Esc** or closing the window stops the loop (and the worker).

## Demo

```bash
python examples/pendulum.py
```

## Notes / limits
- The viewer renders cameras itself from the fed `qpos`; pushing pre-rendered frames via
  `set_frames` is reserved for overlaying real-camera images later.
- `mjr_overlay` text uses fixed grid anchors; the log panel shows a bounded tail.
- macOS: run under plain `python` (not `mjpython`); GLFW owns the main thread here.

MIT licensed.
