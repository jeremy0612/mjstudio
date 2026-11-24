# mjstudio

A thin wrapper around **MuJoCo's own viewer** that adds **mouse-draggable panels** around
it — without reimplementing the 3D view.

It launches the real `mujoco.viewer` (`launch_passive`), so all the interactive 3D (orbit,
zoom, perturbations, camera cycling, keybinds) **and MuJoCo's native left/right UI panels**
come straight from MuJoCo. On top of it, mjstudio overlays:

- a **camera-thumbnail strip** (one live render per named fixed camera) via `set_images`,
- an **info + log panel** (key/value info and a timestamped log, newest first) via `set_images` (PIL) or
  `set_texts` (fallback).

**Grab a panel edge and drag to resize it.** The strip's top edge sets the thumbnail scale;
the info panel's left/bottom edges set its width/height. Panels are sized as fractions of the
3D viewport, so they stay put when you open/close the native panels or resize the window.

```python
from mjstudio import PanelViewer

viewer = PanelViewer(model, data, cameras=["cam_left_wrist", "cam_top"]).open()
while viewer.is_running():
    step()                         # you own model/data and the clock
    viewer.info(step=i, fps=f)     # key/value lines in the panel
    viewer.log("event happened")   # timestamped, newest on top
    viewer.update()                # thumbnails + panel + viewer.sync()
viewer.close()
```

`viewer.update(frames=...)` takes `{cam_name: HWC uint8}` to reuse renders you already have
(e.g. a policy's observation frames); omit it and mjstudio renders the cameras itself.

## Why a patched MuJoCo?

Dragging needs the cursor position. MuJoCo's passive viewer owns the window and the mouse,
and **never exposed the cursor to Python** (only `key_callback`). mjstudio adds it with a small
patch to MuJoCo's `_simulate` (`patches/expose-sim-mouse.patch`) that exposes `Simulate.mouse`
(cursor + buttons, read via `handle._get_sim().mouse` — framebuffer pixels, bottom-left origin,
same frame as the overlay rects, so no conversion).

To keep the 3D scene from orbiting while you drag a panel (MuJoCo turns any left-drag in the 3D
view into a camera orbit, and gives Python no way to veto it), mjstudio switches the viewer
camera to type `USER` on grab — MuJoCo then renders the last pose and ignores the orbit, so
there's no shake — and restores the free camera and its saved pose on release. No patch needed
for that part; it's pure Python via `handle.cam`.

The patch only touches the python `_simulate` extension; **the MuJoCo core library is not
rebuilt** — it is linked from your existing `mujoco` install, so the patched extension is
ABI-compatible with the rest of it. On **unpatched** MuJoCo the panels still render; they just
can't be dragged (a one-time log line says so).

### Install the patched MuJoCo (pick one)

**A. Drop-in the released extension** (fastest — no build):
download `_simulate.cpython-*.so` for your platform from this repo's
[Releases](https://github.com/jeremy0612/mjstudio/releases) and copy it over your install:

```bash
MJ=$(python -c 'import mujoco,os;print(os.path.dirname(mujoco.__file__))')
cp "$MJ"/_simulate.*.so "$MJ"/_simulate.orig        # backup
cp ~/Downloads/_simulate.cpython-312-darwin.so "$MJ"/   # the released file
```

**B. Build it yourself** (any platform / MuJoCo version):

```bash
# venv needs: pip build cmake ninja  (uv: `uv pip install pip build cmake ninja`)
git submodule update --init third_party/mujoco
scripts/build_patched_mujoco.sh            # -> mjstudio/dist/*.whl + _simulate*.so
```

Then drop in `dist/_simulate*.so` as in A, or `pip install dist/mujoco_core-*.whl`.

### Sync with upstream MuJoCo

The upstream source is tracked as the `third_party/mujoco` submodule, pinned to the tag in
`patches/MUJOCO_VERSION`. To move to a new MuJoCo release:

```bash
cd third_party/mujoco && git fetch --tags && git checkout <new-tag> && cd -
echo <new-tag> > patches/MUJOCO_VERSION
git -C third_party/mujoco apply --3way ../../patches/expose-sim-mouse.patch   # re-check it applies
scripts/build_patched_mujoco.sh
```

If the patch no longer applies cleanly, fix the three lines it adds (a `GetMouse()` method and
one `def_property_readonly("mouse", ...)`) and regenerate the patch with `git -C third_party/mujoco diff`.

## Install (the library)

```bash
pip install -e .            # mjstudio itself
pip install -e '.[panel]'   # + Pillow, for the draggable image info panel
```

## Controls
3D view + native UI: all MuJoCo's own (drag to orbit, scroll to zoom, `Tab`/`[`/`]` to cycle
cameras, etc.). Panels: the camera strip fills the viewport width — drag its top edge to change
height (up to the frames' own aspect ratio); the info/log panel's left + bottom edges resize it on both axes. Thumbnails keep their
aspect ratio (letterboxed) — resizing changes the window, not the frame. While you drag a panel
the 3D camera is held completely still (the viewer camera is switched to type `USER` for the
duration of the drag), so the scene doesn't orbit or shake. Hold and pull the scrollbar to scroll the info/log panel.

## Demo

```bash
python   mjstudio/examples/pendulum.py      # Linux / Windows
mjpython mjstudio/examples/pendulum.py      # macOS
```

## Notes
- **Background pump (default)**: `open()` starts a ~60 Hz thread that renders the overlays
  and handles dragging on its own. Your app just feeds data (`info`/`log`/`update(frames=)`),
  so the panels stay live and **draggable even while your thread is blocked** — loading a big
  model, or stalled in a slow step. Pass `open(auto=False)` to drive everything yourself from
  `update()` (single-thread apps). `sync`/`set_images`/`set_texts` lock internally, so the pump
  is thread-safe.
- **Hold `viewer.lock()` around data mutations** when the pump is on: the pump copies your
  `mjData` off-thread for rendering, so any `mj_step`/`mj_forward`/`mjv_updateScene` you run
  must hold the sim mutex or MuJoCo raises *"attempting to copy mjData while stack is in use"*.
  ```python
  while viewer.is_running():
      with viewer.lock():
          mujoco.mj_step(model, data)   # or mj_forward, or render the data yourself
      viewer.info(...); viewer.update()
  ```
- While a panel is being dragged, mjstudio switches the viewer camera to type `USER` so MuJoCo
  ignores the 3D-drag orbit (the scene stays still — no orbit, no shake) and restores the free
  camera + its saved pose on release. Pure Python via `handle.cam`; no patch required.
- **macOS**: the passive viewer requires `mjpython` (MuJoCo's own requirement).
- If thumbnails/panel appear upside down, pass `flip_images=True`.
- Overlay hooks (`set_images`/`set_texts`) need a MuJoCo with passive-viewer overlay support (3.x+).

MIT licensed.
