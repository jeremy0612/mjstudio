#!/usr/bin/env python3
"""Standalone mjstudio demo — wraps MuJoCo's own viewer, adds a camera thumbnail + panel.

A double pendulum swings under gravity. The real MuJoCo viewer handles all the 3D
(orbit/zoom/keybinds); mjstudio overlays one camera thumbnail (bottom) and an info/log
panel (top-right).

    python mjstudio/examples/pendulum.py          # Linux/Windows
    mjpython mjstudio/examples/pendulum.py         # macOS (passive viewer needs mjpython)
"""
import time

import mujoco

from mjstudio import PanelViewer

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <camera name="side" pos="0 -2.5 0.5" xyaxes="1 0 0 0 0 1"/>
    <body pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" rgba="0.8 0.3 0.3 1"/>
      <body pos="0 0 -0.5">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" rgba="0.3 0.5 0.8 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = [2.0, -1.0]

    viewer = PanelViewer(model, data, cameras=["side"], title="mjstudio demo").open()
    t0 = time.time()
    i = 0
    while viewer.is_running():
        with viewer.lock():        # serialize physics with the background pump's sync()
            mujoco.mj_step(model, data)
        if i % 10 == 0:
            viewer.info(step=i, sim_t=round(data.time, 2),
                        fps=round((i + 1) / (time.time() - t0 + 1e-9), 1))
        if i % 500 == 0:
            viewer.log(f"tick {i}: j1={data.qpos[0]:+.2f} j2={data.qpos[1]:+.2f}")
        viewer.update()          # thumbnail + panel + viewer.sync()
        time.sleep(0.002)
        i += 1
    viewer.close()


if __name__ == "__main__":
    main()
