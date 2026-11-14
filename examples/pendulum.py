#!/usr/bin/env python3
"""Standalone mjstudio demo — no robot, no extra deps.

A double pendulum swings under gravity. One fixed camera shows up as a thumbnail; the
side panel streams step / time / fps and a growing log. Drag to orbit, scroll to zoom,
Tab to cycle the main view through the fixed camera, Esc to quit.

    python mjstudio/examples/pendulum.py
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

    viewer = PanelViewer(model, cameras=["side"], title="mjstudio demo — double pendulum")
    viewer.open()  # window up immediately

    def work(v):
        t0 = time.time()
        for i in range(100_000):
            if not v.is_running():
                break
            mujoco.mj_step(model, data)
            v.set_qpos(data.qpos)
            if i % 10 == 0:
                fps = (i + 1) / (time.time() - t0 + 1e-9)
                v.info(step=i, sim_t=round(data.time, 2), fps=round(fps, 1))
            if i % 500 == 0:
                v.log(f"tick {i}: j1={data.qpos[0]:+.2f} j2={data.qpos[1]:+.2f}")
            time.sleep(0.002)

    viewer.run(worker=work)


if __name__ == "__main__":
    main()
