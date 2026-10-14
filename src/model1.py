import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path(
    str(Path(__file__).resolve().parent.parent / "envs" / "printer.xml")
)
d = mujoco.MjData(m)
print([m.actuator(i).name for i in range(m.nu)])  # see what you can control

for i in range(m.nu):
    a = m.actuator(i)
    lo, hi = a.ctrlrange
    j = m.joint(a.trnid[0])  # the joint this actuator drives
    jlo, jhi = j.range
    print(
        f"{a.name:15s} ctrl [{lo:+.3f}, {hi:+.3f}]  "
        f"joint {j.name} [{jlo:+.3f}, {jhi:+.3f}]  "
        f"({np.degrees(jlo):+.0f}° to {np.degrees(jhi):+.0f}°)"
    )

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        d.ctrl[:] = 0.3 * np.sin(d.time) * np.ones(m.nu)  # replace with your controller
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(m.opt.timestep)
