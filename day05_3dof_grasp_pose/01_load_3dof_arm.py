import os
import time

import numpy as np
import mujoco
import mujoco.viewer


current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(
    current_dir, 
    "assets", 
    "three_link_arm.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("==== Model Information ====")
print("nq = ", model.nq)
print("nv = ", model.nv)
print("nu = ", model.nu)

# 设置一个初始姿态
data.qpos[:] = np.deg2rad([
    20.0, 
    40.0, 
    -60.0
])  

mujoco.mj_forward(model, data)

print("Initial q = ", np.rad2deg(data.qpos))

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)