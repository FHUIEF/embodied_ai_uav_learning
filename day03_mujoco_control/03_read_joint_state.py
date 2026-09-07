import os
import time

import mujoco
import mujoco.viewer

current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,
    "assets",
    "one_joint_arm.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        print(
            "qpos:", data.qpos,
            "qvel:", data.qvel,
        )

        viewer.sync()

        time.sleep(model.opt.timestep)