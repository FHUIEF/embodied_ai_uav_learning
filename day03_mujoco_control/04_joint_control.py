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
print("关节自由度 nq =", model.nq)
print("速度自由度 nv =", model.nv)
print("执行器数量 nu =", model.nu)



with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running():

        # 给第一个执行器持续施加 1 N-m 左右的控制量
        data.ctrl[0] = 1.0

        mujoco.mj_step(model, data)

        step_count += 1
        
        if step_count % 100 == 0:
            print(
                f"ctrl = {data.ctrl[0]:.4f}",
                f"tau = {data.qfrc_actuator[0]:.6f}",
                f"constraint = {data.qfrc_constraint[0]:.6f}",
                f"qacc = {data.qacc[0]:.6f}",
                f"qdot = {data.qvel[0]:.6f}",
                f"q = {data.qpos[0]:.6f},"
            )

        viewer.sync()

        time.sleep(model.opt.timestep)

        