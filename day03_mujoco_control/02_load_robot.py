import os
import time

import mujoco
import mujoco.viewer

# 1. XML模型路径
current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir, 
    "assets",
    "one_joint_arm.xml"
)

# 2. 加载模型
# model是：机器人"长什么样"，几个关节、刚体、连杆质量、关节类型、重力、timesleep、geometry、actuator
model = mujoco.MjModel.from_xml_path(xml_path)

# data是：机器人"现在什么状态"，
# 关节位置：data.qpos，
# 关节速度：data.qvel，
# 关节力矩：data.torque，
# 传感器数据：data.sensordata
data = mujoco.MjData(model)

# 3.打开Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():

        # 物理仿真前进一步
        # 根据机器人动力学：M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = tau，计算q, q_dot
        mujoco.mj_step(model, data)

        # 更新画面
        viewer.sync()

        # 控制仿真速度
        time.sleep(model.opt.timestep)  