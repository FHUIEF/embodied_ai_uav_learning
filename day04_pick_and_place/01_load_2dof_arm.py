import mujoco
import mujoco.viewer

import os 
import time
import numpy as np

# 1. 获取 XML模型路径
current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,
    "assets",
    "two_link_arm.xml"
)


# 2. 加载 Mujoco模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


# 3. 检查模型自由度和执行器
print("\n=== Model Information  ===")

print("Position DoF (nq):", model.nq)
print("Velocity DoF (nv):", model.nv)
print("Actuator DoF (nu):", model.nu)

print("\nInitial qpos:", data.qpos)

print("\nInitial qvel:", data.qvel)


# 4. 设置机械臂初始关节角
# joint1 = 30°，joint2 = 60°
data.qpos[0] = np.deg2rad(30)
data.qpos[1] = np.deg2rad(60)

# 手动修改 qpos后
# 让 Mujoco 重新计算所有 body 的位置和姿态
mujoco.mj_forward(model, data) 

print("\nAfter setting initial joint angles:")
print(f"q1: {np.rad2deg(data.qpos[0]):.2f} deg")
print(f"q2: {np.rad2deg(data.qpos[1]):.2f} deg")

# 5. 打开 Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        
        # 推进一步物理仿真
        mujoco.mj_step(model, data)

        # 更新 Viewer
        viewer.sync()

        # 尽量让仿真速度接近真实时间
        time.sleep(model.opt.timestep)
