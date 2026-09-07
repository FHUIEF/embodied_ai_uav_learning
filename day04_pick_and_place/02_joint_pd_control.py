import os
import time

import mujoco
import mujoco.viewer
import numpy as np

# 1.加载 xml
current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,
    "assets",
    "two_link_arm.xml"
)

print("xml_path =", xml_path)


model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2.打印模型信息
print("\n==== Model Information ====")
print("nq =", model.nq)     # 关节位置自由度
print("nv =", model.nv)     # 关节速度自由度    
print("nu =", model.nu)     # 执行器数量

print("\nInitial qpos:", data.qpos)
print("Initial qvel:", data.qvel)


# 3.设置初始姿态（可选）
# 为了更容易看出机械臂运动，可以让他从全零开始
data.qpos[0] = np.deg2rad(0.0)   # joint1 = 0°
data.qpos[1] = np.deg2rad(0.0)   # joint2 = 0°

# 手动设置状态后，需要前向更新
mujoco.mj_forward(model, data)

# 4.设置 PD 控制参数
# 目标关节角：q1 = 30°，q2 = 60°
q_desired = np.deg2rad(
    np.array([30.0, 60.0])
)

# 目标关节速度设为 0
qdot_desired = np.array([0.0, 0.0])

# 两个关节的 PD 参数
Kp = np.array([20.0, 20.0])   # 比例增益
Kd = np.array([10.0, 10.0])     # 微分增益

# actuator 限幅，与 xml 中 trlrange 保持一致
ctrl_min = -20.0
ctrl_max = 20.0

# 5.数据记录（先简单记录，后面可画图）
time_history = []
q_history = []
q_dot_history = []
torque_history = []
error_history = []

# 6.打开 Viewer，开始仿真
simulation_time = 15.0   # 仿真 15 秒

with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running() and data.time < simulation_time:

        # 当前状态
        q = data.qpos[:2].copy()
        q_dot = data.qvel[:2].copy()

        # 位置误差
        position_error = q_desired - q
        # 速度误差
        velocity_error = qdot_desired - q_dot

        # PD 控制律
        # tau = Kp(qd-q) + Kd(qd_dot - q_dot)
        torque = (
            Kp * position_error
            +
            Kd * velocity_error
        )

        # 限幅
        torque = np.clip(torque, ctrl_min, ctrl_max)

        # 输出给两个 actuator
        data.ctrl[:] = torque

        # 仿真推进一步
        mujoco.mj_step(model, data)

        # 记录数据
        time_history.append(data.time)
        q_history.append(data.qpos[:2].copy())
        q_dot_history.append(data.qvel[:2].copy())
        torque_history.append(data.ctrl[:2].copy())
        error_history.append((q_desired - data.qpos[:2]).copy())

        # 打印调试信息
        step_count += 1

        if step_count % 100 == 0:
            print(
                f"time = {data.time:.2f} s | "
                f"q1 = {np.rad2deg(q[0]):.2f} deg | "
                f"q2 = {np.rad2deg(q[1]):.2f} deg | "
                f"q1_dot = {np.rad2deg(data.qvel[0]):.2f} deg/s | "
                f"q2_dot = {np.rad2deg(data.qvel[1]):.2f} deg/s | "
                f"tau1 = {data.ctrl[0]:.3f}, "
                f"tau2 = {data.ctrl[1]:.3f}"
            )


        # 更新 Viewer
        viewer.sync()

        # 控制显示速度
        time.sleep(model.opt.timestep)

print("\nSimulation finished.")

print("\nFinal joint angles ")

print(
    f"q1 = {np.rad2deg(data.qpos[0]):.2f} deg | "
    f"q2 = {np.rad2deg(data.qpos[1]):.2f} deg"
)

print(
    f"q1_desired = {np.rad2deg(q_desired[0]):.2f} deg | "
    f"q2_desired = {np.rad2deg(q_desired[1]):.2f} deg"
)