"""
3DOF Joint PD Control
"""

import os
import time

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# 1. 加载模型
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

# 2. 目标关节角
q_desired = np.deg2rad([
    20.0,
    40.0,
    -60.0
])  # 单位 rad

qdot_desired = np.zeros(3)  # 期望关节速度为零

print("\n==== Desired Joint Angles ====")
print("q_desired = ", q_desired, " deg")

# 3. PD 参数
Kp = np.array([20.0, 20.0, 15.0])  # 比例增益
Kd = np.array([10.0, 10.0, 6.0])    # 微分增益

TORQUE_LIMIT = np.array([20.0, 20.0, 10.0])  # 最大关节力矩限制

# 4. PD 控制器
def joint_pd_control(q, qdot, q_desired, qdot_desired):
    postion_error = q_desired - q
    velocity_error = qdot_desired - qdot

    torque = Kp * postion_error + Kd * velocity_error

    return torque

# 5. 初始化
data.qpos[:] = np.deg2rad([0.0, 0.0, 0.0])  # 初始关节角为零
data.qvel[:] = 0.0  # 初始关节速度为零

mujoco.mj_forward(model, data)

# 6. 数据记录
time_history = []
q_history = []
qdot_history = []
torque_history = []
error_history = []

# 7. 仿真时间
simulation_time = 10.0  # 仿真总时间

# 8. 开始仿真
with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while (viewer.is_running() and data.time < simulation_time):

        # 当前关节状态
        q = data.qpos[:3].copy()
        qdot = data.qvel[:3].copy()

        # PD 控制
        torque = joint_pd_control(
            q, 
            qdot, 
            q_desired, 
            qdot_desired
        )

        # 给 actuator 力矩
        data.ctrl[:] = torque

        # 记录数据
        position_error = q_desired - q

        time_history.append(data.time)

        q_history.append(q.copy())

        qdot_history.append(qdot.copy())

        torque_history.append(torque.copy())

        error_history.append(position_error.copy())

        # 推进仿真
        mujoco.mj_step(model, data)

        # 调试输出
        step_count += 1

        if step_count % 100 == 0:
            print(f"Time = {data.time:.2f}s |")
            print(f"q = {np.round(np.rad2deg(q), 2)} deg | ")
            print(f"error = {np.round(np.rad2deg(position_error), 2)} deg | ")
            print(f"tau = {np.round(torque, 2)} Nm")

        viewer.sync()

        time.sleep(model.opt.timestep)  

# 9. 转成 numpy array
time_history = np.array(time_history)

q_history = np.array(q_history)

qdot_history = np.array(qdot_history)

torque_history = np.array(torque_history)

error_history = np.array(error_history)

# 10. 最终状态
final_q = data.qpos[:3].copy()

final_qdot = data.qvel[:3].copy()

final_error = q_desired - final_q

print("\n==== Final Result ====")
print(f"Desired q = {np.round(np.rad2deg(q_desired), 2)} deg")
print(f"Final q = {np.round(np.rad2deg(final_q), 2)} deg")
print(f"Final error = {np.round(np.rad2deg(final_error), 2)} deg")

# 11. 绘制关节角
plt.figure()
plt.plot(time_history, np.rad2deg(q_history[:, 0]), label='q1')
plt.plot(time_history, np.rad2deg(q_history[:, 1]), label='q2')
plt.plot(time_history, np.rad2deg(q_history[:, 2]), label='q3')

plt.axhline(np.rad2deg(q_desired[0]), linestyle='--', label='q1 desired')
plt.axhline(y=np.rad2deg(q_desired[1]), linestyle='--', label='q2 desired')
plt.axhline(y=np.rad2deg(q_desired[2]),linestyle='--', label='q3 desired')

plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (deg)')
plt.title('3DOF Joint Position')

plt.grid()
plt.legend()

# 12. 绘制关节误差
plt.figure()
plt.plot(time_history, np.rad2deg(error_history[:, 0]), label='q1 error')
plt.plot(time_history, np.rad2deg(error_history[:, 1]), label='q2 error')
plt.plot(time_history, np.rad2deg(error_history[:, 2]), label='q3 error')

plt.xlabel('Time (s)')
plt.ylabel('Error (deg)')
plt.title('3DOF Joint Error')

plt.grid()
plt.legend()

# 13. 绘制关节速度
plt.figure()
plt.plot(time_history, np.rad2deg(qdot_history[:, 0]), label='q1 dot')
plt.plot(time_history, np.rad2deg(qdot_history[:, 1]), label='q2 dot')
plt.plot(time_history, np.rad2deg(qdot_history[:, 2]), label='q3 dot')  

plt.xlabel('Time (s)')
plt.ylabel('Joint Velocity (deg/s)')
plt.title('3DOF Joint Velocity')

plt.grid()
plt.legend()

# 14. 绘制关节力矩
plt.figure()
plt.plot(time_history, torque_history[:, 0], label='tau1')
plt.plot(time_history, torque_history[:, 1], label='tau2')
plt.plot(time_history, torque_history[:, 2], label='tau3')

plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('3DOF Control Torque')

plt.grid()
plt.legend()
plt.show()
