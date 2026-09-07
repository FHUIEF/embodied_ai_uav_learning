import os
import time

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt


current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,  
    "assets",
    "one_joint_arm.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# PD参数
Kp = 10.0  # 比例增益
Kd = 2.0   # 微分增益

# 目标角度：90°
q_desired = np.deg2rad(90.0)

# 目标角速度：0
qdot_desired = 0.0

# 存储数据
time_history = []
q_history = []
q_desired_history = []
qdot_history = []
torque_history = []
error_history = []

# 仿真时间
simulation_time = 8.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running() and data.time < simulation_time:


        # 当前状态（角度和角速度）
        q = data.qpos[0]
        qdot = data.qvel[0]

        # 位置误差
        position_error = q_desired - q
        # 速度误差
        velocity_error = qdot_desired - qdot

        # PD控制
        torque = (
            Kp * position_error 
            +
            Kd * velocity_error
        )

        # 执行器范围限制
        torque = np.clip(torque, -10.0, 10.0)

        # 控制输入
        data.ctrl[0] = torque

        # 仿真一步
        mujoco.mj_step(model, data)

        # 记录数据
        time_history.append(data.time)

        q_history.append(data.qpos[0])

        q_desired_history.append(q_desired)

        qdot_history.append(data.qvel[0])

        torque_history.append(data.ctrl[0])

        error_history.append(
            q_desired - data.qpos[0]
        )

        step_count += 1
        if step_count % 100 == 0:
            print(
                f"target = {np.rad2deg(q_desired):.2f} deg | "
                f"q: {np.rad2deg(data.qpos[0]):.2f} deg | "
                f"qdot: {np.rad2deg(data.qvel[0]):.2f} deg/s | "
                f"torque: {torque:.3f} N-m"
            )

        viewer.sync()

        time.sleep(model.opt.timestep)
    
    time_history = np.array(time_history)
    
    q_history = np.array(q_history)
    
    q_desired_history = np.array(q_desired_history)
    
    qdot_history = np.array(qdot_history)
    
    torque_history = np.array(torque_history)
    
    error_history = np.array(error_history)

def plot_results(
        time_history,
        q_history,
        q_desired_history,
        qdot_history,
        torque_history,
        error_history
):
    # 图1：关节位置响应
    plt.figure()

    plt.plot(
        time_history, 
        np.rad2deg(q_history), 
        label="Actual q"
    )

    plt.plot(
        time_history,
        np.rad2deg(q_desired_history),
        "--",
        label="Desired q"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (deg)")

    plt.title("PD Joint Position Response")

    plt.grid()
    plt.legend()

    plt.show()

    # 图2：误差曲线
    plt.figure()

    plt.plot(
        time_history,
        np.rad2deg(error_history)
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (deg)")

    plt.title("Position Error")

    plt.grid()

    plt.show()

    # 图3：速度响应
    plt.figure()

    plt.plot(
        time_history,
        np.rad2deg(qdot_history)
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint Velocity (deg/s)")

    plt.title("Joint Velocity ")

    plt.grid()

    plt.show()

    # 图4：控制力矩
    plt.figure()

    plt.plot(
        time_history,
        torque_history
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N-m)")

    plt.title("Control Torque")

    plt.grid()

    plt.show()

# 绘图
plot_results(
    time_history,
    q_history,
    q_desired_history,
    qdot_history,
    torque_history,
    error_history
)

# 控制性能分析
# 稳态误差
steady_state_error = error_history[-1]
print(
    "Steady-state error:",
    np.rad2deg(steady_state_error),
    "deg"
)

# 最大超调
q_max = np.max(q_history)

overshoot = (
    (q_max - q_desired)
    /abs(q_desired)
    * 100.0
) 

overshoot = max(0.0, overshoot)

print(
    f"Overshoot: {overshoot:.2f} %"
)

# 上升时间
lower_bound = 0.1 * q_desired
upper_bound = 0.9 * q_desired

idx_10 = np.where(
    q_history >= lower_bound
)[0][0]

idx_90 = np.where(
    q_history >= upper_bound
)[0][0]

rise_time = time_history[idx_90] - time_history[idx_10]

print(
    f"Rise time: {rise_time:.2f} s"
)

# 调节时间
tolerance = 0.02 * abs(q_desired)

outside = np.where(
    np.abs(error_history) > tolerance
)[0]

if len(outside) > 0:
    last_outside_idx = outside[-1]

    if last_outside_idx < len(time_history) - 1:
        settling_time = time_history[last_outside_idx + 1]
    else:
        settling_time = -np.nan
else:
    settling_time = 0.0

print(
    f"Settling time: {settling_time:.2f} s"
)

# 实验报告
print("\n==== PD Control Performance ====")

print(f"Kp = {Kp}")
print(f"Kd = {Kd}")

print(
    f"Final angle = "
    f"{np.rad2deg(q_history[-1]):.3f} deg "
)

print(
    f"Steady-state error = "
    f"{np.rad2deg(steady_state_error):.3f} deg"
)

print(
    f"Overshoot = {overshoot:.2f} %"
)

print(
    f"Rise time = {rise_time:.2f} s"
)

print(
    f"Settling time = {settling_time:.2f} s"
)