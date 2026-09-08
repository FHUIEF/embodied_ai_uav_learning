"""
3DOF 末端位姿控制
(xd, yd, φd) -> Ik -> (q1, q2, q3) -> PDCtrl -> (τ1, τ2, τ3) -> Mujoco
"""

import os
import time

import numpy as np
import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt


# ============================================================
# 1. 加载 MuJoCo 模型
# ============================================================

current_dir = os.path.dirname(__file__)

xml_path = os.path.join(current_dir, "assets", "three_link_arm.xml")

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)


print("===== Model Information =====")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)


# ============================================================
# 2. 机械臂参数
# ============================================================

l1 = 1.0
l2 = 0.8
l3 = 0.3


# ============================================================
# 3. 正运动学
# ============================================================


def forward_kinematics_3dof(q1, q2, q3):

    theta1 = q1
    theta2 = q1 + q2
    theta3 = q1 + q2 + q3

    x = l1 * np.cos(theta1) + l2 * np.cos(theta2) + l3 * np.cos(theta3)

    y = l1 * np.sin(theta1) + l2 * np.sin(theta2) + l3 * np.sin(theta3)

    phi = theta3

    return np.array([x, y, phi])


# ============================================================
# 4. 逆运动学
# ============================================================


def inverse_kinematics_3dof(x_d, y_d, phi_d, elbow="down"):

    # wrist center
    x_w = x_d - l3 * np.cos(phi_d)

    y_w = y_d - l3 * np.sin(phi_d)

    # q2
    cos_q2 = (x_w**2 + y_w**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if abs(cos_q2) > 1.0:
        raise ValueError("Target pose is outside workspace.")

    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    if elbow == "down":
        q2 = np.arccos(cos_q2)

    elif elbow == "up":
        q2 = -np.arccos(cos_q2)

    else:
        raise ValueError("elbow must be 'down' or 'up'")

    # q1
    q1 = np.arctan2(y_w, x_w) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))

    # q3
    q3 = phi_d - q1 - q2

    return np.array([q1, q2, q3])


# ============================================================
# 5. 角度归一化
# ============================================================


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ============================================================
# 6. PD 参数
# ============================================================

Kp = np.array([20.0, 20.0, 15.0])

Kd = np.array([10.0, 10.0, 6.0])


TORQUE_LIMIT = np.array([20.0, 20.0, 10.0])


# ============================================================
# 7. PD 控制器
# ============================================================


def joint_pd_control(q, qdot, q_desired):

    position_error = q_desired - q

    velocity_error = -qdot

    torque = Kp * position_error + Kd * velocity_error

    torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

    return torque


# ============================================================
# 8. 设置目标末端位姿
# ============================================================

x_target = 1.2
y_target = 0.8

phi_target = np.deg2rad(0.0)


target_pose = np.array([x_target, y_target, phi_target])


print("\n===== Target Pose =====")

print(f"x = {x_target:.4f} m")

print(f"y = {y_target:.4f} m")

print(f"phi = {np.rad2deg(phi_target):.2f} deg")


# ============================================================
# 9. IK 求目标关节角
# ============================================================

q_desired = inverse_kinematics_3dof(x_target, y_target, phi_target, elbow="down")


print("\n===== IK Solution =====")

print("q_desired =", np.round(np.rad2deg(q_desired), 4), "deg")


# ============================================================
# 10. 用 FK 验证 IK
# ============================================================

pose_check = forward_kinematics_3dof(q_desired[0], q_desired[1], q_desired[2])


print("\n===== FK Check =====")

print(f"x = {pose_check[0]:.4f} m")

print(f"y = {pose_check[1]:.4f} m")

print(f"phi = {np.rad2deg(pose_check[2]):.4f} deg")


# ============================================================
# 11. 初始状态
# ============================================================

data.qpos[:] = np.deg2rad([0.0, 0.0, 0.0])

data.qvel[:] = 0.0

mujoco.mj_forward(model, data)


# ============================================================
# 12. 数据记录
# ============================================================

time_history = []

q_history = []

pose_history = []

position_error_history = []

orientation_error_history = []

torque_history = []


# ============================================================
# 13. 成功判定参数
# ============================================================

position_threshold = 0.01

orientation_threshold = np.deg2rad(1.0)

stable_counter = 0

required_stable_steps = 200

simulation_time = 12.0


# ============================================================
# 14. 开始仿真
# ============================================================

success = False


with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running() and data.time < simulation_time:

        # ----------------------------------------------------
        # 当前关节状态
        # ----------------------------------------------------
        q = data.qpos[:3].copy()
        qdot = data.qvel[:3].copy()

        # ----------------------------------------------------
        # PD 控制
        # ----------------------------------------------------

        torque = joint_pd_control(q, qdot, q_desired)

        data.ctrl[:] = torque

        # ----------------------------------------------------
        # 当前末端位姿
        # ----------------------------------------------------
        current_pose = forward_kinematics_3dof(q[0], q[1], q[2])

        # ----------------------------------------------------
        # 位置误差
        # ----------------------------------------------------

        position_error_vector = target_pose[:2] - current_pose[:2]

        position_error = np.linalg.norm(position_error_vector)

        # ----------------------------------------------------
        # 姿态误差
        # ----------------------------------------------------

        orientation_error = wrap_to_pi(phi_target - current_pose[2])

        # ----------------------------------------------------
        # 到达判定
        # ----------------------------------------------------

        if position_error < position_threshold and abs(orientation_error) < orientation_threshold:

            stable_counter += 1

        else:

            stable_counter = 0

        if stable_counter >= required_stable_steps:

            print("\n>>> Pose target reached!")

            success = True

            break

        # ----------------------------------------------------
        # 数据记录
        # ----------------------------------------------------

        time_history.append(data.time)

        q_history.append(q.copy())

        pose_history.append(current_pose.copy())

        position_error_history.append(position_error)

        orientation_error_history.append(orientation_error)

        torque_history.append(torque.copy())

        # ----------------------------------------------------
        # 推进 MuJoCo
        # ----------------------------------------------------

        mujoco.mj_step(model, data)

        # ----------------------------------------------------
        # 调试输出
        # ----------------------------------------------------

        step_count += 1

        if step_count % 100 == 0:

            print(
                f"Time={data.time:.2f}s | "
                f"Pose=("
                f"{current_pose[0]:.3f}, "
                f"{current_pose[1]:.3f}, "
                f"{np.rad2deg(current_pose[2]):.2f}deg"
                f") | "
                f"PosErr="
                f"{position_error:.4f} | "
                f"OriErr="
                f"{np.rad2deg(orientation_error):.3f}deg"
            )

        viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# 15. 最终结果
# ============================================================

mujoco.mj_forward(model, data)

final_q = data.qpos[:3].copy()

final_pose = forward_kinematics_3dof(final_q[0], final_q[1], final_q[2])


final_position_error = np.linalg.norm(target_pose[:2] - final_pose[:2])


final_orientation_error = wrap_to_pi(phi_target - final_pose[2])


print("\n===== Final Result =====")

print("Success =", success)

print("Desired q =", np.round(np.rad2deg(q_desired), 4), "deg")

print("Final q =", np.round(np.rad2deg(final_q), 4), "deg")

print(f"Target pose = " f"({x_target:.4f}, " f"{y_target:.4f}, " f"{np.rad2deg(phi_target):.2f} deg)")

print(f"Final pose = " f"({final_pose[0]:.4f}, " f"{final_pose[1]:.4f}, " f"{np.rad2deg(final_pose[2]):.2f} deg)")

print(f"Position error = " f"{final_position_error:.6f} m")

print(f"Orientation error = " f"{np.rad2deg(final_orientation_error):.6f} deg")


# ============================================================
# 16. 转 numpy
# ============================================================

time_history = np.array(time_history)

q_history = np.array(q_history)

pose_history = np.array(pose_history)

position_error_history = np.array(position_error_history)

orientation_error_history = np.array(orientation_error_history)

torque_history = np.array(torque_history)


# ============================================================
# 17. Plot：末端位置
# ============================================================

if len(time_history) > 0:

    plt.figure()

    plt.plot(time_history, pose_history[:, 0], label="x")

    plt.plot(time_history, pose_history[:, 1], label="y")

    plt.axhline(x_target, linestyle="--", label="x target")

    plt.axhline(y_target, linestyle="--", label="y target")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("End-effector Position")

    plt.grid()
    plt.legend()

    # ========================================================
    # 18. Plot：末端姿态
    # ========================================================

    plt.figure()

    plt.plot(time_history, np.rad2deg(pose_history[:, 2]), label="phi")

    plt.axhline(np.rad2deg(phi_target), linestyle="--", label="phi target")

    plt.xlabel("Time (s)")
    plt.ylabel("Orientation (deg)")
    plt.title("End-effector Orientation")

    plt.grid()
    plt.legend()

    # ========================================================
    # 19. Plot：位置误差
    # ========================================================

    plt.figure()

    plt.plot(time_history, position_error_history)

    plt.axhline(position_threshold, linestyle="--", label="threshold")

    plt.xlabel("Time (s)")
    plt.ylabel("Position error (m)")
    plt.title("Task-space Position Error")

    plt.grid()
    plt.legend()

    # ========================================================
    # 20. Plot：姿态误差
    # ========================================================

    plt.figure()

    plt.plot(time_history, np.rad2deg(orientation_error_history))

    plt.axhline(np.rad2deg(orientation_threshold), linestyle="--", label="+threshold")

    plt.axhline(-np.rad2deg(orientation_threshold), linestyle="--", label="-threshold")

    plt.xlabel("Time (s)")
    plt.ylabel("Orientation error (deg)")
    plt.title("Task-space Orientation Error")

    plt.grid()
    plt.legend()

    plt.show()
