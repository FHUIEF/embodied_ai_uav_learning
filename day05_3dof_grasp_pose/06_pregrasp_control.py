import os
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 1. 加载模型
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
# 3. FK
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
# 4. IK
# ============================================================

def inverse_kinematics_3dof(x_d, y_d, phi_d, elbow="down"):
    # ----------------------------
    # wrist center
    # ----------------------------
    x_w = x_d - l3 * np.cos(phi_d)
    y_w = y_d - l3 * np.sin(phi_d)

    # ----------------------------
    # q2
    # ----------------------------
    cos_q2 = (x_w**2 + y_w**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if abs(cos_q2) > 1.0:
        raise ValueError(f"Target pose ({x_d:.3f}, {y_d:.3f}) is outside workspace.")

    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    if elbow == "down":
        q2 = np.arccos(cos_q2)
    elif elbow == "up":
        q2 = -np.arccos(cos_q2)
    else:
        raise ValueError("elbow must be 'down' or 'up'")

    # ----------------------------
    # q1
    # ----------------------------
    q1 = np.arctan2(y_w, x_w) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))

    # ----------------------------
    # q3
    # ----------------------------
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
# 7. Joint PD
# ============================================================

def joint_pd_control(q, qdot, q_desired):
    position_error = q_desired - q
    torque = Kp * position_error - Kd * qdot
    torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
    return torque


# ============================================================
# 8. 设置物体位置
# ============================================================

object_position = np.array([1.2, 0.8])


# ============================================================
# 9. 设置抓取姿态
# ============================================================

phi_grasp = np.deg2rad(0.0)


# ============================================================
# 10. Pre-grasp 距离
# ============================================================

pregrasp_distance = 0.25


# ============================================================
# 11. Approach direction
# ============================================================

approach_direction = np.array([np.cos(phi_grasp), np.sin(phi_grasp)])


# ============================================================
# 12. 计算 Pre-grasp Pose
# ============================================================

pregrasp_position = object_position - pregrasp_distance * approach_direction
pregrasp_pose = np.array([pregrasp_position[0], pregrasp_position[1], phi_grasp])
grasp_pose = np.array([object_position[0], object_position[1], phi_grasp])

print("\n===== Pre-grasp Pose =====")
print(f"x = {pregrasp_pose[0]:.3f}")
print(f"y = {pregrasp_pose[1]:.3f}")
print(f"phi = {np.rad2deg(pregrasp_pose[2]):.2f} deg")

print("\n===== Grasp Pose =====")
print(f"x = {grasp_pose[0]:.3f}")
print(f"y = {grasp_pose[1]:.3f}")
print(f"phi = {np.rad2deg(grasp_pose[2]):.2f} deg")


# ============================================================
# 13. 状态机
# ============================================================

MOVE_TO_PREGRASP = 0
APPROACH = 1
DONE = 2
state = MOVE_TO_PREGRASP
state_names = {MOVE_TO_PREGRASP: "MOVE_TO_PREGRASP", APPROACH: "APPROACH", DONE: "DONE"}


# ============================================================
# 14. Reach 判定
# ============================================================

position_threshold = 0.01
orientation_threshold = np.deg2rad(1.0)
required_stable_steps = 150
stable_counter = 0


# ============================================================
# 15. 初始状态
# ============================================================

data.qpos[:] = np.deg2rad([0.0, 0.0, 0.0])
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)


# ============================================================
# 16. 仿真
# ============================================================

simulation_time = 20.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0

    while viewer.is_running() and data.time < simulation_time:
        # ----------------------------------------------------
        # 当前状态
        # ----------------------------------------------------
        q = data.qpos[:3].copy()
        qdot = data.qvel[:3].copy()
        current_pose = forward_kinematics_3dof(q[0], q[1], q[2])

        # ====================================================
        # 状态 1：MOVE_TO_PREGRASP
        # ====================================================
        if state == MOVE_TO_PREGRASP:
            target_pose = pregrasp_pose

        # ====================================================
        # 状态 2：APPROACH
        # ====================================================
        elif state == APPROACH:
            target_pose = grasp_pose

        # ====================================================
        # 状态 3：DONE
        # ====================================================
        elif state == DONE:
            print("\n===== PREGRASP TASK FINISHED =====")
            break

        # ====================================================
        # IK
        # ====================================================
        q_desired = inverse_kinematics_3dof(target_pose[0], target_pose[1], target_pose[2], elbow="down")

        # ====================================================
        # PD
        # ====================================================
        torque = joint_pd_control(q, qdot, q_desired)
        data.ctrl[:] = torque

        # ====================================================
        # Task-space error
        # ====================================================
        position_error = np.linalg.norm(target_pose[:2] - current_pose[:2])
        orientation_error = wrap_to_pi(target_pose[2] - current_pose[2])

        # ====================================================
        # Stable reach
        # ====================================================
        if position_error < position_threshold and abs(orientation_error) < orientation_threshold:
            stable_counter += 1
        else:
            stable_counter = 0

        # ====================================================
        # 状态转换
        # ====================================================
        if state == MOVE_TO_PREGRASP and stable_counter >= required_stable_steps:
            print("\n>>> Pre-grasp pose reached.")
            state = APPROACH
            stable_counter = 0

        elif state == APPROACH and stable_counter >= required_stable_steps:
            print("\n>>> Grasp pose reached.")
            state = DONE
            stable_counter = 0

        # ====================================================
        # MuJoCo step
        # ====================================================
        mujoco.mj_step(model, data)

        # ====================================================
        # Debug
        # ====================================================
        step_count += 1
        if step_count % 100 == 0:
            print(
                f"Time={data.time:.2f}s | State={state_names[state]} | "
                f"Pose=({current_pose[0]:.3f}, {current_pose[1]:.3f}, "
                f"{np.rad2deg(current_pose[2]):.2f}deg) | "
                f"Target=({target_pose[0]:.3f}, {target_pose[1]:.3f}, "
                f"{np.rad2deg(target_pose[2]):.2f}deg) | "
                f"PosErr={position_error:.4f} | "
                f"OriErr={np.rad2deg(orientation_error):.3f}deg"
            )

        viewer.sync()
        time.sleep(model.opt.timestep)