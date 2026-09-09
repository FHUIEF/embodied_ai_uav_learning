"""
先只做位置到达，给出p_target，求解q_desired，然后让机器人运动过去

阶段 1: Numerical IK

    目标位置 p_target
        ↓
    Jacobian
        ↓
    求 q_desired

阶段 2: MuJoCo dynamics
    
    q_desired
        ↓
    Franka position actuator
        ↓
    机器人真正运动过去

"""

import os 
import time

import numpy as np
import mujoco
import mujoco.viewer

# ============================================================
# 1. 加载模型
# ============================================================

current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,
    "mujoco_menagerie",
    "franka_emika_panda",
    "scene.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)

# ============================================================
# 2. Franka arm joint names
# ============================================================

arm_joint_names = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7"
]

# ============================================================
# 3. 获取 joint ID / qpos address / dof address
# ============================================================

arm_joint_ids = []

arm_qpos_adr = []

arm_dof_adr = []

for joint_name in arm_joint_names:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    arm_joint_ids.append(joint_id)

    # 通过 joint_id 获取 关节在 data.qpos 中地址
    arm_qpos_adr.append(model.jnt_qposadr[joint_id])
    
    # 通过 joint_id 获取 关节在 data.qvel、data.qacc、data.qfrc_* 中地址，读取速度，施加力/扭矩等
    arm_dof_adr.append(model.jnt_dofadr[joint_id])

arm_qpos_adr = np.array(arm_qpos_adr)

arm_dof_adr = np.array(arm_dof_adr)


# ============================================================
# 4. 获取 arm actuator IDs
# ============================================================

arm_actuator_ids = []

for i in range(1, 8):

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"actuator{i}"
    )

    arm_actuator_ids.append(actuator_id)

arm_actuator_ids = np.array(arm_actuator_ids)

# ============================================================
# 5. Gripper actuator
# ============================================================

gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "actuator8"
)

# ============================================================
# 6. Hand Body
# ============================================================

hand_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

# ============================================================
# 7. Home Keyframe
# ============================================================

home_key_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

mujoco.mj_resetDataKeyframe(model, data, home_key_id)

mujoco.mj_forward(model, data)

# ============================================================
# 8. 初始化到 hand position
# ============================================================

initial_hand_position = data.xpos[hand_body_id].copy()

print("\n===== Initial Hand Position =====")

print(np.round(initial_hand_position, 4))

# ============================================================
# 9. 设置 Cartesian target
# ============================================================

# 这里避免使用“相对当前位置的小偏移”
# 避免第一次实验设置一个不可达目标

target_position = (
    initial_hand_position
    + 
    np.array([0.05, 0.05, 0.05])
)

print("\n===== Target Hand Position =====")

print(np.round(target_position, 4))

# ===========================================================
# 10. Numerical IK 参数
# ============================================================

max_iterations = 500

position_tolerance = 1e-4

step_size = 0.3

# ============================================================
# 11. 读取初始 arm q
# ===========================================================

q_ik = np.array([data.qpos[adr] for adr in arm_qpos_adr])

# ============================================================
# 12. Numerical IK 
# ============================================================

print("\n===== Numerical IK =====")

ik_success = False

for iteration in range(max_iterations):

    # -----------------------------------------------------------
    # 将当前 IK 估计写入 qpos
    # -----------------------------------------------------------
    
    for i in range(7):

        data.qpos[arm_qpos_adr[i]] = q_ik[i]

    # Ik 阶段只计算动力学
    data.qvel[:] = 0

    # -----------------------------------------------------------
    # Forward kinematics
    # -----------------------------------------------------------

    mujoco.mj_forward(model, data)

    # -----------------------------------------------------------
    # 当前 hand position
    # -----------------------------------------------------------

    current_position = data.xpos[hand_body_id].copy()

    # -----------------------------------------------------------
    # Cartesian position error
    # -----------------------------------------------------------

    position_error_vector = target_position - current_position

    position_error = np.linalg.norm(position_error_vector)

    # -----------------------------------------------------------
    # 判断是否收敛
    # -----------------------------------------------------------

    if position_error < position_tolerance:

        print(
            f"Ik converged at"
            f"Iteration {iteration}"
        )

        ik_success = True

        break

    # -----------------------------------------------------------
    # 计算完整 body Jacobian
    # -----------------------------------------------------------

    jacp = np.zeros((3, model.nv))

    jacr = np.zeros((3, model.nv)) 

    mujoco.mj_jacBody(model, data, jacp, jacr, hand_body_id)

    # -----------------------------------------------------------
    # 只取 Franka arm 的 7个 DOF
    # -----------------------------------------------------------

    J = jacp[:, arm_dof_adr]

    # -----------------------------------------------------------
    # Jacobian pseudo-inverse
    # -----------------------------------------------------------

    J_pinv = np.linalg.pinv(J)

    # -----------------------------------------------------------
    # Cartesian error -> joint correction
    # -----------------------------------------------------------

    delta_q = J_pinv @ position_error_vector

    # -----------------------------------------------------------
    # 更新 q
    # -----------------------------------------------------------

    q_ik = q_ik + step_size * delta_q

    # -----------------------------------------------------------
    # 简单 joint limit
    # -----------------------------------------------------------

    for i in range(7):

        joint_id = arm_joint_ids[i]

        if model.jnt_limited[joint_id]:

            lower = model.jnt_range[joint_id][0]

            upper = model.jnt_range[joint_id][1]

            q_ik[i] = np.clip(q_ik[i], lower, upper)

    # ----------------------------------------------------------
    # Debug
    # -----------------------------------------------------------

    if iteration % 20 == 0:

        print(
            f"iteration={iteration:3d} |  "
            f"error="
            f"{position_error:.6f}  m "
        )

# ============================================================
# 13. IK 结果
# ============================================================

q_desired = q_ik.copy()


print("\n===== IK Result =====")

print("IK success =", ik_success)

print("q_desired (deg):")
print(np.round(np.rad2deg(q_desired), 3))


# ============================================================
# 14. FK 验证 IK
# ============================================================

for i in range(7):

    data.qpos[arm_qpos_adr[i]] = q_desired[i]


mujoco.mj_forward(model, data)


ik_hand_position = data.xpos[hand_body_id].copy()


ik_position_error = np.linalg.norm(target_position - ik_hand_position)


print("\n===== IK FK Verification =====")

print("Hand position:")

print(np.round(ik_hand_position, 5))

print("Target:")

print(np.round(target_position, 5))

print(
    "IK position error =",
    ik_position_error,
    "m"
)


# ============================================================
# 15. 重置回 Home
#
# 非常重要：
# 上面的 IK 是“数学计算”
# 现在重新从真实 home 状态开始动力学控制
# ============================================================

mujoco.mj_resetDataKeyframe(model, data, home_key_id)

mujoco.mj_forward(model, data)


# ============================================================
# 16. Gripper 保持打开
# ============================================================

data.ctrl[gripper_actuator_id] = 255.0


# ============================================================
# 17. Reach 成功判定
# ============================================================

reach_threshold = 0.01

required_stable_steps = 200

stable_counter = 0

success = False


# ============================================================
# 18. Simulation
# ============================================================

simulation_time = 10.0

step_count = 0


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and data.time < simulation_time:

        # ----------------------------------------------------
        # 发送 7DOF joint position target
        # ----------------------------------------------------

        for i in range(7):

            data.ctrl[arm_actuator_ids[i]] = q_desired[i]


        # ----------------------------------------------------
        # MuJoCo dynamics
        # ----------------------------------------------------

        mujoco.mj_step(model, data)


        # ----------------------------------------------------
        # 当前 hand position
        # ----------------------------------------------------

        current_hand_position = data.xpos[hand_body_id].copy()


        # ----------------------------------------------------
        # Cartesian reach error
        # ----------------------------------------------------

        current_error = np.linalg.norm(target_position - current_hand_position)


        # ----------------------------------------------------
        # Stable reach
        # ----------------------------------------------------

        if (current_error < reach_threshold):

            stable_counter += 1

        else:

            stable_counter = 0


        if (stable_counter >= required_stable_steps):

            print("\n>>> Cartesian target reached!")

            success = True

            break


        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        step_count += 1

        if step_count % 250 == 0:

            current_q = np.array([data.qpos[adr] for adr in arm_qpos_adr])


            print(
                f"\nTime = "
                f"{data.time:.2f}s"
            )

            print("Hand position:")

            print(np.round(current_hand_position, 4))

            print("Position error:")

            print(round(current_error, 5), "m")

            print("Current q (deg):")

            print(np.round(np.rad2deg(current_q), 2))


        viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# 19. Final result
# ============================================================

final_hand_position = data.xpos[hand_body_id].copy()


final_error = np.linalg.norm(target_position - final_hand_position)


print("\n===== Final Result =====")

print("Success =", success)

print("Target position:")

print(np.round(target_position, 4))

print("Final hand position:")

print(np.round(final_hand_position, 4))


print(
    "Final position error =",
    final_error,
    "m"
)