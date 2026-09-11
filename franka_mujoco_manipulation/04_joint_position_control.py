"""关节空间位置控制"""
"""从home姿态出发，通过7关节的位置执行器，逐渐移动到指定的目标关节姿态，并保持夹爪打开"""
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

#============================================================
# 2. 关节名称
#============================================================

arm_joint_names = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7"
]

# ===========================================================
# 3. 获取关节 qpos / qvel 地址
# ===========================================================

arm_joint_ids = []

arm_qpos_adr = []

arm_qvel_adr = []

for joint_name in arm_joint_names:

    joint_id = mujoco.mj_name2id(
        model, 
        mujoco.mjtObj.mjOBJ_JOINT, 
        joint_name
    )

    arm_joint_ids.append(joint_id)

    # joint对应的关节位置 在 qpos 中的地址
    arm_qpos_adr.append(model.jnt_qposadr[joint_id])

    # joint对应的速度 在 qvel 中的地址
    arm_qvel_adr.append(model.jnt_dofadr[joint_id])

# ===========================================================
# 4. 获取 actuator IDs
# ===========================================================

arm_actuator_ids = []

for i in range(1, 8):

    actuator_name = f"actuator{i}"

    actuator_id = mujoco.mj_name2id(
        model, 
        mujoco.mjtObj.mjOBJ_ACTUATOR, 
        actuator_name
    )

    arm_actuator_ids.append(actuator_id)

# ===========================================================
# 5. 目标关节角
# ===========================================================

q_target_deg = np.array([0.0, -20.0, 0.0, -100.0, 0.0, 80.0, -45.0])

q_target = np.deg2rad(q_target_deg)

print("==== Target Joint Angles  ===")

print(q_target_deg, " deg")

# ===========================================================
# 6. 使用官方 home keyframe 初始化
# ===========================================================

home_key_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_KEY, 
    "home"
)

# 把 data 恢复到 XML 中定义的 home keyframe 状态
mujoco.mj_resetDataKeyframe(model, data, home_key_id)

mujoco.mj_forward(model, data)

# ===========================================================
# 7. 控制前查看初始状态
# ===========================================================

initial_q = np.array([data.qpos[adr] for adr in arm_qpos_adr])

print("==== Initial Arm Joint q ===")

print(np.round(np.rad2deg(initial_q), 3), " deg")

# ===========================================================
# 8. 设置 gripper 保持打开
# ===========================================================

gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "actuator8"
)

data.ctrl[gripper_actuator_id] = 255.0  # 保持夹爪打开

# ===========================================================
# 9. 仿真参数
# ===========================================================

simulation_time = 8.0  # 仿真总时间

step_count = 0

# ===========================================================
# 10. Viewer
# ===========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    while (viewer.is_running() and data.time < simulation_time):

        # --------------------------------------------
        # 推进当前 arm joint state
        # --------------------------------------------

        q = np.array([data.qpos[adr] for adr in arm_qpos_adr])

        qdot = np.array([data.qvel[adr] for adr in arm_qvel_adr])

        # --------------------------------------------
        # 给前 7 个 actuator 设置目标关节角
        # --------------------------------------------

        for i in range(7):

            actuator_id = arm_actuator_ids[i]

            data.ctrl[actuator_id] = q_target[i]

        # --------------------------------------------
        # 推进仿真
        # --------------------------------------------
        mujoco.mj_step(model, data)

        # --------------------------------------------
        # 调试输出
        # --------------------------------------------

        step_count += 1

        if step_count % 250 == 0:

            error = q_target - q

            print(f"\nTime = {data.time:.2f}s")

            print(f"Current q(deg):")
            print(np.round(np.rad2deg(q), 2))

            print(f"Error(deg):")
            print(np.round(np.rad2deg(error), 2))

            print("qdot (rad/s):")
            print(np.round(qdot, 4))

        viewer.sync()

        time.sleep(model.opt.timestep)

