import os
import time

import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# 1. 加载模型
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(
    current_dir, "mujoco_menagerie", "franka_emika_panda", "scene.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)


# ============================================================
# 2. Arm joint names
# ============================================================

arm_joint_names = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


# ============================================================
# 3. Arm qpos addresses
# ============================================================

arm_qpos_adr = []


for joint_name in arm_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    qpos_adr = model.jnt_qposadr[joint_id]

    arm_qpos_adr.append(qpos_adr)


arm_qpos_adr = np.array(arm_qpos_adr)


# ============================================================
# 4. Arm actuator IDs
# ============================================================

arm_actuator_ids = []


for i in range(1, 8):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")

    arm_actuator_ids.append(actuator_id)


arm_actuator_ids = np.array(arm_actuator_ids)


# ============================================================
# 5. Finger joints
# ============================================================

finger_joint_names = [
    "finger_joint1",
    "finger_joint2",
]


finger_joint_ids = []

finger_qpos_adr = []

finger_dof_adr = []


for joint_name in finger_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    finger_joint_ids.append(joint_id)

    finger_qpos_adr.append(model.jnt_qposadr[joint_id])

    finger_dof_adr.append(model.jnt_dofadr[joint_id])


finger_qpos_adr = np.array(finger_qpos_adr)

finger_dof_adr = np.array(finger_dof_adr)


# ============================================================
# 6. Gripper actuator
# ============================================================

gripper_actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
)


print("Gripper actuator ID =", gripper_actuator_id)

print("Finger qpos addresses =", finger_qpos_adr)


# ============================================================
# 7. Home keyframe
# ============================================================

home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")


mujoco.mj_resetDataKeyframe(model, data, home_key_id)

mujoco.mj_forward(model, data)


# ============================================================
# 8. 保存 home arm configuration
# ============================================================

q_home = np.array([data.qpos[adr] for adr in arm_qpos_adr])


print("\nHome arm q (deg):")

print(np.round(np.rad2deg(q_home), 3))


# ============================================================
# 9. Gripper control values
# ============================================================

GRIPPER_OPEN = 255.0

GRIPPER_CLOSE = 0.0


# ============================================================
# 10. 仿真阶段时间
# ============================================================

OPEN_TIME = 2.0

CLOSE_TIME = 4.0

REOPEN_TIME = 6.0

simulation_time = 8.0


# ============================================================
# 11. Viewer
# ============================================================

step_count = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < simulation_time:
        # ====================================================
        # A. 保持机械臂在 home
        # ====================================================

        for i in range(7):
            data.ctrl[arm_actuator_ids[i]] = q_home[i]

        # ====================================================
        # B. Gripper state machine
        # ====================================================

        if data.time < OPEN_TIME:
            gripper_command = GRIPPER_OPEN

            state = "OPEN"

        elif data.time < CLOSE_TIME:
            gripper_command = GRIPPER_CLOSE

            state = "CLOSE"

        else:
            gripper_command = GRIPPER_OPEN

            state = "REOPEN"

        # ====================================================
        # C. 给 gripper actuator 命令
        # ====================================================

        data.ctrl[gripper_actuator_id] = gripper_command

        # ====================================================
        # D. MuJoCo dynamics
        # ====================================================

        mujoco.mj_step(model, data)

        # ====================================================
        # E. 读取 finger state
        # ====================================================

        finger_q = np.array([data.qpos[adr] for adr in finger_qpos_adr])

        finger_qdot = np.array([data.qvel[adr] for adr in finger_dof_adr])

        # 两根手指之间的总开度
        gripper_width = finger_q[0] + finger_q[1]

        # ====================================================
        # F. Debug
        # ====================================================

        step_count += 1

        if step_count % 250 == 0:
            print(f"\nTime = {data.time:.2f}s")

            print("State =", state)

            print("Gripper ctrl =", round(gripper_command, 2))

            print("Finger q (m) =")

            print(np.round(finger_q, 5))

            print("Finger qdot (m/s) =")

            print(np.round(finger_qdot, 5))

            print("Gripper width =", round(gripper_width, 5), "m")

        viewer.sync()

        time.sleep(model.opt.timestep)
