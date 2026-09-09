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

print("\n==== Franka Panda State Reader ====")


# ============================================================
# 2. 机械臂 joint names
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

finger_joint_names = [
    "finger_joint1",
    "finger_joint2"
]

# ============================================================
# 3. 根据 joint name 获取 joint id
# ============================================================

arm_joint_ids = []

for joint_name in arm_joint_names:

    joint_id = mujoco.mj_name2id(
        model, 
        mujoco.mjtObj.mjOBJ_JOINT, 
        joint_name
    )

    arm_joint_ids.append(joint_id)

finger_joint_ids = []

for joint_name in finger_joint_names:

    joint_id = mujoco.mj_name2id(
        model, 
        mujoco.mjtObj.mjOBJ_JOINT, 
        joint_name
    )

    finger_joint_ids.append(joint_id)

print("\nArm Joint IDs:", arm_joint_ids)

print("\nFinger Joint IDs:", finger_joint_ids)

# ============================================================
# 4. 获取 qpos / qvel address
# ============================================================

arm_qpos_adr = []

arm_dof_adr = []

# 获取机械臂关节的 qpos / qvel address
for joint_id in arm_joint_ids:

    # 关节的位置数据在 data.qpos 中的地址
    qpos_adr = model.jnt_qposadr[joint_id]

    # 关节的速度数据在 data.qvel 中的地址
    dof_adr = model.jnt_dofadr  [joint_id]

    arm_qpos_adr.append(qpos_adr)

    arm_dof_adr.append(dof_adr)


finger_qpos_adr = []

finger_dof_adr = []


# 获取夹爪关节的 qpos / qvel address
for finger_joint_id in finger_joint_ids:

    qpos_adr = model.jnt_qposadr[finger_joint_id]

    dof_adr = model.jnt_dofadr[finger_joint_id]

    finger_qpos_adr.append(qpos_adr)

    finger_dof_adr.append(dof_adr)


print("\nArm qpos addresses:", arm_qpos_adr)

print("Arm dof addresses:", arm_dof_adr)

print("Finger qpos addresses:", finger_qpos_adr)


# ============================================================
# 5. 找到末端 body
# ============================================================

hand_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)


print("\nHand body ID:", hand_body_id)


# ============================================================
# 6. 找到 actuator
# ============================================================

arm_actuator_ids = []

# 获取机械臂的 actuator id
for i in range(1, 8):

    actuator_name = f"actuator{i}"

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name
    )

    arm_actuator_ids.append(actuator_id)


# 获取夹爪的 actuator id
gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "actuator8"
)


print("Arm actuator IDs:", arm_actuator_ids)

print("Gripper actuator ID:", gripper_actuator_id)


# ============================================================
# 7. Forward kinematics update
# ============================================================

mujoco.mj_forward(model, data)


# ============================================================
# 8. 初始状态
# ============================================================

# 读取机械臂关节角度和速度
print("\n===== Initial Robot State =====")


for i, joint_name in enumerate(arm_joint_names):

    q = data.qpos[arm_qpos_adr[i]]

    qdot = data.qvel[arm_dof_adr[i]]

    print(
        f"{joint_name}: "
        f"q = {q:.6f} rad "
        f"({np.rad2deg(q):.2f} deg), "
        f"qdot = {qdot:.6f} rad/s"
    )

# 读取夹爪关节状态
print("\n===== Finger State =====")


for i, joint_name in enumerate(finger_joint_names):

    q = data.qpos[finger_qpos_adr[i]]

    qdot = data.qvel[finger_dof_adr[i]]

    print(
        f"{joint_name}: "
        f"q = {q:.6f} m, "
        f"qdot = {qdot:.6f} m/s"
    )


# ============================================================
# 9. 末端位置和姿态
# ============================================================

hand_position = data.xpos[hand_body_id].copy()


hand_rotation_matrix = data.xmat[hand_body_id].reshape(3,3).copy()


hand_quaternion = data.xquat[hand_body_id].copy()


print("\n===== End-Effector State =====")

print("Hand position:")

print(hand_position)

print("\nHand rotation matrix:")

print(hand_rotation_matrix)

print("\nHand quaternion [w, x, y, z]:")

print(hand_quaternion)


# ============================================================
# 10. Viewer
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running():

        # --------------------------------------------
        # 推进仿真
        # --------------------------------------------

        mujoco.mj_step(model, data)

        step_count += 1

        # --------------------------------------------
        # 每隔一段时间打印状态
        # --------------------------------------------

        if step_count % 500 == 0:

            print("\n==============================")

            print(f"Time = {data.time:.3f}s")


            # ----------------------------------------
            # Arm q / qdot
            # ----------------------------------------

            arm_q = np.array([
                data.qpos[adr] for adr in arm_qpos_adr
            ])


            arm_qdot = np.array([
                data.qvel[adr] for adr in arm_dof_adr
            ])


            print("Arm q (deg):")

            print(np.round(np.rad2deg(arm_q), 3))


            print("Arm qdot (rad/s):")

            print(np.round(arm_qdot, 5))

            # ----------------------------------------
            # Fingers
            # ----------------------------------------

            finger_q = np.array([
                data.qpos[adr] for adr in finger_qpos_adr
            ])


            print("Finger q (m):")

            print(np.round(finger_q, 5))


            # ----------------------------------------
            # End-effector
            # ----------------------------------------

            hand_position = data.xpos[hand_body_id].copy()


            hand_quaternion = data.xquat[hand_body_id].copy()


            print("Hand position:")

            print(np.round(hand_position, 4))


            print("Hand quaternion:")

            print(np.round(hand_quaternion, 4))


            # ----------------------------------------
            # Control input
            # ----------------------------------------

            print("ctrl:")

            print(np.round(data.ctrl, 4))


        viewer.sync()

        time.sleep(model.opt.timestep)