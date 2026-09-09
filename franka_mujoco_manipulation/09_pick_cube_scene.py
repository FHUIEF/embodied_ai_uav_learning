import os
import time

import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# 1. 加载 Manipulation Scene
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(current_dir, "assets", "franka_pick_cube", "scene.xml")


model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)


# ============================================================
# 2. 打印模型维度
# ============================================================

print("===== Model Information =====")

print("nq =", model.nq)

print("nv =", model.nv)

print("nu =", model.nu)

print("njnt =", model.njnt)

print("nbody =", model.nbody)


# ============================================================
# 3. Franka Arm Joints
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


arm_qpos_adr = []

arm_dof_adr = []


for joint_name in arm_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    arm_qpos_adr.append(model.jnt_qposadr[joint_id])

    arm_dof_adr.append(model.jnt_dofadr[joint_id])


arm_qpos_adr = np.array(arm_qpos_adr)

arm_dof_adr = np.array(arm_dof_adr)


# ============================================================
# 4. Finger Joints
# ============================================================

finger_joint_names = [
    "finger_joint1",
    "finger_joint2",
]


finger_qpos_adr = []


for joint_name in finger_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    finger_qpos_adr.append(model.jnt_qposadr[joint_id])


finger_qpos_adr = np.array(finger_qpos_adr)


# ============================================================
# 5. Arm Actuators
# ============================================================

arm_actuator_ids = []


for i in range(1, 8):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")

    arm_actuator_ids.append(actuator_id)


arm_actuator_ids = np.array(arm_actuator_ids)


# ============================================================
# 6. Gripper Actuator
# ============================================================

gripper_actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
)


# ============================================================
# 7. Cube IDs
# ============================================================

cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")


cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")


cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")


# ============================================================
# 8. Table ID
# ============================================================

table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")


# ============================================================
# 9. Cube qpos / qvel addresses
# ============================================================

cube_qpos_adr = model.jnt_qposadr[cube_joint_id]


cube_dof_adr = model.jnt_dofadr[cube_joint_id]


print("\n===== Cube State Addresses =====")

print("Cube qpos address =", cube_qpos_adr)

print("Cube dof address =", cube_dof_adr)


# ============================================================
# 10. Franka Home Configuration
# ============================================================

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, -0.7853])


finger_home = np.array([0.04, 0.04])


# ============================================================
# 11. 手动初始化 Franka
# ============================================================

for i in range(7):
    data.qpos[arm_qpos_adr[i]] = q_home[i]


for i in range(2):
    data.qpos[finger_qpos_adr[i]] = finger_home[i]


data.qvel[:] = 0.0


# ============================================================
# 12. 设置 actuator 初始目标
# ============================================================

for i in range(7):
    data.ctrl[arm_actuator_ids[i]] = q_home[i]


# gripper fully open
data.ctrl[gripper_actuator_id] = 255.0


# ============================================================
# 13. Forward
# ============================================================

mujoco.mj_forward(model, data)


# ============================================================
# 14. 读取 Cube 初始状态
# ============================================================

cube_qpos = data.qpos[cube_qpos_adr : cube_qpos_adr + 7].copy()


cube_qvel = data.qvel[cube_dof_adr : cube_dof_adr + 6].copy()


print("\n===== Initial Cube State =====")

print("Cube freejoint qpos:")

print(np.round(cube_qpos, 5))


print("\nCube freejoint qvel:")

print(np.round(cube_qvel, 5))


print("\nCube body position:")

print(np.round(data.xpos[cube_body_id], 5))


# ============================================================
# 15. Contact Function
# ============================================================


def check_contact(geom_a, geom_b):
    for i in range(data.ncon):
        contact = data.contact[i]

        g1 = contact.geom1
        g2 = contact.geom2

        if (g1 == geom_a and g2 == geom_b) or (g1 == geom_b and g2 == geom_a):
            return True

    return False


# ============================================================
# 16. 打印 Cube 所有接触对象
# ============================================================


def get_cube_contacts():
    contact_names = []

    for i in range(data.ncon):
        contact = data.contact[i]

        g1 = contact.geom1
        g2 = contact.geom2

        if g1 == cube_geom_id:
            other_geom_id = g2

        elif g2 == cube_geom_id:
            other_geom_id = g1

        else:
            continue

        other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id)

        contact_names.append(other_name)

    return contact_names


# ============================================================
# 17. Simulation
# ============================================================

simulation_time = 5.0

step_count = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < simulation_time:
        # ====================================================
        # A. Franka 保持 Home
        # ====================================================

        for i in range(7):
            data.ctrl[arm_actuator_ids[i]] = q_home[i]

        # ====================================================
        # B. Gripper 保持打开
        # ====================================================

        data.ctrl[gripper_actuator_id] = 255.0

        # ====================================================
        # C. Physics
        # ====================================================

        mujoco.mj_step(model, data)

        # ====================================================
        # D. Debug
        # ====================================================

        step_count += 1

        if step_count % 250 == 0:
            cube_position = data.xpos[cube_body_id].copy()

            cube_velocity = data.qvel[cube_dof_adr : cube_dof_adr + 3].copy()

            cube_table_contact = check_contact(cube_geom_id, table_geom_id)

            cube_contacts = get_cube_contacts()

            print(f"\nTime = {data.time:.2f}s")

            print("Cube position:")

            print(np.round(cube_position, 5))

            print("Cube linear velocity:")

            print(np.round(cube_velocity, 5))

            print("Cube-Table contact =", cube_table_contact)

            print("Cube contacts =", cube_contacts)

            print("Total contacts =", data.ncon)

        viewer.sync()

        time.sleep(model.opt.timestep)
