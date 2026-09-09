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
# 2. 找 joint1
# ===========================================================

# 根据关节名称获取关节ID
joint1_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_JOINT, 
    "joint1"
)

# 获取 joint1 在 qpos 中的地址
joint1_qpos_adr = model.jnt_qposadr[joint1_id]

print(f"joint1 qpos address = ", joint1_qpos_adr)

# ============================================================
# 3. 找 hand body
# ===========================================================

hand_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

# ============================================================
# 4. 定义状态打印函数
# ===========================================================

def print_robot_state(title):

    hand_position = data.xpos[hand_body_id].copy()

    hand_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()

    hand_quaternion = data.xquat[hand_body_id].copy()

    print(f"\n===== {title} =====")

    print("joint1 = ", np.rad2deg(data.qpos[joint1_qpos_adr]), " deg")

    print("\nHand position:")
    print(np.round(hand_position, 4))

    print("\nHand rotation matrix:")
    print(np.round(hand_rotation, 4))

    print("\nHand quaternion [w, x, y, z]:")
    print(np.round(hand_quaternion, 4))

# ===========================================================
# 5. 初始状态
# ===========================================================

mujoco.mj_forward(model, data)

print_robot_state("Initial State")

# ============================================================
# 6. 修改 joint1
# ============================================================

data.qpos[joint1_qpos_adr] = np.deg2rad(90.0)


# ============================================================
# 7. 重新计算 FK
# ============================================================

mujoco.mj_forward(model, data)

print_robot_state("After joint1 = 30 deg")


# ============================================================
# 8. Viewer
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:


    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)


