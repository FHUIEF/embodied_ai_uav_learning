"""计算末端hand的位姿（位置+姿态）"""
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
# 2. 打印所有 site
# ============================================================

print("\n==== Site Names ====")

# model.nsite 附着在机器人某个body上的参考点/参考坐标系
# 没质量、不参与动力学、不负责碰撞，主要用于标记位置和姿态，或者作为传感器的安装位置
print(model.nsite)  # 0: 没有site

for site_id in range(model.nsite):

    site_name = mujoco.mj_id2name(
        model, 
        mujoco.mjtObj.mjOBJ_SITE, 
        site_id
    )

    print(
        f"site_id={site_id},"
        f"site_name={site_name}"
    )

# ============================================================
# 3. 找 hand body
# ============================================================

hand_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

# ============================================================
# 4. 初始化到 home的姿态
# ============================================================

home_key_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

mujoco.mj_resetDataKeyframe(model, data, home_key_id)

mujoco.mj_forward(model, data)

# ============================================================
# 5. 读取 hand body pose
# ============================================================

hand_pos = data.xpos[hand_body_id].copy()

hand_rot = data.xmat[hand_body_id].reshape(3, 3).copy()

hand_quat = data.xquat[hand_body_id].copy()

print("\n===== Hand Body Pose =====")

print("position:")
print(np.round(hand_pos, 4))

print("\nRotation matrix:")
print(np.round(hand_rot, 4))

print("\nQuaternion:")
print(np.round(hand_quat, 4))

# ===========================================================
# 6. Viewer
# ===========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)
