"""机器人模型有什么"""
import os
import time

import mujoco
import mujoco.viewer


# ============================================================
# 1. 获取 xml 路径
# ============================================================

current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,
    "mujoco_menagerie",
    "franka_emika_panda",
    "scene.xml"
)

print("Loading XML: ")
print(xml_path)

# ============================================================
# 2. 加载 Mujoco Model
# ============================================================

model = mujoco.MjModel.from_xml_path(xml_path)  # 机器人模型的固定结构和参数

data = mujoco.MjData(model) # 运行状态

# ============================================================
# 3. 打印基本模型信息
# ============================================================
print("\n==== Model Information ====")

print("nq = ", model.nq)    # 关节自由度数量
print("nv = ", model.nv)    # 速度自由度数量
print("nu = ", model.nu)    # 控制输入数量

print("njnt = ", model.njnt)
print("nbody = ", model.nbody)
print("ngeom = ", model.ngeom)

# ============================================================
# 4. 打印所有 Joint 名字
# ============================================================
print("\n==== Joint Names ====")

for joint_id in range(model.njnt):
    
    joint_name = mujoco.mj_id2name(
                    model, 
                    mujoco.mjtObj.mjOBJ_JOINT, 
                    joint_id
                )
    
    print(f"joint_id: {joint_id}, joint_name: {joint_name}")


# ============================================================
# 5. 打印所有 Actuator 名字
# ============================================================
print("\n==== Actuator Names ====")

for actuator_id in range(model.nu):
    
    actuator_name = mujoco.mj_id2name(
                        model, 
                        mujoco.mjtObj.mjOBJ_ACTUATOR, 
                        actuator_id
                    )
    
    print(f"actuator_id: {actuator_id}, actuator_name: {actuator_name}")

# ============================================================
# 6. 打印 Body 名字
# ============================================================
print("\n==== Body Names ====")

for body_id in range(model.nbody):
    
    body_name = mujoco.mj_id2name(
                    model, 
                    mujoco.mjtObj.mjOBJ_BODY, 
                    body_id
                )
    
    print(f"body_id: {body_id}, body_name: {body_name}")


# ============================================================
# 7. Forward
# ============================================================
mujoco.mj_forward(model, data)

# ============================================================
# 8. 启动 Viewer
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("\n==== Franka Panda loaded successfully ====")

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)