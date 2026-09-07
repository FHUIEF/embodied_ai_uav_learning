import os
import time

import mujoco
import mujoco.viewer
import numpy as np

# 1.加载 MuJoCo 模型
current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,
    "assets",
    "two_link_arm.xml"
)

print("xml_path =", xml_path)


model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2.机械臂参数
l1 = 1.0  
l2 = 1.0   

# 3. Forward Kinematics
def forward_kinematics(q1, q2):
    """
    计算机械臂末端位置
    q: 关节角度 [q1, q2]
    """
    x = (
        l1 * np.cos(q1) 
        + 
        l2 * np.cos(q1 + q2)
    )
    
    y = (
        l1 * np.sin(q1) 
        + 
        l2 * np.sin(q1 + q2)
    )

    return np.array([x, y])

# 4. Inverse Kinematics
def inverse_kinematics(x, y):
    """
    计算机械臂关节角度
    p: 末端位置 [x, y]
    """
    # 计算关节角度 q2
    cos_q2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # 检查目标是否可达
    if abs(cos_q2) > 1.0:
        raise ValueError(
            f"Target ({x:.2f}, {y:.2f}) is outside workspace."
        )
    
    # 一个 IK 解
    q2 = np.arccos(cos_q2)

    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))

    return np.array([q1, q2])

# 5.设置末端位置
target = np.array([1.2, 0.8])

print("Target position:", target)

# 6. IK: 末端位置 -> 目标关节角

q_desired = inverse_kinematics(target[0], target[1])

print("\nIK result")


print(
    f"q_desired ="
    f" {np.rad2deg(q_desired[0]):.2f} deg"
)

print(
    f"q_desired ="
    f" {np.rad2deg(q_desired[1]):.2f} deg"
)

# 7.用 FK 验证 IK 
fk_check = forward_kinematics(q_desired[0], q_desired[1])

print("\nFK check", fk_check)

print("IK/FK error", np.linalg.norm(fk_check - target))


# 8.设置初始状态
data.qpos[:] = np.deg2rad([0.0, 0.0])   # 初始关节角

mujoco.mj_forward(model, data)  

# 9. PD 参数
# 使用上一节已经调好的参数
Kp = np.array([20.0, 20.0])   # 比例增益
Kd = np.array([10.0, 10.0])     # 微分增

qdot_desired = np.zeros(2)   

ctrl_min = -20.0
ctrl_max = 20.0

# 10.仿真时间
simulation_time = 15.0   # 仿真 15 秒

# 11.数据记录
time_history = []
joint_error_history = []
task_error_history = []

# 开始仿真
with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running() and data.time < simulation_time:

        # 当前关节状态
        q = data.qpos[:2].copy()
        q_dot = data.qvel[:2].copy()

        # Joint-space error
        position_error = q_desired - q
        velocity_error = qdot_desired - q_dot

        # PD 控制律
        torque = (
            Kp * position_error
            +
            Kd * velocity_error
        )

        # 限幅
        torque = np.clip(torque, ctrl_min, ctrl_max)

        # 输出给两个 actuator
        data.ctrl[:] = torque

        # 仿真推进一步
        mujoco.mj_step(model, data)

        # 当前实际末端位置
        ee_position = forward_kinematics(q[0], q[1])

        # task-space error
        task_error_vector = target - ee_position

        task_error = np.linalg.norm(task_error_vector)

        # 记录数据
        time_history.append(data.time)

        joint_error_history.append(q_desired - data.qpos[:2])

        task_error_history.append(task_error)

        # 打印
        step_count += 1

        if step_count % 100 == 0:
            print(
                f"Time: {data.time:.2f} s, "
                f"q1 = {np.rad2deg(q[0]):.2f} deg | "
                f"q2 = {np.rad2deg(q[1]):.2f} deg |"
                f"ee = {ee_position[0]:.3f}, "
                f"{ee_position[1]:.3f} | "
                f"Task error: {task_error:.4f}"
            )

        viewer.sync()

        time.sleep(model.opt.timestep)

# 13.最终结果
final_q = data.qpos[:2].copy()

final_ee = forward_kinematics(final_q[0], final_q[1])

final_task_error = np.linalg.norm(target - final_ee)


print("\n===== Final Result =====")

print(
    f"q1 = {np.rad2deg(final_q[0]):.2f} deg, "
    f"q2 = {np.rad2deg(final_q[1]):.2f} deg"
)

print(
    f"End-effector position = "
    f"{final_ee[0]:.4f}"
    f"{final_ee[1]:.4f}"
)

print(
    f"Target = "
    f"{target[0]:.4f}, "
    f"{target[1]:.4f}"
)

print(
    f"Final task error = "
    f"{final_task_error:.6f}"
)