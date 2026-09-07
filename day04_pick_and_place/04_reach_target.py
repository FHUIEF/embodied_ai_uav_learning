import os
import time

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# 1. 加载 MuJoCo 模型
# ============================================================

current_dir = os.path.dirname(__file__)

xml_path = os.path.join(
    current_dir,
    "assets",
    "two_link_arm.xml"
)

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
l2 = 1.0


# ============================================================
# 3. Forward Kinematics
# ============================================================

def forward_kinematics(q1, q2):
    """
    输入:
        q1, q2: 关节角，单位 rad

    输出:
        末端二维位置 [x, y]
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


# ============================================================
# 4. Inverse Kinematics
# ============================================================

def inverse_kinematics(x, y):
    """
    输入:
        x, y: 末端目标位置

    输出:
        [q1, q2]
    """

    cos_q2 = (
        x**2
        + y**2
        - l1**2
        - l2**2
    ) / (2 * l1 * l2)

    # 工作空间检查
    if abs(cos_q2) > 1.0:
        raise ValueError(
            f"Target ({x:.3f}, {y:.3f}) "
            "is outside the workspace."
        )

    # 数值安全处理
    cos_q2 = np.clip(
        cos_q2,
        -1.0,
        1.0
    )

    # 选择一个 IK 解
    q2 = np.arccos(cos_q2)

    q1 = (
        np.arctan2(y, x)
        -
        np.arctan2(
            l2 * np.sin(q2),
            l1 + l2 * np.cos(q2)
        )
    )

    return np.array([q1, q2])


# ============================================================
# 5. 设置目标点
# ============================================================

target = np.array([1.2, 0.8])

print("\n===== Target =====")
print("Target position =", target)


# ============================================================
# 6. IK：目标点 -> 目标关节角
# ============================================================

q_desired = inverse_kinematics(
    target[0],
    target[1]
)

print("\n===== IK Result =====")

print(
    f"q1_desired = "
    f"{np.rad2deg(q_desired[0]):.2f} deg"
)

print(
    f"q2_desired = "
    f"{np.rad2deg(q_desired[1]):.2f} deg"
)


# ============================================================
# 7. 用 FK 检查 IK
# ============================================================

fk_check = forward_kinematics(
    q_desired[0],
    q_desired[1]
)

ik_error = np.linalg.norm(
    target - fk_check
)

print("\n===== IK Verification =====")
print("FK result =", fk_check)
print("IK/FK error =", ik_error)


# ============================================================
# 8. 设置机械臂初始状态
# ============================================================

data.qpos[:] = np.deg2rad(
    [0.0, 0.0]
)

data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# 9. PD 参数
# ============================================================

Kp = np.array([
    20.0,
    20.0
])

# 使用前面实验效果较好的 Kd
Kd = np.array([
    10.0,
    10.0
])

qdot_desired = np.zeros(2)

ctrl_min = -20.0
ctrl_max = 20.0


# ============================================================
# 10. Reach 判定参数
# ============================================================

# 末端距离目标小于 1 cm
reach_threshold = 0.01

# 必须连续稳定一定时间
required_stable_steps = 200

# 当前已经连续满足阈值多少步
stable_counter = 0


# ============================================================
# 11. 最大仿真时间
# ============================================================

max_simulation_time = 20.0


# ============================================================
# 12. 数据记录
# ============================================================

time_history = []
task_error_history = []
q_history = []


# ============================================================
# 13. 开始仿真
# ============================================================

task_success = False

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step_count = 0

    while (
        viewer.is_running()
        and data.time < max_simulation_time
    ):

        # ----------------------------------------------------
        # 1) 当前关节状态
        # ----------------------------------------------------

        q = data.qpos[:2].copy()
        qdot = data.qvel[:2].copy()


        # ----------------------------------------------------
        # 2) Joint-space error
        # ----------------------------------------------------

        position_error = (
            q_desired - q
        )

        velocity_error = (
            qdot_desired - qdot
        )


        # ----------------------------------------------------
        # 3) PD 控制器
        # ----------------------------------------------------

        torque = (
            Kp * position_error
            +
            Kd * velocity_error
        )

        torque = np.clip(
            torque,
            ctrl_min,
            ctrl_max
        )

        data.ctrl[:] = torque


        # ----------------------------------------------------
        # 4) 推进一步仿真
        # ----------------------------------------------------

        mujoco.mj_step(
            model,
            data
        )


        # ----------------------------------------------------
        # 5) 计算当前末端位置
        # ----------------------------------------------------

        ee_position = forward_kinematics(
            data.qpos[0],
            data.qpos[1]
        )


        # ----------------------------------------------------
        # 6) Task-space error
        # ----------------------------------------------------

        task_error_vector = (
            target - ee_position
        )

        task_error = np.linalg.norm(
            task_error_vector
        )


        # ----------------------------------------------------
        # 7) 判断是否进入目标范围
        # ----------------------------------------------------

        if task_error < reach_threshold:

            stable_counter += 1

        else:

            stable_counter = 0


        # ----------------------------------------------------
        # 8) 判断是否已经稳定到达
        # ----------------------------------------------------

        if stable_counter >= required_stable_steps:

            task_success = True

            print(
                "\nTarget reached and stabilized!"
            )

            break


        # ----------------------------------------------------
        # 9) 保存数据
        # ----------------------------------------------------

        time_history.append(
            data.time
        )

        task_error_history.append(
            task_error
        )

        q_history.append(
            data.qpos[:2].copy()
        )


        # ----------------------------------------------------
        # 10) 打印状态
        # ----------------------------------------------------

        step_count += 1

        if step_count % 100 == 0:

            stable_time = (
                stable_counter
                *
                model.opt.timestep
            )

            print(
                f"Time: {data.time:.2f} s | "
                f"q1 = "
                f"{np.rad2deg(data.qpos[0]):.2f} deg | "
                f"q2 = "
                f"{np.rad2deg(data.qpos[1]):.2f} deg | "
                f"ee = "
                f"({ee_position[0]:.3f}, "
                f"{ee_position[1]:.3f}) | "
                f"Error = {task_error:.4f} | "
                f"Stable = {stable_time:.3f} s"
            )


        # ----------------------------------------------------
        # 11) 更新 Viewer
        # ----------------------------------------------------

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )


# ============================================================
# 14. 仿真结束结果
# ============================================================

final_q = data.qpos[:2].copy()

final_ee = forward_kinematics(
    final_q[0],
    final_q[1]
)

final_error = np.linalg.norm(
    target - final_ee
)


print("\n===== Reach Task Result =====")

if task_success:

    print("Task status: SUCCESS")

else:

    print("Task status: FAILED / TIMEOUT")


print(
    f"Simulation time = "
    f"{data.time:.3f} s"
)

print(
    f"Final q1 = "
    f"{np.rad2deg(final_q[0]):.2f} deg"
)

print(
    f"Final q2 = "
    f"{np.rad2deg(final_q[1]):.2f} deg"
)

print(
    f"Final end-effector = "
    f"({final_ee[0]:.4f}, "
    f"{final_ee[1]:.4f})"
)

print(
    f"Target = "
    f"({target[0]:.4f}, "
    f"{target[1]:.4f})"
)

print(
    f"Final task error = "
    f"{final_error:.6f}"
)