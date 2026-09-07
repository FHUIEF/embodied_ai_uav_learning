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
    "pick_place_scene.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


print("===== Model Information =====")

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)
print("nmocap =", model.nmocap)


# ============================================================
# 2. 机械臂参数
# ============================================================

l1 = 1.0
l2 = 1.0


# ============================================================
# 3. Forward Kinematics
# ============================================================

def forward_kinematics(q1, q2):

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

    cos_q2 = ( x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if abs(cos_q2) > 1.0:

        raise ValueError(
            f"Target ({x:.3f}, {y:.3f}) "
            "is outside workspace."
        )

    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

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
# 5. PD 控制器
# ============================================================

Kp = np.array([20.0, 20.0])

Kd = np.array([10.0, 10.0])

qdot_desired = np.zeros(2)

ctrl_min = -20.0
ctrl_max = 20.0


def joint_pd_control(
    q,
    qdot,
    q_desired
):

    position_error = q_desired - q

    velocity_error = qdot_desired - qdot

    torque = (
        Kp * position_error
        +
        Kd * velocity_error
    )

    torque = np.clip(torque, ctrl_min, ctrl_max)

    return torque


# ============================================================
# 6. 初始任务参数
# ============================================================

cube_position = data.mocap_pos[0][:2].copy()

place_position = np.array([0.5, 1.4])


# ============================================================
# 7. 到达判定参数
# ============================================================

reach_threshold = 0.02

required_stable_steps = 150

stable_counter = 0


# ============================================================
# 8. 状态机
# ============================================================

MOVE_TO_CUBE = 0
GRASP = 1
MOVE_TO_TARGET = 2
RELEASE = 3
DONE = 4

state = MOVE_TO_CUBE


# 是否已经抓住方块
cube_grasped = False


# ============================================================
# 9. 设置机械臂初始状态
# ============================================================

data.qpos[:2] = np.deg2rad([0.0, 0.0])

data.qvel[:2] = 0.0

mujoco.mj_forward(model, data)


# ============================================================
# 10. 最大仿真时间
# ============================================================

max_simulation_time = 40.0


# ============================================================
# 11. Viewer
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while (viewer.is_running() and data.time < max_simulation_time):

        # ====================================================
        # 当前机械臂状态
        # ====================================================

        q = data.qpos[:2].copy()

        qdot = data.qvel[:2].copy()


        # ====================================================
        # 当前末端位置
        # ====================================================

        ee_position = forward_kinematics(q[0], q[1])


        # ====================================================
        # 状态 1：移动到方块
        # ====================================================

        if state == MOVE_TO_CUBE:

            current_target = cube_position

            q_desired = inverse_kinematics(
                current_target[0],
                current_target[1]
            )

            task_error = np.linalg.norm(
                current_target
                -
                ee_position
            )


            # ------------------------------
            # 到达稳定判定
            # ------------------------------

            if (
                task_error
                <
                reach_threshold
            ):

                stable_counter += 1

            else:

                stable_counter = 0


            # ------------------------------
            # 稳定到达方块
            # ------------------------------

            if (
                stable_counter
                >=
                required_stable_steps
            ):

                print(
                    "\n>>> Reached cube."
                )

                state = GRASP

                stable_counter = 0


        # ====================================================
        # 状态 2：抓住方块
        # ====================================================

        elif state == GRASP:

            print(
                ">>> Cube grasped."
            )

            cube_grasped = True

            state = MOVE_TO_TARGET

            stable_counter = 0

            continue


        # ====================================================
        # 状态 3：移动到放置位置
        # ====================================================

        elif state == MOVE_TO_TARGET:

            current_target = (
                place_position
            )

            q_desired = inverse_kinematics(
                current_target[0],
                current_target[1]
            )

            task_error = np.linalg.norm(
                current_target
                -
                ee_position
            )


            if (
                task_error
                <
                reach_threshold
            ):

                stable_counter += 1

            else:

                stable_counter = 0


            if (
                stable_counter
                >=
                required_stable_steps
            ):

                print(
                    "\n>>> Reached place target."
                )

                state = RELEASE

                stable_counter = 0


        # ====================================================
        # 状态 4：释放方块
        # ====================================================

        elif state == RELEASE:

            print(
                ">>> Cube released."
            )

            cube_grasped = False

            # 把方块固定在目标位置
            data.mocap_pos[0][0] = (
                place_position[0]
            )

            data.mocap_pos[0][1] = (
                place_position[1]
            )

            state = DONE

            continue


        # ====================================================
        # 状态 5：任务完成
        # ====================================================

        elif state == DONE:

            print(
                "\n===== Pick and Place SUCCESS ====="
            )

            break


        # ====================================================
        # PD 控制
        # ====================================================

        torque = joint_pd_control(
            q,
            qdot,
            q_desired
        )

        data.ctrl[:] = torque


        # ====================================================
        # 如果已经抓住方块
        # 让方块跟随机械臂末端
        # ====================================================

        if cube_grasped:

            data.mocap_pos[0][0] = (
                ee_position[0]
            )

            data.mocap_pos[0][1] = (
                ee_position[1]
            )


        # ====================================================
        # 推进仿真
        # ====================================================

        mujoco.mj_step(
            model,
            data
        )


        # ====================================================
        # 打印状态
        # ====================================================

        step_count += 1

        if step_count % 100 == 0:

            state_names = {
                MOVE_TO_CUBE:
                    "MOVE_TO_CUBE",

                GRASP:
                    "GRASP",

                MOVE_TO_TARGET:
                    "MOVE_TO_TARGET",

                RELEASE:
                    "RELEASE",

                DONE:
                    "DONE"
            }

            print(
                f"Time={data.time:.2f}s | "
                f"State={state_names[state]} | "
                f"EE=({ee_position[0]:.3f}, {ee_position[1]:.3f}) | " 
                f"Target=({current_target[0]:.3f}, {current_target[1]:.3f}) | "
                f"Error={task_error:.4f} | " 
                f"Stable={stable_counter}"
            )


        viewer.sync()

        time.sleep(
            model.opt.timestep
        )


# ============================================================
# 12. 最终结果
# ============================================================

print("\n===== Final Result =====")

print(
    "Cube final position =",
    data.mocap_pos[0]
)

print(
    "Place target =",
    place_position
)