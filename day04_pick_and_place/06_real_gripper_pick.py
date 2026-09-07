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
    "assets",
    "gripper_scene.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("===== Model Information =====")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)


# ============================================================
# 2. 机械臂运动学参数
# ============================================================

l1 = 1.0
l2 = 1.02


def inverse_kinematics(x, y):
    """
    2DOF 平面机械臂解析逆运动学
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
    ) / (
        2 * l1 * l2
    )

    if abs(cos_q2) > 1.0:
        raise ValueError(
            f"Target ({x:.3f}, {y:.3f}) outside workspace."
        )

    cos_q2 = np.clip(
        cos_q2,
        -1.0,
        1.0
    )

    q2 = np.arccos(cos_q2)

    q1 = (
        np.arctan2(y, x)
        -
        np.arctan2(
            l2 * np.sin(q2),
            l1 + l2 * np.cos(q2)
        )
    )

    return np.array([
        q1,
        q2
    ])


# ============================================================
# 3. 机械臂 PD 控制器
# ============================================================

Kp = np.array([
    20.0,
    20.0
])

Kd = np.array([
    10.0,
    10.0
])


def arm_pd_control(
    q,
    qdot,
    q_desired
):
    """
    Joint-space PD
    """

    position_error = (
        q_desired - q
    )

    velocity_error = (
        -qdot
    )

    torque = (
        Kp * position_error
        +
        Kd * velocity_error
    )

    torque = np.clip(
        torque,
        -20.0,
        20.0
    )

    return torque


# ============================================================
# 4. 查找 MuJoCo 对象 ID
# ============================================================

ee_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ee_site"
)

cube_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "cube"
)

place_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "place_target"
)

left_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "left_gripper_act"
)

right_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "right_gripper_act"
)

left_finger_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "left_finger_geom"
)

right_finger_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "right_finger_geom"
)

cube_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "cube_geom"
)


# ============================================================
# 5. 左右夹爪关节位置地址
# ============================================================

left_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "left_finger_joint"
)

right_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "right_finger_joint"
)

left_qpos_adr = model.jnt_qposadr[
    left_joint_id
]

right_qpos_adr = model.jnt_qposadr[
    right_joint_id
]


# ============================================================
# 6. 夹爪参数
# ============================================================

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.065


def set_gripper(target):
    """
    给左右 position actuator 设置目标位置
    """

    data.ctrl[
        left_actuator_id
    ] = target

    data.ctrl[
        right_actuator_id
    ] = target


# ============================================================
# 7. 接触检测函数
# ============================================================

def check_contact(
    geom_a,
    geom_b
):
    """
    判断两个 geom 当前是否有接触
    """

    for i in range(data.ncon):

        contact = data.contact[i]

        g1 = contact.geom1
        g2 = contact.geom2

        if (
            (
                g1 == geom_a
                and g2 == geom_b
            )
            or
            (
                g1 == geom_b
                and g2 == geom_a
            )
        ):
            return True

    return False


# ============================================================
# 8. 状态机定义
# ============================================================

MOVE_TO_PREGRASP = 0
APPROACH_CUBE = 1
CLOSE_GRIPPER = 2
VERIFY_GRASP = 3
MOVE_TO_PLACE = 4
OPEN_GRIPPER = 5
DONE = 6
FAILED = 7


state_names = {
    MOVE_TO_PREGRASP:
        "MOVE_TO_PREGRASP",

    APPROACH_CUBE:
        "APPROACH_CUBE",

    CLOSE_GRIPPER:
        "CLOSE_GRIPPER",

    VERIFY_GRASP:
        "VERIFY_GRASP",

    MOVE_TO_PLACE:
        "MOVE_TO_PLACE",

    OPEN_GRIPPER:
        "OPEN_GRIPPER",

    DONE:
        "DONE",

    FAILED:
        "FAILED"
}


state = MOVE_TO_PREGRASP


# ============================================================
# 9. 状态判定参数
# ============================================================

reach_threshold = 0.04

required_stable_steps = 100

stable_counter = 0


# 抓取验证
required_grasp_steps = 100

grasp_stable_counter = 0


# 防止移动过程中瞬间失去接触就直接判失败
drop_counter = 0

required_drop_steps = 100


state_start_time = 0.0


# ============================================================
# 10. 初始状态
# ============================================================

data.qpos[0] = 0.0
data.qpos[1] = 0.0

data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# 11. 最大运行时间
# ============================================================

max_simulation_time = 40.0


# ============================================================
# 12. Viewer
# ============================================================

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
        # 读取当前状态
        # ----------------------------------------------------

        q = data.qpos[:2].copy()

        qdot = data.qvel[:2].copy()


        ee_xy = data.site_xpos[
            ee_site_id
        ][:2].copy()


        cube_xy = data.xpos[
            cube_body_id
        ][:2].copy()


        place_xy = data.site_xpos[
            place_site_id
        ][:2].copy()


        # ----------------------------------------------------
        # 当前接触状态
        # ----------------------------------------------------

        left_contact = check_contact(
            left_finger_geom_id,
            cube_geom_id
        )

        right_contact = check_contact(
            right_finger_geom_id,
            cube_geom_id
        )


        # ====================================================
        # 1. MOVE_TO_PREGRASP
        # ====================================================

        if state == MOVE_TO_PREGRASP:

            set_gripper(
                GRIPPER_OPEN
            )

            # 从 cube 外侧先停一下
            pregrasp_target = (
                cube_xy
                +
                np.array([
                    0.18,
                    0.0
                ])
            )

            target = pregrasp_target


        # ====================================================
        # 2. APPROACH_CUBE
        # ====================================================

        elif state == APPROACH_CUBE:

            set_gripper(
                GRIPPER_OPEN
            )

            target = cube_xy


        # ====================================================
        # 3. CLOSE_GRIPPER
        # ====================================================

        elif state == CLOSE_GRIPPER:

            set_gripper(
                GRIPPER_CLOSE
            )

            # 保持机械臂末端在抓取位置
            target = cube_xy


            # 给夹爪时间闭合
            if (
                data.time
                -
                state_start_time
                >
                1.0
            ):

                state = VERIFY_GRASP

                state_start_time = (
                    data.time
                )

                grasp_stable_counter = 0


        # ====================================================
        # 4. VERIFY_GRASP
        # ====================================================

        elif state == VERIFY_GRASP:

            set_gripper(
                GRIPPER_CLOSE
            )

            target = cube_xy


            # 左右两根手指都必须接触 cube
            if (
                left_contact
                and
                right_contact
            ):

                grasp_stable_counter += 1

            else:

                grasp_stable_counter = 0


            if (
                grasp_stable_counter
                >=
                required_grasp_steps
            ):

                print(
                    "\n>>> Grasp verified!"
                )

                state = MOVE_TO_PLACE

                state_start_time = (
                    data.time
                )

                stable_counter = 0
                drop_counter = 0


            # 如果很久都夹不上，判定失败
            if (
                data.time
                -
                state_start_time
                >
                3.0
            ):

                print(
                    "\n>>> Grasp failed: "
                    "both fingers did not establish stable contact."
                )

                state = FAILED


        # ====================================================
        # 5. MOVE_TO_PLACE
        # ====================================================

        elif state == MOVE_TO_PLACE:

            set_gripper(
                GRIPPER_CLOSE
            )

            target = place_xy


            # ------------------------------
            # 抓取掉落检测
            # ------------------------------

            if (
                left_contact
                and
                right_contact
            ):

                drop_counter = 0

            else:

                drop_counter += 1


            if (
                drop_counter
                >=
                required_drop_steps
            ):

                print(
                    "\n>>> Cube dropped during transport!"
                )

                state = FAILED


        # ====================================================
        # 6. OPEN_GRIPPER
        # ====================================================

        elif state == OPEN_GRIPPER:

            set_gripper(
                GRIPPER_OPEN
            )

            target = place_xy


            if (
                data.time
                -
                state_start_time
                >
                1.0
            ):

                state = DONE


        # ====================================================
        # 7. DONE
        # ====================================================

        elif state == DONE:

            print(
                "\n===== REAL GRIPPER PICK AND PLACE SUCCESS ====="
            )

            break


        # ====================================================
        # 8. FAILED
        # ====================================================

        elif state == FAILED:

            print(
                "\n===== TASK FAILED ====="
            )

            break


        # ====================================================
        # 13. IK
        # ============================================================

        try:

            q_desired = inverse_kinematics(
                target[0],
                target[1]
            )

        except ValueError as e:

            print(e)

            state = FAILED

            continue


        # ====================================================
        # 14. Arm PD
        # ============================================================

        torque = arm_pd_control(
            q,
            qdot,
            q_desired
        )

        data.ctrl[0:2] = torque


        # ====================================================
        # 15. 当前末端到目标误差
        # ============================================================

        ee_error = np.linalg.norm(
            target - ee_xy
        )


        # ====================================================
        # 16. 到达判定
        # ============================================================

        if state in [
            MOVE_TO_PREGRASP,
            APPROACH_CUBE,
            MOVE_TO_PLACE
        ]:

            if (
                ee_error
                <
                reach_threshold
            ):

                stable_counter += 1

            else:

                stable_counter = 0


        # ----------------------------------------------------
        # PREGRASP -> APPROACH
        # ----------------------------------------------------

        if (
            state
            ==
            MOVE_TO_PREGRASP
            and
            stable_counter
            >=
            required_stable_steps
        ):

            print(
                "\n>>> Pre-grasp reached."
            )

            state = APPROACH_CUBE

            stable_counter = 0

            state_start_time = (
                data.time
            )


        # ----------------------------------------------------
        # APPROACH -> CLOSE
        # ----------------------------------------------------

        elif (
            state
            ==
            APPROACH_CUBE
            and
            stable_counter
            >=
            required_stable_steps
        ):

            print(
                "\n>>> Cube approach position reached."
            )

            state = CLOSE_GRIPPER

            stable_counter = 0

            state_start_time = (
                data.time
            )


        # ----------------------------------------------------
        # PLACE -> OPEN
        # ----------------------------------------------------

        elif (
            state
            ==
            MOVE_TO_PLACE
            and
            stable_counter
            >=
            required_stable_steps
        ):

            print(
                "\n>>> Place target reached."
            )

            state = OPEN_GRIPPER

            stable_counter = 0

            state_start_time = (
                data.time
            )


        # ====================================================
        # 17. 推进 MuJoCo
        # ============================================================

        mujoco.mj_step(
            model,
            data
        )


        # ====================================================
        # 18. 调试输出
        # ============================================================

        step_count += 1

        if step_count % 100 == 0:

            left_finger_pos = (
                data.qpos[
                    left_qpos_adr
                ]
            )

            right_finger_pos = (
                data.qpos[
                    right_qpos_adr
                ]
            )

            print(
                f"Time={data.time:.2f}s | "
                f"State={state_names[state]} | "
                f"EE=({ee_xy[0]:.3f},"
                f"{ee_xy[1]:.3f}) | "
                f"Cube=({cube_xy[0]:.3f},"
                f"{cube_xy[1]:.3f}) | "
                f"Target=({target[0]:.3f},"
                f"{target[1]:.3f}) | "
                f"Error={ee_error:.4f} | "
                f"LContact={left_contact} | "
                f"RContact={right_contact} | "
                f"Lfinger={left_finger_pos:.4f} | "
                f"Rfinger={right_finger_pos:.4f} | "
                f"Contacts={data.ncon}"
            )


        viewer.sync()

        time.sleep(
            model.opt.timestep
        )


# ============================================================
# 19. 最终结果
# ============================================================

mujoco.mj_forward(
    model,
    data
)

final_cube_xy = data.xpos[
    cube_body_id
][:2].copy()

final_place_xy = data.site_xpos[
    place_site_id
][:2].copy()

place_error = np.linalg.norm(
    final_cube_xy
    -
    final_place_xy
)

print("\n===== Final Result =====")

print(
    f"State = "
    f"{state_names[state]}"
)

print(
    f"Cube final position = "
    f"({final_cube_xy[0]:.4f}, "
    f"{final_cube_xy[1]:.4f})"
)

print(
    f"Place target = "
    f"({final_place_xy[0]:.4f}, "
    f"{final_place_xy[1]:.4f})"
)

print(
    f"Cube-place error = "
    f"{place_error:.6f}"
)