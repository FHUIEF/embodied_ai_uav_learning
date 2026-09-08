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
    "three_link_gripper_scene.xml"
)

model = mujoco.MjModel.from_xml_path(
    xml_path
)

data = mujoco.MjData(
    model
)


print("===== Model Information =====")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)


# ============================================================
# 2. 有效机械臂长度
# ============================================================

l1 = 1.0
l2 = 0.8

# joint3 -> link3 -> grasp_site
l3 = 0.50


# ============================================================
# 3. FK
# ============================================================

def forward_kinematics_3dof(
    q1,
    q2,
    q3
):

    theta1 = q1
    theta2 = q1 + q2
    theta3 = q1 + q2 + q3

    x = (
        l1 * np.cos(theta1)
        +
        l2 * np.cos(theta2)
        +
        l3 * np.cos(theta3)
    )

    y = (
        l1 * np.sin(theta1)
        +
        l2 * np.sin(theta2)
        +
        l3 * np.sin(theta3)
    )

    phi = theta3

    return np.array([
        x,
        y,
        phi
    ])


# ============================================================
# 4. IK
# ============================================================

def inverse_kinematics_3dof(
    x_d,
    y_d,
    phi_d,
    elbow="down"
):

    # wrist center
    x_w = (
        x_d
        -
        l3 * np.cos(phi_d)
    )

    y_w = (
        y_d
        -
        l3 * np.sin(phi_d)
    )


    cos_q2 = (
        x_w**2
        +
        y_w**2
        -
        l1**2
        -
        l2**2
    ) / (
        2 * l1 * l2
    )


    if abs(cos_q2) > 1.0:

        raise ValueError(
            "Target pose is outside workspace."
        )


    cos_q2 = np.clip(
        cos_q2,
        -1.0,
        1.0
    )


    if elbow == "down":

        q2 = np.arccos(
            cos_q2
        )

    else:

        q2 = -np.arccos(
            cos_q2
        )


    q1 = (
        np.arctan2(
            y_w,
            x_w
        )
        -
        np.arctan2(
            l2 * np.sin(q2),
            l1 + l2 * np.cos(q2)
        )
    )


    q3 = (
        phi_d
        -
        q1
        -
        q2
    )


    return np.array([
        q1,
        q2,
        q3
    ])


# ============================================================
# 5. wrap angle
# ============================================================

def wrap_to_pi(angle):

    return (
        angle
        +
        np.pi
    ) % (
        2 * np.pi
    ) - np.pi


# ============================================================
# 6. Arm PD
# ============================================================

Kp = np.array([
    20.0,
    20.0,
    15.0
])

Kd = np.array([
    10.0,
    10.0,
    6.0
])

TORQUE_LIMIT = np.array([
    20.0,
    20.0,
    10.0
])


def arm_pd_control(
    q,
    qdot,
    q_desired
):

    torque = (
        Kp
        *
        (
            q_desired
            -
            q
        )
        -
        Kd
        *
        qdot
    )

    torque = np.clip(
        torque,
        -TORQUE_LIMIT,
        TORQUE_LIMIT
    )

    return torque


# ============================================================
# 7. 获取 MuJoCo object IDs
# ============================================================

grasp_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "grasp_site"
)

cube_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "cube"
)

cube_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "cube_geom"
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


# ============================================================
# 8. Contact function
# ============================================================

def check_contact(
    geom_a,
    geom_b
):

    for i in range(data.ncon):

        contact = data.contact[i]

        g1 = contact.geom1
        g2 = contact.geom2

        if (
            (
                g1 == geom_a
                and
                g2 == geom_b
            )
            or
            (
                g1 == geom_b
                and
                g2 == geom_a
            )
        ):

            return True

    return False


# ============================================================
# 9. Gripper control
# ============================================================

GRIPPER_OPEN = 0.0

GRIPPER_CLOSE = 0.065


def set_gripper(target):

    data.ctrl[
        left_actuator_id
    ] = target

    data.ctrl[
        right_actuator_id
    ] = target


# ============================================================
# 10. Object pose
# ============================================================

mujoco.mj_forward(
    model,
    data
)

cube_position = data.xpos[
    cube_body_id
][:2].copy()


print(
    "\nCube position =",
    cube_position
)


# ============================================================
# 11. Grasp orientation
# ============================================================

phi_grasp = np.deg2rad(
    0.0
)


# ============================================================
# 12. Pregrasp
# ============================================================

pregrasp_distance = 0.30


approach_direction = np.array([
    np.cos(phi_grasp),
    np.sin(phi_grasp)
])


pregrasp_xy = (
    cube_position
    -
    pregrasp_distance
    *
    approach_direction
)


grasp_xy = (
    cube_position.copy()
)


pregrasp_pose = np.array([
    pregrasp_xy[0],
    pregrasp_xy[1],
    phi_grasp
])


grasp_pose = np.array([
    grasp_xy[0],
    grasp_xy[1],
    phi_grasp
])


print(
    "Pregrasp pose =",
    pregrasp_pose
)

print(
    "Grasp pose =",
    grasp_pose
)


# ============================================================
# 13. FSM
# ============================================================

MOVE_TO_PREGRASP = 0

APPROACH = 1

HOLD = 2

CLOSE_GRIPPER = 3

VERIFY_GRASP = 4

DONE = 5

FAILED = 6


state = MOVE_TO_PREGRASP


state_names = {

    MOVE_TO_PREGRASP:
        "MOVE_TO_PREGRASP",

    APPROACH:
        "APPROACH",

    HOLD:
        "HOLD",

    CLOSE_GRIPPER:
        "CLOSE_GRIPPER",

    VERIFY_GRASP:
        "VERIFY_GRASP",

    DONE:
        "DONE",

    FAILED:
        "FAILED"
}


# ============================================================
# 14. Reach parameters
# ============================================================

position_threshold = 0.015

orientation_threshold = np.deg2rad(
    1.5
)

required_stable_steps = 100

stable_counter = 0


# ============================================================
# 15. Grasp verification
# ============================================================

required_grasp_steps = 100

grasp_counter = 0


# ============================================================
# 16. Hold
# ============================================================

hold_pose = None

state_start_time = 0.0


# ============================================================
# 17. 初始化
# ============================================================

data.qpos[:3] = np.deg2rad([
    0.0,
    0.0,
    0.0
])

data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# 18. Simulation
# ============================================================

simulation_time = 30.0

step_count = 0


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while (
        viewer.is_running()
        and
        data.time
        <
        simulation_time
    ):

        # ----------------------------------------------------
        # Current arm state
        # ----------------------------------------------------

        q = data.qpos[
            :3
        ].copy()

        qdot = data.qvel[
            :3
        ].copy()


        # ----------------------------------------------------
        # Actual grasp site from MuJoCo
        # ----------------------------------------------------

        grasp_xy_actual = data.site_xpos[
            grasp_site_id
        ][:2].copy()


        # ----------------------------------------------------
        # FK pose
        # ----------------------------------------------------

        current_pose = forward_kinematics_3dof(
            q[0],
            q[1],
            q[2]
        )


        # ----------------------------------------------------
        # Contact
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
        # MOVE TO PREGRASP
        # ====================================================

        if state == MOVE_TO_PREGRASP:

            set_gripper(
                GRIPPER_OPEN
            )

            target_pose = (
                pregrasp_pose
            )


        # ====================================================
        # APPROACH
        # ====================================================

        elif state == APPROACH:

            set_gripper(
                GRIPPER_OPEN
            )

            target_pose = (
                grasp_pose
            )


        # ====================================================
        # HOLD
        # ====================================================

        elif state == HOLD:

            set_gripper(
                GRIPPER_OPEN
            )

            target_pose = (
                hold_pose
            )


            if (
                data.time
                -
                state_start_time
                >
                0.5
            ):

                print(
                    "\n>>> Hold finished."
                )

                state = CLOSE_GRIPPER

                state_start_time = (
                    data.time
                )


        # ====================================================
        # CLOSE GRIPPER
        # ====================================================

        elif state == CLOSE_GRIPPER:

            set_gripper(
                GRIPPER_CLOSE
            )

            target_pose = (
                hold_pose
            )


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

                grasp_counter = 0


        # ====================================================
        # VERIFY GRASP
        # ====================================================

        elif state == VERIFY_GRASP:

            set_gripper(
                GRIPPER_CLOSE
            )

            target_pose = (
                hold_pose
            )


            if (
                left_contact
                and
                right_contact
            ):

                grasp_counter += 1

            else:

                grasp_counter = 0


            if (
                grasp_counter
                >=
                required_grasp_steps
            ):

                print(
                    "\n>>> Grasp verified!"
                )

                state = DONE


            elif (
                data.time
                -
                state_start_time
                >
                3.0
            ):

                print(
                    "\n>>> Grasp failed."
                )

                state = FAILED


        # ====================================================
        # DONE
        # ====================================================

        elif state == DONE:

            set_gripper(
                GRIPPER_CLOSE
            )

            print(
                "\n===== GRASP SUCCESS ====="
            )

            break


        # ====================================================
        # FAILED
        # ====================================================

        elif state == FAILED:

            print(
                "\n===== GRASP FAILED ====="
            )

            break


        # ====================================================
        # IK
        # ====================================================

        q_desired = inverse_kinematics_3dof(
            target_pose[0],
            target_pose[1],
            target_pose[2],
            elbow="down"
        )


        # ====================================================
        # PD
        # ====================================================

        torque = arm_pd_control(
            q,
            qdot,
            q_desired
        )


        data.ctrl[
            0:3
        ] = torque


        # ====================================================
        # Pose error
        # ====================================================

        position_error = np.linalg.norm(
            target_pose[:2]
            -
            current_pose[:2]
        )


        orientation_error = wrap_to_pi(
            target_pose[2]
            -
            current_pose[2]
        )


        # ====================================================
        # Reach check
        # ====================================================

        if state in [
            MOVE_TO_PREGRASP,
            APPROACH
        ]:

            if (
                position_error
                <
                position_threshold
                and
                abs(
                    orientation_error
                )
                <
                orientation_threshold
            ):

                stable_counter += 1

            else:

                stable_counter = 0


        # ====================================================
        # State transitions
        # ====================================================

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
                "\n>>> Pregrasp reached."
            )

            state = APPROACH

            stable_counter = 0


        elif (
            state
            ==
            APPROACH
            and
            stable_counter
            >=
            required_stable_steps
        ):

            print(
                "\n>>> Grasp pose reached."
            )

            # 锁定当前末端 Pose
            hold_pose = (
                current_pose.copy()
            )

            state = HOLD

            state_start_time = (
                data.time
            )

            stable_counter = 0


        # ====================================================
        # Step
        # ====================================================

        mujoco.mj_step(
            model,
            data
        )


        # ====================================================
        # Debug output
        # ====================================================

        step_count += 1

        if step_count % 100 == 0:

            cube_xy_now = data.xpos[
                cube_body_id
            ][:2].copy()


            print(
                f"Time={data.time:.2f}s | "
                f"State={state_names[state]} | "
                f"GraspSite=("
                f"{grasp_xy_actual[0]:.3f},"
                f"{grasp_xy_actual[1]:.3f}) | "
                f"Cube=("
                f"{cube_xy_now[0]:.3f},"
                f"{cube_xy_now[1]:.3f}) | "
                f"PosErr="
                f"{position_error:.4f} | "
                f"OriErr="
                f"{np.rad2deg(orientation_error):.2f}deg | "
                f"LContact="
                f"{left_contact} | "
                f"RContact="
                f"{right_contact} | "
                f"ncon="
                f"{data.ncon}"
            )


        viewer.sync()

        time.sleep(
            model.opt.timestep
        )