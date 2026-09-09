import os
import time

import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# 1. Load scene
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(current_dir, "assets", "franka_pick_cube", "scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)


# ============================================================
# 2. Arm joints
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

arm_joint_ids = []
arm_qpos_adr = []
arm_dof_adr = []


for name in arm_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    arm_joint_ids.append(joint_id)

    arm_qpos_adr.append(model.jnt_qposadr[joint_id])

    arm_dof_adr.append(model.jnt_dofadr[joint_id])


arm_joint_ids = np.array(arm_joint_ids)

arm_qpos_adr = np.array(arm_qpos_adr)

arm_dof_adr = np.array(arm_dof_adr)


# ============================================================
# 3. Finger joints
# ============================================================

finger_joint_names = [
    "finger_joint1",
    "finger_joint2",
]

finger_qpos_adr = []


for name in finger_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    finger_qpos_adr.append(model.jnt_qposadr[joint_id])


finger_qpos_adr = np.array(finger_qpos_adr)


# ============================================================
# 4. Actuators
# ============================================================

arm_actuator_ids = []


for i in range(1, 8):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")

    arm_actuator_ids.append(actuator_id)


arm_actuator_ids = np.array(arm_actuator_ids)


gripper_actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
)


# ============================================================
# 5. Body / geom / site
# ============================================================

cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")


left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")

right_finger_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
)


tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")


# ============================================================
# 6. Home
# ============================================================

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -0.7853])


finger_home = np.array([0.04, 0.04])


# ============================================================
# 7. Initialize robot
# ============================================================

for i in range(7):
    data.qpos[arm_qpos_adr[i]] = q_home[i]


for i in range(2):
    data.qpos[finger_qpos_adr[i]] = finger_home[i]


data.qvel[:] = 0.0


for i in range(7):
    data.ctrl[arm_actuator_ids[i]] = q_home[i]


data.ctrl[gripper_actuator_id] = 255.0


mujoco.mj_forward(model, data)


# ============================================================
# 8. TCP home orientation
# ============================================================

target_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()


# ============================================================
# 9. Rotation error
# ============================================================


def rotation_error_vector(R_current, R_target):
    R_error = R_target @ R_current.T

    cos_theta = (np.trace(R_error) - 1.0) / 2.0

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    axis = np.array(
        [
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1],
        ]
    )

    axis = axis / (2.0 * np.sin(theta))

    return theta * axis


# ============================================================
# 10. Joint limit
# ============================================================


def apply_joint_limits(q):
    q = q.copy()

    for i in range(7):
        joint_id = arm_joint_ids[i]

        if model.jnt_limited[joint_id]:
            lower = model.jnt_range[joint_id, 0]

            upper = model.jnt_range[joint_id, 1]

            q[i] = np.clip(q[i], lower, upper)

    return q


# ============================================================
# 11. TCP 6D IK
# ============================================================


def solve_pose_ik(
    q_init, target_position, target_rotation, max_iterations=1200, step_size=0.2
):
    q = q_init.copy()

    for iteration in range(max_iterations):
        for i in range(7):
            data.qpos[arm_qpos_adr[i]] = q[i]

        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        current_position = data.site_xpos[tcp_site_id].copy()

        current_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()

        position_error_vector = target_position - current_position

        orientation_error_vector = rotation_error_vector(
            current_rotation, target_rotation
        )

        position_error = np.linalg.norm(position_error_vector)

        orientation_error = np.linalg.norm(orientation_error_vector)

        if position_error < 1e-4 and orientation_error < np.deg2rad(0.1):
            print(
                f"IK converged: "
                f"iteration={iteration}, "
                f"pos_error={position_error:.6f} m, "
                f"ori_error="
                f"{np.rad2deg(orientation_error):.4f} deg"
            )

            return q, True

        jacp = np.zeros((3, model.nv))

        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, jacr, tcp_site_id)

        J_position = jacp[:, arm_dof_adr]

        J_rotation = jacr[:, arm_dof_adr]

        J = np.vstack([J_position, J_rotation])

        pose_error = np.concatenate([position_error_vector, orientation_error_vector])

        delta_q = np.linalg.pinv(J) @ pose_error

        # 防止一步跨太大
        delta_q = np.clip(delta_q, -0.10, 0.10)

        q = q + step_size * delta_q

        q = apply_joint_limits(q)

    return q, False


# ============================================================
# 12. Contact utilities
# ============================================================


def finger_contacts_cube(finger_body_id):
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1 = contact.geom1
        geom2 = contact.geom2

        if geom1 == cube_geom_id and model.geom_bodyid[geom2] == finger_body_id:
            return True

        if geom2 == cube_geom_id and model.geom_bodyid[geom1] == finger_body_id:
            return True

    return False


def cube_contacts_table():
    for i in range(data.ncon):
        contact = data.contact[i]

        pair = {contact.geom1, contact.geom2}

        if pair == {cube_geom_id, table_geom_id}:
            return True

    return False


# ============================================================
# 13. Let cube settle
# ============================================================

while data.time < 1.0:
    for i in range(7):
        data.ctrl[arm_actuator_ids[i]] = q_home[i]

    data.ctrl[gripper_actuator_id] = 255.0

    mujoco.mj_step(model, data)


# ============================================================
# 14. Cube position after settling
# ============================================================

cube_initial_position = data.xpos[cube_body_id].copy()


cube_initial_z = cube_initial_position[2]


print("===== Cube Initial Position =====")

print(np.round(cube_initial_position, 5))


# ============================================================
# 15. Generate task targets
# ============================================================

PREGRASP_HEIGHT = 0.12

APPROACH_OFFSET = 0.005

LIFT_HEIGHT = 0.15


pregrasp_position = cube_initial_position + np.array([0.0, 0.0, PREGRASP_HEIGHT])


approach_position = cube_initial_position + np.array([0.0, 0.0, APPROACH_OFFSET])


lift_position = approach_position + np.array([0.0, 0.0, LIFT_HEIGHT])


print("\nPregrasp =", np.round(pregrasp_position, 4))

print("Approach =", np.round(approach_position, 4))

print("Lift =", np.round(lift_position, 4))


# ============================================================
# 16. Solve IK
# ============================================================

print("\n===== Solve PREGRASP IK =====")

q_pregrasp, ok_pre = solve_pose_ik(q_home, pregrasp_position, target_rotation)


print("PREGRASP IK =", ok_pre)


print("\n===== Solve APPROACH IK =====")

q_approach, ok_app = solve_pose_ik(q_pregrasp, approach_position, target_rotation)


print("APPROACH IK =", ok_app)


print("\n===== Solve LIFT IK =====")

q_lift, ok_lift = solve_pose_ik(q_approach, lift_position, target_rotation)


print("LIFT IK =", ok_lift)


if not (ok_pre and ok_app and ok_lift):
    raise RuntimeError("IK failed. Stop before dynamics.")


# ============================================================
# 17. Put robot back Home
# ============================================================

for i in range(7):
    data.qpos[arm_qpos_adr[i]] = q_home[i]


for i in range(2):
    data.qpos[finger_qpos_adr[i]] = finger_home[i]


data.qvel[:] = 0.0


for i in range(7):
    data.ctrl[arm_actuator_ids[i]] = q_home[i]


data.ctrl[gripper_actuator_id] = 255.0


mujoco.mj_forward(model, data)


# ============================================================
# 18. State machine
# ============================================================

PREGRASP = "PREGRASP"
APPROACH = "APPROACH"
CLOSE = "CLOSE"
LIFT = "LIFT"
DONE = "DONE"
FAILED = "FAILED"


state = PREGRASP


pose_stable_counter = 0
contact_counter = 0
lift_success_counter = 0


POSE_STABLE_STEPS = 100
CONTACT_STABLE_STEPS = 50
LIFT_STABLE_STEPS = 100


POSITION_THRESHOLD = 0.008

ORIENTATION_THRESHOLD = np.deg2rad(2.0)


close_start_time = None


# ============================================================
# 19. Viewer
# ============================================================

simulation_duration = 20.0

start_time = data.time

step_count = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < start_time + simulation_duration:
        # ----------------------------------------------------
        # State -> target
        # ----------------------------------------------------

        if state == PREGRASP:
            q_target = q_pregrasp

            cartesian_target = pregrasp_position

            gripper_command = 255.0

        elif state == APPROACH:
            q_target = q_approach

            cartesian_target = approach_position

            gripper_command = 255.0

        elif state == CLOSE:
            q_target = q_approach

            cartesian_target = approach_position

            gripper_command = 0.0

        elif state == LIFT:
            q_target = q_lift

            cartesian_target = lift_position

            # Lift 时必须继续夹紧
            gripper_command = 0.0

        else:
            q_target = q_lift

            cartesian_target = lift_position

            gripper_command = 0.0

        # ----------------------------------------------------
        # Arm control
        # ----------------------------------------------------

        for i in range(7):
            data.ctrl[arm_actuator_ids[i]] = q_target[i]

        # ----------------------------------------------------
        # Gripper control
        # ----------------------------------------------------

        data.ctrl[gripper_actuator_id] = gripper_command

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        mujoco.mj_step(model, data)

        # ----------------------------------------------------
        # Current TCP Pose
        # ----------------------------------------------------

        tcp_position = data.site_xpos[tcp_site_id].copy()

        tcp_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()

        position_error = np.linalg.norm(cartesian_target - tcp_position)

        orientation_error = np.linalg.norm(
            rotation_error_vector(tcp_rotation, target_rotation)
        )

        # ----------------------------------------------------
        # Current Cube State
        # ----------------------------------------------------

        cube_position = data.xpos[cube_body_id].copy()

        cube_height_increase = cube_position[2] - cube_initial_z

        table_contact = cube_contacts_table()

        left_contact = finger_contacts_cube(left_finger_body_id)

        right_contact = finger_contacts_cube(right_finger_body_id)

        # ====================================================
        # PREGRASP
        # ====================================================

        if state == PREGRASP:
            if (
                position_error < POSITION_THRESHOLD
                and orientation_error < ORIENTATION_THRESHOLD
            ):
                pose_stable_counter += 1

            else:
                pose_stable_counter = 0

            if pose_stable_counter >= POSE_STABLE_STEPS:
                print("\n>>> PREGRASP reached")

                state = APPROACH

                pose_stable_counter = 0

        # ====================================================
        # APPROACH
        # ====================================================

        elif state == APPROACH:
            if (
                position_error < POSITION_THRESHOLD
                and orientation_error < ORIENTATION_THRESHOLD
            ):
                pose_stable_counter += 1

            else:
                pose_stable_counter = 0

            if pose_stable_counter >= POSE_STABLE_STEPS:
                print("\n>>> APPROACH reached")

                state = CLOSE

                close_start_time = data.time

                pose_stable_counter = 0

        # ====================================================
        # CLOSE
        # ====================================================

        elif state == CLOSE:
            if left_contact and right_contact:
                contact_counter += 1

            else:
                contact_counter = 0

            if contact_counter >= CONTACT_STABLE_STEPS:
                print("\n>>> Stable bilateral contact")

                print("Start lifting...")

                state = LIFT

                contact_counter = 0

            elif close_start_time is not None and data.time - close_start_time > 2.5:
                print("\n>>> FAILED: bilateral contact not established")

                state = FAILED

        # ====================================================
        # LIFT
        # ====================================================

        elif state == LIFT:
            # Cube 至少抬高 8 cm
            cube_is_lifted = cube_height_increase > 0.08

            # 已经离开桌面
            cube_left_table = not table_contact

            # Cube 没有明显远离 TCP
            relative_position = cube_position - tcp_position

            cube_near_tcp = np.linalg.norm(relative_position) < 0.08

            tcp_reached_lift = position_error < 0.015

            if (
                cube_is_lifted
                and cube_left_table
                and cube_near_tcp
                and tcp_reached_lift
            ):
                lift_success_counter += 1

            else:
                lift_success_counter = 0

            if lift_success_counter >= LIFT_STABLE_STEPS:
                print("\n>>> CUBE LIFT SUCCESS!")

                state = DONE

            # 如果正在 lift，但 Cube 又重新掉到桌面
            if table_contact and tcp_position[2] > approach_position[2] + 0.08:
                print("\n>>> FAILED: Cube slipped.")

                state = FAILED

        # ====================================================
        # Debug
        # ====================================================

        step_count += 1

        if step_count % 250 == 0:
            print(f"\nTime = {data.time:.2f}s")

            print("State =", state)

            print("TCP =", np.round(tcp_position, 4))

            print("Cube =", np.round(cube_position, 4))

            print("Cube lift =", round(cube_height_increase, 4), "m")

            print("Cube-table contact =", table_contact)

            print("Left contact =", left_contact)

            print("Right contact =", right_contact)

        # ====================================================
        # End
        # ====================================================

        if state in [DONE, FAILED]:
            # 保持一会儿方便 Viewer 观察
            for _ in range(500):
                if not viewer.is_running():
                    break

                for i in range(7):
                    data.ctrl[arm_actuator_ids[i]] = q_lift[i]

                data.ctrl[gripper_actuator_id] = 0.0

                mujoco.mj_step(model, data)

                viewer.sync()

                time.sleep(model.opt.timestep)

            break

        viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# 20. Result
# ============================================================

print("\n===== Final Result =====")

print("State =", state)

print("Initial cube z =", cube_initial_z)

print("Final cube z =", data.xpos[cube_body_id][2])

print("Height increase =", data.xpos[cube_body_id][2] - cube_initial_z, "m")
