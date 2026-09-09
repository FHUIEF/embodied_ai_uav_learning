import os
import time

import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# 1. Load Scene
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(current_dir, "assets", "franka_pick_cube", "scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)


# ============================================================
# 2. Arm Joint Information
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


for joint_name in arm_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    arm_joint_ids.append(joint_id)

    arm_qpos_adr.append(model.jnt_qposadr[joint_id])

    arm_dof_adr.append(model.jnt_dofadr[joint_id])


arm_joint_ids = np.array(arm_joint_ids)

arm_qpos_adr = np.array(arm_qpos_adr)

arm_dof_adr = np.array(arm_dof_adr)


# ============================================================
# 3. Finger Joint Information
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
# 4. Arm Actuator IDs
# ============================================================

arm_actuator_ids = []


for i in range(1, 8):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")

    arm_actuator_ids.append(actuator_id)


arm_actuator_ids = np.array(arm_actuator_ids)


# ============================================================
# 5. Gripper Actuator
# ============================================================

gripper_actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
)


# ============================================================
# 6. Body / Geom / Site IDs
# ============================================================

cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")


cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")


table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")


left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")


right_finger_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
)


tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")


# Cube freejoint，用于读取 Cube 的 6D 速度
cube_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
)

cube_dof_adr = model.jnt_dofadr[cube_joint_id]


print("===== Model Check =====")

print("nq =", model.nq)

print("nv =", model.nv)

print("nu =", model.nu)

print("nsite =", model.nsite)

print("tcp_site_id =", tcp_site_id)


# ============================================================
# 7. Franka Home Configuration
# ============================================================

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, -0.7853])


finger_home = np.array([0.04, 0.04])


# ============================================================
# 8. Initialize Robot
# ============================================================

for i in range(7):
    data.qpos[arm_qpos_adr[i]] = q_home[i]


for i in range(2):
    data.qpos[finger_qpos_adr[i]] = finger_home[i]


data.qvel[:] = 0.0


for i in range(7):
    data.ctrl[arm_actuator_ids[i]] = q_home[i]


# Gripper open
data.ctrl[gripper_actuator_id] = 255.0


mujoco.mj_forward(model, data)


# ============================================================
# 9. TCP Target Orientation
#
# 使用 Home 时 TCP 的姿态作为整个抓取过程的目标姿态
# ============================================================

target_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()


print("\n===== TCP Home Pose =====")

print("TCP position:")

print(np.round(data.site_xpos[tcp_site_id], 4))


print("\nTCP rotation:")

print(np.round(target_rotation, 4))


# ============================================================
# 10. Rotation Error
# ============================================================


def rotation_error_vector(R_current, R_target):
    # 当前姿态 -> 目标姿态
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
# 11. Joint Limit Function
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
# 12. 6D Numerical IK for TCP
# ============================================================


def solve_pose_ik(
    q_init, target_position, target_rotation, max_iterations=1200, step_size=0.2
):
    q = q_init.copy()

    for iteration in range(max_iterations):
        # ----------------------------------------------------
        # Write q into MuJoCo
        # ----------------------------------------------------

        for i in range(7):
            data.qpos[arm_qpos_adr[i]] = q[i]

        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        # ----------------------------------------------------
        # Current TCP Pose
        # ----------------------------------------------------

        current_position = data.site_xpos[tcp_site_id].copy()

        current_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()

        # ----------------------------------------------------
        # Position Error
        # ----------------------------------------------------

        position_error_vector = target_position - current_position

        position_error = np.linalg.norm(position_error_vector)

        # ----------------------------------------------------
        # Orientation Error
        # ----------------------------------------------------

        orientation_error_vector = rotation_error_vector(
            current_rotation, target_rotation
        )

        orientation_error = np.linalg.norm(orientation_error_vector)

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        if position_error < 1e-4 and orientation_error < np.deg2rad(0.1):
            print(
                f"IK converged | "
                f"iter={iteration:4d} | "
                f"pos="
                f"{position_error:.6f} m | "
                f"ori="
                f"{np.rad2deg(orientation_error):.4f} deg"
            )

            return (q.copy(), True)

        # ----------------------------------------------------
        # TCP Jacobian
        # ----------------------------------------------------

        jacp = np.zeros((3, model.nv))

        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, jacr, tcp_site_id)

        J_position = jacp[:, arm_dof_adr]

        J_rotation = jacr[:, arm_dof_adr]

        J = np.vstack([J_position, J_rotation])

        pose_error = np.concatenate([position_error_vector, orientation_error_vector])

        # ----------------------------------------------------
        # Pseudoinverse IK
        # ----------------------------------------------------

        delta_q = np.linalg.pinv(J) @ pose_error

        # 防止某次 IK 更新过大
        delta_q = np.clip(delta_q, -0.10, 0.10)

        q = q + step_size * delta_q

        q = apply_joint_limits(q)

        if iteration % 100 == 0:
            print(
                f"iter={iteration:4d} | "
                f"pos_err="
                f"{position_error:.5f} m | "
                f"ori_err="
                f"{np.rad2deg(orientation_error):.3f} deg"
            )

    return (q.copy(), False)


# ============================================================
# 13. Contact Detection
# ============================================================


def finger_contacts_cube(finger_body_id):
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1 = contact.geom1
        geom2 = contact.geom2

        # cube is geom1
        if geom1 == cube_geom_id and model.geom_bodyid[geom2] == finger_body_id:
            return True

        # cube is geom2
        if geom2 == cube_geom_id and model.geom_bodyid[geom1] == finger_body_id:
            return True

    return False


# ============================================================
# 14. Cube-Table Contact
# ============================================================


def cube_contacts_table():
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1 = contact.geom1
        geom2 = contact.geom2

        if (geom1 == cube_geom_id and geom2 == table_geom_id) or (
            geom2 == cube_geom_id and geom1 == table_geom_id
        ):
            return True

    return False


# ============================================================
# 15. Let Cube Settle on Table
# ============================================================

SETTLE_TIME = 1.0


while data.time < SETTLE_TIME:
    for i in range(7):
        data.ctrl[arm_actuator_ids[i]] = q_home[i]

    data.ctrl[gripper_actuator_id] = 255.0

    mujoco.mj_step(model, data)


# ============================================================
# 16. Read Actual Cube Position
# ============================================================

cube_initial_position = data.xpos[cube_body_id].copy()


cube_initial_z = cube_initial_position[2]


print("\n===== Cube Initial Position =====")

print(np.round(cube_initial_position, 5))


print("Cube-table contact =", cube_contacts_table())


# ============================================================
# 17. Generate Pick-and-Place Targets
# ============================================================

PREGRASP_HEIGHT = 0.12

APPROACH_OFFSET = 0.005

LIFT_HEIGHT = 0.15

RETREAT_HEIGHT = 0.12


# ------------------------------------------------------------
# Pick: Pregrasp
# ------------------------------------------------------------

pregrasp_position = cube_initial_position + np.array([0.0, 0.0, PREGRASP_HEIGHT])


# ------------------------------------------------------------
# Pick: Approach
# TCP 稍微高于 Cube 中心
# ------------------------------------------------------------

approach_position = cube_initial_position + np.array([0.0, 0.0, APPROACH_OFFSET])


# ------------------------------------------------------------
# Pick: Lift
# ------------------------------------------------------------

lift_position = approach_position + np.array([0.0, 0.0, LIFT_HEIGHT])


# ============================================================
# Desired Cube Placement Position
# ============================================================

place_cube_position = np.array([0.45, -0.15, cube_initial_z])


# ------------------------------------------------------------
# TCP position when Cube is placed
# ------------------------------------------------------------

place_tcp_position = place_cube_position + np.array([0.0, 0.0, APPROACH_OFFSET])


# ------------------------------------------------------------
# Move above place target
# ------------------------------------------------------------

place_above_position = place_tcp_position + np.array([0.0, 0.0, LIFT_HEIGHT])


# ------------------------------------------------------------
# Lower
# ------------------------------------------------------------

lower_position = place_tcp_position.copy()


# ------------------------------------------------------------
# Retreat after release
# ------------------------------------------------------------

retreat_position = place_tcp_position + np.array([0.0, 0.0, RETREAT_HEIGHT])


print("\n===== Task Targets =====")

print("Pregrasp =", np.round(pregrasp_position, 4))

print("Approach =", np.round(approach_position, 4))

print("Lift =", np.round(lift_position, 4))

print("Place Cube Target =", np.round(place_cube_position, 4))

print("Place Above =", np.round(place_above_position, 4))

print("Lower =", np.round(lower_position, 4))

print("Retreat =", np.round(retreat_position, 4))


# ============================================================
# 18. Solve PREGRASP IK
# ============================================================

print("\n===== Solve PREGRASP IK =====")


q_pregrasp, ok_pre = solve_pose_ik(q_home, pregrasp_position, target_rotation)


print("PREGRASP IK =", ok_pre)


# ============================================================
# 19. Solve APPROACH IK
# ============================================================

print("\n===== Solve APPROACH IK =====")


q_approach, ok_app = solve_pose_ik(q_pregrasp, approach_position, target_rotation)


print("APPROACH IK =", ok_app)


# ============================================================
# 20. Solve LIFT IK
# ============================================================

print("\n===== Solve LIFT IK =====")


q_lift, ok_lift = solve_pose_ik(q_approach, lift_position, target_rotation)


print("LIFT IK =", ok_lift)


# ============================================================
# 21. Solve PLACE ABOVE IK
# ============================================================

print("\n===== Solve PLACE ABOVE IK =====")


q_place_above, ok_place = solve_pose_ik(q_lift, place_above_position, target_rotation)


print("PLACE ABOVE IK =", ok_place)


# ============================================================
# 22. Solve LOWER IK
# ============================================================

print("\n===== Solve LOWER IK =====")


q_lower, ok_lower = solve_pose_ik(q_place_above, lower_position, target_rotation)


print("LOWER IK =", ok_lower)


# ============================================================
# 23. Solve RETREAT IK
# ============================================================

print("\n===== Solve RETREAT IK =====")


q_retreat, ok_retreat = solve_pose_ik(q_lower, retreat_position, target_rotation)


print("RETREAT IK =", ok_retreat)


# ============================================================
# 24. Check All IK
# ============================================================

if not (ok_pre and ok_app and ok_lift and ok_place and ok_lower and ok_retreat):
    raise RuntimeError("One or more IK targets failed.")


# ============================================================
# 25. Put Robot Back to Home
#
# 不 reset 整个 data，
# 否则 Cube 会重新回到 XML 初始位置
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
# 26. State Machine
# ============================================================

PREGRASP = "PREGRASP"

APPROACH = "APPROACH"

CLOSE = "CLOSE"

LIFT = "LIFT"

MOVE = "MOVE"

LOWER = "LOWER"

SETTLE = "SETTLE"

RELEASE = "RELEASE"

RETREAT = "RETREAT"

DONE = "DONE"

FAILED = "FAILED"


state = PREGRASP


# ============================================================
# 27. State Machine Parameters
# ============================================================

pose_stable_counter = 0

contact_counter = 0

lift_success_counter = 0

lower_contact_counter = 0

settle_counter = 0

place_counter = 0


POSE_STABLE_STEPS = 100

CONTACT_STABLE_STEPS = 50

LIFT_STABLE_STEPS = 100

LOWER_CONTACT_STABLE_STEPS = 20

SETTLE_STABLE_STEPS = 100

PLACE_STABLE_STEPS = 100


POSITION_THRESHOLD = 0.008

ORIENTATION_THRESHOLD = np.deg2rad(2.0)


close_start_time = None

settle_start_time = None

release_start_time = None


# SETTLE 阶段保持 Cube 刚接触桌面时的真实机械臂关节角。
# 先用 q_lower 初始化，真正进入 SETTLE 时会被当前实际 q 覆盖。
q_settle = q_lower.copy()


# ============================================================
# 28. Simulation
# ============================================================

simulation_duration = 35.0

start_time = data.time

step_count = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < start_time + simulation_duration:
        # ====================================================
        # A. State -> Target
        # ====================================================

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

            gripper_command = 0.0

        elif state == MOVE:
            q_target = q_place_above

            cartesian_target = place_above_position

            gripper_command = 0.0

        elif state == LOWER:
            q_target = q_lower

            cartesian_target = lower_position

            gripper_command = 0.0

        elif state == SETTLE:
            # Cube 刚接触桌面后，不再继续追 q_lower 向下压。
            # 保持接触瞬间的真实关节位置，并继续夹紧。
            q_target = q_settle

            cartesian_target = lower_position

            gripper_command = 0.0

        elif state == RELEASE:
            # 松手时也保持 q_settle，避免重新跳回 q_lower 再次压 Cube。
            q_target = q_settle

            cartesian_target = lower_position

            gripper_command = 255.0

        elif state == RETREAT:
            q_target = q_retreat

            cartesian_target = retreat_position

            gripper_command = 255.0

        else:
            q_target = q_retreat

            cartesian_target = retreat_position

            gripper_command = 255.0

        # ====================================================
        # B. Arm Control
        # ====================================================

        for i in range(7):
            data.ctrl[arm_actuator_ids[i]] = q_target[i]

        # ====================================================
        # C. Gripper Control
        # ====================================================

        data.ctrl[gripper_actuator_id] = gripper_command

        # ====================================================
        # D. Physics
        # ====================================================

        mujoco.mj_step(model, data)

        # ====================================================
        # E. Read TCP Pose
        # ====================================================

        tcp_position = data.site_xpos[tcp_site_id].copy()

        tcp_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()

        position_error = np.linalg.norm(cartesian_target - tcp_position)

        orientation_error = np.linalg.norm(
            rotation_error_vector(tcp_rotation, target_rotation)
        )

        # ====================================================
        # F. Read Cube State
        # ====================================================

        cube_position = data.xpos[cube_body_id].copy()

        cube_height_increase = cube_position[2] - cube_initial_z

        table_contact = cube_contacts_table()

        left_contact = finger_contacts_cube(left_finger_body_id)

        right_contact = finger_contacts_cube(right_finger_body_id)

        # ====================================================
        # G. PREGRASP
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
        # H. APPROACH
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
        # I. CLOSE
        # ====================================================

        elif state == CLOSE:
            if left_contact and right_contact:
                contact_counter += 1

            else:
                contact_counter = 0

            if contact_counter >= CONTACT_STABLE_STEPS:
                print("\n>>> Stable bilateral contact")

                print("Start LIFT...")

                state = LIFT

                contact_counter = 0

            elif close_start_time is not None and data.time - close_start_time > 2.5:
                print("\n>>> FAILED: bilateral contact not established")

                state = FAILED

        # ====================================================
        # J. LIFT
        # ====================================================

        elif state == LIFT:
            cube_is_lifted = cube_height_increase > 0.08

            cube_left_table = not table_contact

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
                print("\n>>> LIFT reached")

                print("Start MOVE...")

                state = MOVE

                pose_stable_counter = 0

            # 如果 TCP 已经明显上升，
            # 但 Cube 又掉回桌面
            if table_contact and tcp_position[2] > approach_position[2] + 0.08:
                print("\n>>> FAILED: Cube slipped during LIFT")

                state = FAILED

        # ====================================================
        # K. MOVE
        # ====================================================

        elif state == MOVE:
            if (
                position_error < POSITION_THRESHOLD
                and orientation_error < ORIENTATION_THRESHOLD
            ):
                pose_stable_counter += 1

            else:
                pose_stable_counter = 0

            # 搬运过程中 Cube 掉回桌面
            if table_contact and tcp_position[2] > lower_position[2] + 0.05:
                print("\n>>> FAILED: Cube dropped during MOVE")

                state = FAILED

            elif pose_stable_counter >= POSE_STABLE_STEPS:
                print("\n>>> PLACE ABOVE reached")

                print("Start LOWER...")

                state = LOWER

                pose_stable_counter = 0

        # ====================================================
        # L. LOWER
        # ====================================================

        elif state == LOWER:

            # Cube 当前 XY 距离放置目标的误差
            placement_xy_error = np.linalg.norm(
                cube_position[:2] - place_cube_position[:2]
            )

            # Cube 已接触桌面，并且水平位置已经在目标区域附近
            if table_contact and placement_xy_error < 0.03:
                lower_contact_counter += 1
            else:
                lower_contact_counter = 0

            # 一旦 Cube 稳定接触桌面，就停止继续向 q_lower 压下去。
            # 保存此刻机械臂的真实关节角，转入 SETTLE。
            if lower_contact_counter >= LOWER_CONTACT_STABLE_STEPS:
                print("\n>>> Cube contacts table")
                print(
                    "Placement XY error =",
                    placement_xy_error,
                    "m",
                )

                q_settle = np.array(
                    [data.qpos[adr] for adr in arm_qpos_adr]
                )

                settle_start_time = data.time
                settle_counter = 0
                lower_contact_counter = 0

                print("Hold contact pose and start SETTLE...")

                state = SETTLE

        # ====================================================
        # M. SETTLE
        # ====================================================

        elif state == SETTLE:

            # freejoint 的 qvel 顺序：
            # [vx, vy, vz, wx, wy, wz]
            cube_linear_velocity = data.qvel[
                cube_dof_adr : cube_dof_adr + 3
            ].copy()

            cube_angular_velocity = data.qvel[
                cube_dof_adr + 3 : cube_dof_adr + 6
            ].copy()

            linear_speed = np.linalg.norm(
                cube_linear_velocity
            )

            angular_speed = np.linalg.norm(
                cube_angular_velocity
            )

            placement_xy_error = np.linalg.norm(
                cube_position[:2] - place_cube_position[:2]
            )

            # “平稳”不再只看 table_contact：
            # 还要求 Cube 的线速度和角速度都已经很小。
            if (
                table_contact
                and placement_xy_error < 0.03
                and linear_speed < 0.01
                and angular_speed < 0.10
            ):
                settle_counter += 1
            else:
                settle_counter = 0

            if settle_counter >= SETTLE_STABLE_STEPS:
                print("\n>>> Cube settled on table")
                print("Linear speed =", linear_speed, "m/s")
                print("Angular speed =", angular_speed, "rad/s")
                print("Start RELEASE...")

                state = RELEASE
                release_start_time = data.time
                settle_counter = 0

            # 如果长时间都无法稳定，说明 Cube 可能被夹持得过于倾斜，
            # 或 LOWER / 抓取相对位姿仍然存在问题。
            elif (
                settle_start_time is not None
                and data.time - settle_start_time > 3.0
            ):
                print(
                    "\n>>> FAILED: Cube could not settle stably on table"
                )
                print("Linear speed =", linear_speed, "m/s")
                print("Angular speed =", angular_speed, "rad/s")

                state = FAILED

        # ====================================================
        # N. RELEASE
        # ====================================================

        elif state == RELEASE:
            # 给夹爪和 Cube 一段时间
            if release_start_time is not None and data.time - release_start_time > 1.0:
                cube_position_now = data.xpos[cube_body_id].copy()

                xy_error = np.linalg.norm(
                    cube_position_now[:2] - place_cube_position[:2]
                )

                placed_on_table = cube_contacts_table()

                if xy_error < 0.04 and placed_on_table:
                    place_counter += 1

                else:
                    place_counter = 0

                if place_counter >= PLACE_STABLE_STEPS:
                    print("\n>>> CUBE PLACED SUCCESSFULLY")

                    print("Start RETREAT...")

                    state = RETREAT

                    pose_stable_counter = 0

            # 长时间没有成功放置
            if (
                release_start_time is not None
                and data.time - release_start_time > 3.0
                and state == RELEASE
            ):
                print("\n>>> FAILED: Cube placement failed")

                state = FAILED

        # ====================================================
        # O. RETREAT
        # ====================================================

        elif state == RETREAT:
            if (
                position_error < POSITION_THRESHOLD
                and orientation_error < ORIENTATION_THRESHOLD
            ):
                pose_stable_counter += 1

            else:
                pose_stable_counter = 0

            if pose_stable_counter >= POSE_STABLE_STEPS:
                print("\n>>> RETREAT reached")

                print("\n>>> PICK AND PLACE DONE!")

                state = DONE

        # ====================================================
        # P. Debug
        # ====================================================

        step_count += 1

        if step_count % 250 == 0:
            finger_q = np.array([data.qpos[adr] for adr in finger_qpos_adr])

            gripper_width = finger_q[0] + finger_q[1]

            print(f"\nTime = {data.time:.2f}s")

            print("State =", state)

            print("TCP =", np.round(tcp_position, 4))

            print("Cube =", np.round(cube_position, 4))

            print("Position error =", round(position_error, 5), "m")

            print("Orientation error =", round(np.rad2deg(orientation_error), 3), "deg")

            print("Cube lift =", round(cube_height_increase, 4), "m")

            print("Gripper width =", round(gripper_width, 5), "m")

            print("Cube-table contact =", table_contact)

            print("Left contact =", left_contact)

            print("Right contact =", right_contact)

        # ====================================================
        # Q. DONE / FAILED
        # ====================================================

        if state in [DONE, FAILED]:
            # 停留一段时间方便观察
            for _ in range(500):
                if not viewer.is_running():
                    break

                if state == DONE:
                    hold_q = q_retreat

                    hold_gripper = 255.0

                else:
                    hold_q = q_target

                    hold_gripper = gripper_command

                for i in range(7):
                    data.ctrl[arm_actuator_ids[i]] = hold_q[i]

                data.ctrl[gripper_actuator_id] = hold_gripper

                mujoco.mj_step(model, data)

                viewer.sync()

                time.sleep(model.opt.timestep)

            break

        viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# 29. Final Result
# ============================================================

final_cube_position = data.xpos[cube_body_id].copy()


placement_error_xy = np.linalg.norm(final_cube_position[:2] - place_cube_position[:2])


print("\n======================================")

print("        PICK AND PLACE RESULT")

print("======================================")


print("State =", state)


print("\nInitial Cube Position:")

print(np.round(cube_initial_position, 4))


print("\nTarget Cube Position:")

print(np.round(place_cube_position, 4))


print("\nFinal Cube Position:")

print(np.round(final_cube_position, 4))


print("\nPlacement XY Error =", round(placement_error_xy, 5), "m")


print("Cube on table =", cube_contacts_table())
