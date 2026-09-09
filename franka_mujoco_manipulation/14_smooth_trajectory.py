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

# Real simulation data
data = mujoco.MjData(model)

# Separate data for IK calculation.
# Important: numerical IK must not overwrite the real simulation state.
ik_data = mujoco.MjData(model)


# ============================================================
# 2. Arm Joint Information
# ============================================================

arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

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

finger_joint_names = ["finger_joint1", "finger_joint2"]

finger_qpos_adr = []

for joint_name in finger_joint_names:

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    finger_qpos_adr.append(model.jnt_qposadr[joint_id])

finger_qpos_adr = np.array(finger_qpos_adr)


# ============================================================
# 4. Arm Actuators
# ============================================================

arm_actuator_ids = []

for i in range(1, 8):

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")

    arm_actuator_ids.append(actuator_id)

arm_actuator_ids = np.array(arm_actuator_ids)


# ============================================================
# 5. Gripper Actuator
# ============================================================

gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")


# ============================================================
# 6. Body / Joint / Geom / Site IDs
# ============================================================

cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")

cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")

right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

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
# 8. Basic Utilities
# ============================================================


def get_arm_q():

    return np.array([data.qpos[adr] for adr in arm_qpos_adr])


def get_tcp_pose():

    position = data.site_xpos[tcp_site_id].copy()

    rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()

    return position, rotation


def get_cube_pose():

    position = data.xpos[cube_body_id].copy()

    rotation = data.xmat[cube_body_id].reshape(3, 3).copy()

    return position, rotation


def make_transform(position, rotation):

    T = np.eye(4)

    T[:3, :3] = rotation
    T[:3, 3] = position

    return T


# ============================================================
# 9. Initialize Robot
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
# 10. TCP Grasp Orientation
#
# Use the home TCP orientation for top-down grasp.
# ============================================================

grasp_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()


# ============================================================
# 11. Rotation Error
# ============================================================


def rotation_error_vector(R_current, R_target):

    R_error = R_target @ R_current.T

    cos_theta = (np.trace(R_error) - 1.0) / 2.0

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if theta < 1e-8:

        return np.zeros(3)

    # This formula is sufficient here because target changes
    # are small and are not close to 180 degrees.
    axis = np.array([R_error[2, 1] - R_error[1, 2], R_error[0, 2] - R_error[2, 0], R_error[1, 0] - R_error[0, 1]])

    axis = axis / (2.0 * np.sin(theta))

    return theta * axis


# ============================================================
# 12. Joint Limits
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
# 13. 6D Numerical IK
#
# IMPORTANT:
# Use ik_data instead of data so IK does not teleport the
# real robot or disturb the grasped cube.
# ============================================================


def solve_pose_ik(q_init, target_position, target_rotation, max_iterations=1500, step_size=0.2):

    q = q_init.copy()

    # Copy the current real configuration as the background
    # state for the IK model.
    ik_data.qpos[:] = data.qpos[:]
    ik_data.qvel[:] = 0.0

    for iteration in range(max_iterations):

        for i in range(7):

            ik_data.qpos[arm_qpos_adr[i]] = q[i]

        mujoco.mj_forward(model, ik_data)

        current_position = ik_data.site_xpos[tcp_site_id].copy()

        current_rotation = ik_data.site_xmat[tcp_site_id].reshape(3, 3).copy()

        position_error_vector = target_position - current_position

        orientation_error_vector = rotation_error_vector(current_rotation, target_rotation)

        position_error = np.linalg.norm(position_error_vector)

        orientation_error = np.linalg.norm(orientation_error_vector)

        if position_error < 1e-4 and orientation_error < np.deg2rad(0.1):

            print(
                f"IK converged | "
                f"iter={iteration:4d} | "
                f"pos={position_error:.6f} m | "
                f"ori={np.rad2deg(orientation_error):.4f} deg"
            )

            return q.copy(), True

        jacp = np.zeros((3, model.nv))

        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, tcp_site_id)

        J_position = jacp[:, arm_dof_adr]

        J_rotation = jacr[:, arm_dof_adr]

        J = np.vstack([J_position, J_rotation])

        pose_error = np.concatenate([position_error_vector, orientation_error_vector])

        delta_q = np.linalg.pinv(J) @ pose_error

        # Limit each numerical IK increment.
        delta_q = np.clip(delta_q, -0.10, 0.10)

        q = q + step_size * delta_q

        q = apply_joint_limits(q)

        if iteration % 150 == 0:

            print(
                f"iter={iteration:4d} | "
                f"pos_err={position_error:.5f} m | "
                f"ori_err={np.rad2deg(orientation_error):.3f} deg"
            )

    return q.copy(), False


# ============================================================
# 14. Contact Utilities
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

        geom1 = contact.geom1
        geom2 = contact.geom2

        if (geom1 == cube_geom_id and geom2 == table_geom_id) or (geom2 == cube_geom_id and geom1 == table_geom_id):

            return True

    return False


# ============================================================
# 15. Smooth Joint Trajectory
#
# h(s) = 10 s^3 - 15 s^4 + 6 s^5
#
# h(0)=0, h(1)=1
# h'(0)=h'(1)=0
# h''(0)=h''(1)=0
# ============================================================


def quintic_smoothstep(s):

    s = np.clip(s, 0.0, 1.0)

    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


trajectory = {"active": False, "start_q": q_home.copy(), "goal_q": q_home.copy(), "start_time": 0.0, "duration": 1.0}


def start_trajectory(goal_q, duration):

    trajectory["start_q"] = get_arm_q().copy()

    trajectory["goal_q"] = goal_q.copy()

    trajectory["start_time"] = data.time

    trajectory["duration"] = duration

    trajectory["active"] = True


def stop_trajectory():

    trajectory["active"] = False


def trajectory_command():

    if not trajectory["active"]:

        return (get_arm_q(), True)

    elapsed_time = data.time - trajectory["start_time"]

    duration = trajectory["duration"]

    s = elapsed_time / duration

    h = quintic_smoothstep(s)

    q_command = trajectory["start_q"] + h * (trajectory["goal_q"] - trajectory["start_q"])

    finished = s >= 1.0

    return (q_command, finished)


# ============================================================
# 16. Let Cube Settle on Table
# ============================================================

SETTLE_TIME = 1.0

while data.time < SETTLE_TIME:

    for i in range(7):

        data.ctrl[arm_actuator_ids[i]] = q_home[i]

    data.ctrl[gripper_actuator_id] = 255.0

    mujoco.mj_step(model, data)


# ============================================================
# 17. Cube Initial Pose
# ============================================================

cube_initial_position, cube_initial_rotation = get_cube_pose()

cube_initial_z = cube_initial_position[2]

print("\n===== Cube Initial Pose =====")

print("Position =", np.round(cube_initial_position, 5))

print("Cube-table contact =", cube_contacts_table())


# ============================================================
# 18. Pick Targets
# ============================================================

PREGRASP_HEIGHT = 0.12
APPROACH_OFFSET = 0.005
LIFT_HEIGHT = 0.15

pregrasp_position = cube_initial_position + np.array([0.0, 0.0, PREGRASP_HEIGHT])

approach_position = cube_initial_position + np.array([0.0, 0.0, APPROACH_OFFSET])

lift_position = approach_position + np.array([0.0, 0.0, LIFT_HEIGHT])


# ============================================================
# 19. Desired Cube Place Pose
#
# We want the cube to return to an upright pose with the same
# orientation it had while resting on the table initially.
# ============================================================

place_cube_position = np.array([0.45, -0.15, cube_initial_z])

place_cube_rotation = cube_initial_rotation.copy()


# ============================================================
# 20. Solve Pick IK
# ============================================================

print("\n===== Solve PREGRASP IK =====")

q_pregrasp, ok_pre = solve_pose_ik(q_home, pregrasp_position, grasp_rotation)

print("PREGRASP IK =", ok_pre)


print("\n===== Solve APPROACH IK =====")

q_approach, ok_app = solve_pose_ik(q_pregrasp, approach_position, grasp_rotation)

print("APPROACH IK =", ok_app)


print("\n===== Solve LIFT IK =====")

q_lift, ok_lift = solve_pose_ik(q_approach, lift_position, grasp_rotation)

print("LIFT IK =", ok_lift)


if not (ok_pre and ok_app and ok_lift):

    raise RuntimeError("Pick-side IK failed.")


# ============================================================
# 21. Put Robot Back to Home
#
# Do NOT reset the entire data object because the cube has
# already settled physically on the table.
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
# 22. State Machine
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
# 23. State Parameters
# ============================================================

# Smooth trajectory durations
PREGRASP_DURATION = 3.5
APPROACH_DURATION = 3.0
LIFT_DURATION = 3.0

# The important one: move the grasped cube slowly.
MOVE_DURATION = 5.0

# Lower slowly to reduce table impact.
LOWER_DURATION = 4.0

RETREAT_DURATION = 3.0


POSE_STABLE_STEPS = 80
CONTACT_STABLE_STEPS = 50
LIFT_STABLE_STEPS = 60
LOWER_CONTACT_STABLE_STEPS = 20
SETTLE_STABLE_STEPS = 100
PLACE_STABLE_STEPS = 100


POSITION_THRESHOLD = 0.010

ORIENTATION_THRESHOLD = np.deg2rad(2.0)


pose_stable_counter = 0
contact_counter = 0
lift_success_counter = 0
lower_contact_counter = 0
settle_counter = 0
place_counter = 0


close_start_time = None
release_start_time = None


# Runtime placement targets.
# These are computed only AFTER the cube has actually been
# grasped, using the measured TCP-Cube relative transform.
place_tcp_position = None
place_tcp_rotation = None
place_above_position = None
lower_search_position = None
retreat_position = None

q_place_above = None
q_lower = None
q_retreat = None

q_settle = None
settle_tcp_position = None
settle_tcp_rotation = None


# ============================================================
# 24. Start First Smooth Trajectory
# ============================================================

start_trajectory(q_pregrasp, PREGRASP_DURATION)


# ============================================================
# 25. Main Simulation
# ============================================================

simulation_duration = 45.0

start_time = data.time

step_count = 0


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and data.time < start_time + simulation_duration:

        # ====================================================
        # A. State -> q_command / gripper command
        # ====================================================

        if state in [PREGRASP, APPROACH, LIFT, MOVE, LOWER, RETREAT]:

            q_command, trajectory_finished = trajectory_command()

        else:

            trajectory_finished = True

            if state == CLOSE:

                q_command = q_approach.copy()

            elif state in [SETTLE, RELEASE]:

                q_command = q_settle.copy()

            elif state == DONE:

                q_command = q_retreat.copy()

            else:

                q_command = get_arm_q()

        if state in [PREGRASP, APPROACH]:

            gripper_command = 255.0

        elif state in [CLOSE, LIFT, MOVE, LOWER, SETTLE]:

            gripper_command = 0.0

        else:

            gripper_command = 255.0

        # ====================================================
        # B. Control
        # ====================================================

        for i in range(7):

            data.ctrl[arm_actuator_ids[i]] = q_command[i]

        data.ctrl[gripper_actuator_id] = gripper_command

        # ====================================================
        # C. Physics
        # ====================================================

        mujoco.mj_step(model, data)

        # ====================================================
        # D. Read Current State
        # ====================================================

        tcp_position, tcp_rotation = get_tcp_pose()

        cube_position, cube_rotation = get_cube_pose()

        table_contact = cube_contacts_table()

        left_contact = finger_contacts_cube(left_finger_body_id)

        right_contact = finger_contacts_cube(right_finger_body_id)

        cube_height_increase = cube_position[2] - cube_initial_z

        # ====================================================
        # E. State-specific Cartesian reference
        # ====================================================

        if state == PREGRASP:

            cartesian_target = pregrasp_position

            rotation_target = grasp_rotation

        elif state in [APPROACH, CLOSE]:

            cartesian_target = approach_position

            rotation_target = grasp_rotation

        elif state == LIFT:

            cartesian_target = lift_position

            rotation_target = grasp_rotation

        elif state == MOVE:

            cartesian_target = place_above_position

            rotation_target = place_tcp_rotation

        elif state == LOWER:

            cartesian_target = lower_search_position

            rotation_target = place_tcp_rotation

        elif state in [SETTLE, RELEASE]:

            cartesian_target = settle_tcp_position

            rotation_target = settle_tcp_rotation

        elif state in [RETREAT, DONE]:

            cartesian_target = retreat_position

            rotation_target = place_tcp_rotation

        else:

            cartesian_target = tcp_position.copy()

            rotation_target = tcp_rotation.copy()

        position_error = np.linalg.norm(cartesian_target - tcp_position)

        orientation_error = np.linalg.norm(rotation_error_vector(tcp_rotation, rotation_target))

        # ====================================================
        # F. PREGRASP
        # ====================================================

        if state == PREGRASP:

            if (
                trajectory_finished
                and position_error < POSITION_THRESHOLD
                and orientation_error < ORIENTATION_THRESHOLD
            ):

                pose_stable_counter += 1

            else:

                pose_stable_counter = 0

            if pose_stable_counter >= POSE_STABLE_STEPS:

                print("\n>>> PREGRASP reached")

                state = APPROACH

                pose_stable_counter = 0

                start_trajectory(q_approach, APPROACH_DURATION)

        # ====================================================
        # G. APPROACH
        # ====================================================

        elif state == APPROACH:

            if (
                trajectory_finished
                and position_error < POSITION_THRESHOLD
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

                stop_trajectory()

        # ====================================================
        # H. CLOSE
        #
        # After bilateral contact becomes stable:
        #   1) Measure actual TCP-Cube transform.
        #   2) Compute placement TCP pose from desired cube pose.
        #   3) Solve MOVE/LOWER/RETREAT IK without touching the
        #      real simulation state.
        # ====================================================

        elif state == CLOSE:

            if left_contact and right_contact:

                contact_counter += 1

            else:

                contact_counter = 0

            if contact_counter >= CONTACT_STABLE_STEPS:

                print("\n>>> Stable bilateral contact")

                # ------------------------------------------------
                # Measure actual grasp transform
                # ------------------------------------------------

                T_world_tcp = make_transform(tcp_position, tcp_rotation)

                T_world_cube = make_transform(cube_position, cube_rotation)

                T_tcp_cube = np.linalg.inv(T_world_tcp) @ T_world_cube

                # ------------------------------------------------
                # Desired cube world pose
                # ------------------------------------------------

                T_world_cube_desired = make_transform(place_cube_position, place_cube_rotation)

                # T_W_C = T_W_TCP * T_TCP_C
                #
                # therefore:
                #
                # T_W_TCP_des =
                # T_W_C_des * inv(T_TCP_C)
                # ------------------------------------------------

                T_world_tcp_place = T_world_cube_desired @ np.linalg.inv(T_tcp_cube)

                place_tcp_position = T_world_tcp_place[:3, 3].copy()

                place_tcp_rotation = T_world_tcp_place[:3, :3].copy()

                # ------------------------------------------------
                # Move above target first
                # ------------------------------------------------

                place_above_position = place_tcp_position + np.array([0.0, 0.0, 0.15])

                # ------------------------------------------------
                # Lower slightly beyond nominal place pose.
                #
                # We DO NOT intend to reach this final point.
                # LOWER is interrupted as soon as the cube
                # contacts the table.
                # ------------------------------------------------

                LOWER_SEARCH_DEPTH = 0.010

                lower_search_position = place_tcp_position + np.array([0.0, 0.0, -LOWER_SEARCH_DEPTH])

                retreat_position = place_tcp_position + np.array([0.0, 0.0, 0.12])

                print("\n===== Measured Grasp =====")

                print("Actual TCP-Cube transform:")

                print(np.round(T_tcp_cube, 4))

                print("\nComputed place TCP position =", np.round(place_tcp_position, 4))

                # ------------------------------------------------
                # Solve placement-side IK
                # ------------------------------------------------

                print("\n===== Solve PLACE ABOVE IK =====")

                q_place_above, ok_place = solve_pose_ik(q_lift, place_above_position, place_tcp_rotation)

                print("PLACE ABOVE IK =", ok_place)

                print("\n===== Solve LOWER IK =====")

                q_lower, ok_lower = solve_pose_ik(q_place_above, lower_search_position, place_tcp_rotation)

                print("LOWER IK =", ok_lower)

                print("\n===== Solve RETREAT IK =====")

                q_retreat, ok_retreat = solve_pose_ik(q_lower, retreat_position, place_tcp_rotation)

                print("RETREAT IK =", ok_retreat)

                if not (ok_place and ok_lower and ok_retreat):

                    print("\n>>> FAILED: placement-side IK failed")

                    state = FAILED

                else:

                    print("\nStart smooth LIFT...")

                    state = LIFT

                    contact_counter = 0

                    start_trajectory(q_lift, LIFT_DURATION)

            elif close_start_time is not None and data.time - close_start_time > 3.0:

                print("\n>>> FAILED: bilateral contact not established")

                state = FAILED

        # ====================================================
        # I. LIFT
        # ====================================================

        elif state == LIFT:

            cube_is_lifted = cube_height_increase > 0.08

            cube_left_table = not table_contact

            cube_near_tcp = np.linalg.norm(cube_position - tcp_position) < 0.08

            if trajectory_finished and cube_is_lifted and cube_left_table and cube_near_tcp and position_error < 0.015:

                lift_success_counter += 1

            else:

                lift_success_counter = 0

            if lift_success_counter >= LIFT_STABLE_STEPS:

                print("\n>>> LIFT reached")

                print("Start smooth MOVE...")

                state = MOVE

                pose_stable_counter = 0

                start_trajectory(q_place_above, MOVE_DURATION)

            # Detect obvious slip.
            if table_contact and tcp_position[2] > approach_position[2] + 0.08:

                print("\n>>> FAILED: Cube slipped during LIFT")

                state = FAILED

        # ====================================================
        # J. MOVE
        # ====================================================

        elif state == MOVE:

            # If the cube falls while TCP is still high.
            if table_contact and tcp_position[2] > place_tcp_position[2] + 0.05:

                print("\n>>> FAILED: Cube dropped during MOVE")

                state = FAILED

            else:

                if (
                    trajectory_finished
                    and position_error < POSITION_THRESHOLD
                    and orientation_error < ORIENTATION_THRESHOLD
                ):

                    pose_stable_counter += 1

                else:

                    pose_stable_counter = 0

                if pose_stable_counter >= POSE_STABLE_STEPS:

                    print("\n>>> PLACE ABOVE reached")

                    print("Start slow LOWER...")

                    state = LOWER

                    pose_stable_counter = 0

                    start_trajectory(q_lower, LOWER_DURATION)

        # ====================================================
        # K. LOWER
        #
        # Slow quintic descent.
        # Stop immediately when cube-table contact becomes
        # stable instead of forcing q_lower to completion.
        # ====================================================

        elif state == LOWER:

            placement_xy_error = np.linalg.norm(cube_position[:2] - place_cube_position[:2])

            if table_contact and placement_xy_error < 0.03:

                lower_contact_counter += 1

            else:

                lower_contact_counter = 0

            if lower_contact_counter >= LOWER_CONTACT_STABLE_STEPS:

                print("\n>>> Cube contacts table")

                print("Placement XY error =", placement_xy_error, "m")

                # Interrupt downward trajectory immediately.
                stop_trajectory()

                q_settle = get_arm_q().copy()

                settle_tcp_position = tcp_position.copy()

                settle_tcp_rotation = tcp_rotation.copy()

                state = SETTLE

                lower_contact_counter = 0

        # ====================================================
        # L. SETTLE
        #
        # Keep holding the cube but stop descending.
        # Wait until cube linear/angular motion becomes small.
        # ====================================================

        elif state == SETTLE:

            cube_linear_velocity = data.qvel[cube_dof_adr : cube_dof_adr + 3]

            cube_angular_velocity = data.qvel[cube_dof_adr + 3 : cube_dof_adr + 6]

            linear_speed = np.linalg.norm(cube_linear_velocity)

            angular_speed = np.linalg.norm(cube_angular_velocity)

            placement_xy_error = np.linalg.norm(cube_position[:2] - place_cube_position[:2])

            if table_contact and placement_xy_error < 0.03 and linear_speed < 0.01 and angular_speed < 0.10:

                settle_counter += 1

            else:

                settle_counter = 0

            if settle_counter >= SETTLE_STABLE_STEPS:

                print("\n>>> Cube settled on table")

                print("Linear speed =", linear_speed)

                print("Angular speed =", angular_speed)

                print("Start RELEASE...")

                state = RELEASE

                release_start_time = data.time

                settle_counter = 0

        # ====================================================
        # M. RELEASE
        # ====================================================

        elif state == RELEASE:

            if release_start_time is not None and data.time - release_start_time > 1.0:

                cube_linear_velocity = data.qvel[cube_dof_adr : cube_dof_adr + 3]

                cube_angular_velocity = data.qvel[cube_dof_adr + 3 : cube_dof_adr + 6]

                linear_speed = np.linalg.norm(cube_linear_velocity)

                angular_speed = np.linalg.norm(cube_angular_velocity)

                placement_xy_error = np.linalg.norm(cube_position[:2] - place_cube_position[:2])

                if table_contact and placement_xy_error < 0.03 and linear_speed < 0.015 and angular_speed < 0.15:

                    place_counter += 1

                else:

                    place_counter = 0

                if place_counter >= PLACE_STABLE_STEPS:

                    print("\n>>> CUBE PLACED SUCCESSFULLY")

                    print("Start smooth RETREAT...")

                    state = RETREAT

                    pose_stable_counter = 0

                    start_trajectory(q_retreat, RETREAT_DURATION)

            if release_start_time is not None and data.time - release_start_time > 4.0 and state == RELEASE:

                print("\n>>> FAILED: Cube placement failed")

                state = FAILED

        # ====================================================
        # N. RETREAT
        # ====================================================

        elif state == RETREAT:

            if (
                trajectory_finished
                and position_error < POSITION_THRESHOLD
                and orientation_error < ORIENTATION_THRESHOLD
            ):

                pose_stable_counter += 1

            else:

                pose_stable_counter = 0

            if pose_stable_counter >= POSE_STABLE_STEPS:

                print("\n>>> RETREAT reached")

                print("\n>>> SMOOTH PICK AND PLACE DONE!")

                state = DONE

                stop_trajectory()

        # ====================================================
        # O. Debug
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
        # P. DONE / FAILED
        # ====================================================

        if state in [DONE, FAILED]:

            # Keep the scene visible for a short time.
            for _ in range(500):

                if not viewer.is_running():

                    break

                if state == DONE:

                    hold_q = q_retreat

                    hold_gripper = 255.0

                else:

                    hold_q = get_arm_q()

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
# 26. Final Result
# ============================================================

final_cube_position, final_cube_rotation = get_cube_pose()

placement_error_xy = np.linalg.norm(final_cube_position[:2] - place_cube_position[:2])

cube_orientation_error = np.linalg.norm(rotation_error_vector(final_cube_rotation, place_cube_rotation))


print("\n======================================")

print("       SMOOTH PICK AND PLACE RESULT")

print("======================================")

print("State =", state)

print("\nInitial Cube Position:")

print(np.round(cube_initial_position, 4))

print("\nTarget Cube Position:")

print(np.round(place_cube_position, 4))

print("\nFinal Cube Position:")

print(np.round(final_cube_position, 4))

print("\nPlacement XY Error =", round(placement_error_xy, 5), "m")

print("Cube orientation error =", round(np.rad2deg(cube_orientation_error), 3), "deg")

print("Cube on table =", cube_contacts_table())
