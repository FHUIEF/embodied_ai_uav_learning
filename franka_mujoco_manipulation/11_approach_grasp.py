import os
import time

import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# 1. 加载场景
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


for joint_name in arm_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

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


for joint_name in finger_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

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
# 5. Body / Geom / Site IDs
# ============================================================

cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")


left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")

right_finger_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
)


tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")


print("model.nsite =", model.nsite)

print("tcp_site_id =", tcp_site_id)


# ============================================================
# 6. Home configuration
# ============================================================

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, -0.7853])


finger_home = np.array([0.04, 0.04])


# ============================================================
# 7. 初始化机器人
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
# 8. TCP Home Orientation
# ============================================================

tcp_home_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()


print("\n===== TCP Home Pose =====")

print("TCP position:")

print(np.round(data.site_xpos[tcp_site_id], 4))

print("TCP rotation:")

print(np.round(tcp_home_rotation, 4))


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

    # theta 接近 pi 时这套公式数值性较差，
    # 当前任务姿态变化很小，因此够用。
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
# 11. 6D TCP Numerical IK
# ============================================================


def solve_pose_ik(
    q_init, target_position, target_rotation, max_iterations=1000, step_size=0.2
):
    q = q_init.copy()

    for iteration in range(max_iterations):
        # ----------------------------------------------------
        # 当前 q 写入模型
        # ----------------------------------------------------

        for i in range(7):
            data.qpos[arm_qpos_adr[i]] = q[i]

        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        # ----------------------------------------------------
        # TCP 当前 Pose
        # ----------------------------------------------------

        current_position = data.site_xpos[tcp_site_id].copy()

        current_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        position_error_vector = target_position - current_position

        orientation_error_vector = rotation_error_vector(
            current_rotation, target_rotation
        )

        position_error = np.linalg.norm(position_error_vector)

        orientation_error = np.linalg.norm(orientation_error_vector)

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        if position_error < 1e-4 and orientation_error < np.deg2rad(0.1):
            print(
                f"IK converged: "
                f"iter={iteration}, "
                f"pos={position_error:.6f} m, "
                f"ori="
                f"{np.rad2deg(orientation_error):.4f} deg"
            )

            return q, True

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
        # Pseudoinverse
        # ----------------------------------------------------

        delta_q = np.linalg.pinv(J) @ pose_error

        # 避免单次 IK 跳得过大
        delta_q = np.clip(delta_q, -0.10, 0.10)

        q = q + step_size * delta_q

        q = apply_joint_limits(q)

        if iteration % 50 == 0:
            print(
                f"iter={iteration:4d} | "
                f"pos_err={position_error:.5f} m | "
                f"ori_err="
                f"{np.rad2deg(orientation_error):.3f} deg"
            )

    return q, False


# ============================================================
# 12. 先让 Cube 稳定
# ============================================================

while data.time < 1.0:
    for i in range(7):
        data.ctrl[arm_actuator_ids[i]] = q_home[i]

    data.ctrl[gripper_actuator_id] = 255.0

    mujoco.mj_step(model, data)


# ============================================================
# 13. Cube 实际位置
# ============================================================

cube_position = data.xpos[cube_body_id].copy()


print("\n===== Cube Position =====")

print(np.round(cube_position, 5))


# ============================================================
# 14. 定义抓取 Pose
# ============================================================

PREGRASP_HEIGHT = 0.12

APPROACH_Z_OFFSET = 0.005


pregrasp_position = cube_position + np.array([0.0, 0.0, PREGRASP_HEIGHT])


approach_position = cube_position + np.array([0.0, 0.0, APPROACH_Z_OFFSET])


target_rotation = tcp_home_rotation.copy()


print("\nPregrasp position:")

print(np.round(pregrasp_position, 5))


print("Approach position:")

print(np.round(approach_position, 5))


# ============================================================
# 15. 求 PREGRASP IK
# ============================================================

print("\n===== Solve PREGRASP IK =====")


q_pregrasp, pregrasp_ik_success = solve_pose_ik(
    q_home, pregrasp_position, target_rotation
)


print("PREGRASP IK success =", pregrasp_ik_success)


# ============================================================
# 16. 求 APPROACH IK
#
# 从 q_pregrasp 开始求，比从 home 重新求更合理
# ============================================================

print("\n===== Solve APPROACH IK =====")


q_approach, approach_ik_success = solve_pose_ik(
    q_pregrasp, approach_position, target_rotation
)


print("APPROACH IK success =", approach_ik_success)


# ============================================================
# 17. Contact detection
# ============================================================


def finger_contacts_cube(finger_body_id):
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1 = contact.geom1
        geom2 = contact.geom2

        # Cube 是 geom1
        if geom1 == cube_geom_id and model.geom_bodyid[geom2] == finger_body_id:
            return True

        # Cube 是 geom2
        if geom2 == cube_geom_id and model.geom_bodyid[geom1] == finger_body_id:
            return True

    return False


# ============================================================
# 18. 将 Arm 放回 Home
#
# 注意：不重置整个 data，
# 否则 Cube 又会回到 XML 初始位置。
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
# 19. State Machine
# ============================================================

STATE_PREGRASP = "PREGRASP"

STATE_APPROACH = "APPROACH"

STATE_CLOSE = "CLOSE"

STATE_GRASPED = "GRASPED"

STATE_FAILED = "FAILED"


state = STATE_PREGRASP


stable_counter = 0

contact_counter = 0


POSE_STABLE_STEPS = 100

CONTACT_STABLE_STEPS = 50


position_threshold = 0.008

orientation_threshold = np.deg2rad(2.0)


close_start_time = None


# ============================================================
# 20. Viewer
# ============================================================

step_count = 0

simulation_duration = 15.0

start_time = data.time


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < start_time + simulation_duration:
        # ====================================================
        # A. 根据状态选择目标
        # ====================================================

        if state == STATE_PREGRASP:
            q_target = q_pregrasp

            target_position = pregrasp_position

            gripper_command = 255.0

        elif state == STATE_APPROACH:
            q_target = q_approach

            target_position = approach_position

            gripper_command = 255.0

        elif state == STATE_CLOSE:
            q_target = q_approach

            target_position = approach_position

            gripper_command = 0.0

        else:
            q_target = q_approach

            target_position = approach_position

            gripper_command = 0.0

        # ====================================================
        # B. Arm control
        # ====================================================

        for i in range(7):
            data.ctrl[arm_actuator_ids[i]] = q_target[i]

        # ====================================================
        # C. Gripper
        # ====================================================

        data.ctrl[gripper_actuator_id] = gripper_command

        # ====================================================
        # D. Dynamics
        # ====================================================

        mujoco.mj_step(model, data)

        # ====================================================
        # E. TCP current pose
        # ====================================================

        tcp_position = data.site_xpos[tcp_site_id].copy()

        tcp_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()

        position_error = np.linalg.norm(target_position - tcp_position)

        orientation_error = np.linalg.norm(
            rotation_error_vector(tcp_rotation, target_rotation)
        )

        # ====================================================
        # F. PREGRASP / APPROACH 状态转换
        # ====================================================

        if state in [STATE_PREGRASP, STATE_APPROACH]:
            if (
                position_error < position_threshold
                and orientation_error < orientation_threshold
            ):
                stable_counter += 1

            else:
                stable_counter = 0

            if stable_counter >= POSE_STABLE_STEPS:
                if state == STATE_PREGRASP:
                    print("\n>>> PREGRASP reached")

                    state = STATE_APPROACH

                    stable_counter = 0

                elif state == STATE_APPROACH:
                    print("\n>>> APPROACH reached")

                    state = STATE_CLOSE

                    stable_counter = 0

                    close_start_time = data.time

        # ====================================================
        # G. CLOSE + Contact
        # ====================================================

        if state == STATE_CLOSE:
            left_contact = finger_contacts_cube(left_finger_body_id)

            right_contact = finger_contacts_cube(right_finger_body_id)

            if left_contact and right_contact:
                contact_counter += 1

            else:
                contact_counter = 0

            if contact_counter >= CONTACT_STABLE_STEPS:
                print("\n>>> BOTH FINGERS CONTACT CUBE")

                state = STATE_GRASPED

            # 夹了 2 秒还没有双指接触
            if (
                close_start_time is not None
                and data.time - close_start_time > 2.0
                and state == STATE_CLOSE
            ):
                print("\n>>> GRASP FAILED")

                state = STATE_FAILED

        # ====================================================
        # H. Gripper width
        # ====================================================

        finger_q = np.array([data.qpos[adr] for adr in finger_qpos_adr])

        gripper_width = finger_q[0] + finger_q[1]

        # ====================================================
        # I. Debug
        # ====================================================

        step_count += 1

        if step_count % 250 == 0:
            print(f"\nTime = {data.time:.2f}s")

            print("State =", state)

            print("TCP position =", np.round(tcp_position, 4))

            print("Position error =", round(position_error, 5), "m")

            print("Gripper width =", round(gripper_width, 5), "m")

            if state in [STATE_CLOSE, STATE_GRASPED]:
                print("Left contact =", finger_contacts_cube(left_finger_body_id))

                print("Right contact =", finger_contacts_cube(right_finger_body_id))

        # ====================================================
        # J. End
        # ====================================================

        if state in [STATE_GRASPED, STATE_FAILED]:
            # 停留一小段时间便于观察
            for _ in range(300):
                if not viewer.is_running():
                    break

                for i in range(7):
                    data.ctrl[arm_actuator_ids[i]] = q_approach[i]

                data.ctrl[gripper_actuator_id] = 0.0

                mujoco.mj_step(model, data)

                viewer.sync()

                time.sleep(model.opt.timestep)

            break

        viewer.sync()

        time.sleep(model.opt.timestep)


print("\n===== Final State =====")

print("State =", state)
