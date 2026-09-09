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
# 5. Body IDs
# ============================================================

hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")


cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")


# ============================================================
# 6. Home configuration
# ============================================================

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, -0.7853])


finger_home = np.array([0.04, 0.04])


# ============================================================
# 7. 初始化 Franka
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
# 8. 记录 Home hand orientation
# ============================================================

home_hand_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()


print("===== Home Hand Rotation =====")

print(np.round(home_hand_rotation, 4))


# ============================================================
# 9. 先让 Cube 自然落到桌面
# ============================================================

settle_time = 1.0


while data.time < settle_time:
    # arm 保持 home
    for i in range(7):
        data.ctrl[arm_actuator_ids[i]] = q_home[i]

    data.ctrl[gripper_actuator_id] = 255.0

    mujoco.mj_step(model, data)


# ============================================================
# 10. 获取稳定后的 Cube Position
# ============================================================

cube_position = data.xpos[cube_body_id].copy()


print("\n===== Cube Position =====")

print(np.round(cube_position, 5))


# ============================================================
# 11. 自动生成 Pre-grasp Pose
# ============================================================

PREGRASP_HEIGHT = 0.20


pregrasp_position = cube_position + np.array([0.0, 0.0, PREGRASP_HEIGHT])


pregrasp_rotation = home_hand_rotation.copy()


print("\n===== Pre-grasp Pose =====")

print("Position:")

print(np.round(pregrasp_position, 5))


print("\nRotation:")

print(np.round(pregrasp_rotation, 4))


# ============================================================
# 12. Rotation Error
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
# 13. IK 参数
# ============================================================

max_iterations = 1000

position_tolerance = 1e-4

orientation_tolerance = np.deg2rad(0.1)

step_size = 0.2


# ============================================================
# 14. IK 初始 q
# ============================================================

q_ik = np.array([data.qpos[adr] for adr in arm_qpos_adr])


ik_success = False


print("\n===== Pre-grasp IK =====")


# ============================================================
# 15. Numerical 6D IK
# ============================================================

for iteration in range(max_iterations):
    # --------------------------------------------------------
    # 当前 q 写入 MuJoCo
    # --------------------------------------------------------

    for i in range(7):
        data.qpos[arm_qpos_adr[i]] = q_ik[i]

    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # 当前 Hand Pose
    # --------------------------------------------------------

    current_position = data.xpos[hand_body_id].copy()

    current_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()

    # --------------------------------------------------------
    # Position Error
    # --------------------------------------------------------

    position_error_vector = pregrasp_position - current_position

    position_error = np.linalg.norm(position_error_vector)

    # --------------------------------------------------------
    # Orientation Error
    # --------------------------------------------------------

    orientation_error_vector = rotation_error_vector(
        current_rotation, pregrasp_rotation
    )

    orientation_error = np.linalg.norm(orientation_error_vector)

    # --------------------------------------------------------
    # 收敛
    # --------------------------------------------------------

    if (
        position_error < position_tolerance
        and orientation_error < orientation_tolerance
    ):
        print(f"IK converged at iteration {iteration}")

        ik_success = True

        break

    # --------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------

    jacp = np.zeros((3, model.nv))

    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacBody(model, data, jacp, jacr, hand_body_id)

    J_position = jacp[:, arm_dof_adr]

    J_rotation = jacr[:, arm_dof_adr]

    J = np.vstack([J_position, J_rotation])

    pose_error = np.concatenate([position_error_vector, orientation_error_vector])

    # --------------------------------------------------------
    # Pseudoinverse IK
    # --------------------------------------------------------

    delta_q = np.linalg.pinv(J) @ pose_error

    q_ik = q_ik + step_size * delta_q

    # --------------------------------------------------------
    # Joint limits
    # --------------------------------------------------------

    for i in range(7):
        joint_id = arm_joint_ids[i]

        if model.jnt_limited[joint_id]:
            lower = model.jnt_range[joint_id, 0]

            upper = model.jnt_range[joint_id, 1]

            q_ik[i] = np.clip(q_ik[i], lower, upper)

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    if iteration % 20 == 0:
        print(
            f"iter={iteration:3d} | "
            f"position error="
            f"{position_error:.6f} m | "
            f"orientation error="
            f"{np.rad2deg(orientation_error):.4f} deg"
        )


# ============================================================
# 16. IK 结果
# ============================================================

q_pregrasp = q_ik.copy()


print("\n===== IK Result =====")

print("Success =", ik_success)


print("q_pregrasp (deg):")

print(np.round(np.rad2deg(q_pregrasp), 3))


# ============================================================
# 17. 重新初始化 Franka
# ============================================================

for i in range(7):
    data.qpos[arm_qpos_adr[i]] = q_home[i]


for i in range(2):
    data.qpos[finger_qpos_adr[i]] = finger_home[i]


data.qvel[:] = 0.0


mujoco.mj_forward(model, data)


# ============================================================
# 18. Dynamic Pre-grasp Reach
# ============================================================

position_threshold = 0.01

orientation_threshold = np.deg2rad(2.0)

required_stable_steps = 200

stable_counter = 0

success = False


step_count = 0

simulation_time = 10.0


with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = data.time

    while viewer.is_running() and data.time < start_time + simulation_time:
        # ----------------------------------------------------
        # Arm target
        # ----------------------------------------------------

        for i in range(7):
            data.ctrl[arm_actuator_ids[i]] = q_pregrasp[i]

        # ----------------------------------------------------
        # Gripper open
        # ----------------------------------------------------

        data.ctrl[gripper_actuator_id] = 255.0

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        mujoco.mj_step(model, data)

        # ----------------------------------------------------
        # Current hand pose
        # ----------------------------------------------------

        current_position = data.xpos[hand_body_id].copy()

        current_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()

        current_position_error = np.linalg.norm(pregrasp_position - current_position)

        current_orientation_error = np.linalg.norm(
            rotation_error_vector(current_rotation, pregrasp_rotation)
        )

        # ----------------------------------------------------
        # Stable condition
        # ----------------------------------------------------

        if (
            current_position_error < position_threshold
            and current_orientation_error < orientation_threshold
        ):
            stable_counter += 1

        else:
            stable_counter = 0

        if stable_counter >= required_stable_steps:
            print("\n>>> PREGRASP reached!")

            success = True

            break

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        step_count += 1

        if step_count % 250 == 0:
            print(f"\nTime = {data.time:.2f}s")

            print("Hand position:")

            print(np.round(current_position, 4))

            print("Pregrasp target:")

            print(np.round(pregrasp_position, 4))

            print("Position error =", round(current_position_error, 5), "m")

            print(
                "Orientation error =",
                round(np.rad2deg(current_orientation_error), 3),
                "deg",
            )

        viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# 19. Final
# ============================================================

print("\n===== Final Result =====")

print("Success =", success)

print("Final position error =", current_position_error, "m")

print("Final orientation error =", np.rad2deg(current_orientation_error), "deg")
