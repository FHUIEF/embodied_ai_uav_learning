import os
import time

import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# 1. 加载模型
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(
    current_dir, 
    "mujoco_menagerie", 
    "franka_emika_panda", 
    "scene.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)


# ============================================================
# 2. Franka Arm Joint
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
# 3. Actuator IDs
# ============================================================

arm_actuator_ids = []


for i in range(1, 8):

    actuator_id = mujoco.mj_name2id(
        model, 
        mujoco.mjtObj.mjOBJ_ACTUATOR, 
        f"actuator{i}"
    )

    arm_actuator_ids.append(actuator_id)


arm_actuator_ids = np.array(arm_actuator_ids)


gripper_actuator_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_ACTUATOR, 
    "actuator8"
)


# ============================================================
# 4. Hand Body
# ============================================================

hand_body_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_BODY, 
    "hand"
)


# ============================================================
# 5. Home Keyframe
# ============================================================

home_key_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_KEY, 
    "home"
)


mujoco.mj_resetDataKeyframe(model, data, home_key_id)

mujoco.mj_forward(model, data)


# ============================================================
# 6. Rotation Matrix
# ============================================================


def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ============================================================
# 7. Rotation Matrix -> Rotation Error Vector
# ============================================================


def rotation_error_vector(R_current, R_target):
    # 当前姿态转到目标姿态所需的旋转
    R_error = R_target @ R_current.T

    # --------------------------------------------------------
    # axis-angle 中的旋转角
    # --------------------------------------------------------

    cos_theta = (np.trace(R_error) - 1.0) / 2.0

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    # --------------------------------------------------------
    # 姿态已经非常接近
    # --------------------------------------------------------

    if theta < 1e-8:
        return np.zeros(3)

    # --------------------------------------------------------
    # 旋转轴
    # --------------------------------------------------------

    axis = np.array(
        [
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1],
        ]
    )

    axis = axis / (2.0 * np.sin(theta))

    # rotation vector
    return theta * axis


# ============================================================
# 8. Initial Hand Pose
# ============================================================

initial_position = data.xpos[hand_body_id].copy()


initial_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()


print("===== Initial Hand Pose =====")

print("Position:")

print(np.round(initial_position, 4))

print("\nRotation:")

print(np.round(initial_rotation, 4))


# ============================================================
# 9. Target Pose
# ============================================================

target_position = initial_position + np.array([0.04, 0.03, 0.03])


# 在世界 z 方向增加 15 deg
target_rotation = rotation_z(np.deg2rad(15.0)) @ initial_rotation


print("\n===== Target Pose =====")

print("Position:")

print(np.round(target_position, 4))

print("\nRotation:")

print(np.round(target_rotation, 4))


# ============================================================
# 10. Numerical IK Parameters
# ============================================================

max_iterations = 1000

position_tolerance = 1e-4

orientation_tolerance = np.deg2rad(0.1)

step_size = 0.2


# ============================================================
# 11. Initial IK Configuration
# ============================================================

q_ik = np.array([data.qpos[adr] for adr in arm_qpos_adr])


# ============================================================
# 12. Numerical Pose IK
# ============================================================

ik_success = False


print("\n===== 6D Numerical IK =====")


for iteration in range(max_iterations):
    # --------------------------------------------------------
    # q_ik 写入模型
    # --------------------------------------------------------

    for i in range(7):
        data.qpos[arm_qpos_adr[i]] = q_ik[i]

    data.qvel[:] = 0.0

    # --------------------------------------------------------
    # FK
    # --------------------------------------------------------

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # Current Pose
    # --------------------------------------------------------

    current_position = data.xpos[hand_body_id].copy()

    current_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()

    # --------------------------------------------------------
    # Position Error
    # --------------------------------------------------------

    position_error_vector = target_position - current_position

    position_error = np.linalg.norm(position_error_vector)

    # --------------------------------------------------------
    # Orientation Error
    # --------------------------------------------------------

    orientation_error_vector = rotation_error_vector(current_rotation, target_rotation)

    orientation_error = np.linalg.norm(orientation_error_vector)

    # --------------------------------------------------------
    # 收敛判断
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

    # --------------------------------------------------------
    # 只取 Franka 7DOF
    # --------------------------------------------------------

    J_position = jacp[:, arm_dof_adr]

    J_rotation = jacr[:, arm_dof_adr]

    # --------------------------------------------------------
    # 6 x 7 Jacobian
    # --------------------------------------------------------

    J = np.vstack([J_position, J_rotation])

    # --------------------------------------------------------
    # 6D Pose Error
    # --------------------------------------------------------

    pose_error = np.concatenate([position_error_vector, orientation_error_vector])

    # --------------------------------------------------------
    # Pseudoinverse
    # --------------------------------------------------------

    J_pinv = np.linalg.pinv(J)

    # --------------------------------------------------------
    # Pose error -> Joint correction
    # --------------------------------------------------------

    delta_q = J_pinv @ pose_error

    # --------------------------------------------------------
    # Update q
    # --------------------------------------------------------

    q_ik = q_ik + step_size * delta_q

    # --------------------------------------------------------
    # Joint limit
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
            f"iter={iteration:4d} | "
            f"pos_err="
            f"{position_error:.6f} m | "
            f"ori_err="
            f"{np.rad2deg(orientation_error):.4f} deg"
        )


# ============================================================
# 13. IK Result
# ============================================================

q_desired = q_ik.copy()


print("\n===== IK Result =====")

print("IK success =", ik_success)

print("q_desired (deg):")

print(np.round(np.rad2deg(q_desired), 3))


# ============================================================
# 14. FK Verification
# ============================================================

for i in range(7):
    data.qpos[arm_qpos_adr[i]] = q_desired[i]


mujoco.mj_forward(model, data)


ik_position = data.xpos[hand_body_id].copy()


ik_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()


ik_position_error = np.linalg.norm(target_position - ik_position)


ik_orientation_error = np.linalg.norm(
    rotation_error_vector(ik_rotation, target_rotation)
)


print("\n===== FK Verification =====")

print("Position error =", ik_position_error, "m")

print("Orientation error =", np.rad2deg(ik_orientation_error), "deg")


# ============================================================
# 15. Reset Home
# ============================================================

mujoco.mj_resetDataKeyframe(model, data, home_key_id)

mujoco.mj_forward(model, data)


# ============================================================
# 16. Gripper Open
# ============================================================

data.ctrl[gripper_actuator_id] = 255.0


# ============================================================
# 17. Dynamic Reach Parameters
# ============================================================

position_threshold = 0.01

orientation_threshold = np.deg2rad(2.0)

required_stable_steps = 200

stable_counter = 0

success = False


simulation_time = 10.0

step_count = 0


# ============================================================
# 18. Dynamics Simulation
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < simulation_time:
        # ----------------------------------------------------
        # Joint Position Target
        # ----------------------------------------------------

        for i in range(7):
            data.ctrl[arm_actuator_ids[i]] = q_desired[i]

        # ----------------------------------------------------
        # Simulation Step
        # ----------------------------------------------------

        mujoco.mj_step(model, data)

        # ----------------------------------------------------
        # Current Pose
        # ----------------------------------------------------

        current_position = data.xpos[hand_body_id].copy()

        current_rotation = data.xmat[hand_body_id].reshape(3, 3).copy()

        # ----------------------------------------------------
        # Pose Error
        # ----------------------------------------------------

        current_position_error = np.linalg.norm(target_position - current_position)

        current_orientation_error = np.linalg.norm(
            rotation_error_vector(current_rotation, target_rotation)
        )

        # ----------------------------------------------------
        # Reach Judge
        # ----------------------------------------------------

        if (
            current_position_error < position_threshold
            and current_orientation_error < orientation_threshold
        ):
            stable_counter += 1

        else:
            stable_counter = 0

        if stable_counter >= required_stable_steps:
            print("\n>>> 6D Pose target reached!")

            success = True

            break

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        step_count += 1

        if step_count % 250 == 0:
            print(f"\nTime = {data.time:.2f}s")

            print("Position error =", round(current_position_error, 5), "m")

            print(
                "Orientation error =",
                round(np.rad2deg(current_orientation_error), 3),
                "deg",
            )

        viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# 19. Final Result
# ============================================================

print("\n===== Final Result =====")

print("Success =", success)

print("Final position error =", current_position_error, "m")

print("Final orientation error =", np.rad2deg(current_orientation_error), "deg")
