import os
import time

import numpy as np
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer


# ============================================================
# 1. Load model
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(current_dir, "assets", "franka_pick_cube", "scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)

ik_data = mujoco.MjData(model)


# ============================================================
# 2. Franka arm / TCP information
# ============================================================

arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

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


tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")


# ============================================================
# 3. Franka home configuration
# ============================================================

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, -0.7853])


# ============================================================
# 4. Basic utilities
# ============================================================


def set_arm_q(mj_data, q):

    for i in range(7):

        mj_data.qpos[arm_qpos_adr[i]] = q[i]

    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(model, mj_data)


def get_tcp_pose(mj_data):

    position = mj_data.site_xpos[tcp_site_id].copy()

    rotation = mj_data.site_xmat[tcp_site_id].reshape(3, 3).copy()

    return (position, rotation)


def rotation_error_vector(R_current, R_target):

    R_error = R_target @ R_current.T

    cos_theta = (np.trace(R_error) - 1.0) / 2.0

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if theta < 1e-8:

        return np.zeros(3)

    axis = np.array([R_error[2, 1] - R_error[1, 2], R_error[0, 2] - R_error[2, 0], R_error[1, 0] - R_error[0, 1]])

    axis = axis / (2.0 * np.sin(theta))

    return theta * axis


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
# 5. TCP geometric Jacobian
# ============================================================


def get_tcp_jacobian(mj_data):

    jacp = np.zeros((3, model.nv))

    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, mj_data, jacp, jacr, tcp_site_id)

    J_position = jacp[:, arm_dof_adr]

    J_rotation = jacr[:, arm_dof_adr]

    return np.vstack([J_position, J_rotation])


# ============================================================
# 6. Jacobian singularity metrics
# ============================================================


def jacobian_metrics(J):

    singular_values = np.linalg.svd(J, compute_uv=False)

    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]

    condition_number = sigma_max / max(sigma_min, 1e-12)

    manipulability = np.prod(singular_values)

    return (singular_values, sigma_min, condition_number, manipulability)


def sigma_min_at_q(q):

    set_arm_q(ik_data, q)

    J = get_tcp_jacobian(ik_data)

    _, sigma_min, _, _ = jacobian_metrics(J)

    return sigma_min


# ============================================================
# 7. Search for a deliberately near-singular posture
# ============================================================


def find_near_singular_configuration(num_samples=4000, seed=7):

    rng = np.random.default_rng(seed)

    q_seed = np.array([0.0, -0.10, 0.0, -0.12, 0.0, 0.20, -0.7853])

    q_seed = apply_joint_limits(q_seed)

    best_q = q_seed.copy()

    best_sigma = sigma_min_at_q(best_q)

    search_radius = np.array([0.35, 0.35, 0.35, 0.20, 0.35, 0.35, 0.50])

    for _ in range(num_samples):

        candidate = q_seed + rng.uniform(-1.0, 1.0, size=7) * search_radius

        candidate = apply_joint_limits(candidate)

        sigma = sigma_min_at_q(candidate)

        if sigma < best_sigma:

            best_sigma = sigma

            best_q = candidate.copy()

    return (best_q, best_sigma)


# ============================================================
# 8. Adaptive DLS primary-task inverse
# ============================================================


def adaptive_damping(sigma_min, sigma_threshold=0.05, lambda_min=1e-4, lambda_max=0.10):

    r = (sigma_threshold - sigma_min) / sigma_threshold

    r = np.clip(r, 0.0, 1.0)

    return lambda_min + (lambda_max - lambda_min) * r**2


def dls_update(J, error, damping):

    task_dim = J.shape[0]

    A = J @ J.T + damping**2 * np.eye(task_dim)

    return J.T @ np.linalg.solve(A, error)


# ============================================================
# 9. Exact 1D null-space direction of a 6x7 Jacobian
#
# For full-row-rank J in R^(6x7), null(J) has dimension 1.
#
# Full SVD:
#     J = U Sigma V^T
#
# The final right singular vector v_7 satisfies approximately:
#     J v_7 = 0
#
# This is exactly the redundant motion direction we want.
# ============================================================


def get_null_direction(J):

    U, S, Vt = np.linalg.svd(J, full_matrices=True)

    null_direction = Vt[-1, :].copy()

    null_direction = null_direction / max(np.linalg.norm(null_direction), 1e-12)

    return null_direction


# ============================================================
# 10. Choose which null-space direction improves sigma_min
#
# Because a 1D null space has two directions:
#     +n and -n
#
# Probe both directions and choose the one that produces the
# larger smallest singular value.
# ============================================================


def choose_singularity_avoidance_direction(q, J, probe_step=0.015):

    null_direction = get_null_direction(J)

    current_sigma = sigma_min_at_q(q)

    q_plus = apply_joint_limits(q + probe_step * null_direction)

    q_minus = apply_joint_limits(q - probe_step * null_direction)

    sigma_plus = sigma_min_at_q(q_plus)

    sigma_minus = sigma_min_at_q(q_minus)

    if sigma_plus <= current_sigma and sigma_minus <= current_sigma:

        return (np.zeros(7), current_sigma, sigma_plus, sigma_minus)

    if sigma_plus >= sigma_minus:

        chosen_direction = null_direction

    else:

        chosen_direction = -null_direction

    return (chosen_direction, current_sigma, sigma_plus, sigma_minus)


# ============================================================
# 11. Run one controller
#
# Experiment design:
#
# PRIMARY TASK:
#     keep the TCP 6D pose fixed
#
# BASELINE:
#     adaptive DLS only
#
# NULL-SPACE VERSION:
#     adaptive DLS primary task
#     + redundant motion that tries to increase sigma_min(J)
#
# This isolates the meaning of null space very clearly:
# the TCP should stay almost fixed while the elbow/posture moves.
# ============================================================


def run_controller(
    q_init, target_position, target_rotation, use_null_space, iterations=350, task_gain=0.30, null_step=0.012
):

    q = q_init.copy()

    history = {
        "sigma_min": [],
        "condition_number": [],
        "manipulability": [],
        "position_error": [],
        "orientation_error": [],
        "task_step_norm": [],
        "null_step_norm": [],
        "null_task_leak": [],
        "damping": [],
        "q": [],
    }

    for _ in range(iterations):

        set_arm_q(ik_data, q)

        current_position, current_rotation = get_tcp_pose(ik_data)

        position_error_vector = target_position - current_position

        orientation_error_vector = rotation_error_vector(current_rotation, target_rotation)

        pose_error = np.concatenate([position_error_vector, orientation_error_vector])

        position_error = np.linalg.norm(position_error_vector)

        orientation_error = np.linalg.norm(orientation_error_vector)

        J = get_tcp_jacobian(ik_data)

        (_, sigma_min, condition_number, manipulability) = jacobian_metrics(J)

        damping = adaptive_damping(sigma_min)

        # ----------------------------------------------------
        # Primary task:
        # keep the TCP at the requested pose.
        # ----------------------------------------------------

        delta_q_task = task_gain * dls_update(J, pose_error, damping)

        delta_q_task = np.clip(delta_q_task, -0.05, 0.05)

        # ----------------------------------------------------
        # Secondary task:
        # move only in null(J) to improve sigma_min.
        # ----------------------------------------------------

        delta_q_null = np.zeros(7)

        null_task_leak = 0.0

        if use_null_space:

            (null_direction, _, _, _) = choose_singularity_avoidance_direction(q, J)

            delta_q_null = null_step * null_direction

            # First-order effect on TCP should be approximately 0:
            #
            #     J * delta_q_null ~= 0
            #
            null_task_leak = np.linalg.norm(J @ delta_q_null)

        # ----------------------------------------------------
        # Combine primary and secondary motions
        # ----------------------------------------------------

        delta_q = delta_q_task + delta_q_null

        delta_q = np.clip(delta_q, -0.06, 0.06)

        q = q + delta_q

        q = apply_joint_limits(q)

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        history["sigma_min"].append(sigma_min)

        history["condition_number"].append(condition_number)

        history["manipulability"].append(manipulability)

        history["position_error"].append(position_error)

        history["orientation_error"].append(orientation_error)

        history["task_step_norm"].append(np.linalg.norm(delta_q_task))

        history["null_step_norm"].append(np.linalg.norm(delta_q_null))

        history["null_task_leak"].append(null_task_leak)

        history["damping"].append(damping)

        history["q"].append(q.copy())

    return (q, history)


# ============================================================
# 12. Find near-singular initial posture
# ============================================================

print("\n===== Search Near-Singular Configuration =====")

q_singular, _ = find_near_singular_configuration()

set_arm_q(ik_data, q_singular)

initial_position, initial_rotation = get_tcp_pose(ik_data)

initial_J = get_tcp_jacobian(ik_data)

(initial_singular_values, initial_sigma_min, initial_condition_number, initial_manipulability) = jacobian_metrics(
    initial_J
)


print("Initial near-singular q (deg):")

print(np.round(np.rad2deg(q_singular), 3))

print("\nInitial TCP position:")

print(np.round(initial_position, 5))

print("\nInitial singular values:")

print(np.round(initial_singular_values, 8))

print("\nInitial sigma_min =", initial_sigma_min)

print("Initial condition number =", initial_condition_number)

print("Initial manipulability =", initial_manipulability)


# ============================================================
# 13. Same primary task for both controllers
#
# Keep exactly the same TCP pose.
# ============================================================

target_position = initial_position.copy()

target_rotation = initial_rotation.copy()


# ============================================================
# 14. Baseline: no null-space secondary motion
# ============================================================

q_baseline, history_baseline = run_controller(q_singular, target_position, target_rotation, use_null_space=False)


# ============================================================
# 15. Null-space singularity avoidance
# ============================================================

q_null, history_null = run_controller(q_singular, target_position, target_rotation, use_null_space=True)


# ============================================================
# 16. Final metrics
# ============================================================

set_arm_q(ik_data, q_baseline)

baseline_position, baseline_rotation = get_tcp_pose(ik_data)

baseline_J = get_tcp_jacobian(ik_data)

(_, baseline_sigma_min, baseline_condition_number, baseline_manipulability) = jacobian_metrics(baseline_J)


set_arm_q(ik_data, q_null)

null_position, null_rotation = get_tcp_pose(ik_data)

null_J = get_tcp_jacobian(ik_data)

(_, null_sigma_min, null_condition_number, null_manipulability) = jacobian_metrics(null_J)


baseline_position_error = np.linalg.norm(target_position - baseline_position)

baseline_orientation_error = np.linalg.norm(rotation_error_vector(baseline_rotation, target_rotation))


null_position_error = np.linalg.norm(target_position - null_position)

null_orientation_error = np.linalg.norm(rotation_error_vector(null_rotation, target_rotation))


print("\n======================================")

print("       NULL-SPACE COMPARISON")

print("======================================")


print("\n--- Initial ---")

print("sigma_min =", initial_sigma_min)

print("condition number =", initial_condition_number)


print("\n--- Without Null Space ---")

print("sigma_min =", baseline_sigma_min)

print("condition number =", baseline_condition_number)

print("manipulability =", baseline_manipulability)

print("TCP position error =", baseline_position_error, "m")

print("TCP orientation error =", np.rad2deg(baseline_orientation_error), "deg")

print("q_final (deg):")

print(np.round(np.rad2deg(q_baseline), 3))


print("\n--- With Null Space ---")

print("sigma_min =", null_sigma_min)

print("condition number =", null_condition_number)

print("manipulability =", null_manipulability)

print("TCP position error =", null_position_error, "m")

print("TCP orientation error =", np.rad2deg(null_orientation_error), "deg")

print("Max ||J * dq_null|| =", np.max(history_null["null_task_leak"]))

print("q_final (deg):")

print(np.round(np.rad2deg(q_null), 3))


# ============================================================
# 17. Plot 1: sigma_min
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_baseline["sigma_min"], label="Without null space")

plt.plot(history_null["sigma_min"], label="With null-space avoidance")

plt.xlabel("Iteration")

plt.ylabel("sigma_min(J)")

plt.title("Null-Space Singularity Avoidance")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 18. Plot 2: condition number
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_baseline["condition_number"], label="Without null space")

plt.plot(history_null["condition_number"], label="With null-space avoidance")

plt.xlabel("Iteration")

plt.ylabel("Condition number")

plt.title("Jacobian Conditioning")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 19. Plot 3: TCP position error
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_baseline["position_error"], label="Without null space")

plt.plot(history_null["position_error"], label="With null-space avoidance")

plt.xlabel("Iteration")

plt.ylabel("TCP position error (m)")

plt.title("Primary Task: TCP Position Preservation")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 20. Plot 4: TCP orientation error
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(np.rad2deg(history_baseline["orientation_error"]), label="Without null space")

plt.plot(np.rad2deg(history_null["orientation_error"]), label="With null-space avoidance")

plt.xlabel("Iteration")

plt.ylabel("TCP orientation error (deg)")

plt.title("Primary Task: TCP Orientation Preservation")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 21. Plot 5: null-space leakage
#
# If the null-space direction is correct:
#
#     J * dq_null ~= 0
#
# so this curve should stay extremely small.
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_null["null_task_leak"], label="||J dq_null||")

plt.xlabel("Iteration")

plt.ylabel("Task-space leakage")

plt.title("Does Null-Space Motion Affect the TCP?")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 22. Plot 6: joint trajectories with null-space control
# ============================================================

q_history_null = np.array(history_null["q"])

plt.figure(figsize=(9, 6))

for i in range(7):

    plt.plot(np.rad2deg(q_history_null[:, i]), label=f"joint{i + 1}")

plt.xlabel("Iteration")

plt.ylabel("Joint angle (deg)")

plt.title("Joint Posture Changes Inside the Null Space")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 23. Viewer 1: initial near-singular posture
# ============================================================

set_arm_q(data, q_singular)

print("\nViewer 1: initial near-singular posture.")

print("It will stay open for about 4 seconds.")


with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running() and time.time() - start < 4.0:

        viewer.sync()

        time.sleep(0.01)


# ============================================================
# 24. Viewer 2: final posture after null-space avoidance
# ============================================================

set_arm_q(data, q_null)

print("\nViewer 2: final posture WITH null-space avoidance.")

print("Notice that the TCP pose should look almost unchanged,")

print("while the internal arm posture/elbow changes.")

print("It will stay open for about 6 seconds.")


with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running() and time.time() - start < 6.0:

        viewer.sync()

        time.sleep(0.01)


plt.show()
