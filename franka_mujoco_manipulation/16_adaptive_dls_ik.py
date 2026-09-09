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
# 2. Joint / TCP information
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
# 3. Home configuration
# ============================================================

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, -0.7853])


# ============================================================
# 4. Utilities
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
# 5. TCP Jacobian
# ============================================================


def get_tcp_jacobian(mj_data):

    jacp = np.zeros((3, model.nv))

    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, mj_data, jacp, jacr, tcp_site_id)

    J_position = jacp[:, arm_dof_adr]

    J_rotation = jacr[:, arm_dof_adr]

    return np.vstack([J_position, J_rotation])


# ============================================================
# 6. Singularity metrics
# ============================================================


def jacobian_metrics(J):

    singular_values = np.linalg.svd(J, compute_uv=False)

    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]

    condition_number = sigma_max / max(sigma_min, 1e-12)

    manipulability = np.prod(singular_values)

    return (singular_values, sigma_min, condition_number, manipulability)


# ============================================================
# 7. Near-singular configuration search
# ============================================================


def find_near_singular_configuration(num_samples=4000, seed=7):

    rng = np.random.default_rng(seed)

    q_seed = np.array([0.0, -0.10, 0.0, -0.12, 0.0, 0.20, -0.7853])

    q_seed = apply_joint_limits(q_seed)

    best_q = q_seed.copy()

    set_arm_q(ik_data, best_q)

    _, best_sigma_min, _, _ = jacobian_metrics(get_tcp_jacobian(ik_data))

    search_radius = np.array([0.35, 0.35, 0.35, 0.20, 0.35, 0.35, 0.50])

    for _ in range(num_samples):

        candidate = q_seed + rng.uniform(-1.0, 1.0, size=7) * search_radius

        candidate = apply_joint_limits(candidate)

        set_arm_q(ik_data, candidate)

        J = get_tcp_jacobian(ik_data)

        _, sigma_min, _, _ = jacobian_metrics(J)

        if sigma_min < best_sigma_min:

            best_sigma_min = sigma_min

            best_q = candidate.copy()

    return (best_q, best_sigma_min)


# ============================================================
# 8. Fixed DLS
#
# dq = J^T (J J^T + lambda^2 I)^(-1) e
# ============================================================


def dls_update(J, error, damping):

    task_dim = J.shape[0]

    A = J @ J.T + damping**2 * np.eye(task_dim)

    return J.T @ np.linalg.solve(A, error)


# ============================================================
# 9. Adaptive damping
#
# Idea:
#
#   sigma_min large -> lambda approximately lambda_min
#   sigma_min small -> lambda approaches lambda_max
#
# r = clip((sigma_threshold - sigma_min) / sigma_threshold, 0, 1)
#
# lambda = lambda_min
#          + (lambda_max - lambda_min) * r^2
#
# Squaring r makes the transition smoother away from singularity.
# ============================================================


def adaptive_damping(sigma_min, sigma_threshold=0.05, lambda_min=1e-4, lambda_max=0.10):

    r = (sigma_threshold - sigma_min) / sigma_threshold

    r = np.clip(r, 0.0, 1.0)

    damping = lambda_min + (lambda_max - lambda_min) * r**2

    return damping


# ============================================================
# 10. Generic DLS IK solver
#
# method:
#   "fixed"    -> fixed damping
#   "adaptive" -> damping determined from sigma_min(J)
# ============================================================


def solve_dls_ik(
    q_init,
    target_position,
    target_rotation,
    method="adaptive",
    fixed_damping=0.05,
    sigma_threshold=0.05,
    lambda_min=1e-4,
    lambda_max=0.10,
    max_iterations=400,
    step_size=0.25,
    max_joint_step=0.20,
):

    q = q_init.copy()

    history = {
        "pose_error": [],
        "position_error": [],
        "orientation_error": [],
        "joint_step_norm": [],
        "sigma_min": [],
        "condition_number": [],
        "damping": [],
    }

    success = False

    for iteration in range(max_iterations):

        set_arm_q(ik_data, q)

        current_position, current_rotation = get_tcp_pose(ik_data)

        position_error_vector = target_position - current_position

        orientation_error_vector = rotation_error_vector(current_rotation, target_rotation)

        pose_error = np.concatenate([position_error_vector, orientation_error_vector])

        position_error = np.linalg.norm(position_error_vector)

        orientation_error = np.linalg.norm(orientation_error_vector)

        pose_error_norm = np.linalg.norm(pose_error)

        J = get_tcp_jacobian(ik_data)

        (_, sigma_min, condition_number, _) = jacobian_metrics(J)

        if method == "fixed":

            damping = fixed_damping

        elif method == "adaptive":

            damping = adaptive_damping(
                sigma_min, sigma_threshold=sigma_threshold, lambda_min=lambda_min, lambda_max=lambda_max
            )

        else:

            raise ValueError("method must be 'fixed' or 'adaptive'")

        raw_delta_q = dls_update(J, pose_error, damping)

        raw_step_norm = np.linalg.norm(raw_delta_q)

        delta_q = np.clip(raw_delta_q, -max_joint_step, max_joint_step)

        history["pose_error"].append(pose_error_norm)

        history["position_error"].append(position_error)

        history["orientation_error"].append(orientation_error)

        history["joint_step_norm"].append(raw_step_norm)

        history["sigma_min"].append(sigma_min)

        history["condition_number"].append(condition_number)

        history["damping"].append(damping)

        if position_error < 1e-4 and orientation_error < np.deg2rad(0.1):

            success = True

            break

        q = q + step_size * delta_q

        q = apply_joint_limits(q)

    return (q, success, history)


# ============================================================
# 11. Find near-singular pose
# ============================================================

print("\n===== Search Near-Singular Configuration =====")

q_singular, _ = find_near_singular_configuration()

set_arm_q(ik_data, q_singular)

J_singular = get_tcp_jacobian(ik_data)

(singular_values, sigma_min, condition_number, manipulability) = jacobian_metrics(J_singular)


print("Near-singular q (deg):")

print(np.round(np.rad2deg(q_singular), 3))

print("\nSingular values:")

print(np.round(singular_values, 8))

print("\nsigma_min =", sigma_min)

print("condition number =", condition_number)

print("manipulability =", manipulability)


# ============================================================
# 12. Visualize damping law numerically
# ============================================================

print("\n===== Adaptive Damping at Current Pose =====")

current_lambda = adaptive_damping(sigma_min)

print("sigma_min =", sigma_min)

print("adaptive lambda =", current_lambda)


# ============================================================
# 13. Define near-singular finite target
# ============================================================

set_arm_q(ik_data, q_singular)

start_position, start_rotation = get_tcp_pose(ik_data)

target_position = start_position + np.array([0.015, 0.010, 0.010])

target_rotation = start_rotation.copy()


print("\n===== Target =====")

print("Start position =", np.round(start_position, 5))

print("Target position =", np.round(target_position, 5))


# ============================================================
# 14. Fixed DLS
# ============================================================

FIXED_DAMPING = 0.05

q_fixed, success_fixed, history_fixed = solve_dls_ik(
    q_singular, target_position, target_rotation, method="fixed", fixed_damping=FIXED_DAMPING, max_iterations=400
)


# ============================================================
# 15. Adaptive DLS
# ============================================================

q_adaptive, success_adaptive, history_adaptive = solve_dls_ik(
    q_singular,
    target_position,
    target_rotation,
    method="adaptive",
    sigma_threshold=0.05,
    lambda_min=1e-4,
    lambda_max=0.10,
    max_iterations=400,
)


# ============================================================
# 16. Result
# ============================================================

print("\n======================================")

print("      FIXED DLS vs ADAPTIVE DLS")

print("======================================")


print("\nFixed DLS success =", success_fixed)

print("Fixed DLS iterations =", len(history_fixed["pose_error"]))

print("Fixed DLS max raw ||dq|| =", np.max(history_fixed["joint_step_norm"]))

print("Fixed DLS final pose error =", history_fixed["pose_error"][-1])

print("Fixed lambda =", FIXED_DAMPING)


print("\nAdaptive DLS success =", success_adaptive)

print("Adaptive DLS iterations =", len(history_adaptive["pose_error"]))

print("Adaptive DLS max raw ||dq|| =", np.max(history_adaptive["joint_step_norm"]))

print("Adaptive DLS final pose error =", history_adaptive["pose_error"][-1])

print("Adaptive lambda min/max observed =", (np.min(history_adaptive["damping"]), np.max(history_adaptive["damping"])))


print("\nq_fixed (deg):")

print(np.round(np.rad2deg(q_fixed), 3))

print("\nq_adaptive (deg):")

print(np.round(np.rad2deg(q_adaptive), 3))


# ============================================================
# 17. Plot 1: pose error
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_fixed["pose_error"], label="Fixed DLS")

plt.plot(history_adaptive["pose_error"], label="Adaptive DLS")

plt.xlabel("Iteration")

plt.ylabel("6D pose error norm")

plt.title("Fixed vs Adaptive DLS: Pose Error")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 18. Plot 2: raw joint correction
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_fixed["joint_step_norm"], label="Fixed DLS")

plt.plot(history_adaptive["joint_step_norm"], label="Adaptive DLS")

plt.xlabel("Iteration")

plt.ylabel("Raw ||delta q||")

plt.title("Fixed vs Adaptive DLS: Joint Correction")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 19. Plot 3: sigma_min
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_fixed["sigma_min"], label="Fixed DLS path")

plt.plot(history_adaptive["sigma_min"], label="Adaptive DLS path")

plt.xlabel("Iteration")

plt.ylabel("sigma_min(J)")

plt.title("Smallest Jacobian Singular Value")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 20. Plot 4: adaptive damping
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_adaptive["damping"], label="Adaptive lambda")

plt.axhline(FIXED_DAMPING, linestyle="--", label="Fixed lambda")

plt.xlabel("Iteration")

plt.ylabel("Damping lambda")

plt.title("Adaptive Damping Coefficient")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 21. Plot 5: damping law lambda(sigma_min)
# ============================================================

sigma_samples = np.linspace(0.0, 0.10, 300)

lambda_samples = np.array(
    [adaptive_damping(sigma, sigma_threshold=0.05, lambda_min=1e-4, lambda_max=0.10) for sigma in sigma_samples]
)


plt.figure(figsize=(8, 5))

plt.plot(sigma_samples, lambda_samples)

plt.xlabel("sigma_min(J)")

plt.ylabel("Adaptive damping lambda")

plt.title("Adaptive Damping Law")

plt.grid(True)

plt.tight_layout()


# ============================================================
# 22. Viewer
# ============================================================

set_arm_q(data, q_singular)

print("\nViewer shows the near-singular configuration.")

print("Close the viewer to display the comparison plots.")


with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer_start = time.time()

    while viewer.is_running() and time.time() - viewer_start < 8.0:

        viewer.sync()

        time.sleep(0.01)


plt.show()
