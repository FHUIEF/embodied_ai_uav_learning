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
# 5. Full 6 x 7 geometric Jacobian of TCP
# ============================================================


def get_tcp_jacobian(mj_data):

    jacp = np.zeros((3, model.nv))

    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, mj_data, jacp, jacr, tcp_site_id)

    J_position = jacp[:, arm_dof_adr]

    J_rotation = jacr[:, arm_dof_adr]

    J = np.vstack([J_position, J_rotation])

    return J


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


# ============================================================
# 7. Search a near-singular Franka configuration
#
# We deliberately search around an "almost straight" arm.
# The final result is verified numerically using sigma_min(J).
# ============================================================


def find_near_singular_configuration(num_samples=4000, seed=7):

    rng = np.random.default_rng(seed)

    # A nearly extended seed pose.
    # joint4 is kept slightly negative because its valid range
    # does not include zero.
    q_seed = np.array([0.0, -0.10, 0.0, -0.12, 0.0, 0.20, -0.7853])

    q_seed = apply_joint_limits(q_seed)

    best_q = q_seed.copy()

    set_arm_q(ik_data, best_q)

    best_J = get_tcp_jacobian(ik_data)

    _, best_sigma_min, _, _ = jacobian_metrics(best_J)

    # Local random search around the extended posture.
    # Keep the search local so the result remains visually
    # interpretable instead of producing a bizarre random pose.
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
# 8. Pseudoinverse update
# ============================================================


def pinv_update(J, error):

    return np.linalg.pinv(J) @ error


# ============================================================
# 9. Damped Least Squares update
#
# dq = J^T (J J^T + lambda^2 I)^(-1) e
# ============================================================


def dls_update(J, error, damping=0.05):

    task_dim = J.shape[0]

    A = J @ J.T + damping**2 * np.eye(task_dim)

    return J.T @ np.linalg.solve(A, error)


# ============================================================
# 10. Generic iterative IK solver for comparison
# ============================================================


def solve_ik_compare(
    q_init,
    target_position,
    target_rotation,
    method="pinv",
    damping=0.05,
    max_iterations=300,
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

        if method == "pinv":

            raw_delta_q = pinv_update(J, pose_error)

        elif method == "dls":

            raw_delta_q = dls_update(J, pose_error, damping=damping)

        else:

            raise ValueError("method must be 'pinv' or 'dls'")

        # Keep the simulation comparison numerically safe.
        # The RAW update magnitude is still recorded before
        # clipping so we can see amplification near singularity.
        raw_step_norm = np.linalg.norm(raw_delta_q)

        delta_q = np.clip(raw_delta_q, -max_joint_step, max_joint_step)

        history["pose_error"].append(pose_error_norm)

        history["position_error"].append(position_error)

        history["orientation_error"].append(orientation_error)

        history["joint_step_norm"].append(raw_step_norm)

        history["sigma_min"].append(sigma_min)

        history["condition_number"].append(condition_number)

        if position_error < 1e-4 and orientation_error < np.deg2rad(0.1):

            success = True

            break

        q = q + step_size * delta_q

        q = apply_joint_limits(q)

    return (q, success, history)


# ============================================================
# 11. Find deliberately near-singular posture
# ============================================================

print("\n===== Search Near-Singular Configuration =====")

q_singular, searched_sigma_min = find_near_singular_configuration()

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
# 12. Compare with the normal home pose
# ============================================================

set_arm_q(ik_data, q_home)

J_home = get_tcp_jacobian(ik_data)

(singular_values_home, sigma_min_home, condition_number_home, manipulability_home) = jacobian_metrics(J_home)


print("\n===== Home Configuration =====")

print("sigma_min =", sigma_min_home)

print("condition number =", condition_number_home)

print("manipulability =", manipulability_home)


# ============================================================
# 13. Worst instantaneous task-space direction
#
# For J = U S V^T, the last column of U corresponds to the
# task-space direction associated with sigma_min.
#
# A small error along this direction is where pseudoinverse
# amplification is easiest to observe.
# ============================================================

set_arm_q(ik_data, q_singular)

J_singular = get_tcp_jacobian(ik_data)

U, S, Vt = np.linalg.svd(J_singular, full_matrices=False)

u_min = U[:, -1]


# Small differential 6D task error.
# This is ONLY for an instantaneous sensitivity comparison.
worst_task_error = 0.01 * u_min


PINV_raw = pinv_update(J_singular, worst_task_error)

DLS_raw = dls_update(J_singular, worst_task_error, damping=0.05)


print("\n===== Instantaneous Near-Singularity Comparison =====")

print("||task error|| =", np.linalg.norm(worst_task_error))

print("||dq_pinv|| =", np.linalg.norm(PINV_raw))

print("||dq_dls|| =", np.linalg.norm(DLS_raw))

print("PINV / DLS step ratio =", np.linalg.norm(PINV_raw) / max(np.linalg.norm(DLS_raw), 1e-12))


# ============================================================
# 14. Define the same finite IK target for both solvers
#
# Keep orientation fixed and move TCP by a small amount.
# ============================================================

set_arm_q(ik_data, q_singular)

start_position, start_rotation = get_tcp_pose(ik_data)

target_position = start_position + np.array([0.015, 0.010, 0.010])

target_rotation = start_rotation.copy()


print("\n===== Finite IK Target =====")

print("Start position =", np.round(start_position, 5))

print("Target position =", np.round(target_position, 5))


# ============================================================
# 15. Solve with pseudoinverse
# ============================================================

q_pinv, success_pinv, history_pinv = solve_ik_compare(
    q_singular, target_position, target_rotation, method="pinv", max_iterations=300, step_size=0.25, max_joint_step=0.20
)


# ============================================================
# 16. Solve with DLS
# ============================================================

DAMPING = 0.05

q_dls, success_dls, history_dls = solve_ik_compare(
    q_singular,
    target_position,
    target_rotation,
    method="dls",
    damping=DAMPING,
    max_iterations=300,
    step_size=0.25,
    max_joint_step=0.20,
)


# ============================================================
# 17. Final comparison
# ============================================================

print("\n======================================")

print("        PINV vs DLS IK RESULT")

print("======================================")

print("\nPINV success =", success_pinv)

print("PINV iterations =", len(history_pinv["pose_error"]))

print("PINV max raw ||dq|| =", np.max(history_pinv["joint_step_norm"]))

print("PINV final position error =", history_pinv["position_error"][-1])


print("\nDLS success =", success_dls)

print("DLS iterations =", len(history_dls["pose_error"]))

print("DLS max raw ||dq|| =", np.max(history_dls["joint_step_norm"]))

print("DLS final position error =", history_dls["position_error"][-1])


print("\nq_pinv (deg):")

print(np.round(np.rad2deg(q_pinv), 3))

print("\nq_dls (deg):")

print(np.round(np.rad2deg(q_dls), 3))


# ============================================================
# 18. Plot: pose error
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_pinv["pose_error"], label="Pseudoinverse")

plt.plot(history_dls["pose_error"], label="DLS")

plt.xlabel("Iteration")

plt.ylabel("6D pose error norm")

plt.title("IK Error Near Singularity")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 19. Plot: raw joint correction norm
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_pinv["joint_step_norm"], label="Pseudoinverse")

plt.plot(history_dls["joint_step_norm"], label="DLS")

plt.xlabel("Iteration")

plt.ylabel("Raw ||delta q||")

plt.title("Joint Correction Amplification")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 20. Plot: smallest singular value
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history_pinv["sigma_min"], label="Pseudoinverse path")

plt.plot(history_dls["sigma_min"], label="DLS path")

plt.xlabel("Iteration")

plt.ylabel("sigma_min(J)")

plt.title("Smallest Jacobian Singular Value")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 21. Viewer: show the deliberately near-singular posture
# ============================================================

set_arm_q(data, q_singular)

print("\nViewer shows the near-singular configuration.")

print("Close the viewer to show the plots.")


with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer_start = time.time()

    while viewer.is_running() and time.time() - viewer_start < 8.0:

        viewer.sync()

        time.sleep(0.01)


plt.show()
