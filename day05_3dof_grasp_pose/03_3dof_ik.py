"""3DOF 平面机械臂逆运动学"""
# 去掉末端连杆: x_w = x - l3 * cos(phi), y_w = y - l3 * sin(phi)
# 对 wrist center 做 2DOF IK: (x_w, y_w) -> (q1, q2)
# 补偿姿态: q3 = phi - q1 - q2

from math import cos

import numpy as np

# 1. 机械臂参数
l1 = 1.0
l2 = 0.8
l3 = 0.3

# 2. 3DOF 正运动学
def forward_kinematics(q1, q2, q3):
    theta1 = q1
    theta2 = q1 + q2
    theta3 = q1 + q2 + q3

    x = l1 * np.cos(theta1) + l2 * np.cos(theta2) + l3 * np.cos(theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta2) + l3 * np.sin(theta3)

    phi = theta3
    return np.array([x, y, phi])

# 3. 3DOF 逆运动学
def inverse_kinematics(x_d, y_d, phi_d, elbow="down"):
    """
    3DOF 平面机械臂逆运动学
    
    输入：
        x_d, y_d:
            末端目标位置
        
        phi_d:
            末端目标姿态，单位 rad

        elbow:
            "up" or "down", 表示肘部弯曲方向
    
    输出：
        [q1, q2, q3]
    """
    # Step 1: 计算wrist center
    x_w = x_d - l3 * np.cos(phi_d)
    y_w = y_d - l3 * np.sin(phi_d)

    # Step 2: 计算 q2
    cos_q2 = (x_w**2 + y_w**2 - l1**2 - l2**2) / (2 * l1 * l2)       

    # 检查 wrist center 是否可达
    if abs(cos_q2) > 1.0:
        raise ValueError("Target pose is outside the workspace.")

    cos_q2 = np.clip(cos_q2, -1.0, 1.0)  # 防止数值误差导致的 acos 出错

    # elbow-up / elbow-down
    if elbow == "down":
        q2 = np.arccos(cos_q2)
    elif elbow == "up":
        q2 = -np.arccos(cos_q2)
    else:
        raise ValueError("elbow must be 'up' or 'down'.")

    # Step 3: 计算 q1
    q1 = np.arctan2(y_w, x_w) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))

    # Step 4: 计算 q3
    q3 = phi_d - q1 - q2

    return np.array([q1, q2, q3])

# 4. 设置目标位姿
x_target = 1.2
y_target = 0.8
phi_target = np.deg2rad(0.0)

# 计算 wrist center
x_w = x_target - l3 * np.cos(phi_target)
y_w = y_target - l3 * np.sin(phi_target)

# 5. IK
q_solution = inverse_kinematics(x_target, y_target, phi_target, elbow="down")

q1 = q_solution[0]
q2 = q_solution[1]
q3 = q_solution[2]

# 6. 输出 IK 结果
print("==== 3DOF Inverse Kinematics ====")
print(f"Target x = {x_target:.4f} m")
print(f"Target y = {y_target:.4f} m")
print(f"Target phi = {np.rad2deg(phi_target):.2f} deg")

print("\nIK Solution:")
print(f"q1 = {np.rad2deg(q1):.2f} deg")
print(f"q2 = {np.rad2deg(q2):.2f} deg")
print(f"q3 = {np.rad2deg(q3):.2f} deg")

# 7. FK 验证
pose_check = forward_kinematics(q1, q2, q3)
print("\n==== FK Verification ====")
print(f"FK x = {pose_check[0]:.4f} m")
print(f"FK y = {pose_check[1]:.4f} m")
print(f"FK phi = {np.rad2deg(pose_check[2]):.2f} deg")

# 8. 计算误差
position_error = np.linalg.norm(pose_check[:2] - np.array([x_target, y_target]))

orientation_error = (pose_check[2] - phi_target)
# 角度误差归一化到 [-pi, pi]
# def wrap_to_pi(angle):
#     return (angle + np.pi) % (2 * np.pi) - np.pi

print("\n==== Error ====")
print(f"Position error = {position_error:.8f} m")
print(f"Orientation error = {np.rad2deg(orientation_error):.8f} deg")

# 打印 wrist center
print("\n==== Wrist Center ====")
print(f"Wrist center x = {x_w:.4f} m")
print(f"Wrist center y = {y_w:.4f} m")
