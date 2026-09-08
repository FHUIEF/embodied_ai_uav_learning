"""3DOF 平面机械臂正运动学"""
# (q1, q2, q3) -> (x, y, Φ)：Φ为末端执行器的姿态角

import numpy as np
import matplotlib.pyplot as plt


# 1.机械臂参数
l1 = 1.0 
l2 = 0.8 
l3 = 0.3

# 2. 3DOF Forward Kinematics
def forward_kinematics(q1, q2, q3):
    """
    3DOF 平面机械臂正运动学
    
    输入：
        q1, q2, q3
        单位：rad
    
    输出：
        pose = (x, y, phi)

        x, y: 
            末端位置
        phi: 
            末端相对于世界坐标系的姿态角
    """
    # link1 在世界坐标系中的角度
    theta1 = q1

    # link2 在世界坐标系中的角度
    theta2 = q1 + q2

    # link3 在世界坐标系中的角度
    theta3 = q1 + q2 + q3

    # 末端 x 坐标
    x = l1 * np.cos(theta1) + l2 * np.cos(theta2) + l3 * np.cos(theta3)

    # 末端 y 坐标
    y = l1 * np.sin(theta1) + l2 * np.sin(theta2) + l3 * np.sin(theta3)

    # 末端姿态
    phi = theta3
    
    return np.array([x, y, phi])

# 3. 测试关节角
q1 = np.deg2rad(20.0)
q2 = np.deg2rad(40.0)
q3 = np.deg2rad(-60.0)

# 4. 计算 FK
pose = forward_kinematics(q1, q2, q3)

# 5. 输出
print("==== 3DOF Forward Kinematics ====")
print(f"q1 = {np.rad2deg(q1):.2f} deg")
print(f"q2 = {np.rad2deg(q2):.2f} deg")
print(f"q3 = {np.rad2deg(q3):.2f} deg")

print("\nEnd-Effector Pose:")
print(f"x = {pose[0]:.4f} m")
print(f"y = {pose[1]:.4f} m")   
print(f"phi = {np.rad2deg(pose[2]):.2f} deg")

# 获取关节位置
def get_joint_position(q1, q2, q3):
    theta1 = q1
    theta2 = q1 + q2
    theta3 = q1 + q2 + q3

    p0 = np.array([0.0, 0.0])

    p1 = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])

    p2 = p1 + np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)])

    p3 = p2 + np.array([l3 * np.cos(theta3), l3 * np.sin(theta3)])

    return p0, p1, p2, p3

# 绘制机械臂
p0, p1, p2, p3 = get_joint_position(q1, q2, q3)

x_points = [p0[0], p1[0], p2[0], p3[0]]

y_points = [p0[1], p1[1], p2[1], p3[1]]

plt.figure()

plt.plot(x_points, y_points, marker='o')

# 绘制末端姿态，加上一个方向箭头
arrow_length = 0.1

plt.quiver(
    p3[0], 
    p3[1], 
    arrow_length * np.cos(pose[2]), 
    arrow_length * np.sin(pose[2]),
    angles='xy',
    scale_units='xy',
    scale=1
)

plt.axis('equal')
plt.grid()

plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.title('3DOF Planar Arm - Forward Kinematics')

plt.show()

