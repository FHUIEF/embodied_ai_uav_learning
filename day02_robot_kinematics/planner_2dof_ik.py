"""
机械臂逆运动学

为什么可能无解：
    1. 目标点超出机械臂的工作空间
    2. 机械臂的连杆长度无法达到目标点
    |l1 - l2| < sqrt(x^2 + y^2) < l1 + l2
"""
import numpy as np

def inverse_kinematics(x, y, l1=1.0, l2=1.0):
    """
    计算机械臂关节角度
    :param x: 末端位置 x
    :param y: 末端位置 y
    :param l1: 连杆1长度
    :param l2: 连杆2长度
    :return: 关节角度 (q1, q2)，单位：弧度
    """
    # 计算关节2角度
    cos_q2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if np.abs(cos_q2) > 1.0:
        raise ValueError("目标点超出机械臂工作空间")
    
    q2 = np.arccos(cos_q2)  
    
    # 计算关节1角度
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    
    return q1, q2

x_target = 1.0
y_target = 1.0

q1, q2 = inverse_kinematics(x_target, y_target)

print("q1 =", np.rad2deg(q1))
print("q2 =", np.rad2deg(q2))

