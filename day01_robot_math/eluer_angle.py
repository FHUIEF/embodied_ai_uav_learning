"""

偏航角：yaw，绕着y轴旋转
俯仰角：pitch，绕着x轴旋转
滚转角：roll，绕着z轴旋转

"""
import numpy as np

def Rx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def Ry(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def Rz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

roll = np.deg2rad(30)  
pitch = np.deg2rad(45)
yaw = np.deg2rad(60)

R_zyx = Rz(yaw) @ Ry(pitch) @ Rx(roll)

R_xyz = Rx(roll) @ Ry(pitch) @ Rz(yaw)

print("ZYX:")
print(R_zyx)

print("\nXYZ:")
print(R_xyz)

# 欧拉角适合人理解，机器人内部计算经常更喜欢四元数。