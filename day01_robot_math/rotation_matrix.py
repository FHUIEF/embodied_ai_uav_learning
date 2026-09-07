"""
向量、坐标系、旋转矩阵

旋转矩阵三个重要性质：
    R^T R = I
    det(R) = 1
    ||Rp|| = ||p||


"""
import numpy as np

# 原始向量
# p = np.array([1.0, 1.0, 1.0])
p = np.array([1.0, 0, 0])

# 角度：90°
theta = np.deg2rad(90)  # 角度 → 弧度

# 绕Z轴旋转矩阵
Rz = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

# 旋转
p_rotated = Rz @ p  # 矩阵乘法

print("原始向量:")
print(p)

print("\n旋转后矩阵:")
print(Rz)

print("\n旋转后的向量:")
print(p_rotated)

# 验证旋转矩阵的性质
print("\nR.T @ R:")
print(Rz.T @ Rz)

print("\ndet(R):")
print(np.linalg.det(Rz))

print("\n旋转前长度:")
print(np.linalg.norm(p))

print("\n旋转后长度:")
print(np.linalg.norm(p_rotated))

# 验证旋转矩阵表示坐标系转换
# p = np.array([1.0, 0.0, 0.0]) 
# theta = np.deg2rad(90)  

Rx = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

Ry = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])

p_rotated_x = Rx @ p
p_rotated_y = Ry @ p
print(p_rotated_x)
print(p_rotated_y)  
print(p_rotated)
