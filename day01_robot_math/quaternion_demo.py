"""四元数"""
import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


def axis_angle_to_quaternion(axis, theta):
    """
    轴角 -> 四元数
    四元数格式: [w, x, y, z]
    """
    axis = normalize(axis)

    w = np.cos(theta / 2)
    xyz = axis * np.sin(theta / 2)

    q = np.array([w, xyz[0], xyz[1], xyz[2]])
    return q


def quaternion_to_rotation_matrix(q):
    """
    四元数 [w, x, y, z] -> 旋转矩阵
    """
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    R = np.array([
        [
            1 - 2 * (y**2 + z**2),
            2 * (x*y - z*w),
            2 * (x*z + y*w)
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x**2 + z**2),
            2 * (y*z - x*w)
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x**2 + y**2)
        ]
    ])

    return R


# -----------------------------
# 示例：绕 z 轴旋转 90°
# -----------------------------

axis = np.array([0.0, 0.0, 1.0])
theta = np.deg2rad(90)

q = axis_angle_to_quaternion(axis, theta)

print("四元数 [w, x, y, z]:")
print(q)

print("\n四元数长度:")
print(np.linalg.norm(q))


# 四元数 -> 旋转矩阵
R = quaternion_to_rotation_matrix(q)

print("\n对应旋转矩阵:")
print(R)


# 用旋转矩阵旋转向量
p = np.array([1.0, 0.0, 0.0])

p_rotated = R @ p

print("\n原始向量:")
print(p)

print("\n旋转后的向量:")
print(p_rotated)