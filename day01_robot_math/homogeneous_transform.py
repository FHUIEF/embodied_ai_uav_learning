import numpy as np

theta = np.deg2rad(90)  

# B坐标系相对于W坐标系绕z轴旋转90°
R_WB = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

# B坐标系原点在W坐标系中的位置
t_WB = np.array([5.0, 3.0, 2.0])

# 构造齐次变换矩阵
T_WB = np.eye(4)
T_WB[:3, :3] = R_WB
T_WB[:3, 3] = t_WB

# 点在B坐标系中的位置
p_B = np.array([1.0, 0.0, 0.0, 1.0])  # 齐次坐标

# 转换到W坐标系
p_W = T_WB @ p_B

print("T_WB = ")
print(T_WB)

print("\np_B = ")
print(p_B)

print("\np_W = ")
print(p_W)

# 验证变换矩阵的逆，还原会原来坐标系坐标
T_BW = np.linalg.inv(T_WB)

p_B_back = T_BW @ p_W

print("\n恢复后的p_B:")
print(p_B_back)
