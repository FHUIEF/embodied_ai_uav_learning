import numpy as np
import matplotlib.pyplot as plt

theta = np.deg2rad(45)

R_WB = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

t_WB = np.array([2.0, 2.0, 1.0])

# B系中的点
p_B = np.array([1.5, 0.5, 0.5])

# 转换到W系
p_W = R_WB @ p_B + t_WB

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 世界坐标系
ax.quiver(0, 0, 0, 1, 0, 0)
ax.quiver(0, 0, 0, 0, 1, 0)
ax.quiver(0, 0, 0, 0, 0, 1)

# B坐标系三个轴在W系中的方向
# 旋转矩阵的三列，就是B坐标系三个轴在W系中的方向
x_B = R_WB[:, 0]
y_B = R_WB[:, 1]
z_B = R_WB[:, 2]

ax.quiver(*t_WB , *x_B)
ax.quiver(*t_WB , *y_B)
ax.quiver(*t_WB , *z_B)

# 点p
ax.scatter(*p_W, s=60)
ax.text(*p_W, 'p')

ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 4])

plt.show()