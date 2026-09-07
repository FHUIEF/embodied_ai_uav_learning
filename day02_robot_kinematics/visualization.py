"""FK可视化，后续用齐次变换重新做Fk，"""
import numpy as np
import matplotlib.pyplot as plt

l1 = 1.0
l2 = 1.0

q1 = np.deg2rad(30)  # 关节1角度
q2 = np.deg2rad(45)  # 关节2角度

# base
p0 = np.array([0.0, 0.0])

# joint1
p1 = np.array([
    l1 * np.cos(q1), 
    l1 * np.sin(q1)
])

# end effector
p2 = np.array([
    l1 * np.cos(q1) + l2 * np.cos(q1 + q2), 
    l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
])

x = [p0[0], p1[0], p2[0]]
y = [p0[1], p1[1], p2[1]]

plt.plot(x, y, marker='o')
plt.axis('equal')
plt.grid()

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)

plt.xlabel('x')
plt.ylabel('y')

plt.show()