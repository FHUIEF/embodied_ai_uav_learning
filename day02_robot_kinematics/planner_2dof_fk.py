import numpy as np

def forward_kinemaatics(q1, q2, l1=1.0, l2=1.0):
    """
    计算机械臂末端位置
    :param q1: 关节1角度，单位：弧度
    :param q2: 关节2角度，单位：弧度
    :param l1: 连杆1长度
    :param l2: 连杆2长度
    :return: 末端位置 (x, y)
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, y])

q1 = np.deg2rad(30)  # 关节1角度
q2 = np.deg2rad(45)  # 关节2角度

p = forward_kinemaatics(q1, q2)

print("末端位置:")
print(p)