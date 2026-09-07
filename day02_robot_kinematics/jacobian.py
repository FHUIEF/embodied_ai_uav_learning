import numpy as np

def jacobian(q1, q2, l1=1.0, l2=1.0):
    """
    计算机械臂雅可比矩阵
    :param q1: 关节1角度，单位：弧度
    :param q2: 关节2角度，单位：弧度
    :param l1: 连杆1长度
    :param l2: 连杆2长度
    :return: 雅可比矩阵 J
    """
    J = np.array([
        [
            -l1 * np.sin(q1) - l2 * np.sin(q1 + q2),
            -l2 * np.sin(q1 + q2)
        ],
        [
            l1 * np.cos(q1) + l2 * np.cos(q1 + q2), 
            l2 * np.cos(q1 + q2)
        ]
    ])
    return J

q1 = np.deg2rad(30)  # 关节1角度
q2 = np.deg2rad(45)  # 关节2角度

dq = np.array([0.1, 0.2])  

J = jacobian(q1, q2)

dp = J @ dq  

print("Jacobian:")
print(J)

print("\n关节速度:")
print(dq)

print("\n末端速度:")
print(dp)
