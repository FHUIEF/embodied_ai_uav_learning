# Day 1：三维机器人数学基础

## 1. 学习目标

Day 1 的核心目标，是建立后续机器人学、无人机、ROS2、SLAM、计算机视觉都会反复使用的三维空间数学基础。

本日重点：

- 向量与坐标系
- **三维旋转矩阵**
- 主动旋转与**坐标系变换**
- **齐次变换矩阵**
- **欧拉角**
- **四元数**
- 三维坐标系可视化

最终应能够理解：

\[
{}^Wp = {}^WR_B {}^Bp + {}^Wp_B
\]

即：同一个点如何从机体系 \(B\) 转换到世界系 \(W\)。

---

## 2. 向量和坐标不是一回事

空间中的一个物理点 \(P\) 本身不会因为更换坐标系而改变，但它的坐标表示会发生变化。

例如：

\[
{}^Wp =
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
\]

表示点 \(P\) 在世界坐标系 \(W\) 下的坐标。

而：

\[
{}^Bp =
\begin{bmatrix}
2\\
-1\\
3
\end{bmatrix}
\]

表示同一个点在机体坐标系 \(B\) 下的坐标。

关键认识：

> 同一个物理点，在不同坐标系下可以有不同的坐标表示。

机器人学中常见：

- \(W\)：World Frame，**世界坐标系**
- \(B\)：Body Frame，**机体坐标系**
- \(C\)：Camera Frame，**相机坐标系**

---

## 3. 二维旋转矩阵

二维旋转矩阵：

\[
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
\]

若采用列向量：

\[
p' = Rp
\]

例如：

\[
p=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\theta = 90^\circ
\]

则：

\[
p'=
\begin{bmatrix}
0\\
1
\end{bmatrix}
\]

---

## 4. 三维基本旋转矩阵

### 4.1 绕 x 轴旋转

\[
R_x(\theta)=
\begin{bmatrix}
1&0&0\\
0&\cos\theta&-\sin\theta\\
0&\sin\theta&\cos\theta
\end{bmatrix}
\]

绕 x 轴旋转时，x 分量保持不变，y-z 平面内的分量旋转。

### 4.2 绕 y 轴旋转

\[
R_y(\theta)=
\begin{bmatrix}
\cos\theta&0&\sin\theta\\
0&1&0\\
-\sin\theta&0&\cos\theta
\end{bmatrix}
\]

绕 y 轴旋转时，y 分量保持不变，x-z 平面内的分量旋转。

### 4.3 绕 z 轴旋转

\[
R_z(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta&0\\
\sin\theta&\cos\theta&0\\
0&0&1
\end{bmatrix}
\]

绕 z 轴旋转时，z 分量保持不变，x-y 平面内的分量旋转。

---

## 5. 绕某个轴旋转的空间直觉

一个向量绕某个轴旋转时：

- **与旋转轴之间的夹角保持不变**
- **向量长度保持不变**
- 沿旋转轴方向的分量不变
- 垂直于旋转轴方向的分量绕轴转圈

例如：

\[
p=
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix}
\]

### 绕 x 轴旋转 90°

因为向量本身就在 x 轴上：

\[
[1,0,0]^T
\rightarrow
[1,0,0]^T
\]

### 绕 y 轴旋转 90°

根据右手定则：

\[
[1,0,0]^T
\rightarrow
[0,0,-1]^T
\]

### 绕 z 轴旋转 90°

\[
[1,0,0]^T
\rightarrow
[0,1,0]^T
\]

总结：

| 原向量 | 旋转 | 结果 |
|---|---|---|
| \([1,0,0]^T\) | 绕 x 轴 \(90^\circ\) | \([1,0,0]^T\) |
| \([1,0,0]^T\) | 绕 y 轴 \(90^\circ\) | \([0,0,-1]^T\) |
| \([1,0,0]^T\) | 绕 z 轴 \(90^\circ\) | \([0,1,0]^T\) |

---

## 6. 旋转矩阵的三个重要性质

### 6.1 正交性

\[
R^TR=I
\]

因此：

\[
R^{-1}=R^T
\]

### 6.2 行列式

\[
\det(R)=1
\]

### 6.3 保持向量长度

\[
\|Rp\|=\|p\|
\]

说明旋转只改变方向，不改变长度。

---

## 7. 旋转矩阵的列向量含义

若：

\[
{}^WR_B =
\begin{bmatrix}
|&|&|\\
{}^Wx_B&{}^Wy_B&{}^Wz_B\\
|&|&|
\end{bmatrix}
\]

则：

- 第一列：**B 坐标系 x 轴**在 W 坐标系中的方向
- 第二列：**B 坐标系 y 轴**在 W 坐标系中的方向
- 第三列：**B 坐标系 z 轴**在 W 坐标系中的方向

因此：

\[
{}^Wp = {}^WR_B {}^Bp
\]

表示**把点/向量从 B 系表示转换为 W 系表示。**

---

## 8. 主动旋转和被动旋转

### 主动旋转

向量本身真的发生旋转：

\[
p' = Rp
\]

### 被动旋转 / 坐标系变换

物理点不动，只改变描述它所使用的坐标系：

\[
{}^Wp = {}^WR_B{}^Bp
\]

机器人学中更常见的是第二种情况。

---

## 9. 为什么只有旋转矩阵还不够

旋转矩阵只能描述方向关系，不能描述坐标系原点之间的平移。

若 B 坐标系原点在 W 坐标系中的位置为：

\[
{}^Wp_B =
\begin{bmatrix}
5\\
3\\
2
\end{bmatrix}
\]

那么：

\[
\boxed{
{}^Wp = {}^WR_B {}^Bp + {}^Wp_B
}
\]

即：

1. 先旋转
2. 再平移

---

## 10. 齐次变换矩阵

为了将：

\[
Rp+t
\]

统一成一个矩阵乘法，引入齐次坐标。

三维点：

\[
p=
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
\]

扩展为：

\[
\bar p=
\begin{bmatrix}
x\\y\\z\\1
\end{bmatrix}
\]

定义：

\[
T=
\begin{bmatrix}
R&t\\
0&1
\end{bmatrix}
\]

因此：

\[
\boxed{
{}^W\bar p = {}^WT_B {}^B\bar p
}
\]

齐次变换矩阵是 \(4\times 4\) 矩阵。

### 变换矩阵的逆

\[
T^{-1}=
\begin{bmatrix}
R^T&-R^Tt\\
0&1
\end{bmatrix}
\]

注意平移部分不是简单的 \(-t\)，而是：

\[
-R^Tt
\]

---

## 11. 欧拉角

无人机中最常见：

- **Roll：绕 x 轴**
- **Pitch：绕 y 轴**
- **Yaw：绕 z 轴**

常见 ZYX 顺序：

\[
R=
R_z(\psi)
R_y(\theta)
R_x(\phi)
\]

其中：

- \(\phi\)：roll
- \(\theta\)：pitch
- \(\psi\)：yaw

三维旋转一般不满足交换律：

\[
R_xR_y \neq R_yR_x
\]

因此旋转顺序非常重要。

### 欧拉角的缺点

欧拉角存在**万向节锁**（Gimbal Lock），因此机器人内部计算常使用四元数。

---

## 12. 四元数

四元数常写为：

\[
q=
\begin{bmatrix}
w\\x\\y\\z
\end{bmatrix}
\]

单位四元数满足：

\[
w^2+x^2+y^2+z^2=1
\]

若绕单位轴：

\[
u=
\begin{bmatrix}
u_x\\u_y\\u_z
\end{bmatrix}
\]

旋转角度为 \(\theta\)，对应：

\[
q=
\begin{bmatrix}
\cos(\theta/2)\\
u_x\sin(\theta/2)\\
u_y\sin(\theta/2)\\
u_z\sin(\theta/2)
\end{bmatrix}
\]

例如绕 z 轴 90°：

\[
q=
\begin{bmatrix}
0.7071\\
0\\
0\\
0.7071
\end{bmatrix}
\]

Day 1 需要理解：

\[
\text{Axis-Angle}
\leftrightarrow
\text{Quaternion}
\leftrightarrow
\text{Rotation Matrix}
\]

---

## 13. 三种姿态表示方法

| 表示方式 | 优点 | 缺点 |
|---|---|---|
| 欧拉角 | 直观，适合人理解 | 存在万向节锁 |
| 旋转矩阵 | 坐标变换方便 | 9 个元素，存在冗余 |
| 四元数 | 紧凑、数值稳定 | 不够直观 |

实际机器人系统中常见：

- 欧拉角：用于显示
- 四元数：用于状态表示
- 旋转矩阵：用于坐标变换

---

## 14. 本日代码文件

```text
day01_robot_math/
├── rotation_matrix.py
├── homogeneous_transform.py
├── euler_angle.py
├── quaternion_demo.py
└── frame_visualization.py
```

### rotation_matrix.py

完成：

- \(R_x,R_y,R_z\)
- 向量旋转
- 验证 \(R^TR=I\)
- 验证 \(\det(R)=1\)
- 验证长度不变

### homogeneous_transform.py

完成：

\[
{}^Wp = {}^WR_B{}^Bp+t
\]

以及：

\[
T=
\begin{bmatrix}
R&t\\
0&1
\end{bmatrix}
\]

### euler_angle.py

验证：

\[
R_zR_yR_x\neq R_xR_yR_z
\]

### quaternion_demo.py

完成：

- 轴角转四元数
- 四元数转旋转矩阵
- 用旋转矩阵旋转向量
- 验证与直接使用 \(R_z\) 一致

### frame_visualization.py

三维显示：

- 世界坐标系 W
- 机体坐标系 B
- B 系中的一点 P
- P 转换到 W 系的位置

---

## 15. Day 1 核心公式

\[
\boxed{
p'=Rp
}
\]

\[
\boxed{
{}^Wp={}^WR_B{}^Bp
}
\]

\[
\boxed{
{}^Wp={}^WR_B{}^Bp+{}^Wp_B
}
\]

\[
\boxed{
{}^WT_B=
\begin{bmatrix}
{}^WR_B&{}^Wp_B\\
0&1
\end{bmatrix}
}
\]

---

## 16. 复习检查

学完后应能够独立回答：

1. 为什么同一个物理点在两个坐标系中的坐标不同？
2. \(R_x,R_y,R_z\) 分别改变哪些分量？
3. 为什么绕自身方向轴旋转，向量不发生变化？
4. 为什么 \(R^{-1}=R^T\)？
5. 旋转矩阵三列分别表示什么？
6. 为什么需要齐次变换矩阵？
7. 为什么齐次变换矩阵是 \(4\times4\)？
8. 欧拉角为什么存在旋转顺序问题？
9. 欧拉角、旋转矩阵、四元数本质上描述的是否是同一个姿态？
10. 为什么机器人内部常使用四元数？
