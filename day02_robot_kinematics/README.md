# Day 2：2DOF 平面机械臂运动学

## 1. 学习目标

Day 2 从三维空间数学正式进入机器人运动学。

核心问题：

1. 已知关节角，机械臂末端在哪里？
2. 已知末端目标位置，关节角应该是多少？
3. 关节转动速度如何映射为末端速度？

分别对应：

\[
\boxed{FK,\ IK,\ Jacobian}
\]

本日重点：

- 2DOF 平面机械臂建模
- Forward Kinematics 正运动学
- Inverse Kinematics 逆运动学
- Jacobian
- 工作空间
- IK 多解
- 奇异位形

---

## 2. 2DOF 平面机械臂模型

设：

- 第一根连杆长度 \(l_1\)
- 第二根连杆长度 \(l_2\)
- 第一关节角 \(q_1\)
- 第二关节角 \(q_2\)

第一根连杆相对于世界 x 轴旋转 \(q_1\)。

第二根连杆相对于第一根连杆再旋转 \(q_2\)。

因此第二根连杆相对于世界坐标系的总角度为：

\[
q_1+q_2
\]

---

## 3. 正运动学 FK

正运动学回答：

> 已知关节角 \(q\)，求末端位置 \(p\)。

记作：

\[
\boxed{
q\rightarrow p
}
\]

对于 2DOF 平面机械臂：

\[
x=l_1\cos q_1+l_2\cos(q_1+q_2)
\]

\[
y=l_1\sin q_1+l_2\sin(q_1+q_2)
\]

即：

\[
p=f(q)
\]

其中：

\[
q=
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix}
\]

\[
p=
\begin{bmatrix}
x\\
y
\end{bmatrix}
\]

---

## 4. 为什么第二根连杆是 \(q_1+q_2\)

假设：

\[
q_1=30^\circ
\]

第一根连杆相对于世界 x 轴为 \(30^\circ\)。

若：

\[
q_2=45^\circ
\]

它表示第二根连杆相对于第一根连杆再转 \(45^\circ\)。

因此：

\[
30^\circ+45^\circ=75^\circ
\]

这与 Day 1 的旋转叠加概念一致。

---

## 5. FK 与齐次变换

更一般的机械臂不会直接手写末端坐标公式，而是通过连续的坐标变换获得。

二维齐次变换可以写为：

\[
T=
\begin{bmatrix}
\cos\theta&-\sin\theta&t_x\\
\sin\theta&\cos\theta&t_y\\
0&0&1
\end{bmatrix}
\]

第一段变换：

\[
T_{01}
\]

第二段变换：

\[
T_{12}
\]

最终：

\[
\boxed{
T_{02}=T_{01}T_{12}
}
\]

这就是后续 DH 参数法的核心思想：

> 一个关节对应一个局部坐标变换，多个关节通过矩阵连续相乘。

---

## 6. 逆运动学 IK

逆运动学回答：

> 已知末端目标位置 \(p\)，求关节角 \(q\)。

即：

\[
\boxed{
p\rightarrow q
}
\]

IK 比 FK 更复杂，因为可能：

- 无解
- 唯一解
- 多解

---

## 7. IK 中 \(q_2\) 的推导

目标点：

\[
P=(x,y)
\]

原点到目标点距离：

\[
r^2=x^2+y^2
\]

根据余弦定理：

\[
r^2=l_1^2+l_2^2+2l_1l_2\cos q_2
\]

因此：

\[
\boxed{
\cos q_2=
\frac{x^2+y^2-l_1^2-l_2^2}
{2l_1l_2}
}
\]

于是：

\[
q_2=
\pm\cos^{-1}
\left(
\frac{x^2+y^2-l_1^2-l_2^2}
{2l_1l_2}
\right)
\]

正负两个解对应常见的：

- elbow-up
- elbow-down

即肘部朝上和肘部朝下。

---

## 8. IK 中 \(q_1\) 的计算

定义：

\[
\alpha=\operatorname{atan2}(y,x)
\]

\[
\beta=
\operatorname{atan2}
\left(
l_2\sin q_2,\,
l_1+l_2\cos q_2
\right)
\]

则：

\[
\boxed{
q_1=\alpha-\beta
}
\]

---

## 9. 机械臂工作空间

机械臂能够到达的位置构成工作空间。

最大距离：

\[
r_{\max}=l_1+l_2
\]

最小距离：

\[
r_{\min}=|l_1-l_2|
\]

目标点必须满足：

\[
\boxed{
|l_1-l_2|
\le
\sqrt{x^2+y^2}
\le
l_1+l_2
}
\]

否则 IK 无解。

例如：

\[
l_1=l_2=1
\]

最大只能到：

\[
r=2
\]

目标点 \((3,0)\) 不可达。

---

## 10. Jacobian

已知：

\[
p=f(q)
\]

若关节角发生微小变化：

\[
dq
\]

末端位置变化：

\[
dp
\]

两者满足：

\[
\boxed{
dp=J(q)dq
}
\]

速度形式：

\[
\boxed{
\dot p=J(q)\dot q
}
\]

Jacobian 的核心物理意义：

> 将关节速度映射为末端速度。

---

## 11. 2DOF Jacobian 推导

正运动学：

\[
x=l_1\cos q_1+l_2\cos(q_1+q_2)
\]

\[
y=l_1\sin q_1+l_2\sin(q_1+q_2)
\]

Jacobian：

\[
J=
\begin{bmatrix}
\frac{\partial x}{\partial q_1} &
\frac{\partial x}{\partial q_2}\\
\frac{\partial y}{\partial q_1} &
\frac{\partial y}{\partial q_2}
\end{bmatrix}
\]

得到：

\[
\boxed{
J=
\begin{bmatrix}
-l_1\sin q_1-l_2\sin(q_1+q_2)
&
-l_2\sin(q_1+q_2)
\\
l_1\cos q_1+l_2\cos(q_1+q_2)
&
l_2\cos(q_1+q_2)
\end{bmatrix}
}
\]

---

## 12. Jacobian 的速度映射

若：

\[
\dot q=
\begin{bmatrix}
\dot q_1\\
\dot q_2
\end{bmatrix}
\]

则：

\[
\dot p=
\begin{bmatrix}
v_x\\
v_y
\end{bmatrix}
=
J(q)\dot q
\]

即：

```text
joint velocity
      ↓
   Jacobian
      ↓
end-effector velocity
```

---

## 13. Jacobian 反向使用

若希望末端具有指定速度：

\[
\dot p_d
\]

在 \(J\) 可逆时：

\[
\boxed{
\dot q=J^{-1}\dot p_d
}
\]

这就是 Jacobian-based IK / differential IK 的基本思想。

复杂机器人中通常更多使用：

- Jacobian inverse
- Jacobian pseudo-inverse
- Damped least squares

---

## 14. 奇异位形

若：

\[
\det(J)=0
\]

则 Jacobian 不可逆，称为奇异位形。

例如 2DOF 机械臂完全伸直：

```text
O------●------●
```

此时机械臂在某些末端运动方向上的能力会丢失。

奇异位形意味着：

> 某些末端速度方向无法由有限的关节速度实现。

---

## 15. FK、IK、Jacobian 的关系

### FK

\[
\boxed{
q\rightarrow p
}
\]

解决：

> 已知关节角，末端在哪里？

### IK

\[
\boxed{
p\rightarrow q
}
\]

解决：

> 已知末端位置，关节应该转多少？

### Jacobian

\[
\boxed{
\dot p=J(q)\dot q
}
\]

解决：

> 关节速度如何映射为末端速度？

三者构成机械臂运动学的核心骨架。

---

## 16. 本日代码文件

```text
day02_robot_kinematics/
├── planar_2dof_fk.py
├── planar_2dof_ik.py
├── jacobian.py
└── visualization.py
```

### planar_2dof_fk.py

实现：

\[
(q_1,q_2)\rightarrow(x,y)
\]

### planar_2dof_ik.py

实现：

\[
(x,y)\rightarrow(q_1,q_2)
\]

并检查目标是否位于工作空间。

### jacobian.py

实现：

\[
J(q)
\]

以及：

\[
\dot p=J\dot q
\]

### visualization.py

绘制：

- base
- joint
- link1
- link2
- end effector
- target point

---

## 17. 推荐验证流程

对于目标：

\[
p_d=
\begin{bmatrix}
x_d\\
y_d
\end{bmatrix}
\]

执行：

```text
target
  ↓
IK
  ↓
q1, q2
  ↓
FK
  ↓
重新计算末端位置
```

验证：

\[
f(q_1,q_2)\approx p_d
\]

若一致，则说明 FK 与 IK 实现正确。

---

## 18. Day 2 核心公式

\[
\boxed{
x=l_1\cos q_1+l_2\cos(q_1+q_2)
}
\]

\[
\boxed{
y=l_1\sin q_1+l_2\sin(q_1+q_2)
}
\]

\[
\boxed{
\cos q_2=
\frac{x^2+y^2-l_1^2-l_2^2}{2l_1l_2}
}
\]

\[
\boxed{
\dot p=J(q)\dot q
}
\]

---

## 19. 复习检查

学完后应能够独立回答：

1. FK 和 IK 分别解决什么问题？
2. 为什么第二根连杆方向是 \(q_1+q_2\)？
3. 为什么 IK 可能有两个解？
4. 为什么目标点可能无解？
5. 工作空间如何由连杆长度决定？
6. Jacobian 是什么？
7. Jacobian 为什么是偏导矩阵？
8. \(\dot p=J\dot q\) 的物理意义是什么？
9. 什么叫奇异位形？
10. 为什么机械臂完全伸直时容易出现奇异问题？
