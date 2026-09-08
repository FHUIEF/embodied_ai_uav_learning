# Day 5：3DOF 平面机械臂、位姿控制与抓取位姿规划

## 1. 学习目标

Day 5 的核心目标，是解决 Day 4 暴露出来的一个关键问题：

> 2DOF 平面机械臂可以控制末端位置 \((x,y)\)，但无法独立控制末端姿态 \(\phi\)。

在抓取任务中，单纯“到达物体位置”并不够，还必须控制夹爪的朝向。因此 Day 5 将机械臂升级为 3DOF，并建立完整的：

\[
\boxed{
(x_d,y_d,\phi_d)
\rightarrow
IK
\rightarrow
q_d
\rightarrow
PD
\rightarrow
MuJoCo
\rightarrow
(x,y,\phi)
}
\]

学习内容包括：

- 3DOF 平面机械臂建模
- 3DOF 正运动学 FK
- 3DOF 解析逆运动学 IK
- wrist center 的概念
- elbow-up / elbow-down 多解
- 3 关节 PD 控制
- 末端 Pose Control
- Position Error 与 Orientation Error
- 角度归一化 `wrap_to_pi`
- Pre-grasp Pose
- Approach Direction
- Grasp Site / TCP
- 两指夹爪与真实接触抓取尝试
- Day 5 中暴露出的路径与轨迹问题

---

## 2. Day 5 的总体知识链

```text
3DOF Robot Model
        ↓
Forward Kinematics
        ↓
(q1,q2,q3) → (x,y,phi)
        ↓
Inverse Kinematics
        ↓
(x,y,phi) → (q1,q2,q3)
        ↓
Joint-space PD
        ↓
Pose Control
        ↓
Pre-grasp
        ↓
Approach
        ↓
Physical Gripper
```

相比 Day 4，最重要的升级是：

\[
\boxed{
\text{Position Control}
\rightarrow
\text{Pose Control}
}
\]

---

## 3. 为什么要从 2DOF 升级到 3DOF？

对于 2DOF 平面机械臂：

\[
q=
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix}
\]

末端位置由 \(q_1,q_2\) 决定，而末端姿态：

\[
\phi=q_1+q_2
\]

只有两个自由度时，通常只能独立满足两个任务变量 \(x,y\)。一旦位置确定，姿态也被一起决定。

抓取通常至少需要：

\[
\boxed{x,y,\phi}
\]

因此增加第三个关节：

\[
q=
\begin{bmatrix}
q_1\\q_2\\q_3
\end{bmatrix}
\]

并有：

\[
\boxed{\phi=q_1+q_2+q_3}
\]

第三关节就可以用于末端姿态补偿。

---

## 4. 3DOF 平面机械臂结构

本日基础模型：

\[
l_1=1.0,\qquad l_2=0.8,\qquad l_3=0.3
\]

```text
base
  ↓
joint1
  ↓
link1
  ↓
joint2
  ↓
link2
  ↓
joint3
  ↓
link3
  ↓
end-effector
```

基础模型中：

```text
nq = 3
nv = 3
nu = 3
```

分别表示 3 个位置自由度、3 个速度自由度和 3 个 actuator。

---

## 5. `01_load_3dof_arm.py`

目标：

- 加载 3DOF MuJoCo 模型
- 检查 `nq/nv/nu`
- 设置初始关节角
- 观察末端姿态

例如：

```python
data.qpos[:] = np.deg2rad([20.0, 40.0, -60.0])
```

则：

\[
\phi=20^\circ+40^\circ-60^\circ=0^\circ
\]

第三根连杆应保持水平。

关键认识：

\[
\boxed{q_3\text{ 可以补偿前两个关节形成的末端姿态}}
\]

---

## 6. `02_3dof_fk.py`：3DOF 正运动学

目标：

\[
\boxed{(q_1,q_2,q_3)\rightarrow(x,y,\phi)}
\]

各连杆相对于世界坐标系的绝对角度：

\[
\theta_1=q_1
\]

\[
\theta_2=q_1+q_2
\]

\[
\theta_3=q_1+q_2+q_3
\]

因此：

\[
\boxed{\phi=q_1+q_2+q_3}
\]

末端位置：

\[
\boxed{x=l_1\cos q_1+l_2\cos(q_1+q_2)+l_3\cos(q_1+q_2+q_3)}
\]

\[
\boxed{y=l_1\sin q_1+l_2\sin(q_1+q_2)+l_3\sin(q_1+q_2+q_3)}
\]

末端 Pose：

\[
\boxed{
\xi=
\begin{bmatrix}
x\\y\\\phi
\end{bmatrix}}
\]

---

## 7. 关节位置的几何计算

Base：

\[
P_0=\begin{bmatrix}0\\0\end{bmatrix}
\]

Joint 2：

\[
P_1=
\begin{bmatrix}
l_1\cos q_1\\
l_1\sin q_1
\end{bmatrix}
\]

Joint 3：

\[
P_2=P_1+
\begin{bmatrix}
l_2\cos(q_1+q_2)\\
l_2\sin(q_1+q_2)
\end{bmatrix}
\]

末端：

\[
P_3=P_2+
\begin{bmatrix}
l_3\cos(q_1+q_2+q_3)\\
l_3\sin(q_1+q_2+q_3)
\end{bmatrix}
\]

这部分用于几何验证和 Matplotlib 可视化。

---

## 8. `03_3dof_ik.py`：3DOF 解析逆运动学

目标：

\[
\boxed{(x_d,y_d,\phi_d)\rightarrow(q_1,q_2,q_3)}
\]

核心思想不是直接一次求三个关节，而是先求 wrist center，把问题退化成 2DOF IK。

---

## 9. Wrist Center

目标末端 Pose：

\[
(x_d,y_d,\phi_d)
\]

去掉最后一段连杆贡献：

\[
\boxed{x_w=x_d-l_3\cos\phi_d}
\]

\[
\boxed{y_w=y_d-l_3\sin\phi_d}
\]

这样前两根连杆只需要到达 \((x_w,y_w)\)。

---

## 10. 求解 \(q_2\)

由余弦定理：

\[
\boxed{
\cos q_2=
\frac{x_w^2+y_w^2-l_1^2-l_2^2}{2l_1l_2}
}
\]

若：

\[
|\cos q_2|>1
\]

则该目标 Pose 不可达。

---

## 11. Elbow-up / Elbow-down

两组典型解：

\[
q_2=+\arccos(\cos q_2)
\]

与：

\[
q_2=-\arccos(\cos q_2)
\]

对应不同的肘部构型。

因此：

\[
\boxed{\text{同一个末端 Pose 可能对应多组关节角}}
\]

---

## 12. 求解 \(q_1\)

\[
\boxed{
q_1=
\operatorname{atan2}(y_w,x_w)
-
\operatorname{atan2}
\left(l_2\sin q_2,l_1+l_2\cos q_2\right)
}
\]

---

## 13. 求解 \(q_3\)

因为：

\[
\phi_d=q_1+q_2+q_3
\]

所以：

\[
\boxed{q_3=\phi_d-q_1-q_2}
\]

完整流程：

```text
(xd, yd, phid)
      ↓
Wrist Center
      ↓
(xw, yw)
      ↓
2DOF IK
      ↓
q1, q2
      ↓
q3 = phid - q1 - q2
```

---

## 14. 位置可达不等于 Pose 可达

即使 \((x_d,y_d)\) 在机械臂总工作空间内，指定 \(\phi_d\) 后得到的 wrist center 仍可能不可达。

因此：

\[
\boxed{\text{Position Reachable}\not\Rightarrow\text{Pose Reachable}}
\]

---

## 15. IK 后必须进行 FK 验证

求得：

\[
q^*
\]

以后应验证：

\[
q^*\xrightarrow{FK}(x',y',\phi')
\]

检查：

\[
(x',y',\phi')\approx(x_d,y_d,\phi_d)
\]

这是排查以下问题的重要方法：

- IK 推导错误
- 坐标系错误
- rad / deg 错误
- 连杆长度与 XML 不一致
- 角度符号错误

---

## 16. `04_joint_pd_control.py`：3DOF 关节 PD

先直接指定：

\[
q_d=
\begin{bmatrix}
20^\circ\\40^\circ\\-60^\circ
\end{bmatrix}
\]

PD 控制：

\[
\boxed{\tau=K_p(q_d-q)-K_d\dot q}
\]

典型参数：

```python
Kp = np.array([20.0, 20.0, 15.0])
Kd = np.array([10.0, 10.0, 6.0])
```

目标是验证：

\[
q(t)\rightarrow q_d
\]

以及：

\[
\dot q(t)\rightarrow0
\]

---

## 17. 3DOF 动力学耦合

机械臂动力学：

\[
\boxed{M(q)\ddot q+C(q,\dot q)\dot q+g(q)=\tau}
\]

其中 \(M(q)\) 一般不是对角矩阵，因此不同关节之间存在耦合。

当前使用 independent joint PD，并没有显式补偿这些耦合，但在低速和简单任务中仍然可以获得较好效果。

---

## 18. PD 参数理解

\(K_p\) 主要影响：

- 响应速度
- 位置误差

\(K_d\) 主要影响：

- 阻尼
- 超调
- 振荡

控制力矩还必须满足 actuator saturation，因此使用：

```python
np.clip(...)
```

---

## 19. `05_pose_control.py`：完整 Pose Control

目标不再直接给关节角，而是：

```python
x_target = 1.2
y_target = 0.8
phi_target = np.deg2rad(0.0)
```

控制链：

\[
\boxed{
(x_d,y_d,\phi_d)
\rightarrow IK
\rightarrow q_d
\rightarrow PD
\rightarrow\tau
\rightarrow MuJoCo
}
\]

这就是本日最重要的综合程序。

---

## 20. Joint-space Error 与 Task-space Error

PD 控制器使用：

\[
\boxed{e_q=q_d-q}
\]

任务评价使用位置误差：

\[
\boxed{e_p=p_d-p}
\]

以及姿态误差：

\[
\boxed{e_\phi=\phi_d-\phi}
\]

因此：

> 控制器工作在 Joint Space，而任务最终在 Task Space 中评价。

---

## 21. 角度误差的周期性

例如：

\[
\phi=179^\circ,
\qquad
\phi_d=-179^\circ
\]

直接相减得到 \(-358^\circ\)，但真实最短误差只有 \(2^\circ\)。

因此需要：

```python
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
```

将角度限制到：

\[
\boxed{[-\pi,\pi]}
\]

---

## 22. Pose Reach 判定

Day 4 只要求：

\[
\|p_d-p\|<\epsilon_p
\]

Day 5 还要同时满足：

\[
\boxed{|e_\phi|<\epsilon_\phi}
\]

例如：

\[
\epsilon_p=0.01m
\]

\[
\epsilon_\phi=1^\circ
\]

并继续使用 stable counter，避免仅仅瞬间经过目标就判定成功。

---

## 23. `06_pregrasp_control.py`：Pre-grasp Pose

抓取不应该直接：

```text
EE → Cube Center
```

更合理的流程：

```text
Pre-grasp
    ↓
Approach
    ↓
Grasp Pose
```

---

## 24. Approach Direction

若希望抓取姿态为 \(\phi_g\)，则接近方向：

\[
\boxed{
\hat d=
\begin{bmatrix}
\cos\phi_g\\
\sin\phi_g
\end{bmatrix}}
\]

---

## 25. Pre-grasp Pose

物体位置：

\[
p_o=
\begin{bmatrix}
x_o\\y_o
\end{bmatrix}
\]

预抓取距离：

\[
d_{pre}
\]

则：

\[
\boxed{p_{pre}=p_o-d_{pre}\hat d}
\]

即：

\[
x_{pre}=x_o-d_{pre}\cos\phi_g
\]

\[
y_{pre}=y_o-d_{pre}\sin\phi_g
\]

Pre-grasp Pose：

\[
\boxed{\xi_{pre}=(x_{pre},y_{pre},\phi_g)}
\]

Grasp Pose：

\[
\boxed{\xi_g=(x_o,y_o,\phi_g)}
\]

---

## 26. Grasp 的几何意义

抓取至少需要回答：

1. 抓哪里？—— \(p_{grasp}\)
2. 朝什么方向？—— \(\phi_{grasp}\)
3. 从哪里靠近？—— \(p_{pregrasp}\)

因此：

\[
\boxed{Grasp=Position+Orientation+Approach}
\]

---

## 27. `07_3dof_gripper_grasp.py`

Day 5 最后尝试将 Day 4 的两指夹爪重新接入 3DOF 机械臂。

状态机：

```text
MOVE_TO_PREGRASP
        ↓
APPROACH
        ↓
HOLD
        ↓
CLOSE_GRIPPER
        ↓
VERIFY_GRASP
        ↓
DONE
```

这一版只要求“夹住并保持”，暂时不做搬运。

---

## 28. 为什么增加 HOLD？

如果机械臂还在运动时夹爪同时闭合：

```text
arm moving
+
fingers closing
```

很容易把物体推开。

因此使用：

```text
APPROACH
↓
HOLD
↓
CLOSE
```

先稳定末端，再闭合夹爪。

---

## 29. Grasp Site 与 TCP

不能简单让 `gripper_base` 中心去对准物体，因为 gripper base 是实际碰撞实体。

因此定义：

```xml
<site name="grasp_site" ... />
```

并控制：

\[
\boxed{p_{grasp\_site}\rightarrow p_{object}}
\]

这个概念已经接近真实机器人中的：

\[
\boxed{TCP\quad Tool\ Center\ Point}
\]

---

## 30. 数学模型与 MuJoCo 几何必须一致

例如：

```text
joint3
→ link3 = 0.30 m
→ grasp_site = 0.20 m
```

则有效第三段长度：

\[
l_3=0.30+0.20=0.50m
\]

否则：

\[
\boxed{FK_{math}\neq p_{grasp\_site}^{MuJoCo}}
\]

此时 IK 即使数学正确，真实工具点也不会到正确位置。

---

## 31. 真实抓取需要 Contact Verification

距离近不代表抓住。

第一版真实抓取条件：

\[
\boxed{Contact_L\land Contact_R}
\]

即左右两根手指都与 cube 接触，并保持一定时间。

---

## 32. 为什么 `data.ncon` 不够？

`data.ncon` 只表示整个场景当前有多少个 contact。

它无法告诉我们到底是谁与谁碰撞。

因此必须检查：

```python
data.contact[i].geom1
data.contact[i].geom2
```

确定具体的 geometry pair。

---

## 33. Day 5 中真实抓取遇到的新问题

即使升级为 3DOF，仍出现：

> 夹爪挡板或 finger 在 APPROACH 过程中先碰到物体，把物体推开。

这说明：

\[
\boxed{3DOF\ Pose\ Control}
\]

已经解决“姿态自由度不足”，但没有自动解决：

\[
\boxed{\text{运动过程中的路径与碰撞问题}}
\]

---

## 34. 为什么最终 Pose 正确仍然可能失败？

当前控制方式通常是：

```text
Pregrasp Pose
↓
突然切换目标
↓
Grasp Pose
```

然后 Joint-space PD 从一个关节目标运动到另一个关节目标。

最终目标 Pose 虽然正确，但中间实际路径 \(p(t)\) 没有被明确设计。

所以可能发生：

```text
正确终点
+
错误中间路径
=
碰撞 / 推物
```

因此：

\[
\boxed{\text{Endpoint Correct}\not\Rightarrow\text{Path Correct}}
\]

这是 Day 5 最重要的新认识之一。

---

## 35. 简单线性插值的思路

一个最简单的连续接近方法是：

\[
\boxed{
p_d(\alpha)=
(1-\alpha)p_{pre}
+
\alpha p_{grasp}
}
\]

其中：

\[
\alpha\in[0,1]
\]

这样就可以生成连续中间目标点，而不是直接跳目标。

但本日暂不继续深入，因为该问题已经自然进入轨迹生成和轨迹跟踪范畴。

---

## 36. Day 5 暂时保留的真实抓取问题

真实夹爪问题暂时保留，不继续硬调参数。

原因：

> 当前困难已经不再主要是 IK、PD 或自由度问题，而是 Path / Trajectory / Collision 问题。

后续更合理的学习顺序：

```text
Day 5
Pose Control
        ↓
Day 6
Jacobian / Continuous Motion
        ↓
Trajectory Generation
        ↓
Trajectory Tracking
        ↓
Motion Planning
        ↓
重新解决 Grasp Approach
```

---

## 37. Day 5 最重要的失败案例

Day 4 的主要问题：

\[
\boxed{\text{姿态不可独立控制}}
\]

Day 5 已经解决：

\[
(x,y,\phi)
\]

可以同时控制。

但又发现：

\[
\boxed{\text{最终 Pose 正确，但 Approach Path 仍可能错误}}
\]

说明问题已经从“位置/姿态控制”自然升级到了“连续运动控制”。

---

## 38. Day 5 的机器人控制层级

### Task Layer

```text
Pregrasp
Approach
Hold
Close
Verify
```

### Kinematics Layer

```text
FK
IK
Pose
TCP
```

### Control Layer

```text
Joint-space PD
Torque
```

### Physics Layer

```text
Dynamics
Contact
Collision
Friction
```

接下来还要增加：

### Motion Generation Layer

```text
Path
Trajectory
Velocity
Acceleration
```

---

## 39. 与强化学习 / 具身智能的关系

当前经典控制：

```text
Desired Pose
↓
IK
↓
Joint PD
↓
Action
```

以后 RL 可能直接输出：

- joint torque
- joint position target
- joint velocity target
- Cartesian delta pose

例如：

\[
s=[q,\dot q,p_{object},p_{target}]
\]

\[
a=\pi_\theta(s)
\]

但即使未来使用 RL，也仍然必须理解：

- kinematics
- pose
- trajectory
- collision
- contact
- gripper geometry

否则很难判断策略为什么失败。

---

## 40. Day 5 项目结构

```text
day05_3dof_grasp_pose/
├── assets/
│   ├── three_link_arm.xml
│   └── three_link_gripper_scene.xml
│
├── 01_load_3dof_arm.py
├── 02_3dof_fk.py
├── 03_3dof_ik.py
├── 04_joint_pd_control.py
├── 05_pose_control.py
├── 06_pregrasp_control.py
├── 07_3dof_gripper_grasp.py
└── README.md
```

---

## 41. 各程序作用

### `01_load_3dof_arm.py`

建立 3DOF MuJoCo 机械臂，并理解：

\[
\phi=q_1+q_2+q_3
\]

### `02_3dof_fk.py`

完成：

\[
\boxed{q\rightarrow(x,y,\phi)}
\]

### `03_3dof_ik.py`

完成：

\[
\boxed{(x,y,\phi)\rightarrow q}
\]

重点包括 wrist center、多解、Pose 可达性和 FK verification。

### `04_joint_pd_control.py`

完成：

\[
\boxed{q_d\rightarrow q}
\]

验证 3DOF 关节控制层。

### `05_pose_control.py`

完成：

\[
\boxed{(x_d,y_d,\phi_d)\rightarrow IK\rightarrow PD\rightarrow Actual\ Pose}
\]

### `06_pregrasp_control.py`

学习：

- grasp direction
- pregrasp pose
- grasp pose
- FSM 状态转换

### `07_3dof_gripper_grasp.py`

尝试：

- 3DOF Pose Control
- two-finger gripper
- HOLD
- CLOSE
- contact verification

当前保留问题：Approach 阶段的中间路径可能导致推物。

---

## 42. Day 5 核心公式汇总

### FK

\[
\boxed{x=l_1\cos q_1+l_2\cos(q_1+q_2)+l_3\cos(q_1+q_2+q_3)}
\]

\[
\boxed{y=l_1\sin q_1+l_2\sin(q_1+q_2)+l_3\sin(q_1+q_2+q_3)}
\]

\[
\boxed{\phi=q_1+q_2+q_3}
\]

### Wrist Center

\[
\boxed{x_w=x_d-l_3\cos\phi_d}
\]

\[
\boxed{y_w=y_d-l_3\sin\phi_d}
\]

### IK

\[
\boxed{
\cos q_2=
\frac{x_w^2+y_w^2-l_1^2-l_2^2}{2l_1l_2}
}
\]

\[
\boxed{
q_1=
\operatorname{atan2}(y_w,x_w)
-
\operatorname{atan2}(l_2\sin q_2,l_1+l_2\cos q_2)
}
\]

\[
\boxed{q_3=\phi_d-q_1-q_2}
\]

### Joint PD

\[
\boxed{\tau=K_p(q_d-q)-K_d\dot q}
\]

### Position Error

\[
\boxed{e_p=p_d-p}
\]

### Orientation Error

\[
\boxed{e_\phi=\operatorname{wrapToPi}(\phi_d-\phi)}
\]

### Approach Direction

\[
\boxed{
\hat d=
\begin{bmatrix}
\cos\phi_g\\
\sin\phi_g
\end{bmatrix}}
\]

### Pre-grasp

\[
\boxed{p_{pre}=p_o-d_{pre}\hat d}
\]

---

## 43. Day 5 复习问题

学完以后，应能回答：

1. 为什么 2DOF 无法独立控制 \(x,y,\phi\)？
2. 为什么增加第三关节后可以控制末端姿态？
3. 为什么三根连杆绝对角度依次是 \(q_1\)、\(q_1+q_2\)、\(q_1+q_2+q_3\)？
4. wrist center 是什么？
5. 为什么 3DOF IK 可以退化成 2DOF IK？
6. 为什么 \(q_3=\phi_d-q_1-q_2\)？
7. elbow-up 和 elbow-down 有什么区别？
8. 为什么同一个 Pose 可能对应多组关节角？
9. 为什么位置可达不代表 Pose 可达？
10. 为什么 IK 后需要 FK verification？
11. Joint-space PD 使用什么误差？
12. Task-space position error 有什么作用？
13. 为什么 orientation error 要 wrap 到 \([-\pi,\pi]\)？
14. 什么叫 Pose Control？
15. Pose Control 和 Task-space Control 是否完全相同？
16. 什么叫 Pre-grasp Pose？
17. Approach Direction 如何由 \(\phi\) 得到？
18. 什么是 grasp site？
19. grasp site 和 TCP 有什么关系？
20. 为什么数学 \(l_3\) 必须和 XML 中 grasp site 的实际位置一致？
21. 为什么真实抓取不能只根据距离判断成功？
22. 为什么 `data.ncon` 不能直接证明抓取成功？
23. HOLD 状态有什么意义？
24. 为什么 3DOF 已经解决姿态问题，抓取仍可能失败？
25. 为什么“最终 Pose 正确”不代表“中间路径正确”？
26. 什么情况下需要轨迹生成而不是单目标点控制？
27. 线性插值 \(p(\alpha)\) 有什么作用？
28. Day 5 的问题为什么自然引出 Day 6 的 Jacobian 与连续运动控制？

---

## 44. Day 1 ～ Day 5 知识链

```text
Day 1
Coordinate Frame
Rotation
Transformation
Quaternion
        ↓
Day 2
FK
IK
Jacobian 基础
        ↓
Day 3
MuJoCo
Actuator
Dynamics
PD
        ↓
Day 4
2DOF Manipulation
Reach
FSM
Pick-and-Place
Contact
        ↓
Day 5
3DOF
Pose
Pose IK
Pose Control
Pre-grasp
Grasp Geometry
        ↓
Day 6
Jacobian
Velocity Control
Continuous Motion
```

---

## 45. Day 5 最终结论

Day 5 已经建立：

\[
\boxed{
(x_d,y_d,\phi_d)
\rightarrow
IK
\rightarrow
q_d
\rightarrow
PD
\rightarrow
MuJoCo
\rightarrow
(x,y,\phi)
}
\]

即完整的 3DOF Pose Control。

同时进一步认识到：

\[
\boxed{Grasp\neq Reach}
\]

以及：

\[
\boxed{Correct\ Pose\neq Correct\ Motion\ Path}
\]

因此 Day 5 的真实夹爪问题暂时保留，等学习 Jacobian、路径规划、轨迹生成与轨迹跟踪以后再回来解决稳定抓取。
