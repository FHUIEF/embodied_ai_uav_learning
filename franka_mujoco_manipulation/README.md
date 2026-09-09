# Franka Panda MuJoCo 经典 Manipulation 项目复习文档

> 项目目标：基于 MuJoCo 与 Franka Panda，完整走通“模型加载 → 状态读取 → FK → 关节控制 → 3D/6D IK → TCP → 夹爪 → 物体接触 → Pregrasp → Grasp → Lift → Pick-and-Place → 平滑轨迹 → 奇异性分析 → DLS → Adaptive DLS → Null-space Control”的经典机器人 Manipulation 技术链。
>
> 本文按项目实际学习顺序整理，既包含代码中使用的知识，也包含项目过程中反复提问、调试和理解过的关键概念，目标是方便之后复习、写 README、准备面试和写简历。

---

# 1. 项目最终完成了什么

该项目最终完成了一个较完整的 Franka Panda 经典机械臂 Manipulation Pipeline：

\[
\boxed{
Object\ State
\rightarrow
Pregrasp
\rightarrow
6D\ IK
\rightarrow
Smooth\ Trajectory
\rightarrow
Approach
\rightarrow
Grasp
\rightarrow
Lift
\rightarrow
Transport
\rightarrow
Lower
\rightarrow
Settle
\rightarrow
Release
\rightarrow
Retreat
}
\]

并进一步研究了 7DOF 冗余机械臂中的：

- Jacobian 奇异性；
- Pseudoinverse IK；
- Damped Least Squares（DLS）；
- Adaptive DLS；
- Null-space singularity avoidance。

项目最终的平滑 Pick-and-Place 实验中，观测到：

- 状态机最终到达 `DONE`；
- Cube 最终稳定放置于目标区域；
- XY 放置误差约 **2.23 mm**；
- Cube 姿态误差约 **0.596°**；
- Cube 最终与桌面接触正常。

Null-space 实验中，观测到：

- \(\sigma_{\min}(J)\) 得到明显提升；
- Jacobian condition number 明显下降；
- TCP 位置误差仍保持在极小量级；
- TCP 姿态误差也极小；
- \(\|J\Delta q_{\text{null}}\|\) 接近数值 0，验证了 Null-space motion 对主任务的一阶影响几乎为零。

---

# 2. 项目文件与学习路线

项目按学习顺序形成：

```text
franka_mujoco_manipulation/
│
├── 01_load_franka.py
├── 02_read_robot_state.py
├── 03_joint_fk_experiment.py
├── 04_joint_position_control.py
├── 05_end_effector_state.py
├── 06_cartesian_reach.py
├── 07_pose_reach.py
├── 08_gripper_control.py
├── 09_pick_cube_scene.py
├── 10_pregrasp_target.py
├── 11_approach_grasp.py
├── 12_lift_cube.py
├── 13_pick_and_place.py
├── 14_smooth_trajectory.py
├── 15_dls_ik_compare.py
├── 16_adaptive_dls_ik.py
├── 17_null_space_singularity_avoidance.py
│
├── assets/
│   └── franka_pick_cube/
│       ├── assets/
│       ├── panda.xml
│       ├── scene.xml
│       └── README.md
│
├── mujoco_menagerie/
│   └── franka_emika_panda/
│
├── controllers/
├── tasks/
├── utils/
├── results/
├── README.md
└── requirements.txt
```

其中可以把 01~17 按知识层次重新理解为：

```text
第一层：MuJoCo / Robot Model
01 02 03 04 05

第二层：Kinematics / IK
06 07

第三层：Manipulation
08 09 10 11 12 13

第四层：Trajectory Generation
14

第五层：Redundancy / Singularity
15 16 17
```

---

# 3. MuJoCo 的核心数据结构：model 与 data

这是整个项目最基础、也最重要的概念之一。

## 3.1 `MjModel`

```python
model = mujoco.MjModel.from_xml_path(xml_path)
```

可以理解为：

> 机器人的“设计图纸”和固定物理结构。

包括：

- body 数量；
- joint 类型；
- joint range；
- actuator；
- mass；
- inertia；
- geom；
- collision；
- site；
- tendon；
- equality；
- timestep；
- 各种静态索引地址。

它在仿真过程中通常不会变化。

---

## 3.2 `MjData`

```python
data = mujoco.MjData(model)
```

可以理解为：

> 当前这一时刻仿真系统的动态状态。

包括：

```python
data.qpos
data.qvel
data.qacc
data.ctrl
data.xpos
data.xmat
data.xquat
data.contact
data.ncon
```

关系可以总结为：

```text
XML
 ↓
model
 ↓
data
 ↓
当前 q / qdot / contact / pose / control
```

---

# 4. `nq`、`nv`、`nu`、`njnt` 的区别

Franka Panda 初始模型中观测到：

```text
nq = 9
nv = 9
nu = 8
njnt = 9
```

原因是：

- 7 个机械臂 revolute joints；
- 2 个 finger slide joints；
- 共 9 个 1DOF joint；
- 7 个 arm actuators；
- 1 个 gripper actuator；
- 所以 `nu = 8`。

---

## 4.1 `nq`

Configuration position variable 的数量。

对于普通 revolute / prismatic joint：

\[
1\ joint \rightarrow 1\ qpos
\]

---

## 4.2 `nv`

Generalized velocity variable 的数量。

对于普通 1DOF joint：

\[
1\ joint \rightarrow 1\ qvel
\]

---

## 4.3 为什么加入 Cube 后 `nq != nv`

Cube 使用：

```xml
<freejoint name="cube_joint"/>
```

Free joint 的 configuration 用：

\[
[x,y,z,q_w,q_x,q_y,q_z]
\]

所以：

\[
7\ qpos
\]

但是速度是：

\[
[v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]
\]

只有：

\[
6\ qvel
\]

所以添加 Cube 后：

\[
nq = 9 + 7 = 16
\]

\[
nv = 9 + 6 = 15
\]

而 Cube 没有 actuator，所以：

\[
nu = 8
\]

这很好地说明了：

\[
\boxed{nq\neq nv\ 是完全正常的}
\]

---

# 5. Joint ID、qpos address、dof address

通过名字获得 Joint ID：

```python
joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "joint1"
)
```

但 Joint ID 不是直接等于 `data.qpos` 下标。

要进一步通过：

```python
model.jnt_qposadr[joint_id]
```

获得该 joint 在：

```python
data.qpos
```

里的起始位置。

速度使用：

```python
model.jnt_dofadr[joint_id]
```

对应：

```python
data.qvel
```

这两个地址必须区分，尤其存在 free joint 时更加重要。

---

# 6. `mj_forward()` 与 `mj_step()` 的区别

## `mj_forward`

```python
mujoco.mj_forward(model, data)
```

作用：

> 根据当前 q / qvel，重新计算 FK、body pose、site pose、动力学中间量等。

但：

\[
\boxed{\text{不推进仿真时间}}
\]

非常适合：

- 手动设置 q；
- 做 FK；
- Numerical IK；
- 查看某个 configuration 的 TCP pose。

---

## `mj_step`

```python
mujoco.mj_step(model, data)
```

表示真正推进一个仿真 timestep：

\[
t_{k+1}=t_k+\Delta t
\]

执行：

```text
control
→ dynamics
→ acceleration
→ velocity
→ position
→ contact
→ next time step
```

---

# 7. 为什么直接修改 `data.qpos` 不是“控制”

例如：

```python
data.qpos[joint1_adr] = np.deg2rad(30)
mujoco.mj_forward(model, data)
```

只是：

> 把模型状态直接放到某个 configuration。

相当于瞬间设置状态。

不是机器人真实运动。

真正控制需要：

```python
data.ctrl[...] = command
mujoco.mj_step(...)
```

形成动力学过程。

因此项目中始终要区分：

\[
\boxed{\text{State Assignment} \neq \text{Control}}
\]

---

# 8. Franka 初始 q 从哪里来

早期观察到模型加载后机械臂不是所有关节都为 0，例如：

```text
[0, 0, 0, -90°, 0, 90°, -44.994°]
```

其来源一般与模型 XML / keyframe / 初始化设置有关。

项目后期因为加入 Cube freejoint 后原 keyframe qpos 维数不再匹配，所以我们删除了复制版 Panda 中原有 keyframe，并在 Python 中明确指定：

```python
q_home = np.array([
    0.0,
    0.0,
    0.0,
    -np.pi / 2,
    0.0,
    np.pi / 2,
    -0.7853
])
```

这样初始化来源变得完全明确。

---

# 9. 为什么复制 Menagerie Panda，而不是直接修改官方模型

官方模型：

```text
mujoco_menagerie/franka_emika_panda/
```

保留原样。

复制为：

```text
assets/franka_pick_cube/
```

原因：

- 后续要修改 `scene.xml`；
- 添加 table；
- 添加 cube；
- 添加 TCP site；
- 可能修改 contact / geom；
- 保留官方原模型方便对照；
- 出问题时容易判断是官方模型问题还是自己的改动问题。

PowerShell 的：

```powershell
Copy-Item -Recurse ...
```

本质上和鼠标 `Ctrl+C / Ctrl+V` 一样。

复制出来的 `README.md` 本来就是官方模型目录中的说明文件，它不参与仿真，但建议保留用于：

- 模型来源说明；
- License；
- 第三方模型引用。

---

# 10. Body Pose：`xpos`、`xmat`、`xquat`

Body 世界坐标位置：

```python
data.xpos[body_id]
```

表示：

\[
{}^Wp_B
\]

旋转矩阵：

```python
data.xmat[body_id].reshape(3,3)
```

表示：

\[
{}^WR_B
\]

四元数：

```python
data.xquat[body_id]
```

MuJoCo 格式：

\[
[w,x,y,z]
\]

---

# 11. FK：从 Joint Configuration 到 EE Pose

机械臂 FK 的本质是：

\[
\boxed{
q
\rightarrow
T(q)
\rightarrow
(p,R)
}
\]

在项目中，最开始直接通过：

```python
data.qpos[...] = q
mujoco.mj_forward(...)
```

观察 `hand` pose。

后面使用 TCP site，则读取：

```python
data.site_xpos[tcp_site_id]
data.site_xmat[tcp_site_id]
```

---

# 12. 为什么 `hand` 不等于 TCP

`hand` 是一个 body，其 body origin 是模型设计中的结构参考点。

实际抓取发生在两根手指之间的抓取中心附近。

所以：

\[
\boxed{
p_{\text{hand origin}}
\neq
p_{\text{grasp center}}
}
\]

早期使用 `hand` 做 Reach 没问题，但真正抓 Cube 时必须定义更合理的控制点。

---

# 13. TCP 是什么

TCP：

\[
\boxed{Tool\ Center\ Point}
\]

通常表示：

> 工具或夹爪真正执行任务时最关心的参考点。

对于双指夹爪，可定义在：

- 两指之间；
- 抓取中心；
- 指尖接触区域中心。

项目中根据 Franka hand/finger 几何，在 `hand` body 下增加：

```xml
<site
    name="tcp"
    pos="0 0 0.103"
    size="0.008"
    rgba="0 1 0 1"
/>
```

于是：

```python
model.nsite
```

从：

```text
0
```

变成：

```text
1
```

可以通过：

```python
data.site_xpos[tcp_site_id]
data.site_xmat[tcp_site_id]
```

直接读取 TCP pose。

---

# 14. 为什么 TCP site 很有用

刚体变换关系：

\[
{}^Wp_{TCP}
=
{}^Wp_H
+
{}^WR_H {}^Hp_{TCP}
\]

MuJoCo 会自动计算。

不需要每次自己手动做：

```python
p_tcp = p_hand + R_hand @ offset
```

而且还可以直接计算 TCP Jacobian：

```python
mujoco.mj_jacSite(...)
```

---

# 15. Joint Position Control

Franka Menagerie 当前使用的 arm actuator 本质上是 position actuator。

所以：

```python
data.ctrl[arm_actuator_id] = q_desired
```

的物理意义：

> 给 position servo 设置目标关节角。

不是直接：

\[
\tau = ctrl
\]

因此：

\[
\boxed{
data.ctrl
\ 的物理意义必须由 actuator XML 决定
}
\]

不能看到 `ctrl` 就默认说它是 torque。

---

# 16. 关节误差与任务空间误差

Joint-space error：

\[
e_q=q_d-q
\]

用于关节控制。

Task-space position error：

\[
e_p=p_d-p
\]

用于判断末端是否到达目标。

Pose 控制时还需要 orientation error：

\[
e_R
\]

所以：

\[
\boxed{
控制误差与任务成功判据可以是不同空间中的量
}
\]

---

# 17. 3D Cartesian Reach

第一次从：

\[
q_d
\]

升级到：

\[
p_d=[x_d,y_d,z_d]^T
\]

控制链：

```text
p_target
 ↓
Numerical IK
 ↓
q_target
 ↓
Franka position actuator
 ↓
q
 ↓
TCP position
```

---

# 18. Jacobian 的意义

位置 Jacobian：

\[
\boxed{
\dot p
=
J_p(q)\dot q
}
\]

对于 Franka：

\[
J_p\in\mathbb R^{3\times7}
\]

完整几何 Jacobian：

\[
\boxed{
\begin{bmatrix}
v\\
\omega
\end{bmatrix}
=
J(q)\dot q
}
\]

其中：

\[
J=
\begin{bmatrix}
J_p\\
J_R
\end{bmatrix}
\in\mathbb R^{6\times7}
\]

---

# 19. 为什么使用 Pseudoinverse

Franka 是：

\[
7DOF
\]

而 Position task 只有：

\[
3D
\]

Pose task 是：

\[
6D
\]

Jacobians 都不是方阵。

所以不能使用普通：

\[
J^{-1}
\]

而使用 Moore-Penrose Pseudoinverse：

\[
\boxed{
\Delta q
=
J^+e
}
\]

例如 6D Pose：

\[
J\in\mathbb R^{6\times7}
\]

\[
J^+\in\mathbb R^{7\times6}
\]

于是：

\[
(7\times6)(6\times1)=7\times1
\]

得到 7 个 joint correction。

---

# 20. Numerical IK 的基本循环

核心：

\[
q_{k+1}
=
q_k
+
\alpha J^+(q_k)e_k
\]

具体过程：

```text
当前 q
 ↓
FK
 ↓
当前 TCP Pose
 ↓
计算 Pose Error
 ↓
计算 Jacobian
 ↓
Pseudoinverse
 ↓
Δq
 ↓
更新 q
 ↓
重复
```

直到：

\[
\|e_p\|<\epsilon_p
\]

且：

\[
\|e_R\|<\epsilon_R
\]

---

# 21. 为什么需要 `step_size`

如果直接：

\[
q_{k+1}=q_k+\Delta q
\]

可能每次更新太大。

所以：

\[
q_{k+1}
=
q_k
+
\alpha\Delta q
\]

例如：

\[
\alpha=0.2
\]

作用类似 Numerical IK 的“学习率”。

太大：

- 容易振荡；
- 容易跨过解；
- 接近奇异时更危险。

太小：

- 收敛慢。

---

# 22. Joint Limit

机器人 joint 并不是：

\[
(-\infty,\infty)
\]

每个关节都有：

\[
q_i^{min}
\le q_i
\le q_i^{max}
\]

项目中通过：

```python
np.clip(q[i], lower, upper)
```

确保 IK 不产生物理不可行 configuration。

---

# 23. 6D Pose Error

完整 Pose：

\[
(p,R)
\]

位置误差：

\[
\boxed{
e_p=p_d-p
}
\]

姿态误差通过：

\[
R_{err}=R_dR_c^T
\]

得到 axis-angle：

\[
R_{err}
\rightarrow
(\theta,u)
\]

最终定义 rotation vector：

\[
\boxed{
e_R=\theta u
}
\]

组合：

\[
\boxed{
e=
\begin{bmatrix}
e_p\\
e_R
\end{bmatrix}
\in\mathbb R^6
}
\]

---

# 24. `mj_jacBody` 与 `mj_jacSite`

最开始末端参考点是 hand body：

```python
mujoco.mj_jacBody(...)
```

后来真正控制 TCP：

```python
mujoco.mj_jacSite(...)
```

为什么必须换？

若 TCP 与 hand origin 有偏移：

\[
p_{TCP}
=
p_H+R_Hr
\]

速度：

\[
v_{TCP}
=
v_H+
\omega_H\times(R_Hr)
\]

所以：

\[
\boxed{
J_{TCP}\neq J_H
}
\]

特别是位置 Jacobian 会因为 offset 不同而不同。

---

# 25. Franka 冗余性

完整 pose task：

\[
6D
\]

Franka：

\[
7DOF
\]

所以：

\[
7-6=1
\]

意味着：

> 同一个 TCP pose 可以对应多组 joint configuration。

所以：

\[
Pose\rightarrow q
\]

不是唯一映射。

这就是冗余机械臂。

---

# 26. Gripper：Command 不等于 State

Franka 有：

```text
finger_joint1
finger_joint2
```

但只有一个：

```text
actuator8
```

控制整个 gripper。

项目中：

```python
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0
```

但必须区分：

```python
data.ctrl[gripper_actuator_id]
```

与：

```python
finger_q
```

前者表示：

> 命令。

后者表示：

> 两根手指真实位置。

所以：

\[
\boxed{
Command\neq State
}
\]

例如命令 Close：

\[
ctrl=0
\]

但 Cube 挡在中间，finger 并不会真正闭到 0。

---

# 27. Gripper Width

如果：

\[
q_{f1},q_{f2}
\]

是两根 slide finger 的 displacement，则：

\[
\boxed{
w=q_{f1}+q_{f2}
}
\]

完全打开约：

\[
0.04+0.04
=
0.08m
\]

即约 8 cm。

---

# 28. Contact 不能只看 `data.ncon`

`data.ncon > 0` 只能说明：

> 场景中某处存在 contact。

但场景本来就可能存在：

\[
Cube \leftrightarrow Table
\]

所以不能用：

```python
data.ncon > 0
```

判断抓取成功。

真正需要判断：

\[
\boxed{
left\ finger \leftrightarrow cube
}
\]

并且：

\[
\boxed{
right\ finger \leftrightarrow cube
}
\]

还要持续一段时间。

---

# 29. 为什么通过 `geom_bodyid` 判断 finger contact

Franka 每根手指上可能有多个 collision geom，而且并非所有 geom 都有方便的名字。

因此更可靠的方法是：

```python
model.geom_bodyid[geom_id]
```

判断这个碰撞 geom 属于：

```text
left_finger
```

还是：

```text
right_finger
```

---

# 30. Scene：Table + Cube

Table 没有 joint：

```xml
<body name="table">
    <geom .../>
</body>
```

因此固定在世界中。

Cube 有：

```xml
<freejoint name="cube_joint"/>
```

因此是真正 dynamic rigid body。

它会：

- 受 gravity；
- 掉落；
- 碰桌面；
- 被手指夹住；
- 被抬起；
- 被释放。

---

# 31. 为什么先让 Cube settle

XML 中定义：

\[
z=0.24m
\]

但桌面和 Cube 几何对应稳定中心高度约：

\[
0.235m
\]

所以程序先：

```python
while data.time < settle_time:
    mujoco.mj_step(...)
```

让 Cube 真正落到桌面。

然后读取：

```python
cube_position = data.xpos[cube_body_id]
```

控制器应该相信：

\[
\boxed{\text{物体实际状态}}
\]

而不是 XML 中曾经写过的初值。

---

# 32. Pregrasp 的意义

不直接让 gripper 从远处冲向 Cube。

先到：

\[
p_{pre}
=
p_{cube}
+
\begin{bmatrix}
0\\0\\h
\end{bmatrix}
\]

例如：

\[
h=0.12m
\]

得到 Cube 上方安全位置。

标准抓取逻辑：

```text
HOME
 ↓
PREGRASP
 ↓
APPROACH
 ↓
CLOSE
 ↓
LIFT
```

---

# 33. Top-down Grasp

项目采用 top-down grasp。

保持 TCP 朝下的 orientation：

\[
R_{target}=R_{home}
\]

先不研究任意 grasp orientation，以降低变量数量。

---

# 34. Pick 状态机

经过逐步扩展，最终形成：

```text
PREGRASP
 ↓
APPROACH
 ↓
CLOSE
 ↓
LIFT
 ↓
MOVE
 ↓
LOWER
 ↓
SETTLE
 ↓
RELEASE
 ↓
RETREAT
 ↓
DONE
```

这是一个典型的 Finite State Machine（FSM）。

---

# 35. 为什么单次到达阈值不代表成功

如果只判断：

\[
e_p<0.01
\]

一次，就可能因为震荡瞬间穿过阈值而误判。

所以项目中使用：

```python
stable_counter
```

要求连续多个 timestep 满足：

\[
e_p<\epsilon_p
\]

且：

\[
e_R<\epsilon_R
\]

才切换状态。

这就是：

\[
\boxed{\text{稳定到达}}
\]

---

# 36. 为什么双指接触也需要持续计数

单个 timestep 接触可能只是：

- 刮过 Cube；
- 碰一下；
- 瞬时 collision。

所以需要：

```python
contact_counter
```

连续若干 timestep：

\[
left=True
\]

和：

\[
right=True
\]

才认定形成稳定 bilateral contact。

---

# 37. Contact 不等于 Grasp Success

即使：

```text
Left contact = True
Right contact = True
```

仍然可能：

> 一抬就掉。

所以真正的 grasp validation 是：

```text
Bilateral Contact
 ↓
LIFT
 ↓
Cube leaves table
 ↓
Cube follows TCP
```

即：

\[
\boxed{
Contact\neq Grasp
}
\]

真正物理抓取必须通过 Lift 验证。

---

# 38. Lift 成功判据

项目中组合使用：

1. Cube 相对初始高度显著增加：

\[
z_{cube}-z_0>0.08
\]

2. Cube 离开桌面：

\[
CubeTableContact=False
\]

3. Cube 仍靠近 TCP：

\[
\|p_{cube}-p_{tcp}\|<threshold
\]

4. TCP 本身已接近 lift target。

这样的判据比只看：

\[
z_{cube}>某值
\]

更可靠。

---

# 39. 为什么 `13_pick_and_place.py` 会失败

13 已经完成：

- 抓取；
- Lift；
- Move target；
- Lower；
- Release。

但实际出现：

```text
>>> LIFT reached
Start MOVE...

>>> FAILED: Cube dropped during MOVE
```

根因并不是 IK 求不出来，而是：

\[
\boxed{
Waypoint\ 能到达
\neq
Waypoint\ 之间运动合理
}
\]

13 中：

```python
q_target = q_lift
```

下一状态直接：

```python
q_target = q_place_above
```

相当于：

\[
q_d
\]

发生阶跃。

position actuator 会强烈追赶新的目标，造成较大的瞬时运动。

夹住的 Cube 因此：

- 滑动；
- 转动；
- 失去双指约束；
- 掉落。

---

# 40. Path 与 Trajectory 的区别

Path：

> 只描述从哪里到哪里。

例如：

\[
q_A\rightarrow q_B
\]

Trajectory：

> 描述每个时刻应该在哪里。

\[
\boxed{
q(t)
}
\]

同时可以定义：

\[
\dot q(t)
\]

\[
\ddot q(t)
\]

所以：

\[
\boxed{
Final\ Pose\ Correct
\neq
Motion\ Process\ Correct
}
\]

这是整个项目非常重要的学习结论。

---

# 41. 五次时间缩放

14 中使用：

\[
\boxed{
h(s)=10s^3-15s^4+6s^5
}
\]

其中：

\[
s=\frac{t-t_0}{T}
\in[0,1]
\]

Joint trajectory：

\[
\boxed{
q(t)
=
q_0
+
h(s)(q_f-q_0)
}
\]

满足：

\[
h(0)=0,\quad h(1)=1
\]

\[
h'(0)=h'(1)=0
\]

\[
h''(0)=h''(1)=0
\]

所以轨迹段起点和终点：

\[
\boxed{\dot q=0}
\]

以及：

\[
\boxed{\ddot q=0}
\]

比 step command 平滑得多。

---

# 42. 为什么 MOVE 和 LOWER 要更慢

搬运阶段：

\[
Cube\ 已经在 gripper 中
\]

因此任何高加速度都可能导致滑移。

项目中故意设置：

```text
MOVE duration 更长
LOWER duration 更长
```

尤其：

\[
\boxed{LOWER\ 要慢}
\]

减少物体接触桌面的冲击。

---

# 43. 真实 TCP-Cube 相对变换

13 中曾经假设：

\[
p_{TCP}-p_{cube}
=
[0,0,0.005]
\]

这只是理想抓取关系。

实际抓住以后 Cube 可能：

- 偏一点；
- 转一点；
- 在夹爪中不完全居中。

14 中抓住后测量：

\[
{}^{TCP}T_{Cube}
=
({}^{W}T_{TCP})^{-1}
{}^{W}T_{Cube}
\]

这样获得真实 grasp transform。

---

# 44. 根据 Cube 目标反求 TCP 目标

目标是：

\[
{}^WT_{Cube,d}
\]

而实际抓取关系：

\[
{}^{TCP}T_C
\]

满足：

\[
{}^WT_C
=
{}^WT_{TCP}
{}^{TCP}T_C
\]

所以：

\[
\boxed{
{}^WT_{TCP,d}
=
{}^WT_{Cube,d}
({}^{TCP}T_C)^{-1}
}
\]

这一步非常重要。

它把逻辑从：

> “我猜机械臂应该去哪里”

升级为：

> “我知道物体最终应该在哪里，根据实际抓取关系反算机械臂应该在哪里”。

这是更标准的 manipulation 思维。

---

# 45. 为什么 IK 要使用独立 `ik_data`

14 以后 placement-side IK 是在“真实仿真已经抓住 Cube”之后才计算的。

如果 Numerical IK 仍修改真实：

```python
data.qpos
```

会导致：

> 为了算数学解，直接把真实机器人瞬移。

所以创建：

```python
ik_data = mujoco.MjData(model)
```

Numerical IK 用：

```text
ik_data
```

真实物理仿真用：

```text
data
```

这样数学求解与真实仿真完全分离。

这是很重要的工程改进。

---

# 46. LOWER 为什么不是强行到最终 q

理论 place pose 可能有轻微误差。

如果机械臂一定要追到：

\[
q_{lower}
\]

Cube 已经碰桌面后还继续下降，会：

- 压 Cube；
- 推 Cube；
- 让 Cube 倾倒。

所以 14 中 LOWER 是：

```text
Slow downward search
 ↓
Cube contacts table
 ↓
stop trajectory immediately
```

也就是：

\[
\boxed{
Contact-based Stop
}
\]

---

# 47. SETTLE 为什么必要

Cube 与桌面第一次 contact：

\[
table\_contact=True
\]

并不等于：

> Cube 已经稳定平放。

可能只是一个角碰到桌面。

所以增加：

```text
LOWER
 ↓
TABLE CONTACT
 ↓
SETTLE
 ↓
RELEASE
```

要求 Cube：

\[
\|v\|<0.01m/s
\]

以及：

\[
\|\omega\|<0.1rad/s
\]

连续若干 timestep。

只有物体真正基本静止才 RELEASE。

---

# 48. 14 最终为什么成功

14 同时修复了 13 的两个核心问题：

## 问题 1：Waypoint Step

从：

\[
q_A\rightarrow q_B
\]

改成：

\[
q(t)
\]

降低了 MOVE / LOWER 时的冲击。

---

## 问题 2：错误假定 TCP-Cube 关系

从固定 offset：

\[
[0,0,0.005]
\]

升级为实际测量：

\[
{}^{TCP}T_C
\]

最终获得稳定 Pick-and-Place：

- XY error ≈ 2.23 mm；
- orientation error ≈ 0.596°；
- Cube on table = True；
- FSM = DONE。

---

# 49. Jacobian 奇异性

对于：

\[
J\in\mathbb R^{6\times7}
\]

做 SVD：

\[
\boxed{
J=U\Sigma V^T
}
\]

奇异值：

\[
\sigma_1\ge...\ge\sigma_6
\]

最关键：

\[
\boxed{
\sigma_{\min}=\sigma_6
}
\]

如果：

\[
\sigma_{\min}\rightarrow0
\]

说明：

> 机械臂在某个 task-space 方向的运动能力趋近退化。

---

# 50. Condition Number

\[
\boxed{
\kappa(J)
=
\frac{\sigma_{\max}}{\sigma_{\min}}
}
\]

当：

\[
\sigma_{\min}\rightarrow0
\]

则：

\[
\kappa\rightarrow\infty
\]

因此：

- \(\sigma_{\min}\) 越小越危险；
- condition number 越大越危险。

---

# 51. Manipulability

项目使用：

\[
\boxed{
w=\prod_i\sigma_i
}
\]

接近奇异点时：

\[
w\rightarrow0
\]

可以理解为：

> 末端综合运动能力下降。

---

# 52. 为什么 Pseudoinverse 在奇异点附近会爆炸

Pseudoinverse：

\[
J^+
=
V\Sigma^+U^T
\]

其中：

\[
\Sigma^+
=
diag
\left(
\frac1{\sigma_1},
...,
\frac1{\sigma_6}
\right)
\]

如果：

\[
\sigma_{\min}=0.001
\]

那么：

\[
\frac1{\sigma_{\min}}=1000
\]

所以：

\[
\boxed{
小的 task error
\rightarrow
巨大的 joint correction
}
\]

项目实验中 Pseudoinverse 的 raw：

\[
\|\Delta q\|
\]

曾达到非常大的量级，并且出现持续振荡。

---

# 53. 为什么会出现高频振荡

Near singularity：

```text
J^+ 给出巨大 Δq
 ↓
clip 限制
 ↓
跨到奇异区域另一侧
 ↓
下一步 Jacobian 变化
 ↓
Δq 方向可能反转
 ↓
再次跨回
```

最终形成：

\[
\boxed{\text{高频来回振荡}}
\]

因此不是 plotting 错误，而是算法确实在病态区域振荡。

---

# 54. Damped Least Squares（DLS）

DLS：

\[
\boxed{
\Delta q
=
J^T
(JJ^T+\lambda^2I)^{-1}e
}
\]

从 SVD 角度，奇异方向的增益从：

\[
\frac1{\sigma_i}
\]

变成：

\[
\boxed{
\frac{\sigma_i}{\sigma_i^2+\lambda^2}
}
\]

如果：

\[
\sigma_i\rightarrow0
\]

则 DLS 增益：

\[
\rightarrow0
\]

不会发散。

---

# 55. DLS 的本质

DLS 不是：

> 让 IK 一定更快。

而是：

\[
\boxed{
\text{用一定精度换数值稳定性}
}
\]

特点：

- 远离奇异点：和普通 inverse 类似；
- 接近奇异点：抑制大 joint correction；
- 可能收敛更慢；
- 可能存在小的 steady-state error；
- 但更稳定。

---

# 56. Fixed DLS vs Adaptive DLS

Fixed DLS：

\[
\lambda=常数
\]

例如：

\[
\lambda=0.05
\]

问题：

> 不管危险不危险都使用相同阻尼。

---

Adaptive DLS：

\[
\boxed{
\lambda=\lambda(\sigma_{\min})
}
\]

逻辑：

\[
\sigma_{\min}\ 大
\Rightarrow
\lambda\ 小
\]

\[
\sigma_{\min}\ 小
\Rightarrow
\lambda\ 大
\]

可以理解为：

```text
正常区域
→ 少踩刹车

靠近奇异
→ 重踩刹车
```

---

# 57. Adaptive Damping Law

项目中使用：

\[
r
=
\frac{
\sigma_{threshold}-\sigma_{\min}
}{
\sigma_{threshold}
}
\]

并 clip：

\[
r\in[0,1]
\]

然后：

\[
\boxed{
\lambda
=
\lambda_{min}
+
(\lambda_{max}-\lambda_{min})r^2
}
\]

选择 \(r^2\) 的原因：

- 刚进入危险区域时变化平缓；
- 越接近奇异点，阻尼增长越明显。

---

# 58. Adaptive DLS 仍然不会避开奇异点

这是项目中一个非常关键的认识。

Adaptive DLS 只是：

> 到奇异点附近时不爆炸。

它不会主动说：

> 改变 elbow 姿态，离开奇异点。

因此可能看到：

\[
\sigma_{\min}\rightarrow0
\]

但算法仍然稳定，只是误差不再继续下降。

所以：

\[
\boxed{
Adaptive\ DLS
\neq
Singularity\ Avoidance
}
\]

---

# 59. Null Space

Franka：

\[
q\in\mathbb R^7
\]

Pose task：

\[
x\in\mathbb R^6
\]

若：

\[
rank(J)=6
\]

则：

\[
\dim Null(J)
=
7-6
=
1
\]

存在一个 joint motion direction：

\[
z
\]

使：

\[
\boxed{
Jz=0
}
\]

也就是说：

> Joint configuration 可以变化，但 TCP 一阶近似下不变化。

---

# 60. Null-space Control 的经典公式

\[
\boxed{
\dot q
=
J^+\dot x
+
(I-J^+J)\dot q_0
}
\]

第一项：

\[
J^+\dot x
\]

完成主任务。

第二项：

\[
(I-J^+J)\dot q_0
\]

完成 secondary task。

因为：

\[
\boxed{
J(I-J^+J)\approx0
}
\]

所以第二项理论上一阶不影响 TCP 主任务。

---

# 61. 项目中如何找到 1D Null-space Direction

完整 SVD：

\[
J=U\Sigma V^T
\]

对于：

\[
J\in\mathbb R^{6\times7}
\]

\(V^T\) 是：

\[
7\times7
\]

最后一个 right singular vector：

\[
v_7
\]

满足：

\[
\boxed{
Jv_7\approx0
}
\]

所以代码：

```python
null_direction = Vt[-1, :]
```

就是冗余方向。

---

# 62. Null Space 往正方向还是负方向

1D null space 有：

\[
+n
\]

和：

\[
-n
\]

两个方向。

项目中分别试探：

\[
q_+=q+\epsilon n
\]

\[
q_-=q-\epsilon n
\]

分别计算：

\[
\sigma_{\min}(q_+)
\]

和：

\[
\sigma_{\min}(q_-)
\]

选择能让：

\[
\boxed{
\sigma_{\min}\uparrow
}
\]

的方向。

所以这个 secondary task 是：

\[
\boxed{\text{Singularity Avoidance}}
\]

---

# 63. Null-space 实验的主任务

为了最直观看出效果，实验不是让 TCP Reach 新目标，而是：

\[
\boxed{
p_{TCP,d}=p_{TCP,0}
}
\]

\[
\boxed{
R_{TCP,d}=R_{TCP,0}
}
\]

也就是说：

> TCP 原地保持。

然后：

- Baseline：不加 null-space motion；
- Null-space：调整 elbow / joints 提升 \(\sigma_{\min}\)。

---

# 64. Null-space 实验验证了什么

观测结果：

Without Null Space：

- \(\sigma_{\min}\) 基本保持低值；
- condition number 较大；
- robot posture 基本不变。

With Null Space：

- \(\sigma_{\min}\) 明显增大；
- condition number 明显下降；
- joint posture 明显变化；
- TCP position error 仍在极小量级；
- TCP orientation error 也极小。

同时：

\[
\boxed{
\|J\Delta q_{null}\|
\approx0
}
\]

验证了 null-space motion 对主任务的一阶 task-space leakage 极小。

---

# 65. Null Space 的常见用途

项目已经实践：

\[
\boxed{\text{Singularity Avoidance}}
\]

还可以用于：

### Joint Limit Avoidance

让 joint 自动远离极限。

### Obstacle Avoidance

让 elbow / links 避开障碍。

### Preferred Posture

让 robot 保持“舒服”的姿态。

### Energy / Torque Optimization

在满足 TCP task 的多组 configuration 中优化 secondary cost。

---

# 66. 一个必须牢记的区别

## Pseudoinverse

解决：

\[
\boxed{\text{如何从 task error 得到 joint correction}}
\]

---

## DLS / Adaptive DLS

解决：

\[
\boxed{\text{接近奇异时如何避免逆解爆炸}}
\]

---

## Null-space Control

解决：

\[
\boxed{\text{如何利用冗余自由度完成 secondary task}}
\]

---

## Null-space Singularity Avoidance

进一步解决：

\[
\boxed{\text{如何主动远离奇异 configuration}}
\]

这几个概念不能混在一起。

---

# 67. 本项目中最重要的调试经验

## 经验 1：先分层定位问题

抓取失败不能一上来就调所有参数。

应该分：

```text
IK 问题？
 ↓
Trajectory 问题？
 ↓
Collision 问题？
 ↓
Contact 问题？
 ↓
Gripper friction 问题？
 ↓
Task logic 问题？
```

---

## 经验 2：Command 与 State 不同

```python
data.ctrl
```

只是命令。

真正判断系统状态，要读：

```python
data.qpos
data.qvel
data.site_xpos
data.contact
```

---

## 经验 3：Final State 对不代表过程对

13 的 IK waypoint 都可达。

但 MOVE 过程中 Cube 掉了。

所以：

\[
\boxed{
Final\ Pose\ Correct
\neq
Trajectory\ Correct
}
\]

---

## 经验 4：Contact 不等于 Stable Contact

单次 collision 不能直接切状态。

使用稳定计数器是合理做法。

---

## 经验 5：Contact 不等于 Grasp

真正抓取必须通过 Lift 验证。

---

## 经验 6：Table Contact 不等于 Settled

Cube 一个角碰桌面：

```python
table_contact=True
```

但可能仍然在转。

所以要进一步看：

\[
v
\]

和：

\[
\omega
\]

---

## 经验 7：数学参考点必须和真实任务点一致

`hand` body origin 能用于早期 Reach。

真实 grasp 要使用 TCP。

否则：

\[
\boxed{
数学位置对
\neq
实际夹爪抓取中心对
}
\]

---

## 经验 8：Numerical IK 与真实仿真状态分开

后期创建：

```python
ik_data
```

与：

```python
data
```

分离。

这是很有价值的工程实践。

---

# 68. 项目中的关键公式速查

## FK

\[
T=T(q)
\]

---

## Jacobian

\[
\dot x=J(q)\dot q
\]

---

## Position IK

\[
\Delta q=J_p^+e_p
\]

---

## 6D Pose IK

\[
\Delta q=J^+e
\]

其中：

\[
e=
\begin{bmatrix}
e_p\\
e_R
\end{bmatrix}
\]

---

## Rotation Error

\[
R_e=R_dR_c^T
\]

\[
e_R=\theta u
\]

---

## Quintic Time Scaling

\[
h(s)=10s^3-15s^4+6s^5
\]

\[
q(t)=q_0+h(s)(q_f-q_0)
\]

---

## Grasp Transform

\[
{}^{TCP}T_C
=
({}^WT_{TCP})^{-1}
{}^WT_C
\]

---

## Place TCP Transform

\[
{}^WT_{TCP,d}
=
{}^WT_{C,d}
({}^{TCP}T_C)^{-1}
\]

---

## Pseudoinverse IK

\[
\Delta q=J^+e
\]

---

## DLS

\[
\Delta q
=
J^T(JJ^T+\lambda^2I)^{-1}e
\]

---

## Condition Number

\[
\kappa
=
\frac{\sigma_{\max}}{\sigma_{\min}}
\]

---

## Null-space Projector

\[
N=I-J^+J
\]

---

## Null-space Control

\[
\dot q
=
J^+\dot x
+
N\dot q_0
\]

---

# 69. 项目复习时建议重点回答的 20 个问题

## 1. `model` 和 `data` 有什么区别？

`model` 保存固定结构和参数；`data` 保存当前动态状态。

---

## 2. 为什么 `nq` 和 `nv` 可以不同？

freejoint 使用 7 个 configuration variables 表示位姿，但只有 6 个 generalized velocities。

---

## 3. 为什么 joint ID 不能直接拿去索引 qpos？

因为不同 joint 类型在 qpos / qvel 中占用维度不同，需要 `jnt_qposadr` / `jnt_dofadr` 映射。

---

## 4. `mj_forward` 与 `mj_step` 的区别？

前者重新计算当前 configuration 的派生量，不推进时间；后者执行动力学并推进仿真。

---

## 5. 为什么直接修改 `data.qpos` 不是控制？

它是状态赋值 / 瞬移，不是通过 actuator 与 dynamics 形成真实运动。

---

## 6. 为什么 `data.ctrl` 不能默认理解为 torque？

其意义取决于 actuator XML 类型；本项目 Franka arm 使用 position actuator。

---

## 7. `hand` 与 TCP 的区别？

hand 是 body reference origin；TCP 是人为定义的任务中心 / 抓取中心。

---

## 8. 为什么用 `mj_jacSite()`？

因为真正控制的是 TCP site，而不是 hand body origin。

---

## 9. 为什么 Franka IK 使用 pseudoinverse？

因为 Franka 7DOF，而 task 是 3D/6D，Jacobian 不是方阵。

---

## 10. 为什么 Franka 有冗余？

7 joint DOF > 6D pose constraints，所以同一 TCP pose 对应多组 q。

---

## 11. 为什么 command Close 不代表已经抓住？

命令只表示 actuator 目标；真实 finger position / contact / lift 才能反映物理状态。

---

## 12. 为什么双指 contact 仍不能宣布 grasp success？

物体可能一 Lift 就滑落。

---

## 13. 为什么 13 的 Pick-and-Place 会在 MOVE 掉 Cube？

因为不同 waypoint 之间 q_target 阶跃变化，引发较大瞬态运动。

---

## 14. 五次时间缩放为什么适合机器人轨迹？

因为起点终点速度、加速度均为 0，状态切换更加平滑。

---

## 15. 为什么最终放置要记录真实 TCP-Cube transform？

实际 grasp 不可能每次都严格符合理想 offset。

---

## 16. 为什么 Cube 碰桌面不能马上 Release？

第一次 contact 可能只是一角触地，仍存在较大的线速度或角速度。

---

## 17. 为什么 pseudoinverse 接近奇异点会不稳定？

因为 \(1/\sigma_{\min}\) 会非常大，使小 task error 被放大成巨大 joint correction。

---

## 18. DLS 做了什么？

将奇异方向增益从 \(1/\sigma\) 改成 \(\sigma/(\sigma^2+\lambda^2)\)，抑制奇异放大。

---

## 19. Adaptive DLS 与 Fixed DLS 区别？

Adaptive DLS 根据 \(\sigma_{\min}\) 自动调整 \(\lambda\)。

---

## 20. Null-space Control 的核心意义？

在保持主 TCP task 的同时，利用冗余 DOF 优化 secondary objective，例如远离奇异点。

---

# 70. 目前没有完成、不要在简历中写成“已完成”的内容

以下内容目前属于后续方向，不应描述成已经实现：

- Operational Space Control；
- Torque-level impedance control；
- full dynamics compensation；
- RRT / RRT*；
- collision-free motion planning；
- camera perception；
- pose estimation；
- PPO/SAC manipulation；
- imitation learning；
- ACT；
- Diffusion Policy；
- VLA。

简历中一定要严格写实际完成的内容。

---

# 71. 项目如何重构成正式工程结构

当前 01~17 是学习型脚本。

作为 GitHub 项目，建议后续重构：

```text
franka_mujoco_manipulation/
│
├── assets/
│   └── franka_pick_cube/
│
├── controllers/
│   ├── numerical_ik.py
│   ├── dls_ik.py
│   ├── adaptive_dls.py
│   └── nullspace_controller.py
│
├── planners/
│   ├── grasp_pose.py
│   └── joint_trajectory.py
│
├── tasks/
│   ├── reach.py
│   ├── pick.py
│   └── pick_and_place.py
│
├── utils/
│   ├── mujoco_utils.py
│   ├── transforms.py
│   ├── contacts.py
│   └── metrics.py
│
├── experiments/
│   ├── singularity_compare.py
│   ├── adaptive_dls_compare.py
│   └── nullspace_compare.py
│
├── results/
│   ├── pick_place/
│   ├── dls/
│   └── nullspace/
│
├── README.md
└── requirements.txt
```

这样更适合 GitHub 和简历展示。

---

# 72. README 建议结构

```text
1. Project Overview
2. Motivation
3. System Architecture
4. Environment
5. Robot Model
6. Kinematics
7. Numerical IK
8. Gripper and Contact
9. Pick-and-Place FSM
10. Smooth Trajectory
11. Singularity Analysis
12. DLS and Adaptive DLS
13. Null-space Control
14. Results
15. Failure Analysis
16. Project Structure
17. How to Run
18. Future Work
```

尤其建议把：

```text
13 baseline failure
        ↓
14 trajectory improvement
```

写进 README。

这是项目里非常有价值的一段工程故事：

> “初版系统能够完成 Reach、Grasp 和 Lift，但在 waypoint 直接切换的 MOVE 阶段出现 Cube 滑落；通过五次时间缩放、真实 TCP-Cube grasp transform、contact-based lowering 与 settle logic 后，实现稳定 Pick-and-Place。”

相比只写“实现了 Pick-and-Place”，这段更能体现问题定位能力。

---

# 73. 简历项目名称怎么写

推荐：

## 中文

**Franka Panda 机械臂 MuJoCo 抓取与运动控制**

或：

**基于 MuJoCo 的 Franka Panda 7DOF 机械臂 Manipulation 与冗余控制**

第二个更适合偏机器人算法 / 控制岗位。

---

## 英文

**Franka Panda Manipulation and Redundancy Control in MuJoCo**

---

# 74. 简历项目描述——推荐中文版本

可以写成：

### 基于 MuJoCo 的 Franka Panda 7DOF 机械臂 Manipulation 与冗余控制

**技术栈：Python / MuJoCo / NumPy / Matplotlib / Robot Kinematics / Numerical IK**

- 基于 MuJoCo Menagerie 搭建 Franka Panda 7DOF 机械臂与双指夹爪仿真环境，完成机器人状态读取、TCP 建模、6D 末端位姿控制与物体接触检测。
- 基于 geometric Jacobian 与 Moore-Penrose pseudoinverse 实现 6D Numerical IK，并结合关节限位与稳定到达判据完成 Pregrasp、Approach、Grasp、Lift 等 Manipulation 基础动作。
- 设计 `PREGRASP → APPROACH → CLOSE → LIFT → MOVE → LOWER → SETTLE → RELEASE → RETREAT` 有限状态机，实现完整 Pick-and-Place；针对 waypoint 阶跃导致的物体滑落问题，引入五次时间缩放平滑关节轨迹，并基于实际 TCP-Cube 相对变换反算放置 TCP 目标。
- 通过接触反馈、Cube 线/角速度与稳定计数实现 contact-based lowering 与 settle/release 逻辑，最终实现约 **2.23 mm XY 放置误差**、约 **0.596° 物体姿态误差** 的稳定放置。
- 针对 7DOF 冗余机械臂奇异性问题，对比 Pseudoinverse、DLS 与 Adaptive DLS，利用最小奇异值 \(\sigma_{\min}(J)\) 与 condition number 分析 Jacobian 数值状态；进一步基于 SVD Null Space 实现 singularity avoidance，在保持 TCP 位姿基本不变的同时改善机械臂内部姿态与 Jacobian conditioning。

---

# 75. 如果简历空间很少，压缩成 3 条

### 基于 MuJoCo 的 Franka Panda 7DOF 机械臂 Manipulation 与冗余控制

- 基于 MuJoCo 搭建 Franka Panda + Gripper + Cube 仿真环境，完成 TCP 建模、geometric Jacobian、6D Numerical IK、contact detection 与完整 Pick-and-Place FSM。
- 针对 waypoint 直接切换导致 Cube 搬运滑落问题，实现五次时间缩放平滑关节轨迹、真实 TCP-Cube grasp transform 与 contact-based settle/release，最终实现约 **2.23 mm** XY 放置误差。
- 对比 Pseudoinverse、DLS、Adaptive DLS 的奇异点性能，并基于 SVD Null Space 实现 7DOF redundancy singularity avoidance，在保持 TCP 主任务的同时改善 Jacobian conditioning。

---

# 76. 简历英文版本

### Franka Panda Manipulation and Redundancy Control in MuJoCo

- Built a MuJoCo simulation pipeline for a Franka Panda 7-DoF arm with a parallel gripper and dynamic cube, including TCP modeling, geometric Jacobian computation, 6D numerical IK, contact detection, and manipulation state estimation.
- Designed a complete `PREGRASP → APPROACH → CLOSE → LIFT → MOVE → LOWER → SETTLE → RELEASE → RETREAT` pick-and-place FSM; resolved object slippage caused by waypoint step commands using quintic joint trajectory generation and measured TCP-to-object grasp transforms, achieving approximately **2.23 mm XY placement error**.
- Compared pseudoinverse IK, Damped Least Squares, and adaptive DLS near kinematic singularities, and implemented SVD-based null-space singularity avoidance to improve Jacobian conditioning while preserving the primary TCP pose task.

---

# 77. 面试中如何讲这个项目

建议按“问题—方法—结果”讲，而不是从文件 01 开始背。

## 第一层：项目要解决什么

> 我想从零搭建一个经典 7DOF 机械臂 Manipulation pipeline，不依赖现成 RL policy，先把机器人运动学、IK、轨迹生成和接触逻辑完整走通。

---

## 第二层：核心技术

> 用 MuJoCo Menagerie 的 Franka Panda 模型，通过 TCP site 建模末端抓取中心，基于 6x7 geometric Jacobian 实现 Numerical IK，并通过 FSM 组织 Pregrasp、Approach、Grasp、Lift、Move 和 Place。

---

## 第三层：遇到什么问题

> 第一版 Pick-and-Place 能抓住并抬起 Cube，但进入 MOVE 后经常滑落。分析发现 IK waypoint 本身可达，但 q_target 在状态切换时直接阶跃，导致 position actuator 快速追赶目标、引发较大瞬态运动。

---

## 第四层：如何解决

> 后来引入五次时间缩放生成平滑 q(t)，同时抓住 Cube 后实际测量 TCP-Cube 相对齐次变换，根据物体目标 Pose 反算 TCP 放置 Pose，并在下降阶段使用 contact-based stop + settle logic。

---

## 第五层：结果

> 最终完整状态机稳定完成，XY 放置误差约 2.23 mm，Cube 姿态误差约 0.596°。

---

## 第六层：进一步深入

> 然后研究了 Franka 7DOF 冗余带来的 Jacobian 奇异性，对比 pseudoinverse、DLS、Adaptive DLS，并用 Null-space secondary motion 主动提高最小奇异值，在 TCP 基本不动的情况下调整内部姿态。

这一套讲法非常适合机器人算法 / 控制 / 具身智能实习面试。

---

# 78. 简历中推荐的关键词

如果 JD 中有这些关键词，可以自然匹配：

```text
MuJoCo
Franka Panda
Robot Manipulation
Forward Kinematics
Inverse Kinematics
Geometric Jacobian
Numerical IK
Damped Least Squares
Adaptive DLS
Singularity
Redundancy
Null Space
Trajectory Generation
Finite State Machine
Contact Detection
Pick and Place
SE(3)
TCP
Python
NumPy
```

---

# 79. 后续回到原学习路线

这个项目不建议继续无限增加 `18.py`、`19.py`、`20.py`。

当前最合理的收尾：

```text
Franka Project
 ↓
整理 README
 ↓
整理代码结构
 ↓
保存结果图
 ↓
GitHub
 ↓
简历
```

然后返回原路线。

建议下一阶段：

\[
\boxed{
Path\ / Motion\ Planning
}
\]

具体：

```text
A* / Dijkstra
 ↓
RRT
 ↓
RRT*
 ↓
Collision Checking
 ↓
Path
 ↓
Trajectory Generation
 ↓
Tracking
```

你现在已经学过：

\[
\boxed{
Trajectory\ Generation
}
\]

所以再学 Planning 时可以非常自然地理解：

```text
Planner:
生成 path

Trajectory generator:
给 path 加时间

Controller:
跟踪 trajectory
```

之后再进入 Robot Learning：

```text
PPO
 ↓
SAC
 ↓
ManiSkill / MuJoCo Manipulation
 ↓
Imitation Learning
 ↓
ACT
 ↓
Diffusion Policy
```

这样会比现在直接跳 VLA 更扎实。

---

# 80. 这个项目最终应该带走的核心认知

如果只记 10 句话，建议记下面这些：

1. `model` 是结构，`data` 是当前状态。
2. `qpos`、`qvel` 不是永远同维度，freejoint 是最典型例子。
3. FK 是 \(q\to x\)，IK 是 \(x_d\to q_d\)。
4. Jacobian 描述 joint velocity 与 task velocity 的局部映射。
5. 7DOF Franka 做 6D Pose task 是冗余系统，因此 IK 不唯一。
6. Command 不等于 State，Contact 不等于 Grasp，Table Contact 不等于 Settled。
7. Final waypoint 可达，不代表 waypoint 之间的运动合理。
8. Pseudoinverse 接近奇异点会产生巨大的关节修正；DLS 用阻尼换稳定性。
9. Adaptive DLS 能在危险区域自动增加阻尼，但不会主动避开奇异点。
10. Null Space 允许在基本不影响 TCP 主任务的情况下，利用冗余自由度完成 secondary task。

---

# 81. 项目完成状态

当前可明确标记为：

\[
\boxed{
\text{Franka Classical Manipulation Project — Core Complete}
}
\]

已经足够作为：

- 机器人基础项目；
- GitHub 项目；
- 简历项目；
- 后续 Robot Learning 的 classical baseline；
- 面试中讲 FK / IK / Trajectory / Singularity / Redundancy 的实践案例。

后续学习重点不应再是无限扩充这个项目，而是：

\[
\boxed{
\text{整理项目} \rightarrow
\text{Path Planning} \rightarrow
\text{Robot Learning}
}
\]
