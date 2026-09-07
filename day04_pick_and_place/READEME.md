# Day 4：多关节机械臂、IK 控制与 Pick-and-Place

## 1. 学习目标

Day 4 的目标是把前三天的知识真正串起来，形成一个完整的机器人任务链路：

```text
Day 1
坐标系 / 旋转 / 齐次变换

        ↓

Day 2
FK / IK / Jacobian

        ↓

Day 3
MuJoCo / Actuator / PD 控制

        ↓

Day 4
多关节机械臂
→ 末端目标
→ IK
→ Joint-space PD
→ Reach 判断
→ Pick-and-Place 状态机
→ 简化夹爪 / 真实接触夹爪尝试
```

本日主要学习内容：

- MuJoCo 中建立 2DOF 平面机械臂
- 两关节 PD 控制
- 任务空间目标到关节空间目标的 IK 映射
- FK 验证末端位置
- Task-space error 与 Joint-space error
- Reach threshold 与稳定到达判定
- 有限状态机 FSM
- 简化磁吸式 Pick-and-Place
- 两指夹爪的建模与控制
- Contact / friction / grasp verification
- 2DOF 机械臂在真实抓取中的自由度限制

---

## 2. Day 4 的整体知识结构

Day 4 的核心逻辑是：

\[
\boxed{
p_d
\rightarrow
IK
\rightarrow
q_d
\rightarrow
PD
\rightarrow
\tau
\rightarrow
MuJoCo
\rightarrow
q
\rightarrow
FK
\rightarrow
p
}
\]

其中：

- \(p_d\)：末端目标位置
- \(q_d\)：IK 得到的目标关节角
- \(\tau\)：关节控制力矩
- \(q\)：实际关节角
- \(p\)：实际末端位置

这条链把运动学和动力学控制连接起来。

---

## 3. 2DOF 平面机械臂 MuJoCo 模型

本日首先从单关节升级到双关节：

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
```

状态：

\[
q=
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix}
\]

速度：

\[
\dot q=
\begin{bmatrix}
\dot q_1\\
\dot q_2
\end{bmatrix}
\]

控制输入：

\[
\tau=
\begin{bmatrix}
\tau_1\\
\tau_2
\end{bmatrix}
\]

MuJoCo 中正常应有：

```text
nq = 2
nv = 2
nu = 2
```

分别表示：

- 两个位置自由度
- 两个速度自由度
- 两个 actuator

---

## 4. 两关节 PD 控制

目标关节角：

\[
q_d=
\begin{bmatrix}
30^\circ\\
60^\circ
\end{bmatrix}
\]

MuJoCo 内部统一使用弧度，所以必须写：

```python
q_desired = np.deg2rad([30.0, 60.0])
```

不能直接写：

```python
q_desired = np.array([30.0, 60.0])
```

否则 MuJoCo 会把它理解为：

\[
30\text{ rad},\quad 60\text{ rad}
\]

对应：

\[
1718.87^\circ,\quad 3437.75^\circ
\]

这是典型的 degree / radian 单位错误。

### 4.1 双关节 PD 控制律

\[
\boxed{
\tau=
K_p(q_d-q)+K_d(\dot q_d-\dot q)
}
\]

若目标静止：

\[
\dot q_d=0
\]

则：

\[
\boxed{
\tau=
K_p(q_d-q)-K_d\dot q
}
\]

代码中可以采用逐元素形式：

```python
torque = (
    Kp * position_error
    +
    Kd * velocity_error
)
```

再进行 actuator 限幅：

```python
torque = np.clip(
    torque,
    -20.0,
    20.0
)
```

最后：

```python
data.ctrl[:] = torque
```

---

## 5. 2DOF 比 1DOF 更难控制的原因

单关节可以近似看作：

\[
I\ddot q+b\dot q=\tau
\]

双关节机械臂则更接近：

\[
\boxed{
M(q)\ddot q
+
C(q,\dot q)\dot q
+
g(q)
=
\tau
}
\]

其中：

- \(M(q)\)：惯性矩阵
- \(C(q,\dot q)\dot q\)：科氏力 / 离心力项
- \(g(q)\)：重力项

双关节之间存在动力学耦合，因此一个关节的运动会影响另一个关节。

### 5.1 PD 参数实验

本日实验中：

```text
Kp = 20
Kd = 3
```

振荡较明显。

随着：

\[
K_d\uparrow
\]

阻尼增强。

最终：

```text
Kp = 20
Kd = 10
```

效果更好。

这验证了：

\[
\boxed{
K_d \text{ 主要用于抑制速度过大、超调和振荡}
}
\]

---

## 6. 从 Joint Target 升级到 Cartesian Target

此前：

```python
q_desired = np.deg2rad([30, 60])
```

由程序员直接给关节角目标。

Day 4 的下一步改为：

```python
target = np.array([1.2, 0.8])
```

即直接给：

\[
p_d=
\begin{bmatrix}
x_d\\
y_d
\end{bmatrix}
\]

然后通过 IK 自动求：

\[
q_d=
\begin{bmatrix}
q_1^d\\
q_2^d
\end{bmatrix}
\]

流程：

```text
Cartesian target
      ↓
      IK
      ↓
Joint target
      ↓
Joint-space PD
```

---

## 7. 2DOF 解析 IK

对于连杆长度：

\[
l_1,\quad l_2
\]

目标：

\[
(x,y)
\]

先求：

\[
\boxed{
\cos q_2=
\frac{x^2+y^2-l_1^2-l_2^2}
{2l_1l_2}
}
\]

然后：

\[
q_2=\arccos(\cos q_2)
\]

再求：

\[
\boxed{
q_1=
\operatorname{atan2}(y,x)
-
\operatorname{atan2}
\left(
l_2\sin q_2,\,
l_1+l_2\cos q_2
\right)
}
\]

---

## 8. 工作空间检查

若：

\[
|\cos q_2|>1
\]

说明目标不可达。

等价于：

\[
|l_1-l_2|
\le
\sqrt{x^2+y^2}
\le
l_1+l_2
\]

若不满足，应直接报错：

```python
raise ValueError(
    "Target is outside workspace."
)
```

机器人控制不应该对不可达目标盲目输出大力矩。

---

## 9. FK 验证 IK

IK 得到：

\[
q_d
\]

以后，需要通过 FK 再计算：

\[
p_{\text{check}}=f(q_d)
\]

然后检查：

\[
\boxed{
\|p_d-p_{\text{check}}\|
\approx0
}
\]

这是一种非常重要的调试方法：

```text
target
 ↓
IK
 ↓
q_desired
 ↓
FK
 ↓
target_check
```

如果误差不接近 0，说明：

- IK 推导错误
- 连杆长度参数不一致
- 坐标系定义不一致

---

## 10. Joint-space error 与 Task-space error

### 10.1 Joint-space error

\[
\boxed{
e_q=q_d-q
}
\]

PD 控制器真正使用的是这个误差。

### 10.2 Task-space error

\[
\boxed{
e_p=p_d-p
}
\]

实际常使用：

\[
\|e_p\|
=
\|p_d-p\|
\]

它表示机械臂末端距离目标还有多远。

### 10.3 两种误差的关系

当前控制方式是：

```text
Task target
    ↓
IK
    ↓
Joint target
    ↓
Joint-space PD
```

因此这仍然属于：

\[
\boxed{\text{IK + Joint-space PD}}
\]

不是直接的 Task-space PD。

---

## 11. `03_ik_control.py` 的完成结果

对于目标：

\[
p_d=(1.2,\ 0.8)
\]

最终得到类似：

\[
p\approx(1.2055,\ 0.7935)
\]

任务空间误差：

\[
\|e_p\|\approx0.0085m
\]

即约：

\[
8.5\text{ mm}
\]

说明：

\[
\boxed{
IK + PD + MuJoCo + FK
}
\]

整条链已经跑通。

---

## 12. Reach Task：什么时候才算真正“到达”？

不能简单判断：

```python
if task_error < threshold:
    success = True
```

因为机械臂可能只是瞬间经过目标。

例如：

```text
error = 0.0013
↓
error = 0.0067
↓
error = 0.0224
```

说明系统可能进入目标附近后又振出去。

### 12.1 Reach threshold

例如：

\[
\epsilon=0.01
\]

要求：

\[
\boxed{
\|p_d-p\|<0.01
}
\]

即末端距离目标小于 1 cm。

### 12.2 Stable Counter

定义：

```python
stable_counter = 0
required_stable_steps = 200
```

如果：

```python
if task_error < reach_threshold:
    stable_counter += 1
else:
    stable_counter = 0
```

MuJoCo timestep：

\[
\Delta t=0.002s
\]

若：

\[
200\text{ steps}
\]

则要求：

\[
200\times0.002=0.4s
\]

持续在阈值内。

因此成功条件变成：

\[
\boxed{
\|p_d-p(t)\|<\epsilon
\quad
\text{持续一定时间}
}
\]

这比瞬时判断更符合机器人任务逻辑。

---

## 13. 从控制层进入任务层

`03_ik_control.py` 解决：

> 机器人能不能到目标？

`04_reach_target.py` 开始解决：

> 什么时候算任务完成？

这意味着从 Control Layer 开始进入 Task Layer。

---

## 14. 有限状态机 FSM

Pick-and-Place 不是一个动作，而是一系列阶段。

最简单版本：

```text
MOVE_TO_CUBE
      ↓
GRASP
      ↓
MOVE_TO_TARGET
      ↓
RELEASE
      ↓
DONE
```

这就是：

\[
\boxed{\text{Finite State Machine}}
\]

程序形式：

```python
if state == MOVE_TO_CUBE:
    ...

elif state == GRASP:
    ...

elif state == MOVE_TO_TARGET:
    ...

elif state == RELEASE:
    ...

elif state == DONE:
    ...
```

---

## 15. 高层任务与底层控制

此时系统已经具有明显分层结构：

```text
Task Layer
Finite State Machine
        ↓
Target
        ↓
IK
        ↓
q_desired
        ↓
PD Controller
        ↓
Torque
        ↓
MuJoCo Dynamics
```

上层回答：

> 机器人现在应该做什么？

下层回答：

> 机器人应该怎么运动？

这是机器人软件架构中非常重要的思想。

---

## 16. 简化“磁吸式” Pick-and-Place

第一版没有真实夹爪。

抓取通过：

```python
cube_grasped = True
```

然后：

```python
data.mocap_pos[...] = ee_position
```

让 cube 直接跟随末端。

本质上是：

\[
\boxed{\text{Virtual Magnetic Gripper}}
\]

优点：

- 简单
- 容易验证 FSM
- 不需要处理摩擦和接触

缺点：

- 不是真实物理抓取
- 没有夹爪
- 没有抓取稳定性

---

## 17. 磁吸版中的碰撞问题

最初要求：

\[
p_{ee}=p_{cube}
\]

但末端和 cube 都有真实 geometry。

两个实体不可能中心完全重合，因为会先发生碰撞。

因此出现：

```text
EE approaching cube
      ↓
collision
      ↓
constraint force
      ↓
EE cannot reach cube center
      ↓
stable_counter always 0
```

在磁吸教学版中，可临时关闭 cube 碰撞：

```xml
contype="0"
conaffinity="0"
```

因为磁吸版本来就不是依靠真实 contact 抓取。

---

## 18. 真实两指夹爪升级

为了进一步学习 manipulation，进行了一个小升级：

```text
2DOF arm
+
two-finger gripper
+
dynamic cube
+
contact
+
friction
```

结构：

```text
link2
  ↓
gripper base
  ├── left finger
  └── right finger
```

夹爪使用 slide joint：

```xml
type="slide"
```

两根手指沿相反方向移动。

---

## 19. Position Actuator

机械臂关节：

```xml
<motor ... />
```

控制更接近力矩输入。

夹爪：

```xml
<position ... />
```

控制的是：

\[
q_{\text{finger,target}}
\]

例如：

```python
data.ctrl[left_actuator] = 0.065
```

表示：

> 希望手指 slide joint 移动到 0.065 m。

不是：

\[
0.065N\cdot m
\]

---

## 20. 为什么两根手指可以给相同的控制值

左手指 axis：

```xml
axis="0 -1 0"
```

右手指 axis：

```xml
axis="0 1 0"
```

因此：

\[
q_L\uparrow
\]

与：

\[
q_R\uparrow
\]

在世界空间中会朝相反方向运动。

所以可以同时写：

```python
left_ctrl = 0.065
right_ctrl = 0.065
```

实现向中间闭合。

---

## 21. 真实抓取需要 Contact Verification

不能再用：

\[
\|p_{ee}-p_{cube}\|<d
\]

判断抓取成功。

距离近只说明夹爪靠近 cube，并不能说明两根手指真正夹住 cube。

### 21.1 接触检测

MuJoCo 中可以检查：

```python
data.contact[i].geom1
data.contact[i].geom2
```

判断：

- left finger ↔ cube
- right finger ↔ cube

真正的第一版抓取条件：

\[
\boxed{
Contact_L
\land
Contact_R
}
\]

并且要求持续一定时间。

---

## 22. 为什么 `data.ncon` 不够

`data.ncon` 只是当前场景中一共有多少接触。

它可能包括：

- cube ↔ ground
- base ↔ ground
- finger ↔ robot
- finger ↔ cube

所以：

```text
Contacts = 4
```

不能推出：

> 两根手指已经夹住 cube。

必须检查具体 geom pair。

---

## 23. 抓取过程中出现“推着方块走”

这是 Day 4 最有价值的失败案例之一。

现象：

```text
gripper approaches cube
↓
cube moves away
↓
robot keeps following
↓
still cannot grasp
```

主要原因可能包括：

- 抓取中心定义错误
- gripper base 先碰到 cube
- grasp pose 不正确
- approach direction 不正确
- 2DOF 无法独立控制夹爪姿态
- target 实时跟随被推走的 cube，形成“追着推”

---

## 24. Grasp Site：真正应该控制的是抓取中心

不能简单让：

\[
p_{\text{gripper base}}
=
p_{\text{cube}}
\]

因为 gripper base 是实体，会先撞到物体。

应该定义：

```xml
<site
    name="grasp_site"
    ...
/>
```

它位于两根手指之间。

然后控制：

\[
\boxed{
p_{\text{grasp site}}
\rightarrow
p_{\text{cube}}
}
\]

这才符合抓取几何逻辑。

---

## 25. 抓取位姿不只是位置

真实抓取通常不是只要求：

\[
x,y
\]

还要求夹爪姿态：

\[
\phi
\]

即任务实际上更接近：

\[
\boxed{
(x,y,\phi)
}
\]

---

## 26. 2DOF 机械臂真实抓取的自由度限制

当前：

\[
q=
\begin{bmatrix}
q_1\\q_2
\end{bmatrix}
\]

只有 2 个自由度。

可以独立控制：

\[
x,y
\]

但是夹爪末端姿态：

\[
\phi=q_1+q_2
\]

不是一个额外可以自由指定的变量。

因此：

\[
\boxed{
2DOF < 3\text{ task dimensions }(x,y,\phi)
}
\]

一般无法同时满足：

\[
x=x_d,\quad y=y_d,\quad \phi=\phi_d
\]

这就是为什么夹爪可能能到 cube 附近，却以错误姿态接近，从而变成推物体，而不是夹物体。

---

## 27. 为什么“平面”不是根本问题

平面抓取本身完全可以完成。

真正的问题是：

> 自由度不足以独立控制位置和抓取姿态。

若使用 3DOF 平面机械臂：

\[
q=
\begin{bmatrix}
q_1\\
q_2\\
q_3
\end{bmatrix}
\]

就可以控制：

\[
\boxed{
x,\ y,\ \phi
}
\]

第三个关节可以专门调整夹爪姿态。

---

## 28. 3DOF 平面 IK 的基本思路

若末端目标：

\[
(x_d,y_d,\phi_d)
\]

第三段长度：

\[
l_3
\]

先求 wrist center：

\[
\boxed{
x_w=x_d-l_3\cos\phi_d
}
\]

\[
\boxed{
y_w=y_d-l_3\sin\phi_d
}
\]

然后使用已掌握的 2DOF IK 求：

\[
q_1,q_2
\]

最后：

\[
\boxed{
q_3=\phi_d-q_1-q_2
}
\]

这会是后续升级真实抓取的自然方向。

---

## 29. Pre-grasp 与 Approach

真实抓取不应该直接冲向 cube center。

更合理流程：

```text
PREGRASP
    ↓
APPROACH
    ↓
CLOSE GRIPPER
    ↓
VERIFY GRASP
```

Pre-grasp 是一个安全的抓取准备位姿。

Approach 则沿正确方向缓慢进入抓取位置。

---

## 30. 为什么不能实时追着被推走的 cube

如果每一步都：

```python
cube_xy = current_cube_position
target = cube_xy
```

而夹爪开始推 cube：

```text
cube moves
↓
target moves
↓
robot follows
↓
cube moves again
```

就形成：

\[
\boxed{\text{追着方块推}}
\]

更合理做法是：

> 在开始抓取时锁定 grasp target。

如果方块被明显推离，则应该判定抓取失败，而不是一直追。

---

## 31. HOLD + CLOSE

进入闭合阶段时，机械臂最好：

```text
hold position
↓
close fingers
```

而不是：

```text
arm moving
+
fingers closing
```

后者很容易把方块打飞。

因此抓取状态机更合理地写成：

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
```

---

## 32. Day 4 中的三个控制层次

### 32.1 Dynamics Layer

关注：

```text
torque
constraint
contact
friction
```

### 32.2 Motion Control Layer

关注：

```text
IK
q_desired
PD
joint motion
task error
```

### 32.3 Task Layer

关注：

```text
MOVE_TO_OBJECT
GRASP
MOVE_TO_TARGET
RELEASE
DONE
```

这三层在真实机器人系统中通常会被明显分开。

---

## 33. 与强化学习 / 具身智能的连接

当前经典机器人控制：

```text
State Machine
    ↓
IK
    ↓
PD
    ↓
Action
```

以后强化学习可能变成：

```text
Observation
    ↓
Policy Network
    ↓
Action
```

例如：

\[
s=
[
q,\dot q,p_{cube},p_{target}
]
\]

策略：

\[
a=\pi_\theta(s)
\]

动作可能是：

\[
a=
[
\tau_1,\tau_2
]
\]

也可能是：

- joint position target
- joint velocity target
- end-effector delta pose

因此现在学习的 classical robotics，可以帮助理解：

> 强化学习究竟替代了系统中的哪一层。

---

## 34. 本日项目结构

```text
day04_pick_and_place/
├── assets/
│   ├── two_link_arm.xml
│   ├── pick_place_scene.xml
│   └── gripper_scene.xml
│
├── 01_load_2dof_arm.py
├── 02_joint_pd_control.py
├── 03_ik_control.py
├── 04_reach_target.py
├── 05_pick_and_place.py
├── 06a_test_gripper.py
├── 06_real_gripper_pick.py
└── README.md
```

---

## 35. 各程序作用

### `01_load_2dof_arm.py`

- 加载 2DOF 模型
- 检查 `nq/nv/nu`
- 设置初始关节角
- 使用 `mj_forward()` 更新姿态

### `02_joint_pd_control.py`

\[
q_d
\rightarrow
PD
\rightarrow
q
\]

验证两个关节都可以稳定控制。

### `03_ik_control.py`

\[
p_d
\rightarrow
IK
\rightarrow
q_d
\rightarrow
PD
\]

并通过 FK 验证：

\[
p\rightarrow p_d
\]

### `04_reach_target.py`

新增：

- Task-space error
- Reach threshold
- Stable counter
- SUCCESS / TIMEOUT

完成从控制层向任务层的过渡。

### `05_pick_and_place.py`

使用简化磁吸夹爪：

```text
MOVE_TO_CUBE
GRASP
MOVE_TO_TARGET
RELEASE
DONE
```

主要学习 FSM。

### `06a_test_gripper.py`

只测试：

```text
OPEN
↓
CLOSE
↓
OPEN
```

用于验证：

- finger slide joint
- position actuator
- gripper direction

### `06_real_gripper_pick.py`

尝试真实：

- dynamic cube
- left/right contact
- friction
- grasp verification
- drop detection
- task FSM

当前实验暴露了 2DOF 抓取的姿态控制限制。

---

## 36. Day 4 重要调试经验

1. **角度单位统一**：MuJoCo 内部使用 rad，显示时再转 deg。
2. **数学模型必须和 XML 一致**：连杆长度和 grasp site 的几何位置要一致。
3. **Reach ≠ Grasp**：到物体附近不等于抓住物体。
4. **Contact count ≠ Grasp**：`data.ncon` 不能证明两指都接触了 cube。
5. **抓取是位姿问题，不只是位置问题**：真实 grasp 需要位置和姿态。
6. **失败案例很重要**：“推着物体走”说明 grasp pose、approach 或自由度设计有问题。
7. **不要在目标被推走后一直追**：抓取前应锁定 grasp target。
8. **闭合夹爪时机械臂应尽量保持位置**：降低“边移动边挤压”导致的推物现象。

---

## 37. Day 1～Day 4 的完整知识链

```text
Day 1
坐标系 / Rotation / Transformation
          ↓
Day 2
FK / IK / Jacobian
          ↓
Day 3
MuJoCo / Dynamics / Actuator / PD
          ↓
Day 4
Multi-joint control
          ↓
Cartesian target
          ↓
IK
          ↓
Joint-space PD
          ↓
Reach detection
          ↓
FSM
          ↓
Pick-and-Place
          ↓
Contact / Gripper / Grasp
```

这已经形成了一个完整的机器人 manipulation 入门框架。

---

## 38. Day 4 核心公式

2DOF FK：

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

2DOF IK：

\[
\boxed{
\cos q_2=
\frac{x^2+y^2-l_1^2-l_2^2}
{2l_1l_2}
}
\]

Joint-space PD：

\[
\boxed{
\tau=
K_p(q_d-q)-K_d\dot q
}
\]

Task-space error：

\[
\boxed{
e_p=p_d-p
}
\]

Reach condition：

\[
\boxed{
\|e_p\|<\epsilon
}
\]

2DOF 末端姿态：

\[
\boxed{
\phi=q_1+q_2
}
\]

3DOF wrist orientation：

\[
\boxed{
q_3=\phi_d-q_1-q_2
}
\]

---

## 39. 复习检查

学完 Day 4 后，应能够独立回答：

1. 为什么多关节机械臂不能简单看成多个完全独立的单关节系统？
2. MuJoCo 中为什么角度目标必须使用 rad？
3. Joint-space target 和 Task-space target 有什么区别？
4. 为什么要先 IK 再做 Joint-space PD？
5. 为什么要用 FK 验证 IK？
6. Joint-space error 和 Task-space error 分别是什么？
7. Reach threshold 为什么不能只判断一个瞬间？
8. Stable counter 的物理意义是什么？
9. FSM 为什么适合 Pick-and-Place？
10. 高层任务和底层控制有什么区别？
11. 磁吸抓取和真实接触抓取有什么区别？
12. position actuator 和 motor actuator 有什么区别？
13. 为什么 `data.ncon` 不能直接证明抓取成功？
14. 如何检查左右手指是否分别与 cube 接触？
15. 为什么“到达 cube”不等于“抓住 cube”？
16. 为什么夹爪会出现推着物体走的现象？
17. 什么是 grasp site？
18. 什么是 pre-grasp？
19. 为什么抓取需要考虑 orientation？
20. 为什么 2DOF 难以同时控制 \(x,y,\phi\)？
21. 3DOF 平面机械臂为什么更适合真实抓取？
22. 当前经典机器人控制链和强化学习策略控制之间有什么联系？

---

## 40. Day 4 当前结论

Day 4 已经成功完成：

\[
\boxed{
Cartesian\ Target
\rightarrow
Analytical\ IK
\rightarrow
Joint\ PD
\rightarrow
MuJoCo
\rightarrow
Reach
}
\]

并完成：

\[
\boxed{
FSM\ based\ simplified\ Pick-and-Place
}
\]

真实两指夹爪实验则进一步暴露出：

\[
\boxed{
Grasp \neq Reach
}
\]

以及：

\[
\boxed{
抓取需要位置 + 姿态 + 接触 + 摩擦 + 合理的 approach 几何
}
\]

当前 2DOF 平面夹爪“推而不夹”的问题暂时保留，后续可以升级为：

\[
\boxed{
3DOF\ planar\ arm
+
two\ finger\ gripper
}
\]

再继续完成真正稳定的接触抓取。
