# Day 3：MuJoCo 机械臂仿真与 PD 控制

## 1. 学习目标

Day 3 从“机器人运动学”进入“机器人动力学仿真与闭环控制”。

本日完成：

- 安装并测试 MuJoCo
- 理解 MuJoCo 的 model / data
- 加载 1DOF 机械臂
- 读取关节状态
- 配置 actuator
- 开环力矩控制
- 排查碰撞约束问题
- 实现 PD 闭环控制
- 记录与分析 PD 响应曲线

核心跃迁：

```text
Day 2:
关节角 -> 末端位置

Day 3:
控制输入 -> 动力学 -> 关节运动
```

---

## 2. MuJoCo 基本对象

### model

```python
model = mujoco.MjModel.from_xml_path(xml_path)
```

`model` 描述机器人“是什么”。

包含：

- 刚体
- 关节
- 几何形状
- 质量
- 惯量
- actuator
- gravity
- timestep
- joint limit

通常属于模型的固定参数。

### data

```python
data = mujoco.MjData(model)
```

`data` 描述机器人“现在怎么样”。

常用：

```python
data.qpos
data.qvel
data.qacc
data.ctrl
```

分别对应：

\[
q,\quad \dot q,\quad \ddot q,\quad u
\]

---

## 3. MuJoCo 物理仿真

核心：

```python
mujoco.mj_step(model, data)
```

它不是单纯更新画面，而是在执行物理仿真积分。

可以理解为：

```text
当前状态 q, qdot
        +
控制输入 u
        ↓
机器人动力学
        ↓
下一时刻 q, qdot
```

机器人动力学一般形式：

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)=\tau
\]

在 MuJoCo 中，还可能存在：

- contact force
- constraint force
- damping
- friction
- external force

---

## 4. Viewer

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
```

Viewer 负责显示仿真结果。

```python
viewer.sync()
```

负责同步显示。

Viewer 本身不是动力学计算器，真正的动力学推进由：

```python
mujoco.mj_step(...)
```

完成。

---

## 5. XML 中的 body、joint、geom

### body

表示刚体。

```xml
<body name="link1">
```

### joint

表示刚体之间允许的运动自由度。

```xml
<joint
    name="joint1"
    type="hinge"
    axis="0 0 1"
/>
```

`hinge` 表示旋转关节。

`axis="0 0 1"` 表示绕 z 轴旋转。

### geom

表示几何体。

```xml
<geom
    type="capsule"
    fromto="0 0 0 1 0 0"
    size="0.06"
/>
```

用于：

- 可视化
- 质量/惯量
- 碰撞检测

---

## 6. Joint 与 Actuator 的区别

有 joint 并不代表机器人一定有驱动。

```text
Joint
  ↓
定义哪里可以运动

Actuator
  ↓
决定如何驱动运动
```

Actuator 示例：

```xml
<actuator>
    <motor
        name="motor1"
        joint="joint1"
        gear="1"
        ctrllimited="true"
        ctrlrange="-10 10"
    />
</actuator>
```

---

## 7. MuJoCo 中控制相关变量

### data.ctrl

```python
data.ctrl[0]
```

表示发给 actuator 的控制命令。

注意：

> `ctrl` 不应在所有 actuator 模型中都直接理解成 N·m。

对于简单：

```xml
<motor gear="1"/>
```

可以近似理解为 actuator 控制量与关节力矩相对应。

### data.actuator_force

表示 actuator 自身产生的力。

### data.qfrc_actuator

表示 actuator 通过 transmission 后真正作用到关节上的广义力/力矩。

### data.qfrc_applied

表示额外人为施加的外部广义力。

这与 actuator force 不是同一个变量。

正确控制链：

\[
\boxed{
ctrl
\rightarrow
actuator\_force
\rightarrow
qfrc\_actuator
\rightarrow
qacc
\rightarrow
qvel
\rightarrow
qpos
}
\]

---

## 8. 本次调试中遇到的重要问题：打印错变量

最初使用：

```python
data.qfrc_applied[0]
```

观察“tau”，结果一直为 0。

原因：

`qfrc_applied` 是外部额外施加的广义力，而不是 actuator 产生的力矩。

正确应该查看：

```python
data.qfrc_actuator[0]
```

修改后得到：

```text
ctrl = 1
tau  = 1
```

说明 motor 已正确产生控制力矩。

---

## 9. 本次调试中遇到的重要问题：约束力抵消 actuator 力矩

虽然：

\[
\tau_{\text{actuator}}=1
\]

但机械臂仍基本不动。

进一步打印：

```python
data.qfrc_constraint[0]
```

发现：

\[
\tau_{\text{constraint}}
\approx -1.1 \sim -1.3
\]

即：

\[
\tau_{\text{actuator}}
+
\tau_{\text{constraint}}
\approx 0
\]

因此：

\[
\ddot q\approx 0
\]

关节几乎不动。

---

## 10. 为什么产生 constraint force

模型中的 base 与 link1 几何体在关节处存在重叠/接触。

MuJoCo 将它们视为发生碰撞，因此产生接触约束力。

这说明：

> 机器人内部的相邻连杆不一定应该参与 self-collision。

---

## 11. 排除相邻刚体碰撞

在 XML 中增加：

```xml
<contact>
    <exclude body1="base" body2="link1"/>
</contact>
```

意义：

> 排除 base 与 link1 之间的碰撞检测。

不是关闭所有碰撞，而是针对机器人内部相邻刚体排除不合理接触。

修改后：

\[
qfrc_{\text{constraint}}=0
\]

并观察到：

\[
qacc>0
\]

\[
qdot \uparrow
\]

\[
q \uparrow
\]

动力学链路恢复正常。

---

## 12. 开环力矩控制

代码：

```python
data.ctrl[0] = 1.0
```

持续施加固定正控制输入。

对于简化旋转系统，可以近似理解：

\[
I\ddot q+b\dot q=\tau
\]

一开始：

\[
\dot q=0
\]

因此：

\[
I\ddot q=\tau
\]

若：

\[
\tau>0
\]

则：

\[
\ddot q>0
\]

随后：

\[
\dot q\uparrow
\]

\[
q\uparrow
\]

---

## 13. 为什么恒定力矩不能让机械臂停在目标角度

恒定控制：

```python
data.ctrl[0] = 1.0
```

并不知道目标角度在哪里。

因此机械臂不会自动停在：

\[
q_d=90^\circ
\]

这属于：

> 开环控制。

需要反馈控制。

---

## 14. P 控制

目标角度：

\[
q_d
\]

当前位置：

\[
q
\]

误差：

\[
e=q_d-q
\]

P 控制：

\[
\boxed{
\tau=K_p(q_d-q)
}
\]

特点：

- 距离目标越远，控制越强
- 接近目标时，控制变小

但是系统存在惯性，单纯 P 控制可能发生：

- 超调
- 振荡

---

## 15. PD 控制

加入速度反馈：

\[
\boxed{
\tau=
K_p(q_d-q)
+
K_d(\dot q_d-\dot q)
}
\]

对于目标静止：

\[
\dot q_d=0
\]

得到：

\[
\boxed{
\tau=
K_p(q_d-q)-K_d\dot q
}
\]

其中：

- P：根据位置误差提供“拉向目标”的力矩
- D：根据速度提供阻尼，减少超调和振荡

---

## 16. Actuator 限幅

XML：

```xml
ctrlrange="-10 10"
```

代码中也使用：

```python
torque = np.clip(torque, -10.0, 10.0)
```

因此实际控制器是：

\[
\boxed{
\tau=
\operatorname{clip}
\left(
K_pe+K_d\dot e,
-10,
10
\right)
}
\]

这体现了实际机器人中的 actuator saturation。

---

## 17. PD 闭环结构

```text
             q_desired
                 │
                 ↓
          error = qd - q
                 │
                 ↓
          PD Controller
                 │
                 ↓
              torque
                 │
                 ↓
             MuJoCo
                 │
            ┌────┴────┐
            ↓         ↓
            q        qdot
            │         │
            └─────────┘
              feedback
```

这就是标准闭环控制。

---

## 18. PD 参数影响

### \(K_p\)

增大 \(K_p\) 通常意味着：

- 响应更快
- 系统“刚度”更大
- 更容易超调或振荡

### \(K_d\)

增大 \(K_d\) 通常意味着：

- 阻尼增强
- 超调减小
- 振荡减弱
- 过大时响应可能变慢

推荐实验：

```text
Kp = 10, Kd = 0
Kp = 10, Kd = 2
Kp = 10, Kd = 5
```

比较响应差异。

---

## 19. PD 数据记录

实验中建议记录：

\[
t,\quad q_d,\quad q,\quad \dot q,\quad \tau,\quad e
\]

对应：

```python
time_history
q_desired_history
q_history
qdot_history
torque_history
error_history
```

建议在：

```python
mujoco.mj_step(model, data)
```

之后记录状态。

---

## 20. 为什么仿真结束后统一画图

不建议在仿真循环中不断执行：

```python
plt.plot(...)
```

因为会严重影响实时仿真。

更合理流程：

```text
运行仿真
  ↓
记录数据
  ↓
仿真结束
  ↓
统一绘图
```

---

## 21. 推荐绘制的四类响应曲线

### 21.1 关节位置响应

绘制：

\[
q(t)
\]

与：

\[
q_d(t)
\]

观察：

- 上升过程
- 超调
- 稳态

### 21.2 误差曲线

\[
e(t)=q_d-q(t)
\]

理想：

\[
e(t)\rightarrow0
\]

### 21.3 速度响应

\[
\dot q(t)
\]

观察：

- 加速
- 峰值
- 减速
- 最终趋近 0

### 21.4 控制力矩

\[
\tau(t)
\]

观察：

- 初始饱和
- 减小
- 可能反向制动
- 最终趋近 0

---

## 22. 控制性能指标

### 22.1 稳态误差

\[
e_{ss}=q_d-q(\infty)
\]

实际可近似使用：

```python
error_history[-1]
```

### 22.2 最大超调

\[
M_p=
\frac{q_{\max}-q_d}{|q_d|}
\times100\%
\]

### 22.3 上升时间

常用定义：

> 从目标值 10% 上升到 90% 所需时间。

### 22.4 调节时间

常用 2% 标准：

> 系统进入目标值 ±2% 区间，并且以后不再离开。

---

## 23. 本日代码结构

```text
day03_mujoco_control/
├── assets/
│   └── one_joint_arm.xml
├── 01_test_mujoco.py
├── 02_load_robot.py
├── 03_read_joint_state.py
├── 04_joint_control.py
├── 05_pd_control.py
└── README.md
```

### 01_test_mujoco.py

检查 MuJoCo 是否安装成功：

```python
print(mujoco.__version__)
```

### 02_load_robot.py

加载 XML，并打开 Viewer。

### 03_read_joint_state.py

读取：

```python
data.qpos
data.qvel
```

### 04_joint_control.py

完成：

```python
data.ctrl[0] = 1.0
```

并观察：

\[
ctrl\rightarrow\tau\rightarrow qacc\rightarrow qvel\rightarrow q
\]

### 05_pd_control.py

完成：

- PD 控制
- 限幅
- 数据记录
- 响应绘制
- 性能指标分析

---

## 24. Day 3 最重要的调试经验

### 经验 1

不要只看：

```python
data.ctrl
```

因为它只是控制命令。

真正关节力矩要看：

```python
data.qfrc_actuator
```

### 经验 2

`qfrc_applied` 和 `qfrc_actuator` 不一样。

### 经验 3

机器人不动不一定是 actuator 没有输出。

还需要检查：

```python
data.qfrc_constraint
```

### 经验 4

碰撞约束可能把 actuator 力矩完全抵消。

### 经验 5

机器人动力学不是：

\[
\tau\rightarrow\ddot q
\]

这么简单，而是多个动力学项共同决定：

\[
M(q)\ddot q
=
\tau_{\text{actuator}}
+
\tau_{\text{constraint}}
-
\tau_{\text{bias}}
+\cdots
\]

---

## 25. 三天知识之间的连接

### Day 1

解决：

> 坐标系和姿态怎么表示？

核心：

\[
R,\quad T,\quad q_{\text{quaternion}}
\]

### Day 2

解决：

> 机械臂的几何位置和速度怎么计算？

核心：

\[
FK,\quad IK,\quad J
\]

### Day 3

解决：

> 如何通过力矩让机械臂真正运动并稳定到目标？

核心：

\[
\tau,\quad q,\quad \dot q,\quad PD
\]

因此完整逻辑是：

```text
坐标表示
    ↓
机器人运动学
    ↓
机器人动力学仿真
    ↓
闭环控制
```

---

## 26. Day 3 核心公式

机器人动力学基本形式：

\[
\boxed{
M(q)\ddot q+C(q,\dot q)\dot q+g(q)=\tau
}
\]

P 控制：

\[
\boxed{
\tau=K_p(q_d-q)
}
\]

PD 控制：

\[
\boxed{
\tau=
K_p(q_d-q)+K_d(\dot q_d-\dot q)
}
\]

静止目标：

\[
\boxed{
\tau=
K_p(q_d-q)-K_d\dot q
}
\]

---

## 27. 复习检查

学完后应能够独立回答：

1. `model` 和 `data` 的区别是什么？
2. `mj_step()` 在做什么？
3. `qpos/qvel/qacc/ctrl` 分别表示什么？
4. joint 与 actuator 有什么区别？
5. `ctrl` 是否一定等于 N·m？
6. `qfrc_actuator` 和 `qfrc_applied` 有什么区别？
7. 为什么机械臂有 actuator 力矩却可能仍然不动？
8. constraint force 是什么？
9. 为什么要排除相邻刚体之间不合理的 self-collision？
10. 为什么恒定力矩不能让机械臂停在目标角度？
11. P 控制为什么可能振荡？
12. D 项为什么可以减少振荡？
13. actuator saturation 对控制有什么影响？
14. PD 控制实验应该观察哪些响应曲线？
15. 上升时间、超调、调节时间、稳态误差分别表示什么？
