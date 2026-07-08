---
title: Q 函数与 Value 函数
order: 15
tags: [强化学习]
category: 前置知识
---

# 前置知识：Q 函数与 Value 函数——RL 中"好坏"的数学定义

> **一句话**：Value 函数 $V(s)$ 回答"身处状态 $s$ 有多好"，Q 函数 $Q(s,a)$ 回答"在状态 $s$ 做动作 $a$ 有多好"。它们是 RL 所有算法的基石——无论是评估策略好坏、还是引导策略改进，都依赖它们。

**前置概念**：基础概率论（期望的含义）

---

## 贯穿全文的例子

> 一个机械臂在桌面上抓取方块。状态 $s$ 包含手臂位置和方块位置。动作 $a$ 是关节力矩。奖励：每步 $r=-0.01$（时间惩罚），抓到方块 $r=+10$。折扣因子 $\gamma = 0.99$。

---

## 一、为什么需要"价值"的概念

### 1.1 RL 的核心问题

强化学习的目标是最大化**累积奖励**（不是单步奖励）：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

$G_t$ 叫做 **return**（回报），表示"从时刻 $t$ 开始，未来所有奖励的折扣和"。

**问题**：在时刻 $t$ 做决策时，我们不知道未来的奖励是多少——它取决于后续的所有状态和动作。

**解决方案**：用**期望** return 来定义"好坏"——这就是 Value 函数。

### 1.2 两种"好坏"的量化

| 问题 | 数学定义 | 名称 |
|------|---------|------|
| "身处状态 $s$，未来大概能拿多少奖励？" | $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ | State-Value 函数 |
| "在状态 $s$ 做动作 $a$，未来大概能拿多少奖励？" | $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$ | Action-Value 函数（Q 函数） |

---

## 二、State-Value 函数 $V^\pi(s)$

### 2.1 定义

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s\right]
$$

**逐项拆解**：
- $\pi$：策略（决定在每个状态做什么动作）
- $\mathbb{E}_\pi[\cdot]$：在策略 $\pi$ 下的期望（对未来所有随机性取平均）
- $\gamma^k r_{t+k}$：未来第 $k$ 步的折扣奖励
- $s_t = s$：条件——当前处于状态 $s$

**一句话**：$V^\pi(s)$ 就是"如果我在状态 $s$ 开始，一直按策略 $\pi$ 行动，平均能拿多少总奖励"。

### 2.2 代入例子

机械臂当前在方块正上方 5cm 处（状态 $s_A$），使用一个"还不错"的策略 $\pi$：
- 大约 80% 的情况下，5 步内能抓到方块：$G \approx 5 \times (-0.01) + 10 = 9.95$
- 大约 20% 的情况下失败，再花 20 步才抓到：$G \approx 20 \times (-0.01) + 10 \times 0.99^{20} \approx 7.98$

$$
V^\pi(s_A) \approx 0.8 \times 9.95 + 0.2 \times 7.98 = 9.56
$$

如果机械臂在方块正上方 20cm 处（状态 $s_B$），需要更多步才能到达：

$$
V^\pi(s_B) \approx 7.2 \quad (\text{更远，需要更多时间惩罚步})
$$

$V^\pi(s_A) > V^\pi(s_B)$ → 状态 $s_A$ 比 $s_B$ "更好"（更接近目标）。

---

## 三、Action-Value 函数 $Q^\pi(s, a)$

### 3.1 定义

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s, a_t = a\right]
$$

**和 $V$ 的区别**：$V^\pi(s)$ 对动作也取了期望（按 $\pi(a|s)$ 加权平均）；$Q^\pi(s,a)$ 固定了第一步的动作为 $a$，后续才按 $\pi$ 行动。

**关系**：

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^\pi(s, a)] = \sum_a \pi(a|s) \cdot Q^\pi(s, a)
$$

$V$ 是 $Q$ 在动作上的加权平均。

### 3.2 代入例子

在状态 $s_A$（手在方块上方 5cm）：
- $Q^\pi(s_A, \text{下压}) = 9.8$（下压是对的动作，马上就能抓到）
- $Q^\pi(s_A, \text{向上}) = 6.5$（向上是错误动作，需要很多步才能绕回来）
- $Q^\pi(s_A, \text{向左}) = 7.0$（偏离，但不算太远）

策略 $\pi$ 在 $s_A$ 的动作概率：80% 下压、10% 向左、10% 向上：

$$
V^\pi(s_A) = 0.8 \times 9.8 + 0.1 \times 7.0 + 0.1 \times 6.5 = 9.19
$$

### 3.3 Q 函数的核心作用

Q 函数告诉我们**每个动作的"绝对价值"**。如果我们知道所有动作的 Q 值，最优策略就是：

$$
\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q^*(s, a') \\ 0 & \text{otherwise} \end{cases}
$$

**一句话**：知道了 Q 函数，就知道了最优策略——选 Q 值最大的动作就行。

---

## 四、Bellman 方程：Value 函数的递推关系

### 4.1 Bellman 期望方程

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[r(s,a) + \gamma \cdot \mathbb{E}_{s' \sim P}[V^\pi(s')]\right]
$$

$$
Q^\pi(s,a) = r(s,a) + \gamma \cdot \mathbb{E}_{s' \sim P}\left[V^\pi(s')\right] = r(s,a) + \gamma \cdot \mathbb{E}_{s' \sim P}\left[\mathbb{E}_{a' \sim \pi}[Q^\pi(s', a')]\right]
$$

**一句话**：当前的价值 = 即时奖励 + 折扣后的未来价值。这是一个**递推关系**。

**代入数字**：

在 $s_A$ 做"下压"动作：
- 即时奖励：$r(s_A, \text{下压}) = -0.01$（还没抓到，时间惩罚）
- 下一状态 $s'$：手在方块上方 2cm（$V^\pi(s') = 9.7$）
- $Q^\pi(s_A, \text{下压}) = -0.01 + 0.99 \times 9.7 = 9.59$

### 4.2 Bellman 最优方程

对于最优策略 $\pi^*$，Bellman 方程变为：

$$
Q^*(s, a) = r(s,a) + \gamma \cdot \mathbb{E}_{s'}\left[\max_{a'} Q^*(s', a')\right]
$$

**区别**：不再对动作取"按策略加权平均"，而是取 **max**——因为最优策略总是选最好的动作。

**DQN 等 Q-learning 算法**就是通过不断迭代这个方程来逼近 $Q^*$。

---

## 五、Advantage 函数

### 5.1 定义

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

**一句话**：Advantage 衡量"在状态 $s$ 做动作 $a$，比'按策略的平均表现'好多少"。

- $A > 0$：这个动作比平均好
- $A < 0$：这个动作比平均差
- $A = 0$：和平均一样

### 5.2 代入例子

在 $s_A$（$V^\pi(s_A) = 9.19$）：
- $A^\pi(s_A, \text{下压}) = 9.8 - 9.19 = +0.61$（比平均好）
- $A^\pi(s_A, \text{向上}) = 6.5 - 9.19 = -2.69$（比平均差很多！）

### 5.3 为什么策略梯度用 Advantage

[策略梯度](/前置知识/000a_前置知识_策略梯度与PPO) 的更新方向：$\nabla_\theta J \propto \mathbb{E}[\nabla\log\pi \cdot A]$

用 $A$ 而不是 $Q$，是因为 $A$ 天然有零均值（$\mathbb{E}_a[A(s,a)] = 0$），方差更小，训练更稳定。

---

## 六、总结

| 概念 | 定义 | 回答的问题 |
|------|------|-----------|
| $V^\pi(s)$ | $\mathbb{E}_\pi[\sum \gamma^k r_{t+k} \mid s]$ | 这个状态有多好？ |
| $Q^\pi(s,a)$ | $\mathbb{E}_\pi[\sum \gamma^k r_{t+k} \mid s, a]$ | 这个状态做这个动作有多好？ |
| $A^\pi(s,a)$ | $Q^\pi(s,a) - V^\pi(s)$ | 这个动作比平均好多少？ |
| Bellman 方程 | $Q = r + \gamma \mathbb{E}[V']$ | Value 的递推关系 |

---

## 延伸阅读

- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — Advantage 如何驱动策略更新
- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — Q 函数在 off-policy 算法中的使用
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — Value 函数与策略约束的关系
