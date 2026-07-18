---
title: 策略梯度与 PPO
order: 1
tags: [强化学习]
category: 前置知识
---

# 前置知识：策略梯度与 PPO（从"想要什么"到完整算法，把每个字母拆透）

> **为什么要读这篇**：几乎所有机器人 RL 论文（DPPO、AGILE、VLA-RL...）都用 PPO 做核心优化器。这篇文章不满足于"贴公式讲一遍"，而是把 PPO 当成一条**连续的推导链条**——从"我们到底想要什么"开始，每往前走一步都回答"为什么上一步不够、这一步补上了什么"，直到走到你在任何 PPO 代码里都能看到的那几行 loss。走完这条链条，你会知道 PPO 里的每一个符号、每一个系数、每一行代码是从哪里"长出来"的，而不是背下来的。
>
> **涉及原始论文**：
> - REINFORCE (Williams, 1992)
> - TRPO: Trust Region Policy Optimization (Schulman et al., 2015, arXiv:1502.05477)
> - GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015, arXiv:1506.02438)
> - PPO: Proximal Policy Optimization Algorithms (Schulman et al., 2017, arXiv:1707.06347)

**标签**: `#前置知识` `#强化学习` `#策略梯度` `#PPO` `#TRPO` `#GAE` `#Actor-Critic`

**相关阅读（建议读完本文再看，或对照着看）**：
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — 本文会大量用到 $Q(s,a)$、$V(s)$、$A(s,a)$ 的定义，如果这三个符号对你还很陌生，建议先读这篇
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — TRPO/PPO 的约束机制背后的信息论工具

---

## 0. 先看一眼终点：我们最终要写出的东西

在开始推导之前，先把"终点"摆出来，让你知道整条链条要走到哪里去。**下面这几行，就是任何 PPO 实现（CleanRL、Stable-Baselines3、RLinf...）里都会出现的核心代码逻辑**，本文接下来的每一节，都是在解释这里的每一个符号从哪里来、为什么长这样：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, \qquad
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]
$$

$$
L^{\text{total}}(\theta,\phi) = -L^{\text{CLIP}}(\theta) + c_1 \cdot \underbrace{\big(V_\phi(s_t) - \hat{R}_t\big)^2}_{\text{价值损失}} - c_2 \cdot \underbrace{H[\pi_\theta(\cdot|s_t)]}_{\text{熵奖励}}
$$

这两行公式里出现了 $\pi_\theta$、$\pi_{\theta_{\text{old}}}$、$r_t(\theta)$、$\hat{A}_t$、$\text{clip}$、$V_\phi$、$\hat{R}_t$、$H[\cdot]$ 八个"陌生符号"。本文的目标就是：**推导完之后，你能不看任何资料，把这两行公式里的每一个符号讲给别人听**——它是什么、为什么需要它、如果去掉它会发生什么。

我们会按照这条路线走：

$$
\underbrace{\max J(\pi)}_{\text{第 1 步：起点}}
\;\to\;
\underbrace{\text{为什么不能直接最大化 } Q(s,a)}_{\text{第 2 步}}
\;\to\;
\underbrace{\nabla_\theta J(\theta) = \mathbb{E}[\nabla\log\pi_\theta \cdot Q]}_{\text{第 3 步：策略梯度定理}}
\;\to\;
\underbrace{\text{换成 Advantage}}_{\text{第 4 步}}
$$

$$
\to\;
\underbrace{\text{步子太大会崩}}_{\text{第 5 步}}
\;\to\;
\underbrace{\text{TRPO：重要性采样 + KL 约束}}_{\text{第 6 步}}
\;\to\;
\underbrace{\text{PPO：clip 代替 KL}}_{\text{第 7 步}}
\;\to\;
\underbrace{V_\phi \text{ 从哪来}}_{\text{第 8 步}}
\;\to\;
\underbrace{\text{GAE 估计 } \hat{A}_t}_{\text{第 9 步}}
\;\to\;
\underbrace{\text{完整算法 + 代码}}_{\text{第 10 步}}
$$

---

## 贯穿全文的例子

为了让每一步推导都有具体的数字可以代入，我们固定两个例子，全文反复使用：

**例子 A（离散动作，用于策略梯度的字母级演算）**：一个只有两个动作的极简任务。机器人在状态 $s$ 只能选"左"或"右"，选对方向就能更快到达目标。

**例子 B（连续动作，用于 Advantage / Critic / GAE 的数值演算）**：一个机械臂在 2D 桌面上抓取方块。状态 $s$ = (手的位置, 方块位置)，动作 $a$ = (手的位移 $\Delta x, \Delta y$)，每步奖励 $r = -0.01$（时间惩罚），抓到方块给 $+10$，折扣因子 $\gamma = 0.99$。

---

## 第一步：起点——我们到底想要什么

### 1.1 用自然语言说清楚目标

在写任何数学符号之前，先把目标用一句话说清楚：**我们想要一个"决策规则"，让机器人（或任何 agent）从任意起始状态出发，按这个规则一直做决策，能拿到的"未来总奖励"尽可能多。**

这个"决策规则"就是**策略** $\pi$。策略可以是确定性的（给定状态直接输出一个动作），也可以是随机的（给定状态输出一个动作的概率分布，再从中采样）。在深度 RL 里，策略几乎总是用一个神经网络实现，网络的参数记为 $\theta$，所以我们把策略写成 $\pi_\theta(a|s)$——"参数为 $\theta$ 的网络，输入状态 $s$，输出动作 $a$ 的概率"。

agent 和环境交互一整轮（一个 episode），会产生一条**轨迹**：

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)
$$

这条轨迹的**总回报**定义为：

$$
R(\tau) = \sum_{t=0}^{T} \gamma^t r_t = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots
$$

**为什么要有 $\gamma$（折扣因子）**：如果没有折扣（$\gamma=1$），无限长的轨迹的总奖励可能是无穷大，数学上没法比较"哪个策略更好"。加上 $\gamma \in (0,1)$ 之后，越往后的奖励权重指数衰减，总和永远收敛。直觉上也合理：明天的一块钱不如今天的一块钱值钱，越不确定的未来收益，就该打更多折扣。

### 1.2 目标函数 $J(\theta)$——把"策略好不好"变成一个数

**为什么需要这个公式**：一个策略是随机的，同样的策略跑十次可能拿到十个不同的总奖励（因为环境和动作采样都有随机性）。我们没法直接说"这个策略能拿多少分"，只能说"平均能拿多少分"。$J(\theta)$ 就是把这个"平均分"定义清楚，让它变成一个关于参数 $\theta$ 的确定性函数——只有变成了确定性函数，才能对 $\theta$ 求梯度、做优化。

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

> **一句话直觉**：用策略 $\pi_\theta$ 跑无数遍，把每一遍拿到的折扣总奖励平均起来——这个平均数就是 $J(\theta)$，它是我们唯一关心的"策略打分"。

**逐项拆解**（把每个字母都过一遍）：

| 符号 | 数学含义 | 直觉/物理含义 | 在例子 A 中对应什么 |
|------|---------|--------------|-------------------|
| $\theta$ | 策略网络的参数（一堆实数） | "大脑的设置" | 网络权重矩阵里的所有数字 |
| $\pi_\theta$ | 参数为 $\theta$ 的策略，一个条件概率分布 $\pi_\theta(a|s)$ | "在状态 $s$ 下，各个动作的概率是多少" | $\pi_\theta(\text{左}|s)=0.3,\ \pi_\theta(\text{右}|s)=0.7$ |
| $\tau$ | 一条完整轨迹 $(s_0,a_0,r_0,\ldots,s_T,a_T,r_T)$ | "一局游戏的完整录像" | 从出生状态到到达目标的完整过程 |
| $\tau \sim \pi_\theta$ | 轨迹是由 $\pi_\theta$（和环境的转移概率）共同生成的随机变量 | "用这套决策规则玩了一局" | 每局录像都不同，因为策略是随机的 |
| $r_t$ | 第 $t$ 步获得的即时奖励（标量） | "这一步做得好不好的即时反馈" | 每走一步 $r=-1$（时间惩罚），到达目标 $r=+10$ |
| $\gamma^t$ | 折扣因子的 $t$ 次方 | "第 $t$ 步的奖励打几折" | $\gamma=0.9$ 时，第 5 步的奖励打 $0.9^5\approx0.59$ 折 |
| $\sum_{t=0}^T$ | 对整条轨迹所有时间步求和 | "把整局游戏的（打折后的）奖励加总" | 把每一步的折扣奖励累加成一个数 |
| $\mathbb{E}_{\tau\sim\pi_\theta}[\cdot]$ | 对轨迹分布求期望 | "跑无数局，取平均" | 实践中用采样几十/几百条轨迹取样本均值近似 |
| $J(\theta)$ | 最终的标量分数 | "这个策略能打几分" | 分数越高，策略越好 |

**具体数值例子**：例子 A 中，用策略 $\pi_\theta$ 跑了 3 局，$\gamma=0.9$：

- 局 1：$r_0=1, r_1=2, r_2=3$ → $R_1 = 1 + 0.9\times2 + 0.81\times3 = 5.23$
- 局 2：$r_0=0, r_1=1, r_2=1$ → $R_2 = 0 + 0.9 + 0.81 = 1.71$
- 局 3：$r_0=2, r_1=0, r_2=2$ → $R_3 = 2 + 0 + 1.62 = 3.62$

$$
J(\theta) \approx \frac{1}{3}(5.23+1.71+3.62) = 3.52
$$

**为什么是这个形式（而不是别的定义"好坏"的方式）**：你可能会问，为什么不直接用"平均单步奖励"或者"最终是否成功"来定义好坏？因为累积折扣奖励能统一处理"稀疏奖励"（只在结束时给一个 0/1）和"密集奖励"（每步都给），并且天然鼓励"尽快拿到奖励"（因为越晚的奖励打折越多），这对大多数任务都是合理的偏好。

**强化学习的整个目标，从头到尾就是一句话**：

$$
\theta^* = \arg\max_\theta J(\theta)
$$

找到一组参数 $\theta^*$，让上面这个"平均分"最大。接下来所有的推导，都是在回答一个问题：**怎么找到能让 $J(\theta)$ 变大的更新方向？**

---

## 第二步：为什么不能直接"最大化 $Q(s,a)$"

你在很多科普材料里会看到一句简化的说法："RL 就是最大化 $Q(s,a)$"。这句话不算错，但省略了一个关键的技术难题，理解这个难题正是引出"策略梯度"方法的原因。

### 2.1 "最大化 $Q$" 到底是什么意思

先回顾 [Q 函数](/前置知识/000o_前置知识_Q函数与Value函数) 的定义：$Q^\pi(s,a)$ 是"在状态 $s$ 做动作 $a$，之后按策略 $\pi$ 走下去，未来能拿多少期望折扣奖励"。

可以证明（这是一个数学定理，不是巧合）：**$J(\theta)$ 可以完全用 $Q$ 函数重新表达**：

$$
J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}}\Big[\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\big[Q^{\pi_\theta}(s,a)\big]\Big]
$$

其中 $d^{\pi_\theta}(s)$ 是策略 $\pi_\theta$ 下状态的"访问频率分布"（哪些状态被经常访问到）。这个式子说明：**"最大化累积奖励"和"让策略在每个状态都尽量选 $Q$ 值高的动作"，本质上是同一件事**——这就是"最大化 reward = 最大化 $Q(s,a)$"这句话的精确含义。

### 2.2 那为什么不直接 $a^* = \arg\max_a Q(s,a)$ 完事？

如果我们已经知道了精确的 $Q^*(s,a)$（最优 Q 函数），那确实可以直接用

$$
\pi^*(a|s) = \begin{cases} 1 & a = \arg\max_{a'} Q^*(s,a') \\ 0 & \text{otherwise}\end{cases}
$$

这正是 [Q 函数前置知识](/前置知识/000o_前置知识_Q函数与Value函数) 里 Q-learning / DQN 那一路方法的思路：学一个 $Q$ 网络，动作直接 $\arg\max$ 出来。

但这条路在很多场景下走不通：

1. **连续动作空间下 $\arg\max_a Q(s,a)$ 本身就是一个难解的优化问题**。如果 $a \in \mathbb{R}^7$（比如 7 自由度机械臂的力矩），要在每个状态、每一步决策时都对一个 7 维连续函数做全局最大化，代价极高，而且这个 $\arg\max$ 本身还要对 $\theta$ 可导才能训练——一般做不到解析求解。
2. **我们通常不直接学 $Q^*$，而是学一个参数化的策略网络 $\pi_\theta$**（原因见下面）。既然策略本身就是一个待优化的参数化函数，更自然的做法是：直接对 $\theta$ 求梯度，让 $J(\theta)$（等价于让策略选择的动作的 $Q$ 值）越变越大，而不是每一步都单独解一个 $\arg\max$。

**这就是"策略梯度方法"（Policy Gradient）和"值函数方法"（Value-based，如 Q-learning）的根本分岔口**：

| 路线 | 直接学什么 | 动作怎么来 | 适用场景 |
|------|-----------|-----------|---------|
| Value-based（Q-learning, DQN） | $Q_\phi(s,a)$ | $a=\arg\max_a Q_\phi(s,a)$ | 离散、维度不太高的动作空间 |
| **Policy-based（策略梯度, PPO）** | $\pi_\theta(a|s)$ 本身 | 直接从 $\pi_\theta$ 采样 | 连续动作、高维动作、需要随机策略探索的场景 |

PPO 属于第二条路线：**我们不绕道去学 $Q$ 再 $\arg\max$，而是直接对策略参数 $\theta$ 求 $J(\theta)$ 的梯度，用梯度上升更新 $\theta$**。这就引出了下一步的核心问题：$J(\theta)$ 里套了一层对轨迹分布的期望，而这个分布依赖于环境的转移概率（我们通常不知道环境内部机制），$\nabla_\theta J(\theta)$ 到底怎么算？

---

## 第三步：策略梯度定理——怎么对 $\theta$ 求梯度

### 3.1 表面上的困难

$$
\nabla_\theta J(\theta) = \nabla_\theta \, \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

期望的定义是对概率分布积分：$\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] = \int p_\theta(\tau) R(\tau)\, d\tau$，其中 $p_\theta(\tau)$ 是"用策略 $\pi_\theta$ 产生轨迹 $\tau$ 的概率"。问题是：

$$
p_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \underbrace{\pi_\theta(a_t|s_t)}_{\text{策略决定}} \cdot \underbrace{P(s_{t+1}|s_t,a_t)}_{\text{环境转移概率，通常未知}}
$$

$p_\theta(\tau)$ 里混进了环境的转移概率 $P(s_{t+1}|s_t,a_t)$——这是物理世界（或仿真器）决定的，我们**不知道它的解析表达式**，当然也没法对它求 $\nabla_\theta$。看起来 $\nabla_\theta J(\theta)$ 没法直接算。

### 3.2 破局的关键：log-derivative trick（对数导数技巧）

对任意关于 $\theta$ 可导的概率分布 $p_\theta(x)$，有一个纯代数恒等式：

$$
\nabla_\theta \log p_\theta(x) = \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} \quad\Longrightarrow\quad \nabla_\theta p_\theta(x) = p_\theta(x)\cdot \nabla_\theta \log p_\theta(x)
$$

**为什么需要这个公式**：它是整条策略梯度推导唯一的数学"魔法"。它的作用是把"梯度作用在概率分布上"（没法变成期望）转换成"梯度作用在 $\log$ 概率上，再乘回原来的概率"（正好能变成期望，从而能用采样近似）。

> **一句话直觉**：这只是链式法则的逆用——$\log$ 函数求导是 $1/x$，两边乘回 $p_\theta(x)$ 正好把 $\nabla p_\theta$ "还原"出来，代价是多了一个 $\log$。

**逐项拆解**：

| 符号 | 含义 |
|------|------|
| $p_\theta(x)$ | 任意由 $\theta$ 参数化的概率分布（本文中就是轨迹分布 $p_\theta(\tau)$） |
| $\nabla_\theta p_\theta(x)$ | 分布本身对参数的梯度（一个"密度场"如何随参数变化） |
| $\nabla_\theta \log p_\theta(x)$ | 对数密度的梯度，也叫 **score function**（打分函数） |

**证明**（只需一步链式法则）：设 $f(\theta) = \log p_\theta(x)$，则 $\nabla_\theta f = \frac{1}{p_\theta(x)}\nabla_\theta p_\theta(x)$（把 $p_\theta(x)$ 看成中间变量对 $\log$ 求导），两边同乘 $p_\theta(x)$：$p_\theta(x)\nabla_\theta \log p_\theta(x) = \nabla_\theta p_\theta(x)$。证毕。

**为什么恰好是这个形式（为什么不能直接用 $\nabla_\theta p_\theta(x)$）**：因为 $\nabla_\theta p_\theta(x)$ 本身不是任何分布下的期望，没法用"采样几个 $x$ 取平均"来估计。而 $p_\theta(x)\nabla_\theta\log p_\theta(x)$ 里恰好带着一个 $p_\theta(x)$ 因子，可以把它"还给"期望符号——这是下一步的关键。

### 3.3 用 log-derivative trick 推导策略梯度

$$
\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau)\, d\tau
= \int \nabla_\theta p_\theta(\tau) \cdot R(\tau)\, d\tau
$$

（梯度和积分可以交换次序，这是一个技术性假设，实践中总是成立）

$$
= \int p_\theta(\tau)\, \nabla_\theta\log p_\theta(\tau) \cdot R(\tau)\, d\tau
\quad\text{（代入 3.2 的恒等式）}
$$

$$
= \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta \log p_\theta(\tau)\cdot R(\tau)\big]
\quad\text{（积分又变回了期望！）}
$$

**这一步是整个推导的转折点**：原来 $\nabla_\theta J(\theta)$ 是"没法采样估计的东西"（因为梯度在积分外面），现在变成了"$\nabla_\theta\log p_\theta(\tau)\cdot R(\tau)$ 这个量在 $\pi_\theta$ 下的期望"——**这可以用蒙特卡罗采样估计**：跑几条轨迹，把 $\nabla_\theta\log p_\theta(\tau)\cdot R(\tau)$ 算出来取平均即可。

### 3.4 把 $\log p_\theta(\tau)$ 展开，甩掉环境转移概率

现在处理 $\log p_\theta(\tau)$：

$$
\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T}\Big[\log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t,a_t)\Big]
$$

对 $\theta$ 求梯度：

$$
\nabla_\theta \log p_\theta(\tau) = \underbrace{\nabla_\theta \log p(s_0)}_{=0} + \sum_{t=0}^{T}\Big[\nabla_\theta\log \pi_\theta(a_t|s_t) + \underbrace{\nabla_\theta\log P(s_{t+1}|s_t,a_t)}_{=0}\Big]
$$

初始状态分布 $p(s_0)$ 和环境转移概率 $P(s_{t+1}|s_t,a_t)$ 都**完全不依赖策略参数 $\theta$**（环境物理规律不会因为你换了个策略网络就变了），所以它们对 $\theta$ 的梯度都是 0。**这正是我们摆脱"未知环境模型"这个难题的地方**——剩下的只有策略自己的项：

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

### 3.5 策略梯度定理——最终形式

把 3.4 的结果代回 3.3：

**为什么需要这个公式**：我们要用梯度上升更新 $\theta$ 来最大化 $J(\theta)$，但 $J(\theta)$ 表达式里含有未知的环境模型。这个公式给出了一个只依赖"策略自身的 log 概率"和"轨迹的奖励"就能计算的梯度估计，彻底绕开了环境模型。

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot R(\tau)\right]
$$

> **一句话直觉**：对轨迹里出现过的每一个 (状态,动作) 对，如果整条轨迹最终拿到的总奖励 $R(\tau)$ 高，就把这个动作在这个状态下的概率往上调；如果 $R(\tau)$ 低，就往下调。调的"方向"由 $\nabla_\theta\log\pi_\theta(a_t|s_t)$ 决定,调的"力度"由 $R(\tau)$ 决定。

**逐项拆解**：

| 符号 | 数学含义 | 直觉 |
|------|---------|------|
| $\nabla_\theta J(\theta)$ | 目标函数对参数的梯度向量 | "往哪个方向调参数能让平均分变高" |
| $\mathbb{E}_{\tau\sim\pi_\theta}[\cdot]$ | 对当前策略采样出的轨迹分布求期望 | 实践中：采样 $N$ 条轨迹，取样本平均近似 |
| $\sum_{t=0}^T$ | 对轨迹里所有的时间步求和 | 一条轨迹里的每一步都提供一份梯度贡献 |
| $\pi_\theta(a_t\|s_t)$ | 策略在 $t$ 时刻、状态 $s_t$ 下选择动作 $a_t$ 的概率 | 已经实际发生的那次决策的概率 |
| $\log\pi_\theta(a_t\|s_t)$ | 该概率的对数 | 概率越低，$\log$ 越负（越"惊讶"） |
| $\nabla_\theta\log\pi_\theta(a_t\|s_t)$ | 这个 log 概率对参数的梯度，叫 **score function** | "怎么调 $\theta$ 能让这个具体动作的概率增大" |
| $R(\tau)$ | 整条轨迹的折扣总奖励（1.1 节定义） | 给这个动作打的"总分" |

**为什么是 $\log\pi$ 而不是 $\pi$ 本身**：这不是随便选的，是 3.2 节 log-derivative trick 的直接产物——正是因为用了 $\nabla p = p\nabla\log p$ 这个恒等式，才能把 $\nabla_\theta$ 从积分外面搬到期望里面，变成可采样估计的量。如果直接用 $\nabla_\theta\pi_\theta$（没有 $\log$），积分展开后不会自动出现 $\pi_\theta(\tau)$ 这个因子去"配对"期望符号，就没法用蒙特卡罗采样。

**具体数值例子（例子 A）**：在状态 $s$，策略网络输出 $\pi_\theta(\text{左}|s)=0.3$，$\pi_\theta(\text{右}|s)=0.7$。假设某条轨迹采样到了动作"左"，且这条轨迹最终 $R(\tau)=10$（好结果）：

- $\log\pi_\theta(\text{左}|s) = \log(0.3) = -1.20$
- $\nabla_\theta\log\pi_\theta(\text{左}|s)$ 是一个具体的向量，指向"能让 $\pi_\theta(\text{左}|s)$ 变大"的参数调整方向
- 这一步对总梯度的贡献 $= \nabla_\theta\log\pi_\theta(\text{左}|s) \times 10$：**正数乘以"增大左概率"的方向** → 净效果是让"左"的概率进一步增大

如果同样选了"左"，但这条轨迹结果很差，$R(\tau) = -5$：

- 贡献 $= \nabla_\theta\log\pi_\theta(\text{左}|s) \times (-5)$：**负数乘以"增大左概率"的方向 = 反向** → 净效果是让"左"的概率减小

**为什么这个形式合理**：同一个动作，因为轨迹后续表现好坏不同，梯度方向完全相反——这正是我们想要的：不是无条件地强化"左"这个动作，而是"根据这个动作带来的实际后果决定强化还是抑制"。

### 3.6 一个关键的精细化：只用未来奖励（reward-to-go）

3.5 节的公式里用的是**整条轨迹**的总奖励 $R(\tau)$ 来评价第 $t$ 步的动作 $a_t$。但仔细想想：$t$ 时刻**之前**发生的奖励 $r_0,\ldots,r_{t-1}$ 根本不受 $a_t$ 影响（它们已经发生了），把它们也算进 $a_t$ 的"打分"里毫无道理，只会增加不必要的方差。

可以证明（利用"未来动作的选择不影响过去奖励的期望"这一事实），把 $R(\tau)$ 换成只算从 $t$ 时刻往后的**reward-to-go**：

$$
\hat{R}_t = \sum_{t'=t}^{T} \gamma^{t'-t}\, r_{t'}
$$

不会改变梯度的期望值（无偏），但去掉了无关的历史奖励项，方差更小：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta\log\pi_\theta(a_t|s_t)\cdot \hat{R}_t\right]
$$

注意：$\hat{R}_t$ 正是 [Q 函数前置知识](/前置知识/000o_前置知识_Q函数与Value函数) 里 $Q^\pi(s_t,a_t)$ 的**蒙特卡罗（无偏）估计量**——因为 $Q^\pi(s_t,a_t) = \mathbb{E}[\hat{R}_t \mid s_t,a_t]$，用一条采样轨迹的 $\hat{R}_t$ 去近似它的期望值 $Q^\pi$，正是"用样本代替期望"。**这就是"最大化 reward = 最大化 $Q(s,a)$"这句话在策略梯度公式里的具体落地**：策略梯度就是用 $Q$（或它的估计 $\hat R_t$）作为"打分"，去调整动作概率。

### 3.7 REINFORCE 算法——第一个可以写出来的完整算法

$$
\textbf{REINFORCE：}
$$

1. 用当前策略 $\pi_\theta$ 跑 $N$ 条轨迹 $\tau_1,\ldots,\tau_N$
2. 对每条轨迹的每个 $(s_t,a_t)$，计算 $\hat{R}_t = \sum_{t'\geq t}\gamma^{t'-t}r_{t'}$
3. 用样本平均近似梯度：$\displaystyle \nabla_\theta J \approx \frac{1}{N}\sum_{i=1}^N\sum_t \nabla_\theta\log\pi_\theta(a_t^i|s_t^i)\cdot\hat{R}_t^i$
4. $\theta \leftarrow \theta + \alpha\nabla_\theta J$（梯度上升）

### 3.8 REINFORCE 的致命问题：方差爆炸

$\hat{R}_t$ 是一个随机变量——它取决于从 $t$ 时刻开始的**所有**后续随机转移和随机动作采样。哪怕策略完全没变，仅仅因为环境和采样的随机性，同一个 $(s_t,a_t)$ 在不同轨迹里得到的 $\hat{R}_t$ 可能天差地别。

**直觉后果**：梯度估计的"信噪比"很低——你需要采样成百上千条轨迹才能让样本均值稳定下来。这在实践中太慢，几乎没有人直接用纯 REINFORCE。

### 3.9 引入 Baseline：降方差的第一次修正

**核心事实（无偏性）**：对策略梯度公式加上任何一个**不依赖动作 $a$**（只能依赖状态 $s$）的函数 $b(s_t)$，梯度的期望值不变：

$$
\mathbb{E}_{a_t\sim\pi_\theta(\cdot|s_t)}\big[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot b(s_t)\big] = 0
$$

**证明**：把 $b(s_t)$ 提出期望（它不依赖 $a_t$），剩下 $\mathbb{E}_{a_t}[\nabla_\theta\log\pi_\theta(a_t|s_t)]$：

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}\big[\nabla_\theta\log\pi_\theta(a|s)\big] = \int \pi_\theta(a|s)\cdot\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}\,da = \int \nabla_\theta\pi_\theta(a|s)\, da = \nabla_\theta\int\pi_\theta(a|s)\,da = \nabla_\theta(1) = 0
$$

最后一步用了"概率分布对所有动作积分永远等于 1"这个事实，1 对 $\theta$ 求导自然是 0。**这就是为什么减去 baseline 不会引入偏差**——多减掉的这一项，期望本身就是 0，减了等于没减（对期望而言），但对**每一条具体轨迹的方差**却有实质性的影响。

于是把 3.6 节的公式改成：

$$
\nabla_\theta J(\theta) = \mathbb{E}\Big[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\big(\hat{R}_t - b(s_t)\big)\Big]
$$

**为什么减 baseline 能降方差（直觉）**：如果所有轨迹的 $\hat{R}_t$ 都是正数（比如奖励设计成从不为负），那么策略梯度会无差别地"增大所有出现过的动作的概率"，区别只是增大的多少——这浪费了大量"哪个动作更差"的信息。减去一个合理的 baseline 后，比平均水平好的动作变正、比平均水平差的动作变负,信号更干净。

**最优的 baseline 是什么**：理论上可以证明，让方差最小的 baseline 非常接近 $V^\pi(s_t)$（[状态价值函数](/前置知识/000o_前置知识_Q函数与Value函数)——"在 $s_t$ 按策略平均能拿多少"）。用 $V^\pi(s_t)$ 做 baseline 后：

$$
\hat{R}_t - V^\pi(s_t) \approx Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t,a_t)
$$

这正是 [Advantage 函数](/前置知识/000o_前置知识_Q函数与Value函数)！于是策略梯度的**现代标准形式**是：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{(s_t,a_t)\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot A^\pi(s_t,a_t)\big]
$$

> **一句话直觉**：$Q$ 告诉你"这个动作绝对值多少分"，但绝对分数会掺杂"这个状态本身好不好"的信息（比如在必胜局面里随便动都是高分）。减掉 $V(s)$（"这个状态平均值多少分"）之后，剩下的 $A(s,a)$ 才是"这个动作比这个状态下的平均水平好多少"的**纯净信号**——这才是真正该驱动"增大/减小概率"的东西。

到这里，我们已经完整回答了第二步末尾提出的问题：**$J(\theta)$ 的梯度可以只用策略自身的 log 概率和 Advantage 估计算出来，不需要知道环境模型**。但 REINFORCE / Actor-Critic 这类"算完梯度直接更新"的方法还有一个大问题没解决——这就是第五步要讲的"步子太大会崩"。在那之前,我们先说清楚 $Q$、$V$、$A$ 具体是怎么落地估计的(第八、九步),但逻辑上先把"为什么要限制步长"讲透更重要,所以接下来先跳到第五步。

---

## 第五步：步子太大会崩——为什么不能直接梯度上升

### 5.1 问题的本质

有了 3.9 节的梯度公式，最朴素的做法是：

$$
\theta_{\text{new}} = \theta_{\text{old}} + \alpha\cdot\nabla_\theta J(\theta)
$$

这在**监督学习**里天经地义——学习率小一点，多迭代几步，总能收敛。但在 RL 里，这样做有一个监督学习没有的致命问题：

**策略参数变了，数据分布也变了。** 在监督学习中，训练数据是固定的（不随参数变化）。但在 RL 中，$\theta$ 变了 → 策略 $\pi_\theta$ 变了 → 下一轮采集到的状态-动作分布 $d^{\pi_\theta}$ 也变了。如果 $\theta$ 一步跳得太远，新策略可能进入一个从未见过、表现极差的区域，而且**没有回头路**——因为新策略采出来的都是坏数据，用坏数据算出的梯度可能继续把策略推向更差的方向。

**类比**：监督学习像是在固定地形上找山顶，山不会因为你移动而改变。RL 更像是蒙眼在一座会随你脚步移动而"重新长出来"的山上走——你每走一步，脚下的地形都可能剧烈变化，一步踏空可能直接摔进悬崖，且再也回不到原来能看清地形的地方。

### 5.2 神经网络策略让问题更严重

**参数空间的微小变化，可能对应动作空间的巨大变化。** 神经网络是高度非线性的：$\theta$ 在参数空间里移动一小步（比如 $\|\Delta\theta\|=0.01$），在某些区域可能几乎不改变输出，在另一些区域却可能让输出剧烈跳变。梯度上升的步长 $\alpha$ 是在"参数空间"里定义的，但我们真正关心的是"策略行为空间"里的变化幅度——这两者没有直接对应关系，用固定学习率无法保证"策略行为"的变化是可控的。

**我们真正想要的约束，是"新旧策略的行为差异不要太大"，而不是"参数数值差异不要太大"。** 这就引出了下一步：怎么用数学工具衡量"两个策略的行为差异"？答案就是 [KL 散度](/前置知识/000j_前置知识_KL散度与策略约束)。

---

## 第六步：TRPO——用重要性采样 + KL 约束限制更新幅度

### 6.1 目标的重新表述：策略改进量

TRPO 的出发点是一个理论结果：给定旧策略 $\pi_{\text{old}}$ 和新策略 $\pi_{\text{new}}$，两者的性能差可以精确写成：

$$
J(\pi_{\text{new}}) - J(\pi_{\text{old}}) = \mathbb{E}_{s\sim d^{\pi_{\text{new}}},\, a\sim\pi_{\text{new}}}\big[A^{\pi_{\text{old}}}(s,a)\big]
$$

**为什么需要这个公式**：我们想知道"如果换成新策略，性能到底会变好还是变差、变多少"。这个恒等式告诉我们：性能提升量，恰好等于"用新策略的状态分布、新策略的动作，去评估旧策略下的 advantage"的期望。

**逐项拆解**：

| 符号 | 含义 |
|------|------|
| $J(\pi_{\text{new}}) - J(\pi_{\text{old}})$ | 换策略后，性能提升了多少（我们想让它 $>0$ 且尽量大） |
| $d^{\pi_{\text{new}}}$ | 新策略下状态被访问的频率分布 | 
| $A^{\pi_{\text{old}}}(s,a)$ | 用**旧策略**的价值函数算出的 advantage |

**问题**：右边的期望是在 $d^{\pi_{\text{new}}}$（新策略的状态分布）下求的——但我们还没有新策略，没法从它那里采样状态！这是一个"鸡生蛋蛋生鸡"的循环：要评估新策略的好坏，需要新策略产生的数据；但新策略是我们正要去求的东西。

### 6.2 用旧策略的数据近似——重要性采样登场

TRPO 的近似方案：**用旧策略的状态分布 $d^{\pi_{\text{old}}}$ 代替未知的 $d^{\pi_{\text{new}}}$**（当两个策略差别不大时，这个近似误差是可控的二阶小量，这也是为什么 TRPO 要求策略变化幅度小的理论根源）：

$$
J(\pi_{\text{new}}) - J(\pi_{\text{old}}) \approx \mathbb{E}_{s\sim d^{\pi_{\text{old}}},\, a\sim\pi_{\text{new}}(\cdot|s)}\big[A^{\pi_{\text{old}}}(s,a)\big]
$$

但这里 $a\sim\pi_{\text{new}}$ 仍然要用新策略采样——我们同样没有新策略的采样器（训练过程中我们只用旧策略跑环境收集数据）。这里要用统计学里的**重要性采样（Importance Sampling）**恒等式：

$$
\mathbb{E}_{a\sim\pi_{\text{new}}}[f(a)] = \int \pi_{\text{new}}(a|s)\,f(a)\,da = \int \pi_{\text{old}}(a|s)\cdot\frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}\cdot f(a)\,da = \mathbb{E}_{a\sim\pi_{\text{old}}}\left[\frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}\cdot f(a)\right]
$$

**为什么需要这个恒等式**：它能把"对新分布求期望"精确转换成"对旧分布求期望，但要乘一个校正权重"。这个校正权重就是概率比。这样一来，我们就可以**只用旧策略采集的数据**（这是我们唯一有的数据），去估计"如果换成新策略"的期望——不需要真的用新策略去跟环境交互。

> **一句话直觉**：旧策略采样出的动作 $a$，如果新策略也很喜欢选它（$\pi_{\text{new}}(a|s)$ 大），这个样本的权重就调高；如果新策略不喜欢选它，权重就调低——用权重"校正"旧样本，让它看起来像是从新分布采出来的。

**逐项拆解**：

| 符号 | 含义 | 直觉 |
|------|------|------|
| $f(a)$ | 任意关于 $a$ 的函数（这里是 $A^{\pi_{\text{old}}}(s,a)$） | 我们想估计它在新策略下的期望 |
| $\frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}$ | 重要性权重 | "这个样本在新分布下应该被赋予多大权重" |

**数值例子**：假设某个动作 $a_0$ 在 $\pi_{\text{old}}$ 下概率是 $0.2$，$A^{\pi_{\text{old}}}(s,a_0)=5$。

- 若新策略也认为这个动作不错，$\pi_{\text{new}}(a_0|s)=0.3$：权重 $=0.3/0.2=1.5$，贡献 $=1.5\times5=7.5$（放大了这个好动作的影响）
- 若新策略觉得这个动作很差，$\pi_{\text{new}}(a_0|s)=0.05$：权重 $=0.05/0.2=0.25$，贡献 $=0.25\times5=1.25$（削弱了影响，因为新策略本来就不太会选它）

代入后，TRPO/PPO 共同的**代理目标（surrogate objective）**就出现了：

$$
L^{\text{IS}}(\theta) = \mathbb{E}_{s,a\sim\pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\cdot A^{\pi_{\text{old}}}(s,a)\right]
$$

**这就是 3.5 节策略梯度公式的"重要性采样版本"**：如果 $\pi_\theta = \pi_{\text{old}}$，对 $\theta$ 求梯度，会发现 $\nabla_\theta L^{\text{IS}}(\theta)\big|_{\theta=\theta_{\text{old}}} = \mathbb{E}[\nabla_\theta\log\pi_\theta(a|s)\cdot A(s,a)]$，正好等于策略梯度定理的结果——**这也说明了为什么这个代理目标是合理的**：它在 $\theta=\theta_{\text{old}}$ 附近的梯度，和真正的策略梯度完全一致，只是允许我们**用同一批旧数据反复计算、多走几步**（而不是每算一次梯度就丢弃数据重新采样），因为重要性采样权重帮我们做了"分布校正"。

### 6.3 但重要性采样权重会失控——必须加约束

重要性采样的校正是**精确的**（数学恒等式），但它有一个隐藏陷阱：**当 $\pi_{\text{new}}$ 和 $\pi_{\text{old}}$ 差异变大时，权重 $\pi_{\text{new}}/\pi_{\text{old}}$ 的方差会急剧增大**（少数样本的权重可能变得极端大或极端小，导致估计不稳定,甚至上面 6.2 节"$d^{\pi_{\text{new}}}\approx d^{\pi_{\text{old}}}$"这个近似本身也会失效）。所以我们必须**限制新旧策略不能差太远**——这正是第五步提出的需求。TRPO 用 [KL 散度](/前置知识/000j_前置知识_KL散度与策略约束) 显式约束：

$$
\max_\theta\; \mathbb{E}_{s,a\sim\pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}A^{\pi_{\text{old}}}(s,a)\right]
\qquad \text{s.t.}\qquad
\mathbb{E}_{s\sim d^{\pi_{\text{old}}}}\big[D_{\text{KL}}(\pi_{\text{old}}(\cdot|s)\,\|\,\pi_\theta(\cdot|s))\big] \le \delta
$$

**逐项拆解**：

| 符号 | 含义 |
|------|------|
| 目标函数 | 6.2 节的重要性采样代理目标，越大越好 |
| $D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta)$ | 新旧策略在状态 $s$ 下的 [KL 散度](/前置知识/000j_前置知识_KL散度与策略约束) |
| $\delta$ | 允许的最大 KL 散度（一个很小的正数，比如 0.01） |

**为什么用 KL 而不是参数空间的欧氏距离**：KL 散度衡量的是**两个概率分布本身**的差异（"行为"的差异），而不是参数数值的差异。这正好对应第 5.2 节的诉求——我们要约束的是"策略行为"不要变化太多，而不是"参数数值"不要变化太多。

### 6.4 TRPO 的实现困境

上面这个约束优化问题理论上很漂亮，但实际求解很麻烦：

- 需要用**共轭梯度法**近似求解带约束的二次规划
- 需要计算 KL 散度的二阶导数（Fisher 信息矩阵）和它与向量的乘积
- 每一步更新都要做一次线搜索（line search）确认约束确实满足
- 实现复杂、调参困难，很难和大规模并行训练（比如几千个仿真环境同时跑）配合

这就是 PPO 存在的理由——**保留 TRPO 的核心思想（限制策略变化幅度），但用一个极其简单的技巧代替复杂的约束优化**。

---

## 第七步：PPO——用 Clip 代替 KL 约束

### 7.1 核心想法

PPO 的洞察：**我们不需要精确求解带约束的优化问题，只需要在目标函数里"手动"给重要性采样权重设一个安全范围，一旦权重想跑出这个范围，就不再给它更多梯度信号。**

先给 6.2 节的重要性采样权重起一个标准名字，这是你在任何 PPO 代码里都会看到的变量名：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

含义完全继承自 6.2 节：$r_t=1$ 表示新旧策略对这个 $(s_t,a_t)$ 给出相同概率；$r_t=1.5$ 表示新策略把这个动作的概率调高了 50%；$r_t=0.5$ 表示调低了 50%。

### 7.2 PPO-Clip 目标函数

**为什么需要这个公式**：直接用 $r_t(\theta)\hat{A}_t$ 做优化（6.2 节的代理目标），当 $r_t$ 想变得很大（或很小）时，梯度会一直把它往更极端的方向推，没有任何阻力——这正是我们想避免的"步子太大"。我们需要给这个目标函数装一个"安全带"：允许 $r_t$ 有限度地偏离 1，但一旦偏离太多就不再提供额外奖励。

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\big)\Big]
$$

> **一句话直觉**：好动作的概率允许增大，但最多增大到原来的 $(1+\epsilon)$ 倍；差动作的概率允许减小，但最多减小到原来的 $(1-\epsilon)$ 倍——超过这个范围之后，目标函数就不再给"继续推得更远"提供任何梯度奖励。

**逐项拆解**（这是本文最重要的一个公式，逐字拆完）：

| 符号 | 数学含义 | 直觉 |
|------|---------|------|
| $r_t(\theta) = \pi_\theta(a_t\|s_t)/\pi_{\theta_{\text{old}}}(a_t\|s_t)$ | 新旧策略在这个具体样本上的概率比 | 见 7.1 |
| $\hat{A}_t$ | 这个 $(s_t,a_t)$ 的 advantage 估计（第九步 GAE 给出） | 正=这个动作比平均好，负=比平均差 |
| $r_t(\theta)\hat{A}_t$ | 未裁剪的代理目标（就是 6.2 节的重要性采样目标） | "普通"策略梯度的代理形式，没有安全限制 |
| $\text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)$ | 把 $r_t$ 强行截断到区间 $[1{-}\epsilon,1{+}\epsilon]$ 内 | $r_t$ 超出范围就被"拉回"边界 |
| $\text{clip}(\cdots)\hat{A}_t$ | 裁剪后的代理目标 | "安全版本"——即使真实 $r_t$ 跑远了，这一项也只按边界值计算 |
| $\min(\cdot,\cdot)$ | 取两者中较小的一个 | 悲观原则：宁可低估收益，也不给"继续冒进"的动力 |
| $\mathbb{E}_t[\cdot]$ | 对 mini-batch 内所有采样到的 $(s_t,a_t)$ 求平均 | 实践中就是 batch 内取 `.mean()` |
| $\epsilon$ | 裁剪范围超参数，典型值 0.1~0.3 | 越小约束越严格，策略变化越保守 |

**为什么取 $\min$（这是唯一需要花心思理解的地方）**：分两种情况看，把每种情况都算清楚。

**情况一，$\hat{A}_t>0$（好动作）**：我们希望增大 $r_t$（提高这个好动作的概率）。

- 当 $r_t < 1+\epsilon$（还在安全区内）：$r_t\hat{A}_t < \text{clip}(r_t)\hat{A}_t$ 的边界值？此时 $\text{clip}(r_t)=r_t$（没被裁），两项相等，取哪个都一样——正常按 $r_t\hat A_t$ 提供梯度，鼓励继续增大 $r_t$。
- 当 $r_t > 1+\epsilon$（已经冲出安全区）：$\text{clip}(r_t)=1+\epsilon$ 是个常数，$\text{clip}(r_t)\hat A_t = (1+\epsilon)\hat A_t$ 是个常数（不再依赖 $\theta$，梯度为 0）；而未裁剪项 $r_t\hat A_t$ 还在随 $r_t$ 增大。因为 $\hat A_t>0$，$r_t\hat A_t > (1+\epsilon)\hat A_t$，所以 $\min$ 会**选中那个不再变化的常数项**——对 $\theta$ 求梯度时这一项梯度为 0，**优化器不再因为继续增大 $r_t$ 而得到任何奖励**。

**情况二，$\hat{A}_t<0$（差动作）**：我们希望减小 $r_t$（压低这个差动作的概率）。同样的逻辑，但因为 $\hat A_t$ 是负的，不等号方向反过来：当 $r_t<1-\epsilon$（已经把概率压得足够低）时，$\text{clip}(r_t)=1-\epsilon$ 又变成常数，$\min$ 选中它，梯度归零——**不再因为继续压低 $r_t$ 而得到额外奖励**。

**数值例子**（$\epsilon=0.2$，即裁剪区间 $[0.8,1.2]$）：

**场景 A（好动作，冲太远）**：$\hat A_t=+3$，$r_t=1.5$
- 未裁剪项：$1.5\times3=4.5$
- 裁剪项：$\text{clip}(1.5,0.8,1.2)\times3 = 1.2\times3=3.6$
- $\min(4.5,3.6)=3.6$ → 选中裁剪项（常数）→ 对 $\theta$ 的梯度为 0 → **不再鼓励继续增大概率**

**场景 B（好动作，还在安全区）**：$\hat A_t=+3$，$r_t=1.1$
- 未裁剪项：$1.1\times3=3.3$；裁剪项：$\text{clip}(1.1,0.8,1.2)\times3=1.1\times3=3.3$（没被裁）
- $\min(3.3,3.3)=3.3$ → 两项相等，正常提供梯度，**继续鼓励增大概率**

**场景 C（差动作，压太狠）**：$\hat A_t=-2$，$r_t=0.6$
- 未裁剪项：$0.6\times(-2)=-1.2$
- 裁剪项：$\text{clip}(0.6,0.8,1.2)\times(-2)=0.8\times(-2)=-1.6$
- $\min(-1.2,-1.6)=-1.6$ → 选中裁剪项（常数）→ **不再鼓励继续压低概率**

**为什么是这个形式（为什么不直接用 $\text{clip}(r_t,\ldots)\hat A_t$ 而要套一个 $\min$）**：单纯用裁剪后的 $\text{clip}(r_t)\hat A_t$ 作为目标，会在 $r_t$ 已经跑出安全区、但优化方向是"往回走"（比如好动作但 $r_t$ 太大、梯度想让它减小回到安全区）时，仍然给出 0 梯度——阻止了"纠正"的可能。而套上 $\min$ 之后，当 $r_t$ 跑出安全区且继续往外走（对目标更有利）时才被夹住变成常数；一旦 $r_t$ 往安全区方向移动（对目标不利，因为会让未裁剪项变小，$\min$ 会切换回未裁剪项），梯度又会重新出现，允许它"往回走"。**PPO 论文管这个设计原则叫"悲观下界"（pessimistic lower bound）**：$L^{\text{CLIP}}$ 始终是真实（未裁剪）目标的一个下界，取 $\min$ 保证了这一点，同时又不妨碍策略往安全方向移动。

### 7.3 为什么这样就能防止策略崩溃

$$
\text{没有 Clip（普通策略梯度/纯重要性采样）：}
$$
$$
\text{发现某个动作 advantage 很高} \to \text{梯度疯狂增大它的概率} \to \text{几步内变成几乎确定性策略} \to \text{探索消失，可能卡在局部最优，或因为过度自信在实际执行时遇到没见过的状态直接失控}
$$

$$
\text{有 Clip（}\epsilon=0.2\text{）：}
$$
$$
\text{发现某个动作 advantage 很高} \to r_t \text{最多允许涨到 1.2} \to \text{概率只涨 20\%} \to \text{基于变化后的新策略重新采数据、重新估计 advantage} \to \text{如果真的好，下一轮迭代继续涨} \to \text{缓慢、可控地收敛，不会一步冲过头}
$$

**这就是 PPO 名字里"Proximal"（临近的）的含义**：每一步更新，都让新策略"临近"旧策略，用无数次小幅、安全的更新，逐渐逼近最优策略——而不是尝试一步跳到最优。

---

## 第八步：$V(s)$ 从哪来——Critic 网络

### 8.1 为什么必须要学一个 $V(s)$

7.2 节的公式里用到了 $\hat{A}_t$，第九步会讲怎么算它，但不管用什么方法算，几乎所有实用的 advantage 估计都需要知道 $V(s)$（[状态价值函数](/前置知识/000o_前置知识_Q函数与Value函数)）。$V^\pi(s)$ 是一个期望值，理论上要"用策略 $\pi$ 跑无数遍取平均"才能精确知道，这在训练时不现实——所以我们用一个**独立的神经网络** $V_\phi(s)$（参数 $\phi$，通常和策略网络共享大部分底层特征，只是输出头不同）去**近似**它。这个网络在 Actor-Critic 架构里叫 **Critic（评论家）**，负责给状态打分；策略网络本身叫 **Actor（演员）**，负责做决策。

### 8.2 Critic 的训练目标

我们希望 $V_\phi(s_t)$ 尽量接近真实的（或者说，我们能拿到的最好估计的）回报。用回归的方式训练：

**为什么需要这个公式**：$V_\phi$ 是一个需要训练的神经网络，训练神经网络需要一个可以求梯度的损失函数。我们知道 $V^\pi(s_t)$ 的定义就是"未来折扣回报的期望"，所以最自然的监督信号就是"实际观察到的（或估计出的）回报"，用均方误差把预测值往这个目标拉。

$$
L^{V}(\phi) = \mathbb{E}_t\Big[\big(V_\phi(s_t) - \hat{R}_t\big)^2\Big]
$$

> **一句话直觉**：让 Critic 预测的"这个状态大概值多少分"，尽量接近"实际观察到这个状态之后真的拿到了多少分"。

**逐项拆解**：

| 符号 | 含义 | 直觉 |
|------|------|------|
| $V_\phi(s_t)$ | Critic 网络对状态 $s_t$ 的价值预测 | "网络觉得这个状态值多少分" |
| $\hat{R}_t$ | $t$ 时刻的目标回报（第九步会讲清楚具体怎么算，通常是 $\hat A_t + V_\phi(s_t)$，即 GAE advantage 加上旧的 value 预测） | "实际/估计出来这个状态之后真拿了多少分" |
| $(\cdot)^2$ | 平方误差 | 预测偏差越大，惩罚越重（且惩罚是偏差的平方，不是绝对值） |
| $\mathbb{E}_t[\cdot]$ | 对 batch 内所有样本求平均 | 实践中就是 MSE loss |

**为什么用平方误差而不是绝对值误差**：平方误差处处可导（绝对值在 0 点不可导），且对大误差的惩罚呈平方增长，能更快纠正严重偏离的预测，这是回归问题里最标准的选择，其统计意义是"平方误差最小化对应估计条件期望"，正好和 $V(s)=\mathbb{E}[\cdot]$ 的定义吻合。

**数值例子（例子 B）**：机械臂在某状态 $s_t$，Critic 当前预测 $V_\phi(s_t)=8.0$。这一轮实际估计出的目标回报 $\hat R_t = 9.5$（比预测好一点，说明这个状态实际上比 Critic 以为的更有希望）：

$$
L^V = (8.0 - 9.5)^2 = 2.25
$$

反向传播这个 loss，会把 $V_\phi(s_t)$ 往 $9.5$ 的方向调整。

### 8.3 Critic 和 Actor 共享网络主干

在实践中（尤其是图像/高维观测输入），Actor 和 Critic 通常共享底层的特征提取网络（比如 CNN/Transformer backbone），只在最后分叉成两个输出头：一个输出动作分布参数（Actor head），一个输出一个标量（Critic head）。这样做能减少参数量，也让两个头共享的表征学习互相受益。

---

## 第九步：GAE——怎么估计 $\hat{A}_t$

### 9.1 面临的选择：多看几步 vs 少看几步

有了 $V_\phi(s)$，我们可以构造不同"往前看多少步"的 advantage 估计。先定义**单步 TD 误差**：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

**为什么需要这个公式**：这是 [Bellman 方程](/前置知识/000o_前置知识_Q函数与Value函数) 的直接推论——$r_t+\gamma V(s_{t+1})$ 是"用一步真实奖励加上对下一状态的估计"重新构造出的对 $V(s_t)$ 的一次估计，减去 Critic 当前的预测 $V_\phi(s_t)$，差值就是"这一步比 Critic 预期的好多少"。

> **一句话直觉**：$\delta_t$ 就是"这一步实际拿到的（加上对未来的估计）比 Critic 事先预测的多了多少"。

**逐项拆解**：

| 符号 | 含义 |
|------|------|
| $r_t$ | 这一步的即时奖励 |
| $\gamma V_\phi(s_{t+1})$ | 用 Critic 估计的、下一状态开始的折扣未来价值 |
| $r_t+\gamma V_\phi(s_{t+1})$ | 对 $V(s_t)$ 的"一步展开"估计（真实一步 + 估计的剩余部分） |
| $-V_\phi(s_t)$ | 减去 Critic 原本的预测 |

**数值例子**：$\gamma=0.99$，某一步 $r_t=-0.01$（时间惩罚），$V_\phi(s_{t+1})=8.5$，$V_\phi(s_t)=8.0$：

$$
\delta_t = -0.01 + 0.99\times8.5 - 8.0 = -0.01+8.415-8.0 = 0.405
$$

$\delta_t>0$ 说明这一步走完之后的状态，比 Critic 之前预测的（$8.0$）更有希望（下一状态打折后是 $8.415$）——这是"好的一步"。

$\delta_t$ 就是 $A_t$ 的"1 步估计"，记作 $A_t^{(1)}=\delta_t$。如果往后再展开一步：

$$
A_t^{(2)} = \delta_t + \gamma\delta_{t+1} = r_t+\gamma r_{t+1}+\gamma^2 V_\phi(s_{t+2}) - V_\phi(s_t)
$$

以此类推，展开到无穷步就是纯蒙特卡罗估计：

$$
A_t^{(\infty)} = \sum_{l=0}^{\infty}\gamma^l\delta_{t+l} = \hat{R}_t - V_\phi(s_t)
$$

**核心矛盾**：只用 1 步（$A_t^{(1)}$）的估计**方差小**（只涉及一步随机性），但**有偏**（因为用了 $V_\phi(s_{t+1})$ 这个本身不准确的估计值，误差会渗透进来）；用无穷步（纯 MC）**无偏**（用的都是真实观察到的奖励），但**方差极大**（累积了整条轨迹的所有随机性）。

### 9.2 GAE 的解法：指数加权混合所有步数

**为什么需要这个公式**：既然"看 1 步"和"看无穷步"是方差-偏差的两个极端，自然的想法是取一个折中——把所有步数的估计都算出来，按"离当前越远、越不可信"的原则做加权平均。

$$
A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} A_t^{(n)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}
$$

（两个表达式是等价的，右边这个求和形式是实践中真正用来计算的）

> **一句话直觉**：把 1 步、2 步、3 步……所有长度的 advantage 估计混在一起，越近的估计权重越大，越远的估计权重按指数衰减——用 $\lambda$ 一个参数就控制了"信任近处多一点还是远处多一点"。

**逐项拆解**：

| 符号 | 含义 | 直觉 |
|------|------|------|
| $A_t^{\text{GAE}(\gamma,\lambda)}$ | 最终使用的 advantage 估计 | "综合权衡之后，这个动作比平均好多少" |
| $\delta_{t+l}$ | 第 $t+l$ 步的单步 TD 误差 | 每一步都提供一份"局部惊喜度"信息 |
| $(\gamma\lambda)^l$ | 第 $l$ 项的权重 | 双重衰减：$\gamma^l$ 是奖励本身的折扣，$\lambda^l$ 是对远处 TD 误差的信任度衰减 |
| $\lambda\in[0,1]$ | GAE 的核心超参数 | 控制 bias-variance 权衡的"旋钮" |
| $\sum_{l=0}^\infty$ | 对所有未来步数求和 | 实际实现中到 episode 结束就截断 |

**为什么是 $(\gamma\lambda)^l$ 这个具体形式（不是别的衰减方式）**：$\gamma^l$ 这部分不是新引入的，它就是 1.1 节里"折扣因子"本身的自然延伸——奖励本身就该按 $\gamma$ 打折。额外乘上的 $\lambda^l$ 是 GAE 新引入的"信任度衰减"：越远的 $\delta_{t+l}$ 掺杂了越多层"用 $V_\phi$ 估计未来"的误差，可信度递减，所以用 $\lambda^l$ 单独再打一次折。两者独立存在、相乘组合，是这个公式设计上最关键的地方。

**极端情况验证**：
- $\lambda=0$：只剩 $l=0$ 项，$A_t^{\text{GAE}}=\delta_t=A_t^{(1)}$，纯 TD，低方差高偏差
- $\lambda=1$：$(\gamma\cdot1)^l=\gamma^l$，退化为 $\sum_l\gamma^l\delta_{t+l}=A_t^{(\infty)}=\hat R_t-V_\phi(s_t)$，纯 MC，低偏差高方差
- $\lambda=0.95$（实践中最常用）：介于两者之间

**具体数值例子**：$\gamma=0.99$，$\lambda=0.95$，某 3 步的 TD 误差分别是 $\delta_t=2.0$，$\delta_{t+1}=-0.5$，$\delta_{t+2}=1.0$（假设后面截断为 0）：

$$
A_t^{\text{GAE}} = (0.99\times0.95)^0\times2.0 + (0.99\times0.95)^1\times(-0.5) + (0.99\times0.95)^2\times1.0
$$

逐项算：$(\gamma\lambda)^0=1$，$(\gamma\lambda)^1=0.9405$，$(\gamma\lambda)^2=0.8845$

$$
A_t^{\text{GAE}} = 1.0\times2.0 + 0.9405\times(-0.5) + 0.8845\times1.0 = 2.0 - 0.4703 + 0.8845 = 2.414
$$

对比：如果只用 1 步 TD（$\lambda=0$），$A_t=2.0$；GAE 融合了后两步的信息之后变成 $2.414$——多看到了"后面还有一步不错的表现"，估计更准，但也带进了一点额外的方差。

### 9.3 GAE 的高效递归实现

直接按上面的求和公式，每个 $t$ 都要往后求和到 episode 结束，是 $O(T^2)$ 的复杂度。但注意到：

$$
A_t^{\text{GAE}} = \delta_t + \gamma\lambda\,A_{t+1}^{\text{GAE}}
$$

**这是一个从后往前的递推关系**（用归纳法容易验证：把 $A_{t+1}^{\text{GAE}}=\sum_{l=0}^\infty(\gamma\lambda)^l\delta_{t+1+l}$ 代入右边，整理后正好等于左边的定义），于是可以从轨迹最后一步倒着算到第一步，复杂度降到 $O(T)$：

$$
A_T^{\text{GAE}} = \delta_T \quad(\text{最后一步没有"未来"，直接是单步误差})
$$
$$
A_{T-1}^{\text{GAE}} = \delta_{T-1} + \gamma\lambda\,A_T^{\text{GAE}}
$$
$$
\vdots
$$
$$
A_t^{\text{GAE}} = \delta_t + \gamma\lambda\,A_{t+1}^{\text{GAE}}
$$

对应的实现代码（这是几乎所有 PPO 开源实现里都能找到的一段逻辑，先说清楚思路再看代码：**从最后一步往前遍历，维护一个"上一步（其实是时间上的下一步）已经算出的 GAE 值"，每一步用当前的 TD 误差加上这个值乘 $\gamma\lambda$ 折扣**）：

```python
# 输入: rewards[0..T-1], values[0..T]（包含最后一个状态的 value 估计）, dones[0..T-1]
advantages = zeros(T)
last_gae = 0.0
for t in reversed(range(T)):
    # 如果这一步之后是 episode 终止，未来价值直接置 0（没有"下一步"）
    mask = 1.0 - dones[t]
    delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
    advantages[t] = delta + gamma * lam * mask * last_gae
    last_gae = advantages[t]  # 供上一个时刻（t-1）使用
```

**代码中值得注意的细节**：`mask = 1.0 - dones[t]` 处理的是 episode 边界——如果 $t$ 步是终止步（比如任务失败或超时），就不能把"下一状态的价值"计入，因为根本没有下一状态可言（或者说下一状态属于另一个全新的 episode），这行代码保证了跨 episode 的价值不会被错误地"泄漏"过去。

计算出 advantages 之后，Critic 的训练目标（8.2 节的 $\hat R_t$）通常直接用：

$$
\hat{R}_t = A_t^{\text{GAE}} + V_\phi(s_t)
$$

（因为 $A_t \approx \hat R_t - V_\phi(s_t)$，反解出 $\hat R_t$）

### 9.4 Advantage 归一化

实践中几乎总会对一个 batch 内的所有 $A_t$ 做标准化：

$$
\hat{A}_t \leftarrow \frac{A_t - \text{mean}(A)}{\text{std}(A) + 10^{-8}}
$$

**为什么这样做**：不同任务、不同训练阶段的奖励量级差异很大（有的任务奖励在 $[-1,1]$，有的在 $[-1000,1000]$），如果不归一化，7.2 节里的裁剪范围 $\epsilon$（一个"无量纲"的比例参数）在不同任务/阶段的实际意义会完全不同。归一化后 $A$ 的均值为 0、标准差为 1，让 $\epsilon$ 的含义在各种情况下都可比，训练也更稳定，不会因为某几个 episode 奖励特别大就产生数值爆炸的梯度。分母加 $10^{-8}$ 纯粹是防止 batch 内 $A$ 全部相同导致除零。

---

## 第十步：拼出完整的 PPO——从零件到整机

前九步分别推出了三个"零件"：策略更新目标 $L^{\text{CLIP}}$（第七步）、Critic 损失 $L^V$（第八步）、advantage 估计 $\hat A_t$（第九步）。这一步把它们拼装成一个可以运行的完整算法。

### 10.1 补上最后一个零件：熵奖励

在拼装之前，还需要引入一个在 0 节"终点公式"里出现过、但还没解释的项：熵奖励 $H[\pi_\theta(\cdot|s_t)]$。

**为什么需要它**：7.3 节说 clip 能"减缓"策略过快收敛到确定性，但不能完全阻止——策略仍然可能在很多次迭代后逐渐变得几乎确定（熵趋近于 0），一旦发生这种情况，策略就失去了探索能力，容易卡在次优解上。熵奖励直接在损失函数里**奖励"保持随机性"**，作为额外的安全网。

$$
H[\pi_\theta(\cdot|s_t)] = -\sum_a \pi_\theta(a|s_t)\log\pi_\theta(a|s_t) \quad(\text{离散动作})
$$

这正是本文开头"贯穿全文的例子"KL 散度前置知识里 [信息熵](/前置知识/000j_前置知识_KL散度与策略约束) 的定义——策略在状态 $s_t$ 下，动作分布的不确定性。熵越大，策略越"随机"、越有探索性；熵趋于 0，策略趋于确定性。

> **一句话直觉**：在损失函数里"奖励随机性"，防止策略过早变得只会做一件事。

### 10.2 总损失函数——回到 0 节的终点

现在可以完整解释开篇 0 节的两行公式了：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, \qquad
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]
$$

$$
L^{\text{total}}(\theta,\phi) = -L^{\text{CLIP}}(\theta) + c_1\big(V_\phi(s_t)-\hat R_t\big)^2 - c_2\, H[\pi_\theta(\cdot|s_t)]
$$

**逐项拆解**（现在每一项都已经在前面推过）：

| 符号 | 来自哪一步 | 含义 |
|------|-----------|------|
| $-L^{\text{CLIP}}(\theta)$ | 第七步 | 取负号，因为深度学习框架的优化器默认做**梯度下降**（最小化），而我们的原始目标是**最大化** $L^{\text{CLIP}}$——取负后"最小化 $-L^{\text{CLIP}}$"等价于"最大化 $L^{\text{CLIP}}$" |
| $c_1(V_\phi(s_t)-\hat R_t)^2$ | 第八步 | Critic 的回归损失，$c_1$（典型值 0.5）是它相对策略损失的权重 |
| $-c_2 H[\pi_\theta(\cdot|s_t)]$ | 10.1 节 | 熵奖励，取负号也是因为要"最小化损失"，而我们想**最大化**熵，$c_2$（典型值 0~0.01）是权重 |

**为什么三项要加在一起联合优化**：因为 Actor 和 Critic 通常共享底层网络（8.3 节），一次反向传播同时更新共享参数最高效；即使不共享，把三个损失写在一起也只是记号上的方便，本质上梯度是分别流向各自负责的参数（$\theta$ 只受 $L^{\text{CLIP}}$ 和熵项影响，$\phi$ 只受 $L^V$ 影响）。

### 10.3 一个完整训练迭代（iteration）的四个阶段

$$
\textbf{阶段 1：数据采集（rollout）}
$$

用当前策略 $\pi_{\theta_{\text{old}}}$（也就是上一轮迭代结束时的参数）在 $N$ 个并行环境里各跑 $T$ 步，记录下每一步的 $(s_t,a_t,r_t,\text{done}_t,\log\pi_{\theta_{\text{old}}}(a_t|s_t), V_\phi(s_t))$。**注意这里必须把 $\log\pi_{\theta_{\text{old}}}(a_t|s_t)$ 存下来**——它是 7.1 节 $r_t(\theta)$ 公式里的分母，如果不存下来，之后每个 epoch 重新计算 $r_t(\theta)$ 时就没有"旧策略"这个参照了。

$$
\textbf{阶段 2：Advantage 与目标回报计算}
$$

用 Critic $V_\phi$ 和第九步的递归公式，从后往前算出每个 $(s_t,a_t)$ 的 $\hat A_t$，以及 Critic 的训练目标 $\hat R_t = \hat A_t + V_\phi(s_t)$。对 $\hat A_t$ 做 9.4 节的归一化。

$$
\textbf{阶段 3：多轮 mini-batch 更新}
$$

对同一批数据重复 $K$ 个 epoch（典型值 3~10）：每个 epoch 内把数据随机打乱，切成若干 mini-batch，对每个 mini-batch：
1. 用**当前**（正在更新中的）$\theta$ 重新计算 $\log\pi_\theta(a_t|s_t)$
2. 算出 $r_t(\theta)=\exp\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{\text{old}}}(a_t|s_t)\big)$（用 $\exp(\log a - \log b)$ 而不是直接除法，是为了数值稳定，避免极小概率相除溢出）
3. 计算 $L^{\text{total}}$，反向传播，用优化器（通常是 Adam）更新 $\theta,\phi$

$$
\textbf{阶段 4：换代}
$$

$K$ 个 epoch 结束后，令 $\theta_{\text{old}}\leftarrow\theta$，回到阶段 1，用新的"旧策略"重新采集数据。

### 10.4 为什么允许对同一批数据重复用 $K$ 个 epoch

3.7 节 REINFORCE 和最早的 Actor-Critic 方法（如 A2C）都是"数据用一次就扔"——算一次梯度、更新一次参数，立刻重新采数据。原因是标准策略梯度理论要求数据是**严格 on-policy**（用当前参数采的数据算当前参数的梯度），参数一变，旧数据理论上就"过时"了。

PPO 能重复用同一批数据 $K$ 次，靠的正是 6.2 节的重要性采样：只要新旧策略的 $r_t(\theta)$ 还没跑出 $[1-\epsilon,1+\epsilon]$ 太远，用重要性采样权重"校正"旧数据依然是对新策略梯度的合理近似，而 clip 机制恰好保证了每个 epoch 内 $r_t(\theta)$ 的偏移是可控的。这让 PPO 的样本效率比纯 on-policy 方法高出数倍，是 PPO 相比 A2C 的一个重要工程优势。

### 10.5 完整伪代码

```python
for iteration in range(num_iterations):
    # 阶段 1：采集数据
    trajectories = []
    for env in parallel_envs:  # N 个并行环境
        for t in range(T):
            a_t, logp_old = policy_old.sample_with_logprob(s_t)
            v_t = critic(s_t)
            s_next, r_t, done = env.step(a_t)
            trajectories.append((s_t, a_t, r_t, done, logp_old, v_t))
            s_t = s_next

    # 阶段 2：GAE 计算（见 9.3 节递归公式）
    advantages, returns = compute_gae(trajectories, gamma, lam)
    advantages = normalize(advantages)  # 9.4 节

    # 阶段 3：K 个 epoch 的 mini-batch 更新
    for epoch in range(K):
        for batch in shuffle_and_split(trajectories, advantages, returns):
            logp_new = policy.log_prob(batch.s, batch.a)
            ratio = torch.exp(logp_new - batch.logp_old)          # r_t(θ)

            unclipped = ratio * batch.advantages
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * batch.advantages
            policy_loss = -torch.min(unclipped, clipped).mean()    # L^CLIP 取负

            v_pred = critic(batch.s)
            value_loss = ((v_pred - batch.returns) ** 2).mean()

            entropy_loss = -policy.entropy(batch.s).mean()

            total_loss = policy_loss + c1 * value_loss + c2 * entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # 阶段 4：换代
    policy_old.load_state_dict(policy.state_dict())
```

至此，从"我们想要什么"（第一步）到"能跑起来的完整代码"（这里），整条推导链条闭环了。

---

## 第十一步：连续动作空间——把策略换成高斯分布

到目前为止的所有公式（$\pi_\theta(a|s)$、$\log\pi_\theta$、$r_t(\theta)$）对离散和连续动作都成立，唯一的区别是 $\pi_\theta(a|s)$ 具体怎么参数化。机器人控制几乎总是连续动作空间，最常用的参数化是**高斯策略**：

$$
\pi_\theta(a|s) = \mathcal{N}\big(a;\ \mu_\theta(s),\ \sigma^2 I\big)
$$

网络输出均值 $\mu_\theta(s)$（有时也输出 $\sigma$），动作从这个高斯分布中采样。

### 11.1 高斯的 log 概率——PPO 实际要算的公式

**为什么需要这个公式**：10.3 节阶段 3 的第 1 步要求"计算 $\log\pi_\theta(a_t|s_t)$"，对高斯策略而言，这就是高斯分布的对数概率密度公式。

$$
\log\pi_\theta(a|s) = -\frac{1}{2\sigma^2}\|a-\mu_\theta(s)\|^2 - \frac{d}{2}\log(2\pi\sigma^2)
$$

> **一句话直觉**：采样到的动作 $a$ 离网络预测的"最佳动作" $\mu_\theta(s)$ 越远，log 概率越负（这个动作越"不像"是从这个分布里采出来的）。

**逐项拆解**：

| 符号 | 含义 | 直觉 |
|------|------|------|
| $\mu_\theta(s)$ | 网络输出的动作均值向量 | "网络认为最好的动作" |
| $\sigma^2$ | 每个动作维度的方差（标量或对角矩阵） | $\sigma$ 大 = 更随机/更探索，$\sigma$ 小 = 更确定 |
| $\|a-\mu_\theta(s)\|^2$ | 实际动作到均值的欧氏距离平方 | 采样到的动作离"网络推荐值"有多远 |
| $-\frac{1}{2\sigma^2}$ | 距离的惩罚系数 | $\sigma$ 越小，偏离一点惩罚就越重（分布更"尖"） |
| $-\frac{d}{2}\log(2\pi\sigma^2)$ | 归一化常数，保证概率密度积分为 1 | 训练时这一项在新旧策略间通常大部分抵消（如果 $\sigma$ 不变） |
| $d$ | 动作维度 | 例子 B 中 $d=2$（$\Delta x,\Delta y$） |

**数值例子**（例子 B，$d=2$，$\sigma=0.5$）：网络预测 $\mu_\theta(s)=[1.0,2.0]$，实际采样到 $a=[1.3,1.8]$：

$$
\|a-\mu\|^2 = (1.3-1.0)^2+(1.8-2.0)^2 = 0.09+0.04=0.13
$$
$$
\log\pi = -\frac{0.13}{2\times0.25} - \frac{2}{2}\log(2\pi\times0.25) = -0.26 - \log(1.571) = -0.26-0.452=-0.712
$$

如果动作偏离更远，$a=[2.0,3.0]$：

$$
\|a-\mu\|^2 = 1.0+1.0=2.0,\qquad \log\pi = -\frac{2.0}{0.5}-0.452=-4.452
$$

概率比 $\pi(a_{\text{近}})/\pi(a_{\text{远}}) = \exp(-0.712-(-4.452)) = \exp(3.74)\approx 42$——离均值近的动作，被采到的概率是离均值远的动作的 42 倍。

### 11.2 高斯策略的局限：单峰假设

高斯分布只有一个"峰"（均值附近概率最高）。如果真实的最优动作有多个互相分离的选择（比如绕开障碍物"从左边"和"从右边"都行，但"直接走中间"会撞上），高斯策略会在两个峰之间取平均，输出一个两边都不是的、更糟的动作：

$$
\text{真实最优}：a=-1\text{（左绕）或}\ a=+1\text{（右绕），中间}a=0\text{直接撞墙}
$$
$$
\text{高斯策略学到}：\mu\approx 0,\ \sigma\ \text{较大} \to \text{经常采样到接近 0 的动作} \to \text{撞墙}
$$

这正是 [Diffusion Policy](/前置知识/000c_前置知识_Diffusion_Policy) 要用扩散模型代替高斯分布做策略的核心动机——扩散模型可以表示任意复杂的多峰分布。如果想了解 PPO 如何应用到扩散策略上（把每一步去噪当成一次"动作"做 PPO 更新），可以继续阅读 [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) 和相关的 [DPPO 论文精读](/论文综述/001_DPPO_扩散策略策略优化)。

---

## 第十二步：实际调参——超参数一览与失败诊断

### 12.1 超参数表

| 参数 | 符号 | 典型范围 | 出现在哪一步 | 效果 |
|------|------|---------|-------------|------|
| 折扣因子 | $\gamma$ | 0.99~0.999 | 第一步 1.1 | 越大越重视长期奖励 |
| GAE $\lambda$ | $\lambda$ | 0.9~0.99 | 第九步 9.2 | 越大越低偏差高方差 |
| Clip 范围 | $\epsilon$ | 0.1~0.3 | 第七步 7.2 | 越大允许策略变化更多 |
| PPO epoch 数 | $K$ | 3~10 | 第十步 10.3 | 每批数据更新几轮 |
| Mini-batch 大小 | - | 256~8192 | 第十步 10.3 | 梯度估计的精度 |
| 学习率 | $\alpha$ | 1e-5~3e-4 | 第一步末尾 | 太大不稳定，太小收敛慢 |
| 并行环境数 | $N$ | 8~10000+ | 第十步 10.3 | 越多数据越多，advantage 估计越准 |
| 熵系数 | $c_2$ | 0~0.01 | 第十步 10.1/10.2 | 越大越鼓励探索 |
| 价值损失系数 | $c_1$ | 0.5~1.0 | 第十步 10.2 | Critic 训练的相对权重 |

### 12.2 调参直觉

**Clip ratio $\epsilon$**：判断标准是"clipping fraction"——一个 batch 里有多少样本的 $r_t(\theta)$ 真正被裁剪住了（对应 7.2 节的"选中常数项"情形）。理想比例大概 10%~20%。如果太高（比如超过 40%），说明每次更新步子还是太大，考虑减小学习率或 $\epsilon$。

**$\gamma$ 的选择**：本质上 $\gamma$ 决定了"有效视野长度"——大约 $1/(1-\gamma)$ 步之后的奖励权重降到可忽略。episode 长度在 100 步量级时 $\gamma=0.99$（有效视野~100）合适；episode 长达上千步（很多机器人操作任务）时需要 $\gamma=0.999$（有效视野~1000）。

**并行环境数 $N$**：GPU 大规模并行仿真（Isaac Gym 等）通常用 1000~10000 个环境；CPU 仿真（MuJoCo）通常只能到 40~100 个。$N$ 越大，一次 rollout 收集的数据越多，9.2 节的 advantage 估计方差越小，收敛通常越快越稳。

### 12.3 常见失败模式与诊断

| 失败模式 | 可能原因 | 诊断方法 |
|---------|---------|---------|
| 奖励完全不涨 | 学习率太小 / 熵系数太小导致探索不足 / 奖励太稀疏 | 看熵是否在下降但奖励没涨 |
| 奖励先涨后突然崩 | $\epsilon$ 太大 / epoch 数 $K$ 太多 / Critic 预测不准 | 看 KL 散度是否突然跳变 |
| 不同随机种子结果差异很大 | 奖励设计有多个局部最优 / 初始化敏感 | 跑 5+ 个种子对比方差 |
| Value loss 一直很高降不下去 | 奖励量级太大或训练中剧烈变化 | 对奖励做归一化 |

---

## 第十三步：PPO 的位置——它解决了什么、还留下什么

### 13.1 核心优势

1. **On-policy 里样本效率最高的一类**：靠 clip + 重要性采样（10.4 节），能对同一批数据用 $K$ 个 epoch，比纯 on-policy（A2C/REINFORCE）效率高得多
2. **实现简单**：核心逻辑（10.5 节的伪代码）不到 100 行
3. **天然适配大规模并行仿真**：不需要 replay buffer 之类的额外基础设施
4. **调参相对友好**：各超参数耦合度低

### 13.2 局限

1. **样本效率仍不如真正的 off-policy 方法**（如 [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)）：每个 iteration 采的数据用完（$K$ 个 epoch 后）就永久丢弃，不能像 replay buffer 那样反复重用历史数据
2. **依赖大量并行环境**：如果只有一台真实机器人（没有仿真），单个环境的数据采集速度会让 PPO 的训练非常慢
3. **信用分配在超长 horizon 任务上仍然困难**：GAE 的 advantage 估计依赖 Critic 的准确性，horizon 越长，Critic 越难学准

### 13.3 与本文开头"最大化 $Q(s,a)$"这句话的呼应

回到最初的问题：整条推导链条最终告诉我们，"最大化 reward" 在数学上等价于"让策略在每个状态下都倾向于选择 $Q$ 值（或者更精确地说，Advantage 值）高的动作"（第二步 + 3.6 节）。PPO 没有像 Q-learning 那样直接学 $Q$ 再 $\arg\max$，而是**用 $Q$（通过 Advantage）作为"打分信号"，直接驱动策略网络的参数朝着"多选好动作、少选差动作"的方向调整**（3.5 节），并且给这个调整过程加上了"每步只能走一小段"的安全阀（第五到第七步）。这就是从"最大化 $Q(s,a)$"到"PPO 收敛"的完整逻辑链。

---

## 总结：一张表串起全部十三步

| 步骤 | 核心问题 | 核心公式/结论 |
|------|---------|--------------|
| 1 | 想要什么 | $J(\theta)=\mathbb{E}_\tau[\sum\gamma^t r_t]$，找 $\theta^*=\arg\max J$ |
| 2 | 为什么不直接 $\arg\max Q$ | 连续动作下不可解，转向直接对策略参数求梯度 |
| 3 | 怎么求 $\nabla_\theta J$ | log-derivative trick → $\nabla_\theta J=\mathbb{E}[\nabla\log\pi\cdot A]$（带 baseline 后） |
| 4 | （并入第3步）| Advantage $A=Q-V$ 替代 $Q$，降方差 |
| 5 | 为什么不能直接梯度上升 | 策略行为剧变 → 数据分布跟着变 → 崩溃且不可逆 |
| 6 | TRPO 怎么限制 | 重要性采样代理目标 + KL 散度硬约束 |
| 7 | PPO 怎么简化 | Clip 概率比，替代 KL 约束，$L^{\text{CLIP}}=\mathbb{E}[\min(r\hat A,\text{clip}(r)\hat A)]$ |
| 8 | $V(s)$ 哪来 | Critic 网络，MSE 回归到 $\hat R_t$ |
| 9 | $\hat A_t$ 怎么算 | GAE：指数加权混合各步长 TD 误差，$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ |
| 10 | 整机怎么跑 | 采集 → 算 GAE → $K$ 个 epoch 的 mini-batch 更新 → 换代 |
| 11 | 连续动作怎么办 | 高斯策略，$\log\pi$ 变成高斯 log-likelihood |
| 12 | 怎么调参 | $\gamma,\lambda,\epsilon,K,N,c_1,c_2$ 及诊断表 |
| 13 | PPO 的位置 | On-policy 里效率最高，但不如 off-policy 省数据 |

---

## 思考题

1. 3.9 节证明了"减去只依赖状态的 baseline 不改变梯度期望"。如果 baseline 是 $b(s,a)$（依赖动作），这个证明还成立吗？为什么？
2. 6.2 节的重要性采样权重 $\pi_{\text{new}}(a|s)/\pi_{\text{old}}(a|s)$，如果某个动作在 $\pi_{\text{old}}$ 下概率极低（比如 $10^{-6}$），而 $\pi_{\text{new}}$ 恰好大幅提高了它的概率，会发生什么数值问题？PPO 的 clip 机制能否完全避免这个问题？
3. 7.2 节说 $L^{\text{CLIP}}$ 是真实目标的"悲观下界"。如果把 $\min$ 换成 $\max$，会发生什么？
4. 9.2 节 GAE 的 $\lambda$ 和折扣因子 $\gamma$ 是两个独立的参数，但都在控制某种"衰减"。如果只用一个参数（令 $\lambda=\gamma$），会损失什么灵活性？
5. 10.4 节说 PPO 能对同一批数据用 $K$ 个 epoch，是因为 clip 保证了 $r_t(\theta)$ 的偏移可控。如果把 $K$ 设成 100 会发生什么？clip 机制能兜住吗？

---

## 延伸阅读

- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — $Q$、$V$、$A$ 的完整定义与 Bellman 方程
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — 熵、KL 散度的独立详解，以及 PPO clip 与隐式 KL 约束的关系
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — Off-policy 路线的对比方案
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — 当策略从高斯换成扩散模型，PPO 怎么继续用
- [DPPO 论文精读](/论文综述/001_DPPO_扩散策略策略优化) — PPO 在扩散策略上的具体应用
- [深度强化学习方法综述](/论文综述/S01_深度强化学习方法综述) — PPO 在更大的 RL 方法版图中的位置
