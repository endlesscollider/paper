---
title: DDPG（确定性策略梯度）
order: 16
tags: [强化学习]
category: 前置知识
---

# 前置知识：DDPG——Deep Deterministic Policy Gradient

> **一句话**：DDPG 是第一个成功将深度学习应用于连续动作空间 RL 的算法。它结合了 DQN 的 replay buffer + target network 技巧和 Actor-Critic 架构，用确定性策略直接输出连续动作。

**前置概念**：
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Q 值的含义和 Bellman 方程
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — Actor-Critic 的基本思想

---

## 贯穿全文的例子

> 一个单摆（Pendulum）需要学习用连续力矩将摆杆竖直并保持平衡。
> - **状态** $s = [\cos\theta, \sin\theta, \dot\theta] \in \mathbb{R}^3$：摆角和角速度
> - **动作** $a \in [-2, 2]$：施加的力矩（连续值）
> - **奖励**：$r = -(\theta^2 + 0.1\dot\theta^2 + 0.001a^2)$（越竖直、越稳定、力矩越小越好）

---

## 一、动机：DQN 不能处理连续动作

### 1.1 DQN 的局限

DQN 通过 $a^* = \arg\max_a Q(s, a)$ 选最优动作。这要求能对所有动作求 max——当动作空间是离散且有限的（如上下左右 4 个方向），可以逐个比较。

但如果动作空间是连续的（如力矩 $a \in [-2, 2]$），$\arg\max$ 就变成了一个连续优化问题——每步决策都要解一次优化，计算上不可行。

### 1.2 DDPG 的解法

DDPG 的思路：**用一个神经网络直接输出最优动作**，不再对 Q 求 max。

$$
\mu_\theta(s) \approx \arg\max_a Q(s, a)
$$

这个网络叫 **Actor**（策略网络），它输出一个确定性的动作值（不是概率分布）。

---

## 二、算法核心

### 2.1 四个网络

| 网络 | 符号 | 作用 |
|------|------|------|
| Actor | $\mu_\theta(s)$ | 输入状态，输出确定性动作 |
| Critic | $Q_\phi(s, a)$ | 输入状态+动作，输出 Q 值 |
| Target Actor | $\mu_{\bar\theta}(s)$ | Actor 的慢更新副本 |
| Target Critic | $Q_{\bar\phi}(s, a)$ | Critic 的慢更新副本 |

### 2.2 Critic 更新（学 Q 函数）

目标：让 $Q_\phi(s, a)$ 满足 [Bellman 方程](/前置知识/000o_前置知识_Q函数与Value函数)：

$$
L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[(Q_\phi(s,a) - y)^2\right]
$$

$$
y = r + \gamma \cdot Q_{\bar\phi}(s', \mu_{\bar\theta}(s'))
$$

**逐项拆解**：
- $(s,a,r,s')$：从 Replay Buffer $\mathcal{B}$ 中采样的经验
- $y$：Bellman 目标——即时奖励 + 折扣后的下一状态 Q 值
- $\mu_{\bar\theta}(s')$：Target Actor 预测下一状态的动作
- $Q_{\bar\phi}(s', \cdot)$：Target Critic 评估下一步的价值

### 2.3 Actor 更新（确定性策略梯度）

Actor 的目标：输出使 Q 值最大的动作。

$$
\nabla_\theta J = \mathbb{E}_{s \sim \mathcal{B}}\left[\nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]
$$

**一句话**：先问 Critic "Q 值沿哪个方向增大最快"（$\nabla_a Q$），再问 Actor "怎么调参数能让输出往那个方向走"（$\nabla_\theta \mu$）。

**代入例子**：
- 当前状态：摆杆偏左 30°
- Actor 输出：$\mu_\theta(s) = 0.8$（向右施力 0.8）
- Critic 评估：$Q_\phi(s, 0.8) = -2.5$
- Critic 梯度：$\nabla_a Q|_{a=0.8} = +1.2$（力矩再大一点 Q 值更高）
- Actor 更新：调整参数让 $\mu_\theta(s)$ 朝 $> 0.8$ 的方向变化

### 2.4 探索：OU 噪声

DDPG 的策略是确定性的（$\mu_\theta(s)$ 输出唯一一个值），没有内在的随机性。探索完全靠外部加噪声：

$$
a = \mu_\theta(s) + \mathcal{N}_{\text{OU}}
$$

Ornstein-Uhlenbeck (OU) 噪声是一种时间相关的噪声（比白噪声更平滑），适合连续控制。

### 2.5 Target Network 的软更新

$$
\bar\theta \leftarrow (1 - \tau)\bar\theta + \tau\theta, \quad \tau = 0.001
$$

每步只把主网络参数的 0.1% 混入 target 网络。这比 DQN 的"每 N 步硬更新"更平滑稳定。

---

## 三、DDPG 的问题

| 问题 | 表现 | 后续解决方案 |
|------|------|------------|
| Q 值过估计 | Critic 系统性地高估 Q 值，策略被误导 | [TD3](/前置知识/000q_前置知识_TD3) 的双 Q 取最小值 |
| 探索不足 | OU 噪声是无方向的，不知道该往哪探索 | [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) 的最大熵目标 |
| 对超参敏感 | 噪声幅度、学习率稍有变化就崩溃 | TD3 的延迟更新 + SAC 的自动温度 |
| 脆弱性 | 不同随机种子结果差异巨大 | TD3/SAC 的多项稳定化技术 |

---

## 四、DDPG 的历史意义

DDPG (Lillicrap et al., 2015) 是连续控制 Deep RL 的**开山之作**。虽然现在已被 TD3 和 SAC 全面取代，但它建立的框架——"Actor 输出连续动作 + Critic 评估 + Replay Buffer + Target Network"——至今仍是所有连续动作 off-policy 方法的骨架。

---

## 延伸阅读

- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Q 值和 Bellman 方程的基础
- [TD3](/前置知识/000q_前置知识_TD3) — DDPG 的直接改进
- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 用最大熵替代外加噪声
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — 随机策略梯度 vs 确定性策略梯度的对比
