---
title: GRPO：Group Relative Policy Optimization
order: 13
tags: [强化学习, GRPO, 策略优化, VLA]
category: 前置知识
---

# GRPO：Group Relative Policy Optimization（组内相对策略优化）

> **一句话**：GRPO 是一种无需 Critic（价值网络）的策略优化算法，通过同一个输入的多次采样互相比较来估计 advantage，用"组内排名"代替 value function。

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 的 clip 机制和 advantage 估计基础
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — GRPO 中的 KL penalty 来源

---

## 一、为什么需要 GRPO

### 1.1 PPO 的 Critic 问题

回顾 [PPO](/前置知识/000a_前置知识_策略梯度与PPO) 的核心公式：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

其中 advantage $\hat{A}_t$ 通常由 GAE 计算：

$$
\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**核心依赖**：PPO 需要一个 Critic 网络 $V(s)$ 来估计每个状态的价值。

对于大模型（7B 参数的 VLA），这意味着：
- **显存翻倍**：需要同时维护 Actor（7B）+ Critic（7B）= 14B 参数
- **训练不稳定**：在稀疏奖励下，Critic 的 value 估计很不准确，反而误导策略更新
- **工程复杂**：需要同步训练两个大模型，调参难度加倍

### 1.2 GRPO 的核心想法

DeepSeek-R1 提出的 GRPO 用一个简单的观察绕过了 Critic：

> **如果我对同一个输入做多次采样，那么"比平均好"的就是正 advantage，"比平均差"的就是负 advantage。不需要 Critic！**

这就像考试不看绝对分数，只看班级排名——每道题出若干份答卷，互相比较就知道哪个好。

---

## 二、算法详解

### 2.1 采样阶段：Group Sampling

对于给定的输入（在 LLM 中是 prompt，在 VLA 中是观测+指令），采样 $G$ 个独立的输出（在 LLM 中是 response，在 VLA 中是轨迹）：

$$
\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot | q)
$$

其中 $q$ 是输入（query / context），$G$ 是 group size（通常 8–16）。

每个输出获得一个奖励 $r_i = R(q, o_i)$。

**在 VLA 中的具体含义**：
- $q$ = 当前图像 + 语言指令（如 "pick up the red mug"）
- $o_i$ = 一条完整的轨迹（从开始到结束的动作序列）
- $r_i$ = 0 或 1（任务是否成功）

### 2.2 Advantage 估计：组内归一化

**GRPO 的 advantage 估计公式**：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}
$$

**逐项拆解**：
- $r_i$：第 $i$ 个输出的奖励
- $\text{mean}(\{r_j\})$：组内所有奖励的均值，作为 baseline
- $\text{std}(\{r_j\})$：组内所有奖励的标准差，用于归一化

**一句话**：GRPO 的 advantage 就是"你的奖励比组内平均高多少个标准差"。

### 2.3 代入数字的例子

假设 VLA 执行 "pick up the butter" 任务，group size $G = 8$，8 条轨迹的结果为：

| 轨迹编号 | 是否成功 | 奖励 $r_i$ |
|---------|---------|-----------|
| 1 | 失败 | 0 |
| 2 | 失败 | 0 |
| 3 | **成功** | 1 |
| 4 | 失败 | 0 |
| 5 | 失败 | 0 |
| 6 | **成功** | 1 |
| 7 | 失败 | 0 |
| 8 | 失败 | 0 |

计算：
- $\text{mean} = (0+0+1+0+0+1+0+0)/8 = 0.25$
- $\text{std} = \sqrt{\frac{6 \times (0-0.25)^2 + 2 \times (1-0.25)^2}{8}} \approx 0.433$

各轨迹的 advantage：
- 成功轨迹（$r=1$）：$\hat{A} = (1 - 0.25) / 0.433 = +1.73$（应该被奖励）
- 失败轨迹（$r=0$）：$\hat{A} = (0 - 0.25) / 0.433 = -0.577$（应该被惩罚）

### 2.4 策略更新：PPO-style Clip

有了 advantage 后，GRPO 用和 PPO 相同的 clipped surrogate objective 更新策略：

$$
L^{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta) \hat{A}_i, \; \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

**逐项拆解**：
- $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}$：第 $i$ 个输出在第 $t$ 个 token 位置的概率比
- $\hat{A}_i$：该输出的组内归一化 advantage（注意是**轨迹级**的，同一条轨迹内所有 token 共享同一个 advantage）
- $\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$：PPO 的裁剪，限制更新幅度
- $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$：[KL 散度](/前置知识/000j_前置知识_KL散度与策略约束)正则项，防止策略偏离 SFT 模型太远
- $\beta$：KL 惩罚系数

### 2.5 变体：Leave-One-Out (RLOO)

RIPT-VLA 使用的是 GRPO 的一个变体叫 RLOO（Leave-One-Out）：

$$
b_k = \frac{1}{G-1}\sum_{j \neq k} r_j, \quad \hat{A}_k = r_k - b_k
$$

**区别**：RLOO 对每个样本 $k$ 的 baseline 是**去掉自己后**的均值。这降低了 bias（因为 baseline 不包含自己的奖励）。

**代入上面的例子**：对于第 3 条轨迹（成功，$r_3=1$）：
- $b_3 = (0+0+0+0+1+0+0)/7 = 1/7 \approx 0.143$
- $\hat{A}_3 = 1 - 0.143 = 0.857$

对于第 1 条轨迹（失败，$r_1=0$）：
- $b_1 = (0+1+0+0+1+0+0)/7 = 2/7 \approx 0.286$
- $\hat{A}_1 = 0 - 0.286 = -0.286$

---

## 三、GRPO 在 VLA 场景中的挑战

### 3.1 稀疏奖励问题

VLA 场景中奖励通常是 binary（0/1 的成功/失败）。当任务较难时，可能 $G=8$ 条轨迹中只有 0-1 条成功。

**最坏情况**：8 条全部失败
- $r_1 = r_2 = \cdots = r_8 = 0$
- $\text{mean} = 0$, $\text{std} = 0$
- **所有 advantage 都是 0！完全没有梯度信号。**

这是 GRPO 在 VLA 中的根本缺陷。RIPT-VLA 的 "Dynamic Rejection" 就是为了解决这个问题——丢弃这种全零的 group，重新采样直到组内有差异。

### 3.2 和 PPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| Advantage 估计 | Critic 网络 $V(s)$ + GAE | 组内比较 |
| 额外参数 | 需要 Critic（同等规模） | 不需要 |
| 显存 | 2x 模型参数 | 1x 模型参数 |
| 稀疏奖励处理 | Critic 可以学到中间 value | 只有终端奖励信号 |
| 样本效率 | 较高（每步都有信号） | 较低（需要同组有对比） |
| 工程复杂度 | 高（两个模型同步训练） | 低（只训练一个模型） |

**经验结论**（来自多篇 VLA 论文的实验）：
- 当奖励较 dense 或任务较简单时，GRPO 效果接近 PPO 且更省资源
- 当奖励高度稀疏（只有最终 0/1）时，PPO 显著优于 GRPO

---

## 四、GRPO 的适用场景

### 4.1 最适合的场景

1. **模型很大，显存紧张**：7B+ 的 VLA/LLM，无法同时放 Actor + Critic
2. **奖励有一定 density**：不只是最终 0/1，而是有部分成功（如多阶段任务的阶段奖励）
3. **快速原型验证**：不需要训 Critic，减少调参维度

### 4.2 不适合的场景

1. **极端稀疏奖励**：大部分 rollout 全失败时，GRPO 学习信号接近零
2. **需要精细的 credit assignment**：GRPO 把整条轨迹的奖励分配给所有 token，不区分哪一步做对了哪一步做错了
3. **任务太难导致成功率极低**：如果 $G=16$ 条 rollout 中成功率 < 5%，很多 group 内全是 0

---

## 五、总结

| 概念 | 说明 |
|------|------|
| 核心思想 | 同一输入多次采样，用组内比较代替 Critic |
| advantage | $(r_i - \text{mean}) / \text{std}$ |
| 优点 | 无需 Critic，省显存，工程简单 |
| 缺点 | 稀疏奖励下信号弱，credit assignment 粗糙 |
| 代表应用 | DeepSeek-R1, RIPT-VLA, TGRPO |
| VLA 中的主要改进方向 | Dynamic Rejection, Progress Reward (SRPO) |

---

## 延伸阅读

- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 的完整算法
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — GRPO 中 KL penalty 的意义
- [动作 Token 化与自回归策略](/前置知识/000l_前置知识_动作Token化与自回归策略) — 自回归 VLA 的 token-level log-prob 计算
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — GRPO 在 VLA 场景中的全面对比
