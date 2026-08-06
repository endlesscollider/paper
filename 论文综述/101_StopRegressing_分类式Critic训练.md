---
title: Stop Regressing：用分类代替回归训练 Critic
order: 301
tags: [强化学习, Critic, 分类, 交叉熵, HL-Gauss, 分布式RL, Scaling, VLA]
category: 精读
star: 5
---

# Stop Regressing：用分类代替回归训练 Value Function

> **论文标题**: Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
> **作者**: Jesse Farebrother, Jordi Orbay, Quan Vuong, Adrien Ali Taïga 等 (Google DeepMind)
> **发表**: ICML 2024 (Oral)
> **arXiv**: [2403.03950](https://arxiv.org/abs/2403.03950)

**标签**: `#Critic训练` `#分类` `#交叉熵` `#HL-Gauss` `#Scaling` `#Transformer` `#机器人`

**知识链接**：
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Value function 基本定义
- [分布式值函数与类别化回报预测](/前置知识/002q_前置知识_分布式值函数与类别化回报预测) — C51/QR-DQN 等分布式方法
- [Diagnosing Bottlenecks 精读](/论文综述/100_DiagnosingBottlenecks_Q学习瓶颈诊断) — 本文解决的就是那篇发现的"过拟合"和"非平稳 target"问题
- [RECAP 精读](/论文综述/016_RECAP_从真实部署经验中RL学习) — 使用了 categorical value function 的 VLA 实例

---

## 一、核心发现（一句话）

**把 Critic 的训练 loss 从 MSE 回归换成 categorical cross-entropy 分类，几乎不增加成本，但性能和 scaling 能力大幅提升。**

在 Atari（+30%）、机器人操作（+67%）、Chess（+70%）、Wordle（+40%）上都有效，且网络越大提升越明显——终于让 value-based RL 也能 scale up 了。

---

## 二、问题：为什么 MSE 回归训 Critic 在大网络上失效

在监督学习中，越大的网络性能越好（GPT scaling law）。但 value-based RL 中观察到一个反常现象：**把 Q-network 从 ResNet-18 换到 ResNet-101，性能反而下降。**

作者认为根源在于 MSE loss 对 RL 特有挑战的脆弱性：

| RL 特有挑战 | MSE 的问题 |
|------------|-----------|
| **Target 有噪声**（环境随机性、采样方差） | MSE 对离群值梯度线性增长，被噪声 target "拉飞" |
| **Target 非平稳**（策略在改善 → Q* 在变大） | 网络拟合旧 target 后，对新 target 的适应力下降（可塑性丧失） |
| **Bootstrap 误差**（target 本身就不准） | 不准的 target 通过 MSE 直接注入网络权重 |

Cross-entropy loss 对以上三个问题都更鲁棒——这是本文的核心假设，后面用大量实验验证。

---

## 三、方法：HL-Gauss —— 把 TD target 变成分类问题

### 3.1 基本思路

传统方法：$Q_\theta(s,a)$ 输出一个标量，用 MSE 拟合标量 target。

本文方法：$Q_\theta(s,a)$ 输出一个 **categorical 分布**（类似 softmax 后的 $m$ 类概率），target 也是一个分布，用 cross-entropy 训练。

具体步骤：
1. 把 value 的范围 $[v_\min, v_\max]$ 均匀切成 $m$ 个 bin（如 $m = 101$）
2. Q-network 最后一层输出 $m$ 个 logit → softmax → $m$ 个概率 $\hat{p}_i$
3. Q 值 = 这个分布的期望：$Q(s,a) = \sum_i \hat{p}_i \cdot z_i$
4. Target 标量 $y = r + \gamma \max_{a'} Q(s', a')$ 也转成一个分布
5. 用 cross-entropy 训练：$L = -\sum_i p_i \log \hat{p}_i$

### 3.2 如何把标量 target 变成分布：三种方式

| 方法 | 做法 | 效果 |
|------|------|------|
| **One-Hot** | 离散到最近的 bin | 有量化误差，效果差 |
| **Two-Hot** | 概率分配到 target 两侧最近的两个 bin | 无量化误差但无 smoothing |
| **HL-Gauss** ⭐ | 以 target 为均值做一个高斯分布，再投影到 bins 上 | 最优：既无量化误差又有 smoothing |

### 3.3 HL-Gauss 的数学表达

给定 target 值 $y$，构造高斯 $\mathcal{N}(y, \sigma^2)$，然后算每个 bin 的概率：

$$
p_i = F\left(\frac{z_i + \varsigma/2 - y}{\sigma}\right) - F\left(\frac{z_i - \varsigma/2 - y}{\sigma}\right)
$$

**这个公式在做什么**：把以 target 为中心的高斯分布"切"到各个 bin 里，得到每个 bin 的目标概率。$\sigma$ 控制"smoothing"程度。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $z_i$ | 第 $i$ 个 bin 的中心 | 如 $[-10, -9.8, \ldots, 9.8, 10]$ |
| $\varsigma$ | bin 宽度 = $(v_\max - v_\min) / m$ | 如 $20/101 = 0.198$ |
| $\sigma$ | 高斯标准差（控制 smoothing） | 推荐 $\sigma = 0.75 \varsigma$ |
| $F(\cdot)$ | 标准正态 CDF（erf 函数） | |
| $y$ | TD target 值 | 如 $y = 3.5$ |

**数值代入**：$y = 3.5$，$\sigma = 0.15$，$\varsigma = 0.2$，看 $z_{27} = 3.4$ 的概率：

$$
p_{27} = F\left(\frac{3.5 - 3.5}{0.15}\right) - F\left(\frac{3.3 - 3.5}{0.15}\right) = F(0) - F(-1.33) = 0.5 - 0.091 = 0.409
$$

大部分概率落在 target 附近的几个 bin 中，形成"软标签"。

**为什么 $\sigma/\varsigma = 0.75$ 最好？** 这使得约 99.7% 的概率分布在约 5 个 bin 内——既足够 smooth 又不过于分散。实验显示这个值对 bin 数量不敏感。
:::

---

## 四、为什么 Cross-Entropy 比 MSE 好：三个原因

### 4.1 对噪声 target 更鲁棒

作者在 offline RL 数据上人为注入 reward noise $\epsilon \sim U(0, \eta)$，比较 HL-Gauss vs MSE 的 degradation：

| 噪声水平 $\eta$ | MSE 性能下降 | HL-Gauss 性能下降 |
|-----------------|-------------|------------------|
| 0.1 | -5% | -2% |
| 0.3 | -15% | -8% |
| 1.0 | -35% | -18% |

**原因**：MSE loss 对大误差的梯度是 $2(Q - y)$ —— 线性增长，一个离群 target 就能把梯度拉飞。Cross-entropy 的梯度有上界（bounded by logit range），不容易被单个噪声样本"绑架"。

### 4.2 在非平稳 target 下保持可塑性

用 CIFAR10 上的合成实验模拟 RL 中 target 不断变大的过程（模拟策略改善 → Q* 增大）：

- MSE 训练的网络在 target 变化后**逐渐丧失学习能力**——需要越来越多 steps 才能拟合新 target
- HL-Gauss 训练的网络**始终保持接近的学习速度**

这直接验证了 Primacy Bias / Plasticity Loss 的现象——MSE 是帮凶，CE 不是。

### 4.3 学到更好的表征

用 linear probing 评估（冻结特征层，只训最后一层线性头来重新学 Q）：

- HL-Gauss 学到的表征明显优于 MSE —— 说明分类 loss 迫使网络学到更 discriminative 的特征
- C51（分布式 RL）的表征也好，但**HL-Gauss 不需要建模完整回报分布也能达到同样效果**

**核心洞察**：C51 的成功可能主要归功于 cross-entropy loss，而不是"学完整分布"这个概念本身。

---

## 五、关键实验结果

### 5.1 Scaling 效果（最重要的结果）

| 设置 | MSE 在 scale up 后 | HL-Gauss 在 scale up 后 |
|------|-------------------|------------------------|
| Multi-task Atari (ResNet 34→101) | 性能**下降** | 性能**继续上升** |
| SoftMoE (1→8 experts) | 缓慢上升 | 上升且 baseline 就高 30% |
| Multi-game Offline (ResNet 34→101) | 饱和 | IQM +45% over prior SOTA |

**结论**：HL-Gauss 让 value-based RL 终于具备了 "bigger is better" 的 scaling 行为。

### 5.2 机器人操作（Q-Transformer）

在 7-DoF 移动操作器 + 17 种物体抓取任务上，用 Q-Transformer (60M 参数)：
- MSE：peak success rate ≈ 45%
- HL-Gauss：peak success rate ≈ 75%（**+67%**），且学习速度快 2x

### 5.3 HL-Gauss vs Two-Hot vs C51

| 方法 | 在线 RL | 离线 RL | 训练稳定性 |
|------|---------|---------|-----------|
| MSE | baseline | baseline | 后期 loss 反弹 |
| Two-Hot | 略差于 MSE | 略好于 MSE | 不反弹但不如 HL-Gauss |
| C51 | 好于 MSE | 好于 MSE | 稳定 |
| **HL-Gauss** | **最好** | **最好** | **最稳定** |

Two-Hot 虽然也用 CE loss 但效果不如 HL-Gauss——说明 **label smoothing**（概率分散到邻近 bin）是关键。

---

## 六、实操指南：如何在你的代码中替换

改动极小——只改 loss 计算，网络结构几乎不变：

**改动 1**：Q-network 最后一层从 1 个输出改为 $m$ 个（如 101 或 256）
**改动 2**：输出过一个 softmax → 得到 Q = 概率加权求和
**改动 3**：target 用 HL-Gauss 转成分布 → 用 cross-entropy 计算 loss

**超参数**：
- `num_bins`：101（Atari）或 256（RECAP/DreamerV3 用的）
- `v_min, v_max`：取决于环境 reward 范围
- `sigma`：$0.75 \times \text{bin\_width}$

**已经使用这个方法的知名工作**：
- RECAP (π₀.6) — 真实机器人 VLA + Critic
- DreamerV3 — 世界模型中的 value head
- TD-MPC2 — 连续控制的 value function
- Q-Transformer — 机器人操作

---

## 七、对 VLA Critic 训练的启示

1. **无脑替换 MSE → HL-Gauss**：成本几乎为零，在所有场景下都不会更差
2. **大 Critic 可以放心用**：MSE 下大网络会退化，HL-Gauss 下大网络持续获益
3. **Transformer Critic 终于可行**：之前 Transformer 做 value function 很难训，本文证明 CE loss 解决了这个问题
4. **解释了 RECAP 和 DreamerV3 的成功**：它们用 categorical value function 不是因为"学分布更好"，而是因为 CE loss 更稳定

---

## 延伸阅读

- [分布式值函数与类别化回报预测](/前置知识/002q_前置知识_分布式值函数与类别化回报预测) — categorical RL 的数学基础
- [Diagnosing Bottlenecks 精读](/论文综述/100_DiagnosingBottlenecks_Q学习瓶颈诊断) — 本文解决的"过拟合"和"非平稳"正是那篇发现的主要瓶颈
- [Distributional RL 教科书](https://www.distributional-rl.org/) — Bellemare et al., MIT Press, Open Access
- [RECAP 精读](/论文综述/016_RECAP_从真实部署经验中RL学习) — categorical value 在 VLA 中的实际应用
- [The Primacy Bias](/论文综述/102_PrimacyBias_深度RL的初始偏差) — HL-Gauss 能缓解可塑性丧失的理论联系
