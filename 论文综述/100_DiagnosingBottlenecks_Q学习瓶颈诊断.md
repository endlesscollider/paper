---
title: Diagnosing Bottlenecks in Deep Q-learning：Q 学习瓶颈诊断
order: 300
tags: [强化学习, Q-learning, 诊断, 函数逼近, 过拟合, 采样分布, Critic]
category: 精读
star: 5
---

# Diagnosing Bottlenecks in Deep Q-learning：Q 学习瓶颈诊断

> **论文标题**: Diagnosing Bottlenecks in Deep Q-learning Algorithms
> **作者**: Justin Fu, Aviral Kumar, Matthew Soh, Sergey Levine
> **机构**: UC Berkeley / Google Brain
> **发表**: ICML 2019
> **arXiv**: [1902.10250](https://arxiv.org/abs/1902.10250)

**标签**: `#Q-learning` `#诊断` `#函数逼近` `#过拟合` `#采样分布` `#Critic训练`

**知识链接**：
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Q-learning 的基本定义
- [Replay Buffer 经验回放](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — 本文大量讨论 replay 的作用
- [TD 学习与 n 步回报的偏差问题](/前置知识/001k_前置知识_TD学习与n步回报的偏差问题) — bootstrap 误差
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 本文实验使用的算法之一
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — 分布漂移的基础概念

---

## 一、为什么读这篇论文

这篇 2019 年的 ICML 论文做了一件非常有价值的事情：**把 Q-learning 拆成独立的三个误差源，然后用"单元测试"的方式逐一诊断每个源头对最终性能的影响。**

它回答了三个实际问题：
1. **函数逼近**会不会导致 Q-learning 发散？（结论：大网络几乎不会）
2. **采样不足/过拟合**对 Critic 伤害有多大？（结论：是主要瓶颈之一）
3. **分布漂移和非平稳 target** 是不是训练不稳定的元凶？（结论：出乎意料，不是）

这些结论对今天的 VLA Critic 训练仍然非常有指导意义——很多人凭直觉认为"分布漂移最可怕"，但这篇论文用控制实验证明了**过拟合和网络容量**才是更关键的因素。

---

## 二、核心方法：Unit Testing 框架

本文的核心贡献是设计了一个可以用 oracle 替换各个组件的实验框架，从而"隔离"不同误差源。

### 2.1 三种 FQI 变体：逐步去除 oracle

作者设计了三种 Fitted Q-Iteration（FQI）变体，从"全知"到"正常 RL"逐步引入误差：

| 算法 | 知道动力学？ | 用全部状态还是采样？ | 有分布漂移？ |
|------|------------|-------------------|------------|
| **Exact-FQI** | ✅ 精确计算 Bellman backup | ✅ 所有状态-动作对 | ⚠️ 可以控制加权分布 |
| **Sampled-FQI** | ❌ 用采样近似 | ❌ 有限采样 | ⚠️ 固定分布 |
| **Replay-FQI** | ❌ 用采样近似 | ❌ 从 replay buffer 采样 | ✅ 有，类似 DQN |

这个设计的精妙之处在于：Exact-FQI 没有任何采样误差——如果它还出问题，那就是**纯粹的函数逼近**在捣鬼。Sampled-FQI 引入了采样误差但没有分布漂移——如果它比 Exact-FQI 差很多，就说明**过拟合**是瓶颈。

### 2.2 实验域：8 个可以精确计算 Q* 的表格环境

作者选了 8 个可以精确求解 $Q^*$ 的小环境（gridworld、cliffwalk、离散化 pendulum/mountain car 等），这样就能用 $\|Q_\text{learned} - Q^*\|_\infty$ 来**精确度量** Critic 的误差。

---

## 三、关键发现 1：函数逼近几乎不导致发散

### 3.1 实验设置

用 Exact-FQI（无采样误差）+ 不同大小的网络（4×4 到 256×256），在所有 8 个环境上跑，看：
- FQI 找到的解和 $Q^*$ 之间的距离（FQI Error）
- "模型类内最优解"和 $Q^*$ 之间的距离（Projection Error）
- 两者的差 = **bootstrap 过程引入的额外偏差**

### 3.2 结论

1. **发散几乎不发生**：在所有实验中只有 0.9% 出现了 Q 值发散（定义为最大 Q 值超过 $Q^*$ 的 10 倍）
2. **大网络的 bootstrap 偏差极小**：256×256 网络下，FQI Error ≈ Projection Error，说明 bootstrap 过程几乎没有引入额外误差
3. **小网络的 bootstrap 偏差很大**：4×4 网络下，FQI Error 远大于 Projection Error——即使模型类内存在不错的解，bootstrap Q-learning 也找不到它

**核心直觉**：大网络可以在每步 backup 后完成近乎完美的投影（因为容量足够拟合 $\mathcal{T}Q$ 投影到模型类的结果），而小网络投影误差大 → 误差在 bootstrap 链中累积 → 找到的不动点远离最优。

**对 VLA Critic 的启示**：不要怕 Critic 网络太大——大网络的函数逼近带来的好处远大于过拟合风险。VLA 中使用 7B 参数的模型做 Critic head 是完全合理的。

---

## 四、关键发现 2：过拟合是真正的瓶颈

### 4.1 过拟合的量化

用 Sampled-FQI，固定网络大小（256×256），改变每轮迭代用的样本数（从 32 到全部状态），测量：
- **训练 Bellman Error**（在采样数据上的 loss）
- **验证 Bellman Error**（在全部状态上的精确 loss）
- **最终 policy 的 return**

### 4.2 结论

1. **样本越少，验证 loss 越高**——典型的过拟合信号
2. **Replay buffer 的真正好处不是减少分布漂移，而是增加有效样本数**——从 buffer 采样时验证 loss 最低
3. **gradient steps 过多会导致性能下降**——TD3 默认每 step 只做 1 次梯度更新是有道理的，多了就过拟合

### 4.3 缓解过拟合的方法

作者测试了 oracle early stopping（用精确验证 Bellman Error 或精确 return 来决定何时停止 gradient 更新），发现性能有明显提升。这说明：

> **如果能设计一个好的 validation metric（不需要 oracle），就能显著改善 Critic 训练。**

这正是后来 2023 年那篇 "Understanding Sample-Efficient Deep RL"（arXiv:2304.10466）的出发点——他们发现 **validation TD error** 是预测 RL 性能最好的单一指标。

### 4.4 实操建议

- **不要减小网络来对抗过拟合**：小网络的函数逼近偏差 > 大网络的过拟合伤害
- **控制 gradient steps / UTD ratio**：每收集 1 步数据做几步梯度更新，这个比例很关键
- **Replay buffer 要大**：它的核心价值是提供更多有效训练样本，减少单步过拟合

---

## 五、关键发现 3：分布漂移没那么可怕

### 5.1 实验设置

用 Exact-FQI + 256×256 网络，测试不同的加权分布 $\mu$：
- Uniform(s,a)：均匀加权所有状态-动作对
- $\pi(s,a)$：当前策略的 on-policy 分布
- $\pi^*(s,a)$：最优策略的分布
- Random(s,a)：随机策略的分布
- Replay(s,a)：所有历史策略的平均分布（模拟 replay buffer）
- Prioritized(s,a)：按 TD 误差大小加权

度量两个指标：
- **Distribution shift**：相邻迭代之间的分布 TV 距离
- **Loss shift**：分布变化后 Bellman Error 的变化量

### 5.2 结论

**分布漂移大 ≠ 性能差。** Prioritized 的分布漂移最大，但性能很好。On-policy 的分布漂移也不小，性能中规中矩。

真正重要的不是"分布是否在漂移"，而是**分布的覆盖面有多广**：

| 分布特征 | 性能 |
|---------|------|
| 高熵、覆盖广（如 Uniform、Replay） | 🟢 好 |
| 集中在高 TD 误差区域（Prioritized） | 🟢 好 |
| 集中在当前策略附近（On-policy） | 🟡 中等 |
| 集中在最优策略附近（$\pi^*$） | 🔴 较差 |

**核心洞察**：高覆盖率 > 低分布漂移。与其花力气纠正分布漂移（如 importance sampling），不如想办法让训练数据覆盖更多状态-动作对。

**对 VLA Critic 的启示**：
- 不需要过于担心 off-policy 数据"不匹配"——只要数据覆盖够广就行
- 混合多个来源的数据（不同策略、不同任务）训 Critic 可能比只用当前策略数据更好
- VLAC 用人类视频 + 机器人轨迹 + 负样本混合训练是有道理的

---

## 六、Moving Target 也没那么可怕

作者还测试了 $\alpha$-smoothed Bellman backup（类似 soft target update）：

$$
\mathcal{T}_\alpha Q = \alpha \cdot \mathcal{T}Q + (1-\alpha) \cdot Q
$$

**这个公式在做什么**：$\alpha=1$ 是标准 hard update，$\alpha < 1$ 让 target 变化更慢（相当于 Polyak averaging）。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 |
|------|------|
| $\mathcal{T}Q$ | 标准 Bellman backup 结果 |
| $\alpha$ | target 更新速度，$\alpha=1$ 最快 |
| $1-\alpha$ | 保留旧 Q 值的比例 |

收缩率从 $\gamma$ 变为 $1 - \alpha + \alpha\gamma$。如 $\alpha=0.1$, $\gamma=0.99$：收缩率 = $0.9 + 0.1 \times 0.99 = 0.999$。意味着收敛速度大幅下降。

**实验结论**：大网络下 $\alpha=1$（最快 target 变化）性能最好。只有很小的网络（4×4, 16×16）才从慢 target 中获益。
:::

**结论**：target 变化快本身不造成不稳定——不稳定的原因是**网络容量不足以跟上变化的 target**。

---

## 七、本文提出的改进方法：Adversarial Feature Matching (AFM)

基于"高覆盖分布最好"+ "应该重点关注函数逼近误差大的区域"这两个洞察，作者提出了 AFM：

**核心思想**：设计一个对抗性的采样权重 $p_\phi(s,a)$，既要最大化 Bellman Error（关注误差大的地方），又要保证加权后的特征分布和 replay buffer 整体特征分布相近（保持高覆盖）。

这个方法在 TD3 和 SAC 上都有提升，但更重要的是它**验证了作者的诊断结论**：改善采样分布的覆盖性确实能提升 Critic 训练质量。

---

## 八、总结：对 Critic 训练的实操指导

本文的三大 takeaway：

| 直觉假设 | 实验结论 | 实操建议 |
|---------|---------|---------|
| "函数逼近会导致 Q 发散" | 大网络几乎不发散，且 bootstrap 偏差小 | **用大网络**，不要因为"怕发散"而缩小 Critic |
| "过拟合不是 RL 的主要问题" | 过拟合是主要瓶颈之一，尤其是数据少时 | **控制 gradient steps**，增大 replay buffer，考虑 early stopping |
| "分布漂移导致训练不稳定" | 分布漂移和性能没有强相关 | **追求高覆盖**而非低漂移，混合多源数据 |

### 本文局限

- 表格环境的结论能否完全迁移到高维 VLA 场景尚未验证
- 没有研究过估计偏差（后来的 TD3 / Double Q 系列论文补上了）
- 没有研究可塑性丧失（后来的 Primacy Bias / Plasticity Loss 系列补上了）

---

## 延伸阅读

- [Deep RL and the Deadly Triad](https://arxiv.org/abs/1812.02648) — 与本文同期，侧重发散条件的实验分析
- [Stop Regressing: Training Value Functions via Classification](/论文综述/101_StopRegressing_分类式Critic训练) — 2024 年解决 "MSE 训 Critic 不稳定"的方案
- [The Primacy Bias in Deep RL](/论文综述/102_PrimacyBias_深度RL的初始偏差) — 2022 年发现的 Critic 过拟合早期经验现象
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — 本文结论在 VLA 场景的应用
