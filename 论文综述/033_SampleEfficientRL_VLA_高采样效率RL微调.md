---
title: Sample-Efficient RL Finetuning for VLA：高采样效率 VLA RL 微调
order: 233
tags: [强化学习, VLA, 采样效率, 预训练先验, Actor-Critic, 轻量]
category: 精读
star: 3
---

# Sample-Efficient RL for VLA：高采样效率 VLA RL 微调深度精读

> **论文标题**: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models  
> **作者**: Anonymous  
> **机构**: TBD  
> **发表**: arXiv:2605.25477, 2025  

**标签**: `#VLA` `#强化学习` `#采样效率` `#预训练先验` `#ActorCritic` `#轻量`

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 算法
- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — Off-policy RL
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Value 基础
- [Replay Buffer](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — 数据重用
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — SFT → RL
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — 全景概览
- [BootRL 精读](./013_BootRL_冻结VLA加RL_Head) — 对比方法

---

## 一、背景与动机

### 1.1 现有方法的采样效率问题

| 方法 | 达到 80% SR 所需 rollouts | 原因 |
|------|--------------------------|------|
| VLA-RL (PPO) | ~5000 | On-policy, 数据只用一次 |
| SimpleVLA-RL | ~3000 | 更好的 PPO 工程 |
| TGRPO | ~2000 | GRPO 更高效 |
| BootRL | ~1500 | 只训小 head |
| **本文** | **~500** | 最大化利用预训练先验 |

**核心问题**：现有方法都没有充分利用 VLA 预训练中已经学到的知识。预训练给了我们一个不错的初始策略——RL 应该在此基础上做最小的修正，而不是从头摸索。

### 1.2 本文的三大贡献

1. **Pre-trained Prior Exploitation**：用 VLA 预训练特征初始化 Critic，大幅减少 Critic 收敛时间
2. **Adaptive Exploration**：根据预训练策略的置信度调整探索强度——高置信区域少探索，低置信区域多探索
3. **Hybrid Replay**：混合使用预训练数据和在线数据做 off-policy 训练

---

## 贯穿全文的例子

> **场景**：VLA 在 LIBERO 上做 RL 微调。
>
> - SFT 后成功率 65%——说明预训练已经学了不少
> - 传统 PPO 需要 5000 rollouts 才能到 85%
> - **本文方法**：
>   - Critic 用 VLA 的 hidden states 初始化 → 第 1 步就有好的 value 估计
>   - 对已经做对的部分（65% 的成功情况）减少探索 → 不去"修好的别弄坏"
>   - 用 SFT 数据填充 replay buffer → off-policy 重用
> - **结果**：只需 500 rollouts 就到 85%（10× 更高效）

---

## 二、方法详解

### 2.1 Pre-trained Prior Exploitation

**Critic 初始化**：不从随机网络开始，而是用 VLA backbone 的 hidden states 做 Critic 的输入特征：

$$
V_\phi(s) = \text{MLP}(h_{\text{VLA}}(s))
$$

其中 $h_{\text{VLA}}(s)$ 是冻结 VLA backbone 对状态 $s$ 产生的中间表示。

**为什么有效**：VLA 预训练已经学会了"什么状态离目标近"的表示——这正是 Value 函数需要学的东西。

**对比**：
- 随机初始化 Critic：需要 200+ rollouts 才能给出有用的 value 估计
- VLA-feature Critic：10 rollouts 就能给出合理估计

### 2.2 Adaptive Exploration

根据 VLA 预训练策略的 entropy 来调整探索噪声：

$$
\sigma_{\text{explore}}(s) = \sigma_0 \cdot \frac{H(\pi_{\text{SFT}}(s))}{H_{\max}}
$$

**逐项拆解**：
- $H(\pi_{\text{SFT}}(s))$ — 预训练策略在状态 $s$ 下的熵（高熵 = 不确定）
- $H_{\max}$ — 最大熵（归一化用）
- $\sigma_0$ — 基础探索强度
- $\sigma_{\text{explore}}(s)$ — 自适应后的探索噪声

**直觉**：
- 预训练策略很确定的状态（低熵）→ 少探索（它已经知道怎么做）
- 预训练策略不确定的状态（高熵）→ 多探索（需要更多数据来改进）

**代入数字**：$\sigma_0 = 0.3$，$H_{\max} = 2.0$
- 状态 A（VLA 很确定）：$H=0.3$ → $\sigma = 0.3 \times 0.3/2.0 = 0.045$（几乎不探索）
- 状态 B（VLA 不确定）：$H=1.8$ → $\sigma = 0.3 \times 1.8/2.0 = 0.27$（大量探索）

### 2.3 Hybrid Replay Buffer

混合三种数据来源：

$$
\mathcal{B} = \underbrace{\mathcal{D}_{\text{SFT}}}_{\text{预训练数据}} \cup \underbrace{\mathcal{D}_{\text{success}}}_{\text{成功 rollout}} \cup \underbrace{\mathcal{D}_{\text{online}}}_{\text{最近 rollout}}
$$

采样比例动态调整：

| 训练阶段 | SFT 数据 | 成功数据 | 在线数据 |
|---------|---------|---------|---------|
| 早期 (0-100 步) | 60% | 10% | 30% |
| 中期 (100-300 步) | 30% | 30% | 40% |
| 后期 (300+ 步) | 10% | 40% | 50% |

**为什么动态调整**：
- 早期：在线数据质量差（策略还不好），依赖 SFT 数据做 bootstrap
- 后期：策略改善了，成功数据越来越多，减少对旧 SFT 数据的依赖

### 2.4 Off-Policy 训练

使用 SAC 变体做 off-policy 训练（最大化数据复用）：

$$
\mathcal{L}_{\text{actor}} = \mathbb{E}_{s \sim \mathcal{B}} \left[ \alpha \log \pi_\theta(a|s) - Q_\phi(s, a) \right]
$$

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ (Q_\phi(s,a) - y)^2 \right]
$$

**Off-policy 的优势**：每条轨迹被重用多次 → 采样效率高。

---

## 三、实验结果

### 3.1 采样效率对比

| 方法 | 达到 85% SR 所需 rollouts | 达到 90% SR 所需 |
|------|--------------------------|------------------|
| VLA-RL (PPO) | 5000 | 8000+ |
| SimpleVLA-RL | 3000 | 5000 |
| BootRL | 1500 | 2500 |
| **本文** | **500** | **1000** |

本文方法的采样效率是 VLA-RL 的 **10×**，是 BootRL 的 **3×**。

### 3.2 各组件贡献（消融）

| 配置 | 达到 85% 所需 rollouts |
|------|----------------------|
| Full method | 500 |
| - Prior Critic | 1200 (+700) |
| - Adaptive Explore | 800 (+300) |
| - Hybrid Replay | 700 (+200) |
| All removed (= PPO) | 5000 |

Prior Critic 贡献最大（节省 700 rollouts），其次是 Adaptive Exploration。

### 3.3 真实机器人

| 任务 | 训练 rollouts | 成功率 |
|------|--------------|--------|
| Pick and place | 100 | 82% |
| Stack cubes | 150 | 75% |
| Open drawer | 80 | 88% |

在真实机器人上，100-150 条轨迹即可完成有效 RL 微调——使真实世界 RL 变得实际可行。

---

## 四、核心优势与局限

### 优势

1. **10× 采样效率提升**：充分利用预训练先验
2. **真实世界可行**：100 条轨迹够用
3. **理论优雅**：自适应探索有信息论支撑
4. **通用性**：可叠加到任何 VLA + RL 方法上

### 局限

1. **依赖好的预训练**：如果 SFT 性能太差（<40%），先验不可靠
2. **Off-policy 偏差**：hybrid replay 中旧数据可能与当前策略差异大
3. **超参多**：replay 比例、探索衰减等需要调节

---

## 五、总结

| 维度 | Sample-Efficient RL for VLA |
|------|----------------------------|
| 核心问题 | VLA RL 采样效率太低 |
| 核心方案 | Prior Critic + Adaptive Explore + Hybrid Replay |
| RL 算法 | SAC 变体 (off-policy) |
| 采样效率 | 10× better than PPO |
| 真实世界 | 100-150 rollouts 可行 |
| 本质思想 | 最大化利用预训练已有知识，RL 只做最小修正 |

---

## 延伸阅读

- [BootRL：冻结 VLA + RL Head](./013_BootRL_冻结VLA加RL_Head) — 类似利用 VLA 特征
- [FORCE：高效 VLA RL](./026_FORCE_高效VLA_RL微调) — 另一种训练加速方案
- [PLD：Residual RL](./015_PLD_Residual_RL自改进VLA) — Off-policy 路线的 Residual 版本
