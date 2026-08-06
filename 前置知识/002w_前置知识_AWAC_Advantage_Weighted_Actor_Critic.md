---
title: AWAC：优势加权 Actor-Critic（离线到在线 RL 桥接）
order: 42
tags: [强化学习, AWAC, 离线RL, 在线RL, AWR, offline-to-online, 策略优化]
category: 强化学习
---

# 前置知识：AWAC（Advantage Weighted Actor-Critic）

> **为什么要读这篇**：如果你知道 [AWR](/前置知识/000u_前置知识_AWR_优势加权回归)（纯离线的 advantage-weighted regression），AWAC 就是它的"在线扩展版"——先用离线数据做 AWR warmup，然后可以无缝切换到在线收集新数据继续改进策略。它是 **offline-to-online RL** 这条路线最简洁的实现之一，也是 [RLPD](/论文综述/075_RLPD_高效在线RL利用离线数据) 的直接对比基线。在 RECAP 和 Self-Improving EFM 的谱系中，AWAC 是"AWR → CRR → RECAP"这条演化线上的一个关键中间态。

**知识链接**：
- [AWR：优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) — AWAC 的离线阶段就是 AWR
- [SAC：Soft Actor-Critic](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — AWAC 的在线阶段用 SAC 风格的 Q 函数训练
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — AWAC 在离线 RL 方法中的位置
- [RLPD：高效在线 RL 利用离线数据](/论文综述/075_RLPD_高效在线RL利用离线数据) — 更简单但同样有效的 offline-to-online 方法

---

## 一、AWAC 要解决什么问题

### 1.1 两个极端的问题

| 极端 | 方法 | 问题 |
|------|------|------|
| 纯离线 | BC / AWR / CQL | 性能被数据质量封顶，无法超越数据中最好的行为 |
| 纯在线 | SAC / PPO（从零开始） | 初期探索极其低效，在真实机器人上可能需要几万步才开始学到东西 |

**理想情况**：先利用离线数据"热身"到一个还不错的策略，然后在线继续改进。

### 1.2 Naive Offline→Online 的问题

直觉上，可以先用离线 RL（CQL/IQL）预训练，再切到在线 SAC 继续训。但实践中会遇到：

1. **Q 值尺度失配**：CQL 把 Q 值压低了一个未知量，切到在线后策略梯度被这个偏移带歪
2. **需要两阶段代码**：离线和在线用不同的算法，切换点需要手动调
3. **遗忘**：在线阶段的梯度可能"忘掉"离线阶段学到的东西

### 1.3 AWAC 的解法：统一目标函数

AWAC 的核心优雅之处是：**离线和在线用同一个目标函数**，不需要切换算法。

---

## 二、核心公式

### 2.1 AWAC 的策略更新

$$
\pi_{\text{new}} = \arg\max_\pi \mathbb{E}_{(s,a) \sim \mathcal{B}}\left[\frac{\exp(A^{\pi_k}(s,a) / \lambda)}{Z(s)} \cdot \log \pi(a|s)\right]
$$

**这个公式在做什么**：让策略通过 advantage-weighted 最大似然来更新——和 BC 一样"模仿数据中的动作"，但用 advantage 当权重，好动作多学、差动作少学。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 | 典型值 |
|------|------|-----------|--------|
| $\mathcal{B}$ | 数据 buffer | 离线数据 ∪ 在线采集到的数据（同一个 buffer） | 所有可用数据 |
| $A^{\pi_k}(s,a)$ | 当前策略下的 advantage | $Q(s,a) - V(s)$ | 标量 |
| $\lambda$ | 温度参数 | 控制过滤的"硬度"。$\lambda$ 小 → 只学最好的 | 典型 0.1-1.0 |
| $Z(s)$ | 归一化常数 | $\mathbb{E}_{a' \sim \mathcal{B}(\cdot|s)}[\exp(A/\lambda)]$ | 让权重归一化 |
| $\log \pi(a|s)$ | 策略的对数似然 | 标准 BC loss | 标量 |

**数值代入**：假设 $\lambda = 0.5$，某状态下数据中有 3 个动作：
- $a_1$：$A = 2.0$ → 权重 $\propto \exp(2.0/0.5) = e^4 = 54.6$
- $a_2$：$A = 0.5$ → 权重 $\propto \exp(0.5/0.5) = e^1 = 2.7$
- $a_3$：$A = -1.0$ → 权重 $\propto \exp(-1.0/0.5) = e^{-2} = 0.14$

归一化后：$a_1$ 占 $54.6/57.4 = 95\%$ 的权重——策略几乎只学 $a_1$。

**为什么是这个形式**：用 $\exp(A/\lambda)$ 而不是直接用 $A$ 做权重，是因为 advantage 可能为负（负数不能当权重），取指数后保证权重恒正。$\lambda$ 控制"筛选硬度"——$\lambda \to 0$ 退化为只模仿最优动作，$\lambda \to \infty$ 退化为普通 BC。
:::

### 2.2 Q 函数更新

Q 函数用标准的 SAC 风格 TD learning 更新（一个 Q loss + 一个 target network），数据来自同一个混合 buffer $\mathcal{B}$。

### 2.3 统一的在线/离线流程

```python
# AWAC 主循环（离线和在线完全一样！）
buffer = OfflineDataset + OnlineBuffer  # 统一 buffer

for step in range(total_steps):
    # 1. 在线采集（如果在线阶段）
    if online_phase:
        a = policy.sample(s)
        s', r = env.step(a)
        buffer.add(s, a, r, s')
    
    # 2. 从统一 buffer 采 batch
    batch = buffer.sample(batch_size)
    
    # 3. 更新 Q 函数（标准 TD）
    update_critic(batch)
    
    # 4. 更新策略（advantage-weighted regression）
    advantages = Q(batch.s, batch.a) - V(batch.s)
    weights = exp(advantages / lambda)
    policy_loss = -weighted_mean(weights * log_prob(batch.a | batch.s))
    update_policy(policy_loss)
```

**关键点**：离线阶段就是上面这个循环但 `online_phase = False`（不往 buffer 里加新数据）；在线阶段 `online_phase = True`（开始往 buffer 里加新数据）。**算法本身没有任何变化**。

---

## 三、AWAC vs 其他方法

| 维度 | AWR | AWAC | CRR | RECAP |
|------|-----|------|-----|-------|
| 在线能力 | ❌ 纯离线 | ✅ 无缝在线 | ❌ 纯离线 | ⚠️ 迭代式（不是真正在线） |
| 权重函数 | $\exp(A/\lambda)$ | $\exp(A/\lambda)$ | binary 或 $\exp$ | binary indicator |
| Q 函数更新 | 可选（用 MC return 也行） | SAC 风格 TD | 标准 TD | Distributional value |
| buffer 策略 | 只有离线数据 | 离线 ∪ 在线混合 | 只有离线数据 | 每轮新采集替换 |

### AWAC 和 RLPD 的关系

[RLPD](/论文综述/075_RLPD_高效在线RL利用离线数据) 论文（2023）发现：**如果你的在线 RL 算法足够好（SAC + LayerNorm + 大 batch），直接把离线数据扔进 replay buffer 和在线数据混合训练，效果就和 AWAC 一样好甚至更好**——不需要 AWAC 的 advantage weighting。这说明 AWAC 的核心贡献可能不在于"advantage weighting"本身，而在于"统一 buffer + 持续训练"的框架设计。

---

## 四、AWAC 在 2025 年的意义

### 4.1 和 RECAP 的关系

RECAP 可以看作 "AWAC 的离线迭代版 + VLA 适配"：
- AWAC 的"在线采集新数据加入 buffer"→ RECAP 的"每轮迭代重新 rollout 采集"
- AWAC 的"$\exp(A/\lambda)$ 加权"→ RECAP 的"top-k 二值化 indicator"
- AWAC 的"SAC 风格 Q"→ RECAP 的"distributional value model + n-step advantage"

### 4.2 在 D4RL 上的表现

根据 Simple Ingredients for Offline RL (2024) 的研究：增大 AWAC 的网络后，它在 D4RL benchmark 上达到了接近 SOTA 的水平——说明"advantage-weighted regression + 大网络"这个简单组合本身就很强。

---

## 五、总结

| 维度 | 要点 |
|------|------|
| 核心思想 | Advantage-weighted regression，离线和在线用同一个目标函数 |
| 权重 | $\exp(A/\lambda)$（和 AWR 一样，温度 $\lambda$ 控制硬度） |
| 统一性 | 离线→在线无需切换算法，只是 buffer 里开始加新数据 |
| 和 AWR 的区别 | AWR 是离线 only；AWAC 加了在线采集能力 |
| 和 CRR 的区别 | CRR 用 binary filter；AWAC 用 exponential weight |
| 和 RECAP 的关系 | RECAP 是 AWAC 的"迭代离线 + VLA 适配 + distributional value"版本 |
| 适用场景 | 有离线数据 + 可以在线交互的场景 |

---

## 延伸阅读

- [AWR：优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) ← AWAC 的离线基础
- [SAC：Soft Actor-Critic](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) ← AWAC 的 Q 函数训练方式
- [CRR 精读](/论文综述/098_CRR_Critic正则化回归离线RL) ← 同期的离线 RL 方法，用 binary filter 而非 exponential weight
- [RLPD 精读](/论文综述/075_RLPD_高效在线RL利用离线数据) ← 证明了更简单的方法也能达到 AWAC 水平
- [RECAP 精读](/论文综述/016_RECAP_从真实部署经验中RL学习) ← AWAC 在 VLA 上的迭代式继承者
