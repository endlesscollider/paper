---
title: Action-Chunked PPO + Self Behavior Cloning 的 VLA 后训练
order: 230
tags: [强化学习, VLA, PPO, Action Chunking, Self-BC, MetaWorld]
category: 精读
star: 3
---

# Action-Chunked PPO + Self-BC：VLA 后训练深度精读

> **论文标题**: VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning  
> **作者**: Anonymous  
> **机构**: TBD  
> **发表**: arXiv:2509.25718, 2025  

**标签**: `#VLA` `#强化学习` `#PPO` `#ActionChunking` `#SelfBC` `#MetaWorld`

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 核心机制
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — BC + RL 混合训练
- [动作 Token 化与自回归策略](/前置知识/000l_前置知识_动作Token化与自回归策略) — 动作 chunk 表示
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — 防止 RL 崩溃
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — VLA + RL 全景图

---

## 一、背景与动机

### 1.1 VLA RL 后训练的两大痛点

**痛点一：时间一致性差**

VLA 逐步生成动作时，相邻步骤之间可能不一致（突然跳变）。RL 探索进一步加剧这个问题：

$$
a_t = [0.1, 0.2, 0.3], \quad a_{t+1} = [-0.5, 0.8, -0.2] \quad \text{(突变！)}
$$

在真实机器人上，这种突变会导致急停、抖动，甚至损坏设备。

**痛点二：奖励信号利用不充分**

标准 PPO 每步一个 reward → 多数时间 reward=0 → 梯度信号极弱。

### 1.2 本文的两个解法

1. **Action Chunking PPO**：将 $k$ 步动作打包成 chunk，作为一个"super-action" 来优化 → 提高时间一致性 + 奖励密度
2. **Self Behavior Cloning (Self-BC)**：用策略自己产生的成功轨迹做 BC 辅助训练 → 稳定探索

---

## 贯穿全文的例子

> **场景**：VLA 在 MetaWorld 基准上做 50 个操作任务（如 "open drawer"、"push button"）。
>
> - **Chunk size** $k=5$：每次决策输出未来 5 步的动作序列
> - **PPO on chunks**：对整个 5 步序列评估 advantage
> - **Self-BC**：训练过程中，一旦策略成功完成任务，就把该轨迹存入 "success buffer"，后续用 BC loss 正则化
> - **目标**：成功率 93%，平均步数 42 步

---

## 二、方法详解

### 2.1 Action Chunking PPO

将 VLA 的动作生成改为 chunk 模式：

$$
C_t = (a_t, a_{t+1}, \ldots, a_{t+k-1}) = \pi_\theta(o_t)
$$

PPO 在 chunk 级别计算 advantage：

$$
A_{\text{chunk}}(o_t) = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(o_{t+k}) - V(o_t)
$$

**一句话**：把 $k$ 步的累积奖励减去 baseline，得到 chunk 级 advantage。

**代入数字**：$k=5$，$\gamma=0.99$，假设这 5 步的 reward 为 $[0, 0, 0, 0, 1]$（第 5 步触发成功）：
- 累积奖励：$0 + 0 + 0 + 0 + 0.99^4 \times 1 = 0.961$
- Baseline：$V(o_t) = 0.5$（Value 估计）
- Chunk advantage：$0.961 + 0.99^5 \times 0 - 0.5 = +0.461$（正值，应当强化）

### 2.2 时间一致性保证

chunk 内部的动作通过以下方式保证平滑：

$$
\|a_{t+i} - a_{t+i-1}\| \leq \delta_{\max}, \quad i = 1, \ldots, k-1
$$

VLA 在生成 chunk 时，decoder 内部有 causal attention，前一个动作 token 会影响后一个 → 自然保持一致性。

### 2.3 Self Behavior Cloning (Self-BC)

核心思路：**用自己的成功经验来约束探索**。

**维护 Success Buffer $\mathcal{B}_{\text{success}}$**：

```
if episode is successful:
    add trajectory to B_success
```

**BC 辅助 loss**：

$$
\mathcal{L}_{\text{Self-BC}} = \mathbb{E}_{\tau \sim \mathcal{B}_{\text{success}}} \left[ -\log \pi_\theta(C_t | o_t) \right]
$$

**总 loss**：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + \beta \cdot \mathcal{L}_{\text{Self-BC}}
$$

**为什么用 Self-BC 而不是 KL to SFT init**：
- SFT init 的策略可能是次优的（成功率 70%）
- Self-BC 用的是"当前策略产生的最好轨迹"——随训练进行质量不断提高
- 效果：既防止退化，又不限制改进

**类比**：KL to init 像"不要忘记课本上教的"。Self-BC 像"记住你自己做对过的案例"——后者允许你超越课本。

---

## 三、实验结果

### 3.1 MetaWorld 50 Tasks

| 方法 | 成功率 | 平均步数 | 训练稳定性 |
|------|--------|---------|-----------|
| SFT baseline | 72% | 65 | - |
| PPO (per-step) | 85% | 50 | ⚠️ 偶尔崩溃 |
| PPO (chunked, no Self-BC) | 89% | 45 | ✅ |
| **PPO (chunked) + Self-BC** | **93%** | **42** | **✅ 最稳定** |

### 3.2 消融：Chunk size 的影响

| Chunk $k$ | 成功率 | 时间一致性 | 响应性 |
|-----------|--------|-----------|--------|
| 1 (per-step) | 85% | 差 | ✅ 高 |
| 3 | 90% | 中 | 中 |
| **5** | **93%** | **好** | **中** |
| 10 | 91% | 很好 | ❌ 差（反应迟钝） |

$k=5$ 是最佳平衡点：足够的时间一致性，又不会太迟钝。

### 3.3 Self-BC 的动态效果

| 训练阶段 | Success Buffer 大小 | Self-BC 贡献 |
|---------|-------------------|-------------|
| 早期 (0-100 步) | 10 条 | 小（数据少） |
| 中期 (100-300 步) | 50 条 | 大（质量提升） |
| 后期 (300+ 步) | 200 条 | 持续稳定 |

Success Buffer 随训练不断充实，Self-BC 的正则化效果越来越强。

---

## 四、与相关工作的对比

| 方法 | Chunking | BC 正则化 | 特点 |
|------|----------|----------|------|
| VLA-RL | ❌ per-token | KL to init | 标准 PPO |
| TGRPO | ✅ 轨迹级 | ❌ | 无 Critic |
| CO-RFT | ✅ chunk 级 | ✅ (AWR) | 离线 |
| **本文** | ✅ chunk 级 | ✅ (Self-BC) | 在线 + 自适应 BC |

本文的独特之处：**在线 PPO** + **动态质量的 BC 正则化**。

---

## 五、总结

| 维度 | Action-Chunked PPO + Self-BC |
|------|------------------------------|
| 核心创新 | Chunk 级 PPO + 自产成功轨迹做 BC 正则化 |
| RL 算法 | PPO (chunk-level) |
| 正则化 | Self Behavior Cloning（动态 buffer） |
| 关键效果 | 时间一致性好 + 训练稳定 + 高成功率 |
| 基准 | MetaWorld 50 tasks：93% SR |

---

## 延伸阅读

- [TGRPO：轨迹级 GRPO](./019_TGRPO_轨迹级GRPO微调VLA) — 另一种 chunk 级在线 RL
- [CO-RFT：离线分块 RL](./021_CO_RFT_离线分块RL微调VLA) — 离线 chunk RL
- [FORCE：高效 VLA RL](./026_FORCE_高效VLA_RL微调) — 另一种 PPO 训练加速方案
