---
title: The Primacy Bias：深度 RL 为什么过拟合早期经验
order: 302
tags: [强化学习, Critic, 可塑性, Primacy Bias, 周期性重置, 容量丧失]
category: 精读
star: 4
---

# The Primacy Bias：深度 RL 为什么过拟合早期经验

> **论文标题**: The Primacy Bias in Deep Reinforcement Learning
> **作者**: Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville
> **机构**: Mila, Université de Montréal
> **发表**: ICML 2022
> **arXiv**: [2205.07802](https://arxiv.org/abs/2205.07802)

**标签**: `#Critic训练` `#可塑性丧失` `#Primacy Bias` `#周期性重置` `#容量丧失` `#Replay Ratio`

**知识链接**：
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Critic 基础
- [Replay Buffer 经验回放](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — 过拟合与 replay 的关系
- [Diagnosing Bottlenecks 精读](/论文综述/100_DiagnosingBottlenecks_Q学习瓶颈诊断) — 过拟合在 Critic 训练中的角色
- [Stop Regressing 精读](/论文综述/101_StopRegressing_分类式Critic训练) — CE loss 缓解可塑性丧失
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 本文实验使用的算法

---

## 一、核心发现

**深度 RL 的 Critic 和 Actor 网络会过拟合训练最初期的数据，导致后续学到的经验"学不进去"——即使这些新经验更有价值。** 这个现象借用了认知科学的术语，称为 **Primacy Bias（初始偏差）**。

关键实验发现：
- 同一个任务，**只是推迟开始训练 10 万步**（先只收集数据不训练），最终性能可以提升 40-100%
- 原因：早期数据质量差（随机策略产生的），网络过拟合这些垃圾数据后就"僵了"
- **简单解法：周期性重置网络最后几层**——强制恢复可塑性

---

## 二、问题设置：高 Replay Ratio 下的退化

### 2.1 什么是 Replay Ratio (UTD)

Replay Ratio = 每收集 1 条新数据，从 buffer 中采样训练几次。又叫 Update-to-Data ratio (UTD)。

| UTD | 含义 | 代表算法 |
|-----|------|---------|
| 1 | 每条数据只训练一次 | SAC 默认 |
| 4 | 每条数据训练 4 次 | 一些 sample-efficient 方法 |
| 20 | 每条数据训练 20 次 | 极致 sample efficiency |

**常识直觉**：UTD 越高 → 更充分利用数据 → 性能应该更好。

**实际观察**：UTD 从 1 提到 4 确实更好，但继续提到 20 时性能**反而下降**。

### 2.2 为什么高 UTD 会伤害性能？

高 UTD 意味着网络在同一批数据上反复训练。训练初期，buffer 里主要是随机策略的经验（因为策略还没学好）。网络在这些"垃圾数据"上训了 20 遍 → **过拟合到早期数据的模式** → 后来收集到的好数据放进 buffer 后，网络已经学不动了。

这就是 Primacy Bias：**对先见到的数据形成过强的拟合，压制了后续学习能力。**

---

## 三、诊断实验：推迟训练 = 绕过 Primacy Bias

### 3.1 实验设计

在 DeepMind Control Suite 的多个连续控制任务上：
1. 标准训练：从第 0 步开始边收集数据边训练（UTD=20）
2. 延迟训练：先收集 10 万步数据（不训练），然后再开始训练

### 3.2 结果

延迟训练后性能大幅提升！例如在 Humanoid Walk：
- 标准 UTD=20：最终 return ≈ 200
- 延迟 100K 步后 UTD=20：最终 return ≈ 400（**+100%**）

**这证明了是"早期数据质量差"导致网络退化，而不是"UTD 太高本身有问题"。** 如果跳过最垃圾的数据，高 UTD 反而非常有效。

---

## 四、机制分析：网络发生了什么变化

### 4.1 有效秩下降

作者测量了 Critic 隐层特征矩阵的有效秩（用 SVD 奇异值的 entropy 衡量）：

- 标准训练：有效秩在训练初期**快速下降**——说明网络的表达多样性在丧失
- 延迟训练：有效秩下降更慢，最终保持在更高水平

**直觉**：网络把大量容量"锁定"在表示早期数据的模式上（如"随机动作 → 立刻失败"），剩余容量不够表示后来看到的复杂成功行为。

### 4.2 与 Critic 过估计的关系

Primacy bias 还会加剧 Q 值过估计：
- 网络过拟合早期（奖励低的）数据后，对后来看到的高奖励状态给出"虚高"的 Q 值——因为网络对这些状态的特征表示不够精确
- 这导致 Actor 去追逐虚高的 Q 值 → 策略退化 → 收集到更差的数据 → 恶性循环

---

## 五、解法：周期性重置（Periodic Reset）

### 5.1 方法

最简单的解法：每隔 $N$ 步，把网络的**最后几层**参数重新随机初始化。

```
每隔 reset_interval 步:
    随机初始化 Critic 最后 K 层权重
    随机初始化 Actor 最后 K 层权重
    保留所有层的 batch norm 统计量
    Replay buffer 不清空
```

**为什么只重置最后几层？** 前面的特征提取层（如卷积层）学到的低级视觉特征是通用的，不需要重置。最后几层负责"从特征到 Q 值的映射"——这部分最容易过拟合。

### 5.2 效果

| 任务 | 标准 SAC (UTD=20) | SAC + Periodic Reset (UTD=20) |
|------|-------------------|-------------------------------|
| Humanoid Walk | ~200 | **~550** |
| Quadruped Walk | ~300 | **~700** |
| Dog Walk | ~150 | **~400** |

周期性重置让高 UTD 的优势真正释放出来——几乎完全消除了 primacy bias。

### 5.3 和 Target Network 的关系

Target network 的 soft update 也是一种"防过拟合"机制（让 target 不会太快变化）。但它**不能解决** primacy bias——因为 online 网络本身已经僵化了，target 网络只是它的延迟复制。

Periodic reset 更激进：直接让网络"忘掉"过去，重新学习。代价是短期性能会暂时下降，但长期收益很大。

---

## 六、后续发展：从 Primacy Bias 到 Plasticity Loss

本文发表后引发了一系列后续工作（2023-2024），把 "Primacy Bias" 统一到更广泛的"可塑性丧失（Plasticity Loss）"框架下：

| 后续论文 | 核心贡献 |
|---------|---------|
| **Understanding Plasticity** (Lyle et al., ICML 2023) | 分析可塑性丧失和 loss landscape 曲率的关系 |
| **Plasticity Injection** (NeurIPS 2023) | 不用重置整层，而是"注入"新的随机神经元 |
| **Layer Norm** (Lyle et al., 2024) | 发现 LayerNorm + weight decay 组合能维持可塑性，不需要重置 |
| **BBF** (Schwarzer et al., ICML 2023) | 用 periodic reset + SR-SPR 让大网络在 Atari 100K 上达到超人 |
| **Plasticity Loss Survey** (arXiv:2411.04832) | 综述所有可塑性相关工作 |
| **Stop Regressing** (ICML 2024) | CE loss 天然缓解可塑性丧失（不需要显式重置） |

**有意思的联系**：Stop Regressing 论文中的非平稳实验（§5.2.3）实际上验证了本文的机制——CE loss 之所以好，部分原因正是它不容易像 MSE 那样在早期数据上"僵化"。

---

## 七、对 VLA Critic 训练的启示

1. **VLA 的 Primacy Bias 更严重**：VLA 的 SFT 预训练数据 = "早期经验"——Critic 在 SFT 数据上预训后，切换到 RL on-policy 数据时可能已经"僵了"
2. **FORCE 的 Value Warm-Up 本质上是在"消除 primacy bias"**：用 on-policy rollout 重新校准 Q 函数，让它忘掉 SFT 阶段的估值
3. **周期性重置 Critic head 可能有效**：保留 VLM backbone 的表征能力，只重置最后几层 value head
4. **高 UTD + 重置 = sample efficient**：对于昂贵的真实机器人数据，应该用高 UTD 充分利用每条轨迹，但配合重置防止退化
5. **HL-Gauss (CE loss) 是更温和的替代方案**：不需要显式重置也能维持可塑性

---

## 八、总结

| 问题 | 本文的发现 |
|------|-----------|
| 高 replay ratio 为什么反而有害？ | 因为网络过拟合早期（低质量）数据 |
| 如何诊断 primacy bias？ | 监控有效秩 / 延迟训练对比实验 |
| 最简单的解法？ | 周期性重置最后几层 |
| 根本原因？ | 神经网络在非平稳数据流上的可塑性丧失 |
| 和 Critic 过估计的关系？ | Primacy bias 导致特征表达退化 → 加剧 Q 值不准 |

---

## 延伸阅读

- [Plasticity Loss in Deep RL: A Survey](https://arxiv.org/abs/2411.04832) — 综述所有可塑性相关工作
- [Bigger, Better, Faster (BBF)](https://arxiv.org/abs/2305.19452) — 用 periodic reset 让大网络 value RL 工作
- [Dissecting Deep RL with High Update Ratios](https://arxiv.org/abs/2403.05996) — 深入分析高 UTD 下的崩溃机制
- [Stop Regressing 精读](/论文综述/101_StopRegressing_分类式Critic训练) — CE loss 天然缓解可塑性丧失
- [Diagnosing Bottlenecks 精读](/论文综述/100_DiagnosingBottlenecks_Q学习瓶颈诊断) — 过拟合是 Critic 训练的主要瓶颈
- [FORCE 精读](/论文综述/026_FORCE_高效VLA_RL微调) — VLA 中应对 Q 函数分布漂移（本质同源）
