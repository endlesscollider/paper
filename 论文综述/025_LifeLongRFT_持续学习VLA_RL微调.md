---
title: LifeLong-RFT：VLA 持续学习 RL 微调
order: 225
tags: [强化学习, VLA, 持续学习, Process Reward, Action Chunking, GRPO]
category: 精读
star: 4
---

# LifeLong-RFT：VLA 持续学习 RL 微调深度精读

> **论文标题**: Continual Learning VLA Models via Reinforcement Fine-Tuning  
> **作者**: Yuan Liu, Yucheng Xie, et al.  
> **机构**: Peking University, Tsinghua University  
> **发表**: arXiv:2602.10503, 2025  
> **项目页**: https://yuan-liu-lifelong-rft.github.io/

**标签**: `#VLA` `#强化学习` `#持续学习` `#MDPR` `#ActionChunking` `#GRPO` `#不遗忘`

**知识链接**：
- [GRPO](/前置知识/000m_前置知识_GRPO_Group_Relative_Policy_Optimization) — 基础 RL 算法
- [Process Reward Model](/前置知识/000n_前置知识_Process_Reward_Model) — 过程奖励机制
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — 对比方法
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — 防遗忘约束
- [动作 Token 化与自回归策略](/前置知识/000l_前置知识_动作Token化与自回归策略) — 动作表示
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — VLA + RL 全景图
- [TGRPO 精读](./019_TGRPO_轨迹级GRPO微调VLA) — 对比：轨迹级 GRPO

---

## 一、背景与动机

### 1.1 持续学习问题

VLA 模型的理想使用方式不是"训完就不动了"，而是**持续学习新任务**：

- Week 1：学会 "pick up cup"
- Week 2：学会 "open drawer"（同时保持 cup 能力）
- Week 3：学会 "pour water"（同时保持前两个任务）

但现有 RL 微调方法有严重的**灾难性遗忘**问题：学新任务时旧任务性能暴跌。

| 方法 | 新任务成功率 | 旧任务保持率 | 问题 |
|------|------------|------------|------|
| 全参数 RL (PPO) | 85% | 40% (-45%) | 严重遗忘 |
| LoRA RL | 78% | 55% (-30%) | 中度遗忘 |
| SFT replay | 70% | 75% (-10%) | 新任务学不好 |

### 1.2 LifeLong-RFT 的核心创新

两大创新：

1. **Multi-Dimensional Process Reward (MDPR)**：三维度过程奖励，为每个 action chunk 提供密集反馈
2. **Chunking-level On-policy RL**：chunk 级在线 RL，配合持续学习策略

**不依赖在线环境反馈**：MDPR 直接从轨迹本身计算奖励，不需要预训练的 reward model。

---

## 贯穿全文的例子

> **场景**：VLA 模型需要按顺序学会 LIBERO 的 10 个任务（Task 1 → Task 10），每次只有当前任务的 20% 训练数据。
>
> - **挑战**：学到 Task 5 时，Task 1-4 的成功率不能掉
> - **MDPR 的作用**：为每个 action chunk 从"进度"、"安全性"、"效率"三个维度打分
> - **目标**：10 个任务全学完后，平均成功率比纯 SFT 高 22%

---

## 二、方法详解

### 2.1 Multi-Dimensional Process Reward (MDPR)

MDPR 从三个维度为每个 action chunk 计算密集奖励：

$$
R_{\text{MDPR}}(c_i) = \alpha \cdot R_{\text{progress}}(c_i) + \beta \cdot R_{\text{safety}}(c_i) + \gamma \cdot R_{\text{efficiency}}(c_i)
$$

**维度一：进度奖励 $R_{\text{progress}}$**

衡量这个 chunk 让任务"前进了多少"：

$$
R_{\text{progress}}(c_i) = \text{sim}(o_{i+1}, o_{\text{goal}}) - \text{sim}(o_i, o_{\text{goal}})
$$

其中 $\text{sim}$ 是视觉编码器输出的 embedding 余弦相似度。

**直觉**：如果这个 chunk 执行后，画面变得更接近"任务完成"的样子，就给正奖励。

**维度二：安全性奖励 $R_{\text{safety}}$**

惩罚危险动作（如碰撞、超出工作空间）：

$$
R_{\text{safety}}(c_i) = -\lambda \cdot \mathbb{1}[\text{violation}(c_i)]
$$

**维度三：效率奖励 $R_{\text{efficiency}}$**

鼓励用更少的步数完成任务：

$$
R_{\text{efficiency}}(c_i) = -\mu \cdot \frac{1}{T} \cdot \|a_i\|^2
$$

用更小、更精准的动作完成任务 = 更高效。

**代入数字**：$\alpha=1.0, \beta=0.5, \gamma=0.1$，某 chunk：
- 进度：embedding 相似度从 0.3 提升到 0.45 → $R_{\text{progress}} = +0.15$
- 安全：无碰撞 → $R_{\text{safety}} = 0$
- 效率：动作幅度中等 → $R_{\text{efficiency}} = -0.02$
- 总奖励：$1.0 \times 0.15 + 0.5 \times 0 + 0.1 \times (-0.02) = 0.148$

### 2.2 Chunking-Level GRPO

在每个 chunk 级别应用 GRPO：

1. **采样**：对同一状态采样 $G$ 组 chunk
2. **评估**：用 MDPR 为每个 chunk 打分
3. **排序**：组内相对排序得到 advantage
4. **更新**：GRPO 梯度更新

### 2.3 持续学习策略

为防止灾难性遗忘，LifeLong-RFT 采用：

**1. EWC（弹性权重巩固）**

对重要权重施加二次惩罚：

$$
\mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta_i^*)^2
$$

$F_i$ 是 Fisher 信息矩阵对角元素——衡量参数 $\theta_i$ 对旧任务的重要性。

**2. 经验回放**

保留少量旧任务数据（每任务 10 条轨迹），混入新任务训练。

**3. KL 约束**

限制策略偏离参考策略的程度：

$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

---

## 三、实验结果

### 3.1 LIBERO 持续学习基准

按顺序学习 10 个任务后的平均成功率：

| 方法 | 平均成功率 | 新任务 | 旧任务保持 |
|------|-----------|--------|-----------|
| SFT (sequential) | 45% | 68% | 32% |
| EWC + SFT | 52% | 60% | 48% |
| LoRA SFT | 55% | 65% | 50% |
| **LifeLong-RFT** | **67%** | **78%** | **60%** |

**关键结果**：LifeLong-RFT 比纯 SFT 高 22%，且旧任务保持率远优于全参数 RL。

### 3.2 数据效率

使用 20% 训练数据的效果：

| 方法 | 100% 数据 | 20% 数据 | 性能保持 |
|------|----------|----------|---------|
| SFT | 62% | 45% | 73% |
| LifeLong-RFT | 67% | 61% | 91% |

RL 信号让模型能从少量数据中提取更多价值。

---

## 四、核心创新总结

| 组件 | 作用 | 替代方案 |
|------|------|---------|
| MDPR | 无需环境交互的密集奖励 | 需要在线环境 / 需要训练 reward model |
| Chunking-level RL | 减少信用分配难度 | 逐 token RL（稀疏奖励） |
| EWC + Replay + KL | 防止灾难性遗忘 | 无约束（严重遗忘） |

---

## 五、总结

| 维度 | LifeLong-RFT |
|------|--------------|
| 核心问题 | VLA 的持续学习 + RL 微调 |
| 核心创新 | MDPR 三维度过程奖励 + 持续学习策略 |
| RL 算法 | Chunking-level GRPO |
| 独特优势 | 不需要在线环境、不需要预训练 reward model |
| 性能 | 平均成功率 +22%（vs SFT） |
| 适用场景 | 需要持续学习新任务的真实部署场景 |

---

## 延伸阅读

- [TGRPO：轨迹级 GRPO 微调 VLA](./019_TGRPO_轨迹级GRPO微调VLA) — 单任务 GRPO 微调
- [CO-RFT：离线分块 RL 微调](./021_CO_RFT_离线分块RL微调VLA) — 另一种 chunk 级离线方法
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — 全景概览
