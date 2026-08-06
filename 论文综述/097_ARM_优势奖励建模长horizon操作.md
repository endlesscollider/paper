---
title: ARM：优势奖励建模用于长 Horizon 操作
order: 297
tags: [离线RL, VLA, 奖励建模, advantage, 长horizon, 操作, 数据过滤]
category: 精读
star: 3
---

# ARM：优势奖励建模用于长 Horizon 操作深度精读

> **论文标题**: Advantage Reward Modeling for Long-Horizon Manipulation  
> **作者**: Anonymous  
> **发表**: arXiv:2604.03037, 2025  

**标签**: `#离线RL` `#VLA` `#奖励建模` `#advantage` `#长horizon` `#操作` `#数据过滤`

**知识链接**：
- [RECAP：从真实部署经验中 RL 学习](./016_RECAP_从真实部署经验中RL学习) — 同样训 value/advantage 再过滤数据训策略
- [AWR：优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) — advantage-weighted regression 的理论基础
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — filtered BC 的理论定位
- [IG-RFT：交互引导长 Horizon VLA RL](./034_IG_RFT_交互引导长horizon_VLA_RL) — 同为长 horizon 场景的 RL 方法

---

## 一句话概括

**训练一个 Advantage Reward Model 来给 long-horizon 操作轨迹的每一步打分（"这一步对任务完成有多大贡献"），然后在离线 RL pipeline 中用这个分数做自适应动作-奖励重加权，过滤次优样本——核心洞察是 advantage 比 binary reward 信息量大得多，尤其在 long-horizon 任务中。**

---

## 一、问题：Long-Horizon 操作的 Reward 难题

### 1.1 稀疏 Reward 在长 Horizon 下几乎无用

对于一个需要 200+ 步才能完成的 long-horizon 任务（如折叠衣服、组装零件），如果只在最后一步给 +1 reward：

- 前面 199 步都是 0 reward，策略梯度几乎没有信号
- "这一步做对了但最终失败"和"这一步完全瞎做"无法区分
- [RECAP](./016_RECAP_从真实部署经验中RL学习) 的 episode-length-based return 在短 horizon（<100 步）有效，但长 horizon 下分辨率不够

### 1.2 Dense Reward 需要大量工程

手工设计 dense reward 需要对每个任务的子步骤有深入理解，而且：
- 不同任务需要不同的 reward function
- 同一个任务的不同解法可能需要不同的中间 reward
- 容易引入 reward hacking

### 1.3 ARM 的思路：学一个通用的"进度打分器"

ARM 的核心想法：**不要手工设计 reward，而是从数据中学一个 advantage model——它能看一段轨迹，判断"这一步相比平均水平，对最终任务完成贡献了多少"**。

---

## 二、方法

### 2.1 Advantage Reward Model 的训练

ARM 使用**直觉性的训练信号**（不需要手工标注每一步的 reward）：

**训练数据来源**：
1. **成功轨迹 vs 失败轨迹**：同一个任务的成功和失败轨迹对比
2. **时序进度信号**：成功轨迹中越接近尾部的帧，"进度"越高
3. **DAgger 碎片**：人类纠正操作的片段——纠正前是"差"，纠正后是"好"

**训练目标**：让 ARM 学会给每一帧输出一个 scalar advantage 估计 $\hat{A}(s_t, a_t)$：

$$
\hat{A}(s_t, a_t) = \text{ARM}_\theta(s_t, a_t) \in \mathbb{R}
$$

**这个公式在做什么**：ARM 网络接收当前状态和动作，输出一个标量 advantage 估计——正值表示这个动作让任务有进展，负值表示在倒退。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $s_t$ | 时刻 $t$ 的状态 | 机器人观测（图像 + 本体感觉），维度取决于编码器 |
| $a_t$ | 时刻 $t$ 的动作 | 策略输出的动作向量，如 7-DoF 关节角增量 |
| $\text{ARM}_\theta$ | 参数为 $\theta$ 的 ARM 网络 | 一个回归网络，输入 (s, a) 对，输出 scalar |
| $\hat{A}$ | 估计的 advantage 值 | $> 0$ 表示比平均好，$< 0$ 表示比平均差 |
| $\in \mathbb{R}$ | 输出空间为实数 | 不限幅，典型范围约 $[-5, +5]$ |

**数值代入**：假设一个抓取任务：
- 机器人接近目标物体，ARM 输出 $\hat{A} = +1.8$（好动作，在推进任务）
- 机器人原地晃动，ARM 输出 $\hat{A} = -0.3$（略差，没有进展）
- 机器人远离目标，ARM 输出 $\hat{A} = -2.5$（坏动作，在倒退）

**为什么是这个形式**：直接输出实数而不是分类标签，是因为 advantage 本身是连续的——"好一点"和"好很多"对后续加权有不同意义。如果输出 binary（好/坏），就丢失了程度信息。
:::

### 2.2 ARM 如何区别于标准 Value Function

| 维度 | 标准 Value Function | ARM |
|------|-------------------|-----|
| 输入 | 状态 $s$ | 状态 $s$ + 动作 $a$ |
| 输出含义 | "从这个状态出发，未来总共能拿多少 reward" | "这一步动作，比平均动作好多少" |
| 训练信号 | 需要显式 reward | 只需要成功/失败标签 + 时序位置 |
| 在数据中 | 对碎片化 DAgger 数据不适用 | **天然支持碎片化数据** |

### 2.3 离线 RL Pipeline 中的应用

ARM 训好之后，插入离线 RL pipeline 做 **adaptive action-reward reweighting**：

$$
L_{\text{ARM-BC}} = \frac{1}{N}\sum_{i=1}^{N} \sigma(\hat{A}(s_i, a_i) / \tau) \cdot L_{\text{action}}(s_i, a_i)
$$

**这个公式在做什么**：这个 loss 让策略更多地学习 advantage 高的动作，少学或不学 advantage 低的动作——本质是一个 soft-filtered behavior cloning。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $N$ | batch 中样本数 | 一个 mini-batch 的大小，如 256 |
| $\hat{A}(s_i, a_i)$ | ARM 给第 $i$ 个样本打的 advantage 分 | 正 = 好动作，负 = 差动作 |
| $\sigma(\cdot)$ | sigmoid 函数 | 把 advantage 映射到 (0, 1) 作为权重 |
| $\tau$ | 温度参数 | 控制过滤的"硬度"：$\tau \to 0$ 退化为 binary filter |
| $L_{\text{action}}(s_i, a_i)$ | 第 $i$ 个样本的原始动作预测 loss | MSE（flow matching）或交叉熵（token） |

**各项梯度方向**：
- $\sigma(\hat{A}/\tau)$ 项：advantage 越高，权重越大 → 梯度主要来自"好样本"，策略被拉向好动作
- $L_{\text{action}}$ 项：标准 BC loss，让策略模仿数据中的动作

**数值代入**：假设 $\tau = 1.0$，batch 中有 3 个样本：
- 好动作 $\hat{A} = 2.0$：权重 $= \sigma(2.0/1.0) = \sigma(2.0) = 0.88$（高权重，多学）
- 一般动作 $\hat{A} = 0$：权重 $= \sigma(0/1.0) = \sigma(0) = 0.5$（正常权重）
- 差动作 $\hat{A} = -3.0$：权重 $= \sigma(-3.0/1.0) = \sigma(-3.0) = 0.05$（几乎不学）

假设三者的 $L_{\text{action}}$ 分别为 0.8, 0.6, 1.2：

$$
L_{\text{ARM-BC}} = \frac{1}{3}(0.88 \times 0.8 + 0.5 \times 0.6 + 0.05 \times 1.2) = \frac{1}{3}(0.704 + 0.3 + 0.06) = 0.355
$$

对比无加权的均匀 BC loss：$\frac{1}{3}(0.8 + 0.6 + 1.2) = 0.867$。加权后差动作的贡献从 0.4 降到了 0.02。

**为什么是这个形式**：用 sigmoid 而不是 hard threshold（直接丢弃负 advantage 样本），是因为 soft weighting 让梯度更平滑、训练更稳定。温度 $\tau$ 提供了一个旋钮：实际使用中 $\tau=0.5 \sim 2.0$ 范围内调。
:::

---

## 三、和 RECAP 的详细对比

| 维度 | RECAP | ARM |
|------|-------|-----|
| Advantage 来源 | 训练 distributional value model → n-step advantage | 直接训练 advantage model（不经过 value） |
| Return 构造 | Episode length + 成功标签 | 不需要构造 return |
| 二值化 | Top-k indicator（0/1 硬过滤） | Sigmoid 软加权（连续） |
| 碎片化数据 | 不支持（需要完整 episode） | **天然支持**（DAgger 片段直接用） |
| Long-horizon 适应性 | Return 信号在 long-horizon 下分辨率低 | **专门为 long-horizon 设计** |
| 最终效果 | 真机 90%+ | 真机折叠 T 恤 83%（BC 8%） |

**ARM 相比 RECAP 的最大优势**：在 long-horizon（200+ 步）任务上信号更精细。RECAP 的 return 是 $-(L-i-1)/L_{\max}$——在 200 步轨迹中相邻两帧的 return 差异只有 $1/L_{\max} \approx 0.005$，非常微弱。ARM 直接预测 advantage，分辨率不受 horizon 长度影响。

---

## 四、实验结果

### 4.1 真机实验

| 任务 | Vanilla BC | ARM + Filtered BC | 提升 |
|------|-----------|-------------------|------|
| 折叠 T 恤（展平状态） | 8% | **83%** | +75 pp |
| 折叠 T 恤（褶皱状态） | 0% | **67%** | +67 pp |

这个提升幅度在所有离线方法中是最大的之一——说明 long-horizon 任务中，数据过滤的价值极大（原始数据中大量的失败/低效步骤严重拖累了 BC 的性能）。

### 4.2 核心消融

1. **ARM vs Binary Success Filter**：只按成功/失败过滤整条轨迹，效果远不如 ARM 的 per-step advantage 加权——因为即使成功轨迹中也有低效的步骤，即使失败轨迹中也可能有部分好动作
2. **ARM 可迁移性**：在一个任务上训好的 ARM，迁移到相似任务上仍然有效

---

## 五、局限性

1. **ARM 本身需要训练数据**：需要至少一些成功轨迹来提供训练信号——如果任务太难初始完全没有成功数据，ARM 训不出来
2. **目前验证局限于布料操作**：其他类型任务（如 rigid body manipulation）的效果未充分验证
3. **没有迭代机制**：ARM 是"一次性训 advantage model → 一次性过滤训策略"，没有 RECAP 那样的多轮自我提升闭环

---

## 六、在方法谱系中的位置

ARM 处于 "advantage-based offline RL" 家族中，专门解决 long-horizon 的信号稀疏问题：

| 方法 | Horizon 适应性 | 碎片数据 | 迭代 |
|------|--------------|---------|------|
| AWR | 短-中 horizon | ❌ | ❌ |
| CRR | 短-中 horizon | ❌ | ❌ |
| RECAP | 短-中 horizon（return 分辨率受限） | ❌ | ✅ |
| **ARM** | **长 horizon（advantage 直接预测）** | **✅** | ❌ |
| Self-Improving EFM | 中 horizon | ❌ | ✅ |

---

## 七、总结

| 维度 | 要点 |
|------|------|
| 核心创新 | 训练 Advantage Reward Model 为 long-horizon 操作的每一步打分 |
| 训练信号 | 成功/失败对比 + 时序进度 + DAgger 纠正片段 |
| 应用方式 | Sigmoid-weighted BC（soft filter） |
| 最大优势 | Long-horizon 信号精度不受 horizon 长度稀释 + 支持碎片数据 |
| 实验亮点 | 折叠 T 恤从 8% → 83% |
| 局限 | 无迭代机制 + 验证任务类型有限 |

---

## 延伸阅读

- [RECAP 精读](./016_RECAP_从真实部署经验中RL学习) ← 迭代式 advantage-based 离线 RL
- [IG-RFT 精读](./034_IG_RFT_交互引导长horizon_VLA_RL) ← 另一个 long-horizon 解法（在线 RL）
- [AWR：优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) ← advantage-weighted 方法的理论基础
- [Self-Improving EFM 精读](./095_SelfImproving_EFM_自我提升具身基础模型) ← 同为 reward model 替代方案
- [CO-RFT：离线分块 RL 微调 VLA](./021_CO_RFT_离线分块RL微调VLA) ← 另一种离线 RL 后训练 VLA 方法
