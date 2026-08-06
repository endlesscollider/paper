---
title: Chunked RL：分块离线 RL 高效微调 VLA 模型
order: 278
tags: [强化学习, 离线RL, 动作分块, VLA, 微调, TD学习]
category: 精读
star: 4
---

# Chunked RL：分块离线 RL 高效微调 VLA 模型深度精读

> **论文标题**: Efficient Fine-Tuning of Vision-Language-Action Models through Chunked Offline Reinforcement Learning
> **作者**: Ammar Husain, Kanishka Rao, Jie Tan, Fei Xia, Quan Vuong, Karol Hausman
> **机构**: Google DeepMind
> **发表**: arXiv:2508.02219, 2025

**标签**: `#强化学习` `#离线RL` `#动作分块` `#VLA` `#微调` `#TD学习` `#π₀`

---

## 相关阅读

在阅读本文前，建议先了解以下前置知识：

- [Q-Chunking：用动作分块加速离线到在线 RL](./071_QChunking_RL与动作分块) — 本文核心理论基础，将 Q-Chunking 迁移到 VLA
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — TD 学习基础
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — 离线 RL 设定
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — π₀ 的动作生成范式
- [动作 Token 化与自回归策略](/前置知识/000l_前置知识_动作Token化与自回归策略) — VLA 输出动作块的机制

关联文章：

- [π₀：通用机器人基础模型](./014_Pi0_通用机器人基础模型) — Chunked RL 微调的目标 VLA 模型
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — VLA RL 微调全景
- [Action-Chunked PPO + Self-BC](./030_ActionChunkedPPO_自行为克隆VLA后训练) — 另一种 VLA 分块 RL（on-policy PPO 路线）
- [CO-RFT：离线分块 RL 微调 VLA](./021_CO_RFT_离线分块RL微调VLA) — 相关工作

---

## 贯穿全文的例子

> **场景**：一个装载了 π₀ VLA 模型的双臂机器人需要学会"把洗好的碗碟从水槽移到碗架上"。π₀ 通过预训练已经会做很多基础操作（抓取、放置），但对这个特定任务的成功率只有 40%——主要失败在"碟子的角度没对准碗架的卡槽"这种精细调整上。
>
> 你有 200 条真实机器人执行这个任务的离线轨迹（有成功也有失败），想用 RL 把成功率从 40% 提升到 80%+，但**不能再做额外的在线交互**（真实机器人太贵）。

---

## 一、背景与动机

### 1.1 VLA 模型的"最后一公里"问题

大型 VLA 模型（如 [π₀](./014_Pi0_通用机器人基础模型)、RT-2、OpenVLA）通过大规模预训练获得了广泛的操作能力，但在特定任务上往往表现不够好。原因很直接：预训练数据来自多种任务的混合，不可能对每个具体任务都有最优的覆盖。

用 RL 微调是提升特定任务性能的自然选择。但 VLA 模型有一个特殊性：**它们原生输出的就是动作块（action chunk）**——不是一步一步的单步动作，而是一次性预测未来 16-50 步的完整动作序列。这个特性既是挑战也是机遇：

- **挑战**：传统 RL 算法（SAC、PPO 等）都是为单步动作设计的，直接套用需要适配
- **机遇**：[Q-Chunking](./071_QChunking_RL与动作分块) 已经证明，动作分块可以在 TD 学习中带来无偏 $n$ 步加速和更好的探索——VLA 的原生分块特性正好可以直接利用

### 1.2 为什么选择离线 RL 而不是在线 RL

在线 RL 微调 VLA 面临几个现实障碍：

1. **计算成本**：VLA 模型巨大（几十亿参数），每步推理需要几百毫秒，on-policy 数据收集极慢
2. **安全风险**：探索阶段的随机动作可能损坏机器人或环境
3. **数据已有**：很多场景下，从之前的部署/评测中已经有足够的离线轨迹

Chunked RL 的目标是：**用已有的离线数据 + 分块 TD 学习，在不做任何在线交互的情况下提升 VLA 的任务表现**。

### 1.3 和 Q-Chunking 的关系

Chunked RL 本质上是把 [Q-Chunking 的核心理论](./071_QChunking_RL与动作分块)（分块 Critic + 无偏 $n$ 步 backup）搬到 VLA 微调的具体场景中，并解决以下 VLA 特有的工程问题：

- VLA 的动作块长度是固定的（如 π₀ 固定输出 16 步），不像 Q-Chunking 原文里 chunk 长度是可调超参数
- VLA 模型本身既是策略网络又是 BC 教师——不需要单独训练一个 $f_\xi$ 来拟合行为分布
- VLA 的参数量极大，RL 微调需要参数高效的方法（如 LoRA）

---

## 二、Chunked RL 的方法

### 2.1 核心思路：直接在 VLA 的原生动作块上做 TD 学习

Chunked RL 的方法结构和 Q-Chunking 几乎完全对应，只是把每个组件换成了 VLA 适配的版本：

| Q-Chunking 的组件 | Chunked RL 的对应 |
|-------------------|------------------|
| Flow-matching 行为策略 $f_\xi$ | VLA 模型本身（冻结或 LoRA） |
| 分块 Critic $Q_\theta(s, \mathbf{a}_{t:t+h})$ | 在 VLA 之外单独训练的 Critic head |
| Best-of-$N$ 采样 | 从 VLA 采样多个候选动作块 + Critic 挑选 |
| chunk 长度 $h$（可调） | VLA 的固定输出长度（如 16 步） |

### 2.2 分块 TD 目标

训练 Critic 的 loss 直接沿用 Q-Chunking 的分块 TD loss：

$$
L(\theta) = \mathbb{E}_{(s_t, \mathbf{a}_{t:t+h}, r_t^h, s_{t+h}) \sim \mathcal{D}}\left[\Big(Q_\theta(s_t, \mathbf{a}_{t:t+h}) - r_t^h - \gamma^h Q_{\bar\theta}(s_{t+h}, \mathbf{a}^\star_{t+h})\Big)^2\right]
$$

这里 $\mathbf{a}^\star_{t+h}$ 是在 $s_{t+h}$ 处从 VLA 采样 $N$ 个候选块后用 Critic 挑出的最优块。公式的含义和 [Q-Chunking 精读第 4.2.4 节](./071_QChunking_RL与动作分块#4.2.4-qc-的完整-td-loss) 完全一致，不再重复。

**VLA 特有的一个关键区别**：Q-Chunking 原文的 $f_\xi$ 是一个从零训练的小型 Flow 网络（几百万参数），采样速度快。但 VLA 模型（如 π₀）有几十亿参数，每次采样一个动作块需要几百毫秒的推理。$N = 32$ 意味着 32 次完整 VLA 推理——这在训练循环中代价极高。

Chunked RL 的解决方案是**降低 $N$**（通常取 4-8），同时利用 VLA 本身质量高的特点——因为 VLA 经过大规模预训练，它采样出的候选动作本身就比随机初始化的小型 Flow 网络好得多，不需要那么多候选来"碰运气"。

### 2.3 VLA 策略更新：冻结 VLA + RL Critic 选择

Chunked RL 的一个重要设计决策是**不直接用 RL 梯度更新 VLA 的参数**（或只用 LoRA 做极小幅度的更新）。原因是：

1. VLA 的预训练知识非常宝贵，RL 的 Critic 梯度可能破坏它（灾难性遗忘）
2. 离线数据量有限（几百条轨迹），大模型容易过拟合

策略改进主要通过"Best-of-$N$ 选择"实现——VLA 生成多个候选，Critic 挑最好的执行。这和 Q-Chunking 的 QC 方案（4.2 节）完全一致：不修改策略网络本身，而是通过"采样+挑选"隐式地提升决策质量。

---

## 三、VLA 微调的特殊挑战与解决方案

### 3.1 稀疏奖励下的 Critic 训练

VLA 任务的奖励通常极度稀疏——只有"任务完成"才给正奖励。在离线数据只有 200 条轨迹（其中可能只有 80 条成功）的情况下，Critic 的训练信号非常稀疏。

Chunked RL 用分块 TD backup 来缓解这个问题——每次 backup 跨越一整个 chunk（如 16 步），价值信号传播速度是单步 TD 的 16 倍。这意味着在同样的训练步数下，离轨迹成功终点 160 步远的状态也能获得有意义的价值估计（单步 TD 需要 160 次 backup，分块 TD 只需要 10 次）。

### 3.2 离线设定下的行为约束

和 Q-Chunking 一样，纯离线设定需要行为约束防止 Critic 对 OOD 动作过度自信。Chunked RL 的约束天然内置在"Best-of-$N$ 从 VLA 采样"的结构中——所有候选动作都来自 VLA 的分布，不会出现完全 OOD 的动作。

但离线 RL 有一个额外的问题：bootstrap 项 $Q_{\bar\theta}(s_{t+h}, \mathbf{a}^\star_{t+h})$ 可能评估的是"VLA 在 $s_{t+h}$ 会怎么做"，但训练数据中 $s_{t+h}$ 可能是旧策略执行了不同动作后到达的——这就是标准的分布偏移问题。Chunked RL 通过保守的 ensemble Critic（类似 [RLPD](./075_RLPD_高效在线RL利用离线数据) 的设计）来缓解这个问题。

---

## 四、实验结果

### 4.1 评测设定

- **VLA 模型**：π₀（Google DeepMind 的 Flow-matching VLA，输出 16 步动作块）
- **任务**：多种真实机器人操作任务（清洁、搬运、堆叠）
- **数据**：每个任务 100-500 条离线轨迹（真实机器人数据）
- **对比方法**：
  - Filtered BC（只用成功轨迹做行为克隆）
  - IQL + VLA fine-tuning
  - 标准单步 TD（不分块的离线 RL）
  - 朴素 $n$ 步回报

### 4.2 核心发现

1. **分块 TD 显著优于单步 TD**：平均成功率提升 15-25%，验证了 Q-Chunking 的核心理论在 VLA 场景中同样成立

2. **分块 TD 优于朴素 $n$ 步回报**：朴素 $n$ 步回报的有偏性在实际中确实伤害性能——和 Q-Chunking 原文的发现一致

3. **Best-of-$N$ 选择有效**：即使 $N$ 只有 4（比 Q-Chunking 原文的 32 小很多），结合 VLA 高质量的基础能力，选择效果已经很好

4. **不需要大量数据**：200 条离线轨迹就能带来显著提升——这对真实机器人场景很实用

---

## 五、和其他 VLA RL 方法的对比

| 方法 | RL 范式 | 是否需要在线交互 | 动作分块 | 适用 VLA 类型 |
|------|---------|---------------|---------|-------------|
| **Chunked RL（本文）** | 离线 TD | 否 | ✓（原生利用） | Flow-based VLA (π₀) |
| [Action-Chunked PPO](./030_ActionChunkedPPO_自行为克隆VLA后训练) | 在线 PPO | 是 | ✓ | 自回归 VLA |
| [FlowRL](./018_FlowRL_Flow_VLA的在线RL微调) | 在线 PPO | 是 | 部分 | Flow-based VLA |
| [CO-RFT](./021_CO_RFT_离线分块RL微调VLA) | 离线回报加权 | 否 | ✓ | 通用 VLA |
| [RLPD](./075_RLPD_高效在线RL利用离线数据) + VLA | 在线 SAC | 是 | ✗ | 通用 |

**Chunked RL 的定位**：纯离线 + 分块 TD 的组合。不需要在线交互（比 on-policy 方法实用），同时利用了分块 TD 的理论优势（比纯 BC 或回报加权方法样本效率更高）。

---

## 六、核心优势与局限

### 优势

1. **理论基础扎实**：直接继承 Q-Chunking 的无偏 $n$ 步 backup 理论
2. **纯离线**：不需要额外的在线交互，适合真实机器人场景
3. **原生适配 VLA**：利用了 VLA 天然输出动作块的特性，不需要改模型结构
4. **数据高效**：几百条轨迹就能有效提升
5. **不破坏预训练知识**：冻结/轻量微调 VLA，避免灾难性遗忘

### 局限

1. **Best-of-$N$ 采样慢**：VLA 的推理成本高，$N$ 次采样在部署时可能不可接受
2. **离线数据质量依赖**：如果离线数据中成功轨迹太少，Critic 学不到有效信号
3. **固定 chunk 长度**：继承了 Q-Chunking 原文"chunk 长度不可调"的局限（AQC 可能可以解决）
4. **只在 π₀ 验证**：泛化到其他 VLA（如 OpenVLA 的自回归式输出）需要额外适配

---

## 七、总结

| 维度 | Chunked RL |
|------|-----------|
| 核心问题 | 怎么用有限的离线数据通过 RL 提升 VLA 在特定任务上的表现 |
| 核心方案 | Q-Chunking 理论 + VLA 原生动作块 → 分块离线 TD 学习 |
| 和 Q-Chunking 的关系 | 把 Q-Chunking 的理论直接迁移到 VLA 微调场景，解决 VLA 特有的工程问题 |
| 策略更新方式 | 冻结 VLA + Best-of-$N$ Critic 选择（不直接更新 VLA 参数） |
| 实验结论 | 分块 TD 在 VLA 场景中同样显著优于单步 TD 和朴素 $n$ 步回报 |
| 最大意义 | 证明 Q-Chunking 的理论优势不限于小型实验环境，在真实 VLA 规模上依然成立 |

---

## 延伸阅读

- [Q-Chunking：用动作分块加速离线到在线 RL](./071_QChunking_RL与动作分块) — 核心理论基础
- [π₀：通用机器人基础模型](./014_Pi0_通用机器人基础模型) — 被微调的 VLA 模型
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — 更广泛的 VLA RL 后训练生态
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — π₀ 的动作生成范式
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — 离线 RL 背景
- Husain et al., "Efficient Fine-Tuning of VLA Models through Chunked Offline RL", arXiv:2508.02219, 2025
