---
title: OGBench：离线目标条件 RL 基准
order: 276
tags: [强化学习, 离线RL, 目标条件RL, Benchmark, 评测]
category: 精读
star: 3
---

# OGBench：离线目标条件 RL 基准深度精读

> **论文标题**: Benchmarking Offline Goal-Conditioned RL
> **作者**: Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine
> **机构**: UC Berkeley
> **发表**: arXiv:2410.20092, 2024（NeurIPS 2024）
> **代码**: [github.com/seohongpark/ogbench](https://github.com/seohongpark/ogbench)

**标签**: `#强化学习` `#离线RL` `#目标条件RL` `#Benchmark` `#评测` `#长horizon`

---

## 相关阅读

在阅读本文前，建议先了解以下前置知识：

- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Goal-conditioned RL 中 Q 函数的条件化
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — 离线 RL 的通用设定与挑战
- [Replay Buffer 经验回放](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — 数据集结构基础

关联文章：

- [Q-Chunking：用动作分块加速离线到在线 RL](./071_QChunking_RL与动作分块) — 使用 OGBench 作为主要评测平台
- [Decoupled Q-Chunking](./074_DecoupledQChunking_解耦分块价值与执行) — 纯离线设定下在 OGBench 上评测
- [HIQL：分层离线目标条件 RL](./077_HIQL_分层离线目标条件RL) — OGBench 中的代表算法之一

---

## 贯穿全文的例子

> **场景**：一个多关节蚂蚁机器人在迷宫中导航。给定任何起点和终点，它需要找到路径走过去。训练数据来自一个"会走但不一定走最优路线"的随机漫步策略。这就是 OGBench 中 `antmaze` 系列任务的核心设定。

---

## 一、背景与动机

### 1.1 什么是 Offline Goal-Conditioned RL

**目标条件 RL (GCRL)** 的设定比标准 RL 更通用——策略不是追求固定的奖励函数,而是学习"到达任意指定的目标状态"。形式化地：

$$
\pi(a \mid s, g): \text{给定当前状态 } s \text{ 和目标 } g, \text{输出动作 } a
$$

**离线 GCRL** 进一步要求：只用一批预先收集的、**没有奖励标注**的轨迹数据来训练这个策略。奖励是自动推导的——通常定义为"每步 $-1$，到达目标时 $0$"（即最小化到达目标的步数）。

> **一句话直觉**：给一大堆无标注的"机器人到处走"的录像，让模型学会"从 A 走到 B"——不管 A 和 B 是什么位置。

**为什么这个设定重要**：在真实机器人场景中，采集大量无奖励的轨迹数据很便宜（让机器人随机探索即可），但为每个任务标注奖励函数很昂贵。GCRL 的优势在于——训练数据是**任务无关**的（不需要知道将来要到达哪个目标），但学出的策略是**任务有关**的（可以到达任意指定目标）。

### 1.2 为什么需要新的 Benchmark

在 OGBench 之前，离线 GCRL 领域的评测存在几个严重问题：

1. **D4RL AntMaze 被过度使用**：几乎所有论文都只在这一个 benchmark 上评测，而它只有 3-4 个变体，不足以揭示方法间的真正差异
2. **数据集质量不一致**：不同论文使用不同版本的 D4RL 数据，导致结果不可比
3. **任务种类单一**：大多是"导航到一个点"，缺少操作、拼图等需要更复杂规划能力的任务
4. **缺少轨迹拼接测试**：很多方法声称能做 trajectory stitching（把数据集里不同轨迹的片段组合成新路径），但缺少专门设计来测试这种能力的 benchmark

### 1.3 OGBench 的设计目标

OGBench 的目标是提供一个**系统化的、可控的、覆盖面广**的 benchmark，具体包含：
- 8 类不同难度的环境
- 85 个数据集（同一个环境有多种数据质量/分布的变体）
- 6 个代表性算法的参考实现（确保公平对比）

---

## 二、环境设计

### 2.1 八类环境概览

| 环境 | 状态空间 | 动作维度 | 核心挑战 | 任务描述 |
|------|---------|---------|---------|---------|
| PointMaze | 2D 位置+速度 | 2 | 基础导航 | 一个点在迷宫中移动到目标位置 |
| AntMaze | 29D 关节 | 8 | 高维导航 | 四足蚂蚁在迷宫中导航 |
| HumanoidMaze | 376D | 17 | 极高维 | 人形机器人导航 |
| Scene | 机械臂末端+物体位置 | 5-7 | 物体操作 | 移动桌面上的物体到目标位置 |
| Cube | 机械臂+方块位姿 | 5 | 精细抓取 | 抓起方块放到指定位置 |
| Puzzle (3×3) | 滑块拼图状态 | 离散化 | 组合搜索 | 解滑块拼图 |
| Stacking | 多个物体+手臂 | 5 | 多步操作 | 把方块叠起来 |
| Kitchen | 多物体交互 | 9 | 多任务 | 完成多个厨房子任务 |

### 2.2 数据集设计哲学

OGBench 为每个环境设计了多种**数据分布**变体，用于测试不同的能力：

**a) Navigate 数据**：随机策略走出的轨迹。测试的是"在轨迹之间拼接出新路径"的能力——数据里可能没有直接从 A 到 B 的轨迹，但有 A→C 和 C→B 的片段，方法需要把它们组合起来。

**b) Play 数据**：人类操作员随意操作产生的数据。分布更加多样但也更嘈杂。

**c) Sparse vs Dense 变体**：
- Sparse：只在起点和终点有标记，中间路径完全未知
- Dense：轨迹密度更高，覆盖更多中间状态

### 2.3 为什么 cube/scene 系列是 Q-Chunking 最能体现优势的环境

`cube-double`、`cube-triple`、`cube-quadruple` 这几个任务要求机械臂依次抓取 2/3/4 个方块放到指定位置。它们的特殊性在于：

1. **极度稀疏的奖励**：只有全部方块都到位才给 0，否则每步 $-1$
2. **超长 horizon**：完成 4 块需要 80-100+ 步连续正确操作
3. **需要精确的轨迹拼接**：数据里可能有"抓第 1 块"和"抓第 2 块"的片段，但没有"连续抓 1→2→3→4"的完整轨迹

这正是 Q-Chunking 动作分块最能发挥作用的场景——长 chunk 帮助 TD 跨越更多步传播价值信号，连贯的分块探索更容易触达奖励稀疏区域。

---

## 三、参考算法实现

OGBench 提供了 6 个代表性算法的统一实现：

| 算法 | 核心思路 | 价值估计方式 |
|------|---------|------------|
| GCBC | Goal-Conditioned BC | 无（纯模仿） |
| GCIVL | Goal-Conditioned IVL | Implicit Value Learning |
| GCIQL | Goal-Conditioned IQL | Implicit Q-Learning |
| QRL | Quasimetric RL | 用准距离度量到达难度 |
| CRL | Contrastive RL | 对比学习表示目标可达性 |
| HIQL | 分层 IQL | 高层子目标 + 低层执行 |

### 3.1 公平对比的工程保证

OGBench 的一个重要贡献是**统一了训练基础设施**：
- 所有方法用相同的网络宽度、学习率调度、训练步数
- 环境接口完全标准化
- 评估协议统一（每次评测用相同的起点-目标对集合）

这避免了"A 方法在论文里用了 3 层 MLP 而 B 方法用了 5 层，但声称 B 的算法更好"这种不公平对比。

---

## 四、关键实验发现

### 4.1 没有一个方法全面领先

OGBench 最重要的发现是：**没有任何一个现有算法在所有环境上都是最好的**。不同任务类型适合不同的方法：

- 简单导航（PointMaze）：大多数方法都能做到接近完美
- 高维导航（HumanoidMaze）：分层方法（HIQL）明显更好——因为高维状态空间中单步 TD 传播太慢
- 精细操作（Cube 系列）：需要精确的轨迹拼接能力，GCIQL 和 HIQL 各有优势
- 长 horizon（cube-quadruple）：所有方法都表现很差——这正是后续 Q-Chunking、DQC 等工作要攻克的难点

### 4.2 轨迹拼接能力是关键分水岭

OGBench 的实验清楚地表明：**能不能做 trajectory stitching 是区分好方法和差方法的核心能力**。GCBC（纯行为克隆）在数据覆盖好的任务上不错，但一旦目标位置在数据里没有直达轨迹，就彻底失败。而基于 TD 学习的方法（GCIQL、HIQL）能通过 Bellman backup 隐式地拼接不同轨迹片段。

---

## 五、OGBench 对后续工作的意义

### 5.1 为 Q-Chunking 提供实验平台

Q-Chunking（2025）和 DQC（2025）都以 OGBench 作为主要 benchmark，原因是：
- OGBench 的 cube/scene 系列任务天然适合测试"动作分块 + 长 horizon"的效果
- OGBench 的稀疏奖励设定让 TD 传播速度的差异暴露得更明显
- 统一的实现保证了和 HIQL 等 baseline 的公平对比

### 5.2 标准化了 offline GCRL 的研究范式

在 OGBench 之后，该领域的新工作（如 AQC、DQC、TRL 等）都自然地以 OGBench 为评测标准，结束了之前"每篇论文用不同 benchmark 不同设定"的混乱局面。

---

## 六、总结

| 维度 | OGBench |
|------|---------|
| 核心贡献 | 为 offline GCRL 提供系统化的 benchmark：8 类环境、85 数据集、6 算法参考实现 |
| 解决的问题 | 之前该领域 benchmark 碎片化、不公平对比、任务覆盖不足 |
| 关键发现 | 没有全面领先的算法；轨迹拼接能力是核心分水岭；长 horizon 稀疏奖励仍是开放难题 |
| 对后续工作的影响 | 成为 Q-Chunking、DQC、AQC 等动作分块 RL 方法的标准评测平台 |
| 设计哲学 | "好的 benchmark 不只是环境集合，更是一套公平对比的方法论" |

---

## 延伸阅读

- [Q-Chunking：用动作分块加速离线到在线 RL](./071_QChunking_RL与动作分块) — 使用 OGBench 的后续方法论文
- [Decoupled Q-Chunking](./074_DecoupledQChunking_解耦分块价值与执行) — 纯离线设定在 OGBench 上的 SOTA
- [HIQL：分层离线目标条件 RL](./077_HIQL_分层离线目标条件RL) — OGBench 的参考算法之一
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — 离线 RL 的通用背景
- Park et al., "Benchmarking Offline Goal-Conditioned RL", NeurIPS 2024
