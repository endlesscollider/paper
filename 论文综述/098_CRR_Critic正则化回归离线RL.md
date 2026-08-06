---
title: CRR：Critic 正则化回归——离线 RL 的加权/过滤式 BC
order: 298
tags: [离线RL, CRR, DeepMind, advantage, filtered BC, D4RL, 策略优化]
category: 精读
star: 3
---

# CRR：Critic 正则化回归——离线 RL 的加权/过滤式 BC 深度精读

> **论文标题**: Critic Regularized Regression  
> **作者**: Ziyu Wang, Alexander Novikov, Konrad Zolna 等（DeepMind）  
> **发表**: NeurIPS 2020  
> **arXiv**: 2006.15134  

**标签**: `#离线RL` `#CRR` `#DeepMind` `#advantage` `#filteredBC` `#D4RL` `#策略优化`

**知识链接**：
- [AWR：优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) — CRR 的直接前身，两者核心差异在 filter function 选择
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — CRR 在离线 RL 方法谱中的位置
- [CQL：保守 Q 学习](/前置知识/002g_前置知识_CQL保守Q学习) — 对比方法：悲观 Q 值 vs CRR 的 advantage 过滤
- [RECAP：从真实部署经验中 RL 学习](./016_RECAP_从真实部署经验中RL学习) — RECAP 的 indicator 机制是 CRR binary filter 的直系后代
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Advantage = Q − V

---

## 一句话概括

**把离线 RL 的策略优化转化为"用 Critic 打分过滤/加权后的行为克隆"——只克隆数据中 advantage > 0 的动作（或用指数权重放大好动作、压制差动作），完全避免了在 OOD 动作上做策略梯度带来的外推风险。**

---

## 一、CRR 要解决什么问题

### 1.1 离线 RL 的核心困难

离线 RL 面临的核心问题是 **OOD 动作的 Q 值外推误差**：

1. 用 Bellman equation 训练 Q 函数时，target 里有 $\max_{a'} Q(s', a')$
2. 这个 max 会把 $a'$ 推到数据中没见过的动作区域
3. Q 函数在没见过的区域给出不可靠的高估值
4. 策略追逐这些虚高的 Q 值 → 学出的策略在真实环境中完全不能用

### 1.2 已有解法的问题

| 方法 | 解法 | 问题 |
|------|------|------|
| CQL | 惩罚 OOD 动作的 Q 值 | 过于保守，性能被数据质量封顶 |
| BCQ | 只允许选数据中出现过的动作 | 需要训 VAE 来建模数据分布 |
| BEAR | 约束策略不能离数据分布太远 | MMD 约束复杂且不稳定 |
| AWR | 用 advantage 加权做 BC | 加权方式是启发式的，没有理论保证 |

### 1.3 CRR 的思路

**彻底放弃"在策略空间中做梯度优化"这条路**。不用策略梯度、不用 Q 值做 max——只做 **value-filtered regression**：

1. 先正常训一个 Q 函数（和标准 TD 一样）
2. 用 Q 函数计算每个数据中 $(s, a)$ 的 advantage $A(s, a) = Q(s, a) - V(s)$
3. **只对 advantage > 0 的 $(s, a)$ 做行为克隆**（binary filter 模式）

这样策略永远只在"数据中已经出现过的好动作"上学习，完全不存在 OOD 外推的问题。

---

## 二、核心公式

### 2.1 CRR 的策略优化目标

**Step 1：这个公式在做什么**

$$
L_{\text{CRR}}(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}}\left[f(A(s,a)) \cdot \log \pi_\theta(a|s)\right]
$$

**这个公式在做什么**：这个 loss 让策略通过加权最大似然学习数据中的动作——权重 $f(A)$ 由 advantage 决定，advantage 高的动作权重大（多学），advantage 低的动作权重小或为 0（不学）。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 | 典型值 |
|------|------|-----------|--------|
| $\mathcal{D}$ | 离线数据集 | 预先收集好的 $(s, a, r, s')$ transitions | 固定不变 |
| $A(s, a)$ | advantage | $Q(s, a) - V(s)$，正值 = 这个动作比该状态下的平均动作好 | 标量，范围约 $[-5, +5]$ |
| $f(\cdot)$ | filter/weight function | 把 advantage 转化为非负权重，两种选择见下 | 非负标量 |
| $\pi_\theta(a\|s)$ | 策略的动作概率 | 要训练的策略网络输出的动作分布 | 连续高斯 or 离散 softmax |
| $\log \pi_\theta(a\|s)$ | 对数似然 | 和 BC 完全一样的 loss 形式 | 标量，负值 |
| $-\mathbb{E}[\cdots]$ | 取负期望 | 因为要最大化似然，等价于最小化负对数似然 | 通过 mini-batch 近似 |

**数值代入**：假设 mini-batch 中有 3 个样本（binary filter 模式），$A_1 = 1.5$，$A_2 = -0.3$，$A_3 = 0.8$，对应 $\log\pi$ 值分别为 $-2.1$，$-1.8$，$-3.0$：

$$
L = -\frac{1}{3}\left[\mathbb{1}[1.5>0]\cdot(-2.1) + \mathbb{1}[-0.3>0]\cdot(-1.8) + \mathbb{1}[0.8>0]\cdot(-3.0)\right]
$$

$$
= -\frac{1}{3}\left[1\cdot(-2.1) + 0\cdot(-1.8) + 1\cdot(-3.0)\right] = -\frac{1}{3}(-5.1) = 1.7
$$

注意样本 2 因为 $A < 0$ 被完全过滤掉了，策略只从样本 1 和 3 学习。

**为什么是这个形式**：本质就是加权 BC loss。用 $f(A)$ 作为权重而不是直接用 $A$ 本身，是为了保证权重非负（负权重会让梯度方向反转，把好动作推远）。
:::

> **一句话直觉**：正常的 BC 是"数据里什么动作都学"；CRR 是"只学数据里比平均好的动作"。

### 2.2 Filter Function $f$ 的两种选择

CRR 提供了两种 $f$，对应两种不同的"过滤哲学"：

**选择 1：Binary Filter（硬过滤）**

$$
f_{\text{binary}}(A) = \mathbb{1}[A > 0]
$$

**这个公式在做什么**：一个 0/1 开关——advantage 为正就学（权重 = 1），advantage 为负或零就完全不学（权重 = 0）。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $\mathbb{1}[\cdot]$ | 指示函数 | 条件为真返回 1，为假返回 0 |
| $A$ | advantage $Q(s,a) - V(s)$ | 正 = 比平均好，负 = 比平均差 |
| $> 0$ | 阈值判断 | 硬阈值，不可调 |

**数值代入**：$A = 1.2$ → $\mathbb{1}[1.2 > 0] = 1$（学）；$A = -0.3$ → $\mathbb{1}[-0.3 > 0] = 0$（不学）。

**为什么是这个形式**：最简单的过滤方式——不引入任何超参数（不需要调温度 $\beta$），实现极其简单，效果在 D4RL 上 surprisingly 接近更复杂的 exponential 版本。
:::

**选择 2：Exponential Weight（软加权）**

$$
f_{\text{exp}}(A) = \frac{\exp(A / \beta)}{\mathbb{E}_{a' \sim \mathcal{D}(\cdot|s)}[\exp(A(s, a') / \beta)]}
$$

**这个公式在做什么**：用 softmax 形式把 advantage 转换为归一化的非负权重——advantage 越高的动作权重指数级增大，温度 $\beta$ 控制分布的"尖锐程度"。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $\exp(A/\beta)$ | 指数加权 | advantage 越大，权重指数增长 |
| $\beta$ | 温度超参数 | 典型值 1.0；$\beta \to 0$ 退化为 binary filter；$\beta \to \infty$ 退化为均匀加权（普通 BC） |
| 分母 $\mathbb{E}_{a'}[\cdots]$ | 归一化常数 | 对同一状态下数据中所有动作的 $\exp(A/\beta)$ 求均值，使权重归一化 |

**数值代入**：假设状态 $s$ 下有 3 个动作，$A_1 = 2.0$，$A_2 = 0.5$，$A_3 = -1.0$，$\beta = 1.0$：

- $\exp(2.0/1.0) = 7.39$
- $\exp(0.5/1.0) = 1.65$
- $\exp(-1.0/1.0) = 0.37$
- 分母均值 $= (7.39 + 1.65 + 0.37)/3 = 3.14$

权重：$f_1 = 7.39/3.14 = 2.35$，$f_2 = 1.65/3.14 = 0.53$，$f_3 = 0.37/3.14 = 0.12$

→ 动作 1 的学习权重是动作 3 的约 20 倍，好动作被大幅放大。

**为什么是这个形式**：softmax 形式可以证明是 KL 约束优化 $\max_\pi \mathbb{E}[A]$ s.t. $D_{\text{KL}}(\pi\|\pi_{\text{data}}) \leq \epsilon$ 的闭式解（拉格朗日对偶），$\beta$ 对应拉格朗日乘子。
:::

**代入数字**（binary filter 模式）：

假设某状态 $s$ 下数据中有 4 个动作，Q 函数给出的评分：
- $a_1$：$Q = 5.2$，$V = 4.0$ → $A = 1.2 > 0$ → 学！权重 = 1
- $a_2$：$Q = 4.5$，$V = 4.0$ → $A = 0.5 > 0$ → 学！权重 = 1
- $a_3$：$Q = 3.8$，$V = 4.0$ → $A = -0.2 < 0$ → 不学！权重 = 0
- $a_4$：$Q = 2.1$，$V = 4.0$ → $A = -1.9 < 0$ → 不学！权重 = 0

策略只会模仿 $a_1$ 和 $a_2$，完全忽略 $a_3$ 和 $a_4$。

---

## 三、CRR 和 RECAP 的关系

**RECAP 的 indicator 机制就是 CRR binary filter 的直系后代**。核心差异在于：

| 维度 | CRR (2020) | RECAP (2025) |
|------|-----------|-------------|
| Q 函数来源 | 标准 TD 学习 | 不用 Q 函数，用 episode-length-based return + distributional value model |
| 过滤粒度 | Per-sample：每个 $(s,a)$ 独立判断 | Per-task top-k：同一任务下取 advantage 最高的前 k% |
| Filter 阈值 | 固定 = 0 | 动态 = 每个任务自适应确定 |
| 数据来源 | 纯离线（固定数据集） | 迭代采集（每轮用新策略采新数据） |
| 适配场景 | D4RL MuJoCo benchmark | 真实机器人 VLA |
| 人类干预 | 不支持 | 支持（`force_intervention_positive`） |

可以认为 **RECAP = CRR(binary) + 迭代采集 + distributional value + per-task 自适应阈值 + human intervention 融入**。

---

## 四、D4RL 实验结果

CRR 在 D4RL benchmark 上的表现：

### 4.1 Locomotion 任务（HalfCheetah, Hopper, Walker2d）

CRR 在 medium 和 medium-replay 数据集上优于 BC 和 BCQ，和 CQL 持平或略低。CRR+ 变体（2023 年调优后）达到了 D4RL locomotion 的新 SOTA。

### 4.2 高维任务（DM Control Suite）

CRR 在高维状态空间任务上显著优于竞品——这是它被强调的核心优势。论文指出 CRR 在"state+action 维度 > 50"的任务上比 CQL/BCQ 衰退明显更少。

### 4.3 D4RL 综合排名

根据 [Improving and Benchmarking Offline RL (2023)](https://arxiv.org/abs/2306.00972) 的系统对比：

| 方法 | Locomotion 均分 | AntMaze 均分 | Adroit 均分 |
|------|---------------|-------------|------------|
| BC | 44.3 | 2.5 | 18.7 |
| CQL | 67.8 | 42.3 | 48.2 |
| IQL | 68.5 | **52.1** | 55.3 |
| CRR | 63.5 | 28.7 | 42.6 |
| **CRR+** | **72.1** | 45.8 | **57.4** |

CRR+ 通过调整 implementation 细节（layer norm、大网络、正确的 target update 频率）大幅提升了原始 CRR 的表现。

---

## 五、CRR 的理论优势

### 5.1 不需要策略梯度

CRR 完全不做 $\nabla_\theta J(\theta)$ 这种策略梯度更新——它的梯度形式和 supervised learning 完全一样，只是每个样本的 loss 乘了一个非负权重。这意味着：

- **训练稳定性极好**：不会出现策略梯度中常见的方差爆炸
- **和任何策略架构兼容**：只要能算 $\log \pi_\theta(a|s)$，就能用 CRR。Gaussian、GMM、离散 token 都行
- **不需要 importance sampling**：因为不做 off-policy correction

### 5.2 不需要 OOD 动作的 Q 值

CQL 需要在 OOD 动作上**显式评估** Q 值（然后压低它）。CRR 完全不需要——它只在数据中已有的 $(s, a)$ 上评估 Q，然后决定"学还是不学"。这避免了 Q 函数在数据外区域的不可靠外推。

### 5.3 和 AWR 的理论关系

CRR 的 exponential weight 模式可以证明等价于求解以下约束优化问题：

$$
\max_\pi \mathbb{E}_{s \sim \mathcal{D}} \left[\mathbb{E}_{a \sim \pi(\cdot|s)}[A(s,a)]\right] \quad \text{s.t.} \quad D_{\text{KL}}(\pi \| \pi_{\text{data}}) \leq \epsilon
$$

**这个公式在做什么**：在"策略不能偏离数据分布太远"的 KL 约束下，最大化策略的期望 advantage——CRR 的 exponential weight 正是这个约束优化问题的闭式解。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $\max_\pi$ | 优化变量 | 在所有可能的策略中找最优的那个 |
| $\mathbb{E}_{s \sim \mathcal{D}}$ | 对状态求期望 | 从离线数据中采状态 |
| $\mathbb{E}_{a \sim \pi(\cdot\|s)}[A(s,a)]$ | 策略下的期望 advantage | 新策略选出的动作平均比数据策略好多少 |
| $D_{\text{KL}}(\pi \\\| \pi_{\text{data}})$ | KL 散度约束 | 新策略不能离数据中行为策略太远 |
| $\epsilon$ | 约束预算 | KL 散度的上界，越大允许偏离越多 |

**数值代入**：假设某状态下数据策略 $\pi_{\text{data}}$ 在 3 个动作上的分布为 $(0.5, 0.3, 0.2)$，advantage 为 $(2.0, 0.5, -1.0)$。无约束最优解是把所有概率压到 $a_1$：$\pi^* = (1, 0, 0)$，但此时 $D_{\text{KL}} = 0.5\ln(1/0.5) + \cdots = \infty$，违反约束。KL 约束迫使最优策略是 $\pi_{\text{data}}$ 的"指数倾斜"版本：$\pi^*(a) \propto \pi_{\text{data}}(a) \cdot \exp(A(s,a)/\beta)$，其中 $\beta$ 由 $\epsilon$ 决定。

**为什么是这个形式**：直接做 $\max_a Q(s,a)$ 会选到 OOD 动作。加上 KL 约束后，策略被锚定在数据分布附近，只能在数据覆盖的范围内做"有限度的改善"。
:::

即"在 KL 约束下最大化 advantage"——这和 AWR 的推导目标完全一致。**CRR 的 binary filter 模式可以看作 $\beta \to 0$ 的极限情况**。

---

## 六、局限性

1. **Q 函数仍需训练**：虽然策略更新不走策略梯度，但 Q 函数训练仍需标准 TD learning，仍会遇到离线 TD 的有偏估计问题（只是问题被 filter 机制大幅缓解了）
2. **Binary filter 可能浪费数据**：如果 Q 估计有噪声，一些本来好的动作可能因为 $A$ 估计为负被错误丢弃
3. **不适合数据极度稀疏的场景**：如果好动作在数据中占比 < 5%，过滤后可能没有足够样本训练

---

## 七、为什么要在 2025 年回顾 CRR

CRR 本身是 2020 年的工作，但它在 2025 年的 VLA 后训练浪潮中重新变得极其重要，因为：

1. **RECAP 的核心机制就是 CRR 的 binary filter 变体**——理解 CRR 才能理解 RECAP 的设计动机
2. **CO-RFT 的 AWR loss 和 CRR 的 exponential weight 本质相同**——只是 advantage 来源不同
3. **ARM 的 sigmoid-weighted BC 是 CRR 的又一个变体**——用 sigmoid 替代 indicator/exponential
4. **CRR 证明了"过滤式 BC 在离线 RL 中 surprisingly effective"这个核心洞察**——所有后续 VLA 离线后训练方法都在利用这个洞察

---

## 八、总结

| 维度 | 要点 |
|------|------|
| 核心思想 | 把离线策略优化转化为 value-filtered regression |
| Filter 模式 | Binary（$A > 0$ 才学）或 Exponential（$\exp(A/\beta)$ 加权） |
| 不需要 | 策略梯度、OOD 动作的 Q 评估、importance sampling |
| 理论关系 | AWR 的特殊情况；RECAP 的直接前身 |
| 优势 | 训练稳定、高维任务友好、实现极简 |
| D4RL 表现 | CRR+ 版本 locomotion 和 adroit SOTA |
| 对 VLA 的影响 | RECAP、CO-RFT、ARM 的共同理论基础 |

---

## 延伸阅读

- [AWR：优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) ← CRR 的理论前身
- [RECAP 精读](./016_RECAP_从真实部署经验中RL学习) ← CRR binary filter 在 VLA 上的继承者
- [CQL：保守 Q 学习](/前置知识/002g_前置知识_CQL保守Q学习) ← 对比方法：用"悲观"而非"过滤"
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) ← 离线 RL 方法全景
- [CO-RFT：离线分块 RL 微调 VLA](./021_CO_RFT_离线分块RL微调VLA) ← CRR 思想在 VLA chunk 级的应用
- [ARM 精读](./097_ARM_优势奖励建模长horizon操作) ← CRR 思想的 long-horizon 变体
