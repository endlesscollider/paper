---
title: IDQL：隐式扩散 Q 学习
order: 105
tags: [强化学习, 扩散模型]
category: 精读
star: 4
---

# IDQL：Implicit Diffusion Q-Learning 深度精读

> **论文标题**: Implicit Diffusion Q-Learning  
> **作者**: Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, Sergey Levine  
> **机构**: UC Berkeley  
> **发表**: 2023 (arXiv:2304.10573)  
> **代码**: https://github.com/philippe-eecs/IDQL

**标签**: `#扩散策略` `#Q-Learning` `#离线RL` `#Offline-to-Online` `#隐式策略` `#加权回归` `#D4RL` `#样本效率`

**知识链接**：
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — 为什么先 BC 再 RL
- [Diffusion Policy](/前置知识/000c_前置知识_Diffusion_Policy) — 扩散策略基础
- [DPPO：扩散策略策略优化](./001_DPPO_扩散策略策略优化) — 对比：on-policy 路线
- [Online DPRL 综述](./003_Online_DPRL_综述_扩散策略与在线RL) — Q-Weighting 家族的代表
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — Q-Learning 路线的分析

---

## 一、定位：Off-Policy 路线的代表

### 1.1 为什么需要 Off-Policy 方法

DPPO 需要大量并行环境（1000–4096 个）来收集 on-policy 数据。但很多场景中这不现实：

- **真实机器人**：只有 1 个物理环境，每次交互昂贵
- **已有离线数据**：想直接利用之前采集的数据，不必重新交互
- **安全约束**：不能大量在线探索

这些场景都需要**样本效率高**的 off-policy 方法。

### 1.2 在综述中的定位

回忆 Online DPRL 综述的四大家族，IDQL 属于 **Q-Weighting** 家族——不直接改策略参数，而是用 $Q$ 值给动作"打分"来实现策略改进：

```mermaid
graph TD
    A["四大算法家族"] --> B["Action-Gradient<br/>(DIPO)"]
    A --> C["Q-Weighting<br/>(IDQL, AWR) ✓"]
    A --> D["Proximity-Based<br/>(DPPO)"]
    A --> E["BPTT<br/>(DQL, QSM)"]
    style C fill:#e1f5fe
```

---

## 二、核心方法

### 2.1 核心理念：解耦策略训练和策略改进

传统 Actor-Critic 中，Actor 和 Critic 耦合训练——$Q$ 的梯度直接流入 Actor，两者互相影响。IDQL 彻底解耦二者：

```mermaid
flowchart TD
    subgraph "IDQL 三组件（完全独立训练）"
        A["行为策略 π_β<br/>纯 BC 训练<br/>拟合数据分布"]
        B["Q 函数<br/>IQL 训练<br/>评估动作价值"]
        C["V 函数<br/>Expectile 回归<br/>隐式估计 max Q"]
    end
    A -->|"推理时结合"| D["采样 M 个动作 → Q 打分 → 选最好的"]
    B --> D
    style D fill:#e8f5e9
```

**好处**：$\pi_\beta$ 的训练完全是 BC（稳定可靠）；$Q$ 训练是标准 off-policy（成熟技术）；两者互不干扰，不存在"$Q$ 梯度破坏策略"的问题。

### 2.2 扩散行为策略的训练

完全标准的 Diffusion Policy BC 训练，不做任何 RL 修改：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(\mathbf{s},\mathbf{a})\sim\mathcal{D},\; k\sim U(1,K),\; \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})} \left\|\boldsymbol{\epsilon}_\theta(\mathbf{a}_k, k, \mathbf{s}) - \boldsymbol{\epsilon}\right\|^2
$$

**这个公式在做什么**：让扩散网络学会从加噪的动作里猜出加了多少噪声——训练完之后，从纯噪声反复"去猜噪声再减掉"就能生成一个像数据集里的动作。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $(\mathbf{s},\mathbf{a})\sim\mathcal{D}$ | **数据集抽样** | 从离线数据集里随机抽一条 (状态, 动作) 对 |
| $k\sim U(1,K)$ | **随机加噪程度** | 随机选一个扩散时间步，决定往 $\mathbf{a}$ 里加多少噪声 |
| $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ | **真实噪声** | 实际叠加到动作上的高斯噪声，也是网络要猜的目标 |
| $\boldsymbol{\epsilon}_\theta(\mathbf{a}_k, k, \mathbf{s})$ | **网络的猜测** | 网络看到加噪后的动作 $\mathbf{a}_k$、噪声程度 $k$、状态 $\mathbf{s}$，猜"刚才加的噪声是什么样子" |
| $\|\boldsymbol{\epsilon}_\theta(\cdot)-\boldsymbol{\epsilon}\|^2$ | **猜测误差** | 网络猜的噪声和真实噪声的均方差 |
| $\mathbb{E}[\cdot]$ | **训练时的近似** | 实际训练中用 mini-batch 抽样求平均代替这个期望 |

**用人话读**："随机抽一个（状态, 动作），往动作上加一点随机噪声，让网络根据加噪后的样子和状态猜出加了什么噪声，猜错多少就是 loss。"

**为什么是这个形式**：这是标准 DDPM 去噪训练目标，直接搬来做行为克隆——训练目标只关心"学会数据集里的动作分布"，完全不涉及奖励，所以这一步和普通 [Diffusion Policy](/前置知识/000c_前置知识_Diffusion_Policy) 的 BC 训练没有任何区别。
:::

策略 $\pi_\beta$ 纯粹拟合数据分布，不做任何奖励相关优化。

### 2.3 IQL 风格的 $Q$ 函数训练

IDQL 使用 Implicit Q-Learning (IQL) 来训练 $Q$ 和 $V$。标准 Q-Learning 的 target 需要 $\max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}')$，在连续高维动作空间中不可行。IQL 的 trick 是引入 $V(\mathbf{s})$ 来隐式估计这个 max：

**Q 更新**（用 $V$ 替代 max）：

$$
\mathcal{L}_Q = \mathbb{E}_{(\mathbf{s},\mathbf{a},r,\mathbf{s}')\sim\mathcal{D}}\left[\left(Q(\mathbf{s},\mathbf{a}) - r - \gamma V(\mathbf{s}')\right)^2\right]
$$

**这个公式在做什么**：让 $Q$ 网络学会预测"这一步的即时奖励加上未来能拿到的价值"，用 $V(\mathbf{s}')$ 代替标准 Q-Learning 里那个求不出来的 $\max_{\mathbf{a}'}Q(\mathbf{s}',\mathbf{a}')$。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $(\mathbf{s},\mathbf{a},r,\mathbf{s}')\sim\mathcal{D}$ | **一条转移记录** | 从离线数据集里抽一条"状态-动作-奖励-下一状态" |
| $r+\gamma V(\mathbf{s}')$ | **回归目标** | 即时奖励加上打折的下一状态价值，代替了标准 Q-Learning 里的 $r+\gamma\max_{a'}Q(s',a')$ |
| $Q(\mathbf{s},\mathbf{a})$ | **当前预测** | $Q$ 网络对这条记录给出的价值估计 |
| $(Q-r-\gamma V(\mathbf{s}'))^2$ | **预测误差** | 预测值和目标值的均方差，就是要最小化的量 |
| $\mathbb{E}[\cdot]$ | **训练时的近似** | 实际训练用 mini-batch 抽样求平均 |

**用人话读**："让 $Q$ 网络预测的值，尽量接近'这一步奖励 + 下一状态的价值 $V$'，误差就是训练信号。"

**为什么用 $V(\mathbf{s}')$ 而不是 $\max_{a'}Q(s',a')$**：连续高维动作空间下，对 $Q(s',\cdot)$ 做全局最大化本身就是一个难解的优化问题；IQL 的做法是单独训练一个 $V(\mathbf{s}')$ 来隐式近似这个 max（原理见下一条公式），$Q$ 训练时直接用它当目标，避免了每步都要解一次 $\arg\max$。
:::

**V 更新**（Expectile 回归，关键！）：

$$
\mathcal{L}_V = \mathbb{E}_{(\mathbf{s},\mathbf{a})\sim\mathcal{D}}\left[L_\tau\!\left(Q(\mathbf{s},\mathbf{a}) - V(\mathbf{s})\right)\right]
$$

其中 $L_\tau(u) = |\tau - \mathbb{1}(u < 0)| \cdot u^2$ 是 expectile loss，$\tau > 0.5$（通常 0.7–0.9）。

**这个公式在做什么**：用一种"不对称惩罚"的回归让 $V(\mathbf{s})$ 悄悄逼近 $Q(\mathbf{s},\cdot)$ 分布里比较高的那一段，从而不用真的做 $\max$ 运算就能近似出"这个状态下最好能拿多少分"。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q(\mathbf{s},\mathbf{a})-V(\mathbf{s})$ | **残差** | 数据集里这个动作的 $Q$ 值和当前 $V$ 估计的差 |
| $\mathbb{1}(u<0)$ | **方向判断器** | 判断残差是负的（$V$ 高估了）还是正的（$V$ 低估了） |
| $\|\tau-\mathbb{1}(u<0)\|$ | **不对称权重** | $\tau>0.5$ 时，低估（$u>0$，权重 $=\tau$）的惩罚大于高估（$u<0$，权重 $=1-\tau$）的惩罚 |
| $L_\tau(u)=\|\tau-\mathbb{1}(u<0)\|\cdot u^2$ | **expectile loss** | 把不对称权重乘到平方误差上，整体作为 $V$ 的训练损失 |
| $\mathbb{E}_{(\mathbf{s},\mathbf{a})\sim\mathcal{D}}[\cdot]$ | **训练时的近似** | mini-batch 抽样求平均 |

**用人话读**："让 $V$ 去拟合 $Q$，但故意让'$V$ 猜低了'的惩罚比'猜高了'的惩罚更重，逼着 $V$ 悄悄往 $Q$ 分布的高分区域靠。"

**为什么 expectile 能近似 max**：当 $\tau>0.5$ 时，低估（$Q>V$）的惩罚远大于高估（$Q<V$）的惩罚，所以 $V$ 被推向 $Q$ 分布的上尾部，$V(\mathbf{s})\approx\max_{\mathbf{a}}Q(\mathbf{s},\mathbf{a})$；$\tau$ 越接近 1，近似越紧。这样就绕开了连续动作空间里直接求 $\max$ 的难题。
:::

### 2.4 推理时的动作选择

策略改进**完全发生在推理时**，而非训练时：

```mermaid
flowchart LR
    A["给定状态 s"] --> B["从 π_β 采样<br/>M 个动作"]
    B --> C["Q 函数<br/>逐个打分"]
    C --> D["选 Q 值最高<br/>或 softmax 加权"]
    D --> E["执行动作 a*"]
    style E fill:#c8e6c9
```

$$
\mathbf{a}^* = \arg\max_{i \in \{1,\ldots,M\}} Q(\mathbf{s}, \mathbf{a}_i), \quad \mathbf{a}_i \sim \pi_\beta(\cdot|\mathbf{s})
$$

**这个公式在做什么**：从行为策略 $\pi_\beta$ 里抽 $M$ 个候选动作，让 $Q$ 函数给每个候选打分，直接挑分最高的执行——策略本身完全没变，改进来自"事后挑选"。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mathbf{a}_i\sim\pi_\beta(\cdot\|\mathbf{s})$ | **候选动作生成器** | 用训练好的扩散行为策略，在状态 $\mathbf{s}$ 下独立采样出 $M$ 个候选动作 |
| $Q(\mathbf{s},\mathbf{a}_i)$ | **打分员** | 用训练好的 $Q$ 网络给每个候选动作评分 |
| $\arg\max_{i\in\{1,\ldots,M\}}$ | **选拔赛** | 在 $M$ 个候选里挑出打分最高的那一个 |
| $\mathbf{a}^*$ | **最终执行的动作** | 被选中的、实际发给环境执行的动作 |

**用人话读**："让行为策略随便生成几十个候选动作，$Q$ 函数逐个打分，直接执行分数最高的那个。"

**为什么是这个形式**：策略改进不需要修改网络参数，只需要在推理时"多生成几个、选最好的"——这是一种基于拒绝采样的隐式策略改进，好处是训练侧完全解耦（$\pi_\beta$ 和 $Q$ 各自独立训练），代价是推理时要多次采样+打分。
:::

或者用 softmax 加权采样：$w_i \propto \exp\!\left(\beta \cdot Q(\mathbf{s}, \mathbf{a}_i)\right)$。

**和其他方法的本质区别**：

| 方法 | 策略本身变了吗 | 改进机制 |
|---|---|---|
| DQL | ✓（$Q$ 梯度穿去噪链） | 改策略参数 |
| DIPO | ✓（用新目标重训练） | 推动作再拟合 |
| IDQL | **✗（策略不变！）** | 选择效应：好的被执行，差的被丢弃 |

### 2.5 Offline-to-Online 扩展

IDQL 天然支持渐进式改进：

1. **纯 offline 阶段**：用离线数据训练 $\pi_\beta$（BC）和 $Q, V$（IQL），部署时 $Q$ 打分选动作
2. **Online 微调阶段**：继续用新的在线数据更新 $Q$ 和 $V$；$\pi_\beta$ 也可以用新数据微调（仍是 BC loss），或固定不动

在线数据少量就有效（$Q$ 更新比策略梯度更 data-efficient），策略训练保持稳定。

---

## 三、优缺点分析

### 3.1 优点

- **训练极稳定**：策略训练 = 纯 BC，$Q$ 训练 = 标准 IQL，两者解耦互不影响
- **样本效率高**：off-policy，replay buffer 可重复利用数据
- **不需要 $\log\pi$**：扩散策略的概率密度不可算？没关系，只需要从中采样
- **对去噪步数不敏感**：无论 $K=5$ 还是 $K=100$，只在最终动作上做选择

### 3.2 缺点

- **策略改进有上限**：只能选择 $\pi_\beta$ 能生成的动作。如果数据中没有好动作，选不出来
- **推理时需要多次采样**：$M=64$ 次采样 × $K=20$ 步去噪 = 1280 次前向传播，远慢于 DPPO
- **$Q$ 函数在高维动作空间仍然难学**：action chunk 使动作维度达到 56–112 维
- **不利用大规模并行**：off-policy 方法不能充分利用 GPU 仿真的海量 on-policy 数据

### 3.3 和 DPPO 的直接对比

| 维度 | IDQL | DPPO |
|---|---|---|
| 更新方式 | 间接（选择） | 直接（梯度） |
| 策略改进上限 | 受 $\pi_\beta$ 支持集限制 | 理论上无上限 |
| 样本效率 | 高（off-policy） | 中（on-policy） |
| 并行利用率 | 低 | 高 |
| 推理速度 | 慢（$M$ 次采样 + 打分） | 正常（$K$ 次去噪） |
| 复杂任务性能 | 中等 | 最好 |
| 适用场景 | 数据少、不能大量并行 | GPU 仿真、大规模并行 |

---

## 四、关键实验结果

### 4.1 高维操作任务——差距拉开

当动作空间变大（7+ 维 + action chunk），IDQL 和 DPPO 的差距明显：

| 任务 | 动作维度 | IDQL | DPPO |
|---|---|---|---|
| Transport | 14×8 = 112 维 | ~40% | >90% |
| Square | 7×8 = 56 维 | ~70% | ~100% |

差距来源：采样 $M=64$ 个动作覆盖 112 维空间的概率极低；$Q$ 在高维中估计不准；"选最好的"不如"往更好的方向走"。

### 4.2 Online DPRL 综述中的评测

| 维度 | Q-Weighting (IDQL 类) |
|---|---|
| 简单任务 | 性能不错 |
| 复杂任务 | 有上限 |
| 样本效率 | 高 |
| 并行可扩展性 | 低–中 |
| 鲁棒性 | 中等 |

---

## 五、什么时候用 IDQL

### 推荐场景

- 有大量高质量离线数据想直接利用
- 只有 1–few 个真实环境，不能大量并行
- 需要渐进式 offline-to-online 改进
- 动作空间相对低维（≤14 维，无大 action chunk）
- 安全约束严格，不能大量在线探索

### 不推荐场景

- 有 GPU 仿真 + 1000+ 并行环境 → 用 DPPO
- 动作空间高维 (>50 维 action chunk) → $Q$ 学不好
- 需要超越预训练数据分布的策略 → 受限于 $\pi_\beta$ 支持集
- 推理延迟敏感 (<10ms) → 多次采样太慢

---

## 六、个人评价

IDQL 最大的贡献是展示了"不需要让 $Q$ 梯度流入策略"这条路线的可行性。在 DPPO 出现之前，它是扩散策略 + RL 最稳定的方案。DPPO 出现后在大部分场景中被超越，但在数据稀缺的 offline/few-shot 场景中，IDQL 仍有不可替代的价值。

"解耦"的设计思想影响了后续的 AWR 变体和加权回归方法——证明了在某些条件下"简单选择 > 复杂优化"。

---

## 延伸阅读

- Hansen-Estruch et al. (2023) "IDQL" ← 原文
- Kostrikov et al. (2022) "IQL: Offline RL with Implicit Q-Learning" ← $Q$ 函数训练基础
- [DPPO](./001_DPPO_扩散策略策略优化) ← On-policy 对比方案
- [Online DPRL 综述](./003_Online_DPRL_综述_扩散策略与在线RL) ← 统一评测
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) ← Q-Weighting 如何绕过难题
