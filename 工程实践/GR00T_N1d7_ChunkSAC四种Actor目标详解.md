---
title: "GR00T N1.7 Chunk-SAC：四种 Actor 目标的工程实践"
order: 10
tags: [强化学习, SAC, GR00T, Flow Matching, AWR, 后训练, 工程实践]
category: 工程实践
star: 5
---

# GR00T N1.7 Chunk-SAC：四种 Actor 目标的工程实践

> **一句话**：GR00T N1.7 的 Chunk-SAC 系统提供了四种 Actor 更新策略——从保守的 Q 加权 Flow Matching，到激进的直接 Q 最大化，再到完整 SAC 框架——让你根据任务需求和 Critic 可信度选择最合适的策略改进方式。

## 相关阅读

**前置知识**（读本文前建议了解）：
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 最大熵 off-policy RL 的完整推导
- [AWR 优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) — Q 加权模仿学习的数学基础
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — GR00T 动作生成的底层机制
- [Q 加权 Flow 策略](/前置知识/001v_前置知识_Q加权Flow策略_不穿透去噪链的RL训练) — 不穿透去噪链的 RL 训练
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Twin-Q 和 Bellman 方程
- [TD3](/前置知识/000q_前置知识_TD3) — Twin Critic 的来源
- [行为约束策略优化](/前置知识/001l_前置知识_行为约束策略优化) — 约束策略不偏离数据的机制

**关联文章**：
- [RLinf：BC 到 RL 的 ACT 后训练架构](./RLinf_BC到RL的ACT后训练架构) — PPO 路线的对比
- [GR00T N1.7 深度解析系列](/系列/groot_n1d7_deep_dive/index) — 模型架构详解
- [动作分块 RL 基础](/系列/groot_rl_deep_dive/06_动作分块RL基础_QChunking到AQC回顾) — Chunk-level Critic 的理论基础
- [RLinf 算法实现：SAC](/系列/rlinf_deep_dive/10_算法实现_SAC与其他算法) — RLinf 框架中 SAC 的通用实现

---

## 一、这篇文章要解决什么问题

你已经用 GR00T N1.7 的 Flow Matching 管线完成了 BC 预训练，模型能做出基本合理的动作。现在想用环境 reward 做 RL 后训练，让策略超越 BC 天花板。

**核心困难**：GR00T 的动作不是一步前向就出来的——它是一个多步去噪过程：


$$
\mathbf{x}_0 \sim \mathcal{N}(0, I) \;\xrightarrow{v_\theta(\cdot, t_1)}\; \mathbf{x}_1 \;\xrightarrow{v_\theta(\cdot, t_2)}\; \cdots \;\xrightarrow{v_\theta(\cdot, t_K)}\; \mathbf{a}_{0:H}
$$

最终输出是一个**完整动作块** $[B, H, D]$（batch × chunk length × action dim）。标准 SAC 的 `∇_θ Q(s, π(s))` 需要把 Q 的梯度穿过这 K 步去噪链——这就是 [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) 中详细讨论过的梯度爆炸问题。

**四种 Actor objective 就是对"Q 梯度怎么指导策略改进"这个问题的四种不同回答**，从完全不穿透去噪链（`awr_flow`），到部分穿透（`direct_q`、`awr`），再到通过独立 adapter 穿透（`sac_flow_g`）。

---

## 二、贯穿全文的例子

> **任务**：一个双臂人形机器人（GR00T 配置，62 维动作）在 MiArena 仿真中学习打开柜门。
> - 动作块长度 $H = 40$（模型一次输出 40 步动作）
> - 动作维度 $D = 62$（双臂关节角度 + 手指）
> - Critic 输入：状态特征 + 展平后的 $40 \times 62 = 2480$ 维动作向量
> - Reward：柜门角度变化 + 稀疏的成功奖励
> - Replay Buffer 中混合着：随机探索轨迹、部分成功轨迹、完全成功轨迹

---

## 三、共享基础：所有方案都站在同一个地基上

四种 Actor objective 不是四套独立系统。它们共享一个完整的 Chunk-SAC 训练框架，只替换 Actor loss 计算逻辑。

### 3.1 动作块与 Chunk-level Critic

GR00T 的 Actor 一次产生完整动作块：

$$
\pi_\theta(s) \to \mathbf{a}_{0:H} \in \mathbb{R}^{B \times H \times D}
$$

Critic 不对单步动作估值，而是对**整个动作块**打分：

$$
Q(s, \mathbf{a}_{0:H}) \to [Q_1, Q_2] \in \mathbb{R}^{B \times 2}
$$


使用 Twin-Q（取 $\min(Q_1, Q_2)$）降低过估计风险——这和 [TD3](/前置知识/000q_前置知识_TD3) 的设计动机完全一致。

**代入例子**：打开柜门任务中，Critic 评估的是"从当前状态出发，执行这 40 步动作序列后能获得多少累计奖励"。一个好的动作块可能是"先移向柜门把手 → 抓住 → 拉开"，Critic 给出高 Q 值；一个差的动作块可能"手伸过头"，Critic 给出低 Q 值。

### 3.2 Chunk Bellman Target

四种方案共用以下 TD target：

$$
y = \underbrace{\sum_{i=0}^{H-1} \gamma^i \cdot \text{valid}_i \cdot r_i}_{R_{\text{chunk}}} + \gamma^H \cdot m \cdot \Big(\min(Q_1^{\text{target}}(s', \mathbf{a}'), Q_2^{\text{target}}(s', \mathbf{a}')) - \alpha \log \pi(\mathbf{a}'|s')\Big)
$$

> **一句话**：把 chunk 内的折扣奖励加起来，再加上 chunk 结束后下一个状态的 bootstrapped value。

**逐项拆解**：

| 符号 | 含义 | 例子中的对应 |
|------|------|-------------|
| $\gamma^i \cdot \text{valid}_i \cdot r_i$ | 第 $i$ 步的折扣奖励，乘 valid mask 处理 episode 中途结束 | 柜门角度增加了 5° → $r_i = +0.5$，乘上 $\gamma^i$ |
| $R_{\text{chunk}}$ | 40 步内所有折扣奖励之和 | 如果 40 步内柜门打开了，包含稀疏成功奖励 |
| $\gamma^H$ | 40 步后的折扣因子 | $0.99^{40} \approx 0.67$ |
| $m$ | bootstrap mask（episode 未结束=1，结束=0） | 如果 40 步后 episode 还没结束，$m=1$ |
| $Q^{\text{target}}$ | 目标网络对下一状态-动作对的估值 | EMA 更新的 Critic 副本 |
| $\alpha \log \pi$ | 熵项，**只有 `sac_flow_g` 方案使用** | 其他三种方案 $\alpha = 0$ |

**关键区分**：只有 `sac_flow_g` 在 Bellman target 中包含熵项。其他三种方案使用标准的无熵 TD target——这意味着它们的 Critic 学习的是"纯奖励价值"，不考虑策略的随机性。

#### 3.2.1 Bootstrap 怎么算：没有 V 网络，用 Q + 策略采样替代

你可能注意到公式里用的是 $Q^{\text{target}}(s', \mathbf{a}')$ 而不是 $V(s')$。这里没有独立的 Value 网络——bootstrap value 是通过**在下一状态重新采样动作，然后用 target Critic 打分**来实现的。

**$s'$ 是什么？**

$s'$ 是执行完当前 chunk（H=40 步）后，环境到达的下一个状态。代入例子：机器人在状态 $s$ 执行了 40 步动作块，环境做了 40 步物理模拟，到达 $s'$——此时关节角度变了、柜门可能开了一部分。这个 $s'$ 在 rollout 时就记录在 replay buffer 中了。

**$\mathbf{a}'$ 怎么来？**

$\mathbf{a}'$ 是**当前策略在 $s'$ 上重新采样的一整个动作块**：

$$
\mathbf{a}' = \pi_\theta(s') \in \mathbb{R}^{H \times D}
$$

即：拿到 $s'$ → 跑一次 Actor forward（完整的 flow 去噪过程）→ 得到一个新的 40 步动作块。然后把 $(s', \mathbf{a}')$ 送进 target Critic 得到 Q 值估计。

**为什么不直接维护一个 $V(s')$ 网络？**

在标准 SAC 理论中，$V$ 和 $Q$ 有如下等价关系：

$$
V(s') = \mathbb{E}_{\mathbf{a}' \sim \pi(\cdot|s')}\Big[Q(s', \mathbf{a}') - \alpha \log \pi(\mathbf{a}'|s')\Big]
$$

也就是说，$V$ 本质上是"对 $Q - \alpha \log \pi$ 关于动作的期望"。Chunk-SAC 不显式维护 V 网络，而是用**单次策略采样**近似这个期望：

$$
V(s') \approx Q^{\text{target}}(s', \mathbf{a}') - \alpha \log \pi(\mathbf{a}'|s'), \quad \mathbf{a}' \sim \pi(\cdot|s')
$$

这是一个 single-sample Monte Carlo 估计。方差比独立 V 网络略高，但好处是：少一个网络、少一组参数、少一个训练 target。SAC 原论文的 v2 版本也采用了这种"去掉 V 网络"的做法。

**完整计算流程（伪代码）**：

```text
# 从 replay buffer 取一条数据
(s, a_{0:H}, rewards_{0:H}, s', done) = replay.sample()

# Step 1: chunk 内折扣奖励和
R_chunk = Σ_{i=0}^{H-1} γ^i * valid_i * r_i

# Step 2: 当前策略在下一状态采样动作
a' = π_θ(s')                    # 完整的 flow 去噪采样

# Step 3: target Critic 对 (s', a') 打分
Q_next = min(Q1_target(s', a'), Q2_target(s', a'))

# Step 4: 计算 log_pi（仅 sac_flow_g 使用，其他方案此项为 0）
log_pi_next = log π(a'|s')

# Step 5: 组装 Bellman target
y = R_chunk + γ^H * (1 - done) * (Q_next - α * log_pi_next)
```

**代入数字**：假设 $H=40$，$\gamma=0.99$，chunk 内累计奖励 $R_{\text{chunk}}=3.2$，episode 未结束（$m=1$），target Critic 对下一状态的策略动作打分 $Q_{\text{next}}=12.5$，$\alpha \log \pi = 0.3$（仅 `sac_flow_g`）：

- `awr_flow` / `direct_q` / `awr`：$y = 3.2 + 0.99^{40} \times 1 \times 12.5 = 3.2 + 0.669 \times 12.5 = 3.2 + 8.36 = 11.56$
- `sac_flow_g`：$y = 3.2 + 0.669 \times (12.5 - 0.3) = 3.2 + 8.16 = 11.36$

差异只有 0.2——熵项的效果是让 target 略微降低，鼓励策略保持随机性（如果策略过于确定，$-\alpha \log \pi$ 会很大，target 降低，Critic 学到"过于确定的策略值不那么高"）。

#### 3.2.2 $\log \pi(\mathbf{a}'|s')$ 是什么：Flow 策略的对数概率

对高斯策略来说，$\log \pi(a|s)$ 就是高斯分布的对数概率密度——有解析公式，算起来很简单。但 GR00T 是 **Flow Matching 策略**，动作是通过多步 ODE 积分产生的，不像高斯策略有现成的概率密度公式。

**Flow 策略的 $\log \pi$ 怎么算？**

Flow 策略定义了一个从噪声 $\mathbf{x}_0 \sim \mathcal{N}(0, I)$ 到动作 $\mathbf{a}$ 的确定性映射（给定 $s$）。根据变量替换公式（change of variables），这个映射的对数概率密度可以通过**沿 ODE 路径累积散度**来计算：

$$
\log \pi(\mathbf{a}|s) = \log p_0(\mathbf{x}_0) - \int_0^1 \text{div}\, v_\theta(\mathbf{x}_t, t, s)\, dt
$$

| 符号 | 含义 |
|------|------|
| $\log p_0(\mathbf{x}_0)$ | 初始高斯噪声的对数密度（$= -\frac{d}{2}\log 2\pi - \frac{1}{2}\|\mathbf{x}_0\|^2$） |
| $\text{div}\, v_\theta$ | velocity 网络输出的散度（divergence），衡量"流在这个点附近是压缩还是膨胀" |
| $\int_0^1 (\cdot)\, dt$ | 沿整条 ODE 路径积分，实际中用 K 步离散近似 |

> **直觉**：从噪声空间出发时的概率密度是已知的（标准高斯）。沿着 flow 路径走，如果 velocity 场在某处"压缩"了空间（散度 < 0），概率密度就增大（更多轨迹挤到一起）；如果"膨胀"了空间（散度 > 0），概率密度就减小。把这些变化一路累积起来，就得到最终动作点的对数概率。

**实际计算中的近似**：精确计算散度需要 $O(D)$ 次额外前向传播（对每个维度分别求偏导），代价很高。实践中常用 **Hutchinson 估计器**（随机向量投影）来近似散度，将计算量降到常数次额外前向传播。

$$
\text{div}\, v_\theta \approx \mathbf{z}^\top \frac{\partial v_\theta}{\partial \mathbf{x}} \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)
$$

**对三种不使用熵项的方案**（`awr_flow`、`direct_q`、`awr`）：因为 $\alpha = 0$，Bellman target 中的 $\log \pi$ 项被完全消掉——**根本不需要计算 $\log \pi$**。这不仅简化了代码，还省去了 Hutchinson 估计器带来的额外方差。

**只有 `sac_flow_g` 需要计算 $\log \pi$**：因为它是完整 SAC，entropy bonus 是核心组件。额外的计算代价是这个方案复杂度更高的原因之一。

### 3.3 Actor 更新门控

无论选哪种 objective，Actor 更新都需要通过 Worker 层的多重门控：

```mermaid
flowchart TD
    A["Replay 达到 min_buffer_size?"] -->|Yes| B["样本数达到 train_actor_steps?"]
    B -->|Yes| C["完成 critic_warmup_updates?"]
    C -->|Yes| D["当前 update 命中 critic_actor_ratio?"]
    D -->|Yes| E["Critic calibration ready?"]
    E -->|Yes| F["batch 中有合格样本?"]
    F -->|Yes| G["✅ 执行 Actor 更新"]
    A -->|No| X["❌ 跳过"]
    B -->|No| X
    C -->|No| X
    D -->|No| X
    E -->|No| X
    F -->|No| X
```


其中"合格样本"由 `actor_data_filter` 控制：

| 筛选模式 | 保留的数据 | 适用场景 |
|----------|-----------|---------|
| `success_or_progress` | 成功 + 产生正进度的 chunk | 通用默认值 |
| `success` | 仅成功 chunk | Critic 很准时，只学最好的 |
| `all` | 所有非失败 chunk | 探索早期数据稀疏时 |

**注意**：这个筛选 mask 只有 `awr_flow` 真正用作逐样本 loss 权重。其他三种方案只用它判断"本轮是否允许更新"，不在 loss 中区别对待不同样本。

---

## 四、方案一：`awr_flow` — Q 加权的 Flow Matching

### 4.1 核心思想

> **一句话**：不让 Q 梯度穿过动作采样过程，而是用 Critic 给 replay 中的动作"打分"，分高的动作在 flow-matching loss 中获得更大权重。

这是最保守的方案。它的哲学是：**与其让 Q 梯度直接"推"动作（可能推到 Critic 不可靠的区域），不如让 Critic 当"评委"，从 replay 中挑出好动作让 Actor 模仿**。

### 4.2 计算过程详解

**第一步：采样并评估（不反传）**

```text
# 在 torch.no_grad() 下执行
a_policy = π(s)                          # 当前策略采样
Q_policy = min(Q1(s, a_policy), Q2(s, a_policy))  # 当前策略动作的 Q 值
Q_data   = min(Q1(s, a_data),   Q2(s, a_data))    # replay 动作的 Q 值
A        = Q_data - Q_policy              # advantage：replay 比当前策略好多少
```

**第二步：计算权重**

$$
w_{\text{raw}} = \text{selected} \cdot \exp\left(\text{clamp}\left(\frac{A}{\tau}, \; -\infty, \; \log(w_{\max})\right)\right)
$$

$$
w = \frac{w_{\text{raw}}}{\sum_{\text{all ranks}} w_{\text{raw}}}
$$

> **一句话**：advantage 越大的 replay 动作权重越高，但通过 clamp 防止单个样本权重爆炸，通过跨 GPU 归一化保证权重之和为 1。

**逐项拆解**：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $\text{selected}$ | 质量筛选 mask（0 或 1） | `success_or_progress` 的输出 |
| $A$ | replay 动作相对当前策略的优势 | 好动作 $A > 0$，差动作 $A < 0$ |
| $\tau$ | AWR 温度，控制权重的"尖锐度" | 1.0（越小越集中在最好的样本） |
| $w_{\max}$ | 单个样本的最大权重上限 | 20.0 |
| 跨 rank 归一化 | FSDP 下所有 GPU 的权重共同归一化 | 保证分布式一致性 |

**代入数字**：假设 batch 中有 3 个样本：

| 样本 | $Q_{\text{data}}$ | $Q_{\text{policy}}$ | $A$ | $\exp(A/\tau)$ | 归一化后 |
|------|-------------------|---------------------|-----|----------------|----------|
| 样本 1（成功打开柜门） | 15.0 | 8.0 | +7.0 | $e^7 \approx 1097$ → clamp 到 20 | 20/22.37 ≈ 0.89 |
| 样本 2（部分打开） | 10.0 | 8.0 | +2.0 | $e^2 \approx 7.4$ | 7.4/22.37 ≈ 0.10 |
| 样本 3（失败） | — | — | — | selected=0 | 0.0 |

**效果**：成功样本获得了 89% 的训练权重，部分成功样本 10%，失败样本被直接排除。

**第三步：加权 Flow-Matching Loss**

$$
\mathcal{L}_{\text{awr\_flow}} = \text{weighted\_flow\_matching\_loss}(\mathbf{a}_{\text{data}}, \text{valid\_mask}, w)
$$

这里使用的是 GR00T **原生的** flow-matching 训练接口——和 BC 预训练时用的完全相同的 loss 形式，只是给每个样本乘了一个不同的权重。

### 4.3 梯度路径

```mermaid
flowchart LR
    subgraph "有梯度"
        A["replay 动作 a_data"] --> B["Flow-matching loss"]
        B --> C["GR00T Actor 参数 θ"]
    end
    subgraph "无梯度（detached）"
        D["Q_policy, Q_data"] --> E["advantage A"]
        E --> F["权重 w"]
    end
    F -.->|"标量权重"| B
```


**关键特征**：Q 值只用来算权重，不参与反传。Actor 的梯度完全来自 flow-matching loss 对 GR00T velocity 网络的标准训练梯度。这意味着：

1. 不存在"梯度穿过 K 步去噪链"的问题
2. 不会利用 Critic 的 $dQ/da$ 梯度（可能不准确）
3. Actor 更新始终停留在 flow-matching 训练空间中，不会偏离 GR00T 预训练的"流形"

### 4.4 适用场景

- Critic 还不够可信（训练早期、数据少），不敢直接用 Q 梯度推动作
- replay 中已有足够的成功或正进度轨迹（否则无好动作可模仿）
- 希望最大程度保持 GR00T 预训练的生成行为
- 任务允许较慢但稳定的策略改进

### 4.5 关键配置

```yaml
algorithm:
  actor_objective: awr_flow
  actor_data_filter: success_or_progress
  actor_use_awr_weights: true      # false 时所有合格样本等权
  awr_normalize_advantage: true
  awr_temperature: 1.0             # 越小 → 权重越集中在最好样本
  awr_max_weight: 20.0             # 防止单个样本主导整个 batch
```

### 4.6 局限性

**上限受 replay 限制**：如果 replay 中最好的动作只能把柜门打开 70%，`awr_flow` 永远不会产生"打开 100%"的动作——因为它只能模仿已有数据，无法创造新动作。要突破这个上限，需要 `direct_q` 或 `sac_flow_g`。

---

## 五、方案二：`direct_q` — 直接最大化 Twin-Q

### 5.1 核心思想

> **一句话**：让 Q 梯度直接穿过动作采样过程反传到 Actor 参数，把"让 Critic 满意"作为唯一目标。

这是最激进的方案。它的哲学是：**Critic 说哪个方向好，就往哪个方向走**。不需要 replay 中有好动作——策略可以自己"发明"从未出现在数据中的高 Q 值动作。

### 5.2 计算过程

$$
\mathcal{L}_{\text{direct\_q}} = \mathbb{E}_{s \sim \mathcal{D}}\left[\underbrace{c_{\text{ent}} \cdot \log \pi(\mathbf{a}|s)}_{\text{可选熵正则}} - \underbrace{\min(Q_1(s, \mathbf{a}), Q_2(s, \mathbf{a}))}_{\text{最大化 Q 值}}\right]
$$

> **一句话**：loss = 负的 Q 值（加可选的熵惩罚），梯度方向就是"把动作往 Q 值更高的方向推"。

**逐项拆解**：

| 项 | 作用 | 梯度方向 |
|----|------|----------|
| $-\min(Q_1, Q_2)$ | 最大化保守 Q 估计 | 把动作推向 Critic 认为好的区域 |
| $c_{\text{ent}} \cdot \log \pi$ | 惩罚策略过于确定 | 保持一定随机性（防止坍缩到单点） |

当 `path_entropy_coef = 0` 时，目标退化为纯粹的 $-Q$——完全由 Critic 驱动。

### 5.3 梯度路径

```mermaid
flowchart LR
    A["Actor 参数 θ"] --> B["可微 denoising steps"]
    B --> C["采样动作 a_policy"]
    C --> D["固定 online Critic"]
    D --> E["-min(Q1, Q2)"]
    E -.->|"梯度反传"| A
```

**关键理解**：Critic 在这个过程中不被更新——它只是作为一个"可微函数"提供 $dQ/da$ 梯度。Actor optimizer 只更新 Actor 参数；Critic 上产生的临时梯度会被丢弃。

### 5.4 `actor_backprop_steps` 的作用

GR00T 的 flow 采样有 K 步去噪。`actor_backprop_steps` 控制 Q 梯度穿过其中多少步：


| `actor_backprop_steps` | 梯度穿过的步数 | 显存 | 梯度质量 |
|------------------------|---------------|------|----------|
| 1 | 只穿过最后一步 | 低 | 粗糙但稳定 |
| K（全部） | 穿过所有去噪步 | 高 | 精确但可能爆炸 |

**代入例子**：如果 GR00T 用 10 步去噪（K=10），`actor_backprop_steps=1` 意味着只让最后一步的 velocity 网络接收到 Q 梯度。前 9 步作为"给定条件"不参与反传。这是在"梯度信息量"和"训练稳定性"之间的折中。

### 5.5 与标准 SAC 的关键区别

| 维度 | 标准 SAC | `direct_q` |
|------|----------|-----------|
| 熵正则 | 可学习 $\alpha$，有 alpha optimizer | 固定 `path_entropy_coef`，无额外 optimizer |
| Bellman target | 包含 $-\alpha \log \pi$ 熵项 | **无**熵项（$\alpha=0$ in target） |
| Actor loss | $\alpha \log \pi - Q$ | $c_{\text{ent}} \log \pi - Q$ |
| 温度自适应 | 是 | 否 |

这意味着 `direct_q` 不是一个完整的 SAC——它借用了"穿过 Critic 求梯度"的思路，但没有最大熵框架的完整理论保证。

### 5.6 风险：Critic 误差利用

`direct_q` 的最大风险：Actor 会主动寻找 Critic 的"弱点"。

```mermaid
flowchart TD
    A["Actor 优化方向：<br/>max Q(s, a)"] --> B{"Critic 在 OOD 区域<br/>给出过高估计?"}
    B -->|Yes| C["Actor 被吸引到<br/>OOD 高值区域 ❌"]
    B -->|No| D["Actor 朝着真实高值<br/>区域移动 ✅"]
```

**具体场景**：如果 Critic 对"从未见过的极端手臂姿态"错误地给出高 Q 值（因为没有训练数据校准这些区域），`direct_q` 的 Actor 会直接把动作推向这些姿态——导致策略产生不可执行的动作。

**缓解手段**：
1. 确保 Critic calibration 已经 ready（通过 Worker 门控保证）
2. 使用 `temporal_bc_relative` Critic 架构（相对 BC 的增量估值，OOD 区域自然回落到 0）
3. 监控 `actor/action_grad_norm` 和 `critic/td_abs_p90`，发现异常立即停止

### 5.7 关键配置

```yaml
algorithm:
  actor_objective: direct_q

actor:
  model:
    rl_head_config:
      chunk_sac:
        path_entropy_coef: 0.0     # 通常设为 0，纯 Q 最大化
        actor_backprop_steps: 1     # 从 1 开始，稳定后再增大
```

---

## 六、方案三：`sac_flow_g` — 带熵正则的 Flow-G SAC

### 6.1 核心思想

> **一句话**：在预训练 flow velocity 上叠加一个可训练的 adapter（Flow-G），用完整的 SAC 框架（可学习熵温度 + BC warmup + reference 约束）训练这个 adapter，同时冻结 GR00T 主干。

这是四种方案中最复杂、但理论保证最完整的方案。它的哲学是：**不直接改动 GR00T 的预训练权重，而是在它的输出上加一层"修正"，用完整 SAC 理论训练这层修正**。

### 6.2 Flow-G Adapter 是什么

Flow-G 不是另一个完整的 GR00T 模型。它是一个轻量 adapter，对预训练 velocity 施加可训练修正：

$$
v_{\text{total}}(x_t, t, s) = \text{FlowGAdapter}\Big(v_{\text{pretrained}}(x_t, t, s),\; x_t,\; t\Big)
$$

> **一句话**：把预训练模型的输出当作"基础提案"，用一个小网络对它做调整。

**关键设计**：
- `freeze_pretrained_velocity: true` 时，GR00T 主干被冻结，只有 adapter 被更新
- Adapter 初始化为 identity（输出 = 输入），训练开始时行为和原始 BC 策略完全相同
- Actor optimizer 主要更新 adapter 参数，显存需求远小于微调整个 GR00T

**代入例子理解**：GR00T 预训练的 velocity 网络说"向左伸手 5cm/step"，Flow-G adapter 在训练初期输出完全相同的"向左伸手 5cm/step"（因为初始化为 identity）。随着 SAC 训练进行，adapter 可能修正为"向左伸手 5.5cm/step"——只做微小调整，不会产生剧变。

### 6.3 正常 SAC 阶段

当没有额外 reference gate 时：

$$
\mathcal{L}_{\text{actor}} = \mathbb{E}_{s \sim \mathcal{D}}\left[\alpha \cdot \log \pi(\mathbf{a}|s) - \min(Q_1(s, \mathbf{a}), Q_2(s, \mathbf{a}))\right]
$$


> **一句话**：让策略在最大化 Q 值的同时保持足够的随机性（高熵），防止过早收敛。

**与 `direct_q` 的关键区别**：这里的 $\alpha$ 不是固定常数——它是通过以下 loss 自动调节的可学习参数：

$$
\mathcal{L}_{\alpha} = -\alpha \cdot \Big(\mathbb{E}[\log \pi(\mathbf{a}|s)] + \bar{H}\Big)
$$

| 符号 | 含义 |
|------|------|
| $\alpha$ | 熵温度（可学习） |
| $\mathbb{E}[\log \pi]$ | 当前策略的平均负熵 |
| $\bar{H}$ | 目标熵（超参） |

**自动调节逻辑**：
- 如果当前策略的熵 **低于** 目标 → $\mathcal{L}_\alpha < 0$ → $\alpha$ 增大 → Actor loss 中熵项权重增大 → 策略被迫更随机
- 如果当前策略的熵 **高于** 目标 → $\alpha$ 减小 → 策略可以更确定

这是 `sac_flow_g` 独有的能力——只有它创建了 alpha optimizer。

### 6.4 训练的三个阶段

`sac_flow_g` 的训练不是一步到位的，而是分三个阶段逐步推进：

```mermaid
flowchart LR
    A["Critic Warmup<br/>只训练 Critic<br/>Actor 不动"] --> B["BC Warmup<br/>Actor 用 expert 数据<br/>做动作模仿"]
    B --> C["正常 SAC<br/>Actor 用 Q 梯度<br/>+ 可选 BC 正则"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

**阶段一：Critic Warmup**（`critic_warmup_updates` 步）

所有四种方案都有这个阶段——在 Critic 还没学会估值之前不更新 Actor。

**阶段二：BC Warmup**（`sac_flow_bc_warmup_updates` 步，仅 `sac_flow_g`）

```text
update_step ∈ [critic_warmup_updates, critic_warmup_updates + sac_flow_bc_warmup_updates)
```

在这个窗口内，Actor 不使用 Q 梯度，而是纯粹做 expert 动作模仿：

$$
\mathcal{L}_{\text{warmup}} = c_{\text{bc\_warmup}} \cdot \|\mathbf{a}_{\text{policy}} - \mathbf{a}_{\text{expert}}\|^2
$$

**为什么需要这一步？** Flow-G adapter 刚初始化为 identity，进入 Q 优化前需要先确认它能产生合理的动作。如果直接用 Q 梯度推，可能在 adapter 还没"热身"时就把它推到奇怪的状态。

**代入例子**：打开柜门任务中，前 200 步（假设 `sac_flow_bc_warmup_updates: 200`），adapter 只在专家"打开柜门"的示教数据上做模仿。200 步后它已经能稳定产生"伸手→抓→拉"的动作序列，这时再引入 Q 梯度做精细优化。

**阶段三：正常 SAC + 可选 BC 正则**

$$
\mathcal{L} = \underbrace{\alpha \log \pi - \min(Q_1, Q_2)}_{\text{SAC Actor loss}} + \underbrace{c_{\text{bc}} \cdot \|\mathbf{a}_{\text{policy}} - \mathbf{a}_{\text{expert}}\|^2}_{\text{持续 BC 正则（可选）}}
$$

当 `sac_bc_coef > 0` 时，即使在正常 SAC 阶段，也会额外采样 expert batch 做 BC 正则。这防止策略在 Q 优化过程中完全偏离 expert 行为。

### 6.5 冻结 BC Reference 约束

这是 `sac_flow_g` 最独特的安全机制。启用 `actor_reference.enabled` 后：

**Step 1：同一噪声，两条路径**

```text
初始噪声 x₀ → Flow-G Actor → a_actor（当前策略动作）
初始噪声 x₀ → 冻结 BC Policy（禁用 adapter） → a_bc（BC 参考动作）
```

**Step 2：四重 gate 决定是否允许 Q 更新**

只有**同时满足**以下所有条件的样本才执行 Q 最大化：

| 条件 | 含义 | 典型阈值 |
|------|------|----------|
| $Q_1(\text{actor}) - Q_1(\text{bc}) \geq \delta_Q$ | Actor 动作的 Q1 比 BC 动作高 | `min_q_advantage: 0.0` |
| $Q_2(\text{actor}) - Q_2(\text{bc}) \geq \delta_Q$ | 两个 Critic 都认为 Actor 更好 | 同上 |
| $|Q_1 - Q_2|$ 的 disagreement $\leq \epsilon_Q$ | 两个 Critic 意见一致 | `max_critic_disagreement: 0.1` |
| $\|\mathbf{a}_{\text{actor}} - \mathbf{a}_{\text{bc}}\|^2 \leq \epsilon_a$ | 动作没偏离 BC 太远 | `max_action_mse: 0.01` |


**最终 loss 还始终包含 proximity 项**：

$$
\mathcal{L} = \text{gated\_SAC\_loss} + c_{\text{mse}} \cdot \|\mathbf{a}_{\text{actor}} - \mathbf{a}_{\text{bc}}\|^2
$$

> **一句话直觉**：只有当"Actor 确实比 BC 好"且"两个 Critic 都同意"且"动作没飘太远"时，才用 Q 梯度推。否则只用 proximity loss 把 Actor 拉回 BC 附近。

**为什么这么复杂？** 这解决了 `direct_q` 的核心风险：如果 Critic 对某些 OOD 区域过估计，单个 Critic 可能错误引导 Actor。四重 gate 确保只有**两个 Critic 都有信心、且动作没偏离 BC 太远**时才信任 Q 梯度。

**监控指标**：`sac/reference_gate_fraction` 显示通过 gate 的样本比例。如果这个值接近 0，说明 gate 太严——几乎没有样本获得 Q 梯度，Actor 实际上退化为纯 BC 正则。

### 6.6 关键配置

```yaml
algorithm:
  actor_objective: sac_flow_g
  critic_warmup_updates: 800
  sac_flow_bc_warmup_updates: 200
  sac_flow_warmup_bc_coef: 1.0
  sac_bc_coef: 0.1                # 正常阶段的持续 BC 系数
  entropy_tuning:
    alpha_type: softplus           # alpha = softplus(log_alpha)，保证正
    initial_alpha: 0.01
    target_entropy: -1.0
    optim:
      lr: 3.0e-4
      lr_scheduler: torch_constant
  actor_reference:
    enabled: true
    min_q_advantage: 0.0
    max_critic_disagreement: 0.1
    max_action_mse: 0.01
    action_mse_coefficient: 1.0

actor:
  model:
    rl_head_config:
      chunk_sac:
        flow_g:
          enabled: true
          freeze_pretrained_velocity: true
```

### 6.7 数据依赖

`sac_flow_g` 是四种方案中数据要求最高的：

| 数据需求 | 原因 |
|----------|------|
| Expert replay stratum | BC warmup 和持续 BC 正则需要 |
| 冻结 BC reference model | Reference gate 比较需要 |
| 充足的 online rollout | SAC 的 replay 多样性 |

如果你的 replay 中没有 expert stratum（没有标注的成功轨迹），不应该启用 `sac_bc_coef > 0` 或 BC warmup。

---

## 七、方案四：`awr` — 动作空间的 Advantage-Weighted BC

### 7.1 核心思想

> **一句话**：用 Q advantage 给 replay 动作加权，但不在 flow 空间做模仿，而是直接在**最终动作空间**最小化加权 MSE。

这是介于 `awr_flow` 和 `direct_q` 之间的方案。它保留了 AWR 的保守性（Q 只产生权重，不直接提供梯度方向），但梯度通过动作采样过程反传——和 `awr_flow` 的"不反传"形成对比。

### 7.2 计算过程

$$
\mathbf{a}_{\text{policy}} = \pi_\theta(s) \qquad \text{（可微采样）}
$$

$$
A = \min(Q_1(s, \mathbf{a}_{\text{data}}), Q_2(s, \mathbf{a}_{\text{data}})) - \min(Q_1(s, \mathbf{a}_{\text{policy}}), Q_2(s, \mathbf{a}_{\text{policy}}))
$$

$$
w = \frac{\exp\left(\text{clamp}(A / \tau,\; -\infty,\; \log w_{\max})\right)}{\text{mean}(w_{\text{batch}})}
$$

$$
\mathcal{L}_{\text{awr}} = \frac{1}{N} \sum_{i} w_i \cdot \|\mathbf{a}_{\text{policy}}^{(i)} - \mathbf{a}_{\text{data}}^{(i)}\|^2_{\text{masked}}
$$

> **一句话**：策略采样一个动作，计算它和 replay 动作的 MSE，但给 MSE 乘上一个权重——replay 动作越好（advantage 越大），这个 MSE 的权重越大。

**梯度路径**：

```mermaid
flowchart LR
    A["Actor 参数 θ"] --> B["Flow 去噪采样"]
    B --> C["a_policy"]
    C --> D["MSE(a_policy, a_data)"]
    D --> E["× 权重 w"]
    E -.->|"梯度反传"| A

    F["Q(s, a_data) - Q(s, a_policy)"] --> G["权重 w"]
    style F fill:#f5f5f5
    G -.->|"detached"| E
```

### 7.3 与 `awr_flow` 的关键区别

这两者容易混淆。**它们的名字相似但本质完全不同**：

| 维度 | `awr_flow` | `awr` |
|------|-----------|-------|
| 训练空间 | Flow velocity / SFT target | 最终采样动作空间 |
| 梯度是否穿过 Actor forward | **否** | **是** |
| loss 定义 | GR00T 原生 flow-matching loss | 动作 MSE |
| Actor forward 参与 loss？ | 否（只参与 Q 计算来生成权重） | 是（`a_policy` 直接进 MSE） |
| `policy_actions.grad` 指标 | 固定为 0 | 有值 |
| 权重归一化 | 跨 rank 分布式归一化 | 当前 micro-batch 均值归一化 |


**一个比喻帮助区分**：

- `awr_flow` = "用 GR00T 的教学方式（flow-matching）重新教它，只是把教材按好坏排了优先级"
- `awr` = "直接在动作结果上比较，告诉 GR00T '你的输出应该更像这个 replay 动作'"

### 7.4 为什么在动作空间做 MSE 而不是 flow 空间

**优点**：

1. 梯度信号更直接——直接惩罚最终动作的偏差，而不是中间的 velocity 偏差
2. 与 Critic 评估的对象一致——Critic 评估的是最终动作，MSE 也在最终动作上计算
3. 不需要 flow-matching 训练接口——任何能可微采样的 Actor 都能用

**缺点**：

1. 对多模态动作分布，MSE 会把多个有效模式"平均化"——如果打开柜门有"推"和"拉"两种方式，MSE 可能产生介于两者之间的无效中间动作
2. 梯度需要穿过整个去噪链（显存成本高于 `awr_flow`）
3. 在高维动作空间（2480 维），MSE 的每维贡献可能差异很大

### 7.5 关键配置

```yaml
algorithm:
  actor_objective: awr
  awr_temperature: 0.1         # 比 awr_flow 的 1.0 小很多 → 更集中于最好样本
  awr_max_weight: 20.0
```

**注意**：`actor_use_awr_weights` 是给 `awr_flow` 用的 Worker 开关，**不影响** `awr` objective 内部的权重计算。`awr` 始终自己计算 advantage 权重。

---

## 八、Critic 架构：两种选择

四种 Actor objective 可以和两种 Critic 架构自由组合。

### 8.1 `flat_absolute`：简单直接

$$
Q(s, \mathbf{a}) = \text{MLP}\Big(\text{concat}\big[\phi(s),\; \text{flatten}(\mathbf{a}_{0:H})\big]\Big)
$$

- 把状态特征和展平的动作块拼接起来，送入 MLP
- 直接学习绝对 Q 值
- 不需要任何额外参考动作

**代入例子**：状态特征 256 维 + 动作 40×62=2480 维 = 2736 维输入向量，经过 MLP → 标量 Q。

**适用场景**：简单任务、数据充足、不需要结构化先验。

### 8.2 `temporal_bc_relative`：相对 BC 的增量估值

$$
Q(s, \mathbf{a}) = V(s) + A(s, \mathbf{a} - \mathbf{a}_{\text{bc}}) - A(s, \mathbf{0})
$$

> **一句话**：不直接估计"这个动作值多少"，而是估计"这个动作比 BC 动作好/差多少"。

**设计动机**：在预训练策略附近做微调时，大多数动作的绝对 Q 值差异很小（都在 BC 行为附近）。直接学绝对 Q 需要 Critic 在大量接近的输入上区分微小差异——这很难学。改为学"相对于 BC 的增量"后，输入变成了差值 $\mathbf{a} - \mathbf{a}_{\text{bc}}$，微小改进对应的信号被放大了。

**结构细节**：
- 使用**时序卷积**编码动作差值序列（保留时间结构）
- $V(s)$ 部分主要由 expert/BC candidate 数据监督
- $A(s, \cdot)$ 部分主要由 exploration candidate 监督
- 需要 replay 中携带 `chunk_sac_bc_reference_action`

**对 Actor objective 的影响**：
- 当动作偏离 BC 很远时，$A(s, \mathbf{a} - \mathbf{a}_{\text{bc}})$ 自然回落（没有训练数据支撑远处的 advantage 估计）
- 这为 `direct_q` 提供了隐式的行为约束——Actor 不太可能被推到离 BC 很远的地方，因为 Critic 在那些区域的估值不高

### 8.3 组合建议

| Actor Objective | 推荐 Critic | 理由 |
|-----------------|------------|------|
| `awr_flow` | 两者皆可 | 不依赖 Q 梯度质量 |
| `direct_q` | `temporal_bc_relative` | 提供隐式行为约束，防 OOD 利用 |
| `sac_flow_g` | `temporal_bc_relative` | 与 reference gate 互补 |
| `awr` | 两者皆可 | Q 只用于权重，不传梯度 |

---

## 九、四种方案的完整对比

### 9.1 总览表

| 维度 | `awr_flow` | `direct_q` | `sac_flow_g` | `awr` |
|------|-----------|-----------|-------------|-------|
| **Actor 主要目标** | 加权 flow-matching | 最大化 Q | $\alpha\log\pi - Q$ | 加权动作 MSE |
| **Q 梯度穿过动作？** | ❌ 否 | ✅ 是 | ✅ 是（通过 adapter） | ❌ Q 不传，MSE 传 |
| **Q 的角色** | 产生样本权重 | 直接定义 loss | 直接定义 loss + gate | 产生 advantage 权重 |
| **可学习熵温度** | 无 | 无 | ✅ 有 alpha optimizer | 无 |
| **BC warmup** | 无 | 无 | ✅ 支持 | 无 |
| **Reference gate** | 无 | 无 | ✅ 支持 | 无 |
| **Expert 数据依赖** | 否 | 否 | BC warmup/正则时需要 | 否 |
| **训练复杂度** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **突破 replay 上限** | ❌ 不能 | ✅ 能 | ✅ 能 | ❌ 不能 |
| **主要风险** | 受 replay 限制 | 利用 Critic 误差 | 配置复杂，训练阶段多 | MSE 对多模态不友好 |

### 9.2 决策流程图

```mermaid
flowchart TD
    START["选择 Actor Objective"] --> Q1{"Critic 是否可信？<br/>（calibration 正常、TD error 小）"}
    Q1 -->|"不确定"| AWR_FLOW["选 awr_flow<br/>最保守，不用 Q 梯度"]
    Q1 -->|"可信"| Q2{"replay 中有足够<br/>成功数据？"}
    Q2 -->|"有"| Q3{"是否需要突破<br/>replay 行为上限？"}
    Q2 -->|"没有"| DIRECT_Q["选 direct_q<br/>不依赖 replay 好动作"]
    Q3 -->|"不需要"| Q4{"偏好哪种训练空间？"}
    Q3 -->|"需要"| Q5{"是否有 expert 数据<br/>+ 可接受复杂配置？"}
    Q4 -->|"Flow 空间"| AWR_FLOW
    Q4 -->|"动作空间"| AWR["选 awr<br/>动作 MSE 回归"]
    Q5 -->|"是"| SAC_FLOW_G["选 sac_flow_g<br/>完整 SAC + 安全约束"]
    Q5 -->|"否"| DIRECT_Q

    style AWR_FLOW fill:#e3f2fd
    style DIRECT_Q fill:#fce4ec
    style SAC_FLOW_G fill:#e8f5e9
    style AWR fill:#fff3e0
```

---

## 十、实操指南：从零开始配置

### 10.1 推荐的入门路径


**第一次做 GR00T Chunk-SAC 后训练**，建议按以下顺序尝试：

| 步骤 | 方案 | 目的 |
|------|------|------|
| 1 | `awr_flow` | 验证 Critic 训练正常、replay 数据流通、基本指标合理 |
| 2 | `direct_q`（低 backprop_steps） | 验证 Q 梯度能稳定传播、策略在正确方向改进 |
| 3 | `sac_flow_g`（如果需要） | 获得最完整的理论保证和安全约束 |

**不建议**一上来就用 `sac_flow_g`——它的配置项太多，如果出了问题很难定位是哪个环节（BC warmup？alpha？reference gate？）。先用简单方案验证基础设施，再逐步加复杂度。

### 10.2 关键监控指标

训练时必须盯住的核心指标：

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `sac/actor_updated` | 大部分 step 为 True | 长期为 False → 门控条件检查 |
| `sac/actor_q` | 逐渐上升 | 持续下降 → Critic 或 Actor 有问题 |
| `sac/advantage` | 在 0 附近波动，逐渐缩小 | 持续为大负数 → replay 比策略好太多 |
| `actor/action_grad_norm` | 有值（`awr_flow` 除外） | 突然暴增 → 梯度爆炸 |
| `critic/td_abs_p90` | 稳定或下降 | 持续增大 → Critic 不收敛 |
| `sac/awr_weight_mean` | 略大于 1 | 远大于 1 → temperature 太小或 max_weight 太大 |
| `sac/reference_gate_fraction` | 0.3~0.8（`sac_flow_g`） | 接近 0 → gate 太严；接近 1 → gate 太松 |
| `sac/alpha`（`sac_flow_g`） | 逐渐稳定到一个值 | 持续增大 → 策略坍缩；趋近 0 → 探索不足 |

### 10.3 常见故障排除

**问题 1：Actor 长期不更新**

检查门控条件：
1. replay size 够了吗？→ 查 `replay_buffer.min_buffer_size`
2. critic warmup 完了吗？→ 查当前 step vs `critic_warmup_updates`
3. 有合格样本吗？→ 查 `sac/actor_selected_fraction`，如果为 0 说明 filter 太严

**问题 2：`direct_q` 策略飘走**

症状：`actor/action_grad_norm` 暴增、env reward 暴跌。

解法：
- 减小 `actor_backprop_steps`（如从 3 降到 1）
- 换用 `temporal_bc_relative` Critic 架构
- 添加 `path_entropy_coef > 0`（如 0.01）
- 或切换到 `sac_flow_g` 并启用 reference gate

**问题 3：`sac_flow_g` reference gate 通过率为 0**

所有样本都被 gate 拒绝 → Actor 实际上没有 Q 梯度 → 退化为纯 BC 正则。

解法：
- 放松 `max_action_mse`（如从 0.01 到 0.05）
- 放松 `max_critic_disagreement`（如从 0.1 到 0.5）
- 确认 BC warmup 是否足够（adapter 是否已经能产生接近 BC 的动作）

**问题 4：`awr_flow` 的 AWR 权重饱和**

症状：`sac/awr_saturation_fraction` 接近 1.0 — 几乎所有权重都触顶。

解法：
- 增大 `awr_temperature`（如从 0.5 到 2.0）→ 权重更均匀
- 增大 `awr_max_weight`（如从 10 到 50）→ 允许更大差异
- 但要注意：增大后可能单个好样本主导整个 batch

---

## 十一、与 Critic 架构的交互效果

### 11.1 `flat_absolute` + 各方案

| 方案 | 效果 |
|------|------|
| + `awr_flow` | 基础组合，Critic 只需给出正确的相对排序 |
| + `direct_q` | ⚠️ 风险较高：Critic 对 OOD 动作无约束，容易被利用 |
| + `sac_flow_g` | reference gate 提供一定保护，但 Critic 本身无结构化约束 |
| + `awr` | 安全，Q 只做权重不传梯度 |

### 11.2 `temporal_bc_relative` + 各方案

| 方案 | 效果 |
|------|------|
| + `awr_flow` | Critic 学习更精准的增量排序，权重更可靠 |
| + `direct_q` | ✅ 推荐组合：Critic 在远离 BC 的区域 advantage 自然衰减，提供隐式约束 |
| + `sac_flow_g` | ✅ 最安全组合：Critic 约束 + reference gate + BC 正则三重保护 |
| + `awr` | Critic 的 advantage 估计更精确，权重更有意义 |

---

## 十二、常见误区澄清

### 误区 1："`awr_flow` 和 `awr` 是同一种 loss 的两个名字"

❌ 完全不同。前者在 flow/SFT 空间训练（和 BC 用的是同一个 loss 接口），后者在最终动作空间做 MSE。前者 Actor forward 不参与 loss 计算，后者 Actor forward 直接参与。

### 误区 2："`direct_q` 的 `path_entropy_coef` 等同于 SAC 的 alpha"

❌ `path_entropy_coef` 是一个**固定超参**，不会自动调节，也不会影响 Critic 的 Bellman target。SAC 的 alpha 是可学习的、有 optimizer 的、且会加入 target 计算。

### 误区 3："`actor_use_awr_weights: false` 能关闭 `awr` 方案的权重"

❌ `actor_use_awr_weights` 是 Worker 层为 `awr_flow` 准备权重的开关。`awr` objective 内部**始终自己计算** advantage 权重，不受这个开关控制。

### 误区 4："BC warmup = Critic warmup"

❌ Critic warmup 期间 Actor 完全不更新。BC warmup 是 Critic warmup **之后**的一个额外阶段，只有 `sac_flow_g` 支持——在这个阶段 Actor 更新，但只做 expert 模仿，不用 Q 梯度。

时间线：
```text
[0, critic_warmup) → Actor 不动，只训练 Critic
[critic_warmup, critic_warmup + bc_warmup) → Actor 做 expert BC（仅 sac_flow_g）
[critic_warmup + bc_warmup, ∞) → 正常 Actor 更新
```

### 误区 5："`actor_reference` 和 `temporal_bc_relative` 是同一个东西"

❌ 两者都使用 BC reference 动作，但目的完全不同：

| | `actor_reference` | `temporal_bc_relative` |
|---|---|---|
| 属于 | Actor 约束 | Critic 架构 |
| 作用 | 决定哪些样本可以做 Q 优化 | 决定 Critic 估值的数学形式 |
| 仅限 | `sac_flow_g` | 任何 Actor objective |

---

## 十三、总结

### 13.1 选型一句话

- **稳当优先** → `awr_flow`：保持 GR00T 原始行为，慢慢学
- **突破极限** → `direct_q`：直接追 Critic，但要小心
- **理论完备** → `sac_flow_g`：完整 SAC 框架，配置最多但最可控
- **折中之选** → `awr`：在动作空间做保守回归

### 13.2 它们不是孤立的四个开关

四种方案共享：

- 同一个 GR00T N1.7 模型和 checkpoint
- 同一个 Twin Critic 和 Target Critic
- 同一个 Replay Buffer 和数据流
- 同一个 FSDP Worker 和分布式训练基础设施
- 同一套 Actor 更新门控逻辑
- 同一个 Critic calibration 机制

切换方案只需改 `algorithm.actor_objective` 一项配置（以及该方案特有的额外配置），不需要重写训练系统。但注意：**从一个方案切到另一个方案后直接恢复旧 checkpoint 可能触发 config hash 校验失败**——正式切换应视为新的训练语义。

---

## 延伸阅读

- [SAC (Soft Actor-Critic) 完整推导](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 最大熵 RL 的数学基础
- [Q 加权 Flow 策略](/前置知识/001v_前置知识_Q加权Flow策略_不穿透去噪链的RL训练) — `awr_flow` 思路的理论来源
- [AWR 优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) — 加权模仿的一般理论
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — GR00T 动作生成的底层机制
- [RLinf BC 到 RL 的 ACT 后训练架构](./RLinf_BC到RL的ACT后训练架构) — PPO 路线的完整对比
- [GR00T N1.7 深度解析系列](/系列/groot_n1d7_deep_dive/index) — 理解 GR00T 模型架构
- [动作分块 RL 基础](/系列/groot_rl_deep_dive/06_动作分块RL基础_QChunking到AQC回顾) — Chunk-level 估值的理论动机
