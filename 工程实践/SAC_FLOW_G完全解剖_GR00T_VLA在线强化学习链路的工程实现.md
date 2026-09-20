---
title: "SAC_FLOW_G 完全解剖：GR00T VLA 在线强化学习链路的工程实现"
order: 12
tags: [强化学习, SAC, Flow Matching, Flow-G, GR00T, 在线RL, 工程实践, Chunk-SAC]
category: 工程实践
star: 5
---

# SAC_FLOW_G 完全解剖：GR00T VLA 在线强化学习链路的工程实现

> **一句话**：本文逐层拆解 GR00T N1.7 上 SAC-Flow-G 在线 RL 后训练的核心链路——从 Critic 离线预热、到 Flow-G 门控 Actor 更新、到熵温度自适应——五个组件让训练跑起来，一条可复现的基线管线。

**知识链接**：
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 框架的完整原理
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — Flow Matching 基础
- [SAC-Flow：用 SAC 直接训练 Flow 策略](/论文综述/079_SAC_Flow_用SAC直接训练Flow策略) — 学术论文精读
- [GR00T N1.7 四种 RL 方案全景对比](./GR00T_N1d7_四种RL方案全景对比) — 四种方案的宏观定位
- [GR00T N1.7 Chunk-SAC 四种 Actor 目标详解](./GR00T_N1d7_ChunkSAC四种Actor目标详解) — Actor 目标变体
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Critic 基础
- [Replay Buffer 经验回放](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — 离线数据复用

---

## 一、这篇文章要解决什么问题

你手上有一个 BC 预训练好的 GR00T N1.7 模型，成功率约 60%。你决定用 SAC-Flow-G 路线做在线 RL 后训练。

**核心问题**：SAC-Flow-G 链路包含哪些组件？它们如何协作完成"从 BC 到 RL"的跨越？

训练链路被精简为五个核心组件：

1. Episode replay + Chunk-SAC TD Critic 更新
2. 调度型 SAC-Flow-G Actor 更新
3. 可选的连续 BC loss（`sac_bc_coef`）
4. 熵温度自动优化
5. Target-Critic 软更新（`tau`）

本文的目标：**把这五个组件逐一拆解到代码级别，让你能完全理解并复现这条核心链路。**

---

## 二、贯穿全文的例子

> **任务**：GR00T N1.7 双臂人形机器人在 MiArena/Isaac Sim 中执行 `open_drawer`（拉开抽屉）。
>
> - **动作空间**：62 维 camera-frame Rot6D（左臂位置 3D + 旋转 6D + 右臂位置 3D + 旋转 6D + 左手 22D + 右手 22D）
> - **动作块**：chunk_length = 16，replan_steps = 16（每 16 步重新规划）
> - **Critic 架构**：flat_absolute（双 Q 网络，输入状态 + 扁平动作块）
> - **环境并发**：训练 32 envs，评测 48 envs
> - **BC 基线**：约 56-63% 成功率（open_drawer）
> - **目标**：通过在线 RL 提升成功率，不退化

---

## 三、整体架构：两阶段训练

整条管线分为两个阶段，由两个 Hydra 配置驱动：

```mermaid
flowchart TD
    subgraph "Stage 1：Critic 离线预热"
        BC["BC checkpoint<br/>(GR00T N1.7)"] --> REPLAY["BC 数据 → Replay Buffer"]
        REPLAY --> CRITIC["800 次 Critic-only 更新<br/>（Actor 冻结为 identity）"]
        CRITIC --> CKPT["offline_update_800<br/>checkpoint"]
    end

    subgraph "Stage 2：在线 SAC-Flow-G"
        CKPT --> RESUME["恢复 Critic + Identity Flow-G"]
        RESUME --> ROLLOUT["环境 Rollout<br/>（Flow-G Actor 采样动作）"]
        ROLLOUT --> BUFFER["Episode → Replay Buffer"]
        BUFFER --> C_UPDATE["Critic TD 更新<br/>（每 runner step 35 次）"]
        C_UPDATE --> A_UPDATE["Actor 更新<br/>（每 critic_actor_ratio 次 Critic 更新后 1 次）"]
        A_UPDATE --> ALPHA["α 温度更新"]
        ALPHA --> TAU["Target Critic EMA"]
        TAU --> ROLLOUT
    end

    style BC fill:#e1f5fe
    style CKPT fill:#fff3e0
    style A_UPDATE fill:#e8f5e9
```

### 对应的启动配置

| 阶段 | 配置文件 | 关键参数 |
|------|----------|----------|
| Stage 1 | `miarena_r1_sac_flow_g_critic_warmup_gr00t_n1d7.yaml` | `mode: offline_pretrain`, `offline_updates: 800` |
| Stage 2 | `miarena_r1_sac_flow_g_adapter_gr00t_n1d7.yaml` | `mode: online`, `resume_dir: <Stage1 checkpoint>` |

两者统一由 `run_miarena_groot_chunk_sac.sh` 启动。Stage 2 **必须**从 Stage 1 的 `offline_update_800` checkpoint 恢复。

---

## 四、组件一：Episode Replay 与 Chunk-SAC TD Critic 更新

### 5.1 数据流：从环境到 Replay

每个 runner step，32 个并行环境各执行一个完整的动作块（16 步）。每个 episode 被分割为若干 chunk transition：

```
一条 chunk transition = (s, a_chunk, rewards[0:16], valid[0:16], s_next, done)
```

其中：
- `s`：chunk 开始时的状态（图像 + 本体感觉）
- `a_chunk`：16 步动作序列，形状 `[16, 62]`
- `rewards[0:16]`：16 步内的逐步奖励
- `valid[0:16]`：标记哪些步是有效的（episode 提前终止时后续步无效）
- `s_next`：chunk 结束后的下一状态
- `done`：episode 是否在此 chunk 内终止

### 5.2 Chunk-SAC TD Target 构造

Critic 的训练目标是最小化 TD error。目标值 $y$ 的构造如下：

**Step 1：先算 chunk 内的折扣回报 $R_H$**

$$
R_H = \sum_{i=0}^{H-1} \gamma^i \cdot r_i \cdot \mathbb{1}[\text{valid}_i]
$$

**这个公式在做什么**：把 chunk 内每一步的奖励按时间折扣加权求和，episode 提前终止后的无效步不计入。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\gamma^i \cdot r_i$ | **打折奖励** | 第 $i$ 步的奖励乘上随时间衰减的折扣因子，越晚拿到的奖励打折越多 |
| $\mathbb{1}[\text{valid}_i]$ | **有效性开关** | episode 还没结束时开着（=1），一旦提前终止就关掉（=0），后续步骤不再计入 |
| $\sum_{i=0}^{H-1}$ | **chunk 内求和器** | 把 chunk 长度 $H=16$ 步内所有打折奖励加总成一个标量 $R_H$ |

**用人话读**："把这个 16 步动作块里每一步拿到的奖励打折后加起来，一旦 episode 提前结束就不再往后加。"

**为什么是这个形式**：Chunk-SAC 的 TD target 需要知道"这一整个动作块本身带来了多少即时收益"，折扣保证近期奖励权重更高，valid mask 处理 episode 在 chunk 内提前终止的边界情况。

**数值代入**：假设 `terminal_success` 模式，$\gamma=1.0$，episode 在 chunk 第 12 步成功终止：
- $r_0 = r_1 = \ldots = r_{11} = 0$，$r_{12} = 1$
- $\text{valid}_0 = \ldots = \text{valid}_{12} = 1$，$\text{valid}_{13} = \ldots = \text{valid}_{15} = 0$
- $R_H = 1.0^{12} \times 1 = 1.0$

**逐符号拆解补充**：

| 符号 | 含义 | 具体值 |
|------|------|--------|
| $H$ | chunk 长度 | 16 |
| $\gamma$ | 折扣因子 | 0.99（`terminal_success` 模式下用 1.0） |
| $r_i$ | 第 $i$ 步奖励 | `terminal_success` 模式：成功终止步 = +1，其余 = 0 |
:::

**Step 2：完整 Bellman target**

$$
y = R_H + \gamma^H \cdot m \cdot \big(Q^{\text{target}}(s', \mathbf{a}') - \alpha \cdot \log\pi(\mathbf{a}'|s')\big)
$$

**这个公式在做什么**：拼出 Critic 要去拟合的完整 Bellman target——chunk 内真实拿到的奖励，加上 chunk 结束后目标网络对"未来还能拿多少"的估计（扣掉熵奖励）。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $R_H$ | **已到手的钱** | chunk 内 16 步已经实际发生、确定拿到的折扣奖励 |
| $\gamma^H \cdot m$ | **未来价值的折扣开关** | 如果 episode 在 chunk 内已经终止（$m=0$），未来价值直接归零；否则按 $H$ 步折扣打折后计入 |
| $Q^{\text{target}}(s', \mathbf{a}') - \alpha\log\pi(\mathbf{a}'|s')$ | **对未来的估价（扣掉探索津贴）** | 目标网络对"chunk 结束后这个状态还能拿多少分"的估计，再减去当前动作有多"随机"带来的熵奖励 |

**用人话读**："target = 这个 chunk 里已经拿到的真实奖励，加上（如果 episode 还没完）目标网络对后续价值的打折估计。"

**为什么是这个形式**：这是标准 Bellman 方程在"每次决策跨越一整个动作块"场景下的推广——$m$ 处理提前终止的边界情况，$\gamma^H$ 保证折扣尺度和真实经过的步数一致，减去 $\alpha\log\pi$ 项是 SAC 最大化熵目标的体现。

**数值代入**：假设成功 episode（$m=0$）：
- $y = 1.0 + 1.0 \times 0 \times (\ldots) = 1.0$
- 成功 episode 的 target 就是 1.0，Critic 学习"这个状态-动作组合会导致成功"

假设未终止 episode（$m=1$），$Q^{\text{target}}=0.6$，$\alpha \cdot \log\pi = 0.001$：
- $y = 0 + 0.851 \times 1 \times (0.6 - 0.001) = 0.851 \times 0.599 = 0.510$

**逐符号拆解补充**：

| 符号 | 含义 | 具体值 |
|------|------|--------|
| $\gamma^H$ | H 步折扣 | $0.99^{16} = 0.851$ 或 $1.0^{16} = 1.0$ |
| $m$ | bootstrap mask | episode 在 chunk 内终止 = 0，未终止 = 1 |
| $s'$，$\mathbf{a}'$ | 下一状态、下一动作块 | chunk 结束后的观测；当前 Actor 在 $s'$ 重新采样的 16 步动作 |
:::

### 5.3 Critic Loss

$$
L_{\text{critic}} = \frac{1}{2}\Big[\big(Q_1(s, \mathbf{a}) - y\big)^2 + \big(Q_2(s, \mathbf{a}) - y\big)^2\Big]
$$

**这个公式在做什么**：让两个独立的 Q 网络各自去逼近同一个 target $y$，用均方误差衡量预测得准不准。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q_1(s,\mathbf{a})$，$Q_2(s,\mathbf{a})$ | **两个独立打分员** | 两套参数不同的 Q 网络，各自对同一个 (状态, 动作块) 打分 |
| $\big(Q_k(s,\mathbf{a}) - y\big)^2$ | **打分误差** | 打分员的估计和真实 target 差多远，差得越多 loss 越大 |
| $\frac{1}{2}[\cdot + \cdot]$ | **两人平均** | 把两个打分员的误差加起来取平均，一次反传同时训练两套参数 |

**用人话读**："两个 Q 网络各自跟 target 对齐，误差平方后取平均就是 Critic 的 loss。"

**为什么要两个 Q 网络**：这是标准的 SAC twin-Q 设计——训练时把两者的估计取最小值 $\min(Q_1,Q_2)$ 用于计算 target 和 Actor loss，能有效防止单个 Q 网络对某些动作过度乐观（高估）的问题。
:::

两个 Critic 网络独立训练，取 $\min(Q_1, Q_2)$ 作为 Actor 的 Q 估计以防止高估。

### 5.4 代码对应

核心计算在 `rlinf/algorithms/chunk_sac.py` 中：

```python
def chunk_sac_td_target(
    rewards, valid, bootstrap_mask, next_q, next_log_pi, alpha, gamma
):
    chunk_return = discounted_chunk_return(rewards, valid, gamma)
    continuation = gamma ** rewards.shape[-1]
    return chunk_return + continuation * bootstrap_mask * (next_q - alpha * next_log_pi)
```

`discounted_chunk_return` 对 chunk 内 reward 做折扣加权求和，`continuation` 是 $\gamma^H$，`bootstrap_mask` 就是 $m$。

---

## 五、组件二：SAC-Flow-G Actor 更新

这是整条链路最关键的组件——如何让 Q 梯度穿过 Flow 的多步采样更新 Actor 参数，同时保持梯度稳定。

### 6.1 问题回顾：为什么普通 Flow + SAC 会梯度爆炸

GR00T N1.7 的动作头使用 Flow Matching：从初始噪声 $A_0 \sim \mathcal{N}(0,I)$ 出发，经 $K$ 步 Euler 积分生成动作：

$$
A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot v_\theta(t_i, A_{t_i}, s)
$$

**这个公式在做什么**：从纯噪声出发，每一步都让速度网络指出"该往哪个方向挪一小步"，走完 $K$ 步就到达最终动作。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $A_{t_i}$ | **当前位置** | Euler 积分第 $i$ 步时，动作在"从噪声到真实动作"这条路径上的当前坐标 |
| $v_\theta(t_i, A_{t_i}, s)$ | **导航员** | 速度网络看着当前位置、时间、状态 $s$，指出"往哪个方向走" |
| $\Delta t_i \cdot v_\theta(\cdot)$ | **这一步的位移** | 方向乘以步长，就是这一步实际移动的距离 |

**用人话读**："站在当前位置，问速度网络该往哪走，走一小步，重复 $K$ 次就走到终点动作。"

**为什么是这个形式**：这是 Flow Matching 生成动作的标准 Euler 积分公式——用有限步数（GR00T 中 $K=4$）近似连续的常微分方程（ODE）轨迹，步数越多越精确但推理越慢。
:::

SAC 的 Actor loss 需要把 $\nabla_\theta Q$ 从最终动作 $A_{t_K}$ **反传穿过所有 $K$ 步**到达参数 $\theta$。这等价于 K 层 RNN 的 BPTT——梯度指数级增长或衰减。

在 GR00T 的实际配置中 $K=4$（`denoising_steps: 4`），虽然步数不多，但在 62 维动作空间上仍然存在梯度不稳定风险。

### 6.2 Flow-G 门控解决方案

Flow-G 的核心思想是在原始 Flow 速度网络之上插入一个**可学习的门控适配器**，控制每步更新幅度。具体结构：

$$
A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot \Big[v_{\theta_{\text{frozen}}}(t_i, A_{t_i}, s) + g_i \odot \big(\hat{v}_\phi(t_i, A_{t_i}, s) - v_{\theta_{\text{frozen}}}(t_i, A_{t_i}, s)\big)\Big]
$$

**这个公式在做什么**：在冻结的 BC 速度基础上，用一个逐维度门控信号，混入 adapter 给出的候选速度——门开多大，就允许 RL 学到的新方向偏离 BC 多远。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $v_{\theta_{\text{frozen}}}(t_i, A_{t_i}, s)$ | **BC 老司机** | 预训练好、被冻结不再更新的速度网络，代表"安全但不一定最优"的方向 |
| $\hat{v}_\phi(t_i, A_{t_i}, s)$ | **RL 新学员的建议** | 一个独立训练的 adapter 网络，提出它认为更好的速度方向 |
| $g_i \odot (\hat{v}_\phi - v_{\theta_{\text{frozen}}})$ | **按门控比例采纳新建议** | 门控 $g_i$ 决定在多大程度上把老司机的方向替换成新学员的方向 |

**用人话读**："最终方向 = 老司机的方向 + 门控决定的、往新学员方向的偏移量，门开得越大偏移越多。"

**为什么是这个形式**：直接让 adapter 输出速度会导致梯度穿过 $K$ 步链式求导时不稳定；用门控只允许"渐进式偏离"冻结的 BC 速度，相当于给每一步的更新幅度装了一个可学习的安全阀。
:::

$$
g_i = \sigma\big(z_\phi(t_i, A_{t_i}, s)\big) \in (0, 1)^{62}
$$

**这个公式在做什么**：把一个未经限制的 logit 压缩到 $(0,1)$ 区间，作为每个动作维度独立的门控开合程度。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $z_\phi(t_i, A_{t_i}, s)$ | **门控原始信号** | 一个小网络输出的、范围不受限的 62 维 logit 向量 |
| $\sigma(\cdot)$ | **压缩阀门** | sigmoid 函数把任意实数压缩到 $(0,1)$，保证门控是一个合法的"比例" |
| $\in (0,1)^{62}$ | **62 个独立开关** | 62 维动作空间里每一维都有自己独立的门控值，不共享 |

**用人话读**："每个动作维度都有一个 0 到 1 之间的开合度，由一个小网络根据当前状态算出来。"

**为什么每维独立**：62 维动作里不同维度（如手指关节 vs 手臂位置）对 RL 修正的需要程度可能完全不同，逐维度门控比一个全局标量门控更灵活。
:::

**逐符号拆解补充**：

| 符号 | 含义 | 具体值 |
|------|------|--------|
| $v_{\theta_{\text{frozen}}}$ | 冻结的预训练速度网络 | BC checkpoint 中的原始 Flow 权重，不参与梯度 |
| $\hat{v}_\phi$ | Adapter 的候选速度 | 一个独立的 MLP，hidden_dim=256 |

**初始状态**：训练开始时 `gate_bias=0.0`，sigmoid(0)=0.5；但配合 `gate_scale=2.0`，实际初始 gate 输出接近 0（因为 adapter 权重初始化很小），所以 Actor 初始行为约等于冻结的 BC。这就是文档中说的"identity start"。

**代入数字**（1 维简化）：
- 冻结速度 $v_{\text{frozen}} = 0.4$
- Adapter 候选速度 $\hat{v} = 0.7$
- 门控 $g = 0.1$（训练初期，adapter 还没学到什么）
- 实际速度 = $0.4 + 0.1 \times (0.7 - 0.4) = 0.4 + 0.03 = 0.43$
- 只比 BC 偏移了 0.03，非常保守

训练后期 $g = 0.8$：
- 实际速度 = $0.4 + 0.8 \times (0.7 - 0.4) = 0.4 + 0.24 = 0.64$
- 明显偏离 BC，走向 RL 发现的更优方向

### 6.3 为什么梯度不会爆炸

关键在于：即使 $K=4$ 步的链式求导，门控 $g_i \in (0,1)$ 限制了每步的有效 Jacobian 范数。当 gate 接近 0 时，该步对最终动作的贡献趋于零，梯度被"刹住"。网络自动学习在梯度容易爆炸的方向关闭 gate。

这与 GRU 解决 RNN 梯度爆炸的原理完全相同——update gate 控制了信息流。

### 6.4 Actor Loss

$$
L_{\text{actor}} = \alpha \cdot \log\pi(\mathbf{a}^\theta | s) - \min\big(Q_1(s, \mathbf{a}^\theta), Q_2(s, \mathbf{a}^\theta)\big)
$$

**这个公式在做什么**：让 Actor 朝着"Q 值更高"的方向调整参数，同时用熵惩罚项阻止策略过早变得过于确定。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $-\min(Q_1(s,\mathbf{a}^\theta), Q_2(s,\mathbf{a}^\theta))$ | **主驱动力** | 取负号后最小化 loss 就等于最大化两个 Critic 中更保守（较小）的那个 Q 估计 |
| $\alpha \cdot \log\pi(\mathbf{a}^\theta\|s)$ | **保持随机的刹车** | 策略对这个动作越确定（$\log\pi$ 越大），这一项越大，阻止策略过快收敛成确定性动作 |
| $\mathbf{a}^\theta$ | **当场重新采样的动作** | 不是 replay buffer 里存的旧动作，而是当前 Actor（Flow-G）在这个状态下重新生成的动作，这样梯度才能流回 Actor 参数 |

**用人话读**："Actor 的目标是让重新采样出的动作拿到尽量高的 Q 分，同时别让自己变得太死板。"

**为什么要重新采样而不是用旧动作**：只有对当前 Actor 重新采样的动作求梯度，$\nabla_\theta Q$ 才能穿过 Flow-G 的 $K$ 步积分反传到 adapter 参数 $\phi$；用 buffer 里的旧动作则梯度无法回到当前参数。
:::

这里的 $\mathbf{a}^\theta$ 是**当前 Actor 重新采样**的动作（不是 replay buffer 中存储的旧动作）。梯度从 $Q$ 穿过 $\mathbf{a}^\theta$ 再穿过整个 $K$ 步 Flow-G 到达 adapter 参数 $\phi$。

**代入数字**：$\alpha = 4 \times 10^{-6}$，$\log\pi = -50$（Flow 路径的 log-prob 通常是大负数），$Q = 0.6$：
- $L = 4\times10^{-6} \times (-50) - 0.6 = -0.0002 - 0.6 = -0.6002$
- 梯度主要由 Q 项驱动，熵项在初始阶段影响很小

### 6.5 调度：critic_actor_ratio

Actor 不是每次 Critic 更新后都更新。配置参数 `critic_actor_ratio` 控制比例：

- `critic_actor_ratio=8`：每 8 次 Critic 更新后做 1 次 Actor 更新
- `critic_actor_ratio=16`：每 16 次后 1 次
- `critic_actor_ratio=32`：每 32 次后 1 次

每个 runner step 做 35 次 Critic 更新，所以 ratio=8 时每步约 4 次 Actor 更新，ratio=32 时约 1 次。

**为什么需要这个比例**：Critic 需要比 Actor 更快地收敛。如果 Actor 更新太频繁，它会追逐一个还不稳定的 Q landscape，导致策略漂移。实验表明 ratio=8 在 step 25 退化，ratio=16 在 step 40 退化，ratio=32 在 step 40 退化但更平缓。

### 6.6 代码对应

Actor loss 计算在 `rlinf/algorithms/chunk_sac.py`：

```python
def chunk_sac_actor_loss(log_pi, q_value, alpha):
    entropy_coefficient = torch.as_tensor(alpha, device=q_value.device, dtype=q_value.dtype)
    return (entropy_coefficient * log_pi - q_value).mean()
```

Flow-G gate 的配置在 YAML 中：

```yaml
flow_g:
  enabled: true
  freeze_pretrained_velocity: true
  hidden_dim: 256
  gate_bias: 0.0
  gate_scale: 2.0
```

---

## 六、组件三：连续 BC Loss（`sac_bc_coef`）

### 7.1 为什么需要 BC 拉回

纯 SAC 的 Actor 只受 Q 梯度驱动。如果 Critic 有偏（实验证明几乎一定有偏），Actor 可能漂向一个 Critic 误认为好但实际失败的区域。一旦漂出去，由于 off-policy 数据有限，很难纠正。

解决方案：在 Actor loss 中加一个 BC 正则项，把策略拉回专家行为的邻域。

### 7.2 实现方式

完整 Actor loss 变为：

$$
L_{\text{actor}}^{\text{total}} = \underbrace{\alpha \cdot \log\pi - Q}_{\text{SAC 主目标}} + \underbrace{\lambda_{\text{BC}} \cdot \|\mathbf{a}^{\text{actor}} - \mathbf{a}^{\text{BC}}\|_2^2}_{\text{BC 拉回项}}
$$

**这个公式在做什么**：在原来的 SAC 目标（追求高 Q、保持随机性）之外，加一条"绳子"把 Actor 的动作拉回到冻结 BC 动作的邻域，防止 Critic 有偏时策略越走越远。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\alpha\cdot\log\pi - Q$ | **SAC 主目标** | 前面已经讲过的标准 Actor loss，负责"往高 Q 方向走、别太死板" |
| $\lambda_{\text{BC}} \cdot \|\mathbf{a}^{\text{actor}} - \mathbf{a}^{\text{BC}}\|_2^2$ | **安全绳** | 当前 Actor 动作和冻结 BC 动作之间的距离惩罚，距离越远惩罚越重 |
| $\lambda_{\text{BC}}$ | **绳子的松紧度** | 系数越大，Actor 越不敢偏离 BC；系数为 0 就完全放开 |

**用人话读**："Actor 的总损失 = 追求高 Q 值 + 一条把它拉回专家动作附近的安全绳，绳子松紧由 $\lambda_{\text{BC}}$ 控制。"

**为什么需要这条安全绳**：Critic 在训练早期几乎一定存在偏差，如果 Actor 完全信任 Critic 的打分，可能被引导到 Critic 误判为好、但实际会失败的动作区域；一旦偏出专家行为分布，由于 off-policy 数据有限，很难再纠正回来。
:::

**数值代入**：假设某维度 actor=0.5, BC=0.4，$\lambda=0.1$：
- BC loss 贡献 = $0.1 \times (0.5-0.4)^2 = 0.001$
- 相比 $-Q \approx -0.6$，这是一个温和的约束

### 7.3 什么时候可以关闭

Aggressive 实验将 `sac_bc_coef=0.0`，结果是"稳定但无提升"——100 步训练后成功率仍停在 60% 左右，没有崩溃也没有进步。说明 BC loss 在当前设置下主要起"安全网"作用，不是性能瓶颈。

**最小核心保留 `sac_bc_coef=0.1` 作为默认值**，但它是一个配置旋钮而非硬编码逻辑。

---

## 七、组件四：熵温度自动优化

### 8.1 为什么需要自动调温

SAC 的核心是"最大化回报同时最大化熵"。$\alpha$ 控制这两个目标的权衡：
- $\alpha$ 太大 → 策略追求探索，不收敛
- $\alpha$ 太小 → 策略快速坍缩到一个点，丧失探索能力

手动调 $\alpha$ 在 62 维连续动作空间上几乎不可能。自动调温让 $\alpha$ 自适应。

### 8.2 自动调温公式

$$
L_\alpha = -\alpha \cdot \big(\log\pi(\mathbf{a}|s) + \bar{H}\big)
$$

**这个公式在做什么**：把"当前策略实际有多随机"和"我们希望它有多随机（目标熵）"做对比，自动调整温度 $\alpha$ 去缩小这个差距。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $-\log\pi(\mathbf{a}\|s)$ | **实际熵**（隐含在式中） | 策略对这个动作越不确定，$\log\pi$ 越负，熵（$-\log\pi$）越大 |
| $\bar{H}$ | **期望的熵水平** | 我们设定的目标，希望策略的熵至少达到这个值 |
| $\log\pi(\mathbf{a}\|s) + \bar{H}$ | **差距信号** | 实际熵和目标熵的差（符号翻转后的形式），正说明实际熵不够，需要更大的 $\alpha$ 去鼓励探索 |

**用人话读**："如果策略比我们期望的更死板（熵不够），就调大 $\alpha$ 逼它多探索一点；如果已经足够随机，就调小 $\alpha$。"

**为什么要自动调而不是手动设**：$\alpha$ 的合适取值随训练进度和动作维度变化很大，62 维连续动作空间上手动调参几乎不可行；把 $\alpha$ 变成一个可学习参数，用这个 loss 的梯度自动跟踪目标熵更省心也更稳。
:::

**逐符号拆解补充**：

| 符号 | 含义 | 配置值 |
|------|------|--------|
| $\alpha$ | 熵温度 | 用 softplus 参数化，初始约 $4 \times 10^{-6}$ |
| $\bar{H}$ | 目标熵 | `target_entropy: 0.0`（零目标熵 = 不强求探索） |

**数值代入**：假设 $\log\pi = -50$，$\bar{H} = 0$，当前 $\alpha = 4\times10^{-6}$：
- $L_\alpha = -4\times10^{-6} \times (-50 + 0) = 2\times10^{-4}$
- 梯度为正 → $\alpha$ 会增大（因为策略的熵 $-\log\pi = 50$ 远高于目标 0）
- 但由于 $\alpha$ 用 softplus 参数化且 learning rate 为 $3\times10^{-4}$，变化极其缓慢

### 8.3 为什么目标熵设为 0

在 `terminal_success` 奖励模式下（成功=1，其余=0），策略不需要大量探索——成功路径通常是窄的。$\bar{H}=0$ 意味着只要策略不完全坍缩成 delta 函数就行。实际中 Flow 的 SDE path density 天然有噪声（$\sigma_{\text{SDE}}$ 注入），所以 $\alpha$ 会稳定在极小值附近。

### 8.4 配置

```yaml
entropy_tuning:
  alpha_type: softplus
  initial_alpha: 4.0322580645e-6
  target_entropy: 0.0
  optim:
    lr: 3.0e-4
    lr_scheduler: torch_constant
    clip_grad: 10.0
```

---

## 八、组件五：Target-Critic 软更新

### 9.1 为什么需要 Target 网络

如果直接用正在训练的 Critic 计算 TD target，会形成"自我强化"循环——Critic 的错误会被放大并写入自己的训练目标。Target 网络通过延迟更新打破这个循环。

### 9.2 EMA 更新规则

$$
\theta^{\text{target}} \leftarrow (1 - \tau) \cdot \theta^{\text{target}} + \tau \cdot \theta^{\text{online}}
$$

**这个公式在做什么**：让 target 网络的参数缓慢地跟随在线 Critic 参数变化，而不是每次直接复制，避免 target 和自己训练目标形成"自我强化"的循环。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\theta^{\text{online}}$ | **正在训练的 Critic** | 每次梯度下降都会更新的参数，变化快 |
| $\theta^{\text{target}}$（右侧，旧值） | **滞后的参照物** | 上一次的 target 参数，本次更新的起点 |
| $(1-\tau)\cdot\theta^{\text{target}} + \tau\cdot\theta^{\text{online}}$ | **指数滑动平均（EMA）** | 绝大部分保留旧的 target 参数，只混入一小份最新的 online 参数 |

**用人话读**："target 网络的新参数 = 大部分沿用它自己上一步的值，只掺一点点最新训练出来的 Critic 参数。"

**为什么不直接用 online Critic 当 target**：如果直接拿正在训练的 Critic 算 TD target，Critic 的估计误差会被立刻写回自己的训练目标，形成自我强化的正反馈、导致训练发散；EMA 让 target 变化足够慢，相当于给 Bellman 迭代提供一个"稳定的参照系"。
:::

**配置值**：`tau` 通常为 0.005。意味着每次更新，target 只采纳 0.5% 的新 Critic 参数。约 200 次更新后 target 才"追上"online Critic 的当前水平。

**代入数字**：假设某个参数 online=1.0, target_old=0.5, $\tau=0.005$：
- $\text{target\_new} = 0.995 \times 0.5 + 0.005 \times 1.0 = 0.4975 + 0.005 = 0.5025$
- 几乎没变——这就是稳定性的来源

### 9.3 执行时机

每次 Critic 更新后立即执行一次 EMA。在每个 runner step 的 35 次 Critic 更新中，EMA 也执行 35 次。

---

## 九、路径对数概率：SDE Path Density

### 10.1 问题：Flow 没有解析 log-prob

SAC 需要 $\log\pi(a|s)$ 计算熵项和温度更新。但确定性 Flow 的 $K$ 步 Euler 积分给出确定性映射——给定初始噪声 $A_0$，输出 $A_K$ 唯一确定。要计算 marginal $\pi(a|s)$ 需要对所有可能的 $A_0$ 积分，不可行。

### 10.2 解法：注入 SDE 噪声

在每步 Euler 更新中注入微小噪声，把确定性 ODE 变成随机过程：

$$
A_{t_{i+1}} = A_{t_i} + v_\theta(t_i, A_{t_i}, s) \cdot \Delta t_i + \sigma_{\text{SDE}} \cdot \sqrt{\Delta t_i} \cdot \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, I)
$$

**这个公式在做什么**：在原来确定性的 Euler 积分基础上，每一步额外加一点随机高斯扰动，把"确定性 ODE 轨迹"变成"随机 SDE 轨迹"，这样每一步的转移才有明确的概率分布可以算 log-prob。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $v_\theta(t_i, A_{t_i}, s) \cdot \Delta t_i$ | **确定性位移** | 和普通 Flow 一样，沿速度场方向走一步 |
| $\sigma_{\text{SDE}} \cdot \sqrt{\Delta t_i} \cdot \varepsilon_i$ | **随机扰动** | 额外叠加一份高斯噪声，噪声大小由 $\sigma_{\text{SDE}}$ 和步长共同决定 |
| $\varepsilon_i \sim \mathcal{N}(0,I)$ | **噪声来源** | 每一步独立采样的标准高斯随机变量 |

**用人话读**："每一步除了按速度场往前走，还随机抖一下，抖动幅度跟步长的平方根成正比。"

**为什么要加噪声**：纯确定性的 Flow 给定初始噪声 $A_0$ 后输出是唯一确定的，要算 marginal 概率 $\pi(a|s)$ 需要对所有可能的 $A_0$ 积分，没有解析解；加入高斯噪声后每一步的转移变成解析高斯分布，整条路径的 log-prob 就能写成各步高斯 log-prob 之和，这是 SAC 计算熵项和温度更新必需的。
:::

每步的转移概率：

$$
p(A_{t_{i+1}} | A_{t_i}, s) = \mathcal{N}\big(A_{t_i} + v_\theta \cdot \Delta t_i, \; \sigma_{\text{SDE}}^2 \cdot \Delta t_i \cdot I\big)
$$

**这个公式在做什么**：把上面加噪声后的转移写成一个显式的高斯分布——均值是确定性 Euler 步给出的落点，方差由噪声强度和步长决定。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $A_{t_i} + v_\theta \cdot \Delta t_i$ | **分布的中心** | 如果没有噪声本应到达的位置，也就是高斯分布的均值 |
| $\sigma_{\text{SDE}}^2 \cdot \Delta t_i \cdot I$ | **分布的散布程度** | 协方差矩阵，各维度独立同方差，方差随步长线性增长 |
| $\mathcal{N}(\cdot,\cdot)$ | **转移分布** | 给定当前点 $A_{t_i}$，下一点 $A_{t_{i+1}}$ 服从的概率分布 |

**用人话读**："下一步落在哪，服从一个以确定性位移为中心、方差由噪声强度决定的高斯分布。"

**为什么写成显式高斯**：只有转移概率有解析形式，才能对每一步直接算出 $\log p(A_{t_{i+1}}|A_{t_i},s)$，进而把整条路径的 log-prob 累加起来。
:::

路径 log-prob：

$$
\log\pi(\mathbf{a}|s) = \log\mathcal{N}(A_0; 0, I) + \sum_{i=0}^{K-1} \log p(A_{t_{i+1}} | A_{t_i}, s)
$$

**这个公式在做什么**：把"生成这个动作"这件事的总概率，拆成"起点噪声的概率"加上"每一步转移的概率"之和，得到 SAC 需要的完整 $\log\pi(\mathbf{a}|s)$。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\log\mathcal{N}(A_0;0,I)$ | **起点的概率** | 初始噪声 $A_0$ 采样自标准高斯的对数概率，是路径概率的起始项 |
| $\sum_{i=0}^{K-1}\log p(A_{t_{i+1}}\|A_{t_i},s)$ | **逐步转移概率之和** | 把 $K$ 步中每一步的转移 log-prob 累加，链式法则下的联合概率取对数就是求和 |
| $\log\pi(\mathbf{a}\|s)$ | **整条轨迹的对数概率** | 把这条从噪声到动作的完整路径出现的概率取对数，SAC 用它计算熵和温度更新 |

**用人话读**："这个动作出现的概率（取对数）= 起点噪声出现的概率 + 沿途每一步转移概率的累加。"

**为什么可以直接相加**：这是马尔可夫链的联合概率分解——路径上每一步只依赖前一步（马尔可夫性），联合概率是各步条件概率的乘积，取对数后乘法自然变成加法。
:::

### 10.3 为什么不用 tanh squashing

原始 SAC-Flow 论文对最终动作做 tanh 压缩（$a = \tanh(A_K)$），需要加 Jacobian 修正项。但 GR00T 的动作空间已经归一化到 $[-1, 1]$（通过 action statistics），不需要额外的 tanh。配置中 `action_squash: none` 明确关闭了 tanh。

SDE path density 直接在未压缩的动作空间上计算——这简化了实现，也避免了 tanh 在边界处的数值问题。

### 10.4 配置

```yaml
compute_path_log_prob: true
path_density: sde_path
action_squash: none     # GR00T 动作已归一化，不需要 tanh
actor_backprop_steps: 4  # Q 梯度穿过 4 步 Flow
```

---

## 十、完整训练循环伪代码

把五个组件串在一起，一个 runner step 的完整流程：

```
输入：当前 Actor π_φ（Flow-G adapter），Critic Q_ψ，Target Q̄_ψ，Replay B，温度 α

=== Rollout 阶段 ===
for each of 32 envs:
    用 π_φ 采样 16 步动作块（4 步 Flow-G + SDE 噪声）
    在环境中执行 16 步，收集 (s, a_chunk, rewards, valid, s', done)
    存入 Replay Buffer B

=== 更新阶段（重复 35 次） ===
for update_idx in range(35):

    # --- Critic 更新（每次都做）---
    从 B 采 mini-batch
    用当前 π_φ 在 s' 采样下一动作 a'，计算 log π(a'|s')
    target = chunk_return + γ^H * m * (min(Q̄_1, Q̄_2)(s', a') - α * log π)
    L_critic = MSE(Q_1(s,a), target) + MSE(Q_2(s,a), target)
    更新 ψ
    EMA 更新：θ_target ← (1-τ)*θ_target + τ*θ

    # --- Actor 更新（每 critic_actor_ratio 次做 1 次）---
    if update_idx % critic_actor_ratio == 0:
        用 π_φ 在 s 重新采样动作 a_new，计算 log π
        L_actor = α * log π - min(Q_1, Q_2)(s, a_new)
        if sac_bc_coef > 0:
            a_bc = frozen_flow(s)  # 跳过 Flow-G gate
            L_actor += sac_bc_coef * ||a_new - a_bc||²
        更新 φ（只更新 Flow-G adapter 参数）

    # --- α 温度更新（每次 Actor 更新时）---
    if actor_updated_this_step:
        L_α = -α * (log π + target_entropy)
        更新 α
```

### 关键细节

1. **Actor 只更新 adapter 参数**：`freeze_pretrained_velocity: true` 冻结了原始 Flow 速度网络，梯度只流向 Flow-G adapter（约 1M 参数 vs 整个 GR00T 约 2.5B 参数）。
2. **Critic 看到的是扁平化的动作块**：`[B, 16*62]` = `[B, 992]` 维输入，加上状态 embedding。
3. **Expert replay 与 online replay 共存**：Stage 1 的 BC 数据作为 pinned expert 保留在 buffer 中，online 数据轮转最新的 128-512 条 trajectory。

---

## 十一、实验结论与工程教训

### 12.1 关键实验数据（open_drawer 任务）

| 配置 | 退化 step | 峰值成功率 | 备注 |
|------|-----------|-----------|------|
| ratio=8, lr=1e-4 | step 25 | ~30/48 (62.5%) | Actor 更新太频繁 |
| ratio=16, lr=1e-4 | step 40 | 35/48 (72.9%) | 有短暂提升后退化 |
| ratio=32, lr=1e-4 | step 40 | 35/48 (72.9%) | 退化更平缓 |
| ratio=32, lr=3e-5 | step 10+ | 30/48 (62.5%) | 稳定但无明显提升 |
| ratio=32, lr=3e-5, aggressive (no BC) | step 100 | 31/48 (64.6%) | 稳定无崩溃无提升 |
| BC baseline | — | 30/48 (62.5%) | 纯 BC 无 RL |

### 12.2 核心工程教训

**教训一：Critic 质量是瓶颈，不是 Actor 更新策略**

所有 ratio 变体都在 30-50 step 后退化。Critic replay audit 证明 Q 网络主要依赖"任务进度"（状态信息），不能可靠区分不同动作的好坏。原动作与随机 shuffle 动作的 Q 差只有 0.0166。

**教训二：BC loss 是安全网，不是驱动力**

`sac_bc_coef=0.1` 防止策略飘走，但不提供正向指导。关掉它（=0）不崩溃但也不提升——说明当前的提升瓶颈不在约束上，而在 Critic 信号质量上。

**教训三：Identity start 是必须的**

早期实验用 512 步 BC warmup 初始化 Flow-G adapter，结果 adapter 在 BC warmup 阶段就偏离了 identity，后续 RL 从一个不确定的起点开始。改为 identity start（`gate_bias=0`，初始 gate 接近 0）后，评测门禁通过。

**教训四：仿真非确定性是显著噪声源**

同一个 checkpoint，同 48 layout，三次独立评测的成功数分别为 25、29、34。PhysX 的非确定性使得 ±4 个 episode 的波动属于正常范围，不能用单次评测判断策略退化。

---

## 十二、核心配置清单

| 参数 | 典型值 | 作用 |
|------|--------|------|
| `chunk_length` | 16 | 动作块长度 |
| `replan_steps` | 16 | 每次重新规划的间隔 |
| `denoising_steps` | 4 | Flow 积分步数 |
| `critic_architecture` | flat_absolute | 双 Q 网络结构 |
| `flow_g.enabled` | true | 启用 Flow-G 门控适配器 |
| `flow_g.freeze_pretrained_velocity` | true | 冻结预训练 Flow |
| `flow_g.hidden_dim` | 256 | Adapter MLP 隐藏层 |
| `flow_g.gate_bias` | 0.0 | 初始门控偏置（identity start） |
| `critic_actor_ratio` | 8-32 | Critic/Actor 更新比 |
| `sac_bc_coef` | 0.1 | BC 正则系数 |
| `entropy_tuning.target_entropy` | 0.0 | 目标熵 |
| `entropy_tuning.initial_alpha` | ~4e-6 | 初始温度 |
| `tau` | 0.005 | Target EMA 系数 |
| `offline_updates` | 800 | Stage 1 Critic 预热步数 |
| `num_updates_per_step` | 35 | 每 runner step 的 Critic 更新次数 |
| `path_density` | sde_path | log-prob 计算方式 |
| `action_squash` | none | 无 tanh 压缩 |
| `online_reward_mode` | terminal_success | 成功=1 其余=0 |

---

## 十三、总结

SAC-Flow-G 的核心链路清晰而简洁：

1. **Critic 先学会评价**（Stage 1：800 次离线 TD 更新）
2. **Actor 再学会改进**（Stage 2：在线 Flow-G 门控更新）
3. **BC loss 防止飘走**（连续 L2 约束）
4. **温度自适应平衡探索与利用**（自动 $\alpha$）
5. **Target 网络稳定训练**（EMA 软更新）

`critic_actor_ratio` 控制 Actor 更新频率，是训练稳定性最关键的旋钮。当前瓶颈不在训练框架，而在 Critic 的动作区分能力——Critic 倾向于学习状态/进度信息而非动作质量差异。未来方向是改善 Critic 的动作敏感性。

---

## 延伸阅读

- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 的完整数学推导
- [SAC-Flow：用 SAC 直接训练 Flow 策略](/论文综述/079_SAC_Flow_用SAC直接训练Flow策略) — Flow-G 的学术论文原文精读
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — Flow 生成模型基础
- [GR00T N1.7 四种 RL 方案全景对比](./GR00T_N1d7_四种RL方案全景对比) — SAC-Flow-G 在四种方案中的定位
- [GR00T N1.7 Chunk-SAC 四种 Actor 目标详解](./GR00T_N1d7_ChunkSAC四种Actor目标详解) — AWR、Direct-Q、Flow-G 等 Actor 目标对比
- [Replay Buffer 经验回放](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — Replay 机制的通用原理
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — 梯度穿过多步生成的根本困难
