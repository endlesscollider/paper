---
title: "GR00T N1.7 四种 RL 方案全景对比：PPO / QC / SAC Flow-G / ConRFT"
order: 11
tags: [强化学习, GR00T, PPO, SAC, CQL, 后训练, 工程实践, 对比]
category: 工程实践
star: 5
---

# GR00T N1.7 四种 RL 方案全景对比

> **一句话**：GR00T N1.7 的 RL 后训练不是"选 PPO 还是 SAC"这么简单——四种方案在数据协议、Critic 类型、Actor 更新机制、OOD 处理和动作 horizon 上完全不同，checkpoint 也互不兼容。本文帮你理清每种方案在做什么、适合什么场景、以及它们之间为什么不能混用。

## 相关阅读

**前置知识**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 方案的理论基础
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC Flow-G 的理论基础
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — 所有 Q-based 方案的基础
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — CQL/CalQL 的背景
- [行为约束策略优化](/前置知识/001l_前置知识_行为约束策略优化) — 各方案的约束机制对比
- [AWR 优势加权回归](/前置知识/000u_前置知识_AWR_优势加权回归) — QC 隐式约束的理论来源
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — GR00T 动作生成机制

**关联文章**：
- [GR00T N1.7 Chunk-SAC 四种 Actor 目标详解](./GR00T_N1d7_ChunkSAC四种Actor目标详解) — SAC Flow-G 内部的四种 Actor loss 选择
- [RLinf BC 到 RL 的 ACT 后训练架构](./RLinf_BC到RL的ACT后训练架构) — PPO 路线的 ACT 实现细节
- [动作分块 RL 基础](/系列/groot_rl_deep_dive/06_动作分块RL基础_QChunking到AQC回顾) — QC 方案的理论基础
- [RLinf 算法实现：SAC](/系列/rlinf_deep_dive/10_算法实现_SAC与其他算法) — RLinf 中 SAC 的通用架构
- [GR00T N1.7 深度解析系列](/系列/groot_n1d7_deep_dive/index) — 模型架构参考

---

## 一、这篇文章要解决什么问题

你有一个 BC 预训练好的 GR00T N1.7 模型，想用环境 reward 做强化学习后训练。打开 RLinf 框架一看——**有四种完全不同的 RL 方案**可选：

1. **PPO / Flow-SDE**：经典 on-policy 策略梯度
2. **QC Random-40**：Q-Chunking + best-of-N 选择
3. **SAC Flow-G**：Chunk-SAC + Flow-G adapter
4. **ConRFT**：Conservative RL with same-state pair supervision

这不是同一个算法的四种变体——它们的数据流、Critic 类型、Actor 更新方式、replay 需求和 checkpoint 格式全都不同。选错了不仅浪费时间，还可能产生微妙的数据协议错误。

本文的目标：**让你在读完后能根据自己的任务特征和资源约束，做出明确的方案选择**。

---

## 二、贯穿全文的例子

> **任务**：GR00T N1.7 控制一个双臂人形机器人（62 维动作）在 MiArena/Isaac Sim 中执行桌面操作任务。
> - 状态：头部 + 左腕 + 右腕三路图像，加双臂关节状态
> - 动作：62D camera-frame Rot6D，执行前转为 58D simulator action
> - 预训练：已有 BC checkpoint，成功率约 40-60%
> - 目标：通过 RL 后训练提升到 80%+

---

## 三、四种方案共享什么

在讲区别之前，先明确所有方案的共同基础——它们都建立在同一个协议上：


| 共享项 | 说明 |
|--------|------|
| 模型 | GR00T N1.7（Cosmos-Reason2 骨干 + AlternateVLDiT 动作头） |
| 输入 | 三路图像 + 双臂本体感觉状态 |
| 动作格式 | 62D camera-frame Rot6D → 58D sim action |
| 四元数约定 | 统一 `[x, y, z, w]` |
| Processor | GR00T processor + action statistics（训练语义的一部分） |
| 分布式训练 | FSDP 负责 Actor 参数分片 |
| Worker 体系 | RLinf Worker 架构（Actor / Rollout / Env 分离） |

**但共享模型不意味着训练产物兼容**。四种方案的 replay schema、Critic 结构、optimizer state、target network、action horizon 和 checkpoint sidecar 全不相同。**不能仅凭 `model_type: gr00t_n1d7` 互相恢复 checkpoint**。

---

## 四、核心差异一览

先给一张全局对比表，后面再逐一展开：

| 维度 | PPO / Flow-SDE | QC Random-40 | SAC Flow-G | ConRFT |
|------|---------------|-------------|-----------|--------|
| **RL 类型** | On-policy | Off-policy（BC + best-of-N） | Off-policy SAC | Off-policy conservative |
| **Actor 怎么变好** | Clipped ratio × advantage | Actor 做 BC，Q 选动作 | Q 梯度穿过 Flow-G | 可选 Q Actor + BC |
| **Critic 类型** | State value $V(s)$ | Twin/ensemble chunk-Q | Twin chunk-Q | 通用 Q head，可多头 |
| **Q 梯度是否传给 Actor** | 否 | 否 | 是 | 可选 |
| **Replay Buffer** | 无（用当前 rollout） | 必需 | 必需 | 必需 |
| **OOD Q 处理** | 隐式（on-policy 数据） | 候选限于 Actor 分布 | Twin min + reference gate | 显式 CQL/CalQL |
| **动作 horizon** | 正式配置 8-step | 精确连续 40-step | Native 16-step | 配置化 H-step |
| **主要风险** | ratio/log-prob 不稳定 | Q 排序错误 | Critic exploitation | CQL 标度、pair 数据成本 |

---

## 五、方案一：PPO / Flow-SDE

### 5.1 核心思想

> **一句话**：用当前策略收集数据 → 计算 advantage → 用 clipped ratio 更新策略 → 丢弃数据 → 重新收集。

PPO 是最经典的 on-policy 方法。它不训练 $Q(s,a)$，只训练 $V(s)$（state value）。策略改进的信号来自 advantage $\hat{A}_t$——"这个动作比平均好多少"。

### 5.2 数据流

```mermaid
flowchart LR
    A["当前策略 π_old"] --> B["环境 rollout<br/>收集轨迹"]
    B --> C["计算 GAE advantage"]
    C --> D["PPO clipped loss<br/>更新策略 → π_new"]
    D --> E["丢弃旧数据"]
    E --> A
```

**关键特征**：每一轮训练数据用完就丢，不存到 replay buffer。这保证了训练数据始终来自当前策略，但代价是样本效率低。

### 5.3 Actor 更新公式

$$
\mathcal{L}_{\text{PPO}} = -\mathbb{E}\left[\min\Big(r_t \hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\Big)\right]
$$

**这个公式在做什么**：算出新旧策略对同一动作的喜好倍数，用这个倍数放大/缩小动作的优势分，但设一个安全上限防止单步更新跳太远。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $r_t$ | **策略变化倍数计** | 新策略选这个动作的概率 ÷ 旧策略选它的概率 |
| $\hat{A}_t$ | **动作打分器** | 这个动作比平均水平好多少，正=好动作，负=差动作 |
| $\text{clip}(r_t,1-\epsilon,1+\epsilon)$ | **安全绳** | 把 $r_t$ 卡在 $[1-\epsilon,1+\epsilon]$ 区间内 |
| $\min(\cdot,\cdot)$ | **保守裁判** | 取"放开跑"和"卡住跑"两个版本里较小的一个 |
| $-\mathbb{E}[\cdot]$ | **取负号做最小化** | 因为优化器只会做梯度下降，取负号把"最大化目标"变成"最小化损失" |

**用人话读**："算出新旧策略对这个动作喜好程度的比值，乘以这个动作的好坏分，但比值想跑出安全范围太远就不再给更多奖励。"

**为什么是这个形式**：详见 [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) 中 TRPO→PPO 的完整推导，这里的 clip 机制是用廉价操作替代 TRPO 昂贵的 KL 约束优化。
:::

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的概率比值。

> **一句话直觉**：如果一个动作的 advantage > 0（比平均好），就增加它的概率；但增加的幅度被 clip 限制，防止一步跳太远。

详细的 PPO 推导见 [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO)。

### 5.4 GR00T 上的特殊困难

PPO 需要计算 $\log \pi(a|s)$——在高斯策略上这很简单（解析公式），但 **GR00T 是 Flow 策略**，动作经过多步去噪产生。计算 $\log \pi$ 需要沿 ODE 路径累积散度（详见 [GR00T N1.7 Chunk-SAC 四种 Actor 目标详解](./GR00T_N1d7_ChunkSAC四种Actor目标详解#322-log-pimathbfas-是什么flow-策略的对数概率)）。

此外，GR00T 的 denoising latent 是 132D，但实际执行的 action 只有 62D——log-prob 的计算域和执行域不一致。RLinf 的正式配置使用 `token_level` ratio（逐 token 计算比值）而非不稳定的 `chunk_level` ratio（整个 chunk 的 log-prob 求和后再算比值——这会导致 ratio 爆炸）。

### 5.5 同步 vs 异步

| 模式 | 流程 | 优点 | 风险 |
|------|------|------|------|
| 同步 | rollout → GAE → update → 权重同步 → 下一轮 | 数据新鲜，理论保证强 | 环境空闲等待训练 |
| 异步 | rollout 和 update 并行 | 吞吐高 | policy staleness，需额外修正 |

异步 PPO 允许环境持续采数据，但训练用的数据可能来自几步前的旧策略。需要 `staleness_threshold`、behavior/proximal ratio 修正等机制。GR00T 的 log-prob 估计本身已有方差，异步的 stale-policy correction 可能进一步放大偏差。

### 5.6 优缺点

**优点**：
- 不依赖 Q 对 OOD 动作的泛化——只用当前策略数据
- 不需要 replay buffer 和 target network
- PPO 的 trust region 保证策略不会单步崩溃
- 适合 action chunk 较短、可频繁 replan 的场景

**缺点**：
- 样本效率低——Isaac Sim rollout 很贵时成本高
- 对 log-prob、ratio、mask 和 advantage 标度极其敏感
- $V(s)$ 只评价状态，无法用于 best-of-N 动作选择
- 长动作块（如 40 步）的 ratio 方差极大

### 5.7 关键配置片段

```yaml
algorithm:
  loss_type: embodied_ppo
  clip_range: 0.1          # GR00T 上通常比标准 PPO 的 0.2 更保守
  ratio_type: token_level  # 逐 token 算 ratio，避免 chunk 级爆炸
  gae_lambda: 0.95
  gamma: 0.99
```

---

## 六、方案二：QC Random-40

### 6.1 核心思想

> **一句话**：Actor 只做 BC（模仿 replay 中的动作），Critic 负责从 Actor 生成的多个候选动作中选最好的那个执行。


QC 的关键洞察：不让 Q 梯度直接改变 Actor（容易利用 Critic 误差），而是让 Actor 保持"生成合理候选"的能力，由 Critic 在有限候选集中做选择。这形成了一种**隐式 KL 约束**——Critic 只能从 Actor 能生成的动作中选，不会选到 Actor 分布之外的 OOD 动作。

关于 Q-Chunking 的理论基础，详见 [动作分块 RL 基础](/系列/groot_rl_deep_dive/06_动作分块RL基础_QChunking到AQC回顾)。

### 6.2 "Random-40" 是什么意思

这不是把 native 16-step checkpoint 简单改成 40——而是一个**完整的数据协议重建**：

1. **Random-40 BC**：从原始轨迹中随机截取连续 40 步窗口，用这些真实物理步重新训练 BC
2. **Behavior collection**：用 Random-40 BC 策略在环境中执行，收集真实 40-step replay
3. **Offline Q fitting**：在 replay 上训练 Chunk-Q Critic
4. **Online QC**：Critic 从 Actor 候选中选动作执行，新数据加入 replay

**硬约束**——每个训练窗口必须满足：
- 40 步全部是真实物理动作（不是 padding、不是 repeat_last）
- 不跨 episode 边界
- 训练、采集、Critic 和评测使用同一个 processor/statistics

### 6.3 训练阶段详解

```mermaid
flowchart TD
    subgraph "阶段 1: Random-40 BC"
        A["原始轨迹"] --> B["随机截取<br/>连续 40 步窗口"]
        B --> C["Flow-matching BC<br/>训练新 Actor"]
    end

    subgraph "阶段 2: Behavior Collection"
        C --> D["Actor 在环境中<br/>执行完整 40-step chunk"]
        D --> E["记录 (s, a_{0:40}, r_{0:40}, s')"]
        E --> F["Replay Buffer"]
    end

    subgraph "阶段 3: Offline Q Fitting"
        F --> G["Actor: flow BC loss<br/>（继续模仿 replay 动作）"]
        F --> H["Critic: TD loss<br/>（学习评估动作块价值）"]
    end

    subgraph "阶段 4: Online QC"
        I["在当前状态<br/>Actor 生成 N 个候选"] --> J["Critic 打分<br/>选最高 Q 的候选"]
        J --> K["执行选中的候选"]
        K --> L["新数据加入 Replay"]
        L --> G
        L --> H
    end
```

### 6.4 Critic Target

QC 的 Bellman target 和 Chunk-SAC 类似，但 bootstrap 使用 **best-of-N**：

$$
y = \underbrace{\sum_{i=0}^{39} \gamma^i r_i}_{R_{40}} + \gamma^{40} \cdot m \cdot Q^{\text{target}}(s', \mathbf{a}'_{\text{best}})
$$

**这个公式在做什么**：算出这次动作块的训练目标——40 步内真实拿到的折扣奖励，加上 40 步后"best-of-N 选出的最优候选"在目标网络下的估值。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $R_{40}=\sum_{i=0}^{39}\gamma^i r_i$ | **实打实的回报** | 执行完整 40 步动作块期间真实获得的折扣奖励总和 |
| $\gamma^{40}\cdot m$ | **未来价值折扣与截断** | 40 步后的未来价值要打折，episode 若已结束（$m=0$）则未来价值归零 |
| $Q^{\text{target}}(s',\mathbf{a}'_{\text{best}})$ | **最优候选的估值** | 用 best-of-N 选出的下一状态最优动作块，喂给目标网络算出的 Q 值 |

**用人话读**："target = 40 步内真实拿到的奖励，加上 40 步之后从多个候选里选出的最好动作、由目标网络打的分。"

**为什么要 best-of-N 而不是直接用 Actor 的单一输出**：QC 让 Actor 生成多个候选、Critic 挑最优的作为 bootstrap 目标，这样 target 更贴近"这个状态下能达到的最好未来"，而不是 Actor 当前单次采样的随机结果。
:::

其中 $\mathbf{a}'_{\text{best}}$ 的选择过程：

```text
# 在下一状态 s' 生成 N 个候选
C_1, C_2, ..., C_N = Actor(s')     # 每个是 [40, 62] 的完整动作块

# 用 Online Q 选最好的
best = argmax_i Q_online(s', C_i)

# 用 Target Q 评估（防止选择偏差）
future = Q_target(s', C_best)
```

> **为什么用 Online Q 选、Target Q 评？** 这和 Double DQN 的思路一样——用一个网络选动作，用另一个网络评价，避免"选择偏差"导致系统性过估计。

### 6.5 Actor 始终做 BC

QC 中 Actor 的 loss 始终是 flow-matching BC：

$$
\mathcal{L}_{\text{actor}} = \text{FlowBC}(\pi_\theta(s), \mathbf{a}_{\text{replay}})
$$

**这个公式在做什么**：让 Actor 始终模仿 replay 中的真实动作，保持"生成合理候选"的能力,不接受 Q 梯度的直接干预。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\pi_\theta(s)$ | **Actor 的当前输出** | 给定状态 $s$，当前 Actor（flow 策略）生成的动作 |
| $\mathbf{a}_{\text{replay}}$ | **真实执行过的动作** | replay buffer 里记录的、曾经真实执行的动作 |
| $\text{FlowBC}(\cdot,\cdot)$ | **模仿损失** | flow-matching 的行为克隆损失，让 Actor 输出尽量贴近真实动作 |

**用人话读**："Actor 的唯一训练目标就是模仿 replay 里的真实动作，和 Q 值毫无关系。"

**为什么 Actor 只做 BC 不接 Q 梯度**：如果让 Q 梯度直接改动 Actor，Actor 可能被 Critic 的估计误差带偏、生成 OOD 动作；只做 BC 能保证 Actor 生成的候选始终落在"曾经真实执行过"的分布内，让后续 best-of-N 选择是安全的。
:::

**Q 梯度不传给 Actor**。Actor 的作用是"生成覆盖合理动作空间的候选集"，而不是"追着 Q 梯度走"。改进来自 Critic 的选择能力——随着 Critic 训练得更好，它能从候选中挑出更好的动作。

### 6.6 隐式 KL 约束的直觉

为什么这比 `direct_q`（直接沿 Q 梯度推动作）更安全？

```mermaid
flowchart LR
    subgraph "direct_q"
        A1["Q 梯度"] --> B1["动作可以被推到<br/>任意位置 ⚠️"]
        B1 --> C1["可能到达 OOD 区域<br/>Critic 过估计"]
    end

    subgraph "QC best-of-N"
        A2["Critic 打分"] --> B2["只能从 Actor<br/>的 N 个候选中选"]
        B2 --> C2["候选都在 Actor<br/>分布内 ✅"]
    end
```

即使 Critic 对某些 OOD 动作严重过估计，QC 也不会选到那些动作——因为 Actor（做 BC）不会生成 OOD 动作。约束是**结构化的**，不需要额外的 penalty 项。

### 6.7 推理成本

QC 的代价是推理时间。每次决策需要：
1. Actor forward × N 次（生成 N 个候选，每个都是完整的 flow 去噪过程）
2. Critic forward × N 次（对每个候选打分）
3. 选择最高分的候选执行

**代入数字**：如果 N=32，每次 Actor forward 需要 10 步去噪 × 0.01s = 0.1s，那么生成 32 个候选需要 3.2s。加上 Critic 打分约 0.3s，总计约 3.5s 做一次决策。但因为决策后执行完整 40 步（约 0.8s），amortized 后每步决策时间 ≈ 0.088s，在仿真中可接受。

### 6.8 优缺点

**优点**：
- Actor 不利用 $dQ/da$，减少 Critic exploitation 风险
- Replay 可反复使用，样本效率高于 PPO
- best-of-N 可以在不修改 Actor 的情况下提升执行质量
- Actor 始终通过 BC 保持在合理动作分布内

**缺点**：
- 性能上限受 Actor 候选覆盖率限制（Actor 不会生成的好动作永远选不到）
- 每次决策需要生成 N 个完整 chunk，推理成本高
- Q 排序错误 → 直接选出更差的动作（没有梯度做平滑纠正）
- 固定 40-step open-loop 对精细接触阶段不一定合适
- 需要独立的 Random-40 BC 训练，不能复用 native-16 的 replay

---

## 七、方案三：SAC Flow-G

### 7.1 核心思想

> **一句话**：在冻结的 GR00T velocity 上加一个可训练 adapter，用完整 SAC 框架（Twin-Q + 可学习熵温度 + replay）训练这个 adapter，让 Q 梯度直接穿过动作采样链路改进策略。

SAC Flow-G 是四种方案中**性能潜力最高但配置最复杂**的。它不像 QC 那样限制在有限候选中选——策略可以通过连续的 Q 梯度产生从未在 replay 中出现过的好动作。

关于 SAC Flow-G 内部的四种 Actor objective（`awr_flow`/`direct_q`/`sac_flow_g`/`awr`），详见 [GR00T N1.7 Chunk-SAC 四种 Actor 目标详解](./GR00T_N1d7_ChunkSAC四种Actor目标详解)。

### 7.2 Flow-G Adapter

```text
v_actor(x_t, t, s) = FlowGAdapter(v_pretrained(x_t, t, s), x_t, t)
```

- `freeze_pretrained_velocity: true`：GR00T 主干冻结，只更新 adapter
- Adapter 初始化为 identity → 训练起点 = BC 策略
- 主要优化的参数量远小于整个 GR00T

### 7.3 训练结构

```mermaid
flowchart TD
    subgraph "Stage 1：Critic 预训练"
        A["Expert/BC replay"] --> B["Twin Critic TD 训练"]
        B --> C["Target Critic EMA 更新"]
        D["Actor 保持 identity<br/>（不更新）"]
    end

    subgraph "Stage 2：Online SAC"
        E["在线 rollout<br/>产生新数据"] --> F["加入 Replay"]
        F --> G["Critic: TD loss"]
        F --> H["Actor: α log π - min(Q1, Q2)"]
        H --> I["Flow-G adapter 更新"]
    end

    C --> E
    D --> E
```

### 7.4 与 QC 的根本区别

| 维度 | QC | SAC Flow-G |
|------|-----|-----------|
| Actor 如何改进 | 不改进（始终 BC），靠 Critic 选 | Q 梯度直接改进 Actor |
| 能否超越 replay 动作 | ❌ 受限于 Actor 候选覆盖 | ✅ 可产生全新动作 |
| OOD 风险 | 低（结构化约束） | 高（需要 gate/BC/reference 约束） |
| 推理成本 | 高（N 次 forward + 打分） | 低（1 次 forward） |

### 7.5 稳定性机制

因为 Q 梯度直接推动 Actor，SAC Flow-G 需要多重安全网：

| 机制 | 作用 |
|------|------|
| Flow-G identity 初始化 | 起点 = BC，不会突然产生怪动作 |
| BC warmup 阶段 | adapter 先做 expert 模仿再接 Q 梯度 |
| 持续 expert BC loss | 防止完全偏离 expert 行为 |
| Frozen-BC reference gate | 只有 Actor 确实比 BC 好时才用 Q 梯度 |
| Twin-Q minimum | 降低过估计 |
| Critic calibration gate | Critic 没准备好时不更新 Actor |
| `critic_actor_ratio` | Critic 更新 N 次，Actor 才更新 1 次 |

### 7.6 动作协议

当前有效的 SAC Flow-G 使用 **native 16-step**：

```yaml
chunk_length: 16
replan_steps: 16
```

旧的 40-step Flow-G 使用 `repeat_last` 把 native 16 步延长到 40 步——实验已证明这无效，相关 checkpoint 禁止恢复。

### 7.7 优缺点

**优点**：
- Replay 样本效率高
- Actor 能通过连续 Q 梯度超越 replay 中已有动作
- Identity adapter + BC/reference gate 可保护预训练能力
- Twin-Q、target Q、entropy temperature 构成完整 SAC 闭环
- 推理只需 1 次 forward（不需要 best-of-N）

**缺点**：
- 对 Critic 动作梯度的正确性要求最高
- Critic 若主要靠状态预测 Q（忽略动作），Actor 获得错误梯度
- 训练阶段、replay 分层、checkpoint migration 复杂
- Actor 更新过频或 LR 过大容易 policy drift

---

## 八、方案四：ConRFT

### 8.1 核心思想

> **一句话**：先用 conservative objective（CQL/CalQL）训练一个"不会过估计 OOD 动作"的 Critic，再用 same-state pair 监督确保 Critic 真的能区分动作好坏，最后可选地用这个可靠 Critic 更新 Actor。

ConRFT 的哲学与前三种不同：**它首先关注的不是"Actor 怎么变好"，而是"Critic 怎么变可靠"**。在 Critic 没有经过充分验证之前，可以完全不更新 Actor——先把 Critic 的基础打好。

### 8.2 三大 Critic 训练目标

ConRFT 的 Critic loss 由三项加权求和组成：

$$
\mathcal{L}_{\text{critic}} = \underbrace{\mathcal{L}_{\text{TD}}}_{\text{第一项：标准 TD 学习}} + \underbrace{\lambda_{\text{pair}} \cdot \mathcal{L}_{\text{pair}}}_{\text{第二项：同状态 pair 差值监督}} + \underbrace{\lambda_{\text{CQL}} \cdot \mathcal{L}_{\text{CQL}}}_{\text{第三项：Conservative 正则}}
$$

**这个公式在做什么**：把三个不同职责的损失加权求和成 Critic 的总训练目标——基础估值能力、同状态动作区分能力、OOD 保守压制,三者缺一不可。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mathcal{L}_{\text{TD}}$ | **基础估值训练** | 让 Critic 学会"这个状态-动作对未来能拿多少累积奖励" |
| $\lambda_{\text{pair}}\cdot\mathcal{L}_{\text{pair}}$ | **同状态区分训练** | 用 same-state pair 数据强制 Critic 能分清同一状态下两个动作谁更好 |
| $\lambda_{\text{CQL}}\cdot\mathcal{L}_{\text{CQL}}$ | **保守压制训练** | 压低 Critic 对没见过的（OOD）动作的估值，防止过度自信 |

**用人话读**："Critic 的总损失 = 基本估值损失 + 加权的同状态区分损失 + 加权的保守正则损失，三项分别负责不同的可靠性问题。"

**为什么三项各司其职、不能只留一项**：单独 TD 学习不知道怎么处理没见过的动作，容易过估计；pair loss 只管区分好坏，不管压制 OOD；CQL 只压 OOD，不保证区分能力——三者组合才能同时解决"基础能力""区分精度""保守边界"三个独立问题。
:::

> **为什么需要三项？** 单独的 TD 学习有一个固有缺陷：当 Critic 遇到 replay 中没见过的动作（OOD 动作）时，它的输出完全是"凭空外推"的——可能严重过估计。$\mathcal{L}_{\text{CQL}}$ 负责压低这些 OOD 区域的 Q 值；$\mathcal{L}_{\text{pair}}$ 则确保在"有事实依据"的区域内，Critic 能正确区分动作好坏。三项各司其职，缺一不可。

**逐项拆解**：

| 符号 | 对应的是什么 | 训练信号来源 | 解决什么问题 |
|------|-------------|-------------|-------------|
| $\mathcal{L}_{\text{TD}}$ | 标准 Bellman TD loss | replay 中的 $(s, a, r, s')$ | 让 Critic 学会基本的价值估计 |
| $\lambda_{\text{pair}} \cdot \mathcal{L}_{\text{pair}}$ | 同状态两分支的 Q 差值回归 | same-state pair batch | 让 Critic 在同一状态下能区分好动作和差动作 |
| $\lambda_{\text{CQL}} \cdot \mathcal{L}_{\text{CQL}}$ | Conservative Q-Learning 正则 | 随机采样的 OOD 动作 | 压低 Critic 对没见过的动作的估值 |

其中 $\lambda_{\text{pair}}$ 和 $\lambda_{\text{CQL}}$ 是权重超参数（典型值 $\lambda_{\text{pair}} \in [0.1, 1.0]$，$\lambda_{\text{CQL}} \in [0.01, 1.0]$）。

下面逐一详解每一项。

#### 8.2.1 第一项：$\mathcal{L}_{\text{TD}}$ — H-step TD Loss

**这一项在做什么**：让 Critic 学会"从当前状态执行这段动作后能拿到多少累积奖励"的基本估值能力。这是所有 Q-learning 方法共有的基础训练目标。

**Step 1：构造 Bellman target $y$**

$$
y = \underbrace{\sum_{i=0}^{H-1} \gamma^i r_i}_{R_H:\text{ H 步内的折扣奖励和}} + \underbrace{\gamma^H \cdot m \cdot Q^{\text{target}}(s', \mathbf{a}')}_{\text{H 步之后的 bootstrapped 未来价值}}
$$

**这个公式在做什么**：构造标准 Bellman target——把"H 步内真实拿到的奖励"和"H 步后目标网络估计的未来价值"加在一起，作为训练 Critic 的监督信号。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $R_H=\sum_{i=0}^{H-1}\gamma^i r_i$ | **实际拿到的折扣奖励** | 执行完这个动作块期间，环境每一步给的奖励按折扣加总 |
| $\gamma^H\cdot m$ | **未来价值的折扣与截断开关** | $\gamma^H$ 把未来价值折算到当前时刻，$m$ 在 episode 结束时把未来价值清零 |
| $Q^{\text{target}}(s',\mathbf{a}')$ | **目标网络对未来的估计** | 用变化缓慢的 EMA 网络，估计 H 步后状态 $s'$ 下动作 $\mathbf{a}'$ 的价值 |

**用人话读**："target = H 步内真实拿到的奖励，加上 H 步后 Critic（目标网络版本）对剩余未来的估计。"

**为什么要用目标网络而不是当前网络算未来价值**：如果 target 用正在训练的 Critic 自己算，target 和预测同时变化，容易"自己追自己尾巴"发散；目标网络是变化很慢的 EMA 副本，能提供相对稳定的监督信号。
:::

**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $H$ | 动作块步数 | 配置的 `discount_horizon`，例如 16 或 40 |
| $\gamma$ | 折扣因子 | 典型值 0.99，让远期奖励的权重递减 |
| $r_i$ | 第 $i$ 步获得的即时奖励 | 环境每步返回的标量，例如柜门角度增加量 |
| $\sum_{i=0}^{H-1}\gamma^i r_i$ | H 步折扣奖励和 $R_H$ | 把 chunk 内所有奖励按折扣加起来 |
| $m$ | bootstrap mask | episode 结束 = 0（没有未来了），未结束 = 1 |
| $s'$ | H 步之后的状态 | 执行完动作块后环境到达的新状态 |
| $\mathbf{a}'$ | 在 $s'$ 上用当前策略采样的动作 | $\mathbf{a}' = \pi_\theta(s')$，一次 Actor forward |
| $Q^{\text{target}}$ | **目标网络**的 Q 输出 | EMA 版本的 Critic，更新慢、提供稳定的 target |

**代入数字**：假设 $H=16$，$\gamma=0.99$，chunk 内奖励和 $R_{16}=2.1$，episode 未结束（$m=1$），$Q^{\text{target}}(s',\mathbf{a}')=9.5$：

$$
y = 2.1 + 0.99^{16} \times 1 \times 9.5 = 2.1 + 0.851 \times 9.5 = 2.1 + 8.08 = 10.18
$$

**这个公式在做什么**：把 $H=16$、$\gamma=0.99$、$R_{16}=2.1$、$Q^{\text{target}}=9.5$ 这组具体数值代入上面的 Bellman target 公式，算出这一条数据的训练目标数值。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $2.1$ | **这条数据的真实回报** | $R_{16}$，16 步内实际拿到的折扣奖励和 |
| $0.99^{16}\approx0.851$ | **16 步后的折扣系数** | 未来价值要打 85.1% 的折扣才能换算到当前时刻 |
| $0.851\times9.5=8.08$ | **折扣后的未来价值** | 目标网络估计的未来价值 9.5，打折后变成 8.08 |
| $2.1+8.08=10.18$ | **最终 target** | 真实奖励加折扣后未来价值，就是这条数据的训练目标 |

**用人话读**："这条数据实际拿到 2.1 分奖励，16 步后的未来还值 9.5 分，打折到当前是 8.08 分，两者加起来目标是 10.18。"

**为什么要具体代入数字**：抽象公式说明了结构，代入数字能验证折扣系数$0.99^{16}\approx0.851$确实让远期价值明显衰减，帮助直观理解 $H$ 和 $\gamma$ 对训练目标大小的影响。
:::

**Step 2：计算 TD loss**

$$
\mathcal{L}_{\text{TD}} = \text{Loss}\Big(Q_\phi(s, \mathbf{a}) - y\Big)
$$

**这个公式在做什么**：比较当前 Critic 的估值和刚算出的 target，差距就是需要修正的训练信号。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q_\phi(s,\mathbf{a})$ | **当前 Critic 的估值** | 正在被训练的网络对 replay 中真实 $(s,\mathbf{a})$ 的打分 |
| $y$ | **训练目标** | 上一步算出的 Bellman target |
| $\text{Loss}(Q_\phi(s,\mathbf{a})-y)$ | **误差评分函数** | 用 MSE 或 Huber 把差值转换成一个标量损失 |

**用人话读**："拿当前 Critic 的打分和目标值比较，差多少就用 MSE 或 Huber 算出多少损失。"

**为什么留 Loss 函数的选择余地**：MSE 对所有误差一视同仁,大误差样本会主导梯度；Huber 在误差较大时改用线性惩罚，对离群点更鲁棒——具体选哪个取决于训练稳定性需求。
:::

其中 $Q_\phi(s, \mathbf{a})$ 是**当前 Critic**（被训练的那个）对 replay 中"真实执行的 $(s, \mathbf{a})$"的估值，$y$ 是上面算出的 target。

Loss 函数有两种选择：

| Loss 类型 | 公式 | 特点 |
|-----------|------|------|
| MSE | $\frac{1}{2}(Q - y)^2$ | 对所有误差一视同仁 |
| Huber | 当 $|Q-y| \leq \beta$ 时用 MSE，否则用线性 | 对大 TD error 更鲁棒，不让离群点主导梯度 |

**代入数字续**：假设当前 Critic 输出 $Q_\phi(s, \mathbf{a}) = 8.5$，target $y = 10.18$：
- TD error = $8.5 - 10.18 = -1.68$
- MSE loss = $\frac{1}{2}(1.68)^2 = 1.41$
- 梯度方向：推动 Critic 把这个 $(s,\mathbf{a})$ 的估值往上调

**为什么用目标网络？** 如果 target 也用正在被训练的 Critic 计算，会出现"自己追自己尾巴"的问题——target 和 prediction 同时变化，容易发散。目标网络是 Critic 的 EMA 副本，变化很慢（每步只朝当前 Critic 方向挪动 0.5%），提供相对稳定的训练目标。

#### 8.2.2 第二项：$\mathcal{L}_{\text{pair}}$ — Same-State Pair Delta Loss

**这一项在做什么**：单纯的 TD 学习有一个隐患——Critic 可能学会"只看状态猜 Q，忽略动作的差异"。pair loss 专门治这个病：它要求 Critic 在**同一个状态**下，对两个不同动作输出的 Q 差值，必须匹配真实的回报差值。

**数据来源：Pair Batch**

Pair batch 是 ConRFT 独有的数据格式。在仿真中，对同一个起始状态 $s_0$ fork 两条执行路径：

```text
同一个起始观测 s₀（同一个物理快照）
├── main branch:  执行 a_main [H步] → 得到 rewards_main, s'_main, done_main
└── probe branch: 执行 a_probe [H步] → 得到 rewards_probe, s'_probe, done_probe
```

两条分支从**完全相同的物理状态**出发，执行不同的动作块，观察不同的结果。

**Step 1：分别计算两个分支的 Bellman target**

$$
y_{\text{main}} = \sum_{i=0}^{H-1}\gamma^i r_i^{\text{main}} + \gamma^H \cdot m_{\text{main}} \cdot Q^{\text{target}}(s'_{\text{main}}, \mathbf{a}'_{\text{main}})
$$

**这个公式在做什么**：给 main 分支（策略实际执行的动作）算一个 Bellman target，用法和 8.2.1 完全一样。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $y_{\text{main}}$ | **main 分支的 target** | 策略实际执行的动作 $\mathbf{a}_{\text{main}}$ 那条路径算出的 Bellman target |
| $r_i^{\text{main}}$ | **main 分支的即时奖励** | 执行 $\mathbf{a}_{\text{main}}$ 后每一步得到的奖励 |
| $Q^{\text{target}}(s'_{\text{main}},\mathbf{a}'_{\text{main}})$ | **main 分支后续价值** | H 步之后目标网络对 main 分支状态的估值 |

**用人话读**："main 分支的 target 就是按 8.2.1 的方法，用它自己这条路径的奖励和后续状态算一遍。"

**为什么要单独给 main 分支算一遍**：接下来要和 probe 分支的 target 做差值比较，必须先把两条分支的 target 分别算清楚。
:::

$$
y_{\text{probe}} = \sum_{i=0}^{H-1}\gamma^i r_i^{\text{probe}} + \gamma^H \cdot m_{\text{probe}} \cdot Q^{\text{target}}(s'_{\text{probe}}, \mathbf{a}'_{\text{probe}})
$$

**这个公式在做什么**：给 probe 分支（另一个候选动作）算一个 Bellman target，方法和 main 分支完全一样，只是输入换成 probe 分支自己的数据。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $y_{\text{probe}}$ | **probe 分支的 target** | 另一个候选动作 $\mathbf{a}_{\text{probe}}$ 那条路径算出的 Bellman target |
| $r_i^{\text{probe}}$ | **probe 分支的即时奖励** | 执行 $\mathbf{a}_{\text{probe}}$ 后每一步得到的奖励 |
| $Q^{\text{target}}(s'_{\text{probe}},\mathbf{a}'_{\text{probe}})$ | **probe 分支后续价值** | H 步之后目标网络对 probe 分支状态的估值 |

**用人话读**："probe 分支的 target 也按同样方法算,只是用的是 probe 这条路径自己的奖励和后续状态。"

**为什么两条分支要从同一起点 fork**：只有严格共享起始状态，两条分支 target 的差值才能干净地反映"动作差异"而不掺杂"状态差异"，这是下一步做差值监督的前提。
:::

这和 8.2.1 中的 target 计算方式完全相同，只是分别对两条分支各算一个。

**Step 2：计算 Q 差值和 target 差值**

$$
\Delta Q = Q_\phi(s_0, \mathbf{a}_{\text{probe}}) - Q_\phi(s_0, \mathbf{a}_{\text{main}})
$$

**这个公式在做什么**：算出 Critic 主观认为 probe 动作比 main 动作好多少。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q_\phi(s_0,\mathbf{a}_{\text{probe}})$ | **Critic 对 probe 的打分** | 当前 Critic 对 probe 动作在起始状态 $s_0$ 下的估值 |
| $Q_\phi(s_0,\mathbf{a}_{\text{main}})$ | **Critic 对 main 的打分** | 当前 Critic 对 main 动作在同一起始状态下的估值 |
| $\Delta Q$ | **主观判断的差值** | 两者相减，消去共享的状态价值 $V(s_0)$，只留动作差异 |

**用人话读**："Critic 认为 probe 动作比 main 动作好多少分。"

**为什么要相减而不是各自比较绝对值**：相减能抵消掉共享的状态价值 $V(s_0)$，让 Critic 只需要准确判断"动作间的相对差异"，比要求它输出精确的绝对 Q 值容易得多。
:::

$$
\Delta y = y_{\text{probe}} - y_{\text{main}}
$$

**这个公式在做什么**：算出真实数据显示 probe 动作比 main 动作实际好多少，作为 $\Delta Q$ 的监督标签。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $y_{\text{probe}}$ | **probe 分支的真实 target** | 上一步算出的 probe 分支 Bellman target |
| $y_{\text{main}}$ | **main 分支的真实 target** | 上一步算出的 main 分支 Bellman target |
| $\Delta y$ | **客观事实的差值** | 两个 target 相减，代表真实数据告诉我们的动作好坏差异 |

**用人话读**："真实数据显示，probe 这条路径比 main 这条路径实际好多少。"

**为什么这是监督信号而不是预测值**：$y_{\text{main}}$、$y_{\text{probe}}$ 都是由真实观测到的奖励和状态转移计算出来的，不依赖当前 Critic 的判断，所以 $\Delta y$ 可以作为 $\Delta Q$ 的训练标签。
:::

| 符号 | 含义 |
|------|------|
| $\Delta Q$ | Critic 认为 probe 动作比 main 动作好多少 |
| $\Delta y$ | 真实数据告诉我们 probe 实际比 main 好多少 |

**Step 3：用 Huber loss 监督差值**

$$
\mathcal{L}_{\text{pair}} = \text{Huber}(\Delta Q,\; \Delta y,\; \beta)
$$

**这个公式在做什么**：用 Huber loss 把"Critic 判断的差值"和"真实差值"的偏差转换成一个训练损失。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\Delta Q$ | **预测差值** | Critic 认为两个动作差多少 |
| $\Delta y$ | **真实差值** | 数据告诉我们两个动作实际差多少 |
| $\text{Huber}(\cdot,\cdot,\beta)$ | **鲁棒误差函数** | 小误差用平方惩罚，大误差改用线性惩罚，防止离群样本主导梯度 |

**用人话读**："让 Critic 判断的差值尽量贴近真实差值，用 Huber 而不是纯 MSE 来算这个贴近程度的损失。"

**为什么用 Huber 而不是 MSE**：pair 数据中偏差较大的样本（比如极端探索动作）如果用 MSE 会产生过大的梯度，Huber 在这些情况下切换为线性惩罚，训练更稳定。
:::

其中 Huber loss 的定义是：

$$
\text{Huber}(x, y, \beta) = \begin{cases} \frac{1}{2}(x-y)^2 & \text{if } |x-y| \leq \beta \\ \beta \cdot (|x-y| - \frac{\beta}{2}) & \text{otherwise} \end{cases}
$$

**这个公式在做什么**：定义 Huber 函数本身——误差小时表现像 MSE（平滑可导），误差大时表现像线性惩罚（不会爆炸）。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\frac{1}{2}(x-y)^2$（当 $\|x-y\|\le\beta$） | **平方区间** | 误差较小时，和 MSE 完全一样，靠近目标时梯度平滑地减小 |
| $\beta\cdot(\|x-y\|-\frac{\beta}{2})$（否则） | **线性区间** | 误差超过阈值 $\beta$ 后，惩罚只按线性增长，不会像平方项那样爆炸 |
| $\beta$ | **切换阈值** | 决定多大的误差开始从"平方惩罚"切换成"线性惩罚" |

**用人话读**："误差小的时候按平方算损失，误差一旦超过阈值就换成按比例线性算，不让离群点的梯度失控。"

**为什么要两段式定义**：纯 MSE 对大误差样本梯度会线性增长导致训练不稳定；纯 L1（线性）在误差接近 0 时不可导、收敛慢；Huber 结合两者优点，是处理含噪声监督信号时的标准选择。
:::

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $\beta$ | Huber 阈值，小于它用 MSE，大于它用线性 | `pair_huber_beta`，例如 1.0 |

**代入数字**：假设 main 分支（策略动作）在 16 步内获得总折扣奖励 $R_{\text{main}}=2.1$，bootstrap value = 8.0；probe 分支（另一种动作）获得 $R_{\text{probe}}=3.5$，bootstrap value = 8.5。

- $y_{\text{main}} = 2.1 + 0.851 \times 8.0 = 8.91$
- $y_{\text{probe}} = 3.5 + 0.851 \times 8.5 = 10.73$
- $\Delta y = 10.73 - 8.91 = 1.82$（probe 实际比 main 好 1.82）

如果当前 Critic 输出 $Q(s_0, \mathbf{a}_{\text{main}}) = 9.0$，$Q(s_0, \mathbf{a}_{\text{probe}}) = 9.8$：
- $\Delta Q = 9.8 - 9.0 = 0.8$（Critic 认为 probe 只好 0.8）
- 误差 = $0.8 - 1.82 = -1.02$（Critic 低估了差距）
- 梯度方向：推动 Critic 拉大 probe 和 main 之间的 Q 差距

**为什么要做差值而不是直接训练绝对 Q？**

关键洞察：**同一个初始状态抵消了大量 $V(s)$ 噪声**。

假设真实 Q 分别是 $Q(s_0, a_{\text{main}}) = 12.3$ 和 $Q(s_0, a_{\text{probe}}) = 12.8$。要 Critic 精确输出 12.3 和 12.8 非常难——它需要从高维图像+状态中拟合出这个绝对值。但两者的**差值** $12.8 - 12.3 = 0.5$ 只取决于"两个动作在同一状态下的后果差异"——$V(s_0)$ 这个共享的大数完全被抵消了。这使得 pair loss 对 Critic 的约束更精确、噪声更小。

**与 ranking loss 的区别**：

| 方法 | 监督信号 | 信息量 |
|------|----------|--------|
| Ranking loss | 只知道 $Q(s,a_1) > Q(s,a_2)$（谁大） | 1 bit |
| Pair delta loss | 知道 $Q(s,a_1) - Q(s,a_2) \approx 1.82$（大多少） | 连续值 |

Pair delta 包含了"好多少"的精确数值信息，比纯排序 loss 提供了更强的训练信号。

**数据采集成本**：main 和 probe 必须严格共享起始物理状态。在 Isaac Sim 中通过"保存环境快照 → fork 两条路径"实现。这意味着每个 pair 需要两倍的仿真步数——这是 ConRFT 的主要数据成本。

#### 8.2.3 第三项：$\mathcal{L}_{\text{CQL}}$ — Conservative Q-Learning

**这一项在做什么**：TD loss 和 pair loss 都只训练 Critic 在"数据中见过的动作"上的估值。但问题是：当 Actor 后续用 Q 梯度更新策略时，它可能把动作推到"数据中没见过的区域"——而 Critic 在这些区域的输出是不可信的外推值，很可能严重过估计。CQL 的作用就是**在训练 Critic 时，显式地把它对 OOD 动作的估值压低**，让 Critic 在没见过的区域保持"保守悲观"。

**Step 1：采样 OOD 动作**

从动作空间中随机采样 $N$ 个与 replay 数据无关的动作：

$$
a_1^{\text{rand}}, a_2^{\text{rand}}, \ldots, a_N^{\text{rand}} \sim p_{\text{proposal}}(\mathbf{a})
$$

**这个公式在做什么**：从一个和 replay 数据无关的分布里随机采样一批动作，用来代表"Critic 没见过的 OOD 区域"。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $a_i^{\text{rand}}$ | **随机探针动作** | 第 $i$ 个和真实 replay 无关的候选动作 |
| $p_{\text{proposal}}(\mathbf{a})$ | **提议分布** | 采样这些随机动作所用的分布，如均匀分布或高斯分布 |
| $N$ | **采样数量** | 一批随机动作的个数，越多估计越稳定但计算越贵 |

**用人话读**："从一个和真实数据无关的分布里随机撒一批动作，当作 Critic 从没见过的区域的代表样本。"

**为什么要随机采样而不是用 Actor 生成**：CQL 想压制的是"Critic 可能被误导过估计的任意区域"，用与数据无关的随机分布采样能更广泛地覆盖潜在的 OOD 区域，而不是只覆盖 Actor 当前倾向生成的那一小片。
:::

$p_{\text{proposal}}$ 可以是均匀分布 $\text{Uniform}(\text{action\_space})$ 或正态分布 $\mathcal{N}(0, \sigma)$。这些随机动作大概率不在 replay 数据的分布内——它们就是我们想压低的 OOD 区域的代表。

**Step 2：计算 OOD 动作的"软最大" Q 值**

$$
Q_{\text{ood}} = \tau \cdot \log \frac{1}{N+1} \sum_{i=1}^{N} e^{Q_\phi(s, a_i^{\text{rand}})/\tau}
$$

**这个公式在做什么**：把 N 个随机动作的 Q 值汇总成一个"软最大值"，代表 Critic 在 OOD 区域里能打出的最高估值水平。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q_\phi(s,a_i^{\text{rand}})/\tau$ | **温度缩放后的打分** | 每个随机动作的 Q 值除以温度 $\tau$，控制后续 softmax 的尖锐程度 |
| $\log\sum_{i=1}^N e^{(\cdot)}$ | **软最大值运算** | logsumexp，近似于取所有随机动作里 Q 值最高的那个，但保持可导 |
| $\tau\cdot(\cdot)$ | **还原量纲** | 乘回温度，让结果和普通 Q 值的数值范围一致 |

**用人话读**："把所有随机动作的 Q 值做一次软性的取最大值操作，近似算出 Critic 在没见过的区域最高会打多少分。"

**为什么用 softmax 形式而不是直接 max**：直接取 max 不可导，无法用于梯度下降；logsumexp 是 max 的光滑近似，$\tau\to0$ 时退化为真正的 max，同时保持了可微性方便训练。
:::

| 符号 | 含义 | 为什么是这个形式 |
|------|------|-----------------|
| $\tau$ | 温度参数 | 控制 logsumexp 的"软度"，$\tau \to 0$ 退化为 $\max$ |
| $\log\sum\exp(\cdot/\tau)$ | Soft maximum | 对所有随机动作的 Q 值取一个"软的最大值" |
| $N+1$ | 归一化常数 | 使 $Q_{\text{ood}}$ 的量纲与 $Q(s, a_{\text{data}})$ 可比 |

> **直觉**：$Q_{\text{ood}}$ 近似度量了"Critic 给 OOD 区域最高能打多少分"。如果 Critic 对任何随机动作都给出了高分，$Q_{\text{ood}}$ 就会很大。

**Step 3：计算 CQL loss**

$$
\mathcal{L}_{\text{CQL}} = \mathbb{E}_{s \sim \mathcal{D}}\Big[Q_{\text{ood}}(s) - Q_\phi(s, \mathbf{a}_{\text{data}})\Big]
$$

**这个公式在做什么**：让 Critic 在训练时"压低 OOD 区域打分、抬高数据区域打分"，把高估值区域挤压到数据分布内部。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q_{\text{ood}}(s)$ | **OOD 区域的软最大打分** | 上一步算出的、Critic 对随机动作能打出的最高估值 |
| $Q_\phi(s,\mathbf{a}_{\text{data}})$ | **数据动作的打分** | Critic 对 replay 中真实动作的估值 |
| $Q_{\text{ood}}(s)-Q_\phi(s,\mathbf{a}_{\text{data}})$ | **过估计差距** | OOD 区域打分比数据区域高出多少,越大说明过估计越严重 |
| $\mathbb{E}_{s\sim\mathcal{D}}[\cdot]$ | **对 batch 状态取平均** | 对采样到的一批状态取平均，作为最终损失 |

**用人话读**："这个损失衡量 Critic 对没见过的动作打分比对真实数据打分高出多少，最小化它就是逼着 Critic 把没见过的地方分数压下去、真实数据的分数抬上来。"

**为什么是"压低减抬高"而不是只压低一边**：只压低 OOD 分数可能连带压低数据分数附近的合理估值；同时抬高数据动作的分数能保证 Critic 不会因为过度保守而失去区分真实好动作和差动作的能力。
:::

| 项 | 梯度方向 | 含义 |
|----|----------|------|
| $+Q_{\text{ood}}$ | 压低 Critic 对 OOD 动作的输出 | "对没见过的动作别给高分" |
| $-Q_\phi(s, \mathbf{a}_{\text{data}})$ | 抬高 Critic 对 replay 动作的输出 | "对见过的动作保持合理估值" |

> **一句话**：CQL loss = "Critic 对随机动作的估值" 减去 "Critic 对真实数据动作的估值"。最小化这个 loss，就是在**压低 OOD、抬高数据**——迫使 Q 函数的"高值区域"集中在数据分布内部。

**代入数字**：假设某状态 $s$ 下：
- Replay 中的真实动作：$Q_\phi(s, a_{\text{data}}) = 10.0$
- 5 个随机采样动作的 Q 值：$[12.5, 11.0, 9.8, 13.2, 10.5]$
- $\tau = 1.0$

计算 $Q_{\text{ood}}$：
$$
Q_{\text{ood}} = 1.0 \times \log\frac{1}{6}(e^{12.5} + e^{11.0} + e^{9.8} + e^{13.2} + e^{10.5})
$$

**这个公式在做什么**：把 5 个具体的随机动作 Q 值代入软最大值公式，算出这个状态下 $Q_{\text{ood}}$ 的具体数值。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $e^{12.5},e^{11.0},\ldots$ | **各随机动作的指数打分** | 5 个随机动作的 Q 值（$\tau=1.0$ 时直接取指数） |
| $\frac{1}{6}(\cdot)$ | **归一化（$N+1=6$）** | 除以 $N+1$ 做归一化，让结果量纲和普通 Q 值可比 |
| $\log(\cdot)$ | **取对数还原** | 把指数和还原回对数尺度，得到软最大值 |

**用人话读**："把 5 个随机动作的 Q 值取指数、求和、除以 6、再取对数，得到这个状态下 Critic 对 OOD 区域打出的软最大分。"

**为什么最大的那个值（13.2）会主导结果**：指数函数对大值极度敏感,$e^{13.2}$ 远大于其他项，求和结果几乎完全由它决定——这正是 softmax/logsumexp 近似 max 操作的数学原理。
:::

$e^{13.2}$ 主导求和 → $Q_{\text{ood}} \approx 13.2 - \log 6 \approx 13.2 - 1.79 = 11.41$

CQL loss = $11.41 - 10.0 = 1.41 > 0$

梯度方向：
- 把随机动作（特别是 13.2 和 12.5 那两个高分动作）的 Q 估值压下来
- 把真实数据动作的 Q 估值稍微抬一点

**为什么其他方案不用 CQL？**

| 方案 | 如何处理 OOD 过估计 | 为什么不需要 CQL |
|------|-------------------|-----------------|
| PPO | 训练数据始终来自当前策略 | on-policy 数据没有 OOD 问题 |
| QC | Critic 只需要给 Actor 候选排序 | 候选都在 Actor 分布内，不是 OOD |
| SAC Flow-G | Twin-Q min + reference gate + BC 正则 | 隐式约束，但可能不够严格 |
| ConRFT | **显式 CQL** | 直接对 OOD 区域施加惩罚 |

ConRFT 认为隐式机制不够可靠——特别是当 Critic 能力较弱、或动作空间很大时，必须**显式地**告诉 Critic "不要在没见过的地方给高分"。

#### 8.2.4 CalQL：校准版 CQL（防止过度悲观）

**CQL 的副作用**：CQL 可能过于激进——不只压低了 OOD 动作的 Q，连一些**在数据中出现过的好动作**的 Q 也被误压了。这是因为 random proposal 可能碰巧采到和数据接近的动作，也被当成 OOD 压低。

**CalQL 的修正**：当 replay 中有 Monte-Carlo return $R_{\text{MC}}$（某条轨迹的真实累积回报）时，在 CQL 计算中加一个**下界保护**：

$$
Q_{\text{candidate}} \leftarrow \max\Big(Q_{\text{candidate}},\; R_{\text{MC}}\Big)
$$

**这个公式在做什么**：给 CQL 压低后的 Q 值设一个下限——不能压到比这个动作真实获得过的回报还低。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q_{\text{candidate}}$ | **CQL 压低后的候选值** | CQL 正常流程会尽量把这个值往下压 |
| $R_{\text{MC}}$ | **真实发生过的事实** | 这个状态-动作对从真实轨迹中观测到的实际累积回报 |
| $\max(\cdot,\cdot)$ | **下界保护** | 取两者较大的一个，防止 Q 值被压到低于已发生的真实事实之下 |

**用人话读**："Q 值可以被 CQL 压低，但不能压到比这个动作真实拿到过的回报还低。"

**为什么需要这个保护**：CQL 的随机 proposal 有时会碰巧采到接近数据的动作，把这些本来合理的动作也误判为 OOD 压低了；用真实 MC 回报做下界，可以纠正这种过度悲观的错误压制。
:::

| 符号 | 含义 |
|------|------|
| $Q_{\text{candidate}}$ | CQL 中随机候选动作的 Q 值（正常会被压低） |
| $R_{\text{MC}}$ | 这个状态-动作对的真实累积回报（从真实轨迹计算） |

> **一句话直觉**：$R_{\text{MC}}$ 是事实——这个动作从这个状态开始，真实获得了这么多回报。CQL 可以把 Q 压到这个事实之下吗？不行！CalQL 说"Q 值至少不能低于你的真实回报"。

**代入数字**：假设某动作的真实累积回报 $R_{\text{MC}} = 8.0$，但 CQL 把它的 Q 压到了 5.0。CalQL 会把它拉回到 $\max(5.0, 8.0) = 8.0$——不允许低于事实。

**限制**：CalQL 需要 replay 中存在可靠的 MC return。对于在线数据（episode 还没结束），无法计算 MC return。因此配置 `require_mc_returns: true` 时，缺少 MC return 的数据会直接报错，不允许静默跳过。

### 8.3 Actor 更新（可选）

ConRFT 可以完全关闭 Actor 更新，只训练 Critic：

```yaml
algorithm:
  conrft:
    actor_update_enabled: false    # 只训练 Critic
    actor_warmup_steps: 1000       # 或者等 Critic 稳定后再开 Actor
    actor_bc_weight: 0.1           # Actor loss 中的 BC 正则
```

当 Actor 更新打开时：

$$
\mathcal{L}_{\text{actor}} = \alpha \log \pi - Q(s, \pi(s)) + c_{\text{bc}} \cdot \text{BC}(\pi(s), \mathbf{a}_{\text{data}})
$$

**这个公式在做什么**：如果打开 Actor 更新，用"最大化 Q 值 + 保持一定探索熵 + 不完全脱离 BC"三个目标共同训练 Actor。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\alpha\log\pi$ | **熵奖励项** | 鼓励策略保持一定随机性，$\alpha$ 是熵温度系数 |
| $-Q(s,\pi(s))$ | **Q 值最大化项** | 取负号做最小化，本质是让 Actor 生成能拿高 Q 分的动作 |
| $c_{\text{bc}}\cdot\text{BC}(\pi(s),\mathbf{a}_{\text{data}})$ | **BC 正则项** | 限制 Actor 别离真实数据分布太远，权重 $c_{\text{bc}}$ 控制约束强度 |

**用人话读**："Actor 的更新目标是：让 Q 值尽量高，同时保持一点探索性，还不能太脱离真实数据的分布。"

**为什么这里的 Q 是"可信"的才能用**：因为前面 8.2 节的三项 Critic loss（TD + pair + CQL）已经花了大量篇幅确保 Critic 对 OOD 区域保守、对同状态动作有区分能力，只有这样的 Critic 才能安全地用来指导 Actor；这也是 ConRFT 把 Actor 更新设为"可选"步骤的原因。
:::

**与 SAC Flow-G 的区别**：
- SAC Flow-G 重点是 adapter 设计、entropy tuning 和 frozen-BC reference
- ConRFT 重点是确保 Critic 本身可靠（CQL + pair），Actor 更新是可选的后续步骤

### 8.4 优缺点

**优点**：
- CQL/CalQL 明确处理 OOD Q 高估——不像 SAC Flow-G 依赖隐式约束
- Pair delta 直接用事实监督 Critic 的动作区分能力
- 可以先关闭 Actor，单独验证 Critic 是否可靠
- H-step TD、Huber、CQL 权重均可独立调节

**缺点**：
- CQL 权重过大 → 所有 Q 都被压低 → Critic 失去区分能力
- Pair 数据需要 same-state fork，采集成本高
- Random proposal 若与机器人可执行动作分布差太远，CQL 可能"压低"了本来就不可能执行的动作区域（无意义）
- TD + pair + CQL + Actor BC 多项 loss 的相对标度需要仔细审计

---

## 九、动作 Horizon：不是普通超参数

四种方案使用不同的动作 horizon，这是最容易踩坑的地方：

| 方案 | 动作 Horizon | 含义 |
|------|-------------|------|
| PPO（正式配置） | 8-step | 执行 8 步后重新规划 |
| QC Random-40 | 40-step | 完整执行 40 步 open-loop |
| SAC Flow-G | 16-step（native） | 使用原始 checkpoint 的动作长度 |
| ConRFT | 配置化 H-step | 必须与 replay 和 action chunk 一致 |

**为什么不能混用？** 因为 H 同时改变了：

1. **Observation transition**：H=8 时 $s'$ 是 8 步后的状态，H=40 时是 40 步后的
2. **Reward return**：$R_H = \sum_{i=0}^{H-1}\gamma^i r_i$ 的求和范围不同
3. **Bootstrap discount**：$\gamma^8 \approx 0.92$ vs $\gamma^{40} \approx 0.67$
4. **Open-loop 时长**：8 步 ≈ 0.16s vs 40 步 ≈ 0.8s
5. **Critic 输入维度**：$8 \times 62 = 496$ 维 vs $40 \times 62 = 2480$ 维

**禁止操作**：
- 用 `repeat_last` 把 native-16 replay 伪装成 40-step ❌
- 用 Random-40 BC checkpoint 恢复 native-16 Flow-G optimizer ❌
- 用 PPO rollout tensor 直接作为 QC/ConRFT replay ❌
- 修改 H 后继续加载旧 TargetQ、replay 或 resume hash ❌

---

## 十、Checkpoint 与数据兼容性

| 来源 ↓ / 目标 → | PPO | QC | SAC Flow-G | ConRFT |
|-----------------|-----|-----|-----------|--------|
| 原始 BC 权重 | ✅ 可初始化 | ⚠️ 需先做 Random-40 BC 适配 | ✅ 可初始化 | ✅ 可初始化 |
| PPO checkpoint | ✅ 同配置可恢复 | ❌ | ❌ | ❌ |
| QC replay/ckpt | ❌ | ✅ 仅同协议 | ❌ | ⚠️ 需显式转换 |
| Flow-G ckpt | ❌ | ❌ | ✅ 仅同 hash | ❌ |
| ConRFT ckpt | ❌ | ❌ | ❌ | ✅ 仅同配置 |

**核心规则**：最多只能复用兼容的 GR00T model weights。Optimizer state、scheduler、TargetQ、alpha、replay buffer、calibration FIFO、pending episode 和 update counter 必须服从各自方案的 checkpoint contract。

---

## 十一、如何选择：决策流程图

```mermaid
flowchart TD
    START["选择 RL 方案"] --> Q1{"on-policy rollout<br/>成本可接受？"}
    Q1 -->|"是，且不需要 Q"| PPO["选 PPO<br/>最简单，不依赖 Critic 泛化"]
    Q1 -->|"否，需要 replay"| Q2{"是否信任 Critic<br/>的动作梯度 dQ/da？"}
    Q2 -->|"不信任"| Q3{"有高质量 behavior<br/>replay + 可承担<br/>best-of-N 成本？"}
    Q3 -->|"是"| QC["选 QC Random-40<br/>结构化约束，最安全"]
    Q3 -->|"否"| ConRFT_CRITIC["选 ConRFT（关闭 Actor update）<br/>先训练可靠 Critic"]
    Q2 -->|"信任（已审计）"| Q4{"需要显式 OOD<br/>保护 + pair 监督？"}
    Q4 -->|"否，隐式约束够用"| SAC["选 SAC Flow-G<br/>性能潜力最高"]
    Q4 -->|"是"| ConRFT_FULL["选 ConRFT（开 Actor update）<br/>最严格的 Critic 约束"]

    style PPO fill:#e3f2fd
    style QC fill:#fff3e0
    style SAC fill:#e8f5e9
    style ConRFT_CRITIC fill:#fce4ec
    style ConRFT_FULL fill:#fce4ec
```

### 选择建议总结

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| rollout 便宜、action chunk 短、只想快速验证 | PPO | 最简单，不需要 replay/Critic |
| 有大量高质量 replay、不信任 Q 梯度 | QC | 安全，Critic 只做排序不做梯度 |
| 追求最高性能上限、可投入调参时间 | SAC Flow-G | 连续 Q 梯度可超越 replay 上限 |
| 主要瓶颈是 Critic 不可靠/过估计 | ConRFT | 先用 CQL + pair 把 Critic 修好 |
| 能采集 same-state pair 数据 | ConRFT | pair delta 提供最强的 Critic 监督 |
| 想先验证 Critic 再决定是否更新 Actor | ConRFT | 可关闭 Actor update 单独审计 Critic |

---

## 十二、公平比较注意事项

四种方案**不能直接比较各自日志中的 loss 数值**。如果需要 A/B 实验，至少需要固定：

- 相同初始 BC 能力（或记录 BC 适配差异）
- 相同任务、prompt、layout manifest 和 episode horizon
- 相同输入（三相机、processor、statistics、action converter）
- 相同仿真配置（renderer、DLSS、资产、PhysX）
- **相同有效物理步预算**（不是 optimizer update 数）
- 独立的 fixed-layout checkpoint evaluation

此外，由于四种方案的 action horizon 不同，还需要决定实验口径：

1. **保留各方案最自然的动作协议**，比较端到端最佳系统效果
2. **统一动作协议**后比较纯算法差异，但需要为每种方案重新训练兼容的 BC/replay

不能混合这两种口径。

---

## 十三、关键监控指标

### PPO
| 指标 | 关注点 |
|------|--------|
| `approx_kl` / `clip_fraction` | 更新幅度是否受控 |
| ratio 分布 | 是否有爆炸（chunk-level ratio 问题） |
| GAE explained variance | Value head 拟合质量 |
| success rate / PSR | 最终评测标准 |

### QC Random-40
| 指标 | 关注点 |
|------|--------|
| best candidate vs random candidate 真实收益 | Critic 选择是否有效 |
| Twin disagreement | Critic 不确定性 |
| candidate Q spread | 候选之间是否有区分度 |
| BC loss | Actor 是否保持合理生成 |

### SAC Flow-G
| 指标 | 关注点 |
|------|--------|
| `critic/calibration_ready` | Critic 是否允许 Actor 更新 |
| `critic/td_abs_p90` / `p99` | Critic 误差尾部 |
| `sac/alpha` | 熵温度是否稳定 |
| `sac/reference_gate_fraction` | reference gate 通过率 |
| `actor/action_grad_norm` | Actor 梯度是否爆炸 |

### ConRFT
| 指标 | 关注点 |
|------|--------|
| pair delta loss / pair sign accuracy | Critic 能否区分同状态动作好坏 |
| CQL diff（Q_ood - Q_data） | conservative penalty 是否适度 |
| CalQL bound rate | 有多少样本触发了 MC return 下界 |
| Actor gate 状态 | Actor 是否在更新 |

---

## 十四、总结：四条路在解决同一个问题的不同方面

```mermaid
flowchart TD
    PROBLEM["核心问题：<br/>让 GR00T 策略超越 BC 天花板"] --> PPO_ANGLE["PPO 视角：<br/>用当前策略的 advantage<br/>做安全的 trust-region 更新"]
    PROBLEM --> QC_ANGLE["QC 视角：<br/>Actor 保持 BC，<br/>让 Critic 选更好的动作"]
    PROBLEM --> SAC_ANGLE["SAC Flow-G 视角：<br/>用 Q 梯度直接推动 Actor<br/>加多重安全约束"]
    PROBLEM --> CON_ANGLE["ConRFT 视角：<br/>先保证 Critic 可靠<br/>再考虑 Actor 更新"]
```

- **PPO** 说："我不需要 Q，我信任 on-policy advantage"
- **QC** 说："Q 梯度不可信，但 Q 排序可信——让它在候选中选就好"
- **SAC Flow-G** 说："Q 梯度可以信（有足够约束后），直接用它推动 Actor"
- **ConRFT** 说："Q 梯度能不能信取决于 Critic 质量——先把 Critic 修好"

四种方案不是"一个比一个好"的进化关系，而是**对 Critic 信任度的不同假设**下的最优选择。

---

## 延伸阅读

- [GR00T N1.7 Chunk-SAC 四种 Actor 目标详解](./GR00T_N1d7_ChunkSAC四种Actor目标详解) — SAC Flow-G 内部的四种 loss 变体
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 完整推导
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 理论基础
- [离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础) — CQL/CalQL 的动机
- [动作分块 RL 基础](/系列/groot_rl_deep_dive/06_动作分块RL基础_QChunking到AQC回顾) — QC 理论
- [RLinf BC 到 RL 的 ACT 后训练架构](./RLinf_BC到RL的ACT后训练架构) — PPO 路线的 ACT 实现
- [GR00T N1.7 深度解析系列](/系列/groot_n1d7_deep_dive/index) — 模型架构参考
- [RLinf 深度解析系列](/系列/rlinf_deep_dive/index) — 训练框架参考
