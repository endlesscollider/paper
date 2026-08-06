---
title: "08 Chunk 长度与时间尺度：40步 vs 16步 vs 5步 vs 单步"
series:
  id: lerobot_vlarl_deep_dive
  chapter: 8
order: 8
---

# Chunk 长度与时间尺度：Handoff 40步 vs ConRFT 16步 vs SAC-QC 5步

## 相关阅读

- [AWR-Flow Handoff 与 ConRFT 的 Actor 更新机制对比](./AWR_Flow_Handoff与ConRFT的Actor更新机制对比)
- [SAC-QC 与 Handoff Chunk-SAC 的 Actor 架构对比](./SACQC与Handoff_ChunkSAC的Actor架构对比)
- [Q 函数与 Value 函数](/前置知识/000b_前置知识_Q函数与Value函数)

---

## 一、三个系统的 Chunk 配置

| 系统 | chunk_size | 典型 fps | 每 chunk 物理时间 | denoising steps |
|------|-----------|---------|-----------------|----------------|
| **Handoff (RLinf)** | 40 | 10 Hz | **4 秒** | 4 |
| **ConRFT** | 16 | 10-30 Hz | 0.5-1.6 秒 | 1-2 (CP) |
| **SAC-QC** | 5 | 10 Hz | **0.5 秒** | N/A (高斯) |

这些数字不是"参数调了调不一样"——它们代表完全不同的**时间抽象层级**，对 RL 训练的每个环节都有深远影响。

---

## 二、一个 Chunk = 一个 RL Transition

在 Chunk-level RL 中，一个 "transition" 不再是"一帧动作"，而是"一整段动作轨迹"：

```
标准 RL:   s_t → a_t → r_t → s_{t+1}           (1 个物理步)
Chunk RL:  s_t → [a_t, a_{t+1}, ..., a_{t+C-1}] → R → s_{t+C}  (C 个物理步)
```

这意味着：

| 系统 | 同样 1000 个 RL transitions | 实际代表的物理步数 |
|------|---------------------------|-------------------|
| Handoff (C=40) | 1000 × 40 = **40,000 步** | |
| ConRFT (C=16) | 1000 × 16 = **16,000 步** | |
| SAC-QC (C=5) | 1000 × 5 = **5,000 步** | |

所以"训练了 10000 步"在不同系统中含义完全不同。

---

## 三、折扣因子 γ 的时间尺度效应

### 3.1 Chunk-level Discount

在 chunk-level RL 中，discount factor 作用于 **chunk 之间**而非 step 之间：

$$
Q(s_t, \text{chunk}_t) = R_t + \gamma \cdot Q(s_{t+C}, \text{chunk}_{t+C})
$$

这里的 $\gamma$ 的"有效折扣"取决于 chunk 长度。如果我们想要等价的每步折扣 $\gamma_{\text{step}}$，关系是：

$$
\gamma_{\text{chunk}} = \gamma_{\text{step}}^{C}
$$

### 3.2 数值对比

假设我们希望的"有效每步折扣"是 $\gamma_{\text{step}} = 0.99$（RL 中常用）：

| 系统 | C | $\gamma_{\text{chunk}} = 0.99^C$ | 含义 |
|------|---|-----|------|
| Handoff | 40 | $0.99^{40} = 0.669$ | 下一个 chunk 的 Q 只贡献 67% |
| ConRFT | 16 | $0.99^{16} = 0.851$ | 下一个 chunk 的 Q 贡献 85% |
| SAC-QC | 5 | $0.99^{5} = 0.951$ | 下一个 chunk 的 Q 贡献 95% |

**影响**：
- Handoff 的 Critic 几乎只关注"当前 chunk 内的即时收益"——未来被严重折扣
- SAC-QC 的 Critic 能看到较远的未来——更适合需要长期规划的任务
- ConRFT 居中

### 3.3 实际配置中的 γ

实际上各系统用的 γ 是**直接配置在 chunk 级别**的：

| 系统 | 配置的 γ | 等效每步 γ |
|------|---------|-----------|
| Handoff | 0.97（chunk级） | $0.97^{1/40} = 0.9992$ |
| ConRFT | 0.97（chunk级） | $0.97^{1/16} = 0.9981$ |
| SAC-QC | 0.97（chunk级） | $0.97^{1/5} = 0.9939$ |

同样的 `gamma=0.97` 配置，在不同 chunk 长度下的"视野"：

- Handoff：一个 chunk 内 40 步的 reward 会被从 γ⁰ 到 γ⁰（因为 chunk 内 reward 是累加的）衰减
- SAC-QC：一个 chunk 5 步很短，γ 主要影响跨 chunk 的长期规划

---

## 四、Chunk 内 Reward 的处理

### 4.1 Handoff：逐步 reward 在 chunk 内累积

RLinf 在构造 chunk transition 时，chunk 内每一步都有独立的 reward：

```python
chunk_rewards = torch.zeros(num_chunks, chunk_length)  # (N, 40)
# 每步都有 reward 值
chunk_rewards[chunk_index, :length] = rewards[start : start + length]
```

Critic 看到的是 **chunk 内 40 步 reward 的完整向量**（或其折扣累加）。

### 4.2 SAC-QC / ConRFT：chunk reward 是标量

```python
# actor_online_training.py
rewards_chunk = []
for offset in range(self.chunk_size):
    rewards_chunk.append(transitions[src]["reward"])
item["reward"] = torch.cat(rewards_chunk, dim=0)  # (chunk_size,) 但通常只用总和
```

ConRFT 的 Critic 通常只看 chunk 的**总 reward**（或 MC return），不看 chunk 内每一步的分布。

### 4.3 影响

| 处理方式 | 优势 | 劣势 |
|---------|------|------|
| 逐步 reward + valid mask | 精确：Critic 知道 chunk 内哪些步贡献了 reward | 复杂：需要 chunk 内部的时序建模 |
| 总 reward 标量 | 简单：一个数就够 | 粗粒度：不知道 chunk 的哪个部分好/坏 |

Handoff 选择前者是因为 chunk 长达 40 步（4 秒），内部时序结构很重要——"前 1 秒对准了但后 3 秒偏了"和"全程都差"是不同情况。SAC-QC 的 chunk 只有 5 步（0.5 秒），内部差异没那么显著。

---

## 五、对训练效率的影响

### 5.1 样本效率（per physical step）

假设环境中跑了 1000 个物理步：

| 系统 | 可产生的 RL transitions | UTD=2 时的梯度更新次数 |
|------|------------------------|----------------------|
| Handoff (C=40) | 25 个 | 50 次 |
| ConRFT (C=16) | 62 个 | 124 次 |
| SAC-QC (C=5) | 200 个 | 400 次 |

SAC-QC 的训练循环频率比 Handoff 快 8 倍！这在真机 RL（数据极其昂贵）中差别巨大。

### 5.2 但更新质量不同

每次更新中，模型"看到的信息量"不同：

| 系统 | 一个 transition 包含的信息 |
|------|--------------------------|
| Handoff | 40 步动作 + 40 步 reward + 视觉变化 → 信息密度高 |
| SAC-QC | 5 步动作 + 5 步 reward + 微小视觉变化 → 信息密度低 |

长 chunk 的一个 transition 可能包含"接近→接触→抓取"的完整语义段，短 chunk 可能只包含"向右移动一点点"。

---

## 六、Denoising Steps 与推理延迟

### 6.1 Handoff：4 步 denoising

```
初始噪声 z₀ (40 × action_dim)
    → DiT forward ×4 → 动作 chunk (40 × action_dim)
```

每步 DiT forward ~12ms → 总计 ~50ms → 对应 10Hz 控制频率（100ms/step），50ms 推理 + 50ms 余量。

### 6.2 ConRFT：1-2 步 denoising（Consistency Policy）

```
初始噪声 z₀ (16 × action_dim)
    → CP forward ×1 → 动作 chunk (16 × action_dim)
```

Consistency Policy 的优势：1 步就出结果，推理 ~10ms。这让 ConRFT 有余量做 best-of-N（N=4 时 ~40ms）。

### 6.3 SAC-QC：无 denoising

```
obs_encoding
    → MLP forward ×1 → μ + σ → 采样 → chunk (5 × action_dim)
```

纯 MLP，~2ms。极致低延迟。

---

## 七、"相同 max_steps 不等于相同物理量"

这是一个工程中容易出错的地方。当你看到配置文件中：

```yaml
# RLinf
online_steps: 10000     # = 10000 × 40 = 400,000 物理步 = 40,000 秒 @ 10Hz

# ConRFT  
online_steps: 10000     # = 10000 × 16 = 160,000 物理步 = 16,000 秒 @ 10Hz

# SAC-QC
online_steps: 10000     # = 10000 × 5 = 50,000 物理步 = 5,000 秒 @ 10Hz
```

同样的 `online_steps=10000`，在 Handoff 中意味着约 11 小时的机器人运行时间，在 SAC-QC 中只有约 1.4 小时。

**对比实验时必须标准化"物理步数"或"机器人运行时间"，不能直接比较 "optimization steps"。**

---

## 八、对 Episode 结构的影响

假设一个任务平均需要 150 个物理步完成：

| 系统 | 一个 episode 包含的 RL transitions |
|------|----------------------------------|
| Handoff (C=40) | 150/40 ≈ **4 个 chunks** |
| ConRFT (C=16) | 150/16 ≈ **9 个 chunks** |
| SAC-QC (C=5) | 150/5 = **30 个 chunks** |

Handoff 的一个 episode 只有 4 个 RL transitions——这意味着 Critic 只有 4 个"时间点"来学习 Q 函数。如果 episode 有 4 个关键阶段（approach → contact → grasp → lift），每个阶段刚好一个 chunk。

SAC-QC 的 30 个 transitions 给 Critic 提供了更细粒度的时间信息——但每个 transition 的信息量较少。

---

## 九、设计选择建议

| 考虑因素 | 推荐 chunk 长度 | 原因 |
|---------|----------------|------|
| 任务需要精确的接触力控制 | 短（5-10） | 细粒度 reward 反馈 |
| 任务主要是 reach + grasp 两阶段 | 中（16-20） | 一个 chunk 覆盖一个语义段 |
| 任务是长序列操作（>5秒） | 长（40+） | 一个 chunk 包含完整子任务 |
| 推理延迟敏感（>30Hz） | 短 | 短 chunk = 少 denoising steps |
| VLA 预训练用的 chunk_size 固定 | 与预训练一致 | 不匹配会损坏学到的时序模式 |

---

## 十、ReinFlow / ScoRe-Flow 的 Chunk 处理方式

[ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) 和 [ScoRe-Flow](/论文综述/080_ScoRe_Flow_Score引导的Flow策略RL微调) 在 chunk 处理上与前三者都不同：**它们都使用 step-level MDP（单步动作），不使用 action chunk。**

### 10.1 单步动作 vs Action Chunk

| 维度 | Handoff | ConRFT | SAC-QC | **ReinFlow / ScoRe-Flow** |
|------|---------|--------|--------|--------------------------|
| 输出粒度 | 40 步 chunk | 16 步 chunk | 5 步 chunk | **1 步** |
| RL transition | 40 物理步 | 16 物理步 | 5 物理步 | **1 物理步** |
| 每步推理次数 | 1次/40步 | 1次/16步 | 1次/5步 | **每步都推理** |
| γ 的含义 | chunk 间折扣 | chunk 间折扣 | chunk 间折扣 | **标准每步折扣** |

### 10.2 对训练效率的影响

| 系统 | 1000 物理步产生的 RL transitions |
|------|-------------------------------|
| Handoff (C=40) | 25 |
| ConRFT (C=16) | 62 |
| SAC-QC (C=5) | 200 |
| **ReinFlow / ScoRe-Flow (C=1)** | **1000** |

最细粒度的 reward 信号——每步都有独立的 Q 估计。代价是推理频率极高。

### 10.3 为什么不用 Chunk

这两个方法的设计目的是精确控制 Flow 策略的分布。chunk-level RL 让这变得更复杂：
- chunk 内 K 步 denoising 和 chunk 的 T 步物理步是两个不同维度
- 对整个 chunk 用一个 reward → 不知道 chunk 内哪一步需要修正
- PPO 的 log-prob 要求每步都能计算——chunk 级 log-prob 需要联合分布建模

### 10.4 ReinFlow vs ScoRe-Flow 推理对比

| 维度 | ReinFlow | ScoRe-Flow |
|------|---------|-----------|
| 每步操作 | v_θ + σ_ϕ + 噪声 | v_θ + α_ψ·score + σ_ϕ + 噪声 |
| 延迟/步 | ~10ms (K=4) | ~12ms (K=4 + score) |
| 收敛速度 | baseline | 快 2.4 倍 |

### 10.5 完整推理延迟对比

| 系统 | 每物理步推理延迟 | 适合的控制频率 |
|------|-----------------|---------------|
| Handoff | ~50ms/40步 ≈ 1.25ms/步 | 高 |
| ConRFT | ~40ms/16步 ≈ 2.5ms/步 | 高 |
| SAC-QC | ~2ms/5步 ≈ 0.4ms/步 | 极高 |
| ReinFlow | ~10ms/步 | 中 |
| ScoRe-Flow | ~12ms/步 | 中 |

---

## 十一、总结

| 维度 | Handoff (40) | ConRFT (16) | SAC-QC (5) | ReinFlow (1) | ScoRe-Flow (1) |
|------|-------------|-------------|-----------|-------------|---------------|
| **时间尺度** | 4 秒/chunk | 0.5-1.6 秒 | 0.5 秒 | **逐步** | **逐步** |
| **信息密度** | 高 | 中 | 低 | 最低 | 最低 |
| **训练频率** | 低 | 中 | 高 | **最高** | **最高** |
| **γ 视野** | 短程 | 中程 | 长程 | 标准 MDP | 标准 MDP |
| **推理延迟** | ~50ms DiT | ~10ms CP | ~2ms MLP | ~10ms SDE | ~12ms SDE |
| **chunk 内建模** | DiT Self-Attention | CP 内建 | 无（MLP 平铺） | N/A（无 chunk） |
| **Padding 问题** | 有（尾部填充） | 有 | 有 | **无** |

核心 takeaway：
- **chunk_size 不是超参——它决定了 RL 的时间抽象层级**
- Handoff/ConRFT/SAC-QC 都在"chunk-level MDP"框架下操作
- ScoRe-Flow 选择了"step-level MDP"——更精确但推理更频繁
- **没有 padding 问题**是 ScoRe-Flow 的一个隐含优势——每步独立，不存在"episode 尾部 chunk 不完整"的情况
