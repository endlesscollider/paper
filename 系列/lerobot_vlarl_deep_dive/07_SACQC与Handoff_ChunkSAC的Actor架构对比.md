---
title: "07 SAC-QC vs Handoff vs ScoRe-Flow：Actor 架构对比"
series:
  id: lerobot_vlarl_deep_dive
  chapter: 7
order: 7
---

# SAC-QC 与 Handoff Chunk-SAC 的 Actor 架构对比

## 相关阅读

- [AWR-Flow Handoff 与 ConRFT 的 Actor 更新机制对比](./AWR_Flow_Handoff与ConRFT的Actor更新机制对比)
- [SAC Soft Actor-Critic 基础](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)
- [Flow Matching 基础](/前置知识/0013_前置知识_FlowMatching基础)
- [LeRobot-VLARL 全面拆解](/系列/lerobot_vlarl_deep_dive/)

---

## 一、问题定位

两个系统都叫"SAC + Action Chunk"，但 Actor 的内部结构完全不同：

| 系统 | Actor 类型 | 动作生成机制 |
|------|-----------|------------|
| **SAC-QC（dev_wq_temp_temp）** | Gaussian Policy MLP | state → mean/std → 高斯采样 → chunk |
| **Handoff Chunk-SAC（RLinf）** | GR00T Flow Action Head | state + noise → Flow denoising → chunk |

这不是"参数大小不同"这种量的差异——是**动作生成的数学模型**根本不同。

---

## 二、SAC-QC 的 Actor：Gaussian Chunk Policy

### 2.1 网络结构

```
观测编码 (SACQCObservationEncoder)
    ├── 图像 → CNN/ResNet → spatial embedding → (B, latent_dim)
    └── 状态 → Linear → LayerNorm → Tanh → (B, latent_dim)
    ↓ concat
obs_enc (B, total_latent)
    ↓
MLP: [256, 256] with LayerNorm + SiLU
    ↓
features (B, 256)
    ├── mean_layer: Linear(256, chunk_size × action_dim) → μ
    └── std_layer:  Linear(256, chunk_size × action_dim) → log σ → exp → σ
    ↓
TanhMultivariateNormalDiag(μ, σ)
    ↓
rsample() → action_chunk (B, chunk_size, action_dim)  ∈ [-1, 1]
```

### 2.2 动作随机性来源

SAC-QC 的动作随机性**完全来自高斯分布的采样**：

```python
# 策略输出均值和标准差
means = self.mean_layer(features)     # (B, chunk_size * action_dim)
log_std = self.std_layer(features)
std = torch.exp(log_std).clamp(std_min, std_max)

# 从高斯分布采样（reparameterization trick）
dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
actions = dist.rsample()  # μ + σ * ε, 其中 ε ~ N(0, I)
```

每次调用 `rsample()`，高斯噪声 $\varepsilon$ 不同 → 动作不同。std 越大 → 动作越随机（探索越多）。

### 2.3 Log-Probability 计算

因为是标准的参数化分布，log-prob 有解析表达式：

$$
\log \pi(a|s) = \log \mathcal{N}(u; \mu(s), \sigma(s)^2) - \sum_{i=1}^{d} \log(1 - \tanh^2(u_i))
$$

其中 $u = \tanh^{-1}(a)$（Tanh 逆变换），第二项是 Jacobian 修正。

这个 log-prob 可以直接用于：
- **Temperature 自适应**：$\alpha$ 自动调节探索量
- **Actor loss**：$L_{\text{actor}} = \alpha \log\pi(a|s) - Q(s,a)$
- **熵估计**：$H[\pi] \approx -\mathbb{E}[\log\pi]$

---

## 三、Handoff 的 Actor：GR00T Flow Action Head

### 3.1 网络结构

```
观测 (images + state + language)
    ↓
EagleBackbone → vl_embedding (B, N_tokens, 1536)
    ↓
FlowMatchingActionHead:
    初始噪声 z₀ ~ N(0, I)    shape: (B, 40, action_dim)
    ↓ [4 步 Flow denoising]
    for t in [0.0, 0.25, 0.5, 0.75]:
        action_features = ActionEncoder(z_t, t, embodiment_id)
        velocity = DiT(action_features, cross_attn=vl_embedding)
        z_{t+dt} = z_t + dt * velocity
    ↓
    action_chunk (B, 40, action_dim)
```

### 3.2 动作随机性来源

Handoff 的动作随机性来自**Flow path 的初始噪声 $z_0$**：

- 每次推理时，从 $\mathcal{N}(0, I)$ 采一个 40×action_dim 维度的噪声
- 不同的初始噪声沿不同的 ODE 路径积分，得到不同的动作 chunk
- 这个过程**没有解析的概率密度函数**

### 3.3 Log-Probability 问题

Flow Matching 策略没有封闭形式的 $\log\pi(a|s)$。要精确计算需要求解 ODE 的对数行列式（Hutchinson 估计器），计算开销极大。

RLinf 的解决方案：**完全不用 entropy / log-prob**。

```python
# RLinf 中的 alpha 处理
# actor_objective == "awr_flow" 时：
# - 不计算 log_pi
# - 不更新 alpha
# - 直接用 advantage 加权 SFT
# 等效于 alpha = 0（无熵正则）
```

这意味着 Handoff 放弃了 SAC 的"自动温度调节"机制——探索性完全由 Flow denoising 的固有随机性提供（初始噪声的方差）。

---

## 四、核心差异对照

### 4.1 动作分布的数学性质

| 性质 | SAC-QC (Gaussian) | Handoff (Flow) |
|------|-------------------|---------------|
| **分布族** | 参数化高斯 + Tanh squash | 隐式分布（由 ODE 定义） |
| **log π(a\|s)** | ✅ 解析表达式 | ❌ 需要 ODE 求解（不实际） |
| **采样** | μ + σ·ε，一次矩阵运算 | 多步 ODE 积分（4步 DiT forward） |
| **可控随机性** | σ 连续可调（通过 α 自适应） | 固定（由噪声方差决定） |
| **多模态** | 单高斯 → 单模态 | Flow 天然多模态 |

### 4.2 训练信号传导

| 维度 | SAC-QC | Handoff |
|------|--------|---------|
| **Critic → Actor 的梯度** | $\nabla_\theta Q(s, \pi_\theta(s))$ 通过采样动作传导 | 通过 advantage 加权 SFT 间接传导 |
| **温度 α** | 自动调节（SAC 标准做法） | 固定为 0（无熵正则） |
| **Actor loss** | $\alpha \log\pi - \min Q$ | $w_{\text{adv}} \cdot L_{\text{FM}}$ |
| **梯度到达的参数** | mean_layer + std_layer + MLP | DiT + ActionEncoder + ActionDecoder |

### 4.3 推理特性

| 维度 | SAC-QC | Handoff |
|------|--------|---------|
| **推理延迟** | ~2ms（一次 MLP forward） | ~50ms（4步 DiT forward） |
| **确定性推理** | 用 μ 直接作为动作（不采样） | 用固定种子的初始噪声 |
| **动作平滑性** | 取决于训练，可能跳变 | Flow 天然产生平滑轨迹 |
| **chunk 内时序一致性** | MLP 一次输出全部步，无内在时序结构 | DiT self-attention 建模步间关系 |

---

## 五、"Chunk 内时序建模"的本质差异

这是一个容易被忽略但影响巨大的区别。

### 5.1 SAC-QC：平铺输出

SAC-QC 的 `mean_layer` 输出维度是 `chunk_size × action_dim`。它一次性输出整个 chunk 的所有动作，**没有显式建模步与步之间的时序关系**：

```python
# 输出维度
mean_layer = nn.Linear(256, chunk_size * action_dim)  # 如 5*14=70 维
# reshape 回 chunk 形状
means = means.view(B, chunk_size, action_dim)
```

chunk 内不同 step 的动作之间的协调性完全依赖 MLP 隐式学习。

### 5.2 Handoff：DiT Self-Attention

Handoff 的 DiT 在 chunk 维度上做 Self-Attention：第 t 步的动作能"看到"第 t-1、t+1 步的动作。这让 chunk 内的动作轨迹天然具有时序一致性和平滑性。

```
action_features: (B, 40, hidden_dim)
                      ↓ Self-Attention (40 个 token 互相 attend)
model_output:    (B, 40, hidden_dim)  — 每步都考虑了其他步的信息
```

---

## 六、对 Critic 的影响

两者的 Critic 输入方式也不同：

| 系统 | Critic 输入 | 形状 |
|------|------------|------|
| SAC-QC | obs_encoding ⊕ flatten(action_chunk) | `(B, latent + chunk_size*action_dim)` |
| Handoff | obs_encoding ⊕ chunk-level features | 通过 Chunk-SAC 专用 Q head |

SAC-QC 把整个 chunk 展平后拼接——简单粗暴但有效。Handoff 可能有更复杂的 chunk-aware Critic 结构。

---

## 七、第三条路线：ScoRe-Flow 的 Score 引导采样

[ScoRe-Flow](/论文综述/080_ScoRe_Flow_Score引导的Flow策略RL微调) 提供了一条不同的路线：**不用 Gaussian policy，也不用 Flow 的"原生"采样——而是在 Flow 采样过程中注入 score drift + 学习的噪声，使之变成一个有解析 log-prob 的 SDE。**

### 7.1 动作生成方式

```
初始噪声 z₀ ~ N(0, I)
    ↓ [K 步 SDE 积分]
    for k = 0, ..., K-1:
        z_{k+1} = z_k + [v_θ(z_k, t_k) + α_ψ(t_k)·s_t(z_k)] · dt + σ_ϕ(t_k) · ε_k
    ↓
    action (action_dim)
```

### 7.2 与三者的对比

| 维度 | SAC-QC (Gaussian) | Handoff (Flow) | **ScoRe-Flow (SDE)** |
|------|-------------------|---------------|---------------------|
| **分布族** | 参数化高斯 | 隐式（ODE 定义） | 显式 SDE（高斯转移链） |
| **log π(a\|s)** | ✅ 解析 | ❌ 不可用 | ✅ 解析（高斯步累加） |
| **RL 算法** | SAC (off-policy) | AWR / direct Q | **PPO (on-policy)** |
| **α（温度/熵）** | 自动调节 | 固定为 0 | **PPO 的 clip 比例替代** |
| **Flow 网络** | 无 | 被训练更新 | **冻结（只加辅助项）** |
| **多模态性** | 单高斯 → 单模态 | 天然多模态 | 通过 score drift 实现多模态偏移 |
| **推理 steps** | 1 次 MLP | 4 次 DiT | K 次（通常 4-8）|
| **额外可训练参数** | 整个 Actor MLP | 整个 action head | **~几千**（α_ψ + σ_ϕ MLP） |

### 7.3 关键洞察：log-prob 问题的四种解法

Flow Matching 策略做 RL 的核心难题是"没有 log-prob"。四种方案分别绕过这个问题：

| 方案 | 做法 | 代价 |
|------|------|------|
| **SAC-QC** | 根本不用 Flow，用高斯直接参数化 | 失去 Flow 的多模态/平滑优势 |
| **Handoff** | 放弃 log-prob，不用熵正则（α=0），用 advantage 加权 SFT 替代 | 失去自动探索调节 |
| **ReinFlow** | 给 Flow 加噪声变成 SDE，每步高斯转移有解析 log-prob | 只能控制 variance，不能引导 mean |
| **ScoRe-Flow** | ReinFlow + score drift 修正 | 需要额外的 score scheduler |

### 7.4 ReinFlow 的具体机制

ReinFlow 比 ScoRe-Flow 更简单——它**只加噪声，不加方向引导**：

```
初始噪声 z₀ ~ N(0, I)
    ↓ [K 步 SDE 积分]
    for k = 0, ..., K-1:
        z_{k+1} = z_k + v_θ(z_k, t_k) · dt + σ_ϕ(t_k) · ε_k   ← 注意：没有 score drift
    ↓
    action
```

与 ScoRe-Flow 的对比：

| 维度 | ReinFlow | ScoRe-Flow |
|------|---------|-----------|
| SDE 中的项 | v_θ + σ·dW | v_θ + **α·s_t** + σ·dW |
| 额外学习的参数 | 只有 σ_ϕ | σ_ϕ **+ α_ψ** |
| 能引导 mean 方向 | ❌ | ✅ |
| 实现复杂度 | 极低 | 中等（需要计算 score function） |
| 收敛速度 | baseline | 快 2.4 倍 |

**ReinFlow 的定位**：它是"Flow + RL"领域的**最小可行方案**。如果你只想验证"给 Flow 加噪声 + PPO 到底能不能 work"，ReinFlow 是最简单的起点。确认 work 之后再加 score drift（升级为 ScoRe-Flow）。

---

## 八、什么时候用哪个

| 场景 | 推荐 | 原因 |
|------|------|------|
| 无预训练 VLA + 需要快速推理 | **SAC-QC** | 简单、快速、log-prob 可用 |
| 有 GR00T VLA + 需要动作质量 | **Handoff** | 利用 VLA 的多模态理解 |
| 需要精确的熵估计 / α 调节 | **SAC-QC** | 解析 log-prob |
| 需要 chunk 内平滑连续轨迹 | **Handoff** | DiT self-attention |
| 低延迟实时控制（>30Hz） | **SAC-QC** | 2ms vs 50ms |
| chunk_size > 20 | **Handoff** | MLP 平铺输出在大 chunk 上效果差 |

---

## 八、什么时候用哪个

| 场景 | 推荐 | 原因 |
|------|------|------|
| 无预训练 VLA + 需要快速推理 | **SAC-QC** | 简单、快速、log-prob 可用 |
| 有 GR00T VLA + 需要动作质量 | **Handoff** | 利用 VLA 的多模态理解 |
| 需要精确的熵估计 / α 调节 | **SAC-QC** | 解析 log-prob |
| 需要 chunk 内平滑连续轨迹 | **Handoff** | DiT self-attention |
| 低延迟实时控制（>30Hz） | **SAC-QC** | 2ms vs 50ms |
| chunk_size > 20 | **Handoff** | MLP 平铺输出在大 chunk 上效果差 |
| 有预训练 Flow + 不想改网络 + 有仿真器 | **ScoRe-Flow** | 极轻量，PPO 友好 |
| 快速验证"Flow+RL能否work" | **ReinFlow** | 实现最简单，改动最少 |
| ReinFlow 验证 OK 后要提升性能 | **ScoRe-Flow** | 加 score drift 即可 |
| 需要 on-policy PPO 的稳定性保证 | **ReinFlow / ScoRe-Flow** | 唯一有 log-prob + 不改 Flow 结构的方案 |

---

## 九、一句话总结

> **SAC-QC 是"用高斯分布直接参数化一整段动作轨迹"——数学简洁但表达力受限。**
> **Handoff 是"用 Flow denoising 迭代构造轨迹"——表达力强但失去了概率模型的解析性。**
> **ScoRe-Flow 是"在 Flow 路径上加引导把它变回概率模型"——两者兼得但需要 on-policy 采样。**

SAC-QC 适合"低维 + 短 chunk + 需要标准 SAC 全套机制"。Handoff 适合"高维 + 长 chunk + 有预训练 VLA + 可以牺牲 entropy 项"。ScoRe-Flow 适合"有预训练 Flow + 有仿真器 + 不想碰模型参数 + 需要 log-prob"。
