---
title: SAC-Flow：用 SAC 直接端到端训练 Flow 策略
order: 279
tags: [强化学习, Flow Matching, SAC, Off-Policy, 梯度稳定性]
category: 精读
star: 5
---

# SAC-Flow：用 SAC 直接端到端训练 Flow 策略 深度精读

> **论文标题**: SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling  
> **作者**: Yixian Zhang, Shu'ang Yu, Tonghe Zhang, Mo Guang 等  
> **机构**: Tsinghua University, CMU, Li Auto, Shanghai AI Lab  
> **发表**: arXiv:2509.25756v2, Oct 2025  
> **代码**: [github.com/Elessar123/SAC-FLOW](https://github.com/Elessar123/SAC-FLOW)

**知识链接**：
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 的完整原理
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — Flow Matching 基本原理
- [FQL：Flow Q-Learning](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) — 对比：用蒸馏绕开梯度问题的方案
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — 核心困难的背景

---

## 一、这篇论文解决什么问题

### 1.1 背景：Flow 策略 + RL 的梯度困境

Flow Matching 策略（如 π₀ 用的那种）通过多步 ODE 积分生成动作：从噪声 $A_0 \sim \mathcal{N}(0, I)$ 出发，经过 $K$ 步 Euler 更新得到最终动作 $A_1$：

$$
A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot v_\theta(t_i, A_{t_i}, s)
$$

这个多步过程在**行为克隆**中没有问题——每步独立地让 $v_\theta$ 逼近数据的速度场就行。

但如果要用 **off-policy RL（如 SAC）** 训练 flow 策略，就必须把 Q 值的梯度**反传穿过整个 $K$ 步积分**——就像 BPTT（backpropagation through time）穿过一个 $K$ 层的 RNN。这会导致：

- **梯度爆炸**：$K$ 步累乘让梯度指数增长
- **梯度消失**：某些方向的信号被压缩到零

### 1.2 之前的方案都在"绕路"

| 方法 | 策略 | 代价 |
|------|------|------|
| [FQL](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) | 蒸馏一个单步学生网络，对学生做 RL | 丢失了 flow 的多模态表达力 |
| FlowRL | 用 Wasserstein 约束做 surrogate objective，不穿过 flow | 不是直接优化 SAC 目标 |
| ReinFlow | 用 on-policy PPO，不需要穿过 flow 反传 Q 梯度 | 样本效率低 |

**本文的立场**：这些都是 workaround。能不能**直接用 SAC 端到端训练 flow 策略**？

### 1.3 本文的核心洞察

**关键发现**：Flow 的 $K$ 步 Euler 积分，在代数结构上等价于一个**残差 RNN** 的 $K$ 步前向传播。

$$
A_{t_{i+1}} = A_{t_i} + f_\theta(t_i, A_{t_i}, s) \quad \longleftrightarrow \quad h_{i+1} = h_i + \text{RNNCell}(x_i, h_i)
$$

其中 $f_\theta(\cdot) = \Delta t_i \cdot v_\theta(\cdot)$ 就是 RNN cell，$A_{t_i}$ 是 hidden state，$(t_i, s)$ 是输入。

**这解释了为什么标准 flow + SAC 训不稳**：和 vanilla RNN 一样的梯度病态。

**解决方案也顺理成章**：既然问题等价于 RNN 的梯度不稳定，就用解决 RNN 问题的现代方法——**GRU 门控**或 **Transformer decoder**——来重新参数化速度场 $v_\theta$。

---

## 二、方法：两种稳定的速度网络设计

### 2.1 Flow-G：GRU 风格的门控速度

原始 flow 的每步更新是直接加残差 $A_{t_{i+1}} = A_{t_i} + \Delta t \cdot v_\theta$。Flow-G 给它加一个门控：

$$
A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot \big(g_i \odot (\hat{v}_\theta(t_i, A_{t_i}, s) - A_{t_i})\big)
$$

$$
g_i = \sigma\big(z_\theta(t_i, A_{t_i}, s)\big) \in (0, 1)^d
$$

**为什么需要这个公式**：GRU 的核心思想是"门控更新幅度"——在梯度容易爆炸的步骤让门 $g_i \approx 0$（跳过更新），在有效的步骤让门打开。这自动调节了每步对 hidden state 的影响，抑制了梯度的指数增长。

> 一句话直觉：给 flow 的每步更新装一个"刹车"，网络自己学会什么时候踩油门（大幅更新动作），什么时候踩刹车（保持不变）。

**逐项拆解**：
- $g_i \in (0,1)^d$：每个动作维度独立的门控信号。$\sigma$ 是 sigmoid，输出在 0-1 之间
- $\hat{v}_\theta$：候选速度网络（类似 GRU 的候选 hidden state）
- $\hat{v}_\theta - A_{t_i}$：候选"新位置"和当前位置的差——即"要不要跳到候选位置"
- $g_i \odot (\cdots)$：门控决定每个维度"跳多远"。$g_i = 0$ 意味着保持原地，$g_i = 1$ 意味着完全跳到候选位置

**代入数字**（1 维，简化）：
- 当前 $A_{t_i} = 0.3$，候选 $\hat{v}_\theta = 0.9$，门 $g_i = 0.2$
- 差值 = $0.9 - 0.3 = 0.6$
- 门控后 = $0.2 \times 0.6 = 0.12$
- 新位置 = $0.3 + \Delta t \times 0.12$（只移动了一小步，梯度不会爆）
- 如果 $g_i = 0.9$：门控后 = $0.9 \times 0.6 = 0.54$（大步更新）

**为什么是这个形式**：这就是 GRU 的 update gate 机制。区别只是写成了 flow 速度场的形式——$(g_i \odot (\hat{v}_\theta - A_{t_i}))$ 实际上就是门控后的速度。

### 2.2 Flow-T：Transformer Decoder 风格的速度

另一种设计：用 Transformer Decoder 的结构来产生速度。核心思想是把"当前动作-时间 token"通过 cross-attention 和 FFN 逐层精炼：

$$
\Phi_A^{(l)} = Y_i^{(l)} + \text{FFN}_l(\text{LN}(Y_i^{(l)}))
$$
$$
Y_i^{(l)} = \Phi_A^{(l-1)} + \text{CrossAttn}_l(\text{LN}(\Phi_A^{(l-1)}), \text{context}=\text{LN}(\Phi_S))
$$

最终速度由 decoder 输出经线性投影得到：

$$
v_\theta(t_i, A_{t_i}, s) = W_o \cdot \text{LN}(\Phi_A^{(L)})
$$

> 一句话直觉：每步不再是一个简单 MLP 算速度，而是让"动作 token"通过多层 Transformer 查询"状态信息"来精炼自己——每层都有 residual connection + LayerNorm，天然梯度友好。

**为什么 Transformer 稳定**：Pre-norm residual blocks + cross-attention 的结构天然具有良好的梯度特性——每层的 residual connection 保证了梯度至少有一条"直通高速公路"，不会消失；LayerNorm 控制了每层输出的量级，不会爆炸。

### 2.3 两种设计的对比

| | 原始 MLP 速度 | Flow-G（门控） | Flow-T（Transformer） |
|--|-------------|-------------|---------------------|
| 对应的序列模型 | Vanilla RNN | GRU | Transformer Decoder |
| 梯度稳定性 | ❌ 爆炸/消失 | ✅ 门控调节 | ✅ Residual + LN |
| 参数量 | 小 | 中 | 中-大 |
| 表达力 | 有限 | 好 | 最强 |
| 最佳场景 | 不推荐直接 RL | 通用，轻量 | 复杂操作任务 |

---

## 三、怎么算 log-prob：噪声增广 Rollout

### 3.1 问题：确定性 flow 没有解析 log-prob

SAC 需要 $\log\pi(a|s)$ 来计算熵项。但确定性 flow 的 $K$ 步积分是**确定性的**（给定 $A_0$，输出 $A_1$ 唯一确定），它的 marginal density $\pi_\theta(a|s)$ 需要对所有可能的 $A_0$ 积分——计算上不可行。

### 3.2 解法：加噪声让每步变成高斯转移

在每步 Euler 更新中注入微小噪声，把确定性 ODE 变成随机过程：

$$
A_{t_{i+1}} = A_{t_i} + b_\theta(t_i, A_{t_i}, s) \cdot \Delta t_i + \sigma_\theta \sqrt{\Delta t_i} \cdot \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, I)
$$

其中 $b_\theta$ 是修正后的 drift（对原始速度做补偿，保证终点分布不变）。

**关键好处**：现在每步的转移概率是显式高斯：

$$
\eta_\theta(A_{t_{i+1}} | A_{t_i}, s) = \mathcal{N}(A_{t_i} + b_\theta \Delta t_i, \; \sigma_\theta^2 \Delta t_i \cdot I)
$$

整条路径的联合密度可以写成 product of Gaussians：

$$
p_c(\mathcal{A}|s) = \zeta(A_0) \prod_{i=0}^{K-1} \eta_\theta(A_{t_{i+1}} | A_{t_i}, s) \cdot |\det \mathcal{J}(a)|^{-1}
$$

其中 $\mathcal{J}$ 是 tanh squashing 的 Jacobian。

> 一句话直觉：给 flow 的每步加一点噪声，就把它变成了"$K$ 步高斯链"——每步的 log-prob 都能解析算，加起来就是整条路径的 log-prob。

这个 $\log p_c$ 就可以直接当作 SAC 的 $\log\pi$ 使用。$\sigma_\theta$ 通常固定为 0.1（实践中足够）。

---

## 四、完整算法：SAC-Flow 的训练流程

### 4.1 Actor Loss

$$
L_{\text{actor}}(\theta) = \alpha \log p_c(\mathcal{A}^\theta | s_h) - Q_\psi(s_h, a_h^\theta), \quad a_h^\theta = \tanh(A_{t_K}^\theta)
$$

**为什么需要这个公式**：这就是标准 SAC 的 Actor loss，只是把"高斯策略的 $\log\pi$"替换成了"$K$ 步噪声 flow 路径的 $\log p_c$"。结构完全一样——最大化 Q 值同时保持策略熵。

> 一句话直觉：和标准 SAC 完全一样的目标——选好动作但别太确定。只是"策略"从高斯变成了 flow。

**逐项拆解**：
- $\alpha \log p_c(\mathcal{A}^\theta | s_h)$：熵正则项。$p_c$ 越大 = 策略越确定 = 惩罚越大 → 鼓励探索
- $-Q_\psi(s_h, a_h^\theta)$：最大化 Q 值。梯度从 $Q$ 穿过 $a_h^\theta = \tanh(A_{t_K}^\theta)$ 再穿过整个 $K$ 步 flow 到达 $\theta$
- 梯度不会爆炸因为 flow 用了 Flow-G 或 Flow-T 的稳定参数化

### 4.2 Critic Loss

$$
L_{\text{critic}}(\psi) = \Big[Q_\psi(s_h, a_h) - \big(r_h + \gamma Q_{\bar\psi}(s_{h+1}, a_{h+1}) - \alpha \log p_c(\mathcal{A}_{h+1} | s_{h+1})\big)\Big]^2
$$

和标准 SAC Critic loss 完全一样：$(s_h, a_h, r_h, s_{h+1})$ 从 Replay Buffer 采，$a_{h+1}$ 由当前 flow 策略在 $s_{h+1}$ 重新采样。

### 4.3 Offline-to-Online 变种

对于稀疏奖励任务，先用 flow matching 预训练（行为克隆），再切换到 RL。在线阶段的 Actor loss 加一个 proximity regularizer：

$$
L_{\text{actor}}^o(\theta) = \alpha \log p_c(\mathcal{A}^\theta | s_h) - Q_\psi(s_h, a_h^\theta) + \beta \|a_h^\theta - a_h\|_2^2
$$

$\beta$ 项让策略不要偏离 Buffer 中的行为太远——这是 offline-to-online 的标准做法。$\beta$ 在 OGBench 上设为 100-300，Robomimic 上设为 10000（非常保守）。

### 4.4 完整伪代码

```
初始化：Flow 策略 π_θ（Flow-G 或 Flow-T）、Critic Q_ψ、Target Q̄_ψ、Buffer B

每步：
  1. 用 π_θ 采样动作（K 步噪声 flow rollout → tanh 压缩）
  2. 与环境交互，存 (s, a, r, s') 入 Buffer
  3. 从 Buffer 采 batch
  4. 更新 Critic：最小化 Soft Bellman TD error（target 含 log p_c）
  5. 更新 Actor：最小化 α·log p_c - Q（梯度穿过 K 步 flow，稳定无爆炸）
  6. 更新 α：自动温度调节
  7. 软更新 Target Critic（EMA）
```

---

## 五、和其他方法的根本区别

| 方法 | 怎么处理"梯度穿过 flow"问题 | 保留 flow 的多模态？ | Off-policy？ |
|------|--------------------------|-------------------|------------|
| **SAC-Flow（本文）** | 重新参数化速度网络（GRU/Transformer），让梯度稳定 | ✅ 完整保留 | ✅ |
| FQL / QC-FQL | 蒸馏成单步网络，只对学生做 RL | ❌ 学生是单峰高斯 | ✅ |
| FlowRL | 用 Wasserstein surrogate 代替 SAC loss | 部分保留 | ✅ |
| ReinFlow / DPPO | 用 on-policy PPO，不需要 Q 梯度穿过 flow | ✅ | ❌（on-policy） |

**本文的定位**：第一个能**直接用 SAC 端到端训练多步 flow 策略**的方法，无需蒸馏、无需 surrogate、保留完整多模态表达力。

---

## 六、实验结果

### 6.1 From-scratch 训练（MuJoCo 密集奖励）

| 任务 | SAC-Flow-T | SAC-Flow-G | FlowRL | DIME | SAC (高斯) |
|------|-----------|-----------|--------|------|-----------|
| HumanoidStandup | **最高**（+130% over baseline） | 接近最高 | 中 | 中 | 低 |
| Ant | 最高 | 最高 | 中-高 | 中-高 | 中 |
| Walker2d | 最高 | 最高 | 中 | 中 | 中 |

### 6.2 Offline-to-Online（OGBench 稀疏奖励）

在 cube-triple 和 cube-quadruple 任务上，SAC-Flow-T 达到最高成功率，比 QC-FQL 高约 60%。

### 6.3 消融：梯度稳定性

论文测量了训练过程中梯度 norm 随 flow step 的变化：
- **Naive SAC + 标准 MLP flow**：梯度 norm 从 step $k=3$（最后一步）到 $k=0$（第一步）**指数增长**
- **SAC Flow-G / Flow-T**：梯度 norm 在所有步骤保持稳定（最大变化 0.29）

这直接验证了核心假设：梯度不稳定是标准 flow + SAC 失败的根本原因，重新参数化速度网络能解决。

---

## 七、对读者最重要的 Takeaway

1. **Flow 的多步采样 = RNN 的多步前向**。理解了这个等价关系，就理解了为什么 off-policy RL 训 flow 困难——同样的梯度病态。

2. **解决方案是改速度网络的参数化，不是改 RL 算法**。SAC 本身不需要任何修改，只需要把 flow 内部的 MLP 速度网络换成 GRU cell 或 Transformer decoder block。

3. **噪声增广 rollout 解决 log-prob 问题**。给 flow 每步加微小噪声，把确定性 ODE 变成 K 步高斯链，log-prob 可以解析计算。这个 trick 是让 SAC 的熵目标能用在 flow 上的关键。

4. **不再需要蒸馏**。FQL 路线把 flow 蒸馏成单步网络再做 RL——丢失了多模态能力。SAC-Flow 直接端到端训练原始 flow，保留了完整表达力。

---

## 延伸阅读

- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 框架本身的详细原理
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — Flow Matching 的数学基础
- [FQL：Flow Q-Learning](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) — 对比方案：蒸馏路线
- [FlowRL：Flow VLA 的在线 RL 微调](./018_FlowRL_Flow_VLA的在线RL微调) — 对比：on-policy PPO 路线
- [Q-Chunking：RL 与动作分块](./071_QChunking_RL与动作分块) — QC-FQL 的完整精读
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — 更宽泛的背景

**原始论文**：Zhang et al., "SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling", arXiv 2509.25756, 2025
