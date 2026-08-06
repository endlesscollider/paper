---
title: "阶段一 Critic：Cal-QL 保守项"
series:
  id: groot_conrft_rlinf_deep_dive
  chapter: 8
order: 8
---

# 第 08 章 阶段一 Critic：Cal-QL 保守项

> 前情提要：第 06 章讲了 critic 的结构和 TD target，第 07 章讲了带 `chunk_sac_mc_return` 的 BC replay。这一章把两者接起来——阶段一在标准 TD loss 之外加的那一项 Cal-QL 保守项：12 个候选动作从哪来、`logsumexp` 的归一化常数是多少、`cql_scale` 那个分支为什么在当前配置下恒为 1，以及这一项让阶段一慢了多少。

## 知识链接

- 上一章：[阶段一数据侧：BC replay 与 MC 回报](./07_阶段一数据侧)
- 下一章：[阶段一 Actor：BC 加轻量 Q](./09_阶段一Actor目标)
- [系列目录](./index)
- 前置：[CQL 保守 Q 学习](/前置知识/002g_前置知识_CQL保守Q学习) — **必读**，本章是它在 chunk 动作空间上的落地
- 前置：[Cal-QL 校准保守 Q 学习](/前置知识/002h_前置知识_CalQL校准保守Q学习) — **必读**，MC 下界的原理
- 前置：[离线强化学习基础](/前置知识/000s_前置知识_离线强化学习基础)
- 相关：[第 01 章 9.6 / 9.12](./01_全链路总览#9.-问题-风险与实验安排隐患清单)

---

## 1. 挂载点：`forward_critic` 的三行

ConRFT 对 critic 的全部改动就是三行代码。基类算完标准 TD loss，ConRFT 在离线阶段追加保守项：

```python
def forward_critic(self, *args, **kwargs):
    output = super().forward_critic(*args, **kwargs)
    if not self._conrft_offline_stage:
        return output                                          # 在线阶段到此为止
    data = args[0] if args else kwargs["data"]
    cql_loss, cql_metrics = self._compute_conservative_loss(data)
    cql_scale = (
        1.0 / self.gradient_accumulation
        if self.critic_objective is not None
        and self.critic_objective.uses_distributed_normalizers
        else 1.0
    )
    output.loss = output.loss + self.conrft.cql_alpha * cql_scale * cql_loss
    self._conrft_critic_metrics.append({
        "cql_loss": cql_loss.detach().item(),
        **{key: value.item() for key, value in cql_metrics.items()},
    })
    return output
```

这个结构值得注意：**保守项是加在 `CriticObjectiveOutput.loss` 上的，不是单独反向传播的。** 也就是 TD 项和 CQL 项共用一次 `backward()`，梯度自然相加。这是最省事也最不容易出错的做法——不需要担心两次 backward 之间的梯度累积状态。

`_conrft_critic_metrics` 是一个 list，每个 micro-batch 追加一条，`update_one_epoch` 结束时求平均并加上 `conrft/` 前缀。

## 2. 十二个候选动作

保守项的核心是"采一批 OOD 候选动作，压低它们的 Q 值"。在 992 维的 chunk 动作空间里，候选怎么采是决定性的。

```python
def _compute_conservative_loss(self, data):
    actions = data["chunk_sac_action"]
    batch_size = actions.shape[0]
    samples = self.conrft.cql_n_actions                     # 4
    random_actions = torch.empty(
        batch_size, samples, *actions.shape[1:],            # [B, 4, 16, 62]
        device=actions.device, dtype=actions.dtype,
    )
    if self.conrft.cql_action_sample_method == "uniform":
        random_actions.uniform_(-1.0, 1.0)
    else:
        random_actions.normal_()

    current_inputs = _conservative_q_inputs(data)
    with torch.no_grad(), conrft_eval_context(self.model):
        current_policy_actions = self.model(
            forward_type=ForwardType.SAC,
            forward_inputs=_repeat_batch_tree(current_inputs, samples),
        )["actions"].reshape(batch_size, samples, *actions.shape[1:])
        next_policy_actions = self.model(
            forward_type=ForwardType.SAC,
            forward_inputs=_repeat_batch_tree(extract_chunk_sac_next_inputs(data), samples),
        )["actions"].reshape(batch_size, samples, *actions.shape[1:])
    candidate_actions = torch.cat(
        (random_actions, current_policy_actions, next_policy_actions), dim=1
    )                                                        # [B, 12, 16, 62]
```

**`cql_n_actions: 4` 不是候选总数，而是每一路的数量。总候选数是 $3 \times 4 = 12$。** 这一点在读配置时极易误解。

三路来源各自要解决什么问题：

| 来源 | 数量 | 分布 | 它覆盖的风险 |
|------|------|------|--------------|
| `random_actions` | 4 | $\mathcal{U}(-1,1)^{16\times62}$ | 策略当前完全不去、但一旦 $Q$ 虚高就会跑过去的区域 |
| `current_policy_actions` | 4 | $\pi(\cdot \mid s)$ | 策略**现在**最想选的动作，最紧迫的过估计风险点 |
| `next_policy_actions` | 4 | $\pi(\cdot \mid s')$ | 会出现在下一轮 TD target 里的动作，提前压住等于直接抑制 target 虚高 |

**第三路最容易被误读**，单独说清楚：`next_policy_actions` 是在**下一状态 $s'$** 上采的动作，但它们被送去评估的是**当前状态 $s$** 的 Q 值：

```python
repeated_inputs = _repeat_batch_tree(current_inputs, candidate_count)   # ← current，不是 next
sampled_q = self.model(
    forward_type=ForwardType.SAC_Q,
    forward_inputs=repeated_inputs,
    actions=candidate_actions.flatten(0, 1),
).reshape(actions.shape[0], candidate_count, -1)
```

看起来像 bug，其实是 CQL 原论文的做法：保守项本质上是一个 importance-sampled 的 $\log\sum\exp$ 估计器，$\mu$ 的选择只要能覆盖到"可能被高估的动作"就行，从 $\pi(\cdot|s')$ 采出来的动作也是合法的候选。而它的好处是这些动作**恰好就是下一轮 TD target 会查询的那些**——虽然查询的状态不同，但动作分布相近，所以压它们对抑制 target 虚高有直接帮助。这和 LeRobot 的实现一致。

### 2.1 `_repeat_batch_tree` 与 `_conservative_q_inputs`

要在同一批状态上评估 12 个候选，得先把状态复制 12 份。观测是一个嵌套结构（图像字典、状态张量、文本列表），所以需要一个递归的复制函数：

```python
def _repeat_batch_tree(value, repeats):
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, dict):
        return {key: _repeat_batch_tree(item, repeats) for key, item in value.items()}
    if isinstance(value, list):
        return [item for item in value for _ in range(repeats)]
    if value is None:
        return None
    raise TypeError(f"Unsupported ConRFT replay field type: {type(value)}")
```

**注意用的是 `repeat_interleave` 而不是 `repeat`**。前者是 `[a,a,a,b,b,b]`，后者是 `[a,b,a,b,a,b]`。这必须和后面的 `reshape(batch_size, samples, ...)` 对应——`reshape` 假设的是前者的排列。用错会把不同样本的候选混在一起，而且**不会报错**，只会静默给出错误的保守项。这是这段代码里最脆的地方。

那个 `raise TypeError` 是有价值的防御：如果 replay 里将来多了一个不支持的字段类型（比如 numpy 数组），会立刻炸而不是静默漏掉。

复制之前先做了字段裁剪：

```python
_CONRFT_Q_AUX_INPUT_KEYS = frozenset({"chunk_sac_valid", "chunk_sac_bc_reference_action"})

def _conservative_q_inputs(data):
    required_keys = CHUNK_SAC_MODEL_INPUT_KEYS | _CONRFT_Q_AUX_INPUT_KEYS
    return {key: value for key, value in data.items() if key in required_keys}
```

只保留模型前向真正需要的键。理由是成本：`_repeat_batch_tree` 会把每个张量复制 12 份，replay 里那些和前向无关的字段（`chunk_sac_rewards`、`chunk_sac_mc_return`、各种统计标记）复制 12 份纯属浪费显存。

### 2.2 候选范围与 critic 的有效定义域

第 01 章 9.6 讨论过这个问题，结论现在可以精确表述。

`random_actions.uniform_(-1.0, 1.0)` 采的是 $[-1,1]$ 内的动作。而 critic 在收到任何动作之后第一件事是 clamp 到 $[-1,1]$（第 06 章 3.2 节）：

```python
bounded = actions.clamp(-1.0, 1.0)
```

所以 **critic 的有效定义域就是 $[-1,1]^{16\times62}$，均匀候选恰好覆盖了它的全部**。保守项的采样范围没有缺口。

（真正的问题在梯度侧——策略输出可以漂到 $[-1,1]$ 之外，clamp 会切断 Q 项的梯度。这靠 `straight_through_action_clip: true` 解决，见第 06 章。）

### 2.3 高维空间里 12 个候选够吗

诚实的回答是：**从"覆盖动作空间"的角度看远远不够，但保守项不需要覆盖。**

992 维空间里 4 个均匀随机样本的覆盖率可以忽略不计。但 CQL 的机制不是"遍历所有 OOD 动作"，而是"在每次更新时压低当前最容易被高估的那几个动作"。三路候选里有两路是**策略自己采的**，也就是说保守项始终在压"策略当前正在往那边走"的方向。这是一个自适应的、跟随策略移动的压制，不需要全空间覆盖。

均匀随机那一路的作用是补充：它提供一个"背景压制"，防止策略突然跳到一个从未被压过的区域。4 个样本在 992 维里几乎起不到这个作用——**这是 chunk 化对 ConRFT 最不利的影响，而且在原理上没有被解决**（第 02 章 2.2 节）。

实践上的应对是看指标：如果 `conrft/cql_diff` 长期接近 0，说明候选动作的 Q 值和数据动作的 Q 值区分不开，保守项没起作用；如果它很大且在涨，说明 critic 正在拉开数据内外的差距，保守项在工作。

## 3. `conservative_q_loss` 逐行推导

候选准备好之后，剩下的是纯张量运算。

```python
data_q = self.model(forward_type=ForwardType.SAC_Q, forward_inputs=current_inputs, actions=actions)
mc_returns = data.get("chunk_sac_mc_return")
if self.conrft.require_mc_returns and mc_returns is None:
    raise ValueError("ConRFT CalQL requires chunk_sac_mc_return in the replay batch")
return conservative_q_loss(
    data_q, sampled_q,
    temperature=self.conrft.cql_temperature,
    lower_bounds=mc_returns,
    clip_min=self.conrft.cql_clip_min,
    clip_max=self.conrft.cql_clip_max,
)
```

### 3.1 形状检查

函数开头三条检查，其中一条写得很紧凑：

```python
if data_q_values.ndim != 2:
    raise ValueError("data_q_values must have shape [batch, heads]")
if sampled_q_values.ndim != 3:
    raise ValueError("sampled_q_values must have shape [batch, samples, heads]")
if data_q_values.shape != sampled_q_values.shape[::2]:
    raise ValueError("Data and sampled Q values have incompatible shapes")
```

`sampled_q_values.shape` 是 `(B, 12, 2)`，`[::2]` 取步长 2 的切片得到 `(B, 2)`，正好是 `data_q_values.shape`。写法巧妙但可读性差——第一次读到会以为是笔误。

### 3.2 Cal-QL 的下界

```python
candidates = sampled_q_values
bound_rate = data_q_values.new_zeros(())
positive_bound_fraction = data_q_values.new_zeros(())
if lower_bounds is not None:
    if lower_bounds.numel() != data_q_values.shape[0]:
        raise ValueError("MC returns must contain one scalar per replay sample")
    bounds = lower_bounds.reshape(-1, 1, 1).to(sampled_q_values.dtype)
    bound_rate = (candidates < bounds).float().mean()
    positive_bound_fraction = (lower_bounds > 0).float().mean()
    candidates = torch.maximum(candidates, bounds)
```

`bounds` 的形状是 $[B,1,1]$，broadcast 到 $[B,12,2]$，所以每个样本的所有候选、所有 head 都用同一个下界 $G_t$。

**`torch.maximum` 就是 Cal-QL 相对 CQL 的全部改动。** 它的梯度效果：

| 情况 | `maximum` 选中 | 对 $\theta$ 的梯度 |
|------|----------------|--------------------|
| $Q_\theta(s,a_k) > G_t$ | $Q_\theta$ | 正常通过，这个候选**继续被压低** |
| $Q_\theta(s,a_k) \le G_t$ | 常数 $G_t$ | **恒为 0**，这个候选在保守项里不再受力 |

也就是"压到底线就松手"。完整论证见 [Cal-QL 前置知识 3.2 节](/前置知识/002h_前置知识_CalQL校准保守Q学习#3.2-cal-ql-的保守项)。

两个监控量：

- `cql_bound_rate`：被抬升的候选比例，也就是"有多少候选已经被压到底线以下"。理想曲线是训练初期接近 0、后期升到 $0.2\sim0.6$。
- `calql_positive_bound_fraction`：$G_t > 0$ 的样本比例，也就是来自成功 episode 的样本占比（第 07 章 6 节）。当前 run 是 1.0。

### 3.3 拼上数据动作，做归一化的 logsumexp

```python
candidates = torch.cat((candidates, data_q_values.unsqueeze(1)), dim=1)     # [B, 13, 2]
normalizer = temperature * math.log(candidates.shape[1])
ood_q = torch.logsumexp(candidates / temperature, dim=1) * temperature - normalizer
q_diff = ood_q - data_q_values
if clip_min is not None or clip_max is not None:
    q_diff = torch.clamp(q_diff, min=clip_min, max=clip_max)
return q_diff.mean(), {...}
```

**Step 1：这个公式在做什么**

$$
\mathcal{R}_{\text{Cal}} = \underbrace{\Big[\tau \log \sum_{k=1}^{13} \exp\big(\tilde{q}_k / \tau\big) - \tau \log 13\Big]}_{\text{ood\_q：归一化的软最大值}} - \underbrace{Q_\theta(s, a_{\mathcal{D}})}_{\text{数据动作的 Q}}
$$

其中 $\tilde{q}_k = \max\{Q_\theta(s, a_k),\, G_t\}$ 对 12 个候选，$\tilde{q}_{13} = Q_\theta(s, a_{\mathcal{D}})$（数据动作本身）。

**它算出"候选里最高的那个 Q 值，比数据动作的 Q 值高出多少"**，这个差值越小越好。

> **一句话直觉**：如果策略眼里有个动作比人类演示的动作看起来好得多，那大概是估高了，把它压下来。

**逐符号拆解**：

| 符号 | 数学含义 | 直觉 | 具体值 |
|------|----------|------|--------|
| $\tau$ | 温度 | 控制"软"的程度 | `cql_temperature: 1.0` |
| $K = 13$ | 候选总数（含数据动作） | 12 个 OOD + 1 个数据动作 | `candidates.shape[1]` |
| $\tilde q_k$ | 第 $k$ 个候选经下界抬升后的 Q | Cal-QL 处理过的候选值 | — |
| $\tau\log K$ | 归一化常数 | 抵消 logsumexp 相对 max 的系统性偏移 | $1.0 \times \log 13 = \mathbf{2.565}$ |
| $\texttt{ood\_q}$ | 归一化后的软最大值 | 近似 $\max_k \tilde q_k$ | $[B,2]$ |
| $Q_\theta(s,a_{\mathcal{D}})$ | 数据动作的 Q | 锚 | $[B,2]$ |
| `.mean()` | 对 batch 和 head 求平均 | 标量化 | — |

**为什么要减 $\tau\log K$**：logsumexp 有一个已知的夹逼关系（[CQL 前置知识 3.2 节](/前置知识/002g_前置知识_CQL保守Q学习#3.2-实用形式-用-logsumexp-代替算术平均)推过）：

$$
q_{\max} \;\le\; \tau\log\sum_k e^{q_k/\tau} \;\le\; q_{\max} + \tau\log K
$$

也就是它总比真正的最大值大 $0$ 到 $\tau\log K$ 之间。减掉 $\tau\log K$ 之后，`ood_q` 落在 $[q_\max - \tau\log K,\; q_\max]$，也就是**从上界修正成了下界**。

这个修正的实际意义：不减的话，即使全部 13 个候选的 Q 值完全相同（$\tilde q_k \equiv q$），也会得到 $\mathcal{R} = q + \tau\log 13 - q = 2.565 > 0$，保守项会凭空产生一个恒定的正 loss，持续压低所有 Q 值。减掉之后这种情况下 $\mathcal{R} = 0$，**保守项在"候选和数据动作没有区别"时正确地不施加任何力**。

**注意 $K = 13$ 而不是 12**：数据动作被拼进候选集了。这样做的效果是：如果数据动作恰好是所有候选里 Q 值最高的（说明 critic 已经正确地把数据动作排在最前），那么 $\texttt{ood\_q} \approx Q_\theta(s,a_{\mathcal{D}})$，$\mathcal{R} \approx 0$，保守项自动收手。这是一个漂亮的自终止性质。

**代入完整数字**。取本链路的实际量级：$Q$ 范围 $[0,1]$，$\tau = 1.0$，$G_t = 0.61$（对应 episode 中段的成功轨迹）。假设某个样本的 head 0 上：

| 候选 | $Q_\theta$ | 与 $G_t = 0.61$ 比 | $\tilde q_k$ | 是否受压低梯度 |
|------|-----------|---------------------|--------------|----------------|
| random 1 | $0.95$ | 高 | $0.95$ | 是（虚高，正是要压的） |
| random 2 | $0.30$ | 低 | $0.61$ | 否 |
| random 3 | $0.12$ | 低 | $0.61$ | 否 |
| random 4 | $0.48$ | 低 | $0.61$ | 否 |
| $\pi(s)$ 1..4 | $0.72, 0.68, 0.70, 0.66$ | 全高 | 同值 | 是 |
| $\pi(s')$ 1..4 | $0.55, 0.58, 0.52, 0.60$ | 全低 | $0.61 \times 4$ | 否 |
| 数据动作 | $0.64$ | — | $0.64$ | （它是锚，梯度方向相反） |

`bound_rate` 这一行的贡献：12 个候选里 7 个被抬升 → 这个样本贡献 $7/12 = 0.583$。

算 logsumexp（把最大值 $0.95$ 提出来）：

$$
\begin{aligned}
\sum_k e^{\tilde q_k - 0.95} &= e^{0} + 3e^{0.61-0.95} + e^{0.72-0.95} + e^{0.68-0.95} + e^{0.70-0.95} + e^{0.66-0.95} + 4e^{0.61-0.95} + e^{0.64-0.95}\\
&= 1 + 3(0.712) + 0.795 + 0.763 + 0.779 + 0.748 + 4(0.712) + 0.733\\
&= 1 + 2.136 + 3.085 + 2.848 + 0.733 = 9.802
\end{aligned}
$$

$$
\tau\log\sum_k e^{\tilde q_k/\tau} = 0.95 + \log(9.802) = 0.95 + 2.283 = 3.233
$$

$$
\texttt{ood\_q} = 3.233 - 2.565 = 0.668
$$

$$
\mathcal{R}_{\text{Cal}} = 0.668 - 0.64 = \mathbf{0.028}
$$

乘上 $\alpha = 0.1$：对 critic loss 的贡献是 $0.0028$。

**和 TD 项比一下量级**：第 06 章算过 TD target 在 $[0,1]$，所以 TD loss $= (Q - y)^2$ 的典型量级是 $10^{-3} \sim 10^{-2}$。实测的 `critic loss = 0.0718`。所以 $0.0028$ 的 CQL 贡献约占总 loss 的 4%。

**这个比例说明什么**：`cql_alpha: 0.1` 在当前 $Q \in [0,1]$ 的尺度下是一个**偏弱**的保守强度。如果观察到 `conrft/cql_diff` 长期很小（比如 $< 0.01$）而 Q 值又在往不合理的方向跑，可以考虑调大 $\alpha$。反过来如果 `cql_bound_rate` 很快冲到接近 1，说明压得太狠，要调小。

**为什么 `cql_clip_min/max` 都是 `null`**：它们是给 `q_diff` 加硬截断的开关，用来防止保守项在某个 batch 上给出极端梯度。当前不启用，因为 $Q \in [0,1]$ 的尺度下 `q_diff` 本身不会太大。如果换成范围更大的 reward，可能需要打开。

## 4. `cql_scale` 那个分支

第 01 章 9.12 提过这段代码容易误读，这里完整推导一次为什么它是对的。

```python
cql_scale = (
    1.0 / self.gradient_accumulation
    if self.critic_objective is not None and self.critic_objective.uses_distributed_normalizers
    else 1.0
)
output.loss = output.loss + self.conrft.cql_alpha * cql_scale * cql_loss
```

关键是搞清楚**调用方对 `output.loss` 做了什么**。在 `update_one_epoch` 里：

```python
critic_output = self.forward_critic(batch["forward_inputs"], ...)
critic_loss = critic_output.loss
if not self.critic_objective.uses_distributed_normalizers:
    critic_loss = critic_loss / self.gradient_accumulation
critic_loss.backward()
```

分两种情况：

**情况 A：`uses_distributed_normalizers = False`**（本链路，`AbsoluteCriticObjective` 继承基类的默认值）

- `cql_scale = 1.0`，所以 `output.loss = td_loss + 0.1 * cql_loss`
- 调用方把**整体**除以 `gradient_accumulation`：`(td_loss + 0.1*cql_loss) / G`
- 两项被同一个因子缩放，**相对权重保持 $1 : 0.1$**。正确。

**情况 B：`uses_distributed_normalizers = True`**（`TemporalBCRelativeCriticObjective`）

- 这种 critic 的 loss 已经在内部除过全局归一化因子（`batch_normalizers` 返回的跨 rank 计数），所以调用方**不再**除 `gradient_accumulation`
- 但 CQL 项是 ConRFT 自己算的 `q_diff.mean()`，它没有做那个归一化，所以需要自己补上 `1/G`
- `cql_scale = 1/G` 正是在补这一步

**结论：这段代码在两种情况下都给出一致的相对权重，是正确的。** 我第一次读时误判成 bug——因为只看到 `cql_scale = 1.0` 就以为 CQL 项没被除，没有往下看调用方。

**但它仍然是一个脆弱点**：正确性依赖"`forward_critic` 的返回值会被调用方按 `uses_distributed_normalizers` 决定是否除 $G$"这个**隐式契约**。如果哪天有人改了 `update_one_epoch` 的缩放逻辑，这里会静默失效。ConRFT 强制 `flat_absolute`（第 04 章）把情况 B 挡在门外，所以目前只有情况 A 在跑，而情况 B 从来没被验证过。现在有测试锁定 accumulation 缩放，算是补了一层保护。

## 5. 计算成本

Cal-QL 让阶段一慢了多少？按 VLM backbone 前向次数算（这是最贵的部分）。

一次 critic 更新的一个 micro-batch，`_prepare_chunk_sac_features` 的调用次数：

| 来源 | 次数 | batch 倍数 | 说明 |
|------|------|-----------|------|
| `_td_target` 的 `ForwardType.SAC` | 1 | $1\times$ | 在 $s'$ 上采动作 |
| `_td_target` 的 `ForwardType.SAC_Q` | 1 | $1\times$ | target critic |
| `prediction` 的 `ForwardType.SAC_Q` | 1 | $1\times$ | 数据动作的 Q |
| **CQL：当前策略采样** | 1 | $4\times$ | `_repeat_batch_tree(current_inputs, 4)` |
| **CQL：next 策略采样** | 1 | $4\times$ | `_repeat_batch_tree(next_inputs, 4)` |
| **CQL：12 个候选的 Q** | 1 | $12\times$ | `_repeat_batch_tree(current_inputs, 12)` |
| **CQL：数据动作的 Q** | 1 | $1\times$ | 复用 `current_inputs` |

阶段二（无 CQL）是 3 次前向、有效 batch 倍数 3。阶段一是 7 次前向、有效 batch 倍数 $3 + 4 + 4 + 12 + 1 = 24$。

**也就是阶段一每个 micro-batch 的 VLM 计算量约是阶段二的 8 倍。**

再加上 actor 那一次（第 09 章）：阶段一 actor 用 `return_bc_reference=False`，一次 4 步去噪，倍数 1。

**为什么 `micro_batch_size` 只能是 2**：CQL 那一路要把 batch 复制 12 份，$2 \times 12 = 24$ 个样本同时过 VLM。如果 `micro_batch_size = 8`，就是 96 个样本——H20 的显存放不下（每个样本 3 张图 + 2048 维 × 序列长度的激活）。

**优化空间**：三路候选的策略采样都用同一批 `vl_embs`（只是动作不同），理论上 `_prepare_chunk_sac_features` 只需要跑**一次**，然后把 `vl_embs` 和 `state_features` 复制 12 份直接喂给 critic head。当前实现每次都重跑完整前向，浪费了大约 $(4+4+12+1-3)/24 = 75\%$ 的 VLM 计算。这是阶段一最大的一块可优化空间，但需要改 `sac_q_forward` 的接口（让它能接受预先算好的特征）。

**实际耗时**：阶段一 800 次更新，每次 2 个 micro-batch（`gradient_accumulation = 32/2/8 = 2`），在 8 卡 H20 上按 handoff 记录是小时级。

## 6. 在线阶段为什么关掉保守项

```python
if not self._conrft_offline_stage:
    return output
```

三条理由，按重要性排：

1. **不需要**。在线阶段新数据持续进入 replay，OOD 区域的 Q 值会被真实的 TD 信号修正。保守项要防的"错误永不被纠正"的循环在在线阶段不存在。
2. **有害**。保守项会持续压低 Q 值，而在线阶段恰恰需要 Q 值**爬升**到真实尺度才能给出有意义的策略梯度。这正是 Cal-QL 要解决的问题——如果在线还继续压，等于自己制造那个问题。
3. **太贵**。第 5 节算过，保守项让 VLM 计算量涨 8 倍。在线阶段本来就要和环境采集抢 GPU，加不起。

这和 ConRFT 论文的设定一致（阶段一 Cal-ConRFT 有保守项，阶段二 HIL-ConRFT 没有）。

**一个副作用**：阶段二的配置里 `algorithm.conrft.require_mc_returns: true` 是死配置（第 04 章 6 节）。在线阶段 `_bc_mc_return_gamma()` 返回 `None`、`_bc_replay_extra_fields()` 返回空字典、`_compute_conservative_loss` 根本不被调用，所以这个键纯属装饰。留着它是为了两个阶段的 `algorithm.conrft` 块保持字面一致，方便对照。

## 7. 小结

| 主题 | 关键结论 |
|------|----------|
| 挂载方式 | `forward_critic` 里把 CQL 项直接加到 `output.loss`，共用一次 `backward()` |
| 候选总数 | **12**（不是 4）：4 随机 + 4 当前策略 + 4 next 策略 |
| next 策略候选 | 在 $s'$ 上采动作、在 $s$ 上评估 Q。符合 CQL 原论文，作用是提前抑制 target 虚高 |
| 复制方式 | `repeat_interleave`（不是 `repeat`），必须和 `reshape(B, samples, ...)` 对应 |
| 候选范围 | `uniform(-1,1)` 恰好覆盖 critic 的全部有效定义域（因为 critic 内部会 clamp） |
| 高维覆盖率 | 4 个均匀样本在 992 维里可忽略；保守项靠"跟随策略"的两路自适应候选起作用 |
| 归一化常数 | $\tau\log 13 = \mathbf{2.565}$。作用是让"候选与数据动作无差别"时保守项为 0 |
| 为什么 $K=13$ | 数据动作被拼进候选，带来自终止性质 |
| Cal-QL 下界 | `torch.maximum(candidates, G_t)`，跌破下界的候选梯度归零 |
| 数值示例 | $G_t=0.61$ 的样本上 $\mathcal{R}_{\text{Cal}} = 0.028$，乘 $\alpha=0.1$ 后占 critic loss 约 4% |
| `cql_scale` | 当前恒为 1，且正确（调用方统一除 $G$）。情况 B 从未被验证，靠 `flat_absolute` 断言挡住 |
| 计算成本 | 阶段一 VLM 有效 batch 倍数 24 vs 阶段二 3，约 8 倍；可优化空间约 75% |
| 在线关闭 | 不需要 + 有害（阻碍 Q 爬升）+ 太贵 |

## 下章预告

Critic 讲完了，第 09 章讲阶段一的 actor。核心问题是：为什么阶段一的 actor 训在 expert batch 上而不是普通 replay 上（`trains_on_expert_batch` 的机制）、`in_bc_warmup` 语义被 ConRFT 覆写成了什么、BC 项和 Q 项如何在同一次前向里复用同一份采样、以及 W2 项在离线阶段为什么是零张量而不是"算出来恰好为 0"。

→ [第 09 章 阶段一 Actor：BC 加轻量 Q](./09_阶段一Actor目标)
