---
title: "模型层：三个 ForwardType"
series:
  id: groot_conrft_rlinf_deep_dive
  chapter: 5
order: 5
---

# 第 05 章 模型层：三个 ForwardType

> 前情提要：第 04 章讲完了配置怎么合并、哪些断言在保护什么。这一章进模型层，讲 ConRFT 用到的三个模型入口 `ForwardType.SAC` / `SAC_Q` / `SFT` 分别做什么。重点是 `sample_chunk_sac_action` 这个 130 行的函数——它把 4 步 Flow Matching 去噪变成了一个可微的随机策略，并且能在同一次调用里跑出"带 gate"和"不带 gate"两条轨迹。W2 项的全部机制都在这里。

## 知识链接

- 上一章：[配置继承链与硬门禁](./04_配置继承链)
- 下一章：[Critic 架构与 chunk 级 TD target](./06_Critic架构与TD目标)
- [系列目录](./index)
- 前置：[Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流)
- 前置：[随机微分方程 SDE 与扩散模型的联系](/前置知识/001c_前置知识_随机微分方程SDE直觉与扩散模型的联系)
- 前置：[重参数化技巧](/前置知识/002e_前置知识_重参数化技巧)
- 前置：[概率密度函数与高斯分布](/前置知识/002b_前置知识_概率密度函数与高斯分布)
- 相关：[GR00T N1.7 深度解析](/系列/groot_n1d7_deep_dive/) — 基座模型本身
- 相关：[第 01 章 9.7 W2 用了第二次独立采样](./01_全链路总览#9.-问题-风险与实验安排隐患清单)

---

## 1. 三个入口的分工

RLinf 用一个 `forward_type` 枚举把不同的训练需求路由到不同的模型方法。这个设计的目的是让 worker 侧的代码不需要知道模型内部结构——它只说"我要采个动作"或"我要算个 Q 值"。

`gr00t_action_model.py` 里的分发很简单：

```python
def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
    if forward_type == ForwardType.DEFAULT:
        return self.default_forward(**kwargs)
    elif forward_type == ForwardType.AQC:
        return self.aqc_forward(**kwargs)
    elif forward_type == ForwardType.SFT:
        return self.sft_forward(**kwargs)
    elif forward_type == ForwardType.SAC:
        return self.sac_forward(**kwargs)
    elif forward_type == ForwardType.SAC_Q:
        return self.sac_q_forward(**kwargs)
    else:
        raise NotImplementedError
```

ConRFT 只用其中三个：

| ForwardType | 调用者 | 输入 | 输出 | 用途 |
|-------------|--------|------|------|------|
| `SAC` | `ConRFTActorObjective.compute`、critic 的 TD target、Cal-QL 候选采样 | 观测 | `actions`、`log_pi`、可选 `bc_reference_actions` | 采一个**可微的**动作 chunk |
| `SAC_Q` | 同上 | 观测 + 动作 | `[B, 2]` twin-Q | 给一个动作打分 |
| `SFT` | `ConRFTActorObjective.compute` 的 BC 项 | 观测 + 目标动作 + mask | 标量 loss | Flow Matching 的监督学习 loss |

三者的一个共同前置步骤是 `_prepare_chunk_sac_features`，它负责跑 VLM backbone：

```python
def _prepare_chunk_sac_features(self, forward_inputs):
    normalized_input = _normalize_gr00t_forward_inputs(forward_inputs)
    normalized_input = _canonicalize_gr00t_text_forward_inputs(normalized_input, getattr(self, "padding_value", 0))
    backbone_inputs, action_inputs = self.prepare_input(normalized_input)
    with torch.no_grad():                                    # ← backbone 全程不训练
        backbone_outputs = self.backbone(backbone_inputs)
    ...
    state_features = self.action_head._encode_state_features(action_inputs, embodiment_id)
    return backbone_outputs, action_inputs, vl_embs, state_features
```

**`with torch.no_grad()` 这一行很关键**：Cosmos-Reason2-2B backbone **在整条 RL 链路里完全冻结**，不接收任何梯度。可训练的部分只有：

1. Flow-G adapter（约 $2 \times 62 \times 256 + \ldots \approx 3.4$ 万参数）
2. `ChunkSACCritic`（两个 3 层 MLP，输入 $512 + 992$ 维）
3. action head 的部分组件（取决于 FSDP wrap policy，配置里 `module_classes_to_wrap: [ChunkSACCritic, FlowGVelocityAdapter]`）

也就是说这条链路的**可训练参数量非常小**，绝大部分显存花在 backbone 的前向激活上。这解释了为什么 `micro_batch_size` 只有 2——瓶颈是 VLM 前向，不是训练。

**注意 `padding_value` 这个参数**：第 04 章讲过 ConRFT 强制 `padding_value=0`。它在这里被用来做文本输入的 canonicalize。这是它进硬断言的原因之一。

## 2. `ForwardType.SAC`：把去噪链变成随机策略

这是整个模型层最核心的部分。先说清楚要解决什么问题，再看实现。

### 2.1 问题：Flow Matching 是确定性的，SAC 需要随机策略

GR00T 的推理是确定性的 Euler 积分：给定初始噪声 $x_0$，4 步之后得到唯一的动作。用公式写就是

$$
x_{i+1} = x_i + \frac{1}{4} v_\theta(x_i, t_i, s), \qquad i = 0,1,2,3
$$

**Step 1：这个公式在做什么**

**它描述 GR00T 原生的推理过程：从高斯噪声出发，用速度场 $v_\theta$ 迭代 4 次，把噪声"流"成一个合法的动作 chunk。**

> **一句话直觉**：像用 4 步把一团随机噪声捏成一段有意义的机器人轨迹。

**逐符号拆解**：

| 符号 | 数学含义 | 直觉 | 具体是什么 |
|------|----------|------|-----------|
| $x_i$ | 第 $i$ 步的中间状态，形状 $[B, 16, 132]$ | "还没捏完的动作" | 代码里的 `x_t`，注意最后一维是 `model_action_dim=132`，只有前 62 维会被用 |
| $v_\theta$ | 速度场网络 | "往哪个方向捏、捏多快" | 32 层 AlternateVLDiT + action_decoder |
| $t_i = i/4$ | 时间坐标 | 从 0（纯噪声）走到 1（干净动作） | $\{0, 0.25, 0.5, 0.75\}$ |
| $s$ | 状态条件 | 图像 + 机器人状态 + 语言指令 | `vl_embs` + `state_features` |
| $1/4$ | Euler 步长 | 4 步走完 $[0,1]$ | `denoising_steps: 4` |

**代入数字**：$x_0 \sim \mathcal{N}(0,I)$，某个坐标 $x_0 = 0.8$。假设四步的速度分别是 $v = \{-0.4, -0.3, -0.2, -0.1\}$：

$$
x_1 = 0.8 + 0.25\times(-0.4) = 0.70,\quad x_2 = 0.625,\quad x_3 = 0.575,\quad x_4 = 0.550
$$

最终动作这个坐标是 $0.55$。

**问题在哪**：给定 $x_0$，输出是**确定的**。SAC 需要一个有随机性的策略 $\pi(a|s)$，而且需要能算 $\log \pi(a|s)$（用于熵项）。确定性策略的密度是 delta 函数，$\log \pi$ 无定义。

有两条常见解法。一是像 DDPG/TD3 那样干脆用确定性策略 + 外部探索噪声，但那样就没有熵正则了。二是**把去噪过程本身变成一个随机过程**——每一步不再是确定的 Euler 更新，而是从一个高斯分布里采样。本链路走的是第二条。

### 2.2 解法：flow SDE，每一步都是一次高斯采样

`sample_mean_var_val` 返回的不是下一个状态，而是下一个状态的**均值和标准差**：

$$
x_{i+1} \sim \mathcal{N}\big(\mu_i,\ \sigma_i^2 I\big), \qquad
\mu_i = w_i^{(0)} \cdot \hat{x}_0 + w_i^{(1)} \cdot \hat{x}_1
$$

**Step 1：这个公式在做什么**

**它把"确定的一步 Euler 更新"替换成"从一个高斯分布里采一个样"**，均值是 Euler 更新的结果（用两个端点预测的加权），标准差由 SDE 的扩散系数给出。

> **一句话直觉**：还是往同一个方向捏，但每一步手都抖一下，抖的幅度由 SDE 公式规定。

**逐符号拆解**：先要理解 $\hat{x}_0$ 和 $\hat{x}_1$ 这两个"端点预测"。Flow Matching 训练时人为定义了一条从噪声 $x_0$ 到动作 $x_1$ 的直线路径 $x_t=(1-t)x_0+tx_1$，网络学的是这条路径上处处相同的速度 $v=x_1-x_0$（推导见 [Flow Matching 与连续归一化流 §3.1](/前置知识/000g_前置知识_Flow_Matching与连续归一化流#3.1-条件-flow-conditional-flow)）。推理时反过来，把这两个式子当成关于 $x_0,x_1$ 的方程组，代入消元即可从当前 $(x_t, t, v)$ 反解出两个端点：

```python
x0_pred = x_t - v_t * t_input        # 反推纯噪声端
x1_pred = x_t + v_t * (1 - t_input)  # 外推干净动作端
```

| 符号 | 含义 | 代码 |
|------|------|------|
| $\hat{x}_0$ | 对纯噪声端点的预测 | `x0_pred = x_t - v_t * t_input` |
| $\hat{x}_1$ | 对干净动作端点的预测 | `x1_pred = x_t + v_t * (1 - t_input)` |
| $t$ | 当前时间 | `t_input = timesteps[idx]` |
| $\Delta$ | 步长 | `delta = timesteps[idx+1] - timesteps[idx]`，这里是 $0.25$ |
| $\sigma_i$ | 第 $i$ 步的扩散系数 | 见下面的公式 |
| $w_i^{(0)}, w_i^{(1)}$ | 两个端点的权重 | 见下面的公式 |

确定性情况（`mode == "eval"`）很简单——就是标准的插值：

```python
x0_weight = 1 - (t_input + delta)
x1_weight = t_input + delta
x_t_std = torch.zeros_like(t_input)
```

即 $\mu_i = (1 - t - \Delta)\hat{x}_0 + (t+\Delta)\hat{x}_1$，正好是把插值公式的时间推进到 $t + \Delta$。标准差为 0，退化成确定性 Euler。

随机情况（`mode == "train"` 且 `noise_method == "flow_sde"`）多了一个修正项：

```python
sigmas = noise_level * torch.sqrt((1 - timesteps) / torch.where(timesteps == 0, timesteps[1], timesteps))[:-1]
sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
x0_weight = torch.ones_like(t_input) - (t_input + delta) - sigma_i**2 * delta / (2 * (1 - t_input))
x1_weight = t_input + delta
x_t_std = torch.sqrt(delta) * sigma_i
```

$$
\sigma_i = \eta \sqrt{\frac{1 - t_i}{t_i}}, \qquad
w_i^{(0)} = 1 - (t_i + \Delta) - \frac{\sigma_i^2 \Delta}{2(1-t_i)}, \qquad
w_i^{(1)} = t_i + \Delta, \qquad
\sigma_i^{\text{step}} = \sqrt{\Delta}\,\sigma_i
$$

**Step 1（针对这三个式子）：它们在做什么**

**第一个式子给出每一步注入多大的噪声（时间越早噪声越大）；第二、三个式子对均值做一个补偿，使得加噪之后的边际分布仍然近似正确；第四个式子是这一步实际的采样标准差。**

> **一句话直觉**：早期敢多抖（反正后面还能修回来），晚期少抖（要定稿了）；抖了之后均值往回压一点作为补偿。

**逐符号拆解**：

| 符号 | 含义 | 直觉 | 典型值 |
|------|------|------|--------|
| $\eta$ | `noise_level`，全局噪声强度 | 探索幅度的总旋钮 | `rl_config.noise_level` 默认 $0.5$ |
| $\sqrt{(1-t_i)/t_i}$ | 时间相关的衰减因子 | $t\to 0$ 时趋于 $\infty$，$t\to1$ 时趋于 0 | 见下方数值 |
| $\Delta$ | 步长 | $1/4$ | $0.25$ |
| $-\sigma_i^2\Delta / (2(1-t_i))$ | 均值补偿项 | 加了噪声就把均值往 $\hat{x}_0$ 的反方向推一点 | 见下方数值 |
| $\sqrt{\Delta}\sigma_i$ | 单步采样标准差 | 布朗运动的 $\sqrt{\Delta t}$ 缩放 | 见下方数值 |

**代入数字**（$\eta = 0.5$，$\Delta = 0.25$，4 步的时间格点 $\{0, 0.25, 0.5, 0.75\}$）：

$t_0 = 0$ 会让分母为 0，代码里用 `torch.where(timesteps == 0, timesteps[1], timesteps)` 把它替换成 $t_1 = 0.25$。所以：

| $i$ | $t_i$ | 分母用的 $t$ | $\sqrt{(1-t)/t}$ | $\sigma_i$ | $\sigma_i^{\text{step}} = \sqrt{0.25}\sigma_i$ | 均值补偿 $-\sigma_i^2\Delta/(2(1-t_i))$ |
|-----|-------|--------------|------------------|-----------|--------------|----------------------------------------|
| 0 | 0.00 | 0.25 | $\sqrt{3} = 1.732$ | 0.866 | 0.433 | $-0.750\times0.25/2 = -0.0938$ |
| 1 | 0.25 | 0.25 | $\sqrt{3} = 1.732$ | 0.866 | 0.433 | $-0.750\times0.25/(2\times0.75) = -0.125$ |
| 2 | 0.50 | 0.50 | $1.000$ | 0.500 | 0.250 | $-0.250\times0.25/(2\times0.5) = -0.0625$ |
| 3 | 0.75 | 0.75 | $0.577$ | 0.289 | 0.144 | $-0.0835\times0.25/(2\times0.25) = -0.0418$ |

**读这张表**：探索噪声从 $0.433$ 递减到 $0.144$，也就是**去噪早期探索得多、后期收敛**。这和扩散模型的直觉一致——早期决定"大方向"，此时多样性有价值；后期是精修，抖动只会破坏动作质量。

**为什么是这个形式**：这是 flow matching 对应的 probability-flow ODE 加上一个扩散项之后的离散化。$\sqrt{(1-t)/t}$ 这个形状来自 Flow Matching 的条件概率路径的方差；均值补偿项来自 Fokker-Planck 方程配平（加了扩散就要在漂移项上补偿，才能保持边际分布不变）。完整推导见 [SDE 前置知识](/前置知识/001c_前置知识_随机微分方程SDE直觉与扩散模型的联系)。

**采样怎么做**（重参数化，保证可微）：

```python
transition_noise = self.sample_noise(x_t.shape, x_t.device, dtype=x_t.dtype)
x_t = mean + transition_noise * std
```

这就是标准的 [重参数化技巧](/前置知识/002e_前置知识_重参数化技巧)：把随机性挪到一个与参数无关的 $\epsilon \sim \mathcal{N}(0,I)$ 上，$x_{i+1} = \mu_i(\theta) + \sigma_i \epsilon$ 对 $\theta$ 可微。**这是 ConRFT 的 Q 项梯度能传回策略的根本原因。**

### 2.3 `log_pi` 是什么

有了每一步的高斯分布，整条去噪路径的联合对数密度就能算出来：

$$
\log \pi^{\text{path}}(x_{0:4} \mid s) = \underbrace{\log \mathcal{N}(x_0; 0, I)}_{\text{初始噪声}} + \sum_{i=0}^{3} \underbrace{\log \mathcal{N}\big(x_{i+1}; \mu_i, \sigma_i^2 I\big)}_{\text{每一步转移}}
$$

**Step 1：这个公式在做什么**

**它算出"这条完整的去噪轨迹"的对数概率**——注意是**轨迹**的概率，不是最终动作的边际概率。

> **一句话直觉**：把"抽到这个初始噪声"和"每一步抖成这样"的概率全部乘起来（对数就是加起来）。

**逐符号拆解**：

| 项 | 含义 | 代码 |
|----|------|------|
| $\log\mathcal{N}(x_0;0,I)$ | 初始噪声的对数密度 | `base_log_prob = -0.5 * (x_t.square() + torch.log(2*pi*ones))` |
| $\log\mathcal{N}(x_{i+1};\mu_i,\sigma_i^2 I)$ | 第 $i$ 步转移的对数密度 | `step_log_probs.append(self.get_logprob_norm(x_t, mean, std))` |
| 求和 | 联合密度 | `aggregate_sde_path_log_prob` 里的 `base + transitions` |

**关键点：这不是 $\log\pi(a|s)$。** 真正的动作边际密度需要对所有能产生同一个 $a$ 的轨迹积分，那是算不出来的。用路径密度代替边际密度是一个**代理**（surrogate）。这个代理有一个已知的性质：它是边际密度的一个下界相关量（Jensen 不等式），所以最大化路径熵会推动边际熵增大，但两者不相等。

代码里提供了两种口径，由 `path_density` 配置选择：

```python
def select_sde_objective_log_prob(normalized_log_prob, path_log_prob, path_density):
    if path_density == "sde_path":
        return path_log_prob                # 原始路径 log 概率
    if path_density == "normalized_sde_surrogate":
        return normalized_log_prob          # 除以坐标总数
```

归一化版本的除数是：

```python
coordinate_count = (len(transition_log_probs) + 1) * chunk_length * action_dim
return path_log_prob / coordinate_count, path_log_prob
```

代入本链路的数字：$(4+1) \times 16 \times 62 = 4960$。所以 `normalized_log_pi` 是 `path_log_pi` 除以 $4960$——它是"每个坐标平均的 log 概率"，量级上和单个高斯的 log 密度可比，不会随 chunk 长度和动作维度膨胀。

**对 ConRFT 来说，这一整套是白算的。** 配置里 `entropy_tuning.alpha_type: fixed_alpha` + `initial_alpha: 0.0`，熵系数恒为 0：

- TD target 里的 $-\alpha \log\pi$ 项恒为 0
- `ConRFTActorObjective` 完全不使用 `log_pi`（它只把 `log_pi` 放进 `ActorObjectiveOutput` 做日志）

这曾经是白算的：`compute_path_log_prob: true` 从 Flow-G 层继承下来，所以 4 步的 `get_logprob_norm` 和最后的 `aggregate_sde_path_log_prob` 照样跑（张量是 $[B, 16, 132]$ 的，4 步就是 4 份）。

**现在两个阶段都显式设了 `compute_path_log_prob: false`**（第 01 章 9.12）。`config.py` 里那条 `sac_flow_g requires compute_path_log_prob=true` 的断言对 `conrft_actor` 做了豁免：

```python
if not compute_path_log_prob and not conrft_actor:
    raise ValueError("sac_flow_g requires compute_path_log_prob=true")
```

关掉之后 `step_log_probs` 是空列表，走 else 分支：

```python
else:
    path_log_prob = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    normalized_log_prob = path_log_prob
```

**所以在当前配置下 `policy["log_pi"]` 恒为 0。** 记住这一点——`ActorObjectiveOutput` 里的 `log_pi` / `path_log_pi` 字段仍然会被填充并进日志，但它们没有信息量，任何基于 `log_pi` 的诊断都不能用（第 13 章会标出这些"失效指标"）。

### 2.4 梯度回传几步：`actor_backprop_steps`

4 步去噪意味着梯度要穿过 4 层"32 层 DiT"。这个链条很长，显存和计算都吃紧。代码提供了一个开关：

```python
actor_backprop_steps = int(self.chunk_sac_config.get("actor_backprop_steps", self.num_inference_timesteps))
if not 1 <= actor_backprop_steps <= self.num_inference_timesteps:
    raise ValueError(...)
...
track_step_gradient = (
    outer_grad_enabled
    and mode == "train"
    and idx >= self.num_inference_timesteps - actor_backprop_steps
)
with torch.set_grad_enabled(track_step_gradient):
    mean, std, flow_g_gate = self.sample_mean_var_val(...)
    ...
if not track_step_gradient:
    x_t = x_t.detach()
```

**逻辑**：只有**最后** `actor_backprop_steps` 步跟踪梯度，前面的步骤在 `no_grad` 下跑并 `detach`。

本链路配置 `actor_backprop_steps: 4`（继承层 3），等于 `num_inference_timesteps`，所以**全部 4 步都回传**。

代入判断式：$\text{idx} \ge 4 - 4 = 0$，所有 $\text{idx} \in \{0,1,2,3\}$ 都满足。

**如果改成 2 会怎样**：$\text{idx} \ge 2$，只有第 2、3 步回传。显存和计算减半，但策略只能通过"最后两步的速度场"来改变输出，表达能力受限。对 Flow-G 这个"只训一个乘性 gate"的架构，gate 在每一步都参与，所以砍掉前两步等于让前两步的 gate 收不到梯度——不推荐。

## 3. Flow-G adapter：只训一个乘性门

ConRFT 的 actor 训的不是 GR00T 的速度场，而是一个包在它外面的门。

### 3.1 结构

先说设计动机。GR00T 的速度场是在大规模数据上训出来的，直接用 RL 梯度去改它有两个问题：一是参数量太大（32 层 DiT），RL 的稀疏信号撑不起这么多参数；二是容易灾难性遗忘，几十次更新就可能把预训练能力破坏掉。Flow-G 的做法是**冻结速度场，只学一个逐坐标的乘性缩放**：

$$
\tilde{v}(x_t, t, s) = g_\phi(v, x_t, t) \odot v(x_t, t, s)
$$

其中门 $g_\phi$ 是一个两层 MLP：

```python
self.input_dim = self.action_dim * 2 + 1        # velocity(62) + action_state(62) + time(1) = 125
self.net = nn.Sequential(
    nn.Linear(self.input_dim, hidden_dim),      # 125 → 256
    nn.SiLU(),
    nn.Linear(hidden_dim, self.action_dim),     # 256 → 62
)
```

$$
g_\phi(v, x_t, t) = \kappa \cdot \sigma\big(f_\phi([v; x_t; t])\big)
$$

**Step 1：这个公式在做什么**

**它为速度场的每一个坐标产出一个介于 $0$ 和 $\kappa$ 之间的缩放因子**——大于 1 表示"这个方向再走远一点"，小于 1 表示"这个方向少走一点"。

> **一句话直觉**：预训练模型说"往这边走这么多"，gate 说"听你的方向，但幅度我调一下"。

**逐符号拆解**：

| 符号 | 含义 | 直觉 | 配置值 |
|------|------|------|--------|
| $v$ | 冻结速度场的输出，$[B,16,62]$ | 预训练模型的建议 | — |
| $x_t$ | 当前去噪状态 | "现在捏到什么样了" | — |
| $t$ | 时间标量，广播到每个 token | "捏到第几步了" | $\{0, 0.25, 0.5, 0.75\}$ |
| $f_\phi$ | 两层 MLP | 学习"什么情况下该调多少" | 125→256→62，SiLU |
| $\sigma$ | sigmoid | 把输出压到 $(0,1)$ | — |
| $\kappa$ | `gate_scale` | 门的上限 | **2.0** |
| $\odot$ | 逐元素乘 | 每个动作维度独立缩放 | — |

**门的取值范围**：$g \in (0, 2)$。也就是说 RL 最多能把速度放大到 2 倍，或者缩到 0（完全不动）。**它不能反转方向**——这是一个刻意的安全约束：策略只能调整"走多远"，不能调整"往哪走"。

**恒等初始化**：

```python
@torch.no_grad()
def reset_identity(self) -> None:
    self.net[2].weight.zero_()
    self.net[2].bias.fill_(self.gate_bias)
```

最后一层权重清零、偏置设为 `gate_bias = 0.0`。于是无论输入是什么，$f_\phi$ 的输出恒为 0，$g = 2.0 \times \sigma(0) = 2.0 \times 0.5 = 1.0$。

**代入验证**：$\kappa \cdot \sigma(0) = 2.0 \times 0.5 = 1.0$，所以 $\tilde v = 1.0 \odot v = v$，adapter 是恒等映射，模型行为和原始 GR00T 完全一致。

**为什么 `gate_scale = 2.0` 而不是 1.0**：如果 $\kappa = 1$，那么 $g \in (0,1)$，门只能**缩小**速度，永远不能放大。恒等点会落在 sigmoid 的饱和端（需要 $\sigma \to 1$，即 $f_\phi \to +\infty$），初始化不可能做到恒等。$\kappa = 2$ 让恒等点恰好落在 sigmoid 的中心（梯度最大处），既能双向调整，又有最好的初始梯度。

**这就是第 01 章 9.4 那条风险的核心**：`reset_identity()` 会把训练好的 gate 打回 $g \equiv 1$。对"阶段一没训 actor"的 Flow-G 链路无害，对"阶段一训了 800 次 actor"的 ConRFT 是灾难。

### 3.2 门在哪里被应用

在 `sample_mean_var_val` 里，紧接在 action_decoder 之后、Euler 更新之前：

```python
v_t = action_decoder(model_output, embodiment_id) if action_decoder is not None else torch.zeros_like(model_output)
flow_g_config = self.chunk_sac_config.get("flow_g", {})
flow_g_gate = None
if flow_g_config.get("enabled", False) and apply_flow_g:
    if return_flow_g_gate:
        v_t, flow_g_gate = self.flow_g_adapter(v_t, x_t, t_cont, return_gate=True)
    else:
        v_t = self.flow_g_adapter(v_t, x_t, t_cont)
```

注意 `apply_flow_g` 这个参数——它是下一节双轨去噪的开关。

`return_flow_g_gate=True` 时会额外返回 gate 张量，`sample_chunk_sac_action` 用它算两个监控指标：

```python
if flow_g_gate is not None:
    flow_g_gate_means.append(flow_g_gate.float().mean(dim=(1, 2)))
    flow_g_gate_deviations.append((flow_g_gate.float() - 1.0).abs().mean(dim=(1, 2)))
```

- `flow_g_gate_mean`：gate 的平均值。恒等初始化时是 $1.0$，训练后偏离 1 的程度反映 RL 改变了多少。
- `flow_g_gate_deviation`：$|g - 1|$ 的平均。这是"策略偏离预训练模型多远"的**直接度量**，比 W2 loss 更容易解读（W2 是动作空间的距离，gate deviation 是速度场的相对变化）。

这两个指标在第 13 章会作为核心监控项讨论。

## 4. 双轨去噪：`return_bc_reference=True`

这是 W2 项的实现基础，也是本链路相对 LeRobot 实现的一个优势。

### 4.1 要解决的问题

W2 项需要"如果不做 RL，原始 BC 策略在这个状态上会输出什么"。常规做法是**保留一份冻结的模型副本**，前向一次拿到参考动作。对 20 亿参数的 GR00T，这意味着多一份权重（显存翻倍）和多一次完整前向。

Flow-G 的结构提供了一条捷径：**速度场本来就是冻结的，唯一被训练的是 gate。所以"关掉 gate 再跑一遍"就等于"原始 BC 策略"。** 不需要模型副本。

### 4.2 实现

关键是两条轨迹要**共享随机性**，否则比较的就不是"gate 的影响"而是"两次采样的差异"。代码这样做：

```python
bc_x_t = x_t.detach().clone() if return_bc_reference else None      # 同一份初始噪声
...
for idx in range(self.num_inference_timesteps):
    # 主轨迹：带 gate，跟踪梯度
    with torch.set_grad_enabled(track_step_gradient):
        mean, std, flow_g_gate = self.sample_mean_var_val(..., x_t=x_t, ...)   # apply_flow_g 默认 True
        if mode == "train":
            transition_noise = self.sample_noise(x_t.shape, x_t.device, dtype=x_t.dtype)
            x_t = mean + transition_noise * std
        else:
            transition_noise = None
            x_t = mean
    # BC 轨迹：不带 gate，不跟踪梯度，复用同一份 transition_noise
    if bc_x_t is not None:
        with torch.no_grad():
            bc_mean, bc_std = self.sample_mean_var_val(..., x_t=bc_x_t, ..., apply_flow_g=False)
            bc_x_t = bc_mean + transition_noise * bc_std if transition_noise is not None else bc_mean
```

三处共享，逐一说明为什么必要：

| 共享的东西 | 代码 | 不共享会怎样 |
|------------|------|--------------|
| 初始噪声 $x_0$ | `bc_x_t = x_t.detach().clone()` | 两条轨迹从不同起点出发，终点差异里混入了初始噪声的贡献 |
| 每一步的转移噪声 $\epsilon_i$ | `bc_x_t = bc_mean + transition_noise * bc_std` | 同上，每一步都引入无关的差异 |
| 状态条件 | 同一次 backbone 前向的 `vl_embs` / `state_features` | 无法共享的话就要跑两次 VLM，成本翻倍 |

**共享之后，两条轨迹的差异只来自一个原因：主轨迹的速度被 gate 缩放了。** 于是

$$
\mathcal{L}_{\text{W2}} \propto \big\| \pi_\phi(s;\epsilon) - \pi_{\text{BC}}(s;\epsilon) \big\|^2
$$

是一个**逐样本、共同随机数**的比较，方差远小于两次独立采样的比较。这在统计上叫 common random numbers 技巧。

**参考动作的产出**：

```python
if bc_x_t is not None:
    bc_pre_squash_actions = bc_x_t[:, :chunk_length, :action_dim]
    output["bc_reference_actions"] = (
        torch.tanh(bc_pre_squash_actions) if action_squash == "tanh" else bc_pre_squash_actions
    ).float()
```

本链路 `action_squash: none`，所以直接取前 $16\times62$ 的切片。

**成本**：BC 轨迹在 `torch.no_grad()` 下跑，且 `apply_flow_g=False`（省掉 gate 的前向），但**DiT 的 32 层还是要跑 4 次**。所以 `return_bc_reference=True` 会让这次调用的计算量接近翻倍。

**一条防御性检查**：

```python
if return_bc_reference and not flow_g_config.get("enabled", False):
    raise ValueError("A frozen-BC reference requires Flow-G to be enabled")
```

没有 Flow-G 就没有"关掉 gate"这个概念，参考动作无从产生。断言拦住这个组合是对的。

### 4.3 W2 项现在只做一次前向

第 01 章 9.7 记录过一个问题：早期实现里 `ConRFTActorObjective.compute` 做了**两次**独立的 `ForwardType.SAC` 调用，一次给 Q 项、一次给 W2 项，两次的初始噪声不同。后果是逐样本的对抗关系断了（Q 在样本 $k$ 上推得猛时，W2 不会在同一个样本上给出对应强度的反作用），而且策略前向白跑一遍。

**现在的实现是单次前向**：

```python
policy = model(
    forward_type=ForwardType.SAC,
    forward_inputs=data,
    return_bc_reference=not self.offline_stage,   # 只有在线阶段需要参考动作
)
policy["actions"].retain_grad()
q_values = model(forward_type=ForwardType.SAC_Q, forward_inputs=data, actions=policy["actions"])
policy_q = q_values.min(dim=-1).values
q_loss = -policy_q.mean()
...
if not self.offline_stage:
    reference_actions = policy.get("bc_reference_actions")
    if reference_actions is None:
        raise RuntimeError("Online ConRFT requires a frozen-BC reference action")
    w2_loss = _masked_action_mse(policy["actions"], reference_actions, data["chunk_sac_valid"])
```

三个变化：

1. **`return_bc_reference=not self.offline_stage`**：离线阶段不需要 W2，所以不跑 BC 轨迹，省掉一半 DiT 前向。在线阶段才开双轨。
2. **Q 项和 W2 项共用 `policy["actions"]`**：同一份采样，逐样本对齐。
3. **`conrft_eval_context` 从 actor 路径上消失了**。它现在只用在 Cal-QL 的候选采样里（第 08 章）。这也顺带消除了一个语义隐患——原来 W2 那次前向是在 `model.eval()` 下跑的，和 Q 项那次的 `model.train()` 不一致。

**成本对比**（以在线阶段一次 actor 更新为单位，只数 DiT 的 4 步去噪次数）：

| 版本 | 主轨迹 | BC 轨迹 | 第二次主轨迹 | 第二次 BC 轨迹 | expert BC 前向 | 合计 |
|------|--------|---------|--------------|----------------|----------------|------|
| 早期 | 1 | 0 | 1 | 1 | 1（velocity loss，无梯度） | 4 |
| 现在 | 1 | 1 | — | — | 1 | 3 |

离线阶段更省：早期是 1（主）+ 1（velocity BC）= 2，现在是 1（主，`return_bc_reference=False`，BC 项复用它）= **1**。

## 5. `ForwardType.SAC_Q`：给动作打分

这个入口比 SAC 简单得多。它复用 backbone 特征，只跑 critic：

```python
def sac_q_forward(self, forward_inputs, actions, use_target=False, reference_actions=None,
                  return_reference=False, paired_actions=None):
    _, _, vl_embs, state_features = self._prepare_chunk_sac_features(forward_inputs)
    valid = forward_inputs.get("chunk_sac_valid")
    if reference_actions is None:
        reference_actions = forward_inputs.get("chunk_sac_bc_reference_action")
    if paired_actions is None:
        return self.action_head.get_chunk_sac_q_values(
            vl_embs, state_features, actions, use_target=use_target,
            valid=valid, reference_actions=reference_actions, return_reference=return_reference)
    ...
```

三个参数值得注意：

- **`use_target=True`**：切到 `chunk_sac_target_critic`（EMA 副本）。只在 TD target 里用。
- **`paired_actions`**：same-state pair TD 的通路，ConRFT 永不使用（第 04 章的断言禁掉了）。
- **`reference_actions` / `return_reference`**：BC-relative critic 的通路，`flat_absolute` 架构下会直接报错（`raise ValueError("Reference Q values require a BC-relative Critic")`）。ConRFT 强制 `flat_absolute`，所以这两个参数在本链路是无效参数。

critic 内部做什么、动作怎么被 clamp、rot6d 怎么被重新正交化——这些是第 06 章的内容。

**一个成本上的观察**：`sac_q_forward` 每次都会重新跑一遍 `_prepare_chunk_sac_features`，也就是**重新跑一次 VLM backbone**。在 Cal-QL 的保守项里，一次 critic 更新要调 `SAC_Q` 两次（一次算 12 个候选的 Q、一次算数据动作的 Q），加上 TD target 里的一次，加上 actor 那次——**同一个 batch 的 backbone 被跑了 4 遍以上**。虽然都在 `no_grad` 下，但这是阶段一慢的主要原因。第 08 章会算具体的前向次数。

## 6. `ForwardType.SFT`：Flow Matching 的 BC loss

BC 项复用了 GR00T 原本的监督训练路径。ConRFT 侧的调用是：

```python
bc_source = data if bc_data is None else bc_data
flow_data = dict(bc_source)
flow_data["qc_target_action"] = bc_source["chunk_sac_action"]
flow_data["qc_action_mask"] = (
    bc_source["chunk_sac_valid"].unsqueeze(-1).expand_as(bc_source["chunk_sac_action"])
)
bc_loss = model(
    forward_type=ForwardType.SFT,
    data=flow_data,
    sample_weights=torch.ones(bc_source["chunk_sac_action"].shape[0], device=...),
)
```

三点说明：

1. **键名转换**：`chunk_sac_action` → `qc_target_action`，`chunk_sac_valid` → `qc_action_mask`。SFT 路径是 QC（Q-Chunking）那条链路先建的，键名沿用了它的约定。ConRFT 只是做一次改名适配。
2. **mask 的形状扩展**：`chunk_sac_valid` 是 $[B, 16]$（每个时间步是否有效），`expand_as(action)` 变成 $[B, 16, 62]$。意思是"某个时间步无效时，它的全部 62 个维度都不参与 loss"。
3. **`sample_weights` 全是 1**：SFT 路径支持按样本加权（AWR 那条链路用它做优势加权），ConRFT 不用，全部给 1。

BC loss 本身是 Flow Matching 的标准目标：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{\tau \sim \text{Beta}(1.5, 1.0),\ \epsilon \sim \mathcal{N}(0,I)} \Big[ \big\| v_\theta\big(x_\tau, \tau, s\big) - (a^* - \epsilon) \big\|^2 \odot m \Big]
$$

**Step 1：这个公式在做什么**

**它随机取一个时间 $\tau$，把演示动作和噪声按 $\tau$ 混合成 $x_\tau$，然后要求速度场预测出"从噪声指向演示动作"的那个方向。**

> **一句话直觉**：随机挑一个"捏了一半"的中间状态，考网络"接下来该往哪捏"。

**逐符号拆解**：

| 符号 | 含义 | 直觉 | 具体值 |
|------|------|------|--------|
| $\tau$ | 采样的时间点 | 捏到几分熟 | $\text{Beta}(1.5, 1.0)$ 采样后做 $(1-u)\times0.999$ 变换 |
| $\epsilon$ | 高斯噪声 | 起点 | $\mathcal{N}(0,I)$，形状同动作 |
| $a^*$ | 演示动作（目标） | 终点 | `qc_target_action`，$[B,16,62]$ |
| $x_\tau$ | 混合状态 | 中间态 | $(1-\tau)\epsilon + \tau a^*$ |
| $a^* - \epsilon$ | 真实速度 | 从起点直指终点的方向 | 这是 Flow Matching 的监督目标 |
| $m$ | valid mask | 屏蔽 padding | `qc_action_mask`，$[B,16,62]$ |

**代入数字**：某个坐标 $a^* = 0.6$，$\epsilon = -0.9$，采到 $\tau = 0.4$：

$$
x_\tau = (1-0.4)\times(-0.9) + 0.4\times0.6 = -0.54 + 0.24 = -0.30
$$

真实速度 $= 0.6 - (-0.9) = 1.5$。如果网络在 $(x_\tau = -0.30, \tau = 0.4)$ 处预测 $v_\theta = 1.2$，这个坐标的 loss 贡献是 $(1.2 - 1.5)^2 = 0.09$。

**为什么 $\tau$ 用 Beta(1.5, 1.0) 而不是均匀分布**：$\text{Beta}(1.5,1.0)$ 的密度是 $p(u) \propto u^{0.5}$，偏向大的 $u$；经过 $\tau = (1-u)\times0.999$ 变换后偏向**小的 $\tau$**，也就是**更接近纯噪声的那一端**被采得更多。这是因为去噪早期的方向决定了整体轨迹的形状，训练时多花力气在那里更有价值。这是 GR00T 原生的设定，ConRFT 没有改。

**注意 BC 项训的是谁**：$v_\theta$ 是**冻结的**速度场。而 `flow_matching_bc_loss` 的计算图里只有 `action_encoder`、DiT（`self.model`）、`action_decoder` 三个组件：

```python
action_features = self.action_encoder(noisy_actions, timesteps, embodiment_id)
...
model_output = self.model(hidden_states=state_action_features, encoder_hidden_states=vl_embs, timestep=timesteps)
prediction = self.action_decoder(model_output, embodiment_id)
predicted_velocity = prediction[:, -self.action_horizon:]
squared_error = (predicted_velocity[...] - target_velocity[...]).square()
```

**这条路径完全没有 `flow_g_adapter`。** gate 只在 `sample_mean_var_val` 里被应用，而 `flow_matching_bc_loss` 是 GR00T 原生的监督训练路径，直接调 `self.model`。

再看 `model_provider_func`（第 1 节提过）：整个模型先 `requires_grad_(False)`，然后只解冻 `flow_g_adapter.*` 和 `chunk_sac_critic.*`。

**两件事一叠加，结论是**：`bc_loss` 的计算图里没有任何可训练参数，`bc_loss.requires_grad == False`，它是一个常数。actor loss 里 `bc_weight * bc_loss` 这一项对 `backward()` 的贡献精确为 0。

**这曾经是一个已核实的 P0 bug**，完整证据链见 [第 01 章 9.14](./01_全链路总览#9.14-p0-已核实-actor-loss-里权重-1.0-的-bc-项梯度恒为零)。根因是：ConRFT 为了对齐 LeRobot 的实现用了 velocity loss，而 LeRobot 那边训的是整个 action head，所以在那里 velocity loss 是有梯度的——**替换没考虑到本仓库"只训 gate"这个架构差异**。

### 6.1 现在的 BC 项：动作空间的 masked MSE

**修法是改用 Flow-G 那条已经验证过的口径**：先用 `ForwardType.SAC` 采一份动作（经过 gate，有梯度），再和数据里的真实动作做 masked MSE。当前实现是：

```python
bc_source = data if bc_data is None else bc_data
bc_policy_actions = (
    policy["actions"]                                   # 离线：data 就是 expert batch，直接复用主前向
    if bc_data is None
    else model(                                          # 在线：expert batch 是独立的一批，需要额外一次前向
        forward_type=ForwardType.SAC,
        forward_inputs=bc_source,
    )["actions"]
)
bc_loss = _masked_action_mse(
    bc_policy_actions,
    bc_source["chunk_sac_action"],
    bc_source["chunk_sac_valid"],
)
```

三个要点：

1. **离线阶段零额外成本**。离线时 `bc_data is None`（因为 `trains_on_expert_batch=True` 时 actor 直接训在 expert batch 上，`data` 本身就是 expert batch），所以 `bc_policy_actions` 直接复用主前向的 `policy["actions"]`——一次去噪同时供 Q 项和 BC 项使用。
2. **在线阶段多一次前向**。在线时 `data` 是在线 replay、`bc_data` 是 expert replay，两批状态不同，必须各跑一次采样。
3. **口径和 W2 项统一了**。BC 项、W2 项现在都是 `_masked_action_mse`，只是参考对象不同：BC 项比的是**数据里的真实动作**，W2 项比的是**冻结 BC 模型的输出**。

$$
\mathcal{L}_{\text{BC}} = \frac{\sum_{b,h} m_{b,h} \left\| \pi_\phi(s_b)_h - a^{*}_{b,h} \right\|^2}{\left(\sum_{b,h} m_{b,h}\right) \cdot D}
$$

**Step 1：这个公式在做什么**

**它算出"策略输出的动作 chunk"和"演示动作 chunk"之间的平均逐坐标平方误差**，只在有效时间步上统计。

> **一句话直觉**：策略采出来的这 16 步，和人类演示的这 16 步，每个关节差多少。

**逐符号拆解**：

| 符号 | 含义 | 直觉 | 具体值 |
|------|------|------|--------|
| $b$ | batch 索引 | 第几个样本 | $0..B-1$，$B$ = micro_batch_size = 2 |
| $h$ | chunk 内的时间步 | 第几个控制步 | $0..15$ |
| $m_{b,h}$ | valid mask | 这一步是不是 padding | `chunk_sac_valid`，0 或 1 |
| $\pi_\phi(s_b)_h$ | 策略采样动作的第 $h$ 步 | 经过 4 步去噪 + gate 的输出 | $[62]$ |
| $a^{*}_{b,h}$ | 演示动作的第 $h$ 步 | `chunk_sac_action` | $[62]$ |
| $D$ | 动作维度 | 双臂 + 双手 | 62 |
| 分母 | 有效元素总数 | 归一化，使 loss 与 chunk 有效长度无关 | $(\sum m) \times 62$ |

代码里对应：

```python
def _masked_action_mse(actions, reference_actions, valid):
    squared_error = (actions.float() - reference_actions.float()).square()
    valid_float = valid.to(squared_error.dtype)
    return (squared_error * valid_float.unsqueeze(-1)).sum() / (
        valid_float.sum().clamp_min(1.0) * squared_error.shape[-1]
    )
```

**代入数字**：$B=2$，两个样本的有效步数分别是 16 和 11，所以 $\sum m = 27$，分母 $= 27 \times 62 = 1674$。如果所有有效坐标的平方误差之和是 $653.4$：

$$
\mathcal{L}_{\text{BC}} = \frac{653.4}{1674} = 0.390
$$

这正好是实测日志里 warmup 第 35 次更新的 `bc_loss = 0.3903` 的量级。对应的**逐坐标 RMS 误差**是 $\sqrt{0.390} = 0.625$——注意动作是 `q01_q99` 归一化到 $[-1,1]$ 的，所以 0.625 是一个相当大的偏差。这符合预期：策略是随机的（flow SDE 每步注入噪声），单次采样和演示动作差得多是正常的，BC 项管的是**期望上**对齐。

**为什么分母要用有效元素数而不是固定的 $B \times 16 \times 62$**：episode 尾部的 chunk 可能只有几步有效。如果用固定分母，短 chunk 的 loss 会被人为压小，等于降低了它的权重。用有效元素数归一化之后，每个 chunk 无论长短都贡献相同量级的 loss。

**修好之后实测到的量**（warmup 第 35 次更新）：

| 指标 | 值 | 说明 |
|------|-----|------|
| `bc_loss` | 0.3903 | 上面算过 |
| action-gradient norm | 0.0142 | **非零，证明梯度确实流回了 gate** |
| `flow_g_gate_deviation` | 0.0221 | gate 已经偏离恒等约 2.2% |
| critic loss | 0.0718 | TD + Cal-QL |

**代价**：BC 项从 velocity 空间的回归变成了动作空间的回归，不再和 LeRobot ConRFT 的 BC 项逐项对齐。但在"速度场冻结、只训 gate"的架构下，velocity loss 没有可训练参数，这是唯一的选择。第 09 章会讨论这个替换对离线阶段收敛行为的影响。

## 7. 小结

| 主题 | 关键结论 |
|------|----------|
| backbone | 全程 `torch.no_grad()`，2B 参数完全冻结 |
| 可训练参数 | 只有 Flow-G gate（约 3.4 万）+ ChunkSACCritic（两个 3 层 MLP） |
| 随机策略怎么来的 | flow SDE：每步从 $\mathcal{N}(\mu_i, \sigma_i^2)$ 采样，重参数化保证可微 |
| 噪声强度 | 从 $\sigma^{\text{step}} = 0.433$ 递减到 $0.144$（早期探索多） |
| `log_pi` | 是**路径**密度不是动作边际密度；归一化除数 $5\times16\times62 = 4960$。ConRFT 不用它，且现在 `compute_path_log_prob: false`，所以它**恒为 0** |
| 梯度回传 | `actor_backprop_steps: 4`，全部 4 步都回传 |
| Flow-G gate | $g = 2.0\cdot\sigma(f_\phi(\cdot)) \in (0,2)$，恒等初始化 $g\equiv1$，只能改幅度不能改方向 |
| BC 参考动作 | 同一份初始噪声 + 同一份转移噪声，只关掉 gate 再跑一遍。零额外显存 |
| 双轨去噪的代价 | DiT 前向翻倍 |
| BC 项 | 动作空间 masked MSE（不是 velocity loss）。离线复用主前向，在线多一次前向 |
| 已修复的问题 | 9.14（BC 梯度恒为 0）、9.7（两次独立采样）、9.12（`log_pi` 白算）——三条都已修，见第 01 章 |

## 下章预告

动作有了，接下来要给它打分。第 06 章讲 critic：`flat_absolute` twin-Q 的具体结构（为什么把 $16\times62$ 展平成 992 维直接喂进 MLP）、动作在进 critic 之前要经过的 clamp 和 rot6d 重正交化（这解释了第 01 章 9.6 的梯度死区）、以及 chunk 级 TD target 里 $\gamma^{16} = 0.851$ 这个折扣和 bootstrap mask 的语义。

→ [第 06 章 Critic 架构与 chunk 级 TD target](./06_Critic架构与TD目标)
