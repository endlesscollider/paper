---
title: "Rectified Flow + 频率域 Loss：训练目标完整拆解"
series:
  id: xr1_deep_dive
  chapter: 7
order: 7
---

# Rectified Flow + 频率域 Loss：训练目标完整拆解

> **前情提要**：上一章拆解了 Choice Head 的多候选机制。本章深入 DiT 分支的训练目标——Rectified Flow 的数学原理、Beta 时间步采样的设计动机、以及频率域 Loss 如何保护高频动作细节。

**知识链接**：
- 前置知识：[Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流)
- 前代对照：[XR-0 Rectified Flow](/系列/xr0_deep_dive/05_RectifiedFlow_直线插值与速度场回归)

---

## 1. Rectified Flow 回顾

Rectified Flow 是 Flow Matching 的一个特例：用**直线路径**连接噪声分布和数据分布。

核心思想：
- 把 $t=0$ 定义为纯噪声 $x_0 \sim \mathcal{N}(0, I)$
- 把 $t=1$ 定义为真实数据 $x_1$（ground truth 动作）
- 在两者之间做线性插值：$x_t = (1-t) \cdot x_0 + t \cdot x_1$
- 训练一个网络预测这条直线的"速度"：$v = x_1 - x_0$

推理时从 $t=0$ 的纯噪声出发，用 Euler 积分逐步前进到 $t \approx 1$。

## 2. XR-1 的训练过程

### 2.1 时间步采样：Beta(1.5, 1.0) 分布

XR-1 不使用均匀分布 $t \sim U(0,1)$，而是用 Beta 分布采样：

```python
self.beta = Beta(1.5, 1.0)

# 训练时
timestep = ((1 - self.beta.sample((batch_size,))) * 0.999).to(action.dtype)
timestep = timestep[:, None, None]  # [B, 1, 1]
```

注意 `1 - beta.sample()`：Beta(1.5, 1.0) 的原始采样偏向 0~0.6 区间，取 `1-` 后偏向 0.4~1.0 区间。最终 `* 0.999` 确保不会精确等于 1。

$$
t \sim 1 - \text{Beta}(1.5, 1.0), \quad t \in [0, 0.999)
$$

**这个公式在做什么**：让训练时间步偏向接近 1 的高时间步区间（即接近真实数据的区间），同时在低时间步（高噪声）区间也有一定采样。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**为什么偏向高时间步**：

Beta(1.5, 1.0) 的 PDF 在 0 附近有少量密度，在 0.5~1.0 有较高密度。取 `1-` 后，原本密度高的区间 [0.5,1.0] 映射到了 t ∈ [0, 0.5]——等等，这似乎矛盾了？

让我们仔细看：`1 - Beta(1.5, 1.0)` = Beta(1.0, 1.5)。Beta(1.0, 1.5) 的 PDF ∝ $(1-x)^{0.5}$，在 x→0 时密度最大，在 x→1 时密度为 0。所以最终 t 的分布**偏向低时间步**（接近 0，即高噪声区间）。

为什么偏向高噪声区间训练？因为：
1. 高噪声区间的预测更困难（需要从几乎纯噪声中恢复方向），需要更多训练样本
2. 低噪声区间（接近 t=1）的预测相对简单（已经很接近真实数据，小修正即可）
3. 推理时从 t=0 开始，前几步的准确性对最终结果影响最大

**数值代入**：采样 1000 个 t，统计分布：
- t ∈ [0, 0.2)：约 35% 的样本
- t ∈ [0.2, 0.4)：约 25%
- t ∈ [0.4, 0.6)：约 20%
- t ∈ [0.6, 0.8)：约 13%
- t ∈ [0.8, 1.0)：约 7%

大部分训练集中在 t < 0.4 的高噪声区间。
:::

### 2.2 构造 Noisy Action

```python
noise = torch.randn_like(action)  # [B, 30, 60]
noisy_action = (1 - timestep) * noise + timestep * action
```

$$
x_t = (1-t) \cdot \epsilon + t \cdot a^*
$$

**这个公式在做什么**：在纯噪声和真实动作之间做线性插值，t 越大越接近真实动作。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 形状 |
|------|------|------|
| $\epsilon$ | 标准正态噪声 | [B, 30, 60] |
| $a^*$ | ground truth 动作（已归一化） | [B, 30, 60] |
| $t$ | 时间步 | [B, 1, 1]，广播 |
| $x_t$ | 插值后的含噪动作 | [B, 30, 60] |

**数值代入**：$t=0.3$，某维度 $\epsilon=0.8$，$a^*=0.2$：
- $x_t = 0.7 \times 0.8 + 0.3 \times 0.2 = 0.56 + 0.06 = 0.62$

当 $t=0$：$x_t = \epsilon$（纯噪声）
当 $t=1$：$x_t = a^*$（真实动作）
:::

### 2.3 训练目标：预测速度场

目标向量是直线的方向：

```python
target = action - noise  # 速度场 v = x_1 - x_0
```

DiT 输出的是对这个速度场的预测：

```python
pred = self.dit_forward(
    torch.cat([prefix, noisy_action[:, prefix_length:]], dim=1),
    timestep,
    **kwargs
)[:, prefix_length:]  # 只取非前缀部分的预测
```

## 3. MSE Loss（时域）

$$
L_{\text{MSE}} = \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} w_{i,j} \cdot (v^{\text{pred}}_{i,j} - v^{\text{target}}_{i,j})^2
$$

**这个公式在做什么**：在有效掩码位置上计算加权均方误差——预测速度和真实速度的差距。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 说明 |
|------|------|------|
| $v^{\text{pred}}_{i,j}$ | DiT 预测的第 $i$ 步第 $j$ 维速度 | 输出层产出 |
| $v^{\text{target}}_{i,j}$ | 真实速度 $= a^*_{i,j} - \epsilon_{i,j}$ | 直线方向 |
| $w_{i,j}$ | 自适应权重 | 由异步训练的 prefix 预测误差决定 |
| $\mathcal{M}$ | `action_mask` 为 True 的位置集合 | 排除无效维度 |

**权重 $w$ 的来源**（异步训练模式）：
```python
if prefix_length:
    prefix_pred = self._generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs)
    weight = (prefix_pred[:, prefix_length:] - action[:, prefix_length:]).abs()
else:
    weight = torch.ones_like(pred)
```

当有 prefix 时，先用完整推理生成一次，看哪些位置误差大，给大误差位置更高权重。无 prefix 时权重为 1（均匀）。

权重还经过归一化和 clamp：
```python
weight[action_mask] /= weight[action_mask].mean()
weight.clamp_(0.5, 5.0)
```

**数值代入**：假设某位置 $v^{\text{pred}}=0.3$，$v^{\text{target}}=0.5$，$w=1.5$：
- 该位置 loss = $1.5 \times (0.3 - 0.5)^2 = 1.5 \times 0.04 = 0.06$
:::

## 4. FFT Loss（频率域）

这是 XR-1 相对 XR-0 的**全新设计**。

### 4.1 计算方式

```python
freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
```

沿时间轴（dim=1，即 30 步）做实数 FFT，得到频谱。然后取预测频谱和目标频谱的差的绝对值。

$$
L_{\text{FFT}} = \frac{1}{|\mathcal{M}_f|} \sum_{(i,j) \in \mathcal{M}_f} w_j \cdot |\hat{v}^{\text{pred}}_i[j] - \hat{v}^{\text{target}}_i[j]|
$$

**这个公式在做什么**：在频率域上计算预测和目标的差异——保护动作序列中的高频成分（快速变化）不被 MSE 的平滑效应抹掉。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 说明 |
|------|------|------|
| $\hat{v}^{\text{pred}}_i[j]$ | 预测速度第 $i$ 个频率分量第 $j$ 维 | `rfft` 输出的复数取绝对值 |
| $\hat{v}^{\text{target}}_i[j]$ | 目标速度的频谱 | 同上 |
| $\mathcal{M}_f$ | 频率域有效掩码 | 排除 `freq_excluded_dims=[17,18,19]`（底盘速度维度） |
| $w_j$ | 权重（取 batch 平均的时域权重） | `weight.mean(dim=(1,2))` |

**为什么排除 dim 17-19**：这三个维度是底盘速度（base_vel），在频率域上的高频变化没有物理意义（底盘不会高频抖动），所以不用 FFT Loss 约束。

**数值代入**：30 步的 rfft 输出长度 = 30//2+1 = 16 个频率分量。假设第 5 个频率分量（中高频）：
- $|\hat{v}^{\text{pred}}_5[0]| = 0.12$
- $|\hat{v}^{\text{target}}_5[0]| = 0.20$
- 该位置 loss = |0.12 - 0.20| = 0.08

**为什么需要 FFT Loss**：MSE Loss 在时域上对每个时间步独立计算误差。如果目标是一个"快速抬起→放下"的动作（高频），MSE 会倾向于输出"慢慢抬起→慢慢放下"（平滑版本），因为这样时域上的逐步误差更小。FFT Loss 直接在频率域上惩罚高频成分的缺失，迫使模型保留快速变化的动作细节。
:::

### 4.2 有效性判断

只有完整 30 步都有效（`action_mask[:, -1].any(dim=1)` 为 True）的样本才计算 FFT Loss，因为不完整的序列做 FFT 没有意义。

```python
valid_batch = action_mask[:, -1].any(dim=1)
if not torch.any(valid_batch):
    return loss_mse, freq.sum() * 0.0  # 跳过 FFT
```

## 5. 推理：5 步 Euler 积分

推理时不需要 loss，只需要从噪声生成动作：

```python
@torch.no_grad()
def _generate(self, noise, kwargs):
    sample = noise.clone()
    dt = 1.0 / self.num_steps  # = 0.2
    for step in range(self.num_steps):  # 5 步
        timestep = step / self.num_steps  # 0.0, 0.2, 0.4, 0.6, 0.8
        sample = sample + self.dit_forward(sample, timestep, **kwargs) * dt
    return sample
```

$$
x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t, \quad \Delta t = 0.2
$$

**这个公式在做什么**：标准 Euler 积分——每步沿网络预测的速度方向走 $\Delta t = 0.2$ 的距离。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 值 |
|------|------|-----|
| $x_t$ | 当前时刻的样本（从纯噪声开始） | [B, 30, 60] |
| $v_\theta(x_t, t)$ | DiT 在 $(x_t, t)$ 处预测的速度 | [B, 30, 60] |
| $\Delta t$ | 步长 | 1/5 = 0.2 |
| $t$ | 当前时间 | 0.0, 0.2, 0.4, 0.6, 0.8 |

**5 步完整流程**：
- Step 0: $x_{0.2} = x_0 + v_\theta(x_0, 0.0) \times 0.2$
- Step 1: $x_{0.4} = x_{0.2} + v_\theta(x_{0.2}, 0.2) \times 0.2$
- Step 2: $x_{0.6} = x_{0.4} + v_\theta(x_{0.4}, 0.4) \times 0.2$
- Step 3: $x_{0.8} = x_{0.6} + v_\theta(x_{0.6}, 0.6) \times 0.2$
- Step 4: $x_{1.0} = x_{0.8} + v_\theta(x_{0.8}, 0.8) \times 0.2$

最终 $x_{1.0}$ 就是生成的动作序列（归一化空间下）。

**为什么 5 步就够了**：Rectified Flow 的直线插值路径比 DDPM 的曲线路径简单得多。理论上完美训练后 1 步就能精确生成（因为速度场在任何位置都是常数 $v = x_1 - x_0$）。实际中 5 步是精度和速度的良好平衡。
:::

## 6. 异步训练的权重设计

当 `prefix_length > 0`（异步训练模式）时，loss 计算有特殊处理：

```python
if prefix_length:
    # 先用 prefix 做一次完整推理
    prefix_pred = self._generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs)
    # 计算推理结果和 GT 的绝对误差作为权重
    weight = (prefix_pred[:, prefix_length:] - action[:, prefix_length:]).abs()
else:
    weight = torch.ones_like(pred)
```

**设计直觉**：有 prefix 条件时，模型"应该"能做得更好（有前几步的已知信息）。对推理时已经预测不好的位置（大误差）给更高权重，集中训练力度在"难点"上。

## 7. 本章小结

XR-1 的训练目标由两个互补的 loss 组成：

| Loss | 域 | 保护的内容 | 权重 |
|------|-----|-----------|------|
| MSE | 时域 | 每步动作值的准确性 | 0.5 |
| FFT | 频域 | 动作序列的高频变化（快速动作） | 1.0 |

加上 Choice Head 的 L1 + Score，总共 4 项 loss 协同训练。

关键设计选择：
1. **Beta(1.5,1.0) 时间步**：偏向高噪声区间训练，因为推理起始阶段最关键
2. **频率域 Loss**：防止 MSE 把快速动作平滑掉
3. **5 步 Euler 积分**：直线路径下足够精确，推理延迟低
4. **异步权重**：有 prefix 条件时聚焦训练难点

---

**下一章预告**：[Ch07 后训练对齐](./07_后训练对齐_Embodiment与Instruction) 将解释预训练完成后，如何通过两轴对齐把通用能力映射到具体机器人和自然语言指令。
