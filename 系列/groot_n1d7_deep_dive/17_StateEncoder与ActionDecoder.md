---
title: "StateEncoder 与 ActionDecoder：输入输出映射"
series:
  id: groot_n1d7_deep_dive
  chapter: 17
order: 17
---

# StateEncoder 与 ActionDecoder：输入输出映射

> CategorySpecificMLP 在 GR00T 中的两个具体应用——如何把机器人状态编码成 DiT 输入，如何把 DiT 输出解码回物理动作空间。

## 相关阅读

- [CategorySpecificMLP](./16_CategorySpecificMLP_多具身体条件化)（上一章）
- [VLLN 与 VL Self-Attention](./18_VLLN与VL_SelfAttention)（下一章）

---

## 前情提要

上一章我们理解了 `CategorySpecificLinear` 和 `CategorySpecificMLP` 的通用机制。
本章看它们在 GR00T ActionHead 中的两个具体应用：编码输入状态的 `StateEncoder`，
和解码输出动作的 `ActionDecoder`。

---

## 1. StateEncoder：把机器人状态变成 DiT 输入

### 1.1 要解决的问题

机器人状态（如关节角度、末端位姿）通常是一个历史窗口，而不仅仅是当前一瞬间的值。
比如 `state_history_length=1` 时，只用当前一帧；但配置支持大于1，意味着可以同时
输入最近几帧的状态，帮助模型感知"运动趋势"（虽然 GR00T 默认只用当前帧，因为
运动信息主要靠图像帧的历史来提供）。

不管历史长度是多少，状态最终都需要被"拍扁"成一个向量，送入 `CategorySpecificMLP` 编码。

### 1.2 实现

```python
self.state_encoder = CategorySpecificMLP(
    num_categories=config.max_num_embodiments,     # 32
    input_dim=config.max_state_dim * config.state_history_length,  # 132 * 1 = 132
    hidden_dim=self.hidden_size,                     # 1024
    output_dim=self.input_embedding_dim,             # 1536
)
```

`input_dim` 是 `max_state_dim * state_history_length`——如果历史长度是 3，
输入维度就是 `132 * 3 = 396`，把 3 帧状态拼接成一个长向量再一起编码。

### 1.3 使用时的形状变换

```python
# 在 ActionHead.forward() 中：
assert action_input.state.shape[1] == self.config.state_history_length
# state 原始形状: [B, state_history_length, max_state_dim]，例如 [B, 1, 132]

action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)
# reshape 后: [B, 1, state_history_length * max_state_dim]，例如 [B, 1, 132]
# (因为 state_history_length=1，这一步 reshape 实际上没有改变数值，只是"拍平"了维度)

state_features = self.state_encoder(action_input.state, embodiment_id)
# 输出: [B, 1, 1536]
```

**当 `state_history_length > 1` 时的效果**：假设历史长度为3，原始state形状是
`[B, 3, 132]`，reshape 后变成 `[B, 1, 396]`——把3帧的状态首尾拼接成一个396维长向量，
序列长度从3变成1（因为我们把"时间"这个维度"塞进"了特征维度）。
这样处理后，`state_features` 只占 DiT 输入序列中的**一个** token 位置，
无论历史长度是多少。

---

## 2. State Dropout：训练时的正则化

编码完状态后，GR00T 在训练时会以一定概率把整个状态特征"清零"。这是本系列
[配置系统全参数解读](./05_配置系统_全参数解读)中提到过的 `state_dropout_prob=0.2`
的具体实现位置：

```python
if self.training and self.state_dropout_prob > 0:
    do_dropout = (
        torch.rand(state_features.shape[0], device=state_features.device)
        < self.state_dropout_prob
    )  # (B,) 布尔向量，True表示这个样本要被dropout
    do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)  # (B, 1, 1)
    state_features = state_features * (1 - do_dropout)  # 广播乘法
```

逐行看这个操作：`torch.rand(B) < 0.2` 为每个 batch 样本独立抽一个 0-1 随机数，
20% 概率小于 0.2（触发dropout）。`do_dropout[:, None, None]` 把形状从 `(B,)`
扩展到 `(B, 1, 1)`，这样能通过广播机制作用到 `state_features` 的 `(B, 1, 1536)` 上——
`1 - do_dropout` 在触发时是0（清零整个状态向量），未触发时是1（保持原状态）。

**为什么要 dropout 掉整个状态向量，而不是随机丢弃部分维度？**（标准 Dropout 的做法）

因为 GR00T 想让模型学会"即使完全没有状态输入，也能从图像中推断出足够的信息"——
这是应对真实部署中状态传感器可能失效、延迟或缺失的情况。随机丢弃部分维度
达不到这个效果（模型仍然可以从剩余的部分维度中"猜"出状态）；
完整清零则强制模型必须有"纯图像理解"的能力作为后备方案。

---

## 3. ActionDecoder：把 DiT 输出解码回物理动作

### 3.1 结构

`ActionDecoder` 用的也是 `CategorySpecificMLP`，但方向相反——从 DiT 的内部维度
解码回物理动作维度：

```python
self.action_decoder = CategorySpecificMLP(
    num_categories=config.max_num_embodiments,  # 32
    input_dim=self.hidden_size,                   # 1024 (DiT输出维度)
    hidden_dim=self.hidden_size,                   # 1024
    output_dim=self.action_dim,                    # 132 (物理动作维度)
)
```

### 3.2 使用时的切片操作

DiT 输出的序列包含 41 个 token（1 state + 40 action）。解码时先对**整个**序列
做解码，再切片取出属于 action 的部分：

```python
# 训练时 (在 ActionHead.forward 中)：
pred = self.action_decoder(model_output, embodiment_id)  # [B, 41, 132]
pred_actions = pred[:, -actions.shape[1]:]  # 取最后40个 -> [B, 40, 132]
```

为什么要"先解码全部、再切片"，而不是"先切片、再解码"？两种做法在数值上
完全等价（因为 `CategorySpecificMLP` 是逐token独立处理的，不涉及跨token的运算），
先解码全部的写法更简洁——不需要额外记录 state token 具体占了几个位置。

### 3.3 推理时的用法

推理时的用法完全相同，只是这里的 `pred` 代表的是"预测的速度"（velocity），
不是最终动作：

```python
# 推理时 (在 get_action_with_features 中)：
pred = self.action_decoder(model_output, embodiment_id)
pred_velocity = pred[:, -self.action_horizon:]  # [B, 40, 132]

# 用 Euler 积分更新动作估计
actions = actions + dt * pred_velocity * vel_strength
```

---

## 4. State 和 Action 的编解码对称性

观察一下 GR00T 中围绕 state 和 action 的四个模块，会发现一种优雅的对称结构：

| 模块 | 输入维度 | 输出维度 | 作用 |
|------|---------|---------|------|
| StateEncoder | 132 (物理) | 1536 (DiT内部) | 编码：状态进入 |
| ActionEncoder | 132 (物理) | 1536 (DiT内部) | 编码：噪声动作进入 |
| DiT | 1536 | 1024 | 处理：核心计算 |
| ActionDecoder | 1024 (DiT内部) | 132 (物理) | 解码：速度预测输出 |

State 只有编码器（因为state是已知的输入条件，不需要"生成"），
Action 既有编码器（编码带噪声的动作）也共享同一个 Decoder（把DiT输出解码为速度）。
这种设计体现了"编码-处理-解码"三段式管线在多具身体场景下的具体实现。

---

## 5. 总结

StateEncoder 和 ActionDecoder 的设计要点：

1. **StateEncoder**：把（可能多帧的）状态历史拍平成一个长向量，用 `CategorySpecificMLP`
   编码到 DiT 内部维度；训练时有20%概率整体清零做正则化
2. **ActionDecoder**：把 DiT 的输出（包含state+action的完整序列）解码到物理动作维度，
   再通过切片取出属于action部分的预测
3. **两者共享 `CategorySpecificMLP` 基础组件**，但服务于不同方向（编码 vs 解码）
   和不同数据（状态 vs 动作/速度）

至此，我们已经理解了 GR00T ActionHead 中"输入怎么进来、输出怎么出去"的完整链路。
下一章我们看骨干特征进入 DiT 之前的最后一道处理——VLLN 和可选的 VL Self-Attention。

---

## 下一章预告

下一章我们将分析 `vlln`（VL LayerNorm）和 `vl_self_attention` 这两个
容易被忽视但很重要的组件——骨干网络输出的原始特征，在送入 DiT 的 cross-attention
之前，还经过了哪些额外处理？
