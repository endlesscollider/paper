---
title: "Choice Head：5 候选动作生成与评分排序机制"
series:
  id: xr1_deep_dive
  chapter: 6
order: 6
---

# Choice Head：5 候选动作生成与评分排序机制

> **前情提要**：上一章拆解了 DiT 36 层动作头的内部结构。本章聚焦 XR-1 的独有设计——Choice Head，解释为什么要生成 5 个候选动作再选最好的。

**知识链接**：
- 前置知识：[行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式)
- 前代对照：XR-0 没有 Choice Head，直接输出单一动作

---

## 1. 为什么需要 Choice Head？

### 1.1 问题：多模态动作分布

操作任务中一个常见现象：**同一个观测+指令，可能对应多种合理的执行方式**。

例如"把杯子放到桌子右边"——机器人可以：
- 直接平移过去
- 先抬高再横移再放下
- 绕过中间的障碍物走另一条路

这些都是正确答案。如果训练数据中包含了多种路径，纯 MSE Loss 会让模型学到一个"平均"动作（所有合理路径的均值），这个均值本身可能完全不是一条可行路径。

### 1.2 解法：生成多个候选，选最好的

Choice Head 的策略直截了当：
1. 让 VLM 一次性输出 **5 组**候选动作
2. 同时输出每组候选的质量评分
3. 训练时只对**最接近 ground truth 的那一组**计算 loss
4. 推理时选评分最高的那一组

这本质上是 **Winner-Takes-All (WTA)** 训练策略的一种实现——多个"head"并行预测，只有最好的那个被优化，其他的不受惩罚。

## 2. Choice Head 的架构

Choice Head 利用 VLM 输出中的特殊 token：

| Token 类型 | token_id 范围 | 数量 | 用途 |
|-----------|--------------|------|------|
| SCORE | 151669 | 1 per sample | 对应 5 个评分的输出位置 |
| STATE | 151670 | 1 per sample | 传入机器人状态的位置 |
| ACTION | 151671~151730 | N per sample | 对应 5×action_dim 候选输出位置 |

架构组件：

```python
# 状态投影（VLM 分支用的，和 DiT 的 state_projector 独立）
self.state_projector_choice = Projector(60, 2560, num_layers=2)

# 动作候选投影：VLM hidden → 5 组动作
self.action_projector_choice = nn.Sequential(
    Projector(2560, 2560, num_layers=4),     # 4层 MLP
    Projector(2560, 60 * 5),                  # 投影到 300 维 = 60 × 5
)

# 评分投影：VLM hidden → 5 个分数
self.score_projector_choice = nn.Sequential(
    Projector(2560, 2560, num_layers=4),     # 4层 MLP
    Projector(2560, 5),                       # 投影到 5 维
)
```

## 3. 训练时的 Choice Head 前向

### 3.1 从 VLM 提取特征

VLM 前向后，从 `hidden_states` 中按 token 位置提取对应特征：

```python
# 提取所有 ACTION 位置的 hidden states
action_choice = self.action_projector_choice(
    vlm_outputs.hidden_states[
        (input_ids >= ACTION_START_ID) & (input_ids < ACTION_END_ID)
    ]
)
# 形状：[total_action_tokens, 300]  (300 = 60 × 5)

# 提取所有 SCORE 位置的 hidden states
score_choice = self.score_projector_choice(
    vlm_outputs.hidden_states[input_ids == SCORE_ID]
)
# 形状：[B, 5]
```

### 3.2 Choice Loss 计算

Choice Loss 的核心逻辑在 `compute_choice_loss` 函数中：

```python
def compute_choice_loss(self, action_pred, score_pred, target, mask, actual_lengths):
    action_losses, score_losses = [], []
    start = 0
    for sample_index, length in enumerate(actual_lengths):
        end = start + length
        # 1. 把预测reshape成 [5, length, action_dim]
        predictions = action_pred[start:end].reshape(length, 5, -1).transpose(0, 1)
        sample_target = target[start:end].unsqueeze(0).repeat(5, 1, 1)
        sample_mask = mask[start:end].bool().unsqueeze(0).repeat(5, 1, 1)

        # 2. 每组候选和 GT 算 L1 误差
        absolute_error = F.l1_loss(predictions, sample_target, reduction="none")
        choice_error = absolute_error[sample_mask].reshape(5, -1).mean(dim=-1)

        # 3. Winner-Takes-All：只优化误差最小的那一组
        choice_index = choice_error.argmin()
        action_losses.append(choice_error[choice_index])

        # 4. Score loss：让评分准确反映各候选的真实误差
        score_losses.append(
            ((score_pred[sample_index] - choice_error.detach()) ** 2).mean()
        )
        start = end

    return torch.stack(action_losses).mean(), torch.stack(score_losses).mean()
```

## 4. 两个 Loss 的设计动机

### 4.1 Action Choice Loss（L1）

$$
L_{\text{choice}} = \min_{k \in \{1,...,5\}} \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} |a^{(k)}_{i,j} - a^{*}_{i,j}|
$$

**这个公式在做什么**：在 5 组候选动作中找到和 ground truth 最接近的那一组，只对它计算 L1 loss。其他 4 组不参与梯度。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $k$ | 候选索引 | 1~5 中的一个 |
| $a^{(k)}_{i,j}$ | 第 $k$ 组候选的第 $i$ 步、第 $j$ 维 | 预测的动作值 |
| $a^{*}_{i,j}$ | ground truth 动作 | 训练数据中的真实动作 |
| $\mathcal{M}$ | 有效掩码集合 | action_mask 为 True 的位置 |
| $\|\mathcal{M}\|$ | 有效元素数 | 用于平均 |

**数值代入**：假设 action_dim=60, action_length=30, 5 组候选的平均 L1 误差分别为 [0.12, 0.08, 0.15, 0.09, 0.11]：
- argmin = 第 2 组（误差 0.08）
- $L_{\text{choice}} = 0.08$
- 只有第 2 组参与反向传播

**为什么用 L1 而不是 MSE**：L1 对离群点更鲁棒。在多模态场景下，某些候选可能预测了完全不同的路径，L1 不会因为"偶尔大偏差"而过度惩罚。
:::

### 4.2 Score Loss（MSE）

$$
L_{\text{score}} = \frac{1}{5} \sum_{k=1}^{5} (s_k - \bar{e}_k)^2
$$

**这个公式在做什么**：让 Score Head 输出的评分准确地预测每组候选的实际 L1 误差。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $s_k$ | Score Head 对第 $k$ 组的评分 | score_projector 的第 $k$ 维输出 |
| $\bar{e}_k$ | 第 $k$ 组的真实平均 L1 误差 | `choice_error.detach()`，不传梯度 |

**数值代入**：5 组候选真实误差 $\bar{e}$ = [0.12, 0.08, 0.15, 0.09, 0.11]，Score 预测 $s$ = [0.11, 0.10, 0.14, 0.08, 0.12]：
- 各项差值² = [(0.01)², (0.02)², (0.01)², (0.01)², (0.01)²]
- = [0.0001, 0.0004, 0.0001, 0.0001, 0.0001]
- $L_{\text{score}}$ = mean = 0.00016

**为什么这样设计**：
- $\bar{e}_k$ 用 `.detach()` 切断梯度——Score Head 的训练不会影响 Action Head 的参数
- 推理时，Score 预测误差越小的候选→分数越低→选它（argmin score 等价于选最好的候选）
- 注意：score 预测的是**误差**，所以推理时选 score 最小的那组（不是最大的）
:::

## 5. 推理时 Choice Head 的使用

推理时 Choice Head 和 DiT 是两个并行的动作来源：

1. **DiT 分支**：通过 5 步 Euler 积分生成一组动作
2. **Choice Head**：VLM 直接输出 5 组候选 + 评分

具体选择策略（在 `generate` 方法中）：
- 当前开源代码中推理主要走 DiT 分支
- Choice Head 的候选可以作为参考/备选
- 实际部署时可能用 ensemble 策略（如 DiT 输出 + 最佳 Choice 加权平均）

## 6. Choice Head 和 DiT 的关系

一个自然的问题：既然有 DiT 做 Flow 生成，为什么还要 Choice Head？

| 方面 | DiT (Flow) | Choice Head |
|------|-----------|-------------|
| 生成方式 | 迭代去噪（5步） | VLM 单次前向 |
| 多模态处理 | Flow 本身能建模多模态 | WTA 显式选最佳 |
| 计算成本 | 5 次 DiT 前向 | VLM 附带产生，几乎免费 |
| 输出质量 | 通常更好（迭代优化） | 快但不一定最优 |
| 训练信号 | MSE + FFT | L1 + Score |

Choice Head 的价值：
1. **辅助训练信号**：为 VLM 骨干提供额外的动作预测梯度，促进 VLM 学到更好的动作相关表示
2. **快速初筛**：推理时可以先看 Choice Head 的评分快速判断"当前观测是否明确"
3. **互补监督**：DiT 用 MSE（连续优化）、Choice 用 L1（WTA），两种梯度信号互补

## 7. 总 Loss 的组合

XR-1 的总训练 loss：

$$
L_{\text{total}} = 0.5 \cdot L_{\text{MSE}} + 1.0 \cdot L_{\text{FFT}} + 0.5 \cdot L_{\text{choice}} + 0.5 \cdot L_{\text{score}}
$$

**这个公式在做什么**：四项 loss 的加权和。DiT 分支贡献 MSE+FFT，Choice Head 贡献 L1+Score。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 项 | 权重 | 信号来源 | 梯度流向 |
|----|------|---------|---------|
| $L_{\text{MSE}}$ | 0.5 | DiT 预测速度 vs 真实速度 | DiT + VLM（通过 KV-Cache） |
| $L_{\text{FFT}}$ | 1.0 | DiT 预测频谱 vs 真实频谱 | DiT + VLM |
| $L_{\text{choice}}$ | 0.5 | 最佳候选 vs GT（L1） | Choice Projector + VLM |
| $L_{\text{score}}$ | 0.5 | 评分预测 vs 真实误差 | Score Projector（不传到 Action） |

**数值代入**：假设某 batch 的各项 loss = [0.02, 0.01, 0.08, 0.0002]：
- $L_{\text{total}} = 0.5 \times 0.02 + 1.0 \times 0.01 + 0.5 \times 0.08 + 0.5 \times 0.0002$
- $= 0.01 + 0.01 + 0.04 + 0.0001 = 0.0601$

**为什么 FFT 权重是 1.0（最大）**：频率域 loss 的数值通常比 MSE 小（因为是频谱差的绝对值），所以给更高权重来平衡贡献。
:::

## 8. 本章小结

Choice Head 是 XR-1 相对 XR-0 的一个全新设计：

1. **动机**：解决多模态动作分布下单一输出模糊的问题
2. **实现**：VLM 的特殊 token 位置 → MLP → 5 组候选 + 5 个评分
3. **训练策略**：Winner-Takes-All + Score 回归
4. **和 DiT 的关系**：互补而非替代——DiT 主导动作生成，Choice Head 提供辅助信号
5. **几乎零额外推理成本**：候选由 VLM 前向附带产生

---

**下一章预告**：[Ch06 Rectified Flow + 频率域 Loss](./06_RectifiedFlow与频率域Loss) 将完整拆解 DiT 的训练目标——Beta 时间步采样为什么偏向高噪声、频率域 Loss 具体怎么算、两者如何互补。
