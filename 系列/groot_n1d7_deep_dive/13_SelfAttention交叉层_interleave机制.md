---
title: "Self-Attention 交叉层：interleave_self_attention 机制"
series:
  id: groot_n1d7_deep_dive
  chapter: 13
order: 13
---

# Self-Attention 交叉层：interleave_self_attention 机制

> 为什么 GR00T 把 Self-Attention 和 Cross-Attention 放在不同层、交替出现，而不是在同一层中先后执行？这种设计对信息传播和计算效率意味着什么？

## 相关阅读

- [AlternateVLDiT](./12_AlternateVLDiT_交替注意力设计)（上一章）
- [DiT 输出层](./14_DiT输出层_AdaLN_Zero调制)（下一章）
- [Cross-Attention 与交替注意力](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制)

---

## 前情提要

上一章我们深入分析了 AlternateVLDiT 如何在 Cross-Attention 层内交替关注图像和文本。
本章聚焦另一个交替维度——为什么 Self-Attention 和 Cross-Attention 要在**不同层**中出现？

---

## 1. 两种设计方案对比

### 1.1 方案 A：标准 Encoder-Decoder Block（同一层内先后执行）

这是 GPT-2 Decoder、原始 Transformer Decoder 的做法：

```
每一层:
  hidden → Self-Attention → +残差 → Cross-Attention → +残差 → FFN → +残差 → 输出
```

每一层**同时包含** Self-Attn 和 Cross-Attn。一层就做两次注意力计算。

### 1.2 方案 B：Interleaved（GR00T 的做法）

```
偶数层: hidden → Cross-Attention → +残差 → FFN → +残差 → 输出
奇数层: hidden → Self-Attention  → +残差 → FFN → +残差 → 输出
```

每一层**只做一种**注意力。Self 和 Cross 交替出现在不同层。

### 1.3 两种方案的数学等价性

如果方案 A 有 N 层，方案 B 有 2N 层，它们的表达能力是否等价？

**不完全等价**。关键区别在于 FFN 的位置：

- 方案 A (N=16)：`Self → Cross → FFN`（每层有 1 个 FFN，紧跟在 2 次 attention 后面）
- 方案 B (N=32)：`Cross → FFN` + `Self → FFN`（每层各有 1 个 FFN）

方案 B 的 FFN 数量是方案 A 的 **2 倍**——这意味着更多的非线性变换能力。
在 Self-Attn 和 Cross-Attn 之间多了一个 FFN，让网络能在"获取外部信息后"
和"内部交流前"做一次非线性变换——相当于"消化"新获取的信息。

---

## 2. 为什么 GR00T 选择 Interleaved？

### 2.1 原因一：计算效率

每层只做一次 Attention 意味着：

| | 方案 A (16层) | 方案 B (32层) |
|---|---|---|
| 每层 Attention 次数 | 2 | 1 |
| 每层 FFN 次数 | 1 | 1 |
| 总 Attention 次数 | 32 | 32 |
| 总 FFN 次数 | 16 | **32** |
| 每层峰值显存 | Self + Cross 的激活值 | 只有 1 次 Attention 的激活值 |

方案 B 的峰值显存更低——因为任意时刻只需要存储一次 Attention 的中间值（$QK^T$ 矩阵等），
而方案 A 在一层内要存储两次 Attention 的中间值。

对于 gradient checkpointing，方案 B 也更友好——可以按层为单位做 checkpoint，
粒度更细。

### 2.2 原因二："获取→消化"的节奏

Interleaved 模式创造了一种自然的信息处理节奏：

```
Cross-Attn: 从外部获取新信息（"我看到红色方块在左边"）
    ↓ FFN: 初步变换
Self-Attn:  内部消化（action tokens 之间分享这个信息："既然目标在左边，前几步往左移"）
    ↓ FFN: 进一步变换
Cross-Attn: 基于已有理解，从外部获取更精细的信息
    ↓ ...
```

这种"获取→消化→获取→消化"的节奏，让每一次外部信息获取都是在
充分整合了前一次信息的基础上进行的。模型不会"贪婪地一次性塞太多外部信息"。

### 2.3 原因三：对 AlternateVLDiT 的必要性

AlternateVLDiT 强制要求 `interleave_self_attention=True`：

```python
assert self.config.interleave_self_attention, "Interleave self attention must be enabled"
```

为什么是强制的？因为如果没有 Self-Attention 层：
- 文本 Cross-Attn 层获取了文本信息
- 紧接着图像 Cross-Attn 层获取了图像信息
- 但**两种信息从未在 token 之间传播**！

考虑一个场景："把红色方块放到蓝色盘子上"
- 文本层告诉 action token："目标是红色方块"
- 图像层告诉 action token："红色方块在坐标(0.3, 0.5)"
- 但如果没有 Self-Attn：第 5 步的 action token 不知道第 1 步的 action token
  已经开始往(0.3, 0.5)移动了→各步动作缺乏协调

Self-Attention 层让所有 41 个 token（1 state + 40 action）互相可见，
使得信息在空间（不同 token）和时间（不同动作步）上传播。

---

## 3. Self-Attention 层的具体行为

### 3.1 Self-Attention Block 的 forward

当 `encoder_hidden_states=None` 时，`BasicTransformerBlock` 自动退化为 Self-Attention：

```python
# block.forward 中的注意力计算：
if encoder_hidden_states is not None:
    # Cross-Attention: Q 来自 hidden_states, K/V 来自 encoder_hidden_states
    attn_output = self.attn1(norm_h, encoder_hidden_states=encoder_hidden_states, ...)
else:
    # Self-Attention: Q/K/V 全部来自 hidden_states
    attn_output = self.attn1(norm_h, encoder_hidden_states=None, ...)
    # 当 encoder_hidden_states=None 时，Attention 类内部自动用 hidden_states 作为 K/V
```

### 3.2 Self-Attention 中的信息流

hidden_states 的结构是 `[state_token, action_token_0, action_token_1, ..., action_token_39]`。

Self-Attention 让这 41 个 token 互相看到彼此：

```
state_token  ←→ action_token_0 ←→ action_token_1 ←→ ... ←→ action_token_39
     ↕                ↕                ↕                          ↕
  所有 token 都能 attend 到所有其他 token (全连接)
```

**没有 causal mask**——这不是自回归生成，所有 token 双向可见。（如果你不清楚 causal mask 是什么、为什么有些模型需要它，请先阅读 [Causal Attention：因果注意力掩码](/前置知识/001g_前置知识_Causal_Attention因果注意力掩码)）
action_token_39 可以看到 action_token_0，反之亦然。

这很关键：它让模型能做**全局规划**——最后一步的动作可以考虑第一步的起始状态，
第一步的动作也可以考虑最后一步的目标位置。

### 3.3 为什么 Self-Attention 也需要 AdaLayerNorm？

Self-Attention Block 的归一化层同样是 AdaLayerNorm（接收 temb）。

为什么？因为内部信息整合的方式也应该随时间步变化：
- $t=0$（纯噪声）：action tokens 之间的"协调"应该更宽松（因为当前状态很不确定）
- $t=0.75$（接近干净）：action tokens 之间应该做更精细的时序约束（因为轨迹已基本成形）

通过 AdaLN，Self-Attention 层也能适应不同的去噪阶段。

---

## 4. Interleaved 模式下 32 层的完整构建

让我们看 DiT 的 `__init__` 中如何构建 32 个 Block，理解偶数层和奇数层的区别：

构建逻辑的核心思路是：遍历 32 层，根据层号奇偶决定是否给该 Block 配置 `cross_attention_dim`。如果配置了，这个 Block 就是 Cross-Attention Block；如果设为 None，就是 Self-Attention Block。

```python
# DiT.__init__ 中的 Block 构建逻辑
all_blocks = []
for idx in range(num_layers):  # num_layers = 32
    # 判断当前层是否是 Self-Attention 层
    use_self_attn = (idx % 2 == 1) and interleave_self_attention
    # 如果是 Self-Attn 层，cross_attention_dim 设为 None
    curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None
    
    all_blocks.append(BasicTransformerBlock(
        self.inner_dim,                    # 1536
        num_attention_heads,               # 32
        attention_head_dim,                # 48
        cross_attention_dim=curr_cross_attention_dim,  # 2048 或 None
        norm_type="ada_norm",              # 所有层都用 AdaLayerNorm
        # ... 其他参数 ...
    ))
```

关键点：
- `idx % 2 == 1`（奇数层）且 `interleave_self_attention=True` → Self-Attention
- 否则 → Cross-Attention（`cross_attention_dim=2048`，即骨干输出维度）
- 两种 Block 共享相同的 `dim=1536`、`num_heads=32` 等参数
- 唯一区别是 `cross_attention_dim` 是否为 None

---

## 5. 信息流的完整追踪

用一个简化的 4 层例子展示信息如何在 Interleaved 模式中流动：

```
初始 x₀ = [state(当前关节角), action₀(噪声), action₁(噪声), ...]

层0 (Cross-Attn → VL特征):
  state token 从 VL 中了解到 "图中有红色方块在(0.3, 0.5)"
  action tokens 从 VL 中了解到场景信息
  x₁ = x₀ + cross_attn_info

层1 (Self-Attn):
  state token 把"目标在(0.3, 0.5)"分享给所有 action tokens
  action tokens 之间开始协调："既然目标在左边，我们一起往左走"
  x₂ = x₁ + self_attn_info

层2 (Cross-Attn → VL特征):
  现在 action tokens 已经有了大致方向，再从 VL 获取更精细的信息
  "更具体地说，方块的高度是 5cm，需要从上方接近"
  x₃ = x₂ + more_cross_attn_info

层3 (Self-Attn):
  action tokens 精炼轨迹："第0步先抬高，第10步开始下降，第30步闭合夹爪"
  x₄ = x₃ + self_attn_refinement
```

每一层的 Self-Attention 都是"集体会议"——所有 token 坐在一起讨论，
基于从 Cross-Attention 获得的最新外部信息，重新协调彼此的状态。

---

## 6. 总结

`interleave_self_attention` 的设计价值：

| 价值 | 具体体现 |
|------|---------|
| 计算效率 | 每层只做一次 Attention，峰值显存更低 |
| 信息消化 | 外部信息获取后有专门的 Self-Attn 层做内部整合 |
| 全局协调 | Self-Attn 让 40 个 action token 互相可见、互相协调 |
| 对 AlternateVLDiT 的支撑 | 没有 Self-Attn 层，交替的图像/文本信息无法在 token 间传播 |
| 更多 FFN | 32 层各一个 FFN = 32 个非线性变换层，表达能力更强 |

这不是一个随意的设计选择——它是整个 AlternateVLDiT 体系的**必要支撑**。
如果关掉 interleave，AlternateVLDiT 的交替注意力策略就失去了意义——
因为不同 token 之间没有通道来传播从外部获取的信息。

---

## 下一章预告

下一章我们将分析 DiT 的最终输出层——AdaLN-Zero 调制和线性投影。
这是时间步条件信息最后一次注入的机会，也是决定输出"哪些维度活跃、哪些维度安静"
的关键环节。
