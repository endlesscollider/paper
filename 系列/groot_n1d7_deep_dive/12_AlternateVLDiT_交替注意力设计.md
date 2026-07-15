---
title: "AlternateVLDiT：交替注意力的设计哲学"
series:
  id: groot_n1d7_deep_dive
  chapter: 12
order: 12
---

# AlternateVLDiT：交替注意力的设计哲学

> GR00T N1.7 的核心架构创新——为什么让不同层交替关注图像和文本 token？`image_mask` 如何工作？这种设计带来了什么效果？

## 相关阅读

- [DiT 架构逐层拆解](./11_DiT架构逐层拆解)（上一章）
- [Self-Attention 交叉层](./13_SelfAttention交叉层_interleave机制)（下一章）
- [Cross-Attention 与交替注意力](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制)

---

## 前情提要

上一章我们拆解了标准 DiT 的每一层结构。本章聚焦 GR00T N1.7 的**核心创新**——
`AlternateVLDiT`。它继承标准 DiT 的所有组件，但改变了 Cross-Attention 层
"看什么"的策略：不再让每一层都看所有 VL token，而是交替只看图像或只看文本。

---

## 1. 标准 DiT 的问题：为什么要改？

### 1.1 回顾标准 DiT 的 Cross-Attention

标准 DiT 中，每个 Cross-Attention 层都 attend 到**完整的** VL 特征序列：

```
encoder_hidden_states = [图像token₁, 图像token₂, ..., 图像token₂₈₀, 文本token₁, ..., 文本token₂₀]
                         ←——————— 280 个图像 token ————————→  ←— 20 个文本 —→
```

所有 300 个 token 一视同仁地参与 softmax 注意力权重计算。

### 1.2 这会导致什么问题？

**问题的本质是 softmax 的"零和博弈"**——注意力权重必须归一化为总和为 1 的概率分布。

具体来说：假设 action token 对所有 300 个 key 计算注意力分数，经过 softmax 后：
- 如果 280 个图像 token 和 20 个文本 token 的注意力分数相近
- 那么图像 token 总共分走约 280/300 = **93%** 的注意力权重
- 文本 token 只分到约 20/300 = **7%** 的注意力权重

即使网络学会了给文本 token 更高的单个分数，在数量上的 14:1 压制仍然是巨大的。

### 1.3 用数值感受差异

假设一个 action token 想要理解"抓红色方块"这个指令：
- 文本中的"红色"token 携带了关键的颜色信息
- 在 300 个 token 中，"红色"只是 1/300
- 即使网络给"红色"分配了最高注意力分数，经过 softmax 后可能只有 5% 的权重
- 这意味着 95% 的输出由其他 token 决定——"红色"的信号被严重稀释

**如果只看 20 个文本 token**：
- "红色"是 1/20，可以分到 20-30% 的注意力权重
- 信号强度提升了 **4-6 倍**

---

## 2. AlternateVLDiT 的解决方案

### 2.1 核心思路

将 Cross-Attention 层分为两类：
- **文本层**：只 attend 到文本 token（通过 mask 屏蔽图像 token）
- **图像层**：只 attend 到图像 token（通过 mask 屏蔽文本 token）

两类层交替出现。由于残差连接的存在，所有信息最终都会在 hidden_states 中累积——
文本层获取的语义信息不会因为后续的图像层而丢失。

### 2.2 调度策略：`attend_text_every_n_blocks`

GR00T 用 `attend_text_every_n_blocks=2` 来控制交替节奏。这个参数的含义是：
"每 2 个 Cross-Attention 块中，有 1 个看文本"。

具体的判断逻辑是：

```
对于 Cross-Attention 层（偶数层 idx = 0, 2, 4, 6, ...）：
  如果 idx % (2 * attend_text_every_n_blocks) == 0 → 看文本
  否则 → 看图像
```

当 `attend_text_every_n_blocks=2` 时，`2 * 2 = 4`，所以：
- idx=0: 0 % 4 == 0 → **文本**
- idx=2: 2 % 4 == 2 → 图像
- idx=4: 4 % 4 == 0 → **文本**
- idx=6: 6 % 4 == 2 → 图像
- idx=8: 8 % 4 == 0 → **文本**
- ...

结果：Cross-Attention 层中，**一半看文本、一半看图像**——完美对称。

### 2.3 完整 32 层的调度表

| 层号 | 类型 | Attend 目标 | 获取的信息 |
|------|------|-----------|-----------|
| 0 | Cross-Attn | 文本 | "要抓红色方块" |
| 1 | Self-Attn | 自身 | action tokens 互相协调 |
| 2 | Cross-Attn | 图像 | "红色方块在桌子左边" |
| 3 | Self-Attn | 自身 | 整合文本+图像信息 |
| 4 | Cross-Attn | 文本 | 更精细地理解"抓"的含义 |
| 5 | Self-Attn | 自身 | 规划动作轨迹 |
| 6 | Cross-Attn | 图像 | 精确定位方块边缘位置 |
| 7 | Self-Attn | 自身 | 时序协调 |
| ... | ... | ... | ... |
| 28 | Cross-Attn | 文本 | 最终确认任务目标 |
| 29 | Self-Attn | 自身 | 最终精炼 |
| 30 | Cross-Attn | 图像 | 最终位置校准 |
| 31 | Self-Attn | 自身 | 最终输出整合 |

16 个 Cross-Attn 层 → 8 个看文本 + 8 个看图像。
尽管图像 token 数量是文本的 14 倍，两者获得了**均等的注意力资源**。

---

## 3. `image_mask` 的工作机制

### 3.1 mask 从哪里来？

`image_mask` 在骨干网络（Qwen3Backbone）的输出中产生：

```python
# 在 Qwen3Backbone.forward() 中：
image_mask = vl_input["input_ids"] == self.model.config.image_token_id
# 结果: [B, seq_len] 的布尔张量，True = 该位置是图像 token
```

Qwen3-VL 在 tokenize 时，图像区域用一个特殊的 `image_token_id` 占位。
通过比较 input_ids 中每个位置是否等于这个 ID，就能得到 image_mask。

### 3.2 如何用 mask 分离图像和文本

AlternateVLDiT 在 forward 中构建两个 attention mask：

```python
# image_mask: [B, seq_len], True = 图像位置
# backbone_attention_mask: [B, seq_len], True = 有效位置 (非padding)

# 图像 mask：只关注图像 token 的位置
image_attention_mask = image_mask & backbone_attention_mask

# 文本 mask：只关注非图像 token 的位置 (即文本+特殊token)
non_image_attention_mask = (~image_mask) & backbone_attention_mask
```

这两个 mask 的关系：
- `image_attention_mask | non_image_attention_mask == backbone_attention_mask`（并集 = 全部有效位置）
- `image_attention_mask & non_image_attention_mask == 0`（交集为空，互不重叠）

### 3.3 mask 在注意力中如何工作

当 mask 被传入 `BasicTransformerBlock` 的 `encoder_attention_mask` 参数时，
内部的 Attention 模块会：

1. 计算完整的 $QK^T$ 矩阵（所有 query 对所有 key 的相似度）
2. 将 mask 为 False 的位置设为 $-\infty$
3. softmax 后这些位置的权重变为 0（$e^{-\infty} = 0$）

效果：被 mask 掉的 token **完全不参与**注意力输出——就像它们不存在一样。

### 3.4 具体例子

假设 VL 特征序列有 10 个 token：`[img, img, img, img, img, img, img, txt, txt, txt]`

```
image_mask           = [T, T, T, T, T, T, T, F, F, F]
non_image_mask       = [F, F, F, F, F, F, F, T, T, T]
backbone_attn_mask   = [T, T, T, T, T, T, T, T, T, T]  (无padding)

文本层的 mask: non_image_attention_mask = [F, F, F, F, F, F, F, T, T, T]
  → action tokens 只能 attend 到位置 7, 8, 9 (文本)
  → 位置 0-6 (图像) 被屏蔽，注意力权重 = 0

图像层的 mask: image_attention_mask = [T, T, T, T, T, T, T, F, F, F]
  → action tokens 只能 attend 到位置 0-6 (图像)
  → 位置 7-9 (文本) 被屏蔽，注意力权重 = 0
```

---

## 4. 为什么信息不会丢失？（再次强调）

一个容易产生的误解："文本层只看文本，那图像信息不就没了吗？"

**答案：不会丢失。因为残差连接。**

每一层的输出 = 输入 + 新信息。"输入"本身已经包含了前面所有层累积的信息。

```
层0 (文本): 输入=x₀ → 输出=x₀+文本信息 = x₁
层2 (图像): 输入=x₂ (包含 x₁ 中的文本信息) → 输出=x₂+图像信息 = x₃
层4 (文本): 输入=x₄ (包含 x₃ 中的文本+图像信息) → 输出=x₄+更多文本 = x₅
```

每一轮文本/图像注意力都是在**已有全部信息的基础上**"追加"新信息。
32 层下来，最终的 hidden_states 中同时包含了 8 轮文本理解 + 8 轮图像理解 + 16 轮内部整合。

而且越深的层能做越精细的推理——第 28 层的"看文本"已经知道了前面
7 轮图像层提供的所有空间信息，可以在此基础上做"文本+空间"的联合推理。

---

## 5. AlternateVLDiT 与标准 DiT 的代码差异

AlternateVLDiT **继承**自 DiT，只覆盖了两个地方：`__init__` 和 `forward`。

### 5.1 初始化的差异

AlternateVLDiT 的 `__init__` 只是多存了一个参数：

```python
class AlternateVLDiT(DiT):
    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)  # 复用 DiT 的所有初始化
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
```

所有的 Block、TimestepEncoder、输出投影层——全部复用 DiT 的代码。
这意味着从标准 DiT 切换到 AlternateVLDiT **不需要重新训练任何权重**——
只是改变了推理时的 attention mask 策略。

### 5.2 forward 的核心差异

标准 DiT 的 forward 对所有 Cross-Attention 层传入 `encoder_attention_mask=None`（不做任何屏蔽）。

AlternateVLDiT 的 forward 根据层号选择不同的 mask。核心逻辑如下：

首先，它从输入中构建两种互补的 mask——一种只保留图像位置，一种只保留文本位置。
然后在 32 层的循环中：奇数层照常做 Self-Attention（不需要 mask）；
偶数层做 Cross-Attention 时，根据层号的规律选择"这一层看图像还是看文本"。

判断规则很简洁：如果当前 Cross-Attention 层的编号能被 `2 * attend_text_every_n_blocks` 整除，
就看文本；否则看图像。

来看实现：

```python
def forward(self, hidden_states, encoder_hidden_states, timestep=None,
            encoder_attention_mask=None, return_all_hidden_states=False,
            image_mask=None, backbone_attention_mask=None):
    
    assert image_mask is not None, "Image mask is required"
    
    # 1. 编码时间步
    temb = self.timestep_encoder(timestep)
    
    # 2. 构建两种互补的 attention mask
    image_attention_mask = image_mask & backbone_attention_mask       # 图像位置
    non_image_attention_mask = (~image_mask) & backbone_attention_mask  # 文本位置
    
    # 3. 确保 interleave 模式开启
    assert self.config.interleave_self_attention
    
    # 4. 逐层处理
    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1:
            # 奇数层: Self-Attention (无需外部 mask)
            hidden_states = block(hidden_states, encoder_hidden_states=None, temb=temb)
        else:
            # 偶数层: Cross-Attention，选择看图像还是文本
            if idx % (2 * self.attend_text_every_n_blocks) == 0:
                curr_mask = non_image_attention_mask  # 这一层看文本
            else:
                curr_mask = image_attention_mask      # 这一层看图像
            
            hidden_states = block(hidden_states,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=curr_mask,
                                  temb=temb)
    
    # 5. 输出投影 (和标准 DiT 完全相同)
    shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
    hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    return self.proj_out_2(hidden_states), all_hidden_states
```

代码中有几个值得注意的设计细节：

1. **`assert image_mask is not None`**：AlternateVLDiT **强制**要求传入 image_mask，
   如果没有就直接报错。这和标准 DiT 不同（标准 DiT 不需要 image_mask）。

2. **`assert self.config.interleave_self_attention`**：AlternateVLDiT 强制要求
   开启 Self-Attention 交叉模式。因为如果不交叉（所有层都是 Cross-Attention），
   就没有"内部整合"的机会——获取的信息无法在 token 之间传播。

3. **encoder_hidden_states 始终传入完整序列**：虽然 mask 屏蔽了部分 token，
   K 和 V 仍然是完整的张量。被 mask 的位置只是在 softmax 后权重为 0，
   不参与加权求和。这比物理切分张量更高效（避免动态 shape 操作）。

---

## 6. 效果分析：交替 vs 不交替

### 6.1 定性分析

| 场景 | 标准 DiT | AlternateVLDiT |
|------|---------|----------------|
| "抓红色方块" | "红色"信号在 300 个 token 中被稀释 | "红色"在 20 个文本 token 中获得充分注意 |
| 多物体场景 | 所有物体的视觉 token 互相竞争 | 图像层专注空间定位，不受文本干扰 |
| 长指令 | 指令 token 被图像 token 数量压制 | 文本层给指令充分的注意力带宽 |
| 精细操作 | 空间定位信号可能被语义信号覆盖 | 图像层专注于像素级精度 |

### 6.2 计算量对比

交替注意力**不增加计算量**——每层仍然只做一次注意力计算，
只是通过 mask 改变了"看哪些 key"。实际上由于 mask 让部分位置的梯度为 0，
反向传播可能略微更快。

---

## 7. 如果修改 `attend_text_every_n_blocks`？

这个参数控制"文本 vs 图像"的注意力资源分配比例：

| 值 | 文本层占比 | 图像层占比 | 适用场景 |
|----|-----------|-----------|---------|
| 1 | 50% | 50% | 复杂指令 + 简单场景 |
| **2 (默认)** | **50%** | **50%** | **通用** |
| 3 | 33% | 67% | 简单指令 + 复杂场景 |
| 4 | 25% | 75% | 密集物体场景 |

等等——值为 1 和值为 2 都是 50:50？

是的！因为判断逻辑是 `idx % (2 * n) == 0`：
- n=1: `idx % 2 == 0` → 所有 Cross-Attn 层都满足 → 全部看文本！这不对。
  实际上 n=1 时代码逻辑会让 `idx=0` 看文本，`idx=2` 看图像...
  
让我重新分析：对于偶数层 idx = 0, 2, 4, 6, 8...
- n=2: `idx % 4 == 0` → idx=0,4,8,12... 看文本，其余看图像 → 50:50
- n=3: `idx % 6 == 0` → idx=0,6,12,18... 看文本，其余看图像 → 33:67
- n=1: `idx % 2 == 0` → 所有偶数层都满足 → **全部看文本**

所以 n=1 是一个极端值（全部看文本，不看图像），实际使用中 n=2 是最合理的默认值。

---

## 8. 总结

AlternateVLDiT 是 GR00T N1.7 相比标准 DiT 的核心创新，关键设计：

1. **问题**：softmax 的零和博弈让少数文本 token 的信号被多数图像 token 淹没
2. **方案**：交替让不同层专注于不同类型的 token，每种类型独享注意力带宽
3. **实现**：通过 `image_mask` 构建互补的 attention mask，零额外计算开销
4. **安全性**：残差连接确保信息不丢失——每一层都在已有全部信息的基础上追加新信息
5. **灵活性**：`attend_text_every_n_blocks` 控制文本/图像的资源分配比例

这个设计体现了一个工程智慧：**有时候"看得少"反而"看得清"**。
通过限制每一层的视野范围，让注意力机制在有限范围内做更精细的信息提取。

---

## 下一章预告

下一章我们将聚焦另一个交替维度——`interleave_self_attention`。
为什么 Self-Attention 和 Cross-Attention 要放在不同层？如果放在同一层内会怎样？
这种设计对信息流和训练稳定性有什么影响？
