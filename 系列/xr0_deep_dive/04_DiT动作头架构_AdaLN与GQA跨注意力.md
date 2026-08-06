---
title: "DiT 动作头架构：AdaLN-Zero 调制与 GQA 跨注意力"
series:
  id: xr0_deep_dive
  chapter: 4
order: 4
---

# 第四章：DiT 动作头架构 —— AdaLN-Zero 调制与 GQA 跨注意力

> 本章目标：逐层拆解 XR0 的 DiT（Diffusion Transformer）动作头，理解每一层内部 Attention、AdaLN 调制、SwiGLU MLP 是怎么组织起来的，以及"跨模块 Cross-Attention"具体怎么实现。

**前情提要**：第 3 章讲完了 VLM 怎么产出 KV-Cache，本章看 DiT 怎么用这份缓存生成动作。

**知识链接**：
- [AdaLayerNorm 条件化归一化](/前置知识/001f_前置知识_AdaLayerNorm条件化归一化) — AdaLN 的基础原理
- [分组查询注意力 GQA](/前置知识/002l_前置知识_分组查询注意力GQA) — DiT Attention 内部使用的注意力变体
- [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码) — DiT 内部同样使用旋转位置编码
- [Cross-Attention 与交替注意力机制](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制) — DiT 对 VLM KV-Cache 的查询本质是 Cross-Attention

---

## 一、DiT 整体结构：16 层同构 DecoderLayer 的堆叠

```python
class DiT(nn.Module):
    def __init__(self, hidden_size=768, layer_num=8, head_dim=128, kv_heads=2):
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads) for _ in range(layer_num)]
        )

    def forward(self, hidden_states, past_key_values, attn_mask, position_embeds, t_embeds):
        start_idx = max(0, len(past_key_values) - self.layer_num)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, past_key_values[start_idx + i], position_embeds, t_embeds, attn_mask=attn_mask)
        return hidden_states
```

XR0 实际配置是 `hidden_size=1024, layer_num=16, head_dim=128, kv_heads=8`（来自 `dit_hidden_size=1024, dit_num_layers=16`）。每一层结构完全相同，唯一变化的是每层用来做 Cross-Attention 的 VLM KV-Cache 索引不同（第 2 章讲过的层对齐规则）。

## 二、单层 DecoderLayer：三个子模块 + AdaLN 调制

### 2.1 结构总览

```python
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=768, head_dim=64, kv_heads=2):
        self.attn = DiTAttention(hidden_size, head_dim, kv_heads)
        self.mlp = DiTMLP(hidden_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-06)
        self.middle_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-06)
        self.post_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-06)
        self.final_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-06)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
```

每一层有 **4 个 RMSNorm**（比标准 Transformer 多两个）和 **1 个 AdaLN 调制表**。理解这一层的关键是先搞清楚 AdaLN-Zero 具体在调制什么。

### 2.2 AdaLN-Zero：6 组调制参数的来源与作用

**为什么需要这个机制**：DiT 的核心任务是"给定当前的扩散时间步 $t$，判断带噪动作应该往哪个方向修正"。同一套网络参数，在 $t=0$（几乎是纯噪声）和 $t=0.9$（接近真实数据）时应该有不同的行为——网络需要一种机制知道"我现在在整个去噪过程的哪个阶段"。AdaLN 通过让归一化的缩放和偏移参数随时间步动态变化来实现这一点，基础原理见 [AdaLayerNorm 条件化归一化](/前置知识/001f_前置知识_AdaLayerNorm条件化归一化)。

XR0 的具体实现里，每一层从共享的时间步 embedding 出发，投影出 6 个 `[B, hidden_size]` 的调制向量：

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaln_table[None] + t_embeds).chunk(6, dim=1)
```

**这一步在做什么**：`t_embeds` 是从时间步 $t$ 经过 `TimestepEmbedder` 和 `t_projector` 计算出的调制参数（形状 `[B, 6, hidden_size]`，具体计算见第三节），`adaln_table` 是这一层自己的一组可学习偏置（形状 `[6, hidden_size]`）。两者相加后按第二维切成 6 段，分别对应 Attention 子层的 shift/scale/gate 和 MLP 子层的 shift/scale/gate。

**逐项拆解**：

| 符号 | 作用位置 | 具体含义 |
|------|---------|---------|
| $\text{shift}_{\text{msa}}, \text{scale}_{\text{msa}}$ | Attention 子层输入前 | 对归一化后的输入做 `x*(1+scale)+shift`，调整 Attention 看到的输入分布 |
| $\text{gate}_{\text{msa}}$ | Attention 子层输出后 | 控制 Attention 结果对残差流的贡献强度（$0$ 表示完全跳过这个子层） |
| $\text{shift}_{\text{mlp}}, \text{scale}_{\text{mlp}}$ | MLP 子层输入前 | 同理，调整 MLP 看到的输入分布 |
| $\text{gate}_{\text{mlp}}$ | MLP 子层输出后 | 控制 MLP 结果对残差流的贡献强度 |

**为什么叫"Zero"**：`adaln_table` 初始化为 `torch.randn(6, hidden_size) / hidden_size**0.5`，是一个较小的随机值而不是严格的 0，但整体设计思路延续了 DiT 论文里 AdaLN-Zero 的核心思想——让 gate 参数在训练初期趋近于让残差分支"什么都不做"，随着训练逐渐学会何时、多大程度上让每个子层生效，这样训练初期更稳定，不会因为随机初始化的子层输出直接大幅扰乱主干信号。

### 2.3 完整前向传播：两个子层 + AdaLN 调制

```python
def forward(self, hidden_states, past_key_values, position_embeds, t_embeds, attn_mask=None):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaln_table[None] + t_embeds).chunk(6, dim=1)

    # Attention 子层
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = modulate(hidden_states, shift_msa, scale_msa)
    hidden_states = self.attn(hidden_states, past_key_values, position_embeds, attn_mask=attn_mask)
    hidden_states = residual + gate_msa * hidden_states
    hidden_states = self.middle_layernorm(hidden_states)

    # FFN 子层
    residual = hidden_states
    hidden_states = self.post_layernorm(hidden_states)
    hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + gate_mlp * hidden_states
    hidden_states = self.final_layernorm(hidden_states)
    return hidden_states
```

其中 `modulate` 函数就是标准的 AdaLN 仿射变换：

$$
\text{modulate}(x, \text{shift}, \text{scale}) = x \cdot (1 + \text{scale}) + \text{shift}
$$

**为什么需要这个公式**：普通 LayerNorm/RMSNorm 把输入拉到一个标准分布后，用**固定的**可学习参数做缩放偏移；AdaLN 则是用**随时间步变化的**缩放偏移参数，让归一化后的分布能根据当前扩散阶段动态调整。

> **一句话**：先把输入"标准化"，再按照当前时间步的需要重新"拉伸和挪动"这个标准化后的分布。

**逐项拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $x$ | RMSNorm 归一化后的隐藏状态 | 形状 `[B, S, D]` |
| $\text{scale}$ | 缩放调制量 | 由时间步 embedding 投影得到，形状 `[B, 1, D]`（广播到序列维度） |
| $\text{shift}$ | 偏移调制量 | 同上 |
| $1+\text{scale}$ | 实际缩放系数 | 加 1 是为了让 $\text{scale}=0$ 时缩放系数恰好是 1（不改变原有分布），训练初期更稳定 |

**数值例子**：取 $D=2$，某个 token 归一化后的值 $x=[1.0, -0.5]$，假设时间步 embedding 投影出 $\text{scale}=[0.2, -0.1]$，$\text{shift}=[0.05, 0.0]$：

$$
\text{modulate}(x) = [1.0\times(1+0.2)+0.05,\ -0.5\times(1-0.1)+0.0] = [1.25, -0.45]
$$

不同的时间步会产生不同的 $\text{scale}, \text{shift}$，所以同一个归一化后的 $x=[1.0,-0.5]$，在 $t=0.1$ 和 $t=0.9$ 时会被调制成不同的实际输入喂给 Attention/MLP——这就是"同一套参数、不同阶段不同行为"的具体实现方式。

**为什么每层有 4 个 LayerNorm**：细心的读者会发现代码里除了标准的 `input_layernorm`（Attention 前）和 `post_layernorm`（MLP 前）之外，还多了 `middle_layernorm`（Attention 后、残差连接后）和 `final_layernorm`（MLP 后、残差连接后）。这是比标准 DiT 更保守的设计——在每次残差相加之后再做一次归一化，进一步稳定深层网络（16 层）在训练时的数值范围，避免残差流随着层数累积而数值发散。

## 三、时间步 Embedding：从标量 $t$ 到 6 组调制参数

上一节提到的 `t_embeds`（形状 `[B, 6, hidden_size]`）具体是怎么算出来的：

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb[:, None]
```

这是标准的正弦位置编码套路（把连续标量 $t$ 编码成正弦/余弦组合的向量，再过一个小 MLP），和 [标量条件编码](/前置知识/001s_前置知识_标量条件编码_位置编码与时间步嵌入) 里介绍的时间步嵌入方法完全一致。得到 `[B, hidden_size]` 的时间步 embedding 后，再过一个投影层把维度扩展到 $6\times\text{hidden\_size}$：

```python
self.t_projector = MLPProjector(input_dim=self.dit_hidden_size, output_dim=6 * self.dit_hidden_size, bias=True)
...
t_embeds = self.t_embedder(t[:, 0, 0] * 1000)          # [B, 1, hidden_size]
t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)  # [B, 6, hidden_size]
```

**注意这里 `t * 1000`**：Rectified Flow 里时间步 $t \in [0,1]$ 是一个连续小数，但正弦位置编码的频率设计（`max_period=10000`）更适合处理"整数序号"量级的输入。乘以 1000 把 $t$ 映射到 $[0, 1000]$ 的范围，这样不同的 $t$ 值之间的差异能被高频分量充分捕捉到，避免所有 $t$ 值都挤在编码函数的一个很小、变化不明显的区间里。这个技巧在扩散模型的时间步编码里很常见（DDPM 原始实现里时间步本身就是 $0$ 到 $T-1$ 的整数）。

每一层的 `adaln_table` 是该层独有的参数，但所有层**共享同一个** `t_embedder` 和 `t_projector`——时间步信息只需要算一次，然后广播给所有 16 层，每层根据自己的 `adaln_table` 产生该层专属的调制效果。

## 四、DiTAttention：GQA + QK-Norm + 跨模块 Cross-Attention

这是整个 DiT 里最关键的模块，它同时做两件事：（1）动作 token 之间的自注意力（2）动作 token 对 VLM KV-Cache 的跨模块 Cross-Attention——这两者通过"把 VLM 的 K/V 拼接到自己的 K/V 前面"这一个技巧被合并成一次统一的 Attention 计算。

```python
class DiTAttention(nn.Module):
    def __init__(self, hidden_size=768, head_dim=64, kv_heads=2, dropout=0.0):
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(head_dim)
        self.k_norm = Qwen2RMSNorm(head_dim)

    def forward(self, hidden_state, past_key_values, position_embeds, attn_mask=None):
        bsz, q_len, _ = hidden_state.size()
        qkv = self.qkv_proj(hidden_state).view(bsz, q_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states, value_states = [t.transpose(1, 2) for t in (query_states, key_states, value_states)]

        cos, sin = position_embeds
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        k_cache, v_cache = past_key_values
        k_cache, v_cache = repeat_kv(k_cache, self.kv_group), repeat_kv(v_cache, self.kv_group)
        key_states = torch.cat([k_cache, key_states], dim=-2)
        value_states = torch.cat([v_cache, value_states], dim=-2)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask)
        return self.o_proj(attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1))
```

### 4.1 QK-Norm：为什么 Query 和 Key 要单独做归一化

在做完 RoPE 之前，Query 和 Key 各自先过一次 `Qwen2RMSNorm(head_dim)`（对每个头自己的 `head_dim` 维度归一化，不是跨头归一化）。这不是标准 Attention 的必需步骤，而是一种提升训练稳定性的技巧：Attention score 由 $QK^T$ 计算，如果 $Q, K$ 的数值范围随着训练不断漂移（比如某些维度的数值越训练越大），softmax 前的 logits 范围也会跟着漂移，容易导致梯度不稳定。提前把 $Q, K$ 各自归一化到统一的尺度，能让 Attention 的数值行为更可预测。这个技巧在 Qwen 系列模型（前一章看到的 `Qwen3VLTextAttention` 里也有同样的 `q_norm`, `k_norm`）中被广泛采用。

### 4.2 RoPE：动作 token 内部的相对位置编码

DiT 输入序列是 `[sink, state, action_0, action_1, ..., action_29]`，这些 token 之间也需要位置信息——比如动作序列里的第 5 步和第 10 步之间应该有明确的"相对距离"概念。XR0 直接复用了 Qwen3-VL 同款的 `Qwen3VLTextRotaryEmbedding`（第三章介绍过的 RoPE 实现），只是这里作用在 DiT 自己的 Query/Key 上,不涉及图像的空间位置(没有 $h,w$ 维度,退化为纯粹的一维序号 RoPE)。

### 4.3 关键技巧：把 VLM 的 K/V 拼接到 DiT 自己的 K/V 前面

```python
key_states = torch.cat([k_cache, key_states], dim=-2)     # dim=-2 是序列长度维度
value_states = torch.cat([v_cache, value_states], dim=-2)
```

**这一步在做什么**：把 VLM 该层缓存的 Key/Value（序列长度是 VLM 输入的 token 数，记为 $S_{\text{vlm}}$）和 DiT 自己算出的 Key/Value（序列长度是 32，即 sink+state+action）在序列维度上拼接，变成一个长度为 $S_{\text{vlm}}+32$ 的联合 Key/Value 序列。DiT 的 Query（长度仍然是 32）去查询这个联合序列，一次 `scaled_dot_product_attention` 就同时完成了：

- **对 VLM 部分的 Key/Value**：这是标准的 Cross-Attention——DiT token 查询 VLM 已经理解好的视觉-语言信息
- **对 DiT 自己的 Key/Value**：这是标准的 Self-Attention——DiT token 之间互相交流（比如动作 token 需要知道 state token 的当前状态）

用一次拼接把两种不同来源的注意力融合成一次计算，这是一个常见的工程技巧：不需要写两次 Attention 再手动合并结果，直接让 `attn_mask` 去控制"哪些 Query 位置能看到哪些 Key 位置"（VLM 部分通常整段可见，DiT 部分则受局部因果掩码约束，详见 [第 6 章](./06_局部因果掩码_sink_state_action结构)），softmax 会自动在拼接后的整段 Key 上正确地归一化权重。

### 4.4 GQA：为什么 kv_heads 可以比 num_heads 少

```python
self.num_heads = hidden_size // head_dim      # 1024 // 128 = 8
self.kv_group = self.num_heads // kv_heads     # 8 // 8 = 1（XR0 当前配置下退化为标准 MHA）
```

XR0 当前的默认配置 `kv_heads=8` 恰好等于 `num_heads=8`，所以在这个具体超参数设置下退化为标准多头注意力（`kv_group=1`，`repeat_kv` 相当于不做任何重复）。但 `DiTAttention` 的实现是按通用 GQA 形式写的——如果换成更大的 `hidden_size`（比如从其他配置文件加载更大规模的模型变体，`num_heads` 随之变大而 `kv_heads` 保持不变），就会真正触发分组共享，压缩 VLM 侧 KV-Cache 被复制的次数。GQA 的完整原理和显存收益见 [分组查询注意力 GQA](/前置知识/002l_前置知识_分组查询注意力GQA)。

## 五、DiTMLP：标准 SwiGLU

```python
class DiTMLP(nn.Module):
    def __init__(self, hidden_size=768):
        self.intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
```

这和 Qwen3-VL 文本 Decoder 里的 MLP（第三章看到的 `Qwen3VLTextMLP`）结构完全一样——SwiGLU 激活（$\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)$，再投影回原维度），是目前主流大模型 MLP 层的标准配置，比标准 ReLU/GELU MLP 在同等参数量下通常有更好的效果。

## 六、本章小结：一次完整的 DiT 层前向传播

| 输入 | 处理顺序 | 输出 |
|------|---------|------|
| `hidden_states [B,32,1024]` | RMSNorm → AdaLN 调制 → GQA Cross-Attn(拼接 VLM KV) → gate 加权残差 → RMSNorm | Attention 子层输出 |
| Attention 子层输出 | RMSNorm → AdaLN 调制 → SwiGLU MLP → gate 加权残差 → RMSNorm | 该层最终输出，传给下一层 |

**下一章预告**：[第 5 章](./05_RectifiedFlow_直线插值与速度场回归)从 DiT 这个"函数"本身跳出来，看外层的 Rectified Flow 训练算法——为什么用直线插值构造训练目标、为什么用 Beta 分布采样时间步，以及推理时怎么用 5 步 Euler 积分完成生成。
