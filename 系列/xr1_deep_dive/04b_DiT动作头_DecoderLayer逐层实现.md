---
title: "36 层 DiT 动作头（下）：DecoderLayer 逐层实现"
series:
  id: xr1_deep_dive
  chapter: 5
order: 5
---

# 36 层 DiT 动作头（下）：DecoderLayer 逐层实现

> **前情提要**：上一章建立了 DiT 动作头的全局理解——输入构造、信号流方向、与 VLM 的 MoT 耦合关系。本章深入每个组件的代码实现：TimestepEmbedder 的正弦编码、AdaLN-Zero 的调制细节、Attention 内部的 QK-Norm + RoPE + KV 拼接机制、SwiGLU MLP、以及异步模式下的位置编码特殊处理。

**知识链接**：
- 前置知识：[DiT：Diffusion Transformer 架构](/前置知识/002x_前置知识_DiT_Diffusion_Transformer架构)、[AdaLayerNorm 条件化归一化](/前置知识/001f_前置知识_AdaLayerNorm条件化归一化)、[分组查询注意力 GQA](/前置知识/002l_前置知识_分组查询注意力GQA)、[RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码)、[KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码)
- 上一章：[DiT 动作头（上）：整体架构与信号流](./04_DiT动作头_整体架构与信号流)
- 前代对照：[XR-0 DiT 动作头](/系列/xr0_deep_dive/04_DiT动作头架构_AdaLN与GQA跨注意力)

---

## 1. TimestepEmbedder：把标量 $t$ 变成高维向量

### 1.1 设计思路

时间步 $t \in [0, 1)$ 是一个标量，但 DiT 需要一个 1024 维的向量来驱动 AdaLN 调制。这个"标量→向量"的编码分两步：

1. **正弦频率编码**：用不同频率的 sin/cos 把标量映射到 256 维（和 Transformer 的位置编码原理相同）
2. **MLP 投影**：非线性映射到 1024 维

### 1.2 实现

核心思路是：用不同频率的正弦函数"探测"时间步的不同精度层级——低频分量捕捉"早期 vs 晚期"的粗略区别，高频分量捕捉相邻时间步之间的微小差异。

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),  # 256 → 1024
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),              # 1024 → 1024
        )

    def forward(self, timestep):
        # timestep 传入时已经乘以了 1000：t * 1000 → 范围 [0, 1000)
        half = 128  # frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device) / half
        )  # 128 个从高频到低频的频率
        args = timestep[:, None] * frequencies[None]  # [B, 128]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 256]
        return self.mlp(embedding)[:, None]  # [B, 1, 1024]
```

### 1.3 正弦频率编码的数学

$$
\text{freq}_i = \exp\left(-\frac{\log(10000) \cdot i}{128}\right), \quad i = 0, 1, \ldots, 127
$$

**这个公式在做什么**：构造 128 个从高到低的频率值（从 1.0 到 0.0001），用于后续把标量 $t$ 编码成多分辨率的 cos/sin 向量。

::: details 📐 逐符号拆解 + 数值代入（点击展开）

**完整编码公式**（将 $t$ 用这些频率展开为 256 维向量）：

$$
\text{embed}(t) = [\cos(t \cdot \text{freq}_0), \ldots, \cos(t \cdot \text{freq}_{127}), \sin(t \cdot \text{freq}_0), \ldots, \sin(t \cdot \text{freq}_{127})]
$$
**逐符号拆解**：

| 符号 | 含义 | 数值范围 |
|------|------|---------|
| $\text{freq}_0$ | 最高频率 | $e^0 = 1.0$（$t$ 变化 1 就完成一个完整周期） |
| $\text{freq}_{127}$ | 最低频率 | $e^{-\log(10000)} = 1/10000 = 0.0001$ |
| $t$ | 输入时间步（已 ×1000） | [0, 1000) |
| $\cos(t \cdot \text{freq}_i)$ | 第 $i$ 个余弦分量 | [-1, 1] |

**数值代入**（$t=0.4$，乘以 1000 后 $t'=400$）：

- $\text{freq}_0 = 1.0$：$\cos(400 \times 1.0) = \cos(400) \approx -0.85$
- $\text{freq}_{64} = e^{-\log(10000) \times 64/128} = e^{-4.6} \approx 0.01$：$\cos(400 \times 0.01) = \cos(4) \approx -0.65$
- $\text{freq}_{127} = 0.0001$：$\cos(400 \times 0.0001) = \cos(0.04) \approx 1.00$

高频分量（$\text{freq}_0$）对 $t$ 的微小变化剧烈波动——能区分 $t=0.400$ 和 $t=0.401$。低频分量（$\text{freq}_{127}$）几乎不动——只在 $t$ 跨越很大范围时才有明显变化。两者结合，让编码向量对任意两个不同的 $t$ 都能产生可区分的表示。

**为什么乘以 1000**：如果直接用 $t \in [0,1]$，则 $t \cdot \text{freq}_0 = t \in [0,1]$，cos 在 [0,1] 范围内变化极小（$\cos(0)=1, \cos(1)=0.54$），高频分量几乎失效。乘以 1000 后，$t' \cdot \text{freq}_0 \in [0, 1000]$，cos 经历多个完整周期，区分度大幅提升。
:::

### 1.4 从 t_embed 到 6 组调制参数

TimestepEmbedder 输出 $[B, 1, 1024]$ 的 embedding 后，经过 `t_projector` 投影到 $[B, 1, 6144]$，reshape 为 $[B, 6, 1024]$：

```python
# 完整链路
t_embed = self.t_embedder(t[:, 0, 0] * 1000)       # [B, 1, 1024]
t_modulate = self.t_projector(t_embed)               # [B, 1, 6144]
t_modulate = t_modulate.view(-1, 6, self.dit_hidden_size)  # [B, 6, 1024]
```

`t_projector` 是一个 2 层 MLP（1024 → 1024 → 6144），用 SiLU 激活。这个投影把 1024 维的时间步信息"展开"成 6 组独立的调制信号，送给每一层的 DecoderLayer。

---

## 2. DecoderLayer：完整前向传播

### 2.1 结构定义

每层 DecoderLayer 包含 4 个可学习组件 + 1 组调制参数：

```python
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=1024):
        self.attn = Attention(hidden_size=hidden_size)
        self.mlp = MLP(hidden_size=hidden_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        # 每层独有的 AdaLN 基准偏置（6 组 × 1024 维）
        self.adaln_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )
```

注意和 XR-0 的区别：这里只有 **2 个 RMSNorm**（`input_layernorm` 和 `post_layernorm`），而 XR-0 有 4 个。

### 2.2 AdaLN-Zero 调制的实现

每层前向传播的第一步是拆解 6 组调制参数：

```python
def forward(self, hidden_states, past_key_values, position_embeds, timestep, attn_mask):
    # timestep: [B, 6, 1024]（来自上面的 t_modulate）
    # adaln_table: [6, 1024]（本层独有）
    
    # 相加 + 拆解为 6 组
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
        self.adaln_table[None] + timestep  # [1,6,1024] + [B,6,1024] → [B,6,1024]
    ).chunk(6, dim=1)  # 每个 [B, 1, 1024]
```

`adaln_table[None]` 广播到 batch 维度后与 `timestep` 逐元素相加——每层的 `adaln_table` 提供**层特异性偏置**，让不同层即使收到相同的时间步信号，也能产生不同的调制效果。

### 2.3 Attention 子层（带调制）

```python
    # Attention 子层
    residual = hidden_states
    hidden_states = modulate(
        self.input_layernorm(hidden_states),  # 先 RMSNorm
        shift_attn, scale_attn               # 再 AdaLN 调制
    )
    hidden_states = residual + gate_attn * self.attn(
        hidden_states, past_key_values, position_embeds, attn_mask
    )
```

执行顺序：
1. 保存残差
2. RMSNorm 归一化（拉到统一数值范围）
3. modulate（用时间步的 shift/scale 调整分布）
4. 送入 Attention
5. Attention 输出乘以 gate（控制贡献强度）
6. 加回残差

### 2.4 MLP 子层（带调制）

```python
    # MLP 子层
    residual = hidden_states
    hidden_states = modulate(
        self.post_layernorm(hidden_states),
        shift_mlp, scale_mlp
    )
    return residual + gate_mlp * self.mlp(hidden_states)
```

结构完全对称：RMSNorm → Modulate → MLP → Gate × Output → + 残差。

### 2.5 `modulate` 函数

```python
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
```

$$
\text{modulate}(x, \beta, \gamma) = x \cdot (1 + \gamma) + \beta
$$

**这个公式在做什么**：对归一化后的特征做仿射变换——scale 控制"对比度"，shift 控制"中心位置"。`(1+scale)` 的设计让 `scale=0` 时为恒等映射。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 形状 | 含义 |
|------|------|------|
| $x$ | [B, 32, 1024] | RMSNorm 输出（归一化后的隐藏状态） |
| $\gamma$ (scale) | [B, 1, 1024] | 缩放量（广播到序列维度） |
| $\beta$ (shift) | [B, 1, 1024] | 偏移量（广播到序列维度） |

**数值代入**（某一维度）：

- $x = 0.8$（归一化后的值）
- $\gamma = 0.3$（scale，来自时间步调制）
- $\beta = -0.1$（shift）

$$
\text{modulate}(0.8, -0.1, 0.3) = 0.8 \times (1 + 0.3) + (-0.1) = 0.8 \times 1.3 - 0.1 = 0.94
$$

如果 $\gamma = 0, \beta = 0$（训练初期的理想情况）：$0.8 \times 1 + 0 = 0.8$，恒等映射——网络不改变信号。

**为什么是 $(1+\gamma)$ 而不是 $\gamma$**：如果直接用 $\gamma$，初始化时 $\gamma \approx 0$ 会导致 $x \times 0 = 0$——所有信号被抹掉。加 1 后，初始化时 $x \times 1 = x$，信号不受影响，训练过程中 $\gamma$ 逐渐偏离 0 来学习有效的调制。
:::

---

## 3. Attention 模块：QK-Norm + RoPE + KV 拼接

### 3.1 模块定义

```python
class Attention(nn.Module):
    def __init__(self, hidden_size=1024, head_dim=128, kv_heads=8):
        self.num_heads = hidden_size // head_dim  # 1024 // 128 = 8
        self.kv_group = self.num_heads // kv_heads  # 8 // 8 = 1
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)  # → 3072
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(head_dim)  # 对每个头的 128 维做归一化
        self.k_norm = Qwen2RMSNorm(head_dim)
```

关键参数：
- `num_heads = 8`：8 个注意力头
- `head_dim = 128`：每个头 128 维
- `kv_heads = 8`：KV 头数也是 8（所以 `kv_group=1`，退化为标准 MHA，不是真正的 GQA）
- `qkv_proj`：一次性投影出 Q、K、V（$1024 \to 3 \times 1024 = 3072$）

### 3.2 前向传播的 5 个步骤

以下是 Attention forward 的完整逻辑，每一步都有明确的目的：

**Step 1：QKV 投影**

把输入的 1024 维特征投影成 Q、K、V 三组向量，每组 1024 维（8 头 × 128 维/头）：

```python
B, L, _ = hidden_states.shape  # L=32（DiT 序列长度）
qkv = self.qkv_proj(hidden_states)  # [B, 32, 3072]
qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)  # [B, 32, 3, 8, 128]
query, key, value = qkv.unbind(2)  # 各 [B, 32, 8, 128]
```

**Step 2：QK-Norm**

对 Query 和 Key 分别做 RMSNorm——在每个头内部的 128 维上归一化，控制注意力分数的数值范围：

```python
query = self.q_norm(query).transpose(1, 2)  # [B, 8, 32, 128]
key = self.k_norm(key).transpose(1, 2)      # [B, 8, 32, 128]
value = value.transpose(1, 2)               # [B, 8, 32, 128]
```

**为什么需要 QK-Norm**：Attention score = $Q \cdot K^T / \sqrt{d_k}$。如果 Q、K 的数值随训练漂移（某些维度越来越大），softmax 前的 logits 范围也会漂移，导致梯度不稳定。提前归一化让 $||q|| \approx ||k|| \approx \sqrt{d_k}$，确保点积范围可控。这是 Qwen3 系列模型的标准做法。

**Step 3：RoPE 旋转位置编码**

给 Q 和 K 注入位置信息——DiT 内部的 token 之间（Sink/State/Action₀/Action₁/...）需要知道彼此的相对位置：

```python
cos, sin = position_embeds  # 预计算的 RoPE cos/sin
query, key = apply_rotary_pos_emb(query, key, cos, sin)
```

RoPE 的细节见 [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码)。这里的关键点是：DiT 使用**独立的 RoPE 实例**（`self.rotary_emb`），位置 ID 的构造有特殊逻辑（见第 5 节）。

**Step 4：拼接 VLM KV-Cache**

这是整个 Attention 最核心的一步——把 VLM 对应层的 KV-Cache 拼接到 DiT 自己的 KV 前面：

```python
cache_key, cache_value = past_key_values  # VLM 第 i 层的缓存
# repeat_batch: 如果训练重复因子>1，扩展 batch 维度
cache_key = repeat_kv(repeat_batch(cache_key, B), self.kv_group)
cache_value = repeat_kv(repeat_batch(cache_value, B), self.kv_group)

# 拼接！VLM KV 在前，DiT KV 在后
key = torch.cat([cache_key, key], dim=-2)      # [B, 8, S_vlm+32, 128]
value = torch.cat([cache_value, value], dim=-2)  # [B, 8, S_vlm+32, 128]
```

拼接后，DiT 的 32 个 Query token 能同时 attend to：
- VLM 的 $S_{\text{vlm}}$ 个 token（跨模块条件化信息）
- DiT 自己的 32 个 token（内部自注意力）

**Step 5：Scaled Dot-Product Attention + 输出投影**

```python
output = F.scaled_dot_product_attention(
    query, key, value, attn_mask=attn_mask
)  # [B, 8, 32, 128]
output = output.transpose(1, 2).contiguous().view(B, L, -1)  # [B, 32, 1024]
return self.o_proj(output)  # [B, 32, 1024]
```

PyTorch 的 `scaled_dot_product_attention` 自动使用 Flash Attention 2（如果硬件支持），实现高效的 $O(n \cdot m)$ 注意力计算。

### 3.3 `repeat_kv` 的作用

```python
def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states  # XR-1 配置下直接返回（8:8 = 1:1）
    # 如果 kv_group > 1（GQA 模式），把 KV 头复制 n_rep 次匹配 Q 头数
    ...
```

XR-1 当前配置下 `kv_group=1`，所以 `repeat_kv` 实际上是 no-op。但代码保留了 GQA 的通用实现——如果未来某个配置增大 `num_heads`（比如 16 头）而保持 `kv_heads=8`，就会真正触发分组复制。

---

## 4. SwiGLU MLP：门控前馈网络

### 4.1 设计思路

标准 Transformer 的 FFN 是 `Linear → ReLU → Linear`。SwiGLU 改进为**门控结构**：用一个"门"来控制信息是否通过，而不是简单地让 ReLU 截断负值。

### 4.2 实现

```python
class MLP(nn.Module):
    def __init__(self, hidden_size=1024):
        intermediate_size = hidden_size * 4  # = 4096
        self.gate_proj = nn.Linear(1024, 4096, bias=False)  # 门控投影
        self.up_proj = nn.Linear(1024, 4096, bias=False)    # 值投影
        self.down_proj = nn.Linear(4096, 1024, bias=False)  # 下投影
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # gate_proj 决定"开关"，up_proj 决定"信号强度"
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 4.3 数学表示

$$
\text{MLP}(x) = W_{\text{down}} \cdot [\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)]
$$

**这个公式在做什么**：用 `gate_proj` 产生"门信号"（经 SiLU 激活后在 0~∞ 之间），用 `up_proj` 产生"候选值"，两者逐元素相乘实现"门控"——门打开的维度信号通过，门关闭的维度信号被压制。最后 `down_proj` 把 4096 维压回 1024 维。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 形状 | 含义 |
|------|------|------|
| $x$ | [B, 32, 1024] | MLP 输入（经 AdaLN 调制后的特征） |
| $W_{\text{gate}}$ | [4096, 1024] | 门控权重，决定哪些维度"打开" |
| $W_{\text{up}}$ | [4096, 1024] | 值权重，决定"通过门"的内容 |
| $W_{\text{down}}$ | [1024, 4096] | 下投影，恢复到原始维度 |
| $\text{SiLU}(z) = z \cdot \sigma(z)$ | — | Sigmoid Linear Unit：平滑的门控激活 |
| $\odot$ | — | 逐元素乘法（Hadamard 积） |

**数值代入**（跟踪单个维度）：

假设 $x$ 的某一维 = 0.5：
- `gate_proj` 的对应输出 = 1.2
- `up_proj` 的对应输出 = 0.8
- SiLU(1.2) = 1.2 × σ(1.2) = 1.2 × 0.769 = **0.923**（门几乎全开）
- 门控输出 = 0.923 × 0.8 = **0.738**（信号通过）

另一个维度：
- `gate_proj` 输出 = -2.0
- `up_proj` 输出 = 1.5
- SiLU(-2.0) = -2.0 × σ(-2.0) = -2.0 × 0.119 = **-0.238**（门几乎关闭）
- 门控输出 = -0.238 × 1.5 = **-0.357**（信号被大幅压制）

**为什么用 SwiGLU 而不是 ReLU**：
1. ReLU 硬性截断负值 → "神经元死亡"问题（一旦输出为负就永远不更新）
2. SiLU 是平滑函数，所有值都有梯度
3. 门控机制让网络能学会"选择性地通过信息"，比简单的非线性更有表达力
4. 同等参数量下，SwiGLU 在各种 benchmark 上一致优于 ReLU/GELU MLP（实验验证）
:::

---

## 5. 位置编码的特殊处理

### 5.1 DiT 有独立的 RoPE

DiT 不复用 VLM 的位置编码，而是有**自己独立的 RoPE 实例**：

```python
self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=self.config)
```

为什么需要独立的位置编码？因为 DiT 的序列和 VLM 的序列是完全不同的——VLM 处理的是"图像 patch + 文本 token"，DiT 处理的是"Sink + State + Action"。两者的位置含义完全不同，不能共用。

### 5.2 位置 ID 的构造

DiT token 的位置 ID 不是简单的 0, 1, 2, ...，而是**从 VLM 最大位置之后开始编号**：

```python
# VLM 的最大位置（比如 VLM 有 500 个 token，最大位置=499）
max_vlm_pos = position_ids.max(dim=-1).values  # [3, B]

# DiT 的位置 = VLM 最大位置 + 1, +2, +3, ...
dit_position_ids = (
    torch.arange(query_length).view(1, 1, -1).repeat(3, batch_size, 1)
    + max_vlm_pos[..., None]
    + 1
)
# 形状 [3, B, 32]（3维是因为 MRoPE 的 temporal/height/width）
```

**为什么从 VLM 位置之后开始**：虽然 DiT 和 VLM 用不同的 RoPE 实例，但两者的注意力通过 KV 拼接是连通的。如果 DiT 位置从 0 开始，RoPE 计算的相对位置会出现歧义（DiT token 0 和 VLM token 0 会被认为"位置相同"）。从 VLM 最大位置之后开始编号，确保 DiT token 在整个联合序列中有唯一的位置标识。

### 5.3 异步模式的 +10 位置间隔

XR-1 支持异步执行（执行前一轮动作的同时生成下一轮），此时 DiT 的输入分为两部分：
- **前缀动作**（prefix）：上一轮已经确定的动作尾部，提供上下文
- **新生成动作**（suffix）：本轮需要从噪声生成的新动作

代码在两部分之间插入了 **+10 的位置间隔**：

```python
if action_length > prefix_length:
    dit_position_ids[:, :, -(action_length - prefix_length):] += 10
```

**为什么加 10**：这个间隔让 RoPE 产生一个"位置跳跃"——模型通过相对位置差异学会区分"已确定的过去动作"和"需要生成的未来动作"。10 这个值是超参数，足够大让模型清楚感知到两段之间的分界，又不至于大到让位置编码完全"断裂"。

### 5.4 MRoPE：3 维位置编码

注意 `dit_position_ids` 的第一维是 3——这是 Qwen3-VL 的 **Multi-dimensional RoPE (MRoPE)**：

| 维度 | 原始含义（VLM 中） | DiT 中的含义 |
|------|-------------------|-------------|
| 0 | temporal（帧序号） | 退化为统一的时间标记 |
| 1 | height（图像行号） | 退化为统一的时间标记 |
| 2 | width（图像列号） | 退化为统一的时间标记 |

DiT 处理的是 1D 动作序列，不需要 2D/3D 空间信息。但因为复用了 Qwen3-VL 的 RoPE 实现（同一个 `Qwen3VLTextRotaryEmbedding` 类），所以三个维度都填相同的值，等价于普通的 1D RoPE。

---

## 6. 输入/输出投影器

### 6.1 State Projector

机器人状态从 60 维投影到 1024 维——一个 2 层 MLP：

```python
self.state_projector = MLPProjector(
    input_dim=60,       # 双臂(14) + 双手(12) + 底盘(3) + 腰部(1) + ... = 60
    output_dim=1024,
    bias=True
)
# 内部结构：60 → 1024 → SiLU → 1024
```

输入的 60 维包含当前时刻机器人的完整本体感觉（详见 [第 8 章](./08_60维动作空间_数据格式与归一化)）。

### 6.2 Action Projector

带噪动作序列从 (30, 60) 投影到 (30, 1024)：

```python
self.action_projector = MLPProjector(
    input_dim=60,       # 每步动作 60 维
    output_dim=1024,
    bias=True
)
# 对 30 步的每一步独立做同一个投影（参数共享）
noisy_action_embed = self.action_projector(noisy_action * action_mask)  # [B, 30, 1024]
```

**`action_mask` 的作用**：异步模式下，前缀动作是"已知的真实动作"（不带噪声），后缀动作是"从噪声生成的"。`action_mask` 把前缀动作置零，只保留需要去噪的部分。

### 6.3 Action Output Layer

DiT 36 层处理完后，取 Action Token 位置的输出，投影回 60 维：

```python
self.action_output_layer = MLPProjector(
    input_dim=1024,
    output_dim=60,    # 投影回动作空间
    bias=True
)
# 只取 Action Token（跳过 Sink 和 State）
predicted_velocity = self.action_output_layer(hidden_states[:, -action_length:, :])
# [B, 30, 60]
```

这个输出就是 DiT 对速度场 $v_\theta(x_t, t)$ 的预测。训练时和 ground truth $v^* = x_1 - x_0$ 做 MSE Loss。

---

## 7. 完整 DiT 前向传播代码

把所有组件串起来的完整逻辑（简化版）：

```python
class DiT(nn.Module):
    def forward(self, noisy_action, state, timestep, past_key_values, 
                position_ids, attn_mask, action_mask):
        batch_size = noisy_action.shape[0]
        
        # === 1. 时间步编码 ===
        t_embed = self.t_embedder(timestep[:, 0, 0] * 1000)  # [B, 1, 1024]
        t_modulate = self.t_projector(t_embed).view(-1, 6, self.dit_hidden_size)  # [B, 6, 1024]
        
        # === 2. 输入投影 ===
        sink = self.sink.weight[None].repeat(batch_size, 1, 1)       # [B, 1, 1024]
        state_embed = self.state_projector(state)                     # [B, 1, 1024]
        action_embed = self.action_projector(noisy_action * action_mask)  # [B, 30, 1024]
        
        # === 3. 拼接输入序列 ===
        hidden_states = torch.cat([sink, state_embed, action_embed], dim=1)  # [B, 32, 1024]
        
        # === 4. 计算位置编码 ===
        dit_position_ids = self._compute_dit_positions(position_ids, batch_size)
        position_embeds = self.rotary_emb(hidden_states, dit_position_ids)
        
        # === 5. 通过 36 层 DecoderLayer ===
        start_idx = max(0, len(past_key_values) - self.layer_num)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start_idx + i],  # VLM 第 i 层的 KV-Cache
                position_embeds,
                t_modulate,
                attn_mask
            )
        
        # === 6. 提取动作输出 ===
        action_output = hidden_states[:, -action_length:, :]  # 跳过 Sink+State
        return self.action_output_layer(action_output)  # [B, 30, 60]
```

### 7.1 关键实现细节

**`start_idx` 的计算**：

```python
start_idx = max(0, len(past_key_values) - self.layer_num)
```

`past_key_values` 包含 VLM 所有层的 KV-Cache。如果 VLM 有 36 层而 DiT 也有 36 层，则 `start_idx=0`——DiT 的第 0 层用 VLM 第 0 层的 KV。如果是 XR-0（DiT 16 层，VLM 36 层），则 `start_idx=20`——DiT 的第 0 层用 VLM 第 20 层的 KV。

**`repeat_batch` 的作用**：

训练时有一个"重复因子"设计——VLM 跑一次前向产出 KV-Cache，然后 DiT 用这份 KV-Cache 训练多次（不同的 $t$ 和噪声）。此时 DiT 的 batch_size 是 VLM batch_size 的 4 倍，需要把 VLM 的 KV-Cache 重复 4 次来匹配。

---

## 8. Sink Token 的实现细节

### 8.1 定义方式

```python
self.sink = nn.Embedding(1, self.dit_hidden_size)  # 1 个 embedding，1024 维
```

用 `nn.Embedding` 而非 `nn.Parameter` 只是实现偏好——两者效果等价。训练时这个 embedding 通过梯度正常更新。

### 8.2 为什么需要 Sink Token

在 Transformer 的注意力机制中，softmax 要求每个 Query 的注意力权重和为 1。当大部分 Key 都是噪声（训练初期 Action tokens 就是纯噪声），softmax 不得不把权重"分配"给这些无意义的位置——注意力被"稀释"。

Sink Token 提供了一个**有意义的注意力锚点**：
- 它不含噪声（可学习参数，训练后收敛到有意义的值）
- 所有 Action tokens 都能 attend to 它获取"全局基线信息"
- 在去噪早期（大量噪声），Sink 可能吸收大部分注意力权重，防止模型被噪声 token 误导

### 8.3 输出时的处理

DiT 输出时，Sink Token 和 State Token 对应位置的输出被丢弃——只取 Action 位置的输出：

```python
action_output = hidden_states[:, -action_length:, :]  # 只要最后 30 个 token
```

Sink 和 State 在中间层起到"辅助信息聚合"的作用，但最终不产生输出。

---

## 9. 数值验证：跟踪一个 token 穿过一层

为了加深理解，我们跟踪一个具体的 Action Token（维度简化到 $d=4$）穿过一层 DecoderLayer 的完整计算：

**初始状态**：
- 输入 $h = [0.5, -0.2, 0.8, -0.4]$
- 时间步 $t=0.4$，经 TimestepEmbedder + t_projector 得到 6 组参数
- 本层 adaln_table + t_modulate 后得到：
  - shift_attn = $[0.05, -0.02, 0.01, 0.03]$
  - scale_attn = $[0.2, 0.1, -0.1, 0.15]$
  - gate_attn = $[0.7, 0.7, 0.7, 0.7]$
  - shift_mlp = $[0.01, 0.0, -0.01, 0.02]$
  - scale_mlp = $[0.15, 0.05, 0.1, -0.05]$
  - gate_mlp = $[0.6, 0.6, 0.6, 0.6]$

**Attention 子层**：

1. RMSNorm: $\text{rms}(h) = \sqrt{(0.25+0.04+0.64+0.16)/4} = \sqrt{0.2725} = 0.522$
   - $h_{\text{norm}} = h / 0.522 = [0.958, -0.383, 1.533, -0.766]$

2. Modulate: $h' = h_{\text{norm}} \times (1 + \text{scale}) + \text{shift}$
   - $h'_0 = 0.958 \times 1.2 + 0.05 = 1.20$
   - $h'_1 = -0.383 \times 1.1 + (-0.02) = -0.44$
   - $h'_2 = 1.533 \times 0.9 + 0.01 = 1.39$
   - $h'_3 = -0.766 \times 1.15 + 0.03 = -0.85$

3. Attention: 假设输出 $a = [0.3, 0.1, -0.2, 0.15]$

4. Gate + 残差: $h_{\text{out}} = h + 0.7 \times a = [0.5+0.21, -0.2+0.07, 0.8-0.14, -0.4+0.105]$
   - $= [0.71, -0.13, 0.66, -0.295]$

**MLP 子层**（同样的 RMSNorm → Modulate → MLP → Gate + 残差流程）→ 最终输出。

关键观察：gate=0.7 意味着 Attention 只贡献了原始强度的 70%。如果 gate 更小（比如 0.1），Attention 结果几乎不影响残差流——这就是 AdaLN-Zero "渐进式启用子层"的效果。

---

## 10. 本章小结

本章完整实现了 DiT 动作头的每个组件：

| 组件 | 核心实现要点 |
|------|-------------|
| TimestepEmbedder | 正弦频率编码（t×1000）+ 2 层 MLP → 1024d |
| DecoderLayer | RMSNorm → AdaLN Modulate → Attn/MLP → Gate × Output → + 残差 |
| Attention | QKV 投影 → QK-Norm → RoPE → VLM KV 拼接 → Scaled Dot-Product |
| SwiGLU MLP | gate_proj(SiLU) ⊙ up_proj → down_proj |
| 位置编码 | 独立 MRoPE，从 VLM 最大位置 +1 开始，异步模式 +10 间隔 |
| Sink Token | nn.Embedding(1, 1024)，全局注意力锚点 |

所有 36 层结构完全相同，唯一不同是各层的 `adaln_table` 参数值和使用的 VLM KV-Cache 层索引。

---

**下一章预告**：[Ch05 Choice Head](./05_ChoiceHead_多候选动作与评分机制) 将拆解 XR-1 独有的 5 候选动作生成 + 评分排序机制——VLM 除了提供 KV-Cache 之外，还通过特殊的 ACTION/SCORE token 直接参与动作候选的生成和评分。
