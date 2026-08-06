---
title: "Qwen3-VL 骨干：视觉编码、MRoPE 与 KV-Cache 生成"
series:
  id: xr0_deep_dive
  chapter: 3
order: 3
---

# 第三章：Qwen3-VL 骨干 —— 视觉编码、MRoPE 与 KV-Cache 生成

> 本章目标：理解图像怎么被切成 patch、怎么编码空间位置、怎么和文本 token 一起进入 Decoder，并最终产出供 DiT 复用的 KV-Cache。

**前情提要**：第 2 章给出了整体数据流，本章深入 VLM 骨干网络（Qwen3-VL-4B-Instruct）内部，看第一阶段"图像+指令 → KV-Cache"具体是怎么算出来的。

**知识链接**：
- [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码) — MRoPE 是 RoPE 的多模态扩展
- [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码) — 本章末尾产出的正是这份缓存
- [分组查询注意力 GQA](/前置知识/002l_前置知识_分组查询注意力GQA) — Qwen3-VL 文本部分的 Attention 也用 GQA

---

## 一、视觉编码器：从像素到 Patch Token

### 1.1 Patch Embedding：把图像切成方块

Qwen3-VL 的视觉编码器不是逐像素处理图像,而是先把图像切成固定大小的方块(patch),每个 patch 被一个 3D 卷积压缩成一个向量:

```python
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
```

这里用的是 3D 卷积(而不是更常见的 2D 卷积),是因为 Qwen3-VL 的视觉编码器设计上同时兼容图像和视频输入——图像可以看成时间维度长度为 1 的特殊视频。`temporal_patch_size` 控制时间维度上每个 patch 覆盖多少帧,`patch_size` 控制空间维度上每个 patch 覆盖多大的像素方块(通常是 14×14 或 16×16)。

对于 XR0 的场景(每帧任务只用当前时刻的单帧图像),时间维度上退化为长度 1,本质上和标准的 ViT patch embedding 是一样的效果:一张图像被切成 $H/p \times W/p$ 个 patch($p$ 是 patch 边长),每个 patch 变成一个 `embed_dim` 维的向量。

### 1.2 位置编码:两套并行的机制

视觉 token 需要知道"自己在图像里的空间位置"。Qwen3-VL 的视觉编码器用了两套并行的位置编码机制,分别服务不同的目的:

**机制一:绝对位置嵌入(通过插值适配任意分辨率)**

```python
pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
hidden_states = hidden_states + pos_embeds
```

这是一个可学习的位置嵌入表(`self.pos_embed`),固定训练时的网格大小(比如 32×32)。但实际输入图像的分辨率是可变的,patch 网格大小可能是任意的 $h \times w$。`fast_pos_embed_interpolate` 通过双线性插值,把固定大小的位置嵌入表"拉伸"或"压缩"到实际的网格大小上——这类似于把一张固定分辨率的位置编码图,缩放贴合到任意尺寸的输入图像网格上。

**机制二:2D 旋转位置编码(用于 Attention 内部)**

```python
rotary_pos_emb = self.rot_pos_emb(grid_thw)
```

这是视觉 Attention 内部使用的旋转位置编码,和第二节要讲的 MRoPE 是同一套思想在纯视觉场景下的应用——只是这里只需要 2 个维度(行、列),不需要额外的时间维度(因为 XR0 每次只处理单帧图像)。具体的旋转机制原理见 [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码)。

### 1.3 Deepstack:让浅层视觉特征也能影响语言模型

Qwen3-VL 视觉编码器的一个特点是 **Deepstack** 机制:

```python
deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[...](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)
```

**这一步在做什么**:标准做法是只用视觉编码器**最后一层**的输出接入语言模型,但这样会丢失浅层的细粒度视觉信息(比如边缘、纹理这类局部特征,在深层被逐渐抽象掉)。Deepstack 的做法是在视觉编码器的若干个中间层各抽取一份特征(经过对应的 `deepstack_merger_list` 投影),额外注入到语言模型 Decoder 的前几层里(见下面 `Qwen3VLTextModel._deepstack_process`)。

这样语言模型不仅能拿到"高度抽象后的整体场景理解",也能拿到"浅层的局部视觉细节",对需要精细视觉定位的任务(比如机器人抓取需要精确判断物体边界)更友好。

### 1.4 Patch Merger:降低视觉 token 数量

原始 patch 网格通常很密(比如 32×32=1024 个 patch),直接把这么多 token 喂给语言模型开销很大。`Qwen3VLVisionPatchMerger` 把 `spatial_merge_size × spatial_merge_size`(通常是 2×2=4 个)相邻 patch 合并成一个 token:

```python
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)
```

这把 4 个 patch 的特征拼接成一个更宽的向量,再用两层 MLP 压缩投影回目标维度——本质上是一次"下采样",用 4 倍的 token 数量减少换来单个 token 更丰富的语义内容。这也是为什么最终喂给语言模型的图像 token 数量,通常是原始 patch 数量的 1/4。

## 二、MRoPE:多模态位置编码

### 2.1 为什么纯文本的 RoPE 不够用

普通 RoPE 假设位置是一个一维序号(第几个 token)。但图像 token 展开后失去了原本的二维空间结构——如果只用一维序号,模型无法知道"这个 patch 和那个 patch 在图像里是不是相邻"。

MRoPE(Multimodal RoPE)把位置编号从一个整数,扩展成三元组 $(t, h, w)$,详细机制见 [RoPE 旋转位置编码 第四节](/前置知识/002k_前置知识_RoPE旋转位置编码)的"多模态位置编码 MRoPE"一节。这里补充 XR0 场景下的具体应用:

- **文本 token**(指令文字):$t, h, w$ 三个值相等,退化为普通一维 RoPE,和处理纯文本时行为一致
- **图像 token**(某个 patch):$t$ 是这张图像在序列中的整体位置(比如"这是第几张图"),$h, w$ 是这个 patch 在图像网格里的行号、列号

### 2.2 三个位置维度的插值合并:apply_interleaved_mrope

```python
def apply_interleaved_mrope(self, freqs, mrope_section):
    freqs_t = freqs[0]  # 先假设全部按时间维度算
    for dim, offset in enumerate((1, 2), start=1):  # 高度、宽度维度
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t
```

**这一步在做什么**:RoPE 的频率组一共有 `head_dim // 2` 组(参见 [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码)的"从二维扩展到高维"一节),`mrope_section=[24, 20, 20]` 把这些频率组分成三段,分别指定用 $t$、$h$、$w$ 中的哪一个来计算该组的旋转角度。代码里的做法是"以时间维度为底,再用高度、宽度维度的频率去覆盖对应位置的切片",用交错索引(`slice(offset, length, 3)`)让三种维度的频率均匀分布在整个频率谱里,而不是简单地拼接成三段。

这样设计出来的位置编码,同一个 token 的旋转角度同时携带了"第几帧""第几行""第几列"三种信息,让 Attention 分数天然能反映"这两个 patch 在空间上有多接近""这个文字 token 和那个图像 patch 的相对位置关系"这类跨模态的位置结构。

## 三、语言模型 Decoder:处理拼接后的多模态序列

### 3.1 输入序列的构成

图像经过视觉编码器压缩成图像 token 后,和文本指令的 token 拼接成一个统一的序列,一起喂给 36 层的 `Qwen3VLTextDecoderLayer`。XR0 里典型的输入指令模板(见 [第 9 章](./09_数据管线_JSON标注与相对动作计算))形如:

```
<|im_start|>user
The following observations are captured from multiple views.
# Base View
<|vision_start|><|image_pad|><|vision_end|>
# Left-Wrist View
<|vision_start|><|image_pad|><|vision_end|>
Generate robot actions for the task:
把耳机放进收纳盒 /no_cot<|im_end|>
<|im_start|>assistant
<cot></cot><|im_end|>
```

其中 `<|image_pad|>` 会被替换成对应数量的图像 token(取决于该图像 patch 网格切分后的 token 数)。

### 3.2 每一层 Decoder 的标准结构

```python
class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states, position_embeddings, attention_mask, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ..., use_cache=use_cache, ...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

这是标准的 Pre-LayerNorm Transformer Decoder 结构:RMSNorm → Self-Attention → 残差连接 → RMSNorm → SwiGLU MLP → 残差连接。Self-Attention 内部用 GQA(分组查询注意力,原理见 [前置知识](/前置知识/002l_前置知识_分组查询注意力GQA))压缩 KV 头数,并对 Query、Key 做了额外的 RMSNorm(`q_norm`, `k_norm`)——这是一种提升训练稳定性的技巧,避免 Attention score 因为 Query/Key 数值范围漂移而不稳定。

### 3.3 KV-Cache 的产生:use_cache=True 是唯一的开关

关键代码只有一行——XR0 调用 VLM 时传入 `use_cache=True`:

```python
vlm_outputs = self.vlm(**batch, use_cache=True)
past_key_values = list(vlm_outputs.past_key_values)
```

`use_cache=True` 让 `Qwen3VLTextModel` 内部为每一层创建一个 `DynamicCache`,在每一层的 `self_attn` 计算完 Key、Value 后,把它们存进缓存(而不是像标准推理那样在生成完当前 token 后立即丢弃中间结果)。整个 36 层跑完,`past_key_values` 就是一个长度为 36 的列表,每个元素是该层的 `(key, value)` 张量对。

**为什么 XR0 要这样用 KV-Cache**:标准场景下 KV-Cache 是为了"避免自回归生成时重复计算历史 token 的 K/V"(详见 [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码))。而 XR0 这里根本没有做自回归文字生成——它只是借用了"KV-Cache 存储中间计算结果"这个机制,把 VLM 一次前向传播的产出**完整保留**下来,提供给下游完全独立的 DiT 模块去做 Cross-Attention 查询。这是 KV-Cache 概念的一个巧妙的跨模块复用案例。

## 四、本章小结:VLM 阶段的输入输出

| 输入 | 处理 | 输出 |
|------|------|------|
| 3 张多视角图像 | Patch Embed → Deepstack ViT → Patch Merger | 图像 token 序列(每张图像对应若干个 token) |
| 文本指令 | Tokenizer | 文本 token 序列 |
| 拼接后的多模态序列 | 36 层 Decoder(GQA Self-Attn + MRoPE + SwiGLU),`use_cache=True` | 每层的 `(key, value)`,共 36 组 |

这份 36 组 KV-Cache 就是第 2 章数据流图里标注的"过河船票"——它是 VLM 阶段唯一传递给 DiT 阶段的信息通道。

**下一章预告**:[第 4 章](./04_DiT动作头架构_AdaLN与GQA跨注意力)进入动作头内部,看 DiT 的每一层具体怎么用这份 KV-Cache 做 Cross-Attention,以及 AdaLN 调制机制如何让同一套网络参数在不同的扩散时间步表现出不同的行为。
