---
title: "Joint Cross-Modal Attention 逐行拆解"
series:
  id: rynnworld4d_deep_dive
  chapter: 4
order: 4
---

# Joint Cross-Modal Attention 逐行拆解

> **前情提要**：[第 2 章](./02_TriBranch架构总览_从单分支Wan2.2到三分支世界模型)讲了三分支怎么从单分支 Wan2.2 克隆出来，[第 3 章](./03_三阶段渐进式融合_训练策略总览)讲了为什么要用三阶段训练、而不是一步到位打开跨模态注意力。这两章都还没有回答一个具体问题：**Joint Cross-Modal Attention 内部到底怎么算**——`joint_kv_video/depth/flow`、`joint_q_video/depth/flow`、`joint_align_*`、`joint_out_*`、`joint_gate_*`、`modality_embed_*` 这一整套参数，具体怎么组合成一次前向传播。这一章把 `module_joint.py` 里的 `JointRynnWorld4DTransformerBlock` 逐段拆开讲清楚。

## 相关阅读

- [Cross-Attention 与交替注意力机制](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制)——理解 Q 来自一处、K/V 来自另一处这种注意力的基本计算方式
- [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码)——第 5 节讲跨模态 RoPE 时会用到
- [Zero-Init 门控与渐进式模块注入](/前置知识/002n_前置知识_Zero-Init门控与渐进式模块注入)——第 7 节门控计算的设计依据
- [第 2 章：Tri-Branch 架构总览](./02_TriBranch架构总览_从单分支Wan2.2到三分支世界模型)——三分支各自独立的 Self-Attn/FFN 是这一章要接入的对象
- [第 3 章：三阶段渐进式融合](./03_三阶段渐进式融合_训练策略总览)——本章讲的模块具体是在哪个训练阶段被启用、被冻结

## 0. 这个模块插在哪、替代了什么

第 2 章讲过，`RynnWorld4DTransformerBlock` 里每条分支各自跑完 Self-Attention → Cross-Attention（文本条件）→ FFN 之后，三条分支之间除了共享的 `condition_embedder`、RoPE、AdaLN 参数，没有任何直接的信息交换。`fusion_mode="joint"` 对应的类是 `JointRynnWorld4DTransformerBlock`，它在这套流程里插入了一个新步骤——**Self-Attention 做完之后、Cross-Attention 之前**，让三条分支的隐藏状态先互相"看一眼对方现在算到哪了"，再各自继续走文本条件和 FFN。

用一句话概括这个模块要解决的问题：三条分支各自的 Self-Attention 只能看到自己模态内部的 token，不知道另外两条分支这一层算出了什么。Joint Cross-Modal Attention 就是插在这个信息真空里的一次额外的注意力计算，让 depth 分支的每个 token 能去"查询"video 分支和 flow 分支同一层的隐藏状态，反过来也一样（如果不是 unidirectional 模式）。这不是替换掉原有的任何计算，是在原有流程里**多插一步**。

## 1. 共享 KV 设计：为什么是 12 个 Linear，不是 18 个

三条分支要两两互相查询信息，最直接的想法是给每一对分支（video-depth、video-flow、depth-flow）各自设计一套独立的 Cross-Attention，每一对需要一套 Q/K/V/Out 四个 Linear，三对就是 12 个 Linear，但这只覆盖了"从 A 查询 B"这一个方向，如果要双向（A 查 B 和 B 查 A 都算，且参数不共享），就要 18 个 Linear——源码注释里写的正是这个数字对比：

```python
"""
  - New params per block = 3*(Q + K + V + Out) = 12 linear layers (vs 18 in naive design).
"""
```

RynnWorld-4D 用的不是"每一对独立配一套"的方案，而是**每条分支只负责生产一份 K/V，供其余两条分支共用**。具体到代码：

```python
self.joint_kv_video = nn.Linear(dim, dim * 2, bias=True)   # K + V from video
self.joint_kv_depth = nn.Linear(dim, dim * 2, bias=True)   # K + V from depth
self.joint_kv_flow = nn.Linear(dim, dim * 2, bias=True)    # K + V from flow

self.joint_q_video = nn.Linear(dim, dim, bias=True)
self.joint_q_depth = nn.Linear(dim, dim, bias=True)
self.joint_q_flow = nn.Linear(dim, dim, bias=True)
```

**这段代码在做什么**：每条分支各建一个 `nn.Linear(dim, dim*2)` 同时算出 K 和 V（一次线性变换、切成两半），再各建一个 `nn.Linear(dim, dim)` 算自己的 Q。三条分支各自 1 个 KV Linear + 1 个 Q Linear，一共 6 个；后面还有 3 个输出投影 `joint_out_video/depth/flow`，加起来 9 个——再算上代码注释里提到的"每个分支一个"这个计数口径（KV+Q+Out 各 3 个，共 9 个，注释说 12 个是把 K 和 V 拆开单独计数,`3*(Q+K+V+Out)`确实是 12），无论怎么数,核心结论是：**video 分支的这份 K/V，同时供 depth 分支和 flow 分支查询,不需要为"depth 查 video"和"flow 查 video"分别准备两套不同的 K/V**。这正是省下参数的地方——如果每一对查询关系都要自己的 K/V，depth 查 video 和 flow 查 video 用的会是两份不同的 video 特征投影，但实际上"video 分支这一层的隐藏状态里有什么信息"是唯一确定的一件事，没必要让两个不同的"提问者"各自用不同的方式去"翻译"同一份原始信息。一份 K/V、多个提问者共用，这是标准 Multi-Query 式的参数节省思路,应用在跨模态场景下的具体实现。

## 2. `joint_align_*`：为什么要在算 KV/Q 之前先各自归一化一次

三条分支的隐藏状态在这一层之前已经各自走过 Self-Attention 和 AdaLN 调制，数值分布未必一致——RGB 分支承接的是 Wan2.2 预训练权重，激活值的统计特性已经被大规模视频数据"喂"得很稳定；depth、flow 分支是从 RGB 权重克隆初始化、但训练数据的统计特性完全不同（深度图是单调距离场，光流图是位移矢量场），三者的激活值量级可能有明显差异。

```python
self.joint_align_video = nn.LayerNorm(dim, elementwise_affine=True)
self.joint_align_depth = nn.LayerNorm(dim, elementwise_affine=True)
self.joint_align_flow = nn.LayerNorm(dim, elementwise_affine=True)
```

**为什么需要这一步**：跨模态注意力的核心计算是点积（Q·K），如果 video 分支的激活值天生比 depth 分支大出好几倍，跨模态的点积结果会被 video 分支的数值量级主导，注意力权重的分布不再反映"语义上谁和谁更相关"，而是反映"谁的数值本身更大"。`joint_align_*` 是三个各自独立的 `LayerNorm`，在计算 joint KV/Q 之前，各自把自己分支的隐藏状态拉到均值 0、方差 1（再加上各自可学习的缩放/平移）的量级上，确保三条分支进入跨模态注意力时站在同一个数值起跑线上，注意力权重才能真正反映"内容相关性"而不是"谁的数值绝对值更大"。

这一步的动机和 [Zero-Init 门控与渐进式模块注入](/前置知识/002n_前置知识_Zero-Init门控与渐进式模块注入)不是同一个问题——`joint_align` 解决的是"输入到跨模态注意力之前的数值尺度问题"，Zero-Init 解决的是"跨模态注意力的输出加回主干时会不会破坏已有能力"的问题,两者在同一个模块里各自负责不同阶段的稳定性。

## 3. `_joint_cross_modal_attention`：完整计算流程逐段拆解

这是整个模块最核心的方法，接收三条分支当前的隐藏状态，返回三条分支各自应该"吸收"的跨模态信息。先看方法签名和整体骨架，再逐段深入：

```python
def _joint_cross_modal_attention(
    self,
    hidden_states_video, hidden_states_depth, hidden_states_flow,
    num_frames, rotary_emb=None,
):
    hidden_states_video = self.joint_align_video(hidden_states_video)
    hidden_states_depth = self.joint_align_depth(hidden_states_depth)
    hidden_states_flow = self.joint_align_flow(hidden_states_flow)

    kv_v = self.joint_kv_video(hidden_states_video)
    kv_d = self.joint_kv_depth(hidden_states_depth)
    kv_f = self.joint_kv_flow(hidden_states_flow)
    k_v, v_v = kv_v.chunk(2, dim=-1)
    k_d, v_d = kv_d.chunk(2, dim=-1)
    k_f, v_f = kv_f.chunk(2, dim=-1)

    q_video = self.joint_norm_q(self.joint_q_video(hidden_states_video))
    q_depth = self.joint_norm_q(self.joint_q_depth(hidden_states_depth))
    q_flow = self.joint_norm_q(self.joint_q_flow(hidden_states_flow))

    k_v = self.joint_norm_k(k_v)
    k_d = self.joint_norm_k(k_d)
    k_f = self.joint_norm_k(k_f)
```

这一段做的就是第 1、2 节讲过的事情：先各自 `joint_align` 归一化，再各自过 KV Linear 拆成 K/V，再各自过 Q Linear。有个细节值得停下来看——`joint_norm_q` 和 `joint_norm_k` 是**三条分支共用**的两个 `nn.RMSNorm`（不像 `joint_align` 那样各分支各自独立一份）：

```python
self.joint_norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
self.joint_norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
```

这是标准 Transformer 里常见的 QK-Norm 技巧——在算点积之前对 Q、K 各自做一次 RMSNorm，把每个 token 的向量长度归一化掉，只保留方向信息参与点积，这样可以让注意力分数的数值范围更稳定，不会因为某个 token 的 Q 或 K 向量长度异常大就产生过大的点积分数（进而导致 softmax 输出退化成近似 one-hot）。这一步和 `joint_align` 是两个不同粒度的归一化：`joint_align` 是每条分支各自一份、作用在整个隐藏状态向量上；`joint_norm_q/k` 是三分支共享一份、专门作用在算完 Q/K 投影之后的结果上，因为 Q 和 K 的角色（发起查询 vs. 被查询）是跨分支通用的，不需要为每个分支单独学一份 Q/K 的归一化参数。

## 4. 每个分支具体 attend 哪两个分支：非 unidirectional 路径

拿到三条分支各自的 Q、K、V 之后，接下来要决定"谁去查询谁"。默认（`joint_unidirectional=False`）情况下,三条分支两两互相查询，每条分支的 Q 去拼接另外两条分支的 K/V：

```python
# --- Video attends to [depth, flow] ---
k_for_v = torch.cat([k_d, k_f], dim=1)
v_for_v = torch.cat([v_d, v_f], dim=1)
out_v = self._compute_attention(q_video, k_for_v, v_for_v)
out_v = self.joint_out_video(out_v)

# --- Depth attends to [video, flow] ---
k_for_d = torch.cat([k_v, k_f], dim=1)
v_for_d = torch.cat([v_v, v_f], dim=1)
out_d = self._compute_attention(q_depth, k_for_d, v_for_d)
out_d = self.joint_out_depth(out_d)

# --- Flow attends to [video, depth] ---
k_for_f = torch.cat([k_v, k_d], dim=1)
v_for_f = torch.cat([v_v, v_d], dim=1)
out_f = self._compute_attention(q_flow, k_for_f, v_for_f)
out_f = self.joint_out_flow(out_f)
```

**这三段代码在做什么**：每条分支的 Q 都要去"看"**另外两条**分支的 K/V（不看自己），具体做法是先把另外两条分支的 K 沿 token 维拼接（`torch.cat([k_d, k_f], dim=1)`），V 也同样拼接，再送进标准的多头注意力计算，最后过各自的输出投影。

**逐符号拆解**（以 depth 分支为例）：

| 符号 | 数学含义 | 具体是什么 | 典型形状 |
|------|---------|-----------|----------|
| $q_{\text{depth}}$ | depth 分支发起查询的 Q | `joint_q_depth` 投影后再经 `joint_norm_q` 的结果 | $(B, N, \text{dim})$，$N$ 是 token 数 |
| $k_v, k_f$ | video、flow 分支各自的 K | `joint_kv_video/flow` 投影后拆出来、再经 `joint_norm_k` 的结果 | 各 $(B, N, \text{dim})$ |
| $k_{\text{for d}} = [k_v; k_f]$ | 拼接后的 Key 集合 | depth 要查询的对象——video 和 flow 的 K 拼在一起 | $(B, 2N, \text{dim})$ |
| $v_{\text{for d}} = [v_v; v_f]$ | 拼接后的 Value 集合 | 和 K 一一对应的 Value | $(B, 2N, \text{dim})$ |
| $\text{Attn}(q_{\text{depth}}, k_{\text{for d}}, v_{\text{for d}})$ | 标准的 scaled dot-product attention | depth 分支每个 token 会根据自己的 Q，对 video 和 flow 的全部 token 算一次注意力权重，再按权重加权求和 V | 输出形状 $(B, N, \text{dim})$，和 $q_{\text{depth}}$ 一致 |
| $\text{joint\_out\_depth}(\cdot)$ | depth 专属的输出投影 | 把 attention 输出重新映射一次，Zero-Init 初始化（第 6 章会讲这一步的门控） | $(B, N, \text{dim})$ |

**为什么要把 video 和 flow 的 K/V 拼在一起、而不是分两次单独 attend 再相加**：拼接后一次 softmax，意味着 depth 的每个 token 在"该更相信 video 还是更相信 flow"这件事上是**互相竞争**的——如果拼接前做两次独立的 attention 再相加，video 和 flow 各自的注意力权重内部先各自归一化成 1，两边贡献的总"份额"是被人为限制成对等的（各占一半的注意力质量），不管实际内容哪边更相关。拼接之后统一做一次 softmax，如果这个 depth token 在语义上和 video 里某个 token 高度相关、和 flow 里所有 token 都不相关，softmax 自然会把绝大部分权重分给那个 video token，而不是被"必须给 flow 也留一份"这种人为约束限制住。这是让模型自己学会"这一刻该更依赖哪个分支"的关键设计。

**数值代入**：简化到单头、$\text{dim}=2$，$N=1$（每条分支只有一个 token，方便手算）。假设 $q_{\text{depth}} = [1, 0]$，$k_v = [1, 0]$（和 $q_{\text{depth}}$ 方向一致），$k_f = [0, 1]$（和 $q_{\text{depth}}$ 方向垂直），$v_v = [10, 0]$，$v_f = [0, 5]$。点积（忽略缩放因子简化）：$q\cdot k_v = 1$，$q\cdot k_f=0$。Softmax$([1,0]) \approx [0.731, 0.269]$。加权输出：$0.731\times[10,0] + 0.269\times[0,5] = [7.31, 1.35]$——depth 分支这个 token 的跨模态输出主要吸收了 video 分支的信息（权重 0.731），只有一小部分来自 flow（0.269），因为它的 Q 方向和 video 的 K 更接近。如果换成分别做两次 attention 再相加（各自内部归一化成 1）,由于只有一个 K,单头 attention 的 softmax 会恒等于 1,输出会变成 $1\times[10,0]+1\times[0,5]=[10,5]$——两个分支的贡献被强制拉平成相同权重,无法体现"这个 depth token 其实更该信任 video"这件事。这个简化例子说明了拼接统一 softmax 相比分别计算再相加的关键差异。

## 5. `joint_frame_wise`：把跨模态注意力限制在同一帧内

上一节讲的注意力计算是"全局"的——depth 的任意一个 token 都能去 attend video 和 flow 的**所有** token，不管它们分别处在第几帧。这在训练初期（第 3 章讲过 Stage2 会打开 `joint_frame_wise=True`）容易出问题：如果第 5 帧的 depth token 去关注第 20 帧的 video token，这种跨越长时间跨度的注意力对"对齐同一时刻的场景信息"这个目标没有帮助，反而可能引入噪声（比如物体运动导致第 5 帧和第 20 帧的场景完全不同）。

`joint_frame_wise=True` 时的做法是把注意力计算范围收窄到"同一帧内"：

```python
def to_per_frame(x):
    return x.reshape(B, num_frames, S, dim).reshape(B * num_frames, S, dim)

def from_per_frame(x):
    return x.reshape(B, num_frames, S, dim).reshape(B, N, dim)
```

**这段代码在做什么**：把原本形状 $(B, T\times S, \text{dim})$ 的 token 序列（$T$ 是帧数、$S$ 是每帧的 token 数,比如第 2 章例子里 $S=15\times26=390$），重新排列成 $(B\times T, S, \text{dim})$——**把"帧"这个维度从 token 序列里挪出来、并到 batch 维上**。这样一来,后续调用 `_compute_attention` 时，batch 维度变成了 $B\times T$，注意力计算天然只会在每个 $(B\times T)$ 切片内部的 $S$ 个 token 之间进行——因为标准的多头注意力从来不会跨 batch 维度计算，第 $b$ 个样本第 $t$ 帧的 token，永远只会和同样是"第 $b$ 个样本第 $t$ 帧"这个切片里的 token 算注意力。

**代入具体数字**：假设 $B=1$（一个样本），$T=7$（7 个 latent 帧,对应第 2 章例子），$S=390$（每帧 390 个 token），$\text{dim}=3072$。原始张量形状是 $(1, 2730, 3072)$（$7\times390=2730$）。`to_per_frame` 之后变成 $(7, 390, 3072)$——batch 维从 1 变成了 7，每个"batch 切片"精确对应原来的一帧。调用 `_compute_attention` 时,第 0 个切片（第 1 帧）的 390 个 token 只会互相计算注意力权重，第 1 个切片（第 2 帧）的 390 个 token 是完全独立的另一组计算，两者之间不存在任何交叉——这就实现了"帧 $i$ 只关注帧 $i$"的效果。计算完之后 `from_per_frame` 把形状reshape回 $(1, 2730, 3072)$，恢复成原本的 token 序列排列，供后续的 `joint_out_*` 投影和残差加法使用。

`joint_frame_wise=True` 时，$k_v, v_v, k_d, v_d, k_f, v_f$ 以及各分支的 Q 都要先各自过一次 `to_per_frame`，再做第 4 节的拼接和 attention（这也是为什么代码里 `unidirectional` 和非 `unidirectional` 两条路径分别在 `joint_frame_wise` 分支下重复写了一遍——两套逻辑要分别应用 frame-wise 重排）。

## 6. `joint_unidirectional`：让 video 只当 K/V 源，不被修改

第 3 章讲过 `joint_unidirectional=True` 的动机——RGB 分支质量最高，不希望被质量还不稳定的 depth/flow 分支通过跨模态注意力"拖累"。这个设计在 `_joint_cross_modal_attention` 里对应一条单独的代码路径：

```python
if self.joint_unidirectional:
    # --- Depth attends to [video] only ---
    out_d = self._compute_attention(q_depth, k_v, v_v)
    out_d = self.joint_out_depth(out_d)

    # --- Flow attends to [video] only ---
    out_f = self._compute_attention(q_flow, k_v, v_v)
    out_f = self.joint_out_flow(out_f)

    out_v = None
```

和第 4 节的非 unidirectional 路径对比，差异非常直接：depth 和 flow 各自只 attend `k_v, v_v`（video 一家的 K/V），不再像之前那样把两条别的分支拼在一起；而 video 分支这一侧**完全没有对应的计算**——`out_v = None`，video 分支不会产生任何跨模态输出，它的隐藏状态在整个 Joint Attention 步骤里保持不变，只是被"借用"了一次自己的 K/V 供别人查询。

对应到 `forward` 方法里，`joint_unidirectional=True` 时不会给 video 分支加 `modality_embed_video`（因为反正没用），也不会给 video 分支做门控相加（因为 `joint_v` 是 `None`）：

```python
if self.joint_unidirectional:
    hidden_states_depth = hidden_states_depth + self.modality_embed_depth
    hidden_states_flow = hidden_states_flow + self.modality_embed_flow
    joint_v, joint_d, joint_f = self._joint_cross_modal_attention(...)
    hidden_states_depth = hidden_states_depth + joint_d * self.joint_gate_depth.tanh()
    hidden_states_flow = hidden_states_flow + joint_f * self.joint_gate_flow.tanh()
```

这条路径下只有 `hidden_states_depth` 和 `hidden_states_flow` 两行加法，`hidden_states`（video）从头到尾没有出现在这段更新逻辑里——这与第 3 章"video 只提供信息、不接收信息"的设计描述完全对应，这里能看到它具体是怎么在代码层面落地的：不是靠一个门控系数把 video 的更新压到零，而是**从计算图上根本不存在这条更新路径**。

## 7. Modality Embedding：给共享的注意力机制一个"这是哪个模态"的标签

`_joint_cross_modal_attention` 内部的 Q/K/V 投影层（`joint_kv_video`、`joint_q_depth` 等）虽然是每条分支各自独立的，但注意力计算本身（`_compute_attention`）是完全通用、不区分模态的标准多头点积注意力。如果不做任何额外处理，模型在算注意力权重时,唯一能利用的信息就是 Q、K 向量本身的内容——它不知道"现在这个 K 来自哪个分支"这件事本身可能也是一个有用的信号。

```python
self.modality_embed_video = nn.Parameter(torch.zeros(1, 1, dim))
self.modality_embed_depth = nn.Parameter(torch.zeros(1, 1, dim))
self.modality_embed_flow = nn.Parameter(torch.zeros(1, 1, dim))
```

`forward` 里在调用 `_joint_cross_modal_attention` 之前，先把这三个可学习的向量分别加到对应分支的隐藏状态上：

```python
hidden_states = hidden_states + self.modality_embed_video * self.joint_gate_video_decay
hidden_states_depth = hidden_states_depth + self.modality_embed_depth
hidden_states_flow = hidden_states_flow + self.modality_embed_flow
```

**为什么要加这个而不是留给模型自己隐式学**：Q、K、V 的投影层（`joint_kv_video` vs `joint_kv_depth` vs `joint_kv_flow`）本身权重不同，理论上确实已经能让不同模态的 K 落在特征空间的不同区域——但这是一种"隐式"的区分,依赖训练把三套投影权重训得足够不同。加一个显式的、每个模态各自专属、和输入内容无关的可学习偏置向量，相当于在特征空间里给每个模态直接烫上一个固定的"标签坐标"，不需要指望投影权重自己练出可分性，模型可以直接利用这个显式标签去判断"这个 token 来自哪个模态"，再据此调整注意力策略（比如学到"对来自 video 的 K 要给更高权重"这类模态层级的偏好，而不只是 token 内容层级的相关性）。这三个参数初始化为全零（`torch.zeros`），保证了它们和后面第 8 节讲的输出投影 Zero-Init 一样，刚插入时不改变任何行为，靠训练慢慢学出有用的标签值。

## 8. Gate 与 Decay：`forward` 里两次门控相乘的完整链路

前面几节讲完了怎么算出 `joint_v/joint_d/joint_f` 这三个跨模态注意力的输出，最后一步是把它们加回各自分支的隐藏状态——但不是直接加，中间要经过门控缩放。非 unidirectional 路径下 video 分支的更新公式是：

```python
hidden_states = hidden_states + joint_v * self.joint_gate_video.tanh() * self.joint_gate_video_decay
hidden_states_depth = hidden_states_depth + joint_d * self.joint_gate_depth.tanh()
hidden_states_flow = hidden_states_flow + joint_f * self.joint_gate_flow.tanh()
```

**逐符号拆解**：

| 符号 | 含义 | 是否可学习 | 初始值 |
|------|------|-----------|--------|
| `joint_v/d/f` | 第 4/5/6 节算出的跨模态注意力原始输出（已过 `joint_out_*` 投影） | 是（间接，通过 `joint_out_*` 的权重） | `joint_out_*` 权重和 bias 全部清零，所以 `joint_v/d/f` 初始恒为 0 |
| `joint_gate_video/depth/flow` | 每条分支各自的门控标量参数 | 是 | `torch.ones(1)`，初始化为 1，不是 0 |
| `.tanh()` | 把门控值压缩到 $(-1,1)$ | — | $\tanh(1)\approx 0.762$ |
| `joint_gate_video_decay` | 一个**非参数**的 buffer（`register_buffer`，不参与梯度更新） | 否，由训练脚本手动调度 | 初始为 1，Stage3 若开启 `joint_video_decay` 会按余弦调度从 1 衰减到 0 |

这里 `joint_out_*` 清零 + `joint_gate_*` 初始化为 1（不是 0）的组合，正是 [Zero-Init 门控与渐进式模块注入](/前置知识/002n_前置知识_Zero-Init门控与渐进式模块注入)里讲过的方案——`joint_out_*` 的权重清零保证了 Joint Attention 插入瞬间贡献恒为零（$joint_v = joint_out\_video(\text{attn output}) = 0$，因为权重是 0），但门控 $\tanh(1)\approx0.762$ 不为零，保证了反向传播时 `joint_out_video` 权重能立刻拿到非零梯度（梯度里带着这个 $0.762$ 的因子），不会陷入"两个都是零、谁都学不动"的死锁。

`joint_gate_video_decay` 只出现在 video 分支这一行、且只在非 unidirectional 路径下生效——这对应第 3 章提到的一个可选高级选项 `joint_video_decay`：如果 Stage3 训练时把这个开关打开，训练脚本会在每个 step 手动把这个 buffer 从 1 按余弦曲线降到 0（`0.5 * (1 + cos(π * progress))`），效果是让 depth/flow 向 video 的信息注入强度随训练推进逐渐关掉，最终让 video 分支的前向路径退化成和 Stage1（`fusion_mode=none`）完全一致的独立计算，这是为了在训练末期彻底排除跨模态干扰对 RGB 生成质量的任何影响,同时保留 depth/flow 从跨模态注意力里学到的能力。这个机制不是必选项，`joint_unidirectional=True`（第 6 节讲的路径,`joint_v` 恒为 `None`）时根本不会执行到这一行，因为 video 分支从设计上就不接收任何 joint 输出，`joint_gate_video_decay` 这个乘法自然也无从谈起。

## 9. 三条分支之外：`enable_joint` 怎么控制哪些层插入这个模块

最后补一句模型层级的控制逻辑。`JointRynnWorld4DTransformer3DModel` 不是让每一层 Transformer Block 都插入 Joint Attention，而是按区间和步长控制：

```python
enable_joint=(
    (i >= joint_start_layer)
    and (i < joint_end_layer)
    and ((i - joint_start_layer) % joint_every_n_layers == 0)
)
```

**这在做什么**：对第 $i$ 层判断是否启用 Joint Attention——必须落在 `[joint_start_layer, joint_end_layer)` 这个区间内,并且相对起始层的偏移量要整除 `joint_every_n_layers`。第 3 章表格里 Stage2/3 用的是 `joint_start_layer=0, joint_end_layer=30, joint_every_n_layers=3`，代入这个公式，启用的层是第 $0,3,6,9,...,27$ 层，共 10 层（$30/3=10$）。不需要每一层都插入这个模块的原因很直接——跨模态信息交换不需要在每一层都发生，隔几层做一次既能让信息在网络深度方向上逐步扩散、对齐，又能省下额外的参数量和计算量（每多一层 Joint Attention，就多一份第 1 节讲过的 `joint_kv/q/out` 等参数）。

## 10. 小结

这一章把 Joint Cross-Modal Attention 从参数定义到 forward 计算链路走了一遍，核心结论可以归纳成四点：

1. **共享 KV**：每条分支只生产一份 K/V 供别人查询，不是每对分支各自配一套，省了三分之一的参数
2. **两层归一化**：`joint_align_*` 解决三模态数值尺度不一致的问题，`joint_norm_q/k`（QK-Norm）解决点积数值范围不稳定的问题，两者作用的粒度和位置都不一样
3. **注意力范围可以按需收窄**：`joint_frame_wise` 把注意力限制在同一帧内，`joint_unidirectional` 把信息流限制成单向（depth/flow 查 video，video 不查任何人）
4. **Zero-Init 门控保证渐进式接入**：`joint_out_*` 清零 + `joint_gate_*` 初始化为 1，是插入新模块不破坏已有能力、同时不陷入梯度死锁的标准做法；`joint_gate_video_decay` 是这套门控机制之上的一个额外的、手动调度的高级选项

这些设计逐一对应了第 3 章讲过的训练阶段安排——`freeze_non_joint` 冻结的正是本章讲的这一整套 `joint_*` 参数之外的部分，`branch_dropout` 逼着 depth/flow 分支在自己输入被致盲时,必须通过本章讲的这套跨模态查询机制去借力。

## 11. 下一章预告

本章讲清楚了 Joint Cross-Modal Attention **怎么算**，但它每一层输出的 `joint_v/d/f` 最终要服务于一个具体的训练目标——[第 5 章：训练细节](./05_训练细节_FlowMatching目标与分支随机丢弃)会把整个训练 step 的 loss 计算逐行拆开，包括共享噪声、时间步的非线性偏移、首帧条件注入，以及本章第 6 节提到的 branch dropout 具体在训练代码里作用在哪个变量上。

## 知识链接

- [Cross-Attention 与交替注意力机制](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制)——理解 Q/K/V 来自不同来源的标准注意力计算
- [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码)——`joint_use_rope` 让跨模态 Q/K 也带上位置信息的机制基础
- [Zero-Init 门控与渐进式模块注入](/前置知识/002n_前置知识_Zero-Init门控与渐进式模块注入)——第 8 节门控计算的完整设计依据
- [第 2 章：Tri-Branch 架构总览](./02_TriBranch架构总览_从单分支Wan2.2到三分支世界模型)——本章插入的模块所依附的三分支基础结构
- [第 3 章：三阶段渐进式融合](./03_三阶段渐进式融合_训练策略总览)——本章讲的模块在三个训练阶段里分别被怎样冻结/解冻
- [第 5 章：训练细节](./05_训练细节_FlowMatching目标与分支随机丢弃)——本章输出如何服务于最终的训练目标
