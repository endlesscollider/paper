---
title: "分组查询注意力 GQA：用更少的 Key/Value 换更快的推理"
order: 33.2
tags: [Transformer, Attention, GQA, KV-Cache, 大模型, 推理优化]
category: 前置知识
star: 4
---

# 分组查询注意力 GQA：用更少的 Key/Value 换更快的推理

> 标准多头注意力里,Query、Key、Value 各有相同数量的"头"。GQA 打破了这个对称性:让多个 Query 头共享同一组 Key/Value。本文讲清楚这样做的动机、具体机制,以及它和 KV-Cache 内存开销之间的直接联系。

## 相关阅读

- [Cross-Attention 与交替注意力机制](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制) — Attention 的基础概念
- [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码) — GQA 节省的正是 KV-Cache 的内存
- [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码) — 常与 GQA 搭配使用
- [XR0 深度解析系列](/系列/xr0_deep_dive/) — DiT 动作头中 GQA 的具体应用

---

## 一、先回顾:标准多头注意力(MHA)的开销在哪

### 1.1 多头注意力的基本结构

标准多头注意力(Multi-Head Attention, MHA)把输入的隐藏向量投影成三组张量:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

然后把 $Q, K, V$ 各自沿隐藏维度切成 $H$ 个"头",每个头独立做一次 Attention,最后把结果拼接起来。这里的 $H$(头数)对 $Q, K, V$ 是**完全相同的一个数字**——每个 Query 头都配备了自己专属的一个 Key 头和 Value 头。

### 1.2 推理时的瓶颈:KV-Cache 占用的显存

在自回归生成场景中(比如 LLM 逐词生成,或者 XR0 的 DiT 逐层缓存 VLM 的中间结果),每生成一个新 token 都要用到**之前所有 token 算好的 Key 和 Value**。为了避免重复计算,系统会把每一层、每一个已生成 token 的 $K, V$ 存下来,这就是 KV-Cache(具体机制见 [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码))。

KV-Cache 的显存占用和"KV 头的数量"直接成正比。如果一个模型有 32 个 Query 头,标准 MHA 要求配 32 个 Key 头和 32 个 Value 头,序列越长、层数越多,缓存的 K/V 张量就越大,很容易成为推理时显存的主要瓶颈——尤其是在需要长上下文或者大批量并发推理的场景。

**核心问题**:有没有办法减少需要缓存的 K/V 数量,同时不明显损失模型表达能力?

---

## 二、GQA 的核心思路:分组共享

### 2.1 一句话直觉

> 把 $H$ 个 Query 头分成 $G$ 组,同一组内的多个 Query 头**共享同一对** Key/Value 头——相当于几个"侦察兵"(Query)共用同一份"情报资料"(K/V),而不是每人各配一份。

### 2.2 三种配置的对比

| 配置名称 | Query 头数 | KV 头数 | 关系 |
|---------|-----------|--------|------|
| **MHA**(标准多头注意力) | $H$ | $H$ | 每个 Query 头独享一对 K/V |
| **GQA**(分组查询注意力) | $H$ | $G$($1 < G < H$) | 每 $H/G$ 个 Query 头共享一对 K/V |
| **MQA**(多查询注意力) | $H$ | $1$ | 所有 Query 头共享同一对 K/V(GQA 的极端情况) |

GQA 是 MHA 和 MQA 之间的一个折中方案:MHA 表达力最强但缓存开销最大,MQA 缓存开销最小但表达力损失较多,GQA 通过调节分组数 $G$ 在两者之间找平衡点。

### 2.3 具体计算过程

给定 $H$ 个 Query 头和 $G$ 个 KV 头($H$ 能被 $G$ 整除),定义分组倍数:

$$
n_{\text{rep}} = \frac{H}{G}
$$

前向传播分三步:

**第一步:分别投影得到数量不对等的 Q、K、V**

$$
Q \in \mathbb{R}^{B \times H \times S \times d_h}, \qquad K, V \in \mathbb{R}^{B \times G \times S \times d_h}
$$

**这一步在做什么**:Query 的投影矩阵输出 $H$ 组头,而 Key/Value 的投影矩阵只输出 $G$ 组头(注意这里的 $G < H$,所以 K/V 投影层的参数量、以及存下来的 K/V 张量本身,都比 MHA 少了 $n_{\text{rep}}$ 倍)。

**逐项拆解**:

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| $B$ | batch size | 一批处理的样本数 |
| $H$ | Query 头数 | 例如 32 |
| $G$ | KV 头数(组数) | 例如 8 |
| $S$ | 序列长度 | 当前序列的 token 数 |
| $d_h$ | 每个头的维度 | 例如 128 |
| $n_{\text{rep}} = H/G$ | 每组内共享的 Query 头数 | 例如 $32/8=4$,即每 4 个 Query 头共用 1 对 K/V |

**第二步:把 K、V 重复扩展到和 Q 头数一致**

$$
K^{\text{rep}} = \text{repeat\_interleave}(K, n_{\text{rep}}, \text{dim=heads}), \qquad V^{\text{rep}} = \text{repeat\_interleave}(V, n_{\text{rep}}, \text{dim=heads})
$$

**这一步在做什么**:因为标准的 Attention 计算(softmax、加权求和)要求 Q、K、V 的头数对齐才能逐头做矩阵乘法,所以在实际做 Attention 运算之前,需要把只有 $G$ 组的 K、V 沿着头这个维度"重复"$n_{\text{rep}}$ 次,变成和 $Q$ 一样的 $H$ 组——注意这只是计算时的临时展开,**存储的 KV-Cache 依然只有 $G$ 组**,重复操作在用到的时候才做。

**具体数值例子**:取 $H=8$,$G=2$,$n_{\text{rep}}=4$。原始 $K$ 的头编号是 $[k_0, k_1]$(共 2 个头)。重复扩展后:

$$
K^{\text{rep}} = [k_0, k_0, k_0, k_0, k_1, k_1, k_1, k_1]
$$

即 Query 头 0-3 共享 $k_0$,Query 头 4-7 共享 $k_1$。

**第三步:标准 Attention 计算**

$$
\text{Attention}(Q, K^{\text{rep}}, V^{\text{rep}}) = \text{softmax}\left(\frac{Q (K^{\text{rep}})^T}{\sqrt{d_h}}\right) V^{\text{rep}}
$$

这一步和标准 MHA 完全一样,区别只在于用来做计算的 $K^{\text{rep}}, V^{\text{rep}}$ 是从更少的原始 K、V 复制出来的。

### 2.4 完整数值走查

假设 $H=8$(Query 头数),$G=2$(KV 头数),$d_h=4$(每头维度),序列长度 $S=1$(只看一个 token,方便手算)。

- Query 头 0 的向量:$q_0 = [1, 0, 0, 0]$
- KV 头 0(被 Query 头 0-3 共享):$k_0 = [1, 0, 0, 0]$,$v_0 = [10, 20, 30, 40]$
- KV 头 1(被 Query 头 4-7 共享):$k_1 = [0, 1, 0, 0]$,$v_1 = [50, 60, 70, 80]$

对 Query 头 0(属于第 0 组,用 $k_0, v_0$):
$$
\text{score} = q_0 \cdot k_0 / \sqrt{4} = (1\times1+0+0+0)/2 = 0.5
$$
只有一个 KV 对时 softmax 后权重恒为 1,输出 $=v_0=[10,20,30,40]$。

对 Query 头 4(属于第 1 组,用 $k_1, v_1$)即使 $q_4$ 的数值和 $q_0$ 不同,只要它落在第 1 组,就只会和 $k_1, v_1$ 交互,输出会基于 $v_1=[50,60,70,80]$。

**关键结论**:8 个 Query 头最终产生 8 个不同的输出(因为每个头的 Query 向量不同,即使共享同一个 K/V,attention score 也不同),但整个过程只需要存储和计算 2 组 K/V,而不是 8 组——**这就是显存节省的来源**。

---

## 三、为什么这样做不会让性能损失太多

### 3.1 直觉解释

Attention 的表达力主要来自于:(1)每个头能学到不同的"查询模式"(体现在 $Q$ 的多样性上);(2)不同头能关注序列中不同的信息(体现在 $K, V$ 提取到的特征上)。GQA 保留了 $Q$ 的完全多样性(仍然是 $H$ 个独立的 Query 头),只压缩了 $K, V$ 的多样性。经验上,多个 Query 头共享同一份"被检索的内容"(K/V)但仍能各自算出不同的注意力权重分布,信息损失比想象中小。

### 3.2 分组数 $G$ 的选择是一个权衡

| $G$ 的取值 | 效果 |
|-----------|------|
| $G=H$(即 MHA) | 表达力最强,KV-Cache 最大 |
| $G$ 适中(如 $H/4$) | 表达力小幅下降,KV-Cache 减少到 $1/4$ |
| $G=1$(即 MQA) | KV-Cache 最小,但表达力损失相对更明显 |

实践中 $G$ 通常取 $H$ 的 $1/4$ 到 $1/8$,在几乎不损失下游任务效果的前提下,把 KV-Cache 压缩到很小的比例。

### 3.3 XR0 中的具体配置

XR0(Xiaomi-Robotics-0)的 DiT 动作头里,`DiTAttention` 模块的构造参数是 `hidden_size=1024, head_dim=128, kv_heads=8`,算出 Query 头数为 $1024/128=8$。也就是说在这个具体配置下 $H=G=8$,退化为标准 MHA;但代码框架本身按 GQA 的通用形式实现(`kv_group = num_heads // kv_heads`,当 `kv_heads` 设为比 `num_heads` 更小的值时即可启用真正的分组共享),这样设计是为了在不同规模的模型变体之间复用同一套 Attention 实现,具体细节参见 [XR0 深度解析 - DiT 动作头架构](/系列/xr0_deep_dive/05_DiT动作头架构_AdaLN与GQA跨注意力)。

---

## 四、延伸阅读

- [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码) — GQA 节省的正是这里的缓存
- [RoPE 旋转位置编码](/前置知识/002k_前置知识_RoPE旋转位置编码) — 常与 GQA 一起构成现代 Attention 的标准组合
- [XR0 深度解析系列](/系列/xr0_deep_dive/) — GQA 在 DiT 动作头中的实际配置
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023 — GQA 原始论文
