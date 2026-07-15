---
title: "Causal Attention：因果注意力掩码"
order: 33
tags: [Transformer, Attention, 自回归, 掩码]
category: 前置知识
---

# Causal Attention：因果注意力掩码

> 为什么 GPT 生成文字时后面的词看不到前面的词的"未来"？为什么 GR00T 的动作生成不需要这个限制？本文讲清楚 Causal Attention 的动机、实现和适用场景。

## 相关阅读

- [Cross-Attention 与交替注意力机制](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制)
- [GR00T N1.7 - Self-Attention 交叉层](/系列/groot_n1d7_deep_dive/13_SelfAttention交叉层_interleave机制)

---

## 1. 问题的起点：自回归生成

### 1.1 什么是自回归生成？

很多序列生成任务（如语言模型写文章）采用**自回归**（Autoregressive）方式：
一个 token 一个一个地生成，每次生成都基于**已经生成的部分**。

$$
P(x_1, x_2, \ldots, x_n) = P(x_1) \cdot P(x_2 | x_1) \cdot P(x_3 | x_1, x_2) \cdots P(x_n | x_1, \ldots, x_{n-1})
$$

> **一句话直觉**：整句话的概率被拆解成"一个接一个预测下一个词"的连乘。

**具体例子**：生成 "The cat sat down"
```
P("The")                          → 生成第1个词
P("cat" | "The")                  → 给定"The"，生成第2个词
P("sat" | "The", "cat")           → 给定"The cat"，生成第3个词
P("down" | "The", "cat", "sat")   → 给定"The cat sat"，生成第4个词
```

推理时，模型必须严格按顺序：先有 "The"，才能生成 "cat"；有了 "The cat"，才能生成 "sat"。**未来的词在生成那一刻还不存在**。

### 1.2 训练时的矛盾

推理时模型是"一个词一个词蹦出来的"，但**训练**时我们通常把整句话
`"The cat sat down"` 一次性喂给模型（这样并行度高、训练快）。

矛盾出现了：如果不加任何限制，Self-Attention 会让每个位置看到**整个输入序列**——
包括它"未来"的词。这意味着模型在学习"预测 cat 后面是什么"时，
直接能看到答案 "sat" 就在输入里，根本不需要学习任何东西。

> 这就像考试时，试卷上直接印着每道题的正确答案在旁边——你不需要推理，抄就行。

---

## 2. Causal Attention：解决矛盾的掩码机制

### 2.1 核心思想

给 Self-Attention 加一个**掩码**（mask），强制规定：
第 $i$ 个位置只能 attend 到位置 $1, 2, \ldots, i$（自己和前面），**看不到** $i+1$ 之后的位置。

这样即使训练时把整句话一次性喂进去，每个位置的预测**仍然只依赖它前面的内容**——
和推理时"一个词一个词生成"的信息可见性完全一致。

### 2.2 数学表示

标准 Self-Attention：
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Causal Attention 在 softmax 之前加一个掩码矩阵 $M$：

$$
\text{CausalAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

> **一句话直觉**：在算注意力权重之前，把"不该看到"的位置的分数强行拉到负无穷，softmax 之后这些位置的权重变成 0。

**逐项拆解**：
- $M \in \mathbb{R}^{n \times n}$：掩码矩阵，$M_{ij} = 0$（如果 $j \le i$，允许看）或 $M_{ij} = -\infty$（如果 $j > i$，禁止看）
- $QK^T + M$：加上掩码后，被禁止的位置分数变为 $-\infty$
- $\text{softmax}(-\infty) = 0$：这些位置最终权重为 0，完全不参与加权求和

### 2.3 具体数值例子

4 个 token 的序列，掩码矩阵：

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

假设某一行 $QK^T$ 的原始分数是 $[2, 3, 1, 4]$（第 3 个 token，即第 3 行）：

```
原始分数:        [2,     3,     1,    4  ]
加上第3行的mask:  [2,     3,     1,   -∞ ]   (第4个位置被屏蔽)
softmax后:       [0.24,  0.65,  0.11, 0.0]   (第4个位置权重=0)
```

第 3 个 token 的输出完全不受第 4 个 token（"未来"）的影响——
即使 $QK^T$ 原始分数显示第 4 个 token 很相关（分数=4，是最高的），
掩码强行把它压制为 0。

### 2.4 为什么叫"下三角矩阵"？

如果把掩码矩阵中 $M_{ij}=0$（允许）标记为 1、$M_{ij}=-\infty$（禁止）标记为 0，
得到的是一个**下三角矩阵**：

```
[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

对角线及以下是 1（可见），对角线以上是 0（不可见）。这就是为什么
Causal Attention 常被称为"下三角掩码注意力"。

---

## 3. 代码实现

```python
import torch

def causal_attention(Q, K, V):
    """
    Q, K, V: [B, num_heads, seq_len, head_dim]
    """
    seq_len = Q.shape[-2]
    scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)  # [B, H, T, T]
    
    # 构建下三角掩码 (True = 允许看)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    
    # 被禁止的位置设为负无穷
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ V
```

`torch.tril` 生成下三角矩阵（True 表示"可以看"），`masked_fill` 把不能看的位置
（上三角部分）填成 $-\infty$。

---

## 4. Causal Attention vs 双向 Attention（Bidirectional）

| 维度 | Causal（单向） | Bidirectional（双向） |
|------|--------------|---------------------|
| 每个位置能看到 | 自己 + 之前的位置 | **所有**位置（前后都能看） |
| 典型应用 | GPT 系列（自回归文本生成） | BERT（文本理解）、ViT（图像理解） |
| 训练目标 | Next-token prediction | Masked language modeling / 分类 |
| 是否可并行生成 | 否（必须逐个生成） | 是（一次性输出所有位置的表示） |
| 掩码矩阵形态 | 下三角 | 全 1（无限制）或任务特定 mask |

**判断标准**：如果任务在推理时需要"边生成边参考已生成内容"（自回归），
就需要 causal mask。如果任务是"给定完整输入，一次性输出全部结果"（如理解、分类、
并行生成），则不需要 causal mask。

---

## 5. GR00T 中为什么不用 Causal Attention？

GR00T 的 DiT 中，Self-Attention 层（41 个 token：1 state + 40 action）
**没有使用** causal mask——所有 token 可以双向互相看到。

**原因**：GR00T 的动作生成不是自回归的！

回忆 [Flow Matching 数学基础](/系列/groot_n1d7_deep_dive/09_Flow_Matching数学基础)：
GR00T 一次性预测**整个** 40 步动作轨迹的速度场，不是"先生成第1步，再基于第1步生成第2步"。

```
自回归方式（GR00T 没有采用）：
  action_0 = f(state)
  action_1 = f(state, action_0)     ← 需要看到 action_0
  action_2 = f(state, action_0, action_1)  ← 需要看到前面所有
  ...

GR00T 实际采用的并行方式：
  [action_0, action_1, ..., action_39] = DiT(state, noisy_actions, VL特征)
  ← 一次前向传播同时输出所有40步的速度预测
```

因为是并行预测整个轨迹，模型在生成"第 5 步动作"时，
完全可以（也应该）参考"第 30 步的目标状态"来规划一条连贯的轨迹——
这正是双向注意力的价值：**全局规划**能力。

如果加上 causal mask，第 5 步的 action token 就看不到第 10、20、30 步的信息，
会严重削弱模型做长程规划的能力（比如"因为终点在这里，所以中间路径应该这样走"）。

---

## 6. 什么场景需要用到 Causal Attention？

在机器人 VLA 领域，Causal Attention 仍然有其用途：

| 场景 | 是否需要 Causal | 原因 |
|------|----------------|------|
| 语言模型生成文本描述 | 需要 | 文本是严格顺序的自回归生成 |
| VLM 骨干的语言部分（如 Qwen3-VL 的 LLM） | 需要 | LLM 内部仍然是自回归架构 |
| GR00T 的动作 DiT | 不需要 | 一次性并行预测整个动作块 |
| RT-2 式的离散动作 token 生成 | 需要 | 把动作当作文本 token 逐个自回归生成 |
| Diffusion Policy / Flow Matching 动作生成 | 不需要 | 并行去噪，非自回归 |

**关键规律**：只要生成方式是"逐 token 自回归"，就需要 causal mask；
只要是"并行整体生成"（如扩散模型的去噪），就不需要。

这也解释了为什么 GR00T 的骨干网络（Qwen3-VL 的 LLM 部分）内部用 causal attention
（因为它本质是个语言模型），但动作头的 DiT 用双向 attention
（因为动作生成是并行去噪，不是自回归）。

---

## 7. 总结

| 要点 | 内容 |
|------|------|
| **是什么** | 强制每个位置只能看到自己和之前的位置，看不到之后的位置 |
| **为什么需要** | 让并行训练（teacher forcing）和串行推理（自回归生成）的信息可见性一致 |
| **怎么实现** | 在 $QK^T$ 上加一个下三角掩码，未来位置设为 $-\infty$ |
| **和双向注意力的区别** | 双向注意力所有位置互相可见，适合"一次性理解/生成全部" |
| **GR00T 中的应用** | LLM 骨干需要（自回归文本生成），DiT 动作头不需要（并行去噪） |

Causal Attention 是自回归生成模型（GPT 系列）的基石，但不是所有 Transformer 都需要它——
判断标准始终是："这个任务是逐步生成还是一次性并行输出？"
