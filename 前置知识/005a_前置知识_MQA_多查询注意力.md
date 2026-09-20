---
title: "MQA：多查询注意力——把 KV Cache 砍到极限的注意力变体"
order: 33.05
tags: [Transformer, Attention, MQA, KV-Cache, 大模型, 推理优化]
category: 前置知识
---

# MQA：多查询注意力——把 KV Cache 砍到极限的注意力变体

> **一句话**：标准多头注意力（Multi-Head Attention, MHA）里，每个 Query 头都配一份专属的 Key、Value。多查询注意力（Multi-Query Attention, MQA）把这件事改成——**所有 Query 头共用同一份 Key、Value**。代价是牺牲一部分模型质量，换来的是推理时缓存显存和内存带宽的巨大节省。本文讲清楚这笔交易具体划不划算，划算在哪，不划算在哪。

## 相关阅读

- [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码) — 理解本文之前必须先知道 KV Cache 是什么、为什么需要它
- [分组查询注意力 GQA](/前置知识/002l_前置知识_分组查询注意力GQA) — GQA 是本文讲的 MQA 和标准 MHA 之间的插值方案，读完本文建议接着读这篇
- [Causal Attention 因果注意力掩码](/前置知识/001g_前置知识_Causal_Attention因果注意力掩码) — 自回归生成的基础机制
- [MLA：多头潜在注意力](/前置知识/005b_前置知识_MLA_多头潜在注意力) — 沿着"压缩 KV"这条思路继续往前走的更精细方案
- [Attention 变体全景综述](/论文综述/S23_Attention变体全景综述) — MHA→MQA→GQA→MLA 完整技术图谱

---

## 一、先把问题摆清楚：KV Cache 为什么会变成瓶颈

### 1.1 贯穿全文的例子

假设你要部署一个 7B 参数规模的自回归语言模型做长文本处理，具体配置是：

- 层数 $L = 32$
- 每层的注意力头数 $H = 32$
- 每个头的维度 $d_h = 128$（所以隐藏维度 $d = H \times d_h = 4096$，这是 7B 级模型的典型配置）
- 处理的文本长度 $S = 2048$ 个 token
- 用 bfloat16 存储（每个数字占 2 字节）

这个具体的模型配置会贯穿本文，每引入一个新概念就在这个例子上算一遍具体数字。

### 1.2 自回归解码为什么必须缓存 K、V

一个自回归模型生成第 2000 个 token 时，需要用它的 Query 向量去和前面已经生成的全部 1999 个 token 的 Key、Value 做注意力计算。如果每生成一个新 token 都从头把前面所有 token 的 Key、Value 重新算一遍，计算量会随序列长度平方增长——这在实践中完全不可接受。

标准做法是把每个 token 算出来的 Key、Value 存进显存里的一块缓冲区，也就是 KV Cache，下一步直接从缓存里读，不再重新计算。KV Cache 具体的产生机制、为什么只缓存 K/V 不缓存 Q、代码怎么实现，这些内容在 [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码) 里已经完整讲过，这里不再重复推导，只关注一件事：**这份缓存到底有多大，能不能变小**。

### 1.3 缓存大小的公式，以及它带来的具体数字

KV Cache 的显存占用，取决于要缓存多少组独立的 Key/Value：

$$
\text{KV Cache 显存} = 2 \times L \times S \times H_{\text{kv}} \times d_h \times \text{bytes}
$$

**这个公式在做什么**：把"总共要存多少个数字"乘上"每个数字占几个字节"，算出 KV Cache 真实吃掉的显存量。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $2$ | **两份账本** | Key 和 Value 各存一份，缺一不可 |
| $L$ | **层数** | 每一层都有自己独立的 K、V 投影，缓存必须逐层分别存 |
| $S$ | **已处理的 token 数** | 缓存随生成长度线性增长，这是长上下文推理的核心压力来源 |
| $H_{\text{kv}}$ | **KV 头的数量** | 决定"每个 token 要存几份 Key/Value"——这正是 MQA 要动刀的地方 |
| $d_h$ | **每个头的维度** | 每一份 Key 或 Value 向量本身有多大 |
| bytes | **数字的胖瘦** | bf16 占 2 字节，fp32 占 4 字节，量化到 int8 只占 1 字节 |

**用人话读**："KV Cache 的大小 = 2（K和V）× 层数 × 序列长度 × KV头数 × 每头维度 × 每个数占的字节数，五个因子相乘。"

**为什么是这个形式**：这是纯粹的存储量清点公式，没有设计动机可言——它只是把"到底存了多少个浮点数，每个数占几个字节"如实地乘起来。真正有设计空间的是 $H_{\text{kv}}$ 这一项：标准 MHA 里 $H_{\text{kv}}$ 恒等于 Query 头数 $H$，而这正是可以被压缩的自由度。
:::

代入贯穿全文的例子（$L=32, S=2048, d_h=128$，bf16 即 2 字节），标准 MHA 下 $H_{\text{kv}} = H = 32$：

$$
2 \times 32 \times 2048 \times 32 \times 128 \times 2 = 1{,}073{,}741{,}824 \text{ 字节} \approx 1024\text{ MB}
$$

**这个公式在做什么**：把上一节的抽象公式代入本文贯穿全文的具体配置数字，算出标准 MHA 下单条序列的 KV Cache 到底要占多少显存。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $32$（层数位置） | **层数 $L$** | 本文例子设定的 32 层 Transformer |
| $2048$ | **序列长度 $S$** | 本文例子处理的文本长度 |
| $32$（头数位置） | **KV 头数 $H_{\text{kv}}$** | 标准 MHA 下等于 Query 头数，本文例子里是 32 |
| $128$ | **每头维度 $d_h$** | 本文例子设定的每头维度 |
| $2$（末位） | **每个数占的字节数** | bf16 精度，2 字节 |
| $1024\text{ MB}$ | **最终显存占用** | 五个因子相乘再除以 $1024^2$ 换算成 MB 得到的结果 |

**用人话读**："把本文设定的具体配置数字，代入上一节的显存公式，算出标准做法下一条 2048 长度的序列要占用 1024 MB 显存。"

**为什么是这个形式**：这一步纯粹是数值代入，没有额外设计动机——它的作用是把抽象公式变成一个具体到"1 GB"的直观数字，为后面对比 MQA 压缩后的效果提供基准。
:::

单个样本、单条 2048 长度的序列，仅 KV Cache 就要吃掉 **1 GB** 显存。如果要支持批量推理（比如同时服务 32 个用户）或者更长的上下文（比如 32K token），这个数字会再乘以对应的倍数，很容易膨胀到几十甚至上百 GB——这就是为什么 KV Cache 是长上下文、大批量推理场景下显存的第一大瓶颈。

---

## 二、MQA 的核心思路：把 $H_{\text{kv}}$ 砍到 1

### 2.1 一句话直觉

> 标准 MHA 里，32 个 Query 头各自配一个专属的 Key 头和 Value 头，像是 32 个侦察兵各自背着自己的一份情报资料。MQA 的做法是：**所有 32 个 Query 头共用同一份 Key、Value**，情报资料只印一份，32 个侦察兵传阅同一份文件。

### 2.2 具体的前向计算

标准 MHA 的 Query、Key、Value 投影公式是三个形状相同的线性层，各自把隐藏向量 $\mathbf{h}_t$ 投影到 $H \times d_h$ 维，再切成 $H$ 个头。MQA 只改了 K、V 这两个投影层的输出维度：

$$
\mathbf{q}_t = W_Q \mathbf{h}_t \in \mathbb{R}^{H \cdot d_h}, \qquad \mathbf{k}_t = W_K \mathbf{h}_t \in \mathbb{R}^{d_h}, \qquad \mathbf{v}_t = W_V \mathbf{h}_t \in \mathbb{R}^{d_h}
$$

**这个公式在做什么**：三个投影层里，Query 的输出维度不变（仍然覆盖全部 $H$ 个头），但 Key 和 Value 的输出维度被砍到只剩**一个头的大小**——这一步直接决定了后面需要缓存的数据量。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mathbf{h}_t$ | **原材料** | 当前 token 经过上一层输出的隐藏向量，是三路投影共同的输入 |
| $W_Q \in \mathbb{R}^{H d_h \times d}$ | **提问模具** | 把输入冲压成 $H$ 个头的 Query，形状和标准 MHA 完全一样，一点没削减 |
| $W_K, W_V \in \mathbb{R}^{d_h \times d}$ | **共享情报模具** | 只冲压出**一个头**大小的 Key/Value，而不是 $H$ 个头 |
| $\mathbf{q}_t$ | **32 份不同的提问** | 切成 $H$ 个头后，每个头仍然代表一种独立的"查询角度" |
| $\mathbf{k}_t, \mathbf{v}_t$ | **唯一的一份情报** | 不再切分成多个头，所有 Query 头之后都会用这唯一的一份 |

**用人话读**："Query 还是老老实实算出 32 份不同的提问，但 Key 和 Value 只算一份，不再像标准做法那样也搞 32 份。"

**为什么是这个形式**：注意力的信息检索能力主要来自"提问角度的多样性"（Query 的头数），而不是"被检索内容的份数"（K/V 的头数）。MQA 的设计赌的正是这一点——只削减 K/V 的头数，尽量保留 Query 的多样性，希望信息损失能压到最低。
:::

接下来做注意力计算时，$H$ 个 Query 头**全部**去和这唯一的一份 $\mathbf{k}_t, \mathbf{v}_t$ 交互：

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{\mathbf{q}_{t,i}^T \mathbf{k}_j}{\sqrt{d_h}}\right) \mathbf{v}_j, \qquad i = 1, \dots, H
$$

**这个公式在做什么**：对每一个 Query 头 $i$，都用它自己独特的 $\mathbf{q}_{t,i}$ 去查询同一份 Key/Value 序列，算出各自不同的输出——头之间的差异完全来自 Query 的不同，而不是看到了不同的 Key/Value。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mathbf{q}_{t,i}$ | **第 $i$ 个侦察兵的提问** | 32 个头各自不同，这是信息多样性唯一的来源 |
| $\mathbf{k}_j, \mathbf{v}_j$ | **公共情报库里第 $j$ 页** | 对所有 32 个头都一样，不带下标 $i$ |
| $\mathbf{q}_{t,i}^T \mathbf{k}_j / \sqrt{d_h}$ | **相关度打分** | 第 $i$ 个头觉得"第 $j$ 个历史 token 跟我现在想查的东西有多相关" |
| $\text{Softmax}_j(\cdot)$ | **归一化成权重** | 把打分转成一组和为 1 的关注度分布 |
| $\sum_j (\cdot) \mathbf{v}_j$ | **按权重取内容** | 用这组权重去加权平均公共情报库里的内容，得到这个头的最终输出 |

**用人话读**："32 个头各自拿着不同的问题，去问同一批 Key/Value；因为问题不同，即便情报源相同，每个头查出来的答案权重分布也不一样，最终输出仍然是 32 份不同的结果。"

**为什么是这个形式**：这一步和标准 Attention 公式在数学形式上完全一样，唯一的区别只是 $\mathbf{k}_j, \mathbf{v}_j$ 不再带头下标 $i$——这保证了 MQA 在计算逻辑上是标准 Attention 的一个特例，不需要额外设计新的计算规则，工程实现的改动量很小。
:::

### 2.3 用具体数字看会不会真的丢信息

取 $H=4$（简化版，方便手算），$d_h=2$，只看一个历史 token 的 Key/Value：$\mathbf{k} = [1, 0]$，$\mathbf{v} = [10, 20]$。四个 Query 头各不相同：

- $\mathbf{q}_1 = [2, 0]$：打分 $= 2/\sqrt{2} = 1.41$
- $\mathbf{q}_2 = [0.5, 0]$：打分 $= 0.5/\sqrt{2} = 0.35$
- $\mathbf{q}_3 = [1, 1]$：打分 $= 1/\sqrt{2} = 0.71$
- $\mathbf{q}_4 = [-1, 0]$：打分 $= -1/\sqrt{2} = -0.71$

因为只有一个历史 token，softmax 后权重恒为 1，四个头的输出都等于 $\mathbf{v} = [10, 20]$——**在只有一个 Key 的极端情况下，四个头的输出完全一样，Query 的差异被完全浪费了**。但一旦历史 token 数增加到多个，不同的打分会导致 softmax 权重分布不同，四个头就会从同一份 Key/Value 序列里"注意"到不同的部分，输出重新出现差异。

这个手算例子暴露了 MQA 的本质权衡：**当序列变长、需要检索的信息变丰富时，Query 的多样性还能发挥作用；但所有头能看到的"素材"始终是同一份，信息的上限被这唯一一份 K/V 锁死了**。这就是为什么 MQA 会有性能损失——不是损失在"问的角度不够多"，而是损失在"能查的素材种类不够多"。

---

## 三、具体的收益：KV Cache 缩小多少倍

### 3.1 直接代入公式

回到贯穿全文的例子（$L=32, S=2048, d_h=128$，bf16），MQA 下 $H_{\text{kv}} = 1$：

$$
2 \times 32 \times 2048 \times 1 \times 128 \times 2 = 33{,}554{,}432 \text{ 字节} \approx 32\text{ MB}
$$

**这个公式在做什么**：把 $H_{\text{kv}}$ 从 32 改成 MQA 的 1，其余四个因子保持不变，重新算一遍同一条序列的 KV Cache 大小，直接对比出压缩效果。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $1$（头数位置） | **MQA 下的 $H_{\text{kv}}$** | 唯一改变的因子，从标准 MHA 的 32 变成了 1 |
| 其余四个因子 | **保持不变的部分** | 层数、序列长度、每头维度、字节数都和上一节 MHA 的计算完全一样 |
| $32\text{ MB}$ | **压缩后的显存占用** | 相比 MHA 的 1024 MB，只因为 $H_{\text{kv}}$ 从 32 变成 1，结果直接缩小到 1/32 |

**用人话读**："只把 KV 头数这一个因子从 32 改成 1，其他都不变，重新算一遍，得到的显存占用变成了原来的 1/32。"

**为什么是这个形式**：因为 KV Cache 显存公式里 $H_{\text{kv}}$ 是纯线性因子，改变它、其余不变，压缩比例就直接等于 $H_{\text{kv}}$ 前后比值的倒数——这正是 MQA"砍头数"这个设计能直接换算成具体压缩倍数的原因。
:::

对比标准 MHA 的 1024 MB，**MQA 把 KV Cache 从 1 GB 压到 32 MB，缩小了整整 32 倍**——这个 32 倍恰好就是头数 $H=32$，因为 KV Cache 大小和 $H_{\text{kv}}$ 是线性关系，$H_{\text{kv}}$ 从 32 降到 1，缓存自然缩小 32 倍。

| 配置 | $H_{\text{kv}}$ | KV Cache（单条 2048 长度序列） | 相对 MHA 的倍数 |
|------|----------------|-------------------------------|-----------------|
| MHA（标准） | 32 | 1024 MB | 1× |
| MQA | 1 | 32 MB | **1/32** |

### 3.2 省下的显存能换来什么

省下的显存不是白白省下——它可以直接转化成两种收益：

1. **更大的批量（batch size）**：显存腾出来了，同一张卡能同时服务更多用户请求，单位时间吞吐量上升
2. **更长的上下文**：同样的显存预算下，能处理的序列长度上限提高了 32 倍

而且解码阶段（decode）本身是**内存带宽瓶颈**而不是算力瓶颈——每生成一个新 token，GPU 要把整份 KV Cache 从显存搬到计算单元里参与矩阵乘法，这个搬运的数据量直接决定了解码速度。缓存缩小到 1/32，搬运的数据量也缩小到 1/32，解码速度会明显加快。这也是为什么 Multi-Query Attention 最早由 Noam Shazeer 在论文《Fast Transformer Decoding: One Write-Head is All You Need》（2019）中提出——标题里的"one write-head"说的就是只保留一个 KV 头。

---

## 四、代价：性能损失有多大

天下没有免费的午餐。DeepSeek-V2 论文（2024）在附录里专门做了一组对照实验，训练三个参数量对齐到 7B、结构完全相同（只有注意力机制不同）的稠密模型，都用 1.33T token 训练，在四个高难度基准上评测：

| 基准（指标） | MQA | GQA | MHA |
|-------------|-----|-----|-----|
| BBH（EM，3-shot） | 33.2 | 35.6 | 37.0 |
| MMLU（Acc，5-shot） | 37.9 | 41.2 | 45.2 |
| C-Eval（Acc，5-shot） | 30.0 | 37.7 | 42.9 |
| CMMLU（Acc，5-shot） | 34.6 | 38.4 | 43.5 |

数据来源：DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv:2405.04434, Appendix D.1, Table 8。

四个基准上都是同一个规律：**MHA > GQA > MQA**，而且差距不小——MMLU 上 MHA 比 MQA 高出 7.3 个百分点，C-Eval 上高出接近 13 个百分点。这说明"只保留一份 K/V 给所有 Query 头共享"确实会实打实地损失模型能力，尤其在需要精细知识检索、复杂推理的高难度任务上损失更明显。

### 4.1 为什么会损失，损失在哪

结合第二节的手算例子，可以把损失的机制说清楚：

- Query 的多样性能让模型学到"从不同角度提问"，这部分能力 MQA 完整保留了
- 但 Key、Value 只有一份，意味着模型**能拿来回答这些提问的素材种类被锁死了**——不同的头本来可以各自专注于文本中不同类型的信息（比如一个头专注于语法结构，另一个头专注于指代关系），但现在它们被迫从同一份"素材库"里各自挑选，素材库本身的信息容量成了瓶颈
- 训练时这种约束会传导到梯度上：所有头的梯度都要通过同一个 $W_K, W_V$ 反传，这个共享投影层要同时兼顾所有头的需求，学习难度更大，Shazeer 原始论文里也报告了 MQA 训练相对不稳定的问题

### 4.2 训练稳定性问题

除了最终精度下降，MQA 在训练过程中还容易出现不稳定的情况——因为 $W_K, W_V$ 这两个共享投影层承受的"梯度压力"比标准 MHA 大得多（相当于把原本 32 份独立的学习信号硬塞进一份参数里），训练时更容易出现损失震荡或收敛变慢的问题。这也是后续 GQA 被提出的直接动机之一。

---

## 五、代码实现

MQA 的实现思路很直接：把 K、V 的投影层输出维度改成只有一个头的大小，然后在注意力计算前把这唯一一份 K、V 广播（broadcast）到匹配 Query 头数的形状。下面是一个从零实现的最小版本：

```python
import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    """MQA：Query 有 H 个头，Key/Value 只有 1 个头，被所有 Query 头共享"""
    def __init__(self, d_model, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** 0.5

        # Query 投影：输出维度覆盖全部 H 个头
        self.W_Q = nn.Linear(d_model, num_heads * head_dim, bias=False)
        # Key/Value 投影：输出维度只有 1 个头的大小 —— 这是 MQA 唯一的改动点
        self.W_K = nn.Linear(d_model, head_dim, bias=False)
        self.W_V = nn.Linear(d_model, head_dim, bias=False)
        self.W_O = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        """
        x: [batch, seq_len, d_model]
        kv_cache: (cached_k, cached_v) 或 None，各自形状 [batch, past_len, head_dim]
        """
        batch, seq_len, _ = x.shape

        q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k_new = self.W_K(x)  # [batch, seq_len, head_dim] —— 注意没有 num_heads 这一维
        v_new = self.W_V(x)  # [batch, seq_len, head_dim]

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k_new], dim=1)  # [batch, total_len, head_dim]
            v = torch.cat([cached_v, v_new], dim=1)
        else:
            k, v = k_new, v_new

        # 把唯一的一份 K/V 广播到匹配 num_heads 的形状，供每个 Query 头查询
        # k: [batch, total_len, head_dim] -> [batch, 1, total_len, head_dim] -> 隐式广播
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k_expand = k.unsqueeze(1)  # [batch, 1, total_len, head_dim]，广播维为 1
        v_expand = v.unsqueeze(1)

        scores = torch.matmul(q, k_expand.transpose(-2, -1)) / self.scale
        # scores: [batch, num_heads, seq_len, total_len] —— 广播机制让每个头都用上同一份 k_expand
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v_expand)  # [batch, num_heads, seq_len, head_dim]

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.W_O(out), (k, v)
```

关键点在于 `W_K`、`W_V` 的输出维度只有 `head_dim`，而不是标准 MHA 里的 `num_heads * head_dim`——这直接决定了参数量和 KV Cache 都缩小到 `1/num_heads`。后面用 `unsqueeze(1)` 把 K、V 在头这一维广播成 1，PyTorch 的广播机制会自动让所有 `num_heads` 个 Query 头共享同一份张量参与矩阵乘法，不需要真的复制出多份数据。

---

## 六、MQA 在 MHA 和 GQA 之间的位置

### 6.1 三种方案放在一起看

| 方案 | KV 头数 $H_{\text{kv}}$ | 相对 MHA 的 KV Cache 倍数 | 相对 MHA 的性能 |
|------|----------------------|---------------------------|-----------------|
| MHA | $H$（等于 Query 头数） | 1× | 基准 |
| GQA | $1 < G < H$ | $G/H$ | 接近 MHA，略有下降 |
| **MQA** | **1** | **1/H**（本文的 32 倍压缩） | 明显下降（如上表 MMLU 差 7.3 点） |

### 6.2 GQA 是 MHA 和 MQA 之间的插值

看这张表就能发现一个自然的问题：MQA 把 $H_{\text{kv}}$ 砍到 1，换来最大的显存压缩，但性能损失也最大；MHA 保留完整的 $H_{\text{kv}}=H$，性能最好但显存开销最大。**有没有中间地带？**

答案就是 [分组查询注意力 GQA](/前置知识/002l_前置知识_分组查询注意力GQA)——它把 Query 头分成 $G$ 组（$1 < G < H$），每组内的多个 Query 头共享一份 K/V，而不是全部头共享同一份（MQA），也不是每个头都有自己独立的一份（MHA）。用本文的记号看，**MQA 就是 GQA 在 $G=1$ 时的特例，MHA 是 GQA 在 $G=H$ 时的特例**——GQA 通过调节 $G$ 这一个超参数，在"显存开销"和"模型质量"这两个相互冲突的目标之间连续地滑动，找一个符合具体场景需求的平衡点。GQA 文章里的完整数值走查和分组共享的具体实现细节可以接着读那一篇，本文不再重复。

### 6.3 但分组共享不是唯一的思路

GQA/MQA 这条路线的核心手段都是"减少独立 K/V 的份数"——本质上是做一种**结构性的裁剪**：要么全砍到 1（MQA），要么砍到若干组（GQA）。这种裁剪天然会丢信息，因为被合并到一组里的 Query 头被迫共享完全相同的 K/V 内容，没有任何缓和空间。

另一条思路是不裁剪"份数"，而是把每一份 K/V 本身在维度上压缩变小，需要时再用一个投影矩阵还原出来——这样理论上可以保留更丰富的信息，同时依然大幅压缩存储。这正是 [MLA：多头潜在注意力](/前置知识/005b_前置知识_MLA_多头潜在注意力) 的思路，接下来的文章会讲清楚它具体怎么做，以及它为什么能同时压缩缓存又不像 MQA/GQA 这样明显损失表达力。

---

## 七、总结

| 要点 | 内容 |
|------|------|
| **核心机制** | 所有 Query 头共享同一份 Key、Value（$H_{\text{kv}}=1$），Query 头数保持不变 |
| **KV Cache 收益** | 相对 MHA 缩小 $H$ 倍——本文例子中从 1024 MB 压到 32 MB，缩小 32 倍 |
| **性能代价** | DeepSeek-V2 论文实测：7B 规模下 MMLU 掉 7.3 点，C-Eval 掉近 13 点，明显劣于 MHA 和 GQA |
| **为什么会损失** | Query 多样性保留了，但所有头能检索的"素材库"被压缩成一份，信息容量成了瓶颈；共享投影层的梯度压力也更大，训练更不稳定 |
| **提出背景** | Noam Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019 |
| **在技术谱系中的位置** | MHA 和 MQA 是两个极端，GQA 是介于二者之间的可调插值方案 |
| **后续演进方向** | MLA 用"压缩维度而非砍头数"的思路，试图同时兼顾缓存大小和模型表达力 |

## 延伸阅读

- [分组查询注意力 GQA](/前置知识/002l_前置知识_分组查询注意力GQA) — MHA 与 MQA 之间的插值方案，完整的分组共享机制和数值走查
- [KV-Cache 与自回归解码](/前置知识/002m_前置知识_KV_Cache与自回归解码) — KV Cache 的产生原理、显存公式的完整推导
- [MLA：多头潜在注意力](/前置知识/005b_前置知识_MLA_多头潜在注意力) — 不砍头数、改压缩维度的下一代方案
- [DeepSeek-V2 MLA 精读](/论文综述/104_DeepSeekV2_MLA多头潜在注意力) — MHA/GQA/MQA 消融实验数据的原始出处
- [Attention 变体全景综述](/论文综述/S23_Attention变体全景综述) — 完整技术图谱与选型指南
- Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv:1911.02150, 2019 — MQA 原始论文
