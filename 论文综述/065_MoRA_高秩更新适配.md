---
title: "MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning"
order: 65
tags: [LoRA, MoRA, 高秩更新, 参数高效微调]
category: 精读
---

# MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning

> **论文信息**：Jiang et al., 2024  
> **一句话概括**：LoRA 的低秩约束 $\Delta W = BA$ 限制了它学习新知识（如记忆新信息）的能力。MoRA 用方阵映射（$M \in \mathbb{R}^{r' \times r'}$，$r' = \sqrt{dr/k}$）替代低秩分解，在**相同参数预算**下实现更高秩的权重更新——尤其擅长记忆密集型任务。

**相关阅读**：
- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — LoRA 的低秩约束
- [AdaLoRA 精读](./057_AdaLoRA_自适应秩分配) — 自适应秩的另一种思路

---

## 贯穿全文的例子

> 场景：用 LoRA 微调 LLaMA-7B 来记忆一本专业医学教材（大量需要精确记忆的专有名词和事实）。
>
> **LoRA ($r=256$) 的问题**：$\Delta W$ 秩 ≤ 256 → 能表达的独立"知识方向"不超过 256 个。但医学教材可能包含数千个需要独立记忆的事实。
>
> **MoRA 的解决**：用相同参数量，实现 $\Delta W$ 的秩可以达到 $\min(d, k)$（理论上满秩）→ 可以编码更多独立信息。
>
> 代价：推理时不能像 LoRA 那样零开销合并（但可以近似合并）。

---

## 一、论文动机

### 1.1 LoRA 的低秩瓶颈

LoRA 的 $\Delta W = BA$ 其秩**永远不超过** $r$。这意味着：

$$
\text{rank}(\Delta W) \leq r
$$

**这个公式在做什么**：给出 LoRA 权重更新量的秩上限——无论怎么训练，$\Delta W$ 能表达的独立"变化方向"数量都不会超过 $r$。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\Delta W$ | **实际的权重改变量** | LoRA 训练出来加到原权重上的增量矩阵 $BA$ |
| $\text{rank}(\Delta W)$ | **独立方向计数器** | 数出 $\Delta W$ 能表达多少个线性独立的"变化方向" |
| $\leq r$ | **天花板** | 这个数量永远不会超过 LoRA 设定的秩 $r$ |

**用人话读**："LoRA 学到的权重改变量，最多只能同时表达 $r$ 个互相独立的调整方向，多一个都不行。"

**为什么是这个形式**：$\Delta W = BA$ 是两个矩阵的乘积，其中 $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$，线性代数的基本结论是"两个矩阵乘积的秩不超过参与乘法的任意一个矩阵的秩"，而 $B$、$A$ 的秩本身都不超过 $r$（因为它们只有 $r$ 列/行），所以乘积的秩也被 $r$ 锁死。

**矩阵秩的含义**：秩为 $r$ 的矩阵只能表达 $r$ 个线性独立的"变化方向"。如果任务需要模型在 $r$ 个以上的独立方向上调整，LoRA 就力不从心了。
:::

### 1.2 什么任务需要高秩更新？

| 任务类型 | 所需秩 | LoRA 表现 |
|---------|--------|----------|
| 风格转换（格式/语气） | 低 | 好 |
| 分类/NLU | 低~中 | 好 |
| 代码生成 | 中 | 好 |
| **新知识记忆** | **高** | **差** |
| **多语言翻译** | **高** | **中** |
| **大量实体关系编码** | **高** | **差** |

**直觉解释**：
- 风格转换只需要改变"怎么说"→ 几个方向就够
- 新知识记忆需要改变"知道什么"→ 每条知识可能需要一个独立方向

### 1.3 增大 $r$ 不是解决方案

如果增大 $r$ 到 1024 或更大：
- 参数量暴增（$r=1024$ 时约 640M 参数）
- 失去了 PEFT 的意义
- 且由于 [rsLoRA](./062_rsLoRA_秩稳定缩放) 指出的缩放问题，大 $r$ 的利用效率很低

**MoRA 的思路**：在**不增加参数量**的前提下，实现更高秩的更新。

---

## 二、方法详解

### 2.1 核心思想

LoRA 用两个瘦长矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 相乘：
- 参数量：$dr + rk$
- 结果秩：$\leq r$

MoRA 的替代方案：用一个**方阵** $M \in \mathbb{R}^{r' \times r'}$ 配合降维/升维操作：
- 参数量：${r'}^2 \approx dr + rk$（相同预算）
- 结果秩：$\leq r'$（$r' \gg r$！）

**关键洞察**：给定参数预算 $P = dr + rk$，如果使用方阵 $M$，其边长为：

$$
r' = \sqrt{P} = \sqrt{dr + rk} = \sqrt{r(d+k)}
$$

**这个公式在做什么**：在参数量和 LoRA 完全相同的前提下，算出用方阵表示时，方阵的边长 $r'$ 能有多大——这个 $r'$ 就是 MoRA 能达到的秩上限。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $P=dr+rk$ | **参数预算** | LoRA 用两个瘦长矩阵消耗的总参数量，作为"公平对比"的基准 |
| $\sqrt{P}$ | **等预算下的方阵边长** | 一个 $r'\times r'$ 的方阵恰好用掉 $P$ 个参数时，边长该是多少 |
| $\sqrt{r(d+k)}$ | **化简后的表达式** | 把 $P=r(d+k)$ 代入根号，方便直接用 $r,d,k$ 算出结果 |

**用人话读**："把 LoRA 花的参数量原封不动地拿来搭一个正方形矩阵，这个正方形的边长就是 MoRA 能达到的新秩上限。"

**为什么是这个形式**：一个 $r'\times r'$ 的方阵秩最高可以等于 $r'$（满秩），参数量是 $r'^2$；只要让 $r'^2=P=r(d+k)$，就能在同样的参数预算下把秩上限从 $r$ 提升到 $\sqrt{r(d+k)}$——因为 $d,k$ 通常远大于 $r$，这个新秩上限往往是原来的几十倍。
:::

以 $d = k = 4096$, $r = 8$ 为例：
- LoRA 参数量：$8 \times 4096 + 8 \times 4096 = 65536$
- MoRA 方阵边长：$r' = \sqrt{65536} = 256$
- **MoRA 的 $\Delta W$ 秩可以达到 256**，而 LoRA 只能达到 8！

### 2.2 MoRA 的前向传播

由于 $M \in \mathbb{R}^{r' \times r'}$ 而输入是 $x \in \mathbb{R}^k$, 输出需要 $\mathbb{R}^d$，需要降维和升维操作：

$$
h = W_0 x + \text{Decompress}(M \cdot \text{Compress}(x))
$$

**这个公式在做什么**：算出 MoRA 层的实际前向输出——原模型的计算结果，加上"压缩输入 → 方阵变换 → 解压回原维度"这条新增路径的贡献。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $W_0 x$ | **原模型的输出** | 冻结不动的原始权重对输入 $x$ 的计算结果 |
| $\text{Compress}(x)$ | **降维安检口** | 把 $k$ 维输入压缩到方阵能处理的 $r'$ 维 |
| $M\cdot\text{Compress}(x)$ | **方阵内部加工** | 在低维空间里对压缩后的向量做一次可学习的线性变换 |
| $\text{Decompress}(\cdot)$ | **升维出口** | 把 $r'$ 维的变换结果重新展开回 $d$ 维，好和 $W_0x$ 维度对齐 |
| $h$ | **最终输出** | 原模型输出和新增路径输出相加 |

**用人话读**："输入先走原模型正常算一遍，同时另开一条小路：先把输入压小、送进方阵里变换、再撑大回原来的维度，两条路的结果加在一起就是最终输出。"

**为什么需要 Compress/Decompress**：方阵 $M$ 是 $r'\times r'$ 的，而输入是 $k$ 维、输出要求是 $d$ 维，$r'$ 通常和 $d,k$ 都不相等，所以必须先把输入压缩到 $r'$ 维才能喂给 $M$，算完之后再把 $r'$ 维的结果展开回 $d$ 维。
:::

其中：
- $\text{Compress}: \mathbb{R}^k \to \mathbb{R}^{r'}$（将 $k$ 维输入压缩到 $r'$ 维）
- $\text{Decompress}: \mathbb{R}^{r'} \to \mathbb{R}^d$（将 $r'$ 维输出展开到 $d$ 维）

### 2.3 压缩和解压缩的实现

论文探索了多种压缩/解压缩策略：

**策略 1：截断（Truncation）**
$$
\text{Compress}(x) = x_{1:r'} \quad \text{（取前 } r' \text{ 维）}
$$

**这个公式在做什么**：最简单粗暴的压缩方式——直接砍掉输入向量后面的部分，只保留前 $r'$ 维。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $x$ | **完整输入** | 原始的 $k$ 维输入向量 |
| $x_{1:r'}$ | **保留的前半部分** | 只取输入向量的前 $r'$ 个维度 |
| $\text{Compress}(x)$ | **压缩结果** | 直接等于截取出来的前 $r'$ 维，作为送进方阵 $M$ 的输入 |

**用人话读**："压缩就是直接砍掉输入向量后面 $k-r'$ 维，只留前面 $r'$ 维。"

**为什么是这个形式**：这是最容易实现的压缩方式，几乎不需要额外计算，但代价是后 $k-r'$ 维的信息被完全丢弃，没有参与任何计算——这正是下一段说的缺点。
:::

简单但丢弃了后 $k - r'$ 维的信息。

**策略 2：分组求和（Grouped Sum）**

将 $k$ 维输入分成 $r'$ 组，每组求和：
$$
\text{Compress}(x)_i = \sum_{j \in \text{group}_i} x_j
$$

**这个公式在做什么**：把输入的 $k$ 个维度分成 $r'$ 组，每组内部所有值加起来，得到压缩结果的第 $i$ 维——这样每一维原始输入都参与了计算，不会被直接丢弃。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\text{group}_i$ | **第 $i$ 个小组** | 把 $k$ 个维度平均分成 $r'$ 组后，属于第 $i$ 组的那些维度下标 |
| $x_j$ | **组内成员** | 属于第 $i$ 组的某个原始输入维度的值 |
| $\sum_{j\in\text{group}_i}x_j$ | **组内投票汇总** | 把这一组所有维度的值加起来，得到一个标量 |
| $\text{Compress}(x)_i$ | **压缩结果的第 $i$ 维** | 把上面这个组内求和的结果，作为压缩后向量的第 $i$ 个分量 |

**用人话读**："把输入的所有维度分成 $r'$ 个小组，每组内部的数值全部加起来，得到 $r'$ 个数——这就是压缩后的向量。"

**为什么是这个形式**：相比截断策略直接丢弃后面的维度，分组求和让**每一个**原始维度都对压缩结果有贡献（哪怕只是被加进了某个组的总和里），信息利用更充分；代价是求和是"多对一"的不可逆操作，没法从压缩结果精确还原原始输入。
:::

保留了所有维度的信息（通过加和），但不可逆。

**策略 3：共享行/列操作**

将输入 reshape 为矩阵形式再处理。

**论文推荐策略**：分组求和（简单有效）。

### 2.4 完整流程

```mermaid
flowchart LR
    X["输入 x ∈ ℝ^k"] --> Comp["Compress<br/>ℝ^k → ℝ^r'"]
    Comp --> M["方阵 M ∈ ℝ^(r'×r')"]
    M --> Decomp["Decompress<br/>ℝ^r' → ℝ^d"]
    Decomp --> Plus["⊕"]
    X --> W0["W₀x (冻结)"]
    W0 --> Plus
    Plus --> H["输出 h ∈ ℝ^d"]
    
    style W0 fill:#ddd,stroke:#666
    style M fill:#bfb,stroke:#393
```

### 2.5 秩分析

**定理**：$\Delta W = \text{Decompress} \circ M \circ \text{Compress}$ 的秩为：

$$
\text{rank}(\Delta W) = \min(\text{rank}(M), \text{rank(Compress)}, \text{rank(Decompress)})
$$

**这个公式在做什么**：算出 MoRA 整条"压缩 → 方阵变换 → 解压"链路最终能达到的秩——由链路中最"瘦"的那一环决定。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\text{rank}(M)$ | **方阵的秩** | 中间可学习矩阵 $M$ 本身能表达的独立方向数 |
| $\text{rank(Compress)}$ | **压缩环节的秩** | 压缩操作本身作为一个线性映射的秩 |
| $\text{rank(Decompress)}$ | **解压环节的秩** | 解压操作本身作为一个线性映射的秩 |
| $\min(\cdot,\cdot,\cdot)$ | **短板效应** | 三者中最小的那个，决定了整条链路的最终秩 |

**用人话读**："压缩、方阵变换、解压这三步串联起来，整条链路能表达的独立方向数，取决于这三步里最'瘦'的那一步。"

**为什么是这个形式**：这是线性代数里"复合映射的秩不超过任意一环的秩"这一性质的直接应用——三个线性映射依次作用，最终能保留的独立方向数不可能超过其中任何一环，所以取三者的最小值。这也解释了为什么 Compress/Decompress 的设计很重要：只要它们不是"瓶颈"（即秩足够高），整条链路的秩就完全由 $M$ 决定，能达到 $r'$。
:::

- 如果 Compress 和 Decompress 都是满秩映射（如分组求和 + 重复展开）
- 则 $\text{rank}(\Delta W) = \text{rank}(M)$，可达 $r'$

与 LoRA 对比（相同参数预算 $P$）：
- LoRA 最大秩：$r = P / (d + k)$
- MoRA 最大秩：$r' = \sqrt{P}$

以 $P = 65536$, $d = k = 4096$：
- LoRA：$r = 65536 / 8192 = 8$
- MoRA：$r' = \sqrt{65536} = 256$ → **32 倍的秩上限！**

---

## 三、实验结果

### 3.1 记忆任务

在知识密集型任务上（需要记忆新事实）：

| 方法 | 参数量 | UUID 记忆准确率 | 长上下文记忆 | 知识问答 |
|------|--------|----------------|------------|---------|
| LoRA ($r=8$) | 65K | 12.3% | 45.2% | 62.1% |
| LoRA ($r=256$) | 2M | 48.5% | 68.7% | 71.3% |
| **MoRA (同 $r=8$ 预算)** | **65K** | **38.7%** | **62.3%** | **68.5%** |
| **MoRA (同 $r=256$ 预算)** | **2M** | **72.1%** | **81.5%** | **78.2%** |

**MoRA 在记忆任务上远超 LoRA**——特别是在相同参数预算下。

### 3.2 通用任务

在 NLU/NLG 通用任务上：

| 方法 | MMLU | HellaSwag | ARC | 平均 |
|------|------|-----------|-----|------|
| LoRA ($r=16$) | 44.5 | 78.1 | 77.8 | 66.8 |
| **MoRA** (同预算) | 44.2 | 77.8 | 77.5 | 66.5 |

**在通用任务上 MoRA 与 LoRA 持平**——高秩不一定比低秩好（通用任务确实不需要高秩）。

### 3.3 总结规律

| 任务类型 | 更适合的方法 | 原因 |
|---------|------------|------|
| 新知识记忆 | **MoRA** | 需要高秩编码独立事实 |
| 风格/格式适配 | LoRA | 低秩足够 |
| 推理/理解 | 两者相当 | 不依赖权重秩 |
| 多语言翻译 | **MoRA** | 每种语言可能需要独立方向 |

---

## 四、MoRA 的代价

### 4.1 不能像 LoRA 那样合并

LoRA：$W_{\text{merged}} = W_0 + BA$（直接矩阵加法，推理零开销）

MoRA：$\Delta W = \text{Decompress}(M \cdot \text{Compress}(\cdot))$ 不是一个简单的矩阵加法——因为 Compress 和 Decompress 是非线性操作（如分组求和/展开）。

**解决方案**：训练完成后，可以用 $W_0 + \Delta W_{\text{approx}}$ 近似合并：
1. 计算完整的 $\Delta W$ 矩阵（$d \times k$）
2. 加到 $W_0$ 上

这需要临时用 $O(dk)$ 内存计算一次，但之后推理就是零开销了。

### 4.2 训练速度

方阵乘法 $M \cdot z$（$r' \times r'$）的计算量 vs LoRA 的两步乘法 $B(Ax)$：

- LoRA：$O(r \cdot k + r \cdot d)$ = $O(r(d+k))$
- MoRA：$O({r'}^2)$ = $O(r(d+k))$（相同参数预算时相等）

加上 Compress/Decompress 的开销，MoRA 略慢但差异不大。

---

## 五、代码实现

```python
import torch
import torch.nn as nn
import math

class MoRALinear(nn.Module):
    """MoRA: 高秩更新的参数高效微调"""
    
    def __init__(self, original_linear: nn.Linear, r: int = 8):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        
        d, k = original_linear.out_features, original_linear.in_features
        
        # 计算等效参数预算下的方阵大小
        lora_params = d * r + r * k  # LoRA 同等预算
        r_prime = int(math.sqrt(lora_params))
        self.r_prime = r_prime
        self.d = d
        self.k = k
        
        # 可训练的方阵
        self.M = nn.Parameter(torch.zeros(r_prime, r_prime))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        self.M.data *= 0.01  # 缩小初始值
        
        # 计算分组参数
        self.groups_in = k // r_prime if k >= r_prime else 1
        self.groups_out = d // r_prime if d >= r_prime else 1
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """将 k 维输入压缩到 r' 维（分组求和）"""
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(*batch_shape, self.r_prime, -1)  # 分组
        return x_flat.sum(dim=-1)  # 组内求和 → [batch, r']
    
    def decompress(self, z: torch.Tensor) -> torch.Tensor:
        """将 r' 维输出展开到 d 维（重复展开）"""
        batch_shape = z.shape[:-1]
        return z.unsqueeze(-1).expand(*batch_shape, self.r_prime, 
               self.d // self.r_prime).reshape(*batch_shape, self.d)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.original(x)
        
        # MoRA 路径
        compressed = self.compress(x)         # [batch, seq, r']
        transformed = compressed @ self.M.T   # [batch, seq, r']
        decompressed = self.decompress(transformed)  # [batch, seq, d]
        
        return h + decompressed
```

---

## 六、总结

### 核心贡献

1. **指出了 LoRA 低秩约束在记忆任务上的根本限制**
2. **提出了方阵映射方案**：相同参数下实现 32x 更高的秩
3. **在记忆密集型任务上大幅超越 LoRA**
4. **在通用任务上保持竞争力**

### MoRA vs LoRA 选择指南

- **需要精确记忆大量新信息** → 选 MoRA
- **风格/格式/推理能力适配** → 选 LoRA
- **需要推理零开销** → 选 LoRA（或训练后合并 MoRA）
- **多任务快速切换** → 选 LoRA（适配器更小）

### 延伸阅读

- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — 低秩约束的含义
- [AdaLoRA 精读](./057_AdaLoRA_自适应秩分配) — 自适应秩选择
- [PiSSA 精读](./066_PiSSA_主成分初始化LoRA) — 另一种最大化 LoRA 效果的方法
