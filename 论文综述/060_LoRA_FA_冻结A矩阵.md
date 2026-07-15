---
title: "LoRA-FA: Memory Efficient Low-Rank Adaptation for Large Language Models Fine-Tuning"
order: 60
tags: [LoRA, LoRA-FA, 显存优化, 参数高效微调]
category: 精读
---

# LoRA-FA: Memory Efficient Low-Rank Adaptation for Large Language Models Fine-Tuning

> **论文信息**：Zhang et al., 2023  
> **一句话概括**：标准 LoRA 中 $A$ 和 $B$ 都是可训练的，但实验发现**冻结 $A$（随机初始化后不再更新）、只训练 $B$**，效果几乎不变，而激活值显存减少近一半——因为冻结的 $A$ 不需要存储激活值用于反向传播。

**相关阅读**：
- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — LoRA 中 $A, B$ 的作用
- [LoRA+ 精读](./058_LoRAPlus_不同学习率适配) — A、B 的不对称学习动力学

---

## 贯穿全文的例子

> 在 4 张 A100-40GB 上微调 LLaMA-65B（使用标准 LoRA，$r=16$，对所有线性层）。
>
> **瓶颈**：即使用了 LoRA，long sequence（4096 tokens）训练时激活值仍占大量显存
> - 标准 LoRA 激活值显存：~每层需要存 $x$（给 $A$ 的反向传播）和 $Ax$（给 $B$ 的反向传播）
> - **LoRA-FA 激活值显存**：只需要存 $Ax$（因为 $A$ 冻结不需要梯度，不需要存 $x$）
>
> 这让 batch size 可以翻倍，或者用更少的 GPU 完成同样规模的训练。

---

## 一、论文动机：LoRA 的激活值显存瓶颈

### 1.1 大家忽视的显存消耗：激活值

讨论 LoRA 显存时，人们通常关注：
- ✅ 参数量（很少，不是问题）
- ✅ 优化器状态（很少，不是问题）
- ❌ **激活值**（被低估的大头！）

**什么是激活值？** 在反向传播中，计算梯度需要用到前向传播时的中间结果。这些中间结果必须存储在显存中。

对于标准 LoRA 的分支 $h_{\text{lora}} = BAx$：
- 计算 $\frac{\partial \mathcal{L}}{\partial A}$ 需要用到 $x$（输入）和 $B$
- 计算 $\frac{\partial \mathcal{L}}{\partial B}$ 需要用到 $Ax$（A 的输出）

所以需要存储：
1. $x \in \mathbb{R}^{b \times s \times k}$（完整输入）
2. $Ax \in \mathbb{R}^{b \times s \times r}$（降维后的中间激活）

其中 $b$ 是 batch size，$s$ 是序列长度。

### 1.2 数值分析

以 LLaMA-7B, $r=16$, $b=4$, $s=2048$ 为例：

| 存储项 | 形状 | 大小（BF16） |
|--------|------|-------------|
| $x$（为 A 的梯度） | $4 \times 2048 \times 4096$ | 64 MB/层 |
| $Ax$（为 B 的梯度） | $4 \times 2048 \times 16$ | 0.25 MB/层 |
| **总计/层** | | **64.25 MB/层** |
| **32 层总计** | | **~2 GB** |

如果**冻结 A**，则不需要存 $x$：
| 存储项 | 形状 | 大小 |
|--------|------|------|
| $Ax$（为 B 的梯度） | $4 \times 2048 \times 16$ | 0.25 MB/层 |
| **32 层总计** | | **~8 MB** |

**节省**：从 ~2 GB 降到 ~8 MB，节省了 **99.6%** 的 LoRA 分支激活值显存！

### 1.3 为什么冻结 A 而不冻结 B？

回顾初始化：$A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$

- **冻结 A**：A 作为随机投影矩阵，将输入投影到随机低维子空间。B 负责在这个子空间中学习有意义的映射
- **冻结 B**：B 初始化为零，如果冻结 B=0，则 $BA = 0$ 永远为零，什么也学不到

所以**必须冻结 A 而不是 B**。而且，随机投影的理论（Johnson-Lindenstrauss 定理）告诉我们：随机高斯矩阵是一个很好的降维映射，它可以近似保持距离关系。

---

## 二、方法详解

### 2.1 LoRA-FA 的前向传播

与标准 LoRA 完全相同：

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

唯一区别：$A$ 没有 `requires_grad`。

### 2.2 梯度计算

标准 LoRA 的梯度：
$$
\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T \quad \text{(需要存储 } x \text{)}
$$
$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T \quad \text{(需要存储 } Ax \text{)}
$$

LoRA-FA 只需要计算第二个：
$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T \quad \text{(只需要存储 } Ax \text{)}
$$

### 2.3 为什么 $Ax$ 很"便宜"？

$Ax$ 的形状是 $\mathbb{R}^{b \times s \times r}$，而 $x$ 的形状是 $\mathbb{R}^{b \times s \times k}$。

$$
\frac{\text{size}(Ax)}{\text{size}(x)} = \frac{r}{k} = \frac{16}{4096} = 0.39\%
$$

存储 $Ax$ 的开销与存储 $x$ 相比可以忽略不计。

---

## 三、理论支撑

### 3.1 随机投影理论

LoRA-FA 的有效性可以用随机投影理论来解释：

**Johnson-Lindenstrauss 引理**：如果 $A \in \mathbb{R}^{r \times k}$ 的每个元素是 i.i.d. $\mathcal{N}(0, 1/r)$，则对任意 $x, y \in \mathbb{R}^k$：

$$
(1-\epsilon) \|x-y\|^2 \leq \|Ax - Ay\|^2 \leq (1+\epsilon) \|x-y\|^2
$$

以高概率成立（只要 $r = O(\log n / \epsilon^2)$）。

**含义**：随机固定的 $A$ 可以保持输入特征之间的距离关系。所以 $B$ 看到的 $Ax$ 仍然包含了区分不同输入所需的信息。

### 3.2 实证验证

论文通过以下实验验证了冻结 A 的合理性：

1. 训练标准 LoRA → 分析训练完成后 $A$ 相对于初始化的变化量
2. 发现 $\|A_{\text{final}} - A_{\text{init}}\| / \|A_{\text{init}}\|$ 很小（通常 <5%）
3. 这说明 $A$ 在训练中没有发生显著变化 → 冻结它损失很小

---

## 四、实验结果

### 4.1 语言模型微调

在 LLaMA 系列上的对比：

| 方法 | 模型 | 可训练参数 | 激活值显存 | MMLU | HellaSwag |
|------|------|-----------|-----------|------|-----------|
| LoRA | 7B | 20M | ~2 GB | 38.4 | 78.1 |
| **LoRA-FA** | 7B | **10M** | **~8 MB** | **38.2** | **77.8** |
| LoRA | 13B | 40M | ~4 GB | 44.5 | 82.3 |
| **LoRA-FA** | 13B | **20M** | **~16 MB** | **44.1** | **82.0** |

**结论**：性能下降不到 0.5%，但激活值显存节省了两个数量级。

### 4.2 显存使用对比

| 配置 | LoRA | LoRA-FA | 节省 |
|------|------|---------|------|
| LLaMA-7B, bs=4, seq=2048 | 32.5 GB | 30.8 GB | 5% |
| LLaMA-7B, bs=4, seq=4096 | 45.2 GB | 41.3 GB | 9% |
| LLaMA-13B, bs=4, seq=4096 | 67.8 GB | 59.2 GB | 13% |
| LLaMA-65B, bs=2, seq=2048 | 156 GB | 134 GB | 14% |

**趋势**：序列越长、模型越大，LoRA-FA 的相对节省越明显。这是因为长序列时激活值占显存的比例更大。

### 4.3 与梯度检查点的对比

梯度检查点（Gradient Checkpointing）也是减少激活值显存的方法：

| 方法 | 激活值显存 | 计算开销 | 效果 |
|------|-----------|---------|------|
| 无优化 | 100% | 1x | 基线 |
| 梯度检查点 | ~30% | 1.3x（重计算） | 相同 |
| **LoRA-FA** | ~50% | **1x**（无额外计算） | 几乎相同 |
| **LoRA-FA + 梯度检查点** | **~15%** | 1.3x | 几乎相同 |

LoRA-FA 可以与梯度检查点组合使用，进一步节省显存。

---

## 五、代码实现

```python
import torch
import torch.nn as nn
import math

class LoRAFALinear(nn.Module):
    """LoRA-FA: 冻结 A，只训练 B"""
    
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        d, k = original_linear.out_features, original_linear.in_features
        self.r = r
        self.scaling = alpha / r
        
        # A 是冻结的随机投影（不需要梯度！）
        self.A = nn.Parameter(torch.empty(r, k), requires_grad=False)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
        # B 是唯一可训练的 LoRA 参数
        self.B = nn.Parameter(torch.zeros(d, r))  # 初始化为零
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.original(x)
        
        # LoRA 路径：由于 A 无梯度，x 不需要被存储用于 A 的反向传播
        # 只有 Ax 需要被存储（用于 B 的反向传播），而 Ax 很小
        lora_out = (x @ self.A.T) @ self.B.T
        
        return h + self.scaling * lora_out
```

---

## 六、LoRA-FA 的局限与适用场景

### 6.1 局限

1. **可训练参数减半**：只有 $B$ 可训练，参数量从 $dr + rk$ 降到 $dr$
2. **表达能力略受限**：$A$ 的方向固定，$B$ 只能在随机投影后的空间中学习
3. **小 $r$ 时效果下降更明显**：$r$ 越小，随机投影的信息损失越大
4. **与 LoRA+ 的互动**：LoRA+ 的理论基础是 A 和 B 都在训练，LoRA-FA 不适用

### 6.2 最佳适用场景

- **显存极度紧张**（如单卡微调超大模型）
- **长序列训练**（如 8k~32k context length）
- **batch size 受限**（希望通过节省显存来增大 batch size）
- **对微小性能损失可以容忍**（如实验性探索、快速原型）

### 6.3 与其他方法的组合

| 组合 | 效果 | 推荐度 |
|------|------|--------|
| LoRA-FA + QLoRA | 4-bit 量化 + 冻结 A → 极致省显存 | ⭐⭐⭐⭐⭐ |
| LoRA-FA + 梯度检查点 | 双重减少激活值 | ⭐⭐⭐⭐ |
| LoRA-FA + 大 $r$ | 用更大的 r 补偿冻结 A 的表达损失 | ⭐⭐⭐ |
| LoRA-FA + DoRA | 方向更新受限（A 冻结），幅度仍独立 → 兼容但收益可能有限 | ⭐⭐ |

---

## 七、总结

### 核心贡献

1. **发现了 LoRA 中激活值显存的被低估问题**
2. **提出了极简的解决方案**：冻结 A 矩阵
3. **理论支撑**：随机投影理论保证了信息保持
4. **实验验证**：性能损失 <0.5%，显存节省显著

### 延伸阅读

- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — 基本原理
- [QLoRA 精读](./056_QLoRA_量化低秩适配) — 量化路径的显存优化
- [VeRA 精读](./063_VeRA_向量化秩一适配) — 将"冻结投影"思想推到极致
- [GaLore 精读](./061_GaLore_梯度低秩投影训练) — 另一种显存优化思路
