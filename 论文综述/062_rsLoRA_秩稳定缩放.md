---
title: "rsLoRA: A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"
order: 62
tags: [LoRA, rsLoRA, 缩放因子, 参数高效微调]
category: 精读
---

# rsLoRA: A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA

> **论文信息**：Kalajdzievski, 2024  
> **一句话概括**：标准 LoRA 的缩放因子 $\frac{\alpha}{r}$ 在增大秩 $r$ 时会导致每个 rank-1 分量的贡献被过度压缩，使得大 $r$ 的 LoRA 难以有效利用额外容量。rsLoRA 将缩放因子改为 $\frac{\alpha}{\sqrt{r}}$，使不同 $r$ 的 LoRA 在初始化阶段具有稳定的输出方差——大 $r$ 终于能发挥全部潜力。

**相关阅读**：
- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — LoRA 的 $\alpha/r$ 缩放
- [LoRA+ 精读](./058_LoRAPlus_不同学习率适配) — 另一种让 LoRA 训练更高效的方法

---

## 贯穿全文的例子

> 在 LLaMA-7B 上做代码生成任务。我们先用 $r=8$ 的 LoRA 微调，效果不错。为了进一步提升性能，我们把 $r$ 增大到 64。
>
> **标准 LoRA 的困境**：$r=64$ 的 LoRA 效果居然没比 $r=8$ 好多少！甚至在某些 seed 下更差。为什么给了更多参数反而没用？
>
> **原因**：标准 LoRA 的缩放 $\frac{\alpha}{r}$ 导致 $r=64$ 时每个 rank-1 分量的影响力只有 $r=8$ 时的 $\frac{8}{64} = \frac{1}{8}$。大部分额外秩分量被"静音"了。
>
> **rsLoRA 的解决**：将缩放改为 $\frac{\alpha}{\sqrt{r}}$，让每个分量在不同 $r$ 下都有合适的初始影响力。$r=64$ 的 rsLoRA 终于能显著超越 $r=8$。

---

## 一、论文动机：为什么 LoRA 增大 $r$ 效果不涨？

### 1.1 LoRA 的"秩饱和"现象

LoRA 原始论文已经观察到：$r$ 从 4 增大到 256，WikiSQL 准确率几乎没变（73.4 → 73.1）。当时的解释是"微调本身只需要极低的秩"。

但 rsLoRA 的作者提出了另一个解释：**不是不需要大 $r$，而是标准缩放因子让大 $r$ 失效了。**

### 1.2 问题根源：信号强度与 $r$ 的关系

考虑 LoRA 的输出（训练初始阶段）：

$$
h_{\text{lora}} = \frac{\alpha}{r} \cdot BAx = \frac{\alpha}{r} \sum_{i=1}^{r} B_{:,i} \cdot (A_{i,:} \cdot x)
$$

这是 $r$ 个 rank-1 项的加权求和。关键问题：**这个求和的总方差是多少？**

假设 $A_{i,j} \sim \mathcal{N}(0, 1/k)$（Kaiming 初始化），$B$ 初始化为零（所以前几步分析需要看梯度更新后的 $B$）。

对于训练几步后的 $B$（假设每个元素量级为 $\sigma_B$），输出方差：

$$
\text{Var}[h_{\text{lora}}] \approx \left(\frac{\alpha}{r}\right)^2 \cdot r \cdot \sigma_B^2 = \frac{\alpha^2 \cdot \sigma_B^2}{r}
$$

**注意**：方差与 $r$ 成反比！$r$ 越大，LoRA 分支的信号越弱。

### 1.3 直觉理解

**类比**：想象你有一支由 $r$ 人组成的团队，团队预算固定为 $\alpha$。
- 标准 LoRA 的 $\frac{\alpha}{r}$：每人分到 $\frac{\alpha}{r}$ 的预算。团队越大，每人分到的越少。
- 当团队足够大时，每人的预算少到做不了什么有意义的事 → 加人无用。
- rsLoRA 的 $\frac{\alpha}{\sqrt{r}}$：总预算随 $\sqrt{r}$ 增长 → 每人分到 $\frac{\alpha}{\sqrt{r} \cdot r} = \frac{\alpha}{r^{3/2}}$... 不对，让我重新理解。

**正确的理解**：缩放因子作用于整个求和结果。$r$ 个 i.i.d. 随机变量的求和标准差正比于 $\sqrt{r}$。为了让输出量级稳定：
- 求和贡献 $\sqrt{r}$ 的方差增长
- 缩放应该除以 $\sqrt{r}$ 来抵消 → $\frac{\alpha}{\sqrt{r}}$

标准 LoRA 除以 $r$（过度缩放），导致 $r$ 大时信号过弱。

---

## 二、方法详解

### 2.1 rsLoRA 缩放因子

将 LoRA 的缩放从 $\frac{\alpha}{r}$ 改为 $\frac{\alpha}{\sqrt{r}}$：

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}} \cdot BAx
$$

**就这一个改动。** 没有其他架构修改。

### 2.2 数学推导

**目标**：不论 $r$ 取什么值，LoRA 分支输出的方差应该保持不变（称为"秩稳定"条件）。

设 $A$ 的元素为 $\mathcal{N}(0, \sigma_A^2)$，$B$的元素为 $\mathcal{N}(0, \sigma_B^2)$（分析训练过程中某时刻的统计量），缩放因子为 $\gamma(r)$：

$$
h_{\text{lora}} = \gamma(r) \cdot BAx
$$

$BAx$ 的单个输出元素可以写为：
$$
[BAx]_j = \sum_{i=1}^{r} B_{j,i} \cdot [Ax]_i = \sum_{i=1}^r B_{j,i} \sum_{l=1}^k A_{i,l} x_l
$$

假设各项独立：
$$
\text{Var}[BAx]_j = r \cdot \sigma_B^2 \cdot k \cdot \sigma_A^2 \cdot \sigma_x^2 = r \cdot C
$$

其中 $C = \sigma_B^2 \cdot k \cdot \sigma_A^2 \cdot \sigma_x^2$ 是与 $r$ 无关的常数。

要使 $\text{Var}[\gamma(r) \cdot BAx] = \gamma(r)^2 \cdot r \cdot C$ 与 $r$ 无关：

$$
\gamma(r)^2 \cdot r = \text{const} \implies \gamma(r) = \frac{\text{const}}{\sqrt{r}}
$$

所以最优缩放是 $\gamma(r) \propto \frac{1}{\sqrt{r}}$，即 $\frac{\alpha}{\sqrt{r}}$。

### 2.3 标准 LoRA 为什么用 $\frac{\alpha}{r}$？

LoRA 原始论文使用 $\frac{\alpha}{r}$ 的初衷是：
- 固定 $\alpha$，调 $r$ 时不需要调学习率
- 在小 $r$（4~8）的实验中这个缩放工作得还行

但从信号传播的角度，$\frac{\alpha}{r}$ 是过度缩放。在 $r$ 小时问题不严重（因为 $1/r$ 和 $1/\sqrt{r}$ 差距不大），但 $r$ 大时（如 64~256）问题就很明显了。

| $r$ | $\alpha/r$ | $\alpha/\sqrt{r}$ | 比值 |
|-----|-----------|-------------------|------|
| 4 | 0.25 | 0.5 | 2x |
| 8 | 0.125 | 0.354 | 2.83x |
| 16 | 0.0625 | 0.25 | 4x |
| 64 | 0.0156 | 0.125 | 8x |
| 256 | 0.0039 | 0.0625 | 16x |

（假设 $\alpha = 1$ 用于对比）

可以看到 $r=256$ 时，标准缩放比 rsLoRA 弱 16 倍！

---

## 三、实验结果

### 3.1 不同 $r$ 下的效果对比

在 LLaMA-7B 上做指令微调，对比不同 $r$：

| $r$ | LoRA (标准 $\alpha/r$) | rsLoRA ($\alpha/\sqrt{r}$) | 差距 |
|-----|----------------------|---------------------------|------|
| 4 | 5.71 | 5.74 | +0.03 |
| 8 | 5.76 | 5.82 | +0.06 |
| 16 | 5.82 | 5.93 | +0.11 |
| 32 | 5.85 | 6.01 | +0.16 |
| 64 | 5.87 | **6.12** | **+0.25** |
| 128 | 5.86 | **6.18** | **+0.32** |

**关键发现**：
1. 标准 LoRA 在 $r > 16$ 后基本饱和（从 5.82 只涨到 5.87）
2. rsLoRA 在 $r$ 增大时持续提升（从 5.74 涨到 6.18）
3. $r$ 越大，rsLoRA 的优势越明显
4. rsLoRA 让大 $r$ 终于能发挥作用了

### 3.2 训练收敛速度

| 方法 | 达到 5.8 分所需步数 |
|------|-------------------|
| LoRA ($r=16$) | 8000 步 |
| rsLoRA ($r=16$) | 6000 步 |
| rsLoRA ($r=64$) | 4000 步 |

rsLoRA 不仅最终效果更好，收敛速度也更快。

---

## 四、与其他改进的兼容性

rsLoRA 的改动极小（只改缩放因子），与几乎所有 LoRA 变体兼容：

| 组合 | 兼容性 | 效果 |
|------|--------|------|
| rsLoRA + QLoRA | ✅ 完全兼容 | 量化模型 + 秩稳定缩放 |
| rsLoRA + LoRA+ | ✅ 完全兼容 | 秩稳定 + 不同学习率 |
| rsLoRA + DoRA | ✅ 兼容 | 方向-幅度分解 + 秩稳定 |
| rsLoRA + AdaLoRA | ⚠️ 需小心 | 秩在变化，缩放也要相应调整 |

---

## 五、代码实现

```python
import torch
import torch.nn as nn
import math

class rsLoRALinear(nn.Module):
    """rsLoRA: 使用 α/√r 缩放代替 α/r"""
    
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 16):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        
        d, k = original_linear.out_features, original_linear.in_features
        self.r = r
        
        # 关键区别：缩放因子用 √r 而不是 r
        self.scaling = alpha / math.sqrt(r)  # rsLoRA!
        
        self.A = nn.Parameter(torch.empty(r, k))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.original(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return h + self.scaling * lora_out

# Hugging Face PEFT 中使用 rsLoRA（从 PEFT v0.9.0 开始支持）
from peft import LoraConfig
config = LoraConfig(
    r=64,
    lora_alpha=64,
    use_rslora=True,  # 启用 rsLoRA 缩放！
    target_modules="all-linear",
)
```

---

## 六、总结

### 核心贡献

1. **诊断了 LoRA 秩饱和的真正原因**：缩放因子 $\frac{\alpha}{r}$ 过度压缩了大 $r$ 的信号
2. **提出了极简修复**：将 $\frac{\alpha}{r}$ 改为 $\frac{\alpha}{\sqrt{r}}$
3. **解锁了大 $r$ 的潜力**：让增加秩真正能带来效果提升
4. **理论严谨**：基于方差稳定性的数学推导

### 实践建议

- 如果你使用 $r \leq 8$：rsLoRA 与标准 LoRA 差异不大，可以不用
- 如果你使用 $r \geq 16$：**强烈建议使用 rsLoRA**
- 如果你在搜索最优 $r$：使用 rsLoRA 后 $r$ 的搜索更可预测（大 $r$ 一定不差于小 $r$）

### 延伸阅读

- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — 缩放因子的基础定义
- [LoRA+ 精读](./058_LoRAPlus_不同学习率适配) — 学习率优化
- [DoRA 精读](./059_DoRA_权重分解低秩适配) — 权重分解优化
