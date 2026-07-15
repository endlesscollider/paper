---
title: "LoRA+: Efficient Low Rank Adaptation of Large Models"
order: 58
tags: [LoRA, LoRA+, 学习率, 参数高效微调]
category: 精读
---

# LoRA+: Efficient Low Rank Adaptation of Large Models

> **论文信息**：Hayou et al., ICML 2024  
> **一句话概括**：LoRA 中的矩阵 $A$ 和 $B$ 承担不同的功能角色——$A$ 负责"特征提取方向"，$B$ 负责"映射回输出空间"。用相同学习率训练它们是次优的；给 $B$ 一个比 $A$ 大得多的学习率（通常 $\eta_B / \eta_A \approx 16$）可以让 LoRA 收敛更快、效果更好。

**相关阅读**：
- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — LoRA 的 $A, B$ 矩阵定义
- [LoRA 原始论文精读](./055_LoRA_低秩适配微调大模型) — LoRA 的完整技术细节

---

## 贯穿全文的例子

> 我们用 LoRA（$r=16$）微调 LLaMA-7B 做指令跟随任务。
>
> - **标准 LoRA**：$A$ 和 $B$ 使用相同学习率 $\eta = 2\text{e-}4$，训练 3 epochs 后收敛
> - **LoRA+**：$A$ 用 $\eta_A = 2\text{e-}4$，$B$ 用 $\eta_B = 16 \times 2\text{e-}4 = 3.2\text{e-}3$
>
> 结果：LoRA+ 在相同训练步数下 loss 更低，或者达到相同 loss 所需步数减少 ~2 倍。改动极小（只改学习率），但一致性地提升效果。

---

## 一、论文动机：A 和 B 是对称的吗？

### 1.1 标准 LoRA 的隐含假设

标准 LoRA 对 $A$ 和 $B$ 使用相同的学习率。这隐含地假设了两者具有对称的学习动力学。但这个假设成立吗？

回顾 LoRA 的设置：
- $A \in \mathbb{R}^{r \times k}$：从 $k$ 维输入空间投影到 $r$ 维低秩空间
- $B \in \mathbb{R}^{d \times r}$：从 $r$ 维低秩空间映射到 $d$ 维输出空间
- **初始化**：$A \sim \mathcal{N}(0, \sigma^2)$，$B = 0$

注意到 $A$ 和 $B$ 在几个方面是**不对称**的：
1. **初始化不同**：$A$ 是随机的，$B$ 是零
2. **功能不同**：$A$ 选择"关注输入的哪些方向"，$B$ 决定"如何映射到输出"
3. **维度通常不同**：如果 $d \neq k$（如 MLP 层的 gate 投影），两者形状不同

### 1.2 直觉：$B$ 的梯度信号更弱

考虑训练开始时（$B = 0$）的梯度：

$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot (Ax)^T
$$

$$
\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \cdot B^T \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^T
$$

在训练初期 $B \approx 0$ 时：
- $\frac{\partial \mathcal{L}}{\partial B}$ 的量级正常（由 $Ax$ 决定，而 $A$ 已初始化为非零）
- $\frac{\partial \mathcal{L}}{\partial A}$ 的量级很小（因为乘了 $B^T \approx 0$）

**但这不意味着 $A$ 的学习率应该更大**。论文的深入分析给出了相反的结论。

---

## 二、理论分析

### 2.1 无限宽度极限分析

论文使用**特征学习理论**（$\mu P$-like 分析）来研究 LoRA 在大模型极限下的最优学习率配比。

核心结论：

> **在宽度 $d \to \infty$ 的极限下，要使 LoRA 的特征学习效率最大化，$B$ 的学习率应该比 $A$ 的学习率大 $\Theta(d)$ 倍。**

**为什么？** 直觉解释：
- $A$ 需要学习的是"输入空间中的特征方向"——这是一个相对低维的问题
- $B$ 需要学习的是"如何将低秩表示映射到高维输出空间"——这需要在 $d$ 维空间中精确定位

可以类比为：
- $A$ 像是"选择用哪个滤镜看世界"（低维选择）
- $B$ 像是"用选好的滤镜精确绘制出结果"（高维映射）

后者需要更大的学习率来在有限步数内完成高维空间中的搜索。

### 2.2 最优学习率比值

论文通过理论推导给出了最优学习率比值的形式：

$$
\frac{\eta_B}{\eta_A} = \Theta\left(\frac{d}{r}\right)
$$

对于实际模型（如 LLaMA-7B，$d=4096$，$r=16$）：
$$
\frac{\eta_B}{\eta_A} \propto \frac{4096}{16} = 256
$$

但论文发现实际最优比值通常在 **$2 \sim 32$** 之间，推荐默认值为 **$\frac{\eta_B}{\eta_A} = 16$**。

**为什么实际比理论小？** 因为：
1. 有限宽度效应
2. Adam 优化器已经做了一定程度的自适应缩放
3. 非线性效应（理论是线性假设下推导的）

### 2.3 固定 $\eta_A$，增大 $\eta_B$

论文推荐的实践方案非常简单：

```python
# 标准 LoRA
optimizer = Adam([
    {'params': lora_A_params, 'lr': 2e-4},
    {'params': lora_B_params, 'lr': 2e-4},  # 相同学习率
])

# LoRA+
optimizer = Adam([
    {'params': lora_A_params, 'lr': 2e-4},       # η_A 不变
    {'params': lora_B_params, 'lr': 2e-4 * 16},  # η_B = 16 × η_A
])
```

**就这么简单。** 不需要修改架构、不需要额外超参数搜索、不增加计算成本。

---

## 三、实验结果

### 3.1 语言模型微调

在 LLaMA-7B 上的指令微调（Alpaca 数据集）：

| 方法 | 学习率设置 | MT-Bench 分数 | 训练步数至收敛 |
|------|-----------|---------------|--------------|
| LoRA | $\eta = 2\text{e-}4$ (均) | 5.82 | 10000 |
| LoRA+ ($\eta_B / \eta_A = 4$) | $\eta_A = 2\text{e-}4$ | 5.94 | 8000 |
| LoRA+ ($\eta_B / \eta_A = 16$) | $\eta_A = 2\text{e-}4$ | **6.05** | **7000** |
| LoRA+ ($\eta_B / \eta_A = 64$) | $\eta_A = 2\text{e-}4$ | 5.98 | 7500 |

**关键发现**：
- $\eta_B / \eta_A = 16$ 是最优sweet spot
- 比值过大（64）效果略下降（$B$ 更新过于激进）
- 收敛速度提升约 30%

### 3.2 不同模型规模

| 模型 | LoRA | LoRA+ ($\times 16$) | 提升 |
|------|------|---------------------|------|
| RoBERTa-Base (125M) | 90.3 | 90.7 | +0.4 |
| GPT-2 Medium (355M) | 23.1 (PPL) | 22.5 (PPL) | -0.6 |
| LLaMA-7B | 5.82 | 6.05 | +0.23 |
| LLaMA-13B | 6.21 | 6.42 | +0.21 |

**趋势**：模型越大，LoRA+ 的提升越一致。这与理论预测一致（$d$ 越大，不对称性越明显）。

### 3.3 不同 $r$ 值

| $r$ | LoRA | LoRA+ | $\eta_B / \eta_A$ 最优值 |
|-----|------|-------|------------------------|
| 4 | 5.71 | 5.88 | 32 |
| 8 | 5.76 | 5.95 | 16~32 |
| 16 | 5.82 | 6.05 | 16 |
| 32 | 5.87 | 6.08 | 8~16 |
| 64 | 5.90 | 6.09 | 8 |

**趋势**：$r$ 越小，最优比值越大。这也与理论一致：$\frac{\eta_B}{\eta_A} \propto \frac{d}{r}$，$r$ 小则比值大。

---

## 四、为什么 LoRA+ 有效？梯度动力学分析

### 4.1 训练初期的动力学

在 $B = 0$ 的初始化下，前几步更新：

**$A$ 的更新**：
$$
A_1 = A_0 - \eta_A \cdot \frac{\partial \mathcal{L}}{\partial A}\bigg|_{B=0} = A_0 - \eta_A \cdot \underbrace{0^T}_{B^T} \cdot (\cdots) = A_0
$$

**$A$ 在第一步根本没有更新！** 因为 $B=0$ 导致 $A$ 的梯度为零。

**$B$ 的更新**：
$$
B_1 = 0 - \eta_B \cdot \frac{\partial \mathcal{L}}{\partial B}\bigg|_{B=0} = -\eta_B \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot (A_0 x)^T
$$

$B$ 有非零梯度，可以开始学习。

**这意味着**：训练的前几步几乎完全是 $B$ 在更新。给 $B$ 更大的学习率可以加速这个"破冰"过程。

### 4.2 训练中后期

当 $B$ 变为非零后，$A$ 也开始接收梯度。但此时：
- $A$ 的梯度与 $B$ 的当前值成正比
- 如果 $B$ 已经通过大学习率快速收敛到一个好的方向，$A$ 就能基于稳定的 $B$ 高效学习

**类比**：$B$ 先"搭好脚手架"，$A$ 再在脚手架上精细施工。

---

## 五、代码实现

```python
import torch
from torch.optim import AdamW

def get_lora_plus_optimizer(model, lr_A=2e-4, lr_B=3.2e-3, weight_decay=0.01):
    """为 LoRA+ 构建不同学习率的优化器"""
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'lora_A' in name:
            param_groups.append({
                'params': [param],
                'lr': lr_A,
                'weight_decay': weight_decay,
            })
        elif 'lora_B' in name:
            param_groups.append({
                'params': [param],
                'lr': lr_B,  # B 的学习率更大！
                'weight_decay': weight_decay,
            })
        else:
            # 其他可训练参数（如分类头）
            param_groups.append({
                'params': [param],
                'lr': lr_A,
                'weight_decay': weight_decay,
            })
    
    return AdamW(param_groups)

# 使用
optimizer = get_lora_plus_optimizer(
    model, 
    lr_A=2e-4, 
    lr_B=2e-4 * 16  # ratio = 16
)
```

---

## 六、与其他学习率技巧的关系

| 方法 | 核心思想 | 与 LoRA+ 的关系 |
|------|---------|----------------|
| **层级学习率衰减** (LLRD) | 底层用小学习率，顶层用大学习率 | 正交（可以同时使用） |
| **$\mu$P** (Maximal Update Parameterization) | 根据宽度自动缩放超参数 | LoRA+ 是 $\mu$P 在 LoRA 中的应用 |
| **LAMB / LARS** | 层自适应学习率 | 自适应方式不同 |
| **rsLoRA** | 根据 $r$ 调整缩放因子 | 互补（rsLoRA 调缩放，LoRA+ 调学习率） |

---

## 七、局限性

1. **理论-实践 gap**：理论最优比值 $\Theta(d/r)$ 远大于实际最优比值 16
2. **对 Adam 优化器有依赖**：Adam 的自适应性已经部分缓解了不对称性，SGD 下效果可能更显著
3. **超参搜索仍需要**：虽然默认 16 通常好用，但不同任务的最优比值可能不同（4~32）
4. **与其他优化技巧的交互不明确**：如与 gradient clipping、warmup 的最佳组合

---

## 八、总结

### 核心贡献

1. **发现了 LoRA 训练动力学的不对称性**：$A$ 和 $B$ 的角色本质不同
2. **理论推导了最优学习率比值**：基于无限宽度极限分析
3. **提出了极其简单的改进**：只改学习率，零额外开销，一致性提升效果
4. **实践价值高**：改动极小，可以直接应用于任何 LoRA 训练

### LoRA+ 的使用建议

- **默认设置**：$\eta_B / \eta_A = 16$
- **小 $r$（4~8）时**：可以尝试 $\eta_B / \eta_A = 32$
- **大 $r$（64+）时**：可以缩小到 $\eta_B / \eta_A = 8$
- **与其他方法兼容**：可以同时与 QLoRA、rsLoRA 等方法组合使用

### 延伸阅读

- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — 基本原理
- [LoRA 原始论文精读](./055_LoRA_低秩适配微调大模型) — 原始方法
- [rsLoRA 精读](./062_rsLoRA_秩稳定缩放) — 另一种解决秩缩放问题的方法
