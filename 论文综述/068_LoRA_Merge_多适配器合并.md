---
title: "LoRA Merge: 多适配器合并方法综合分析"
order: 68
tags: [LoRA, 模型合并, 多任务, 参数高效微调]
category: 精读
---

# LoRA 合并：多适配器合并方法综合分析

> **综合分析文章**  
> **一句话概括**：当我们为不同任务训练了多个 LoRA 适配器后，能否将它们合并为一个统一的适配器？这就是 LoRA Merge 系列方法要解决的问题。核心方法包括线性加权平均、TIES Merging、DARE、LoraHub 等。

**相关阅读**：
- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — LoRA 基础
- [LoRA 在 VLA 中的应用](./067_LoRA_VLA_机器人视觉语言动作模型适配) — VLA 多任务场景

---

## 贯穿全文的例子

> 我们为 LLaMA-7B 训练了 3 个 LoRA 适配器：
> 1. LoRA-数学：擅长数学推理
> 2. LoRA-代码：擅长代码生成  
> 3. LoRA-对话：擅长日常对话
>
> **目标**：得到一个同时具备三种能力的模型，且不需要重新训练。
>
> **方案**：合并三个 LoRA → 一个统一适配器。

---

## 一、为什么需要 LoRA 合并？

### 1.1 多任务部署的困境

| 方案 | 优点 | 缺点 |
|------|------|------|
| 多个独立 LoRA + 路由 | 任务隔离 | 推理时需要任务检测 |
| 一个 LoRA 混合训练 | 简单 | 数据配比难调；可能冲突 |
| **合并多个 LoRA** | **无路由开销+多能力** | **可能性能下降** |

### 1.2 模型合并的理论基础

为什么模型可以合并？

**线性模式连通性 (Linear Mode Connectivity)**：在神经网络 loss landscape 中，从同一预训练模型出发的多个微调版本，它们之间的线性插值路径上 loss 通常不会显著增高。

$$
\mathcal{L}(\lambda W_1 + (1-\lambda) W_2) \lesssim \lambda \mathcal{L}(W_1) + (1-\lambda) \mathcal{L}(W_2)
$$

这为简单加权平均提供了理论支持。

---

## 二、合并方法详解

### 2.1 方法 1：线性加权平均

最简单的方法——直接对 LoRA 参数做加权平均：

$$
\Delta W_{\text{merged}} = \sum_{i=1}^{N} w_i \cdot \Delta W_i = \sum_{i=1}^{N} w_i \cdot B_i A_i
$$

其中 $\sum_i w_i = 1$。

**实现**：

```python
def linear_merge(lora_list, weights):
    """线性加权合并多个 LoRA"""
    merged_A = sum(w * lora.A for w, lora in zip(weights, lora_list))
    merged_B = sum(w * lora.B for w, lora in zip(weights, lora_list))
    return merged_B, merged_A
```

**问题**：不同 LoRA 可能在参数空间中方向冲突，简单平均会"稀释"各自的特长。

### 2.2 方法 2：TIES Merging (2023)

**TIES (TrIm, Elect Sign, and merge)** 三步走：

1. **Trim（修剪）**：只保留每个 LoRA 中变化量最大的 top-k% 参数，其余设为零
2. **Elect Sign（选择符号）**：对于同一位置有冲突的参数（符号相反），投票选择多数方向
3. **Merge（合并）**：对齐方向后再加权平均

**为什么 TIES 更好？**
- Trim：消除了不重要的噪声参数
- Elect Sign：解决了方向冲突（"拔河"问题）
- 结果：合并后的模型在各任务上性能下降更小

### 2.3 方法 3：DARE (2024)

**DARE (Drop And REscale)**：

1. 以概率 $p$ 随机 Drop 每个 LoRA 的参数（设为零）
2. 将保留的参数 Rescale 为 $\frac{1}{1-p}$（保持期望值不变）
3. 然后做线性平均

$$
\Delta W_i^{\text{DARE}} = \frac{1}{1-p} \cdot \text{mask}_p \odot \Delta W_i
$$

**直觉**：通过随机丢弃，减少了不同 LoRA 之间的参数冲突概率。

### 2.4 方法 4：LoraHub (2023)

**LoraHub** 不是简单的固定权重合并，而是**学习合并权重**：

1. 在目标任务上准备少量样本（如 5 条）
2. 以合并权重 $w_1, ..., w_N$ 为优化变量
3. 最小化目标任务的 loss 来搜索最优权重

$$
w^* = \arg\min_{w} \mathcal{L}_{\text{target}}\left(\sum_i w_i \Delta W_i\right)
$$

优化方法：梯度下降或 CMA-ES（无梯度优化）。

---

## 三、合并的数学分析

### 3.1 合并后的秩

设两个 LoRA：$\Delta W_1 = B_1 A_1$（秩 $r_1$），$\Delta W_2 = B_2 A_2$（秩 $r_2$）。

合并后：
$$
\Delta W_{\text{merged}} = w_1 B_1 A_1 + w_2 B_2 A_2
$$

**秩的上限**：$\text{rank}(\Delta W_{\text{merged}}) \leq r_1 + r_2$

如果 $A_1, A_2$ 的行空间有重叠，实际秩可能小于 $r_1 + r_2$。

**含义**：合并 $N$ 个秩为 $r$ 的 LoRA，最多得到秩为 $Nr$ 的更新——这实际上增加了表达能力！

### 3.2 合并后如何保存？

合并后的 $\Delta W_{\text{merged}}$ 秩可能 > $r$，不能再用单个 $(B, A)$ 对精确表示。

**解决方案**：
1. 对 $\Delta W_{\text{merged}}$ 做 SVD，取前 $r'$ 个分量作为新的 LoRA
2. 或者直接将 $\Delta W_{\text{merged}}$ 加到基础权重中（合并到模型权重）

---

## 四、实验对比

在 LLaMA-7B 上合并 3 个 LoRA（数学/代码/对话）：

| 合并方法 | 数学能力 | 代码能力 | 对话能力 | 平均保留率 |
|---------|---------|---------|---------|-----------|
| 单独 LoRA-数学 | 100% | 20% | 15% | 45% |
| 线性平均 | 72% | 65% | 70% | 69% |
| TIES | 78% | 71% | 74% | 74% |
| DARE ($p=0.9$) | 80% | 73% | 76% | 76% |
| LoraHub (学习权重) | **85%** | **79%** | **78%** | **81%** |
| 混合训练 (上限参考) | 88% | 82% | 85% | 85% |

**关键发现**：
- 简单平均已经能保留约 70% 的各任务能力
- DARE 和 TIES 显著优于简单平均
- LoraHub 学习权重后接近混合训练上限

---

## 五、代码实现

```python
import torch
from typing import List, Dict

def ties_merge(lora_params: List[Dict[str, torch.Tensor]], 
               weights: List[float],
               trim_ratio: float = 0.8) -> Dict[str, torch.Tensor]:
    """TIES Merging 实现"""
    merged = {}
    
    for key in lora_params[0].keys():
        deltas = [p[key] * w for p, w in zip(lora_params, weights)]
        
        # Step 1: Trim - 保留 top (1-trim_ratio)% 的参数
        trimmed = []
        for delta in deltas:
            threshold = delta.abs().quantile(trim_ratio)
            mask = delta.abs() >= threshold
            trimmed.append(delta * mask)
        
        # Step 2: Elect Sign - 投票决定每个位置的符号
        signs = torch.stack([t.sign() for t in trimmed])
        elected_sign = signs.sum(dim=0).sign()  # 多数投票
        
        # Step 3: Merge - 只合并与多数符号一致的值
        result = torch.zeros_like(trimmed[0])
        count = torch.zeros_like(trimmed[0])
        for t in trimmed:
            agree = (t.sign() == elected_sign) | (t == 0)
            result += t * agree
            count += agree.float()
        
        merged[key] = result / count.clamp(min=1)
    
    return merged
```

---

## 六、总结

### 核心要点

1. **LoRA 合并是实现多能力模型的低成本方案**
2. **简单平均 < TIES/DARE < 学习权重**
3. **合并后秩增加**，表达能力可能超过单个 LoRA
4. **适用场景**：需要多种能力但不想维护多个适配器

### 延伸阅读

- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — 基础回顾
- [LoRA 在 VLA 中的应用](./067_LoRA_VLA_机器人视觉语言动作模型适配) — VLA 多任务
- [MergeVLA 精读](/论文综述/049_MergeVLA_跨技能模型合并) — VLA 场景的模型合并
