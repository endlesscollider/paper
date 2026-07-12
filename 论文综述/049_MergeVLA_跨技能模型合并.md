---
title: MergeVLA：跨技能模型合并的通用 VLA 代理
order: 249
tags: [VLA, 模型合并, LoRA, 稀疏适配器, 动作专家, CVPR2026]
category: 精读
---

# MergeVLA：用稀疏 LoRA Mask + 可组合动作专家合并多个技能 VLA

> **论文**: *MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent*<br>
> **会议**: CVPR 2026<br>
> **版本**: arXiv:2511.18810, 2025<br>
> **一句话**: MergeVLA 提出稀疏激活的 LoRA 适配器（通过任务 mask 保留一致参数、减少冲突）和仅用 cross-attention 的动作专家模块，实现多个技能特化 VLA 的无干扰合并，无需重训练即可获得通用策略。

---

## 相关阅读

| 类型 | 链接 |
|------|------|
| 前置知识 | [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) |
| 前置知识 | [KL散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) |
| 综述 | [持续/终身 VLA 强化学习综述](./S07_持续终身VLA强化学习综述) |
| 精读 | [Stellar VLA：技能知识空间持续进化](./048_StellarVLA_技能知识空间持续进化) |

---

## 贯穿全文的例子

> **设定**：我们有 3 个独立训练的技能特化 VLA：
> - VLA-A：专精"桌面抓取"（LoRA-A 在 LIBERO 抓取任务上微调）
> - VLA-B：专精"抽屉操作"（LoRA-B 在抽屉相关任务上微调）
> - VLA-C：专精"堆叠任务"（LoRA-C 在方块堆叠任务上微调）
>
> **目标**：将三者合并为一个 VLA-ABC，能同时处理所有技能，且不需要重新训练，性能不下降。
>
> **挑战**：直接平均 LoRA 权重会导致严重干扰——"抓取"的参数和"堆叠"的参数可能在同一位置做了相反方向的修改。

---

## 一、问题：为什么不能简单合并

### 1.1 模型合并的基本思路

给定基座模型 $W_0$ 和 $N$ 个任务特化的 LoRA 增量 $\{\Delta W_1, \Delta W_2, \ldots, \Delta W_N\}$，最简单的合并方式是加权平均：

$$
W_{\text{merged}} = W_0 + \frac{1}{N}\sum_{i=1}^N \Delta W_i
$$

### 1.2 为什么简单平均会失败

**参数干扰（Parameter Interference）**：不同任务可能对同一个参数做了"方向相反"的修改。

**数值例子**：假设基座某个权重 $w_0 = 1.0$：
- LoRA-A 将其改为 $w_0 + \Delta w_A = 1.0 + 0.5 = 1.5$（抓取需要加大）
- LoRA-B 将其改为 $w_0 + \Delta w_B = 1.0 - 0.3 = 0.7$（抽屉操作需要减小）
- 简单平均：$1.0 + (0.5 + (-0.3))/2 = 1.1$ — 既不是 1.5 也不是 0.7，两个任务都不满意

**冲突率统计**：实验发现，两个随机技能 LoRA 之间约 35-45% 的非零参数位置存在符号冲突（一个正一个负）。

### 1.3 与持续学习的关系

模型合并和持续学习解决的是相关但不同的问题：

| 维度 | 持续学习 | 模型合并 |
|------|---------|---------|
| 训练方式 | 顺序训练一个模型 | 独立训练多个模型后合并 |
| 遗忘来源 | 新任务梯度覆盖旧知识 | 合并时参数冲突 |
| 优势 | 可以利用跨任务迁移 | 可完全并行训练 |
| 代表方法 | [Simple Recipe](./045_SimpleRecipe_VLA天然持续学习者), [Stellar VLA](./048_StellarVLA_技能知识空间持续进化) | MergeVLA (本文) |

---

## 二、MergeVLA 的两大创新

### 2.1 创新一：稀疏 LoRA Task Mask

核心思想：不是所有 LoRA 参数对每个任务都重要。通过为每个任务学习一个**二值 mask**，只保留对该任务重要的参数位置，合并时不同任务就只在各自重要的位置上有值，大幅减少冲突。

**形式化**：对任务 $i$，LoRA 增量变为：

$$
\Delta W_i^{\text{sparse}} = M_i \odot \Delta W_i
$$

其中 $M_i \in \{0, 1\}^{d \times d}$ 是二值 mask，$\odot$ 是逐元素乘法。

**Mask 的学习**：通过 magnitude pruning + task-specific importance scoring：

$$
M_i[j,k] = \begin{cases} 1 & \text{if } |\Delta W_i[j,k]| \geq \text{top-}p\text{ percentile} \\ 0 & \text{otherwise} \end{cases}
$$

保留每个 LoRA 中 magnitude 最大的 $p\%$ 参数。

**数值例子**：假设 $p = 30\%$（只保留最重要的 30%）。

LoRA-A 的原始增量（4个参数为例）：$\Delta W_A = [0.5, 0.02, -0.3, 0.01]$
- Mask-A = $[1, 0, 1, 0]$（保留 |0.5| 和 |-0.3|，去掉小值）
- Sparse LoRA-A = $[0.5, 0, -0.3, 0]$

LoRA-B 的原始增量：$\Delta W_B = [0.01, 0.4, 0.03, -0.6]$
- Mask-B = $[0, 1, 0, 1]$
- Sparse LoRA-B = $[0, 0.4, 0, -0.6]$

**合并后**：$\Delta W_{\text{merged}} = [0.5, 0.4, -0.3, -0.6]$ — **零冲突**！

因为两个 mask 的非零位置不重叠，合并时各自的参数"分管"不同位置。

### 2.2 Mask 重叠度分析

当任务数 $N$ 增加或稀疏度 $p$ 增大时，mask 重叠不可避免。重叠概率为：

$$
P(\text{overlap at position } (j,k)) = 1 - (1-p)^N
$$

**数值例子**：
- $p = 30\%$, $N = 3$ 个任务：$P = 1 - 0.7^3 = 0.657$
- $p = 10\%$, $N = 3$ 个任务：$P = 1 - 0.9^3 = 0.271$
- $p = 10\%$, $N = 10$ 个任务：$P = 1 - 0.9^{10} = 0.651$

因此，本文选择 $p = 10\%$ 作为默认稀疏度——每个任务只保留 10% 的 LoRA 参数。

### 2.3 重叠位置的冲突解决

对于不可避免的重叠，使用**一致性感知合并**：

$$
\Delta W_{\text{merged}}[j,k] = \begin{cases}
\text{mean}(\{\Delta W_i[j,k] : M_i[j,k]=1\}) & \text{if signs are consistent} \\
\Delta W_{i^*}[j,k] & \text{if signs conflict, } i^* = \arg\max_i |\Delta W_i[j,k]|
\end{cases}
$$

即：
- 符号一致时取平均（都是正或都是负 → 平均不会"抵消"）
- 符号冲突时保留 magnitude 最大的那个

---

## 三、创新二：Cross-Attention 动作专家

### 3.1 动机

VLA 的 backbone（如 LLaMA-based）主要处理视觉和语言理解。动作生成（action decoding）是一个不同性质的任务：

- 视觉理解在不同技能间高度共享
- 语言理解同样高度共享
- **动作映射是技能特化的**：抓取的运动模式和堆叠的运动模式差异很大

因此，MergeVLA 将动作生成部分独立出来作为**可组合的动作专家**。

### 3.2 动作专家架构

```mermaid
flowchart TD
    subgraph VLA Backbone (共享)
        A["视觉编码器"] --> B["多模态 Transformer"]
        C["语言编码器"] --> B
        B --> D["隐藏状态 h ∈ R^d"]
    end
    
    subgraph Action Expert (技能特化)
        D --> E["Cross-Attention\nQ=action queries\nK,V=h"]
        E --> F["前馈网络"]
        F --> G["动作输出 a ∈ R^{dim_a}"]
    end
```

**关键设计**：动作专家**只使用 cross-attention**，不使用 self-attention。

为什么？

1. **Cross-attention 是"只读"的**：它从 backbone 隐藏状态中提取信息，但不修改 backbone
2. **参数隔离**：不同技能的动作专家之间完全没有参数共享，不会互相干扰
3. **可插拔**：推理时只需根据任务选择对应动作专家，backbone 不变

### 3.3 Cross-Attention 的具体计算

动作专家维护一组可学习的 action queries $Q_{\text{action}} \in \mathbb{R}^{L_a \times d_q}$，其中 $L_a$ 是 action chunk 长度。

$$
\text{Attention}(Q_{\text{action}}, K_h, V_h) = \text{softmax}\left(\frac{Q_{\text{action}} K_h^T}{\sqrt{d_q}}\right) V_h
$$

其中 $K_h = W_K h$，$V_h = W_V h$ 来自 backbone 的隐藏状态。

**数值例子**：假设 action chunk 长度 $L_a = 8$（预测未来 8 步动作），$d_q = 64$：
- Action queries：$8 \times 64$ 的可学习矩阵
- Backbone 输出：$(T_{\text{context}} \times d)$ 的序列（如 $50 \times 2048$）
- Attention 输出：$8 \times 2048$
- 经 FFN 映射到动作维度：$8 \times 7$（7-DoF 机械臂的 8 步动作）

每个技能的动作专家有独立的 $Q_{\text{action}}$、$W_K$、$W_V$ 和 FFN——完全不共享。

### 3.4 动作专家的参数量

| 组件 | 参数量计算 | 具体值 (d=2048, d_q=64, L_a=8, dim_a=7) |
|------|-----------|------|
| Action queries | $L_a \times d_q$ | 512 |
| $W_Q$ | $d_q \times d$ | 131K |
| $W_K$ | $d \times d$ | 4.2M |
| $W_V$ | $d \times d$ | 4.2M |
| FFN | $2 \times d \times d_{\text{ff}}$ | ~8M |
| **总计/专家** | | **~17M** |

对 7B 的 backbone，每个动作专家只占 0.24%。即使 10 个技能也只增加 2.4%。

---

## 四、合并流程

### 4.1 完整的 MergeVLA 合并管线

```mermaid
flowchart TD
    subgraph 训练阶段(并行)
        A1["任务A: 训LoRA-A + Expert-A"] 
        A2["任务B: 训LoRA-B + Expert-B"]
        A3["任务C: 训LoRA-C + Expert-C"]
    end
    
    subgraph 合并阶段(一次性)
        B1["对每个LoRA计算 mask\n(保留top-10%)"]
        B2["稀疏LoRA合并\n(一致性感知)"]
        B3["动作专家直接拼接\n(无需合并)"]
    end
    
    subgraph 推理阶段
        C1["输入(o,l)"] --> C2["共享backbone\n+ 合并后的LoRA"]
        C2 --> C3["路由器选择动作专家"]
        C3 --> C4["对应Expert输出动作"]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> C2
    B3 --> C3
```

### 4.2 推理时的路由

推理时需要决定使用哪个动作专家。路由基于语言指令的语义匹配：

$$
\text{expert\_id} = \arg\max_i \cos(f_{\text{LM}}(l), e_i)
$$

其中 $f_{\text{LM}}(l)$ 是语言模型对指令 $l$ 的嵌入，$e_i$ 是每个技能的预计算原型嵌入。

**数值例子**：
- 指令"pick up the red cup"的嵌入与"抓取"原型的余弦相似度 = 0.92
- 与"抽屉"原型的余弦相似度 = 0.23
- 与"堆叠"原型的余弦相似度 = 0.31
- → 选择 Expert-A（抓取专家）

---

## 五、实验

### 5.1 实验设置

| 项目 | 详情 |
|------|------|
| 基座模型 | OpenVLA (7B) |
| LoRA rank | 16 |
| 稀疏度 $p$ | 10% |
| 技能数 | 3-10 |
| 环境 | LIBERO, RoboCasa, RLBench |
| Baseline | Task Arithmetic, TIES, DARE, Fisher Merging |

### 5.2 合并质量对比

| 方法 | 平均成功率 ↑ | 性能保持率 ↑ | 最差技能 |
|------|------------|------------|---------|
| Individual (上界) | 91.2% | 100% | 88.5% |
| Task Arithmetic | 62.4% | 68.4% | 41.2% |
| TIES Merging | 71.8% | 78.7% | 55.3% |
| DARE | 74.2% | 81.4% | 58.7% |
| Fisher Merging | 76.5% | 83.9% | 62.1% |
| **MergeVLA** | **87.3%** | **95.7%** | **83.8%** |

**性能保持率** = 合并后成功率 / 单独训练的成功率。MergeVLA 达到了 95.7%——几乎无损合并。

### 5.3 稀疏度消融

| 稀疏度 $p$ | 性能保持率 | 合并冲突率 |
|-----------|-----------|-----------|
| 100% (无mask) | 68.4% | 43.2% |
| 50% | 82.1% | 22.5% |
| 30% | 89.5% | 12.8% |
| **10%** | **95.7%** | **3.1%** |
| 5% | 92.3% | 0.8% |
| 1% | 78.5% | 0.01% |

**关键发现**：$p=10\%$ 是最优平衡点。太稀疏（1%）会损失单任务性能，太密集（50%+）会引入大量冲突。

### 5.4 动作专家的必要性

| 配置 | 平均成功率 |
|------|-----------|
| 合并 LoRA + 共享 action head | 79.2% |
| 合并 LoRA + per-task action head (FFN) | 83.5% |
| 合并 LoRA + per-task action expert (cross-attn) | **87.3%** |

Cross-attention 专家比简单 FFN 高 3.8%，因为它能更有选择性地从 backbone 中提取与当前技能相关的信息。

---

## 六、与 Stellar VLA 的比较

| 维度 | [Stellar VLA](./048_StellarVLA_技能知识空间持续进化) | MergeVLA (本文) |
|------|------|------|
| 训练方式 | 顺序训练，动态添加专家 | 并行训练，一次性合并 |
| 适用场景 | 任务逐步到达 | 所有任务都已知 |
| 路由机制 | 语义嵌入 + softmax | 语言指令匹配 |
| 参数冲突处理 | 冻结旧专家 | Sparse mask 隔离 |
| 正迁移 | 通过共享技能专家 | 通过合并后的 LoRA |
| 扩展性 | 专家随技能数增长 | LoRA mask + 动作专家增长 |

**何时选择哪个？**

- 任务持续到来、不确定未来有什么 → **Stellar VLA**
- 所有技能已知、需要一个通用模型部署 → **MergeVLA**
- 两者可以组合：先用 Stellar VLA 持续学习，定期用 MergeVLA 方法"整理"多个专家

---

## 七、理论分析：为什么稀疏 mask 有效

### 7.1 参数重要性的幂律分布

经验发现：LoRA 参数的 magnitude 遵循幂律分布：

$$
P(|\Delta w| > x) \propto x^{-\alpha}, \quad \alpha \approx 2.5
$$

这意味着：绝大多数参数变化很小（对任务贡献微弱），只有少数参数有大变化（决定性作用）。

**数值例子**：对 rank=16 的 LoRA（约 70M 参数）：
- top 10% 的参数贡献了约 85% 的 Frobenius 范数
- top 30% 贡献了约 97%
- 剩余 70% 只贡献 3%

因此，丢弃 90% 的参数只损失约 15% 的"信号强度"——这就是为什么 $p=10\%$ 仍能保持高性能。

### 7.2 稀疏 mask 减少冲突的数学

两个任务在位置 $(j,k)$ 冲突的条件：
1. 两个 mask 都为 1：$M_A[j,k] = 1 \wedge M_B[j,k] = 1$
2. 两个参数符号相反：$\text{sign}(\Delta W_A[j,k]) \neq \text{sign}(\Delta W_B[j,k])$

冲突概率：

$$
P(\text{conflict}) = p^2 \times P(\text{sign mismatch}) \approx p^2 \times 0.5
$$

当 $p = 10\%$：$P(\text{conflict}) = 0.01 \times 0.5 = 0.005 = 0.5\%$ — 极低。

当 $p = 100\%$（无 mask）：$P(\text{conflict}) = 1.0 \times 0.5 = 50\%$ — 非常高。

### 7.3 合并误差的上界

设合并后某一层的权重误差为：

$$
\epsilon_{\text{merge}} = \|W_{\text{merged}} - W_{\text{ideal}}\|_F
$$

其中 $W_{\text{ideal}}$ 是理想的多任务权重。可以证明：

$$
\epsilon_{\text{merge}} \leq \underbrace{\sqrt{N} \cdot (1-p) \cdot \sigma_{\text{tail}}}_{\text{稀疏化截断误差}} + \underbrace{N \cdot p^2 \cdot \sigma_{\text{conflict}}}_{\text{重叠冲突误差}}
$$

两项之间存在 trade-off：$p$ 越小，冲突少但截断大；$p$ 越大，截断小但冲突大。最优 $p$ 大约在 $p^* \approx 1/\sqrt{N}$。

**数值**：$N = 10$ 个任务时，$p^* \approx 0.316$。但论文发现实际最优 $p = 0.1$，比理论预测更激进——原因是幂律分布使得截断误差比理论均匀假设更小。

---

## 八、局限性

### 8.1 需要预知任务分组

合并前需要知道"这些 LoRA 对应哪些技能"——推理时路由需要。如果任务描述模糊或多义，路由可能出错。

### 8.2 对基座模型的依赖

所有技能 VLA 必须从**同一个基座模型**出发训练。不同基座训练的 VLA 无法用本方法合并。

### 8.3 动态合并

当前是一次性合并。如果后续有新技能训练完成，需要重新运行合并流程（虽然很快，不需要重训练）。

---

## 九、总结

| 贡献 | 意义 |
|------|------|
| Sparse LoRA Task Mask | 大幅减少合并冲突（从~45% 降到~3%）|
| Cross-Attention 动作专家 | 技能特化的动作生成，完全无干扰 |
| 无需重训练的合并 | 训练完成后 O(1) 时间合并 |
| 95.7% 性能保持率 | 几乎无损的多技能通用化 |

**核心洞察**：与其训练一个通用 VLA（困难且可能次优），不如分别训练专精 VLA 然后智能合并。关键是：(1) 让每个 LoRA 只修改少量位置以避免冲突，(2) 将动作生成彻底隔离为可插拔模块。

---

## 延伸阅读

- [Stellar VLA：技能知识空间持续进化](./048_StellarVLA_技能知识空间持续进化)：顺序学习场景下的持续进化方案
- [Simple Recipe Works：VLA 天然持续学习者](./045_SimpleRecipe_VLA天然持续学习者)：当不需要合并时的最简方案
- [Forget Me Not：预训练 VLA 抗遗忘](./046_ForgetMeNot_预训练VLA抗遗忘)：replay 方案的极限
- [持续/终身 VLA RL 综述](./S07_持续终身VLA强化学习综述)：所有方法的系统对比
