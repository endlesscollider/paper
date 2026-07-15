---
title: "CategorySpecificMLP：多具身体条件化线性层"
series:
  id: groot_n1d7_deep_dive
  chapter: 16
order: 16
---

# CategorySpecificMLP：多具身体条件化线性层

> 一个模型如何同时给 32 种不同的机器人独立"配备"权重？本章拆解 CategorySpecificLinear 的批量矩阵乘法技巧和维度扩展机制。

## 相关阅读

- [ActionEncoder](./15_ActionEncoder_动作轨迹时间步联合编码)（上一章）
- [StateEncoder 与 ActionDecoder](./17_StateEncoder与ActionDecoder)（下一章）
- [从 N1.5 到 N1.7 架构升级](./03_从N1d5到N1d7_架构升级)

---

## 前情提要

上一章我们理解了 `ActionEncoder` 如何联合编码动作和时间步。本章聚焦
GR00T N1.7 实现"一个模型驾驭多种机器人"的核心机制——`CategorySpecificLinear`。

---

## 1. 问题：普通线性层为什么不够用？

一个普通的 `nn.Linear(in_dim, out_dim)` 只存储**一组**权重矩阵 $W \in \mathbb{R}^{d_{in} \times d_{out}}$。
无论输入来自哪个机器人，都用同一个 $W$ 做变换：

$$
y = xW + b
$$

问题在于：不同机器人的"输入语义"完全不同。比如 132 维的统一动作向量中：
- Franka 机械臂：前 8 维是"7个关节角 + 1个夹爪"，后 124 维全是填充的 0
- Unitree G1：前 50 维是"双臂+双手+腰+导航"的复杂组合，后 82 维填充 0

如果用同一个 $W$ 处理这两种完全不同的语义，模型必须学会"根据数值模式猜出这是哪种机器人"——
这是一种隐式的、脆弱的区分方式，容易在训练数据不均衡时出现负迁移
（一个机器人的梯度更新"污染"了另一个机器人的表现）。

## 2. 解决思路：显式的多组权重

既然问题是"一组权重要处理多种语义"，那么最直接的解法就是——
**给每种机器人分配一组独立的权重矩阵**，通过 `embodiment_id` 来选择使用哪一组。

数学表示：不再是单个 $W$，而是一个权重张量 $W \in \mathbb{R}^{N \times d_{in} \times d_{out}}$
（$N$ 是支持的机器人种类数）。对于 embodiment_id 为 $c$ 的样本：

$$
y = x W_c + b_c
$$

> **一句话直觉**：每种机器人都有自己专属的"翻译规则"，模型根据输入是哪种机器人，自动切换到对应的翻译规则。

这样一来，Franka 的权重矩阵只需要学习"怎么翻译 Franka 的动作语义"，
完全不会受到 G1 训练数据的干扰。代价是参数量随机器人种类数线性增长，
但由于每组权重本身不大（是一个小 MLP），总代价是可控的。

---

## 3. 实现关键：怎么高效地"按样本选权重"？

一个 batch 里通常混合了不同机器人的样本（比如 batch_size=8，其中 3 个是 Franka、5 个是 G1）。
如果用 for 循环逐个样本选权重再做矩阵乘法，会很慢（无法利用 GPU 的并行计算能力）。

高效的做法是用 **批量矩阵乘法（batch matrix multiplication, `bmm`）**——
先把每个样本对应的那组权重"取出来"拼成一个新的批量张量，再一次性做 bmm。

### 3.1 存储：一个大的权重张量

```python
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        # 存储 num_categories 组独立的权重矩阵，一次性初始化
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
```

`self.W` 的形状是 `(N, input_dim, hidden_dim)`——想象成 N 个独立的 `input_dim × hidden_dim` 矩阵叠在一起。

### 3.2 前向传播：索引 + bmm

给定 batch 中每个样本的 `cat_ids`（形状 `(B,)`，取值范围 `0` 到 `N-1`），
先用 PyTorch 的高级索引一次性取出这个 batch 需要的所有权重：

```python
def forward(self, x, cat_ids):
    """
    x: (B, T, input_dim) - 输入
    cat_ids: (B,) - 每个样本对应的 embodiment ID
    """
    selected_W = self.W[cat_ids]  # (B, input_dim, hidden_dim)
    selected_b = self.b[cat_ids]  # (B, hidden_dim)
    return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
```

### 3.3 逐行解释这个关键操作

`self.W[cat_ids]` 是 PyTorch 的高级索引（fancy indexing）。假设：
- `self.W` 形状是 `(32, 132, 1536)`（32种机器人，每种一个 132→1536 的矩阵）
- `cat_ids = [26, 26, 25, 26]`（batch中4个样本，分别对应不同机器人）

那么 `self.W[cat_ids]` 会返回形状 `(4, 132, 1536)` 的张量——
第0、1、3个样本取出的是第26组权重（同一组，因为它们都是同一种机器人），
第2个样本取出的是第25组权重。**这一步是并行的，不需要 for 循环**。

`torch.bmm(x, selected_W)`：`x` 形状是 `(4, T, 132)`，`selected_W` 形状是 `(4, 132, 1536)`。
`bmm`（batch matrix multiply）对 batch 中的每一个样本**独立**做矩阵乘法：
`x[i] @ selected_W[i]`，最终输出 `(4, T, 1536)`。

> **一句话理解 bmm**：普通的矩阵乘法 `A @ B` 只能处理2维矩阵；`bmm` 让你一次性对一个批次的"矩阵对"分别做乘法，效率等价于把每对矩阵分开算，但代码写法上是"一行"、计算上是并行的。

### 3.4 具体数值例子

简化维度：`input_dim=2`, `hidden_dim=3`, `num_categories=2`（只有2种机器人）

```
self.W:
  category 0: [[1, 0, 1],
               [0, 1, 0]]   # (2, 3)
  category 1: [[2, 1, 0],
               [1, 0, 2]]   # (2, 3)

batch: cat_ids = [0, 1]  (第一个样本是机器人0，第二个是机器人1)
x = [[[1, 1]],    # 样本0的输入 (T=1, input_dim=2)
     [[2, 0]]]    # 样本1的输入

selected_W = self.W[[0, 1]]  # 取出对应两组权重
  = [[[1, 0, 1], [0, 1, 0]],    # 样本0用category 0的权重
     [[2, 1, 0], [1, 0, 2]]]    # 样本1用category 1的权重

bmm(x, selected_W):
  样本0: [1,1] @ [[1,0,1],[0,1,0]] = [1*1+1*0, 1*0+1*1, 1*1+1*0] = [1, 1, 1]
  样本1: [2,0] @ [[2,1,0],[1,0,2]] = [2*2+0*1, 2*1+0*0, 2*0+0*2] = [4, 2, 0]

输出: [[[1, 1, 1]], [[4, 2, 0]]]
```

样本0和样本1用了完全不同的权重矩阵，通过一次 `bmm` 调用并行完成，没有 for 循环。

---

## 4. CategorySpecificMLP：两层堆叠

有了 `CategorySpecificLinear` 这个基础组件，GR00T 用两层堆叠构成一个完整的 MLP，
和普通 MLP（Linear→ReLU→Linear）的结构完全一样，只是每层都换成了"多具身体版"：

```python
class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))  # 第一层 + ReLU
        return self.layer2(hidden, cat_ids)        # 第二层
```

这个 `CategorySpecificMLP` 在 GR00T 中被用作 **StateEncoder** 和 **ActionDecoder**
（下一章会详细讲）。两处用法的核心思想相同：给每种机器人独立的编解码权重。

---

## 5. 维度扩展机制：向后兼容新机器人

一个实际问题：假如你训练好了一个支持"最大动作维度=64"的模型，后来想接入一个
128 维动作的新机器人怎么办？直接改配置重新训练所有权重代价太大。

`CategorySpecificLinear` 提供了 `expand_action_dimension` 方法，
通过**复制已有权重**的方式扩展维度，而不是随机初始化：

```python
def expand_action_dimension(self, old_action_dim, new_action_dim, 
                             expand_input=False, expand_output=False):
    if new_action_dim <= old_action_dim:
        raise ValueError(f"New dim {new_action_dim} must be larger than old dim {old_action_dim}")

    # 扩展输入维度 (适用于像 W1 这样"吃"动作向量的层)
    if expand_input and self.W.shape[1] == old_action_dim:
        repeat_times = new_action_dim // old_action_dim
        remainder = new_action_dim % old_action_dim
        
        new_W_parts = [self.W] * repeat_times  # 重复整块
        if remainder > 0:
            new_W_parts.append(self.W[:, :remainder, :])  # 补上余数部分
        
        new_W = torch.cat(new_W_parts, dim=1)
        self.W = nn.Parameter(new_W)
```

### 5.1 具体例子

假设 `old_action_dim=64`，想扩展到 `new_action_dim=132`：
- `repeat_times = 132 // 64 = 2`
- `remainder = 132 % 64 = 4`
- 新权重 = [原权重(64维), 原权重(64维), 原权重的前4维] 拼接起来 → 132维

这种"重复+截断补齐"的策略，让新维度的权重不是从零开始，而是继承了
已训练好的旧权重的"模式"——相当于给新维度一个合理的初始猜测，
比随机初始化更容易收敛。

---

## 6. 参数量分析

回顾一下之前提到的参数量对比。用具体数字看 `CategorySpecificLinear` 相比
普通 `nn.Linear` 的代价：

| 配置 | 普通 Linear 参数量 | CategorySpecificLinear 参数量 (32种机器人) | 倍数 |
|------|-------------------|------------------------------------------|------|
| 132→1536 (W1) | 202,752 | 6,488,064 | 32x |
| 3072→1536 (W2) | 4,718,592 | 150,994,944 | 32x |
| 1536→1536 (W3) | 2,359,296 | 75,497,472 | 32x |

看起来倍数很大，但要注意：这些模块**只是整个模型的一小部分**
（DiT 的 32 层才是参数大头，约 790M）。CategorySpecificMLP 系列模块
（StateEncoder + ActionEncoder + ActionDecoder）的总参数量约在 200M 量级，
相比整个 3B 模型的规模，多具身体化带来的参数增长是可接受的代价。

---

## 7. 为什么不用更"参数高效"的方案？

一个自然的问题：既然担心参数量，为什么不用类似 LoRA 的低秩分解，
给每种机器人一个小的"增量"权重，而不是完全独立的一大组权重？

可能的原因：
1. **简单性**：完全独立的权重逻辑最简单，bmm 实现高效，不需要额外的秩选择超参数
2. **彻底隔离**：LoRA 式的"共享基座+低秩增量"仍然会有梯度耦合（共享基座部分），
   而完全独立权重做到了 100% 隔离
3. **模块本身不大**：StateEncoder/ActionEncoder/ActionDecoder 本身是小 MLP，
   即使乘以 32 倍也不会成为整个模型的参数瓶颈

---

## 8. 总结

`CategorySpecificLinear` 是 GR00T N1.7 实现多具身体支持的基础组件，核心设计：

1. **存储**：一个 `(N, in, out)` 的权重张量，N 组独立权重叠放在一起
2. **选择**：用 `cat_ids` 做高级索引，一次性取出 batch 中每个样本对应的权重
3. **计算**：用 `bmm` 并行完成 batch 内所有样本的矩阵乘法，无需 for 循环
4. **扩展**：通过重复+截断的方式扩展维度，继承已训练权重而非随机初始化

这个设计让 GR00T 能用一套训练流程、一次前向传播，同时服务 32 种不同的机器人——
每种机器人的编解码逻辑完全独立，互不干扰。

---

## 下一章预告

下一章我们把 `CategorySpecificMLP` 放回到具体场景中——理解 `StateEncoder`
如何处理"状态历史"的展平和拼接，以及 `ActionDecoder` 如何把 DiT 输出切片解码
回物理动作空间。
