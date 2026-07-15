---
title: "ActionEncoder：动作轨迹 + 时间步的联合编码"
series:
  id: groot_n1d7_deep_dive
  chapter: 15
order: 15
---

# ActionEncoder：动作轨迹 + 时间步的联合编码

> 噪声动作轨迹是怎么变成 DiT 能处理的向量的？本章拆解 ActionEncoder 的三层设计，理解动作和时间步如何被联合编码。

## 相关阅读

- [DiT 输出层](./14_DiT输出层_AdaLN_Zero调制)（上一章）
- [CategorySpecificMLP](./16_CategorySpecificMLP_多具身体条件化)（下一章）
- [Flow Matching 数学基础](./09_Flow_Matching数学基础)

---

## 前情提要

上一章我们看完了 DiT 输出的最后一步。现在往回看——DiT 的输入是从哪里来的？
本章聚焦 `ActionEncoder`（在 GR00T N1.7 中具体是 `MultiEmbodimentActionEncoder`
的基础版本），理解带噪声的动作轨迹是如何变成 DiT 能处理的 1536 维向量的。

---

## 1. 待解决的问题

回忆 Flow Matching 训练流程：每个训练样本会构造一个"带噪声的动作轨迹"
$x_t = (1-t)\epsilon + ta$，形状是 `[B, 40, 132]`（40 步，每步 132 维物理动作）。

DiT 内部处理的向量维度是 1536（`input_embedding_dim`）。我们需要一个模块，
把 `[B, 40, 132]` 的物理动作转换成 `[B, 40, 1536]` 的 DiT 输入向量——
同时还要把"当前是第几步去噪"（时间步 $t$）也编码进去。

这正是 `ActionEncoder` 的职责：**同时编码"动作是什么"和"现在处于去噪的哪个阶段"**。

---

## 2. 设计思路

一个自然的想法是：分别编码动作和时间步，然后想办法把两者结合起来。

具体设计分三步：

1. **动作嵌入**：用一个线性层把 132 维的动作映射到 hidden_size 维（这一步不涉及时间步）
2. **时间步嵌入**：用 sinusoidal 编码把标量时间步转换为 hidden_size 维向量（这一步不涉及动作）
3. **融合**：把两者拼接起来，再用两层非线性变换把拼接后的信息"揉"成一个统一的向量

为什么不直接把动作和时间步相加？因为相加要求两者维度相同且语义可比较——
但动作信息和时间步信息的性质完全不同。拼接（concat）保留了两种信息的完整性，
再用可学习的线性层去决定如何组合它们，比简单相加更灵活。

---

## 3. 逐步拆解实现

### 3.1 第一步：SinusoidalPositionalEncoding 编码时间步

时间步是一个标量（比如 $t=250$），需要转换成一个高维向量。这里用的方法和
Transformer 经典的位置编码完全一样的数学形式——不同频率的 sin/cos 组合：

$$
\text{enc}(t)_{2i} = \sin(t \cdot f_i), \quad \text{enc}(t)_{2i+1} = \cos(t \cdot f_i)
$$

其中 $f_i$ 是不同的频率，随 $i$ 指数递减。

> **一句话直觉**：用一组不同"节奏"的正弦波去描述同一个数字，节奏越快的波能区分相近的数字，节奏越慢的波能区分相差很大的数字。

来看代码实现：

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: (B, T) - 注意这里输入是2D的，T是动作序列长度
        timesteps = timesteps.float()
        B, T = timesteps.shape
        half_dim = self.embedding_dim // 2
        
        # 计算不同频率
        exponent = -torch.arange(half_dim, dtype=torch.float) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, embedding_dim)
        return enc
```

注意这里的输入 `timesteps` 形状是 `(B, T)`——每个 batch 中每个时间步位置都有一个值，
而不是单个标量。这是因为 GR00T 需要把**同一个** batch 级时间步 $t$（一个标量），
广播复制到动作序列的每一步上（40 步）。

### 3.2 第二步：动作嵌入 W1

这一步很直接——用一个线性层把物理动作维度映射到隐藏维度：

```python
self.W1 = nn.Linear(action_dim, hidden_size)  # (d -> w)，例如 132 -> 1536

a_emb = self.W1(actions)  # actions: (B, T, action_dim) -> (B, T, hidden_size)
```

### 3.3 第三步：融合（拼接 + 非线性变换）

现在有了 `a_emb`（动作嵌入，形状 `(B, T, w)`）和 `tau_emb`（时间步嵌入，形状 `(B, T, w)`）。
接下来要把它们融合成一个统一的表示。

设计选择是"拼接后过一个非线性 MLP"：

```python
self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # (2w -> w)
self.W3 = nn.Linear(hidden_size, hidden_size)      # (w -> w)

x = torch.cat([a_emb, tau_emb], dim=-1)  # (B, T, 2w) - 拼接
x = swish(self.W2(x))                     # (B, T, w) - 融合 + 激活
x = self.W3(x)                            # (B, T, w) - 再一次变换
```

其中 `swish(x) = x * sigmoid(x)`，是一种平滑的非线性激活函数
（比 ReLU 更平滑，梯度不会突然截断）。

### 3.4 完整前向传播代码

把三步串起来，就是完整的 `ActionEncoder.forward`：

```python
def forward(self, actions, timesteps):
    """
    actions:   (B, T, action_dim)
    timesteps: (B,) - 一个标量，对整个batch中每个样本各一个
    returns:   (B, T, hidden_size)
    """
    B, T, _ = actions.shape

    # 1) 把标量时间步广播到序列的每一步
    timesteps = timesteps.unsqueeze(1).expand(-1, T)  # (B,) -> (B, T)

    # 2) 动作嵌入
    a_emb = self.W1(actions)  # (B, T, w)

    # 3) 时间步编码
    tau_emb = self.pos_encoding(timesteps)  # (B, T, w)

    # 4) 融合
    x = torch.cat([a_emb, tau_emb], dim=-1)  # (B, T, 2w)
    x = swish(self.W2(x))                     # (B, T, w)
    x = self.W3(x)                            # (B, T, w)

    return x
```

---

## 4. 具体数值追踪

假设 `action_dim=4`（简化), `hidden_size=8`, `B=1`, `T=3`（3步动作）：

```
输入:
  actions = [[0.5, -0.2, 0.1, 0.3],   # 第1步
             [0.6, -0.1, 0.2, 0.4],   # 第2步
             [0.7,  0.0, 0.3, 0.5]]   # 第3步
  timesteps = [250]  (batch中只有一个样本)

Step 1: 广播时间步
  timesteps: [250] -> [[250, 250, 250]]  (广播到3步)

Step 2: 动作嵌入 (W1: 4->8)
  a_emb[0] = W1(actions[0])  # 8维向量
  a_emb[1] = W1(actions[1])
  a_emb[2] = W1(actions[2])
  → a_emb 形状 (1, 3, 8)

Step 3: 时间步编码 (sinusoidal, ->8维)
  tau_emb[0,0] = [sin(250*f0), sin(250*f1), sin(250*f2), sin(250*f3),
                   cos(250*f0), cos(250*f1), cos(250*f2), cos(250*f3)]
  (第1、2、3步的 tau_emb 完全相同，因为都是同一个 t=250)
  → tau_emb 形状 (1, 3, 8)

Step 4: 拼接 + 融合
  x = cat([a_emb, tau_emb])  → (1, 3, 16)
  x = swish(W2(x))            → (1, 3, 8)
  x = W3(x)                   → (1, 3, 8)

输出: action_features, 形状 (1, 3, 8)
```

注意关键点：**同一个 batch 样本内，40 步动作共享同一个时间步 $t$**——
因为整个 40 步的动作块是被**一起**去噪的（不是逐步自回归）。
但每一步的**动作值不同**（`actions[0], actions[1], actions[2]` 各不相同），
所以 `a_emb` 在不同步之间是不同的，只有 `tau_emb` 相同。

---

## 5. 为什么要"联合"编码而不是分开处理？

一个可能的问题：为什么不直接把 `a_emb` 和 `tau_emb` 相加，或者分别输入 DiT？

**相加的问题**：动作信息和时间步信息的"尺度"和"语义"不同。直接相加可能导致
一方的信息淹没另一方（类似于交替注意力章节讨论的"数量压制"问题，但这里是"尺度压制"）。

**分别输入的问题**：DiT 的输入接口设计为单一的 `hidden_states` 序列，
如果要分别处理动作和时间步，需要重新设计整个 DiT 的输入接口，增加复杂度。

**拼接+MLP 的优势**：让网络自己学习"如何组合"两种信息，而不是人为规定
组合方式（相加/相乘）。这是深度学习中常见的设计哲学——
用可学习的变换代替手工设计的组合规则。

---

## 6. 和 N1.7 的 MultiEmbodimentActionEncoder 的关系

本章讲的 `ActionEncoder` 是 N1.5 使用的**通用版本**（W1/W2/W3 权重被所有机器人共享）。

N1.7 使用的是升级版 `MultiEmbodimentActionEncoder`——设计思路完全相同
（sinusoidal 编码 + 拼接 + 两层非线性变换），唯一的区别是把 `nn.Linear`
换成了 `CategorySpecificLinear`，让每种机器人有独立的 W1/W2/W3。

下一章我们将详细拆解这个升级——理解 `CategorySpecificLinear` 如何用一个
"批量矩阵乘法（bmm）"技巧，同时维护多组独立权重。

---

## 7. 总结

ActionEncoder 的核心设计可以概括为"分离编码，联合融合"：

1. **动作嵌入**（W1）：把物理动作映射到隐藏空间，不涉及时间信息
2. **时间步编码**（sinusoidal）：把标量时间步转换为向量，不涉及动作信息
3. **融合**（W2 + Swish + W3）：拼接后用非线性 MLP 学习两者的组合方式

这个三步流程把"噪声动作 + 去噪进度"这两种异质信息统一编码成 DiT 能理解的
单一向量表示——为后续 32 层的处理做好准备。

---

## 下一章预告

下一章我们深入 `CategorySpecificMLP` 和 `CategorySpecificLinear`——
GR00T N1.7 实现多具身体支持的核心机制。你将看到"每种机器人独立权重矩阵"
是如何通过批量矩阵乘法高效实现的，以及维度扩展机制如何支持向后兼容。
