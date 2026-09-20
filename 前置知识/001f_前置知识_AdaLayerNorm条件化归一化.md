---
title: "AdaLayerNorm：条件化归一化——让网络行为随条件动态调整"
order: 32
tags: [Transformer, DiT, 归一化, 条件生成, 扩散模型]
category: 前置知识
---

# AdaLayerNorm：条件化归一化

> 标准 LayerNorm 对所有输入做相同的归一化。AdaLayerNorm 让归一化的行为**随条件动态变化**——告诉网络"你现在在去噪的第几步"，让网络的工作模式自适应调整。

## 相关阅读

- [DiT：Diffusion Transformer 架构](/前置知识/002x_前置知识_DiT_Diffusion_Transformer架构)
- [Cross-Attention 与交替注意力](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制)
- [Flow Matching 数学基础](/系列/groot_n1d7_deep_dive/09_Flow_Matching数学基础)
- [GR00T N1.7 - DiT 架构](/系列/groot_n1d7_deep_dive/11_DiT架构逐层拆解)

---

## 1. 先回顾：标准 LayerNorm 做了什么？

### 1.1 LayerNorm 的公式

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta
$$

> **一句话直觉**：把输入的分布"拉"到均值为 0、方差为 1 的标准分布，然后用可学习的 $\gamma$ 和 $\beta$ 做缩放和偏移。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\frac{x-\mu}{\sigma+\epsilon}$ | **标准化器** | 把输入的每个维度拉到均值 0、方差 1 的标准分布，消除数值尺度差异 |
| $\mu,\sigma$ | **当前输入的统计量** | 输入向量自身各维度的均值和标准差，随输入实时计算，不是可学习参数 |
| $\gamma,\beta$ | **固定的缩放/偏移参数** | 训练后固定下来的可学习参数，把标准化后的分布再缩放和偏移到网络喜欢的范围 |
| $\epsilon$ | **安全垫** | 防止 $\sigma$ 太小时除零，通常取 $10^{-5}$ |

**用人话读**："先把输入拉成标准分布消除尺度差异，再用两组固定学到的参数把它缩放偏移到合适的范围。"

**为什么是这个形式**：深层网络每层输入的分布会随训练剧烈波动（内部协变量偏移），先标准化再用可学习的 $\gamma,\beta$ 恢复表达能力，是稳定训练最直接的办法。

**具体数值例子**（$d=4$）：

输入：$x = [2, 4, 6, 8]$
- $\mu = 5$，$\sigma = \sqrt{5} \approx 2.24$
- 归一化：$\frac{x - 5}{2.24} = [-1.34, -0.45, 0.45, 1.34]$
- 假设 $\gamma = [1,1,1,1]$，$\beta = [0,0,0,0]$
- 输出：$[-1.34, -0.45, 0.45, 1.34]$
:::

### 1.2 LayerNorm 的局限

标准 LayerNorm 的 $\gamma$ 和 $\beta$ 是**固定的**可学习参数——
一旦训练完成，不管输入是什么、条件是什么，都用同一组 $\gamma, \beta$。

但在扩散模型中，我们希望网络在不同的去噪时间步有**不同的行为**：
- $t=0$（纯噪声）：网络需要做"大幅度修正"
- $t=0.75$（接近干净）：网络只需要做"微小微调"

用同一组 $\gamma, \beta$ 无法表达这种"时间步相关"的动态行为。

---

## 2. AdaLayerNorm：让归一化随条件变化

### 2.1 核心思想

**Adaptive Layer Normalization (AdaLN)**：用条件信息（如时间步 embedding）
动态生成 $\gamma$ 和 $\beta$，替代固定的可学习参数。

$$
\text{AdaLN}(x, c) = (1 + \gamma_c) \cdot \text{LN}(x) + \beta_c
$$

**这个公式在做什么**：把标准 LayerNorm 里固定的 $\gamma,\beta$ 换成"由条件 $c$ 实时算出来"的 $\gamma_c,\beta_c$，让归一化的强度随条件（如时间步）动态变化。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\text{LN}(x)$ | **标准化后的输入** | 已经拉成均值 0、方差 1 分布的特征，不带自己的 $\gamma,\beta$ |
| $\gamma_c$ | **条件决定的缩放力度** | 由条件 $c$（如时间步）动态算出的 scale，控制这一层要"用多大力气工作" |
| $\beta_c$ | **条件决定的偏移量** | 由条件 $c$ 动态算出的 shift |
| $(1+\gamma_c)$ | **围绕 1 的相对缩放** | 加 1 是为了让默认状态（$\gamma_c\approx0$）对应"不缩放" |

**用人话读**："先把输入标准化，再用条件（比如现在是去噪第几步）实时算出的缩放和偏移参数去调制它，而不是用训练完就固定不变的参数。"

**为什么是这个形式**：扩散模型需要网络在不同去噪时间步表现出不同的行为（早期大幅修正、后期微调），固定的 $\gamma,\beta$ 做不到这一点；让 $\gamma_c,\beta_c$ 随条件 $c$ 变化，相当于给每一层装了一个可以按需调节的"力度旋钮"。
:::

其中 $\gamma_c$ 和 $\beta_c$ 是从条件 $c$ 计算得到的：

$$
[\gamma_c, \beta_c] = \text{Linear}(\text{SiLU}(c))
$$

**这个公式在做什么**：具体说明 $\gamma_c,\beta_c$ 这两组"调节旋钮"是怎么从条件 $c$ 算出来的——过一个非线性激活再过一个线性层，输出直接切成两半。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $c$ | **原始条件** | 如时间步的 embedding 向量 |
| $\text{SiLU}(c)$ | **非线性变换** | 对条件做一次平滑的非线性激活，增强表达能力 |
| $\text{Linear}(\cdot)$ | **投影并拆分** | 输出维度是输入的 2 倍，前一半当 $\gamma_c$，后一半当 $\beta_c$ |
| $[\gamma_c,\beta_c]$ | **两组调节参数** | 切分后分别用作缩放和偏移 |

**用人话读**："条件先过一个非线性激活，再过一个线性层，输出的前后两半分别当作缩放参数和偏移参数。"

**为什么用一个线性层同时输出两者**：把 $\gamma_c,\beta_c$ 的计算共享同一套底层特征，参数量更省，且训练中两者天然联合优化，没有必要拆成两个独立网络。
:::

> **一句话直觉**：条件（时间步）通过 scale 和 shift 告诉每一层"用什么强度工作"——去噪早期用大力，后期用小力。

### 2.2 数学表示（GR00T 中的实现）

$$
\text{AdaLN}(x, \text{temb}) = \text{LN}(x) \cdot (1 + s) + b
$$

**这个公式在做什么**：这是上面 AdaLN 公式在 GR00T 里的具体记号版本——把条件从抽象的 $c$ 换成"时间步 embedding" `temb`，缩放/偏移参数从 $\gamma_c,\beta_c$ 换成 $s,b$，公式结构完全一样。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\text{temb}$ | **时间步的向量表示** | 从 TimestepEncoder 得到的、代表"现在是去噪第几步"的 embedding |
| $s$ | **scale 参数** | 对应上面的 $\gamma_c$，由 temb 算出 |
| $b$ | **shift 参数** | 对应上面的 $\beta_c$，由 temb 算出 |
| $\text{LN}(x)\cdot(1+s)+b$ | **调制后的输出** | 标准化后的特征乘以缩放再加偏移 |

**用人话读**："这就是把'条件 $c$'具体换成'时间步 embedding'之后的 AdaLN，公式本质不变。"

**为什么是这个形式**：GR00T 中唯一需要注入的标量条件就是去噪时间步，所以直接用 temb 代替通用的 $c$，命名上更贴合实现代码。
:::

其中：
$$
[s, b] = \text{Linear}(\text{SiLU}(\text{temb}))
$$

**这个公式在做什么**：具体给出 $s,b$ 的计算方式——和上面 $[\gamma_c,\beta_c]$ 的算法完全相同，只是输入换成了 temb。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\text{SiLU}(\text{temb})$ | **非线性变换后的时间步特征** | 增强表达能力 |
| $\text{Linear}(\cdot)$ | **投影并拆分** | 输出维度是 temb 的 2 倍，切成两半 |
| $s,b$ | **scale 和 shift** | 分别用来乘和加，调制归一化后的特征 |

**用人话读**："时间步 embedding 过一次非线性激活再过一个线性层，输出前半段当缩放、后半段当偏移。"

**为什么是这个形式**：和上一个公式共用同一套设计逻辑——用一个线性层同时产出两组参数，省参数量、联合训练。
:::

**逐项拆解**：
- $x \in \mathbb{R}^{B \times T \times D}$：输入特征
- $\text{temb} \in \mathbb{R}^{B \times D}$：时间步 embedding（从 TimestepEncoder 得到）
- $\text{SiLU}(\text{temb})$：非线性激活（$\text{SiLU}(x) = x \cdot \sigma(x)$）
- $\text{Linear}$：线性层，输入维度 $D$，输出维度 $2D$（一半是 scale，一半是 shift）
- $s \in \mathbb{R}^{B \times D}$：scale 参数（对应 $\gamma$）
- $b \in \mathbb{R}^{B \times D}$：shift 参数（对应 $\beta$）
- $(1 + s)$：加 1 是为了让 scale 的默认值为 1（未经训练时 $s \approx 0$ → 不缩放）

### 2.3 GR00T 中的代码

```python
class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim)  # 输出 scale + shift
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)  # 无可学习参数的 LN
    
    def forward(self, x, temb):
        # temb: [B, D] 时间步embedding
        temb = self.linear(self.silu(temb))  # [B, 2D]
        scale, shift = temb.chunk(2, dim=1)  # 各 [B, D]
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]  # [:, None] 扩展到序列维度
        return x
```

注意 `self.norm` 使用 `elementwise_affine=False`——即标准 LayerNorm **没有**自己的 $\gamma, \beta$。
所有的缩放和偏移完全由外部条件 `temb` 控制。

---

## 3. 具体数值例子

假设 $D=4$，batch 中有 2 个样本，对应不同的去噪时间步。

**样本 1**：$t=0$（纯噪声阶段）
```
temb_1 = [2.0, -1.0, 0.5, 1.5]  (时间步 embedding)
经过 SiLU + Linear 后:
  scale_1 = [0.8, 0.8, 0.8, 0.8]  (大 scale → 大幅调制)
  shift_1 = [0.3, -0.2, 0.1, 0.4]
```

**样本 2**：$t=0.75$（接近干净阶段）
```
temb_2 = [-0.5, 0.3, -0.2, 0.1]  (不同的时间步 embedding)
经过 SiLU + Linear 后:
  scale_2 = [0.1, 0.1, 0.1, 0.1]  (小 scale → 轻微调制)
  shift_2 = [0.01, 0.02, -0.01, 0.01]
```

相同的输入 $x$，因为时间步不同，经过 AdaLN 后的输出完全不同：
- 样本 1：$\text{LN}(x) \times 1.8 + 0.3$ → 特征被大幅放大和偏移
- 样本 2：$\text{LN}(x) \times 1.1 + 0.01$ → 特征几乎不变

---

## 4. 为什么 DiT 要用 AdaLayerNorm？

### 4.1 扩散模型中条件注入的需求

扩散模型（包括 DDPM 和 Flow Matching）的核心流程是：
- 训练时：给定带噪声的数据 $x_t$ 和时间步 $t$，预测噪声/速度
- 推理时：从 $t=0$ 逐步积分到 $t=1$

关键需求：**网络必须知道当前是"去噪的第几步"**。

为什么？因为同样的输入 $x$：
- 如果 $t=0$（几乎全是噪声），网络需要预测"大方向"
- 如果 $t=0.75$（几乎已经干净），网络需要预测"微小修正"

如果网络不知道 $t$，它无法区分这两种情况。

### 4.2 条件注入的三种方式对比

| 方式 | 做法 | 缺点 |
|------|------|------|
| 拼接 | 把 $t$ 的 embedding 拼在输入后面 | 只在第一层有效，深层可能被忘记 |
| 相加 | $x + \text{temb}$ | 改变了输入的分布，可能导致训练不稳定 |
| **AdaLN** | 通过 scale/shift 调制每一层 | ✅ 每一层都直接受 $t$ 控制，效果最强 |

AdaLN 的优势：**每一层**的归一化都被条件调制——
时间步信息在网络的每一层都被"注入"，不会随着层数增加而衰减。

### 4.3 在 GR00T 中的信息流

```mermaid
flowchart LR
    T["时间步 t=250"]
    TE["TimestepEncoder<br/>sinusoidal → MLP"]
    TEMB["temb [B, 1536]"]
    
    T --> TE --> TEMB
    
    TEMB -->|"scale, shift"| L0["层0 AdaLN"]
    TEMB -->|"scale, shift"| L1["层1 AdaLN"]
    TEMB -->|"scale, shift"| L2["层2 AdaLN"]
    TEMB -->|"scale, shift"| DOTS["... 每一层都用同一个 temb"]
    TEMB -->|"scale, shift"| OUT["输出层 AdaLN-Zero"]
```

同一个 `temb` 被送入每一层的 AdaLN——每一层都"知道"当前的去噪进度。

---

## 5. AdaLN-Zero：输出层的特殊变体

GR00T 的 DiT 在最终输出时还使用了一个变体—— **AdaLN-Zero**：

```python
# DiT 输出层
conditioning = temb                    # [B, 1536]
shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)  # 各 [B, 1536]
hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
output = self.proj_out_2(hidden_states)  # 投影到 output_dim
```

和中间层 AdaLN 的区别：
- 中间层 AdaLN 后面还有注意力和 FFN
- 输出层 AdaLN 直接跟最终投影——是最后一次条件调制的机会

"Zero" 的含义：初始化时让 scale 和 shift 接近 0，
使得网络初始行为接近**恒等映射**（输出约等于输入）。
这有助于训练稳定性——刚开始训练时网络不会输出垃圾值。

---

## 6. 和其他条件化技术的对比

| 技术 | 条件注入位置 | 强度 | 计算开销 | 使用场景 |
|------|------------|------|---------|---------|
| 拼接输入 | 只在输入层 | 弱 | 低 | 简单条件 |
| Cross-Attention | 每个 cross-attn 层 | 中等 | 中等 | 序列化条件（如文本） |
| **AdaLayerNorm** | **每一层的归一化** | **强** | 低 | 标量/向量条件（如时间步） |
| FiLM (Feature-wise Linear Modulation) | 任意层 | 中等 | 低 | AdaLN 的泛化版本 |

GR00T 中两种主要条件同时使用：
- **时间步** → AdaLayerNorm（标量条件，每层注入）
- **VL 特征** → Cross-Attention（序列条件，交替注入）

两者互不冲突，各自负责不同类型的条件信息。

---

## 7. 总结

| 要点 | 内容 |
|------|------|
| **是什么** | 用外部条件动态生成 LayerNorm 的 scale 和 shift 参数 |
| **为什么需要** | 让网络行为随去噪时间步动态调整（早期大力，后期轻柔） |
| **怎么实现** | `[scale, shift] = Linear(SiLU(condition))`，然后 `LN(x) * (1+scale) + shift` |
| **和标准 LN 的区别** | 标准 LN 的 γ/β 是固定参数；AdaLN 的 scale/shift 每次根据条件动态计算 |
| **GR00T 中的应用** | DiT 的每一层都用 AdaLN 注入时间步信息 |
| **优势** | 每一层都直接受条件控制，信息不会衰减 |

AdaLayerNorm 是所有现代扩散 Transformer（DiT、SD3、Flux 等）的标准组件。
理解它是理解 DiT 架构的关键——没有 AdaLN，DiT 就不知道自己在去噪的哪一步。
