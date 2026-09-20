---
title: SAC-Flow：用 SAC 直接端到端训练 Flow 策略
order: 279
tags: [强化学习, Flow Matching, SAC, Off-Policy, 梯度稳定性, GRU, Transformer]
category: 精读
star: 5
---

# SAC-Flow：用 SAC 直接端到端训练 Flow 策略 深度精读

> **论文标题**: SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling
> **作者**: Yixian Zhang, Shu'ang Yu, Tonghe Zhang, Mo Guang, Haojia Hui, Kaiwen Long, Yu Wang, Chao Yu, Wenbo Ding
> **机构**: Tsinghua University、Carnegie Mellon University、Li Auto、Zhongguancun Academy、Shanghai AI Laboratory
> **发表**: arXiv:2509.25756, Sep 2025
> **代码**: [github.com/Elessar123/SAC-FLOW](https://github.com/Elessar123/SAC-FLOW)

**知识链接**：
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 本文直接复用的 RL 算法框架
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — Flow 策略的多步 ODE 采样过程
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — 梯度穿过多步生成链的通用困难
- [梯度消失与梯度爆炸：为什么长链反向传播会失控](/前置知识/005m_前置知识_梯度消失与梯度爆炸_雅可比连乘效应) — 本文 1.2 节问题的完整数学原理，建议先读
- [FQL：Flow Q-Learning](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) — 对比方案：用蒸馏绕开梯度问题
- [Q-Chunking：RL 与动作分块](/论文综述/071_QChunking_RL与动作分块) — QC-FQL（本文的强 baseline）出处
- [ReinFlow：Flow 策略的噪声注入 RL 微调](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) — 对比方案：on-policy PPO 路线
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Critic、Bellman 方程的基础
- [Wasserstein 距离与最优传输](/前置知识/002s_前置知识_Wasserstein距离与最优传输) — FlowRL（Lv et al.）baseline 用到的约束工具

---

## 贯穿全文的例子

> 一个 UR-5 机械臂要把桌上散落的三个方块依次搬到指定格子里（对应论文里 OGBench 的 cube-triple 任务）。这个任务天然是**多模态**的——先搬哪个方块、绕开另外两个方块走哪条路径，都有好几种同样合理的方案。我们希望用一个 Flow 策略来表达这种多模态行为，同时用 SAC 这种数据利用率很高的 off-policy 算法去训练它，让机器人从少量交互里快速学会。

---

## 一、这篇论文解决什么问题

### 1.1 背景：Flow 策略怎么生成一个动作

[Flow Matching](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) 策略不直接输出动作，而是学习一个**速度场** $v_\theta(t, A_t, s)$，再靠数值积分把一个纯噪声"推"成一个动作。具体来说，从标准高斯噪声 $A_{t_0} \sim \mathcal{N}(0, I_d)$ 出发，经过 $K$ 步 Euler 更新：

$$
A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot v_\theta(t_i, A_{t_i}, s), \quad 0 = t_0 < t_1 < \cdots < t_K = 1
$$

**这个公式在做什么**：这是"把噪声一步步搬运成动作"的具体操作规则——每一步都往速度场指的方向挪一点，挪 $K$ 次之后停在 $A_{t_K}$，就是最终生成的动作（论文里还会再过一个 $\tanh$ 把它压缩到合法的动作范围）。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $A_{t_i}$ | **当前位置** | 积分路径上第 $i$ 步时刻，噪声演化到的中间状态（还不是最终动作） |
| $v_\theta(t_i, A_{t_i}, s)$ | **导航员** | 一个神经网络，输入"现在在哪、几点了、状态是什么"，输出"接下来该往哪个方向走" |
| $\Delta t_i \cdot v_\theta(\cdot)$ | **一步的位移** | 导航员指的方向乘上这一步的时间跨度，就是这一步实际挪动的距离 |
| $A_{t_0} \sim \mathcal{N}(0, I_d)$ | **出发点** | 纯随机噪声，和任务、状态完全无关 |

**用人话读**："从一个随机点出发，问网络'往哪走'，走一小步，再问一次，再走一小步……重复 $K$ 次，走到的地方就是要执行的动作。"

**为什么是这个形式**：这是标准的 [ODE](/前置知识/001b_前置知识_常微分方程ODE直觉与数值求解) 数值积分（Euler 法）——把连续的 $\frac{dA_t}{dt} = v_\theta(t, A_t, s)$ 离散化成有限步。本文采用 Rectified Flow 的直线路径设定，$v_\theta$ 训练目标是回归 $A_1 - A_0$（详见 [Flow Matching 前置知识](/前置知识/000g_前置知识_Flow_Matching与连续归一化流)）。
:::

在**行为克隆**里，这个多步过程完全没有问题——每一步的 $v_\theta$ 独立地对着数据集里的真实速度做回归就行，训练目标里没有任何跨步的梯度传递。

### 1.2 换成 off-policy RL，问题出在哪

如果想用 **SAC** 这种 off-policy 算法直接训练 Flow 策略，Actor 更新的梯度必须从 Critic 的输出 $Q_\psi(s, a)$，一路反传穿过 $\tanh$，再穿过全部 $K$ 步 Euler 更新，最后落到 $v_\theta$ 的参数 $\theta$ 上。这条反传路径的长度正好是 $K$——和一个 $K$ 层深的循环网络做 [BPTT](/前置知识/000f_前置知识_为什么扩散策略难以RL微调)（Backpropagation Through Time）完全是同一种数学结构。

**为什么这条链容易出问题**：每一步的雅可比矩阵近似地把梯度乘上一个"局部放大系数"。如果这个系数持续略大于 1，连乘 $K$ 次后梯度会指数级放大（梯度爆炸）；如果持续略小于 1，会指数级缩小到几乎为零（梯度消失）。这正是训练 vanilla RNN 时的经典病理，完整的数学推导、数值算例和图示见 [梯度消失与梯度爆炸：为什么长链反向传播会失控](/前置知识/005m_前置知识_梯度消失与梯度爆炸_雅可比连乘效应)；[FQL 前置知识](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) 里也有对这个连乘效应的一版数值推导，可以对照阅读。

### 1.3 之前的方案都在"绕路"

| 方法 | 策略 | 代价 |
|------|------|------|
| [FQL](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) / [QC-FQL](/论文综述/071_QChunking_RL与动作分块) | 蒸馏一个单步学生网络，只对学生做 RL 更新 | 学生是单峰分布，丢失了 Flow 教师的多模态表达力 |
| FlowRL（Lv et al., 2025） | 用 [2-Wasserstein 距离约束](/前置知识/002s_前置知识_Wasserstein距离与最优传输) 构造 surrogate objective，梯度不穿过 Flow rollout | 不是在直接优化 SAC 的目标函数，只是一个近似 |
| [ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) | 用 on-policy PPO，不需要把 Q 值的梯度反传穿过 Flow | 舍弃了 off-policy 方法的高数据效率 |

**本文的立场**：以上三条路线本质上都是在"回避"梯度穿过多步 Flow 这件事本身。既然回避有代价，能不能正面解决这个问题——**直接**用 SAC 端到端训练 Flow？

### 1.4 本文的核心洞察

**关键发现**：Flow 的 $K$ 步 Euler 积分，在代数结构上和一个**残差循环神经网络**（Residual RNN）的 $K$ 步前向传播完全等价：

$$
A_{t_{i+1}} = A_{t_i} + f_\theta(t_i, A_{t_i}, s), \quad f_\theta(\cdot) = \Delta t_i \cdot v_\theta(\cdot)
$$

**这个公式在做什么**：把 Flow 的更新规则重新写成"隐藏状态 + 残差更新"的形式，让它和 RNN 的更新规则在形式上完全对齐——只要把 $A_{t_i}$ 看成 RNN 的隐藏状态、$(t_i, s)$ 看成输入、$f_\theta$ 看成 RNN cell，两者就是同一个数学对象。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $A_{t_i}$ | **RNN 的隐藏状态** $h_i$ | 携带着"走到目前为止积累的全部信息" |
| $(t_i, s)$ | **RNN 的输入** $x_i$ | 每一步喂给网络的额外信息（时间进度 + 环境状态） |
| $f_\theta(t_i, A_{t_i}, s)$ | **RNN cell** | 根据当前隐藏状态和输入，算出"这一步要往隐藏状态上加多少" |
| $A_{t_i} + f_\theta(\cdot)$ | **残差更新** | 新隐藏状态 = 旧隐藏状态 + 增量，而不是完全重新计算 |

**用人话读**："Flow 的每一步 Euler 更新，就是 RNN 的每一步残差式状态更新，两者是同一件事换了个名字。"

**为什么是这个形式**：这不是一个凑出来的类比，而是直接从 Euler 积分公式恒等变形得到的——$\Delta t_i \cdot v_\theta$ 本身就是加到 $A_{t_i}$ 上的增量，天然满足残差 RNN 的定义（Goel et al., 2017 提出的 R2N2 架构正是这种残差 RNN）。
:::

这个等价关系直接**解释了**为什么标准 Flow + SAC 训不稳：它面临和 vanilla RNN 完全一样的梯度病态。**解决方案也就顺理成章**——既然问题的数学本质是"深层循环计算的梯度不稳定"，就照搬 RNN 领域已经验证过的解法：用 **GRU 式门控**或 **Transformer decoder** 重新设计速度场 $v_\theta$ 的内部结构，SAC 算法本身完全不用改。

---

## 二、方法：两种稳定的速度网络设计

### 2.1 Flow-G：GRU 风格的门控速度

标准 Flow 的每步更新是"无条件"地加上残差 $A_{t_{i+1}} = A_{t_i} + \Delta t \cdot v_\theta$——不管这一步的更新是好是坏，全盘接受。Flow-G 的想法是给这次更新加一道"闸门"，让网络自己学会什么时候该大步走、什么时候该原地不动：

$$
A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot \big(g_i \odot (\hat{v}_\theta(t_i, A_{t_i}, s) - A_{t_i})\big), \qquad g_i = \sigma\big(z_\theta(t_i, A_{t_i}, s)\big) \in (0, 1)^d
$$

**这个公式在做什么**：给"要不要更新、更新多少"这件事装一个可学习的刹车。网络先算出一个"候选新位置" $\hat{v}_\theta$，再用门 $g_i$ 决定往这个候选位置挪多远——门开得越大，挪得越多；门几乎关闭，就基本停在原地不动，从而截断了这一步可能带来的梯度放大。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $z_\theta(t_i, A_{t_i}, s)$ | **门控网络的原始输出** | 一个小型 MLP，还没经过 sigmoid 压缩，取值范围是全体实数 |
| $g_i = \sigma(z_\theta(\cdot)) \in (0,1)^d$ | **每维独立的闸门** | 把原始输出压到 0~1 之间，$d$ 个动作维度各有自己的开关，互不干扰 |
| $\hat{v}_\theta(t_i, A_{t_i}, s)$ | **候选网络** | 另一个小型 MLP，提出"如果不设限，我想跳到哪个新位置" |
| $\hat{v}_\theta - A_{t_i}$ | **候选位移** | 候选新位置和当前位置的差值——即"想挪动的距离和方向" |
| $g_i \odot (\hat{v}_\theta - A_{t_i})$ | **门控后的实际速度** | 按逐维度的门控系数缩小候选位移，得到真正采用的速度 $v_\theta$ |

**用人话读**："网络先猜一个候选目标点，再用一个 0~1 之间的旋钮控制真正往那个目标点挪多远——旋钮拧到 0 就原地不动，拧到 1 就完全采纳候选目标。"

**为什么是这个形式**：这正是 [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) 的更新门（update gate）机制，只是写成了 Flow 速度场的语言。GRU 用门控来解决 vanilla RNN 的梯度问题，本质是让梯度反传路径上出现一个可学习的"衰减系数"——训练早期如果某一步容易导致梯度爆炸，网络可以自己学会把该步的门关小，切断这条路径上的梯度放大。
:::

**代入数字直觉理解**（简化成 1 维）：

- 当前位置 $A_{t_i} = 0.3$，候选目标 $\hat{v}_\theta = 0.9$，差值 $= 0.6$
- 若门 $g_i = 0.2$（几乎关闭）：门控后速度 $= 0.2 \times 0.6 = 0.12$，只挪了一小步，反传时这一步对梯度的放大效应被压得很小
- 若门 $g_i = 0.9$（几乎全开）：门控后速度 $= 0.9 \times 0.6 = 0.54$，大步跳向候选目标

下图展示了 sigmoid 门控函数 $g = \sigma(z)$ 的形状——这就是决定"刹车松紧"的核心非线性：

![Sigmoid 门控函数——原始信号 z 如何被压缩成 0~1 之间的开关系数](/sac_flow_sigmoid_gate.png)

> 图中标出了两个典型工作点：$z=-3$ 时门几乎完全关闭（$g\approx0.05$），这一步基本不更新，梯度也基本不通过；$z=3$ 时门几乎完全打开（$g\approx0.95$），网络几乎全盘接受候选速度。中间的过渡区（$z$ 接近 0）是门控网络学习"该不该更新"的敏感区间。

### 2.2 Flow-T：Transformer Decoder 风格的速度

第二种设计彻底换了一套结构：不再用 MLP 直接算速度，而是让"当前动作-时间 token"通过若干层 **state-only cross-attention**（只查询状态、不和其它时间步的 token 混合信息）逐层精炼自己：

$$
\Phi_{A_i}^{(0)} := \Phi_{A_i} = E_A(\phi_t(t_i), A_{t_i}), \qquad \Phi_S = E_S(\phi_s(s))
$$

**这个公式在做什么**：先把"当前动作 + 时间"打包成一个 token，把"环境状态"单独打包成另一个 token——这两个 token 是接下来所有层要处理的原始材料。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $E_A(\phi_t(t_i), A_{t_i})$ | **动作 token 的构造方式** | 把时间嵌入 $\phi_t(t_i)$ 和当前中间动作 $A_{t_i}$ 拼接后线性投影,压缩成统一维度的 token 向量 |
| $E_S(\phi_s(s))$ | **状态 token 的构造方式** | 把环境状态 $s$ 编码后线性投影,得到一个全局共享的 token,后面每一层都会被查询 |

**用人话读**："先把'现在几点、动作走到哪了'打包成一个 token,把'环境是什么样'单独打包成另一个 token。"

**为什么是这个形式**：Cross-attention 需要 query 和 context 是两个独立的 token 集合,这一步就是在准备这两个集合。
:::

$$
Y_i^{(l)} = \Phi_{A_i}^{(l-1)} + \text{Cross}_l\big(\text{LN}(\Phi_{A_i}^{(l-1)}), \; \text{context}=\text{LN}(\Phi_S)\big), \qquad \Phi_{A_i}^{(l)} = Y_i^{(l)} + \text{FFN}_l(\text{LN}(Y_i^{(l)}))
$$

**这个公式在做什么**：动作 token 在第 $l$ 层里"向状态 token 提问"（cross-attention）获得一次修正，再过前馈网络做一次自我精炼——两次修正都是以残差形式叠加，不是替换。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\text{Cross}_l(\cdot, \text{context}=\cdot)$ | **状态查询模块** | 动作 token 去"问"状态 token："基于当前状态,我该怎么调整自己?"——这是外部信息进入速度场的唯一通道 |
| $Y_i^{(l)} = \Phi_{A_i}^{(l-1)} + \text{Cross}_l(\cdot)$ | **第一次残差连接** | Cross-Attention 的输出是增量,加回原 token,不是替换掉原 token |
| $\Phi_{A_i}^{(l)} = Y_i^{(l)} + \text{FFN}_l(\cdot)$ | **第二次残差连接** | 前馈网络的输出同样以增量形式叠加 |

**用人话读**："每一层里,动作 token 先向状态提一次问、再自我精炼一次,两步都是'加上一点修正'而不是'推翻重来'。"

**为什么是这个形式**：去掉自注意力（不和其它时间步的 token 混合信息）是为了保持 Flow 的马尔可夫性质——$A_{t_{i+1}}$ 只能依赖当前的 $A_{t_i}$ 和 $s$,不能偷看路径上其它时刻的信息。两次残差连接分别对应标准 Transformer decoder block 里的 attention 子层和 FFN 子层。
:::

$$
A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot W_o\big(\text{LN}(\Phi_{A_i}^{(L)})\big)
$$

**这个公式在做什么**：把最后一层精炼完的 token 投影回速度空间的维度，得到这一步 Euler 更新要用的速度 $v_\theta(t_i, A_{t_i}, s) = W_o(\text{LN}(\Phi_{A_i}^{(L)}))$。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\Phi_{A_i}^{(L)}$ | **最后一层的精炼结果** | 经过 $L$ 层"提问+精炼"之后,动作 token 携带的最终信息 |
| $W_o(\text{LN}(\cdot))$ | **解码头** | 先归一化,再线性投影回动作维度,得到实际可用的速度向量 |

**用人话读**："把 Transformer 层层精炼完的 token,最后转换成一个和动作维度一样大小的速度向量,直接拿去做 Euler 更新。"

**为什么是这个形式**：这里刻意去掉了 Transformer 常见的**自注意力（跨时间步的 token 混合）**——只保留 cross-attention。原因是 Flow 的采样过程要保持**马尔可夫性质**：$A_{t_{i+1}}$ 只能依赖当前的 $A_{t_i}$ 和状态 $s$，不能"偷看"路径上其它时间步的信息，否则会破坏 Flow 本身的数学定义。稳定性来自两处标准设计：**pre-norm residual connection** 保证梯度至少有一条不经过任何非线性变换的"直通高速公路"（恒等映射的梯度恒为 1）；LayerNorm 则把每层输出的数值量级重新拉回到统一范围，防止逐层累积的数值漂移。
:::

### 2.3 两种设计的对比

| | 原始 MLP 速度 | Flow-G（门控） | Flow-T（Transformer） |
|--|-------------|-------------|---------------------|
| 对应的序列模型 | Vanilla RNN | GRU | Transformer Decoder |
| 稳定性机制 | 无 | 逐维度可学习门控 | Residual + Pre-LN |
| 梯度稳定性 | ❌ 随步数指数级爆炸/消失 | ✅ 稳定 | ✅ 更稳定，对步数 $K$ 增大更鲁棒 |
| 参数量 | 小 | 中 | 中-大 |
| 对采样步数 $K$ 的鲁棒性 | 差 | 好 | 最好（论文 Fig.7 显示 $K=4,7,10$ 下表现一致） |

下图用示意曲线直观展示论文消融实验（Fig. 6a）的核心发现——沿着反向传播路径（从最后一步 $k=K-1$ 向第一步 $k=0$ 走），Naive Flow 的梯度范数指数级增长，而 Flow-G/Flow-T 全程稳定：

<img src="/sac_flow_gradient_norm_schematic.png" alt="梯度范数随反传深度变化的示意图——Naive Flow 指数增长，Flow-G/Flow-T 保持稳定" class="img-wide">

> 注意这张图是根据论文文字描述（"最大变化量仅 0.29"、"Naive baseline 梯度范数从 $k=3$ 到 $k=0$ 急剧上升"）绘制的**示意曲线**，用于建立直觉，并非论文原始实验数据的复现。

---

## 三、怎么算 log-prob：噪声增广 Rollout

### 3.1 问题：确定性 Flow 没有解析 log-prob

[SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) 的熵正则项需要计算 $\log \pi(a|s)$。但第一节里的 $K$ 步 Euler 积分是**确定性的**——给定固定的 $A_{t_0}$，输出的 $A_{t_K}$ 是唯一确定的。策略的真实密度 $\pi_\theta(a|s)$ 需要对所有可能的初始噪声 $A_{t_0}$ 积分求出，这个积分在高维空间中没有闭式解，计算上不可行。

### 3.2 解法：给每一步加噪声，让确定性积分变成随机过程

论文的解法是在每一步 Euler 更新里注入一点各向同性的高斯噪声，同时给漂移项 $b_\theta$ 加一个补偿量，使得**最终动作的边缘分布不变**：

$$
A_{t_{i+1}} = A_{t_i} + b_\theta(t_i, A_{t_i}, s) \cdot \Delta t_i + \sigma_\theta \sqrt{\Delta t_i} \cdot \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, I_d)
$$

**这个公式在做什么**：把原来"每一步走到哪都是确定的"改成"每一步走到哪是一个高斯分布"——多了一点随机抖动。只要漂移 $b_\theta$ 补偿得恰当，抖动不会改变最终动作 $A_{t_K}$ 的整体分布，但换来的好处是：现在每一步的转移概率都是一个**显式高斯**，可以直接写出解析的对数密度。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $b_\theta(t_i, A_{t_i}, s)$ | **补偿后的漂移** | 不是原始速度场 $v_\theta$ 本身，而是经过修正的版本——修正量正好抵消掉后面加噪声带来的分布偏移 |
| $\sigma_\theta \sqrt{\Delta t_i} \cdot \varepsilon_i$ | **注入的随机抖动** | 每一步额外加的高斯噪声，$\sigma_\theta$ 通常固定为一个小常数（论文用 0.10），$\sqrt{\Delta t_i}$ 是标准 SDE 离散化里噪声项的标准写法 |
| $\varepsilon_i \sim \mathcal{N}(0, I_d)$ | **噪声源** | 每一步独立采样一次的标准高斯随机变量 |

**用人话读**："每走一步除了按导航方向挪动，再随机抖动一下——抖动的幅度固定，方向的修正量经过精心设计，保证最终落点的整体分布和不抖动时一样。"

**为什么是这个形式**：这是 SDE（[随机微分方程](/前置知识/001c_前置知识_随机微分方程SDE直觉与扩散模型的联系)）离散化的标准写法。漂移项的具体补偿公式（论文附录 A.1 给出闭式表达式）保证了这个随机化过程"绕了个圈子"但终点分布不变——之所以要这么绕，是因为只有变成随机过程后，每一步的转移才有显式的高斯密度可以计算 log-prob。
:::

这样，每一步的转移概率就是一个显式高斯：

$$
\eta_\theta(A_{t_{i+1}} | A_{t_i}, s) = \mathcal{N}\big(A_{t_i} + b_\theta \Delta t_i, \; \sigma_\theta^2 \Delta t_i \cdot I_d\big)
$$

**这个公式在做什么**：把上一段文字描述的"每一步转移是显式高斯"写成正式的数学形式——均值是加了漂移补偿后的新位置，方差由固定噪声强度 $\sigma_\theta$ 和步长 $\Delta t_i$ 共同决定。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $A_{t_i} + b_\theta \Delta t_i$ | **高斯的均值** | 加噪声之前"应该"到达的位置——补偿后的漂移乘以步长 |
| $\sigma_\theta^2 \Delta t_i \cdot I_d$ | **高斯的协方差** | 各维度独立、方差相等的对角协方差矩阵，方差大小由噪声强度和步长共同决定 |

**用人话读**："第 $i+1$ 步的落点服从一个高斯分布，中心在补偿后应该到达的位置，散布程度由噪声强度和步长决定。"

**为什么是这个形式**：这是上一个公式（噪声增广 Euler 更新）逐项对号写出的转移密度——因为更新规则里加的是高斯噪声，转移天然就是高斯分布，均值和方差直接对应更新公式里的漂移项和噪声项。
:::

整条采样路径 $\mathcal{A} = (A_{t_0}, \ldots, A_{t_K})$ 的联合密度就是这些高斯转移的乘积，再乘上初始噪声密度和 $\tanh$ 压缩带来的雅可比修正：

$$
p_c(\mathcal{A}|s) = \zeta(A_{t_0}) \prod_{i=0}^{K-1} \eta_\theta(A_{t_{i+1}} | A_{t_i}, s) \cdot \|\det \mathcal{J}(a)\|^{-1}, \qquad a = \tanh(A_{t_K})
$$

**这个公式在做什么**：把"整条路径出现的概率"拆成三块乘在一起——起点的概率、每一步转移的概率（连乘）、以及最后把 $A_{t_K}$ 压缩进 $[-1,1]$ 区间时坐标变换带来的密度修正。这就是 SAC 熵项里要用到的 $\log \pi_\theta$ 的**可计算替身**。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\zeta(A_{t_0})$ | **起点密度** | 标准高斯 $\mathcal{N}(0, I_d)$ 在起点处的概率密度值，闭式可算 |
| $\prod_{i=0}^{K-1} \eta_\theta(\cdot)$ | **路径转移的连乘** | 每一步都是显式高斯，$K$ 项乘起来就是"沿着这条特定路径走"的总概率 |
| $\|\det \mathcal{J}(a)\|^{-1}$ | **压缩变换的密度修正** | $\tanh$ 把无界的 $A_{t_K}$ 压缩到有界区间，密度也要跟着按雅可比行列式的倒数缩放（对逐元素 $\tanh$，$\|\det\mathcal{J}(a)\| = \prod_j (1-a_j^2)^{-1}$） |

**用人话读**："一条路径出现的可能性 = 起点在哪的可能性 × 每一步恰好走到下一个点的可能性连乘 × 最后压缩坐标系带来的修正因子。"

**为什么是这个形式**：这是标准的马尔可夫链联合密度分解（链式法则）加上变量替换公式（change of variables）。取对数后连乘变成连加，$\log p_c(\mathcal{A}|s) = \log\zeta(A_{t_0}) + \sum_i \log\eta_\theta(\cdot) - \log\|\det\mathcal{J}(a)\|$，每一项都能解析算出，梯度也能顺着这条链反传——这正是把它塞进 SAC 熵项能成立的原因。
:::

$\sigma_\theta$ 在从零开始训练时通常固定为 0.10，这个 $\log p_c$ 就直接替代标准 SAC 里的 $\log \pi_\theta$ 使用。

---

## 四、完整算法：SAC-Flow 的训练流程

### 4.1 Actor Loss

$$
L_{\text{actor}}(\theta) = \alpha \log p_c(\mathcal{A}^\theta | s_h) - Q_\psi(s_h, a_h^\theta), \qquad \mathcal{A}^\theta \sim \pi_\theta(\cdot|s_h), \quad a_h^\theta = \tanh(A_{t_K}^\theta)
$$

**这个公式在做什么**：这就是标准 [SAC 的 Actor loss](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)，唯一的区别是把"高斯策略的 $\log\pi$"替换成了上一节构造出的"$K$ 步噪声 Flow 路径的 $\log p_c$"。结构完全没变——同时最大化 Q 值、保留策略的随机性（熵）。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\alpha \log p_c(\mathcal{A}^\theta \| s_h)$ | **熵正则项** | $\alpha$ 是温度系数；$p_c$ 越大代表策略越"确定"，这一项就越大，作为 loss 的一部分被最小化时会推动策略保持随机性 |
| $-Q_\psi(s_h, a_h^\theta)$ | **价值最大化项** | 让选出的动作 $a_h^\theta$ 尽量得到高 Q 值；负号是因为要最大化 Q 就是最小化 $-Q$ |
| $a_h^\theta = \tanh(A_{t_K}^\theta)$ | **重参数化采样的动作** | $A_{t_K}^\theta$ 是从当前策略 $\pi_\theta$ 采样（带着噪声增广 rollout）得到的终点，梯度可以顺着这条采样路径回传到 $\theta$ |

**用人话读**："和标准 SAC 完全一样——选出让 Q 值高的动作，但别选得太'死板'（保留一点随机性）。只是这里的'动作'是走完 $K$ 步噪声 Flow 之后才得到的。"

**为什么是这个形式**：梯度要从 $Q_\psi(s_h, a_h^\theta)$ 一路穿过 $\tanh$，再穿过整个 $K$ 步噪声 Flow rollout，最后到达 $\theta$。这条链之所以不会梯度爆炸/消失，正是因为 $v_\theta$ 用了第二节的 Flow-G 或 Flow-T 参数化——这是全篇论文最终落地成一个可训练目标的地方。
:::

### 4.2 Critic Loss

$$
L_{\text{critic}}(\psi) = \Big[Q_\psi(s_h, a_h) - \big(r_h + \gamma Q_{\bar\psi}(s_{h+1}, a_{h+1}) - \alpha \log p_c(\mathcal{A}_{h+1} | s_{h+1})\big)\Big]^2
$$

**这个公式在做什么**：让 Critic 的预测值 $Q_\psi(s_h, a_h)$ 去逼近一个"目标值"——目标值由即时奖励加上下一步的（软）价值构成，这就是标准的 TD（时序差分）误差平方。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $Q_\psi(s_h, a_h)$ | **当前的预测** | Critic 网络对"这个状态-动作对到底值多少"的当前估计 |
| $r_h + \gamma Q_{\bar\psi}(s_{h+1}, a_{h+1})$ | **一步展开的目标价值** | 走一步拿到的真实奖励，加上折扣后"下一步还能值多少"的估计 |
| $-\alpha \log p_c(\mathcal{A}_{h+1} \| s_{h+1})$ | **软化项（熵红利）** | 下一步的策略越随机（$\log p_c$ 越小/越负），这一项越大，鼓励价值估计给"保持探索"留出空间——这正是"软"（Soft）Q 学习的来源 |
| $[\cdot]^2$ | **平方误差** | 当前预测和目标值的差距，梯度只更新 $\psi$，$\bar\psi$ 和目标里的 $\log p_c$ 项都不参与反传 |

**用人话读**："让 Critic 的打分尽量接近'真实拿到的奖励 + 下一步的软价值'，差多少就按平方误差惩罚多少。"

**为什么是这个形式**：这是不带任何修改的标准 SAC Critic loss（见 [SAC 前置知识](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)），唯一的变化是 $\log\pi_\theta$ 换成了本文构造的 $\log p_c$——因为两者在数学上扮演的是完全相同的角色（策略的对数密度），可以直接替换。
:::

这和标准 [SAC 的 Critic TD loss](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) 完全一样：$(s_h, a_h, r_h, s_{h+1})$ 从 Replay Buffer 里采样，$a_{h+1}$ 由当前 Flow 策略在 $s_{h+1}$ 上重新采样得到,$\bar\psi$ 是延迟更新的目标网络参数,梯度不经过它反传。

### 4.3 Offline-to-Online 变种

对稀疏奖励任务，先用行为克隆式的 Flow Matching 目标在专家数据上预训练策略，再切换到在线阶段。在线阶段的 Actor loss 额外加一个"贴近 Buffer"的正则项：

$$
L_{\text{actor}}^o(\theta) = \alpha \log p_c(\mathcal{A}^\theta | s_h) - Q_\psi(s_h, a_h^\theta) + \beta \|a_h^\theta - a_h\|_2^2, \qquad (s_h, a_h) \sim \mathcal{B}
$$

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\alpha \log p_c(\mathcal{A}^\theta \| s_h) - Q_\psi(s_h, a_h^\theta)$ | **原有的 SAC Actor 目标** | 和从零训练时完全一样的两项——保熵、提 Q 值 |
| $\beta \|a_h^\theta - a_h\|_2^2$ | **贴近 Buffer 的惩罚项** | 新策略生成的动作和 Buffer 里已知真实动作之间的欧式距离平方，系数 $\beta$ 决定"贴多紧" |
| $(s_h, a_h) \sim \mathcal{B}$ | **数据来源** | 这一对状态-动作直接从 Replay Buffer 采样，而不是重新生成 |

**用人话读**："在标准 SAC 目标之外，额外要求新策略生成的动作别离 Buffer 里的已知动作太远。"

**为什么需要这个公式**：稀疏奖励环境下，如果一开始就让策略自由探索、只受 Q 值驱动，很容易在 Q 网络还没学准之前就跑偏。这一项强制新策略生成的动作 $a_h^\theta$ 不要离 Buffer 里已知的真实动作 $a_h$ 太远,是标准的 offline-to-online 保守化手段。$\beta$ 在 OGBench 的 cube 系列任务上设为 100~300,在 Robomimic 上设为 10000（非常保守,因为 Robomimic 数据量更小、更容易被 Q 值误导）。
:::

### 4.4 完整算法流程

```
Algorithm: SAC Flow (from scratch)
1. 初始化 Critic Q_ψ、目标 Critic Q_ψ̄、Flow 策略 π_θ（用 Flow-G 或 Flow-T 参数化 v_θ）、Replay Buffer B
2. for each update do
3.     用 π_θ 与环境交互；把 (s_t, a_t, r_t, s_{t+1}) 存入 B
4.     从 B 中采样一个 batch {(s_h, a_h, r_h, s_{h+1})}
5.     Actor: 用 K 步噪声增广 rollout 采出 a_h^θ；最小化 L_actor(θ)
6.     Critic: 最小化 L_critic(ψ)；用指数滑动平均更新目标网络 ψ̄
7. end for
```

Offline-to-online 版本结构相同，区别是先用 $\mathcal{D}_{\text{expert}}$ 初始化 Buffer,在离线阶段（前 $L_{\text{off}}$ 次更新）额外做 Flow-matching 预训练,并把 Actor loss 换成 4.3 节带正则项的版本。

---

## 五、和其他方法的根本区别

| 方法 | 怎么处理"梯度穿过 Flow"问题 | 保留 Flow 的多模态？ | Off-policy？ |
|------|--------------------------|-------------------|------------|
| **SAC-Flow（本文）** | 重新参数化速度网络（GRU/Transformer），让梯度本身稳定 | ✅ 完整保留 | ✅ |
| [FQL](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) / [QC-FQL](/论文综述/071_QChunking_RL与动作分块) | 蒸馏成单步网络,只对学生做 RL | ❌ 学生是单峰分布 | ✅ |
| FlowRL（Lv et al., 2025） | 用 [Wasserstein-2 距离约束](/前置知识/002s_前置知识_Wasserstein距离与最优传输) 代替直接优化 SAC 目标 | 部分保留 | ✅ |
| [ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) | 用 on-policy PPO,不需要 Q 梯度穿过 Flow | ✅ | ❌（on-policy） |

**本文的定位**：第一个能**直接用 SAC 端到端训练多步 Flow 策略**的方法,不需要蒸馏、不需要 surrogate objective,完整保留了 Flow 策略的多模态表达力。

---

## 六、实验结果

### 6.1 From-scratch 训练（MuJoCo 密集奖励）

论文在 Hopper、Walker2d、HalfCheetah、Ant、Humanoid、HumanoidStandup 六个标准连续控制任务上,对比 SAC Flow-T、SAC Flow-G 和五个 baseline（QSM、DIME、FlowRL、Gaussian SAC、Gaussian PPO）。SAC Flow-T 和 SAC Flow-G 在全部六个任务上都取得了最优表现,尤其在高维、高难度的 Humanoid 与 HumanoidStandup 任务上优势最明显——这两个任务的动作维度高达 17,Flow 策略的多模态表达力在这里体现得最充分。所有 Flow-based baseline（含 FlowRL）的收敛速度普遍快于 Diffusion-based baseline（DIME、QSM）。

### 6.2 Offline-to-Online（OGBench + Robomimic 稀疏奖励）

在 cube-double/triple/quadruple（OGBench）和 lift/can/square（Robomimic）任务上,对比 [ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调)、[FQL](/前置知识/001p_前置知识_FQL_Flow_Q_Learning)、[QC-FQL](/论文综述/071_QChunking_RL与动作分块) 三个 baseline。所有方法先做 1M 次离线更新,再做 1M 步在线交互。在难度最高的 cube-triple 和 cube-quadruple 上,SAC Flow-T 收敛最快、最终成功率最高;在 Robomimic 上,由于正则化系数 $\beta$ 设得很大（10000）限制了 Flow 模型的学习空间,SAC Flow-T/G 的表现与 QC-FQL 接近。相比 on-policy 的 ReinFlow,SAC Flow-T/G 在同样 1M 在线步数内表现更好,体现出 off-policy 方法固有的数据效率优势。

### 6.3 消融实验：梯度稳定性验证

论文直接测量了训练过程中梯度范数随反传深度（从最后一步 $k=K-1$ 走到第一步 $k=0$）的变化：

- **Naive SAC + 标准 MLP 速度**：梯度范数从最后一步到第一步呈**急剧上升**趋势（第五节的示意图就是根据这个结论绘制的）
- **SAC Flow-G / Flow-T**：梯度范数在所有反传步骤上都保持稳定,论文报告的最大波动幅度只有 0.29

这个消融直接验证了第一节的核心假设——梯度不稳定确实是标准 Flow + SAC 训练失败的根本原因,而不是别的什么因素;重新参数化速度网络的结构（而不是调学习率、加梯度裁剪之类的表面手段）能从根源上解决这个问题。论文进一步测试了不同采样步数 $K=4,7,10$ 下的表现,SAC Flow-T 对步数增加的鲁棒性尤其突出。

---

## 七、对读者最重要的 Takeaway

1. **Flow 的多步采样 = 残差 RNN 的多步前向**。理解了这个代数等价关系,就理解了为什么直接用 off-policy RL 训练 Flow 策略会遇到困难——这和训练一个很深的 RNN 面对的是完全相同的梯度病态,而不是 RL 算法本身有什么问题。

2. **解决方案是改速度网络的内部参数化,而不是改 RL 算法**。[SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) 本身不需要任何修改,只需要把 Flow 内部原本用 MLP 实现的速度场 $v_\theta$,替换成 GRU 式的门控速度（Flow-G）或 Transformer decoder 式的解码速度（Flow-T）。

3. **噪声增广 rollout 是让 SAC 熵目标能用在 Flow 上的关键技巧**。给 Flow 每一步加一点精心设计的噪声,把原本没有解析密度的确定性 ODE 变成一条 $K$ 步高斯链,log-prob 就变得可以精确计算。

4. **不再需要蒸馏**。[FQL](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) 路线把 Flow 蒸馏成单步网络再做 RL,丢失了多模态表达能力。SAC-Flow 直接端到端训练原始 Flow 模型,保留了完整表达力,也少了一套额外的蒸馏网络需要维护。

---

## 延伸阅读

- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 框架本身的详细原理
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — Flow Matching 的数学基础
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — 梯度穿过多步生成链的通用背景
- [FQL：Flow Q-Learning](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) — 对比方案：蒸馏路线,以及"连乘 $N$ 步雅可比矩阵"效应的详细数值推导
- [ReinFlow：Flow 策略的噪声注入 RL 微调](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) — 对比方案：on-policy PPO 路线
- [Q-Chunking：RL 与动作分块](/论文综述/071_QChunking_RL与动作分块) — QC-FQL 的完整精读,本文重要的对比 baseline
- [Wasserstein 距离与最优传输](/前置知识/002s_前置知识_Wasserstein距离与最优传输) — 理解 FlowRL（Lv et al.）baseline 用到的约束工具
- [GR00T-N1.7 四种 RL 方案全景对比](/工程实践/GR00T_N1d7_四种RL方案全景对比) — SAC-Flow 思路在真实 VLA 工程中的落地对比
- [SAC-FLOW-G 完全解剖：GR00T VLA 在线强化学习链路的工程实现](/工程实践/SAC_FLOW_G完全解剖_GR00T_VLA在线强化学习链路的工程实现) — 本文方法的真实代码级工程实现

**原始论文**：Zhang, Yu, Zhang, Guang, Hui, Long, Wang, Yu, Ding, "SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling", arXiv:2509.25756, 2025
