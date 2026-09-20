---
title: BootRL：冻结 VLA 加轻量 RL Head 的高效在线强化学习
order: 213
tags: [强化学习, VLA, PPO, RL Token, 轻量适配, 冻结backbone]
category: 精读
star: 3
---

# BootRL：冻结 VLA + 轻量 RL Head 深度精读

> **论文标题**: Bootstrapping Online RL with Vision-Language-Action Models  
> **作者**: Yito de Morais, Corentin Léger, Yann Berthelot, et al.  
> **机构**: Hugging Face, INRIA  
> **发表**: arXiv:2604.23073, 2025  
> **代码**: https://github.com/huggingface/bootrl

**标签**: `#VLA` `#强化学习` `#PPO` `#RLToken` `#轻量适配` `#冻结backbone` `#Actor-Critic`

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 核心机制
- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 另一种 Actor-Critic 算法
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — SFT → RL 范式
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — 防止策略崩溃
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — VLA + RL 全景图
- [VLA-RL 精读](./006_VLA_RL_PPO直接训练自回归VLA) — 对比：全模型 RL 训练

---

## 一、背景与动机

### 1.1 全模型 RL 训练的代价

现有 VLA RL 方法（如 VLA-RL、RIPT-VLA、SimpleVLA-RL）都需要对整个 VLA 模型做 RL 微调：

| 方法 | 训练参数量 | 显存需求 | 训练时间 | 泛化保持 |
|------|-----------|---------|---------|---------|
| VLA-RL (LoRA) | ~50M（LoRA） | 48GB+ | 48h | 部分丢失 |
| RIPT-VLA (Full) | 7B（全量） | 80GB+ | 24h | 明显丢失 |
| SimpleVLA-RL | ~50M（LoRA） | 64GB+ | 18h | 部分丢失 |

**两个核心问题**：
1. **计算成本**：即使用 LoRA，7B 模型的反向传播仍然很贵
2. **灾难性遗忘**：RL 微调会破坏 VLA 预训练获得的泛化能力——在目标任务上变好，在其他任务上变差

### 1.2 BootRL 的核心思路

**完全冻结 VLA backbone，只训一个极小的 Actor-Critic head（~100M 参数）。**

核心观察：VLA 的 hidden states 已经编码了丰富的视觉-语言-运动信息。我们不需要改变这些表示——只需要在其之上学一个"如何利用这些表示做出更好动作"的小网络。

**类比**：就像一个经验丰富的建筑师（VLA backbone）已经画好了蓝图（hidden states），我们只需要雇一个小施工队（RL head）来按蓝图执行，不需要让建筑师重新学习。

### 1.3 贯穿全文的例子

> **场景**：桌面机械臂执行 pick-and-place 任务——"把绿色积木放到蓝色盒子里"。
>
> VLA backbone 已经"理解"了这个任务（能生成大致正确的动作），但成功率只有 65%。我们想用 RL 提升到 90%+，但不想破坏 VLA 在其他任务上的泛化能力。
>
> BootRL 的方案：冻结 VLA，在其 hidden states 上加一个 100M 的小 head 做 RL 训练。

---

## 二、方法：RL Token + Actor-Critic Head

### 2.1 RL Token 机制

BootRL 的核心创新是 **RL Token**——在 VLA 的输入序列中插入一个特殊的 token 位置，专门用于 RL head 的信息提取。

```mermaid
flowchart LR
    subgraph Input["VLA 输入序列"]
        IMG["[IMG tokens]"] --> INST["[指令 tokens]"] --> RLT["[RL Token]"] --> ACT["[动作 tokens]"]
    end
    subgraph VLA["冻结的 VLA Backbone"]
        Input --> TF["Transformer Layers<br/>(全部冻结)"]
    end
    subgraph Output["输出"]
        TF --> H_RL["h_RL = 隐藏状态<br/>（RL Token 位置）"]
        TF --> H_ACT["h_ACT = 隐藏状态<br/>（动作 Token 位置）"]
        H_RL --> HEAD["RL Head<br/>(~100M, 可训练)"]
        H_ACT --> VLA_ACT["VLA 原始动作<br/>(anchor)"]
        HEAD --> RL_ACT["RL 修正动作"]
    end
```

**RL Token 的作用**：
- VLA transformer 在处理完图像 + 指令后，在 RL Token 位置积累了对当前状态的**全局理解**
- 这个位置的 hidden state $h_{\text{RL}}$ 是对当前场景最浓缩的表征
- RL head 只需要从这一个 vector 出发做决策——极其高效

### 2.2 Actor-Critic Head 结构

RL Head 是一个小型的 MLP 网络，同时包含 Actor 和 Critic：

$$
\text{Actor}: \quad \mu_{\text{RL}}, \sigma_{\text{RL}} = f_{\text{actor}}(h_{\text{RL}}) \in \mathbb{R}^{d} \times \mathbb{R}^{d}
$$

**这个公式在做什么**：Actor 网络吃进 VLA 在 RL Token 位置的隐藏状态，吐出一个高斯分布的均值和标准差——这个分布决定了 RL head 打算对 VLA 原始动作做多大的修正。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $h_{\text{RL}} \in \mathbb{R}^{4096}$ | **场景的浓缩摘要** | VLA backbone 在 RL Token 位置输出的隐藏状态，装着这一帧场景的全局理解（7B 模型通常是 4096 维） |
| $f_{\text{actor}}$ | **小施工队的决策脑** | 3 层 MLP（4096→1024→512→2d），把摘要向量翻译成"往哪修、修多少、有多不确定"的分布参数 |
| $\mu_{\text{RL}}\in\mathbb{R}^d$ | **修正量的中心猜测** | RL head 建议的修正均值，$d$ 是动作空间维度（如 7 维：xyz + rotation + gripper） |
| $\sigma_{\text{RL}}\in\mathbb{R}^d$ | **修正量的不确定性** | 建议的标准差，越大表示越倾向于探索更大范围的修正 |

**用人话读**："把 VLA 对当前场景的理解喂给一个小网络，它输出'该往哪个方向修正动作、修正幅度大概多大、还有多不确定'。"

**为什么是这个形式**：Actor 输出分布参数而不是直接输出一个确定值，是为了保留 PPO 训练需要的随机性（探索）；标准差训练初期设大、后期衰减（见 5.2 节技巧 3）。
:::

$$
\text{Critic}: \quad V(s) = f_{\text{critic}}(h_{\text{RL}}) \in \mathbb{R}
$$

**这个公式在做什么**：Critic 网络用同一个隐藏状态打一个分，估计"当前状态未来还能拿多少期望回报"——这是 PPO 训练算 advantage 时用的基准线。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $h_{\text{RL}}$ | **场景的浓缩摘要** | 和 Actor 共享同一个输入来源，不需要额外的 backbone forward |
| $f_{\text{critic}}$ | **打分裁判** | 3 层 MLP（4096→1024→512→1），把摘要向量映射成一个标量分数 |
| $V(s)$ | **这个状态的估值** | 预测从当前状态出发、按当前策略走下去能拿到的期望折扣回报 |

**用人话读**："同样用 VLA 给出的场景摘要，另一个小网络单独打一个'这个状态大概值多少分'的分数。"

**为什么是这个形式**：Actor 和 Critic 共享 $h_{\text{RL}}$ 这一个输入，只是各自接一个独立的小 MLP head，两者都只依赖冻结 backbone 一次 forward 的结果，训练成本极低。
:::

**参数量对比**：

| 组件 | 参数量 |
|------|--------|
| VLA backbone（冻结） | 7B |
| RL Actor head | ~50M |
| RL Critic head | ~50M |
| **总可训练参数** | **~100M（仅占 1.4%！）** |

### 2.3 动作输出：Anchoring 到 VLA 原始预测

BootRL 的最终动作不是 RL head 独立预测的，而是**锚定到 VLA 的原始动作预测上**：

$$
a_{\text{final}} = \underbrace{a_{\text{VLA}}}_{\text{VLA backbone 原始预测}} + \underbrace{\Delta a_{\text{RL}}}_{\text{RL head 的修正量}}
$$

**这个公式在做什么**：最终执行的动作不是 RL head 独立生成的，而是在 VLA 原本就会输出的动作上叠加一个小修正量——这样即使 RL head 什么都没学到，动作也不会比原始 VLA 差。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $a_{\text{VLA}}$ | **老师傅的原始方案** | 冻结的 VLA backbone 直接给出的动作预测，不依赖 RL head |
| $\Delta a_{\text{RL}}$ | **小施工队的修正建议** | RL head 学出来的一份修正量，叠加在原始方案上 |
| $a_{\text{final}}$ | **最终拍板执行的动作** | 两者相加得到的、真正发给机械臂执行的动作 |

**用人话读**："最终动作 = VLA 原本想做的动作 + RL head 学到的一点点修正。"

**为什么是这个形式**：见下方"为什么用 residual 形式"的三点说明——这是 anchoring 设计的核心，保证训练起点稳定、任务更容易学。
:::

其中：

$$
\Delta a_{\text{RL}} \sim \mathcal{N}(\mu_{\text{RL}}, \sigma_{\text{RL}}^2)
$$

**这个公式在做什么**：修正量本身不是一个固定值，而是从上一步 Actor 网络输出的高斯分布里采样出来的一个随机样本——这份随机性正是 PPO 训练所需要的探索。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mu_{\text{RL}},\sigma_{\text{RL}}$ | **分布的两个刻度盘** | 上一步 Actor 网络算出的均值和标准差，决定这个高斯分布的中心和宽度 |
| $\mathcal{N}(\cdot,\cdot)$ | **随机抽签器** | 标准高斯分布记号，实际训练/推理时从这个分布里采一个具体的修正量出来 |
| $\Delta a_{\text{RL}}$ | **本次实际用的修正量** | 采样得到的具体数值，会被代入上一个公式里加到 $a_{\text{VLA}}$ 上 |

**用人话读**："修正量不是网络直接吐出的一个死数字，而是围着网络给出的中心值、按给出的宽度随机抖一下抽出来的。"

**为什么是这个形式**：随机采样让 PPO 可以计算 $\log$ 概率、估计梯度方向（详见 2.4 节），也让策略保留探索能力，不会一开始就固化成确定性动作。
:::

**为什么用 residual 形式？**

1. **保持 VLA 的先验**：如果 RL head 输出 $\Delta a = 0$，就完全使用 VLA 的原始动作——保底不会比 VLA 差
2. **学习更容易**：RL head 只需要学"在 VLA 的基础上微调多少"，而不是从零学完整的动作
3. **初始化稳定**：$\mu_{\text{RL}}$ 初始化为 0 → 训练一开始就等于纯 VLA，不会崩溃

**代入数字的例子**：

假设机械臂当前在抓取物体前的位置调整阶段：
- VLA 原始预测：$a_{\text{VLA}} = [0.05, -0.02, -0.03, 0, 0, 0, 0]$（向右移 5cm，向下移 3cm）
- RL head 修正：$\Delta a_{\text{RL}} = [0.01, 0.005, -0.01, 0, 0, 0, 0]$（再多右移 1cm，多下移 1cm）
- 最终动作：$a_{\text{final}} = [0.06, -0.015, -0.04, 0, 0, 0, 0]$

RL head 学到了"VLA 总是差一点点，需要微调位置精度"。

### 2.4 训练目标

RL head 用 PPO 训练（详见 [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO)）：

$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}_t\left[\min\left(r_t \hat{A}_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

**这个公式在做什么**：这是标准 PPO 的 clip 目标（详见 [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO)），套用在 RL head 上——用新旧 RL head 的概率比去放大/缩小修正量的好坏分，但设一个安全上限防止一步更新太猛。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $r_t\hat A_t$ | **自由更新方向** | 概率比乘 advantage，这个修正量带来的收益有多大 |
| $\text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t$ | **限速版本** | 把概率比卡在 $[1-\epsilon,1+\epsilon]$ 区间内再算收益 |
| $\min(\cdot,\cdot)$ | **保守裁判** | 两个版本取较小的，防止 RL head 一步更新偏离太远 |
| $\mathbb{E}_t[\cdot]$ | **batch 平均** | 对采集到的一批时间步取平均，训练里就是 `.mean()` |

**用人话读**："和标准 PPO clip 目标完全一样，只是这里的策略是 RL head 输出的修正量分布，不是完整动作分布。"

**为什么是这个形式**：直接复用 PPO 的成熟机制，不需要为 RL head 重新设计约束方式，clip 的作用和标准 PPO 一致——防止修正量的概率一次跳变太大。
:::

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_t\left[(V_\phi(s_t) - V_t^{\text{target}})^2\right]
$$

**这个公式在做什么**：让 Critic 网络的估值尽量贴近实际观测到的回报目标——这是标准的价值函数回归损失。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $V_\phi(s_t)$ | **Critic 的猜测** | 当前 Critic 网络对状态 $s_t$ 的估值 |
| $V_t^{\text{target}}$ | **更靠谱的参考答案** | 用采样轨迹算出来的回报目标（如 GAE 的 $\hat R_t$ 或 5.2 节的 $G_t$） |
| $(\cdot)^2$ | **偏差评分** | 猜测和目标之间的平方误差 |
| $\mathbb{E}_t[\cdot]$ | **batch 平均** | 对一批时间步取平均 |

**用人话读**："让 Critic 打的分尽量接近轨迹实际拿到的回报，打得越准，将来给 Actor 的 advantage 估计就越可靠。"

**为什么是这个形式**：标准的回归损失（MSE），Critic 打分准确是后续算 advantage、指导 Actor 更新方向的基础。
:::

其中概率比的计算只涉及 RL head 的参数：

$$
r_t = \frac{\pi_{\text{RL}}(\Delta a_t | h_{\text{RL},t})}{\pi_{\text{RL,old}}(\Delta a_t | h_{\text{RL},t})}
$$

**这个公式在做什么**：算出"新 RL head 比旧 RL head 更喜欢这个修正量多少倍"——这个比值只涉及 RL head 自己的参数，完全不涉及冻结的 VLA backbone。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\pi_{\text{RL}}(\Delta a_t\|h_{\text{RL},t})$ | **新 RL head 的喜好** | 当前（更新后）RL head 给这个修正量的概率密度 |
| $\pi_{\text{RL,old}}(\Delta a_t\|h_{\text{RL},t})$ | **旧 RL head 的喜好** | 采集这批数据时用的（更新前）RL head 给的概率密度 |
| $r_t$ | **喜好倍数** | 两者比值，$>1$ 说明新 head 更喜欢这个修正量 |

**用人话读**："只看 RL head 自己新旧两个版本对同一个修正量的打分比值，VLA backbone 完全不参与这个计算。"

**为什么是这个形式**：这正是 BootRL 高效的原因——重要性采样比值只依赖 100M 的 RL head 参数，不需要对 7B 的 backbone 做任何梯度计算。
:::

$$
\pi_{\text{RL}}(\Delta a | h) = \mathcal{N}(\Delta a; \mu_{\text{RL}}(h), \sigma_{\text{RL}}(h)^2)
$$

**这个公式在做什么**：明确写出 RL head 的策略分布形式——给定隐藏状态 $h$，修正量服从一个由 Actor 网络参数化的高斯分布。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $h$ | **输入** | RL Token 位置的隐藏状态 |
| $\mu_{\text{RL}}(h),\sigma_{\text{RL}}(h)$ | **分布参数** | 2.2 节 Actor 网络的输出，随 $h$ 变化 |
| $\mathcal{N}(\Delta a;\cdot,\cdot)$ | **概率密度函数** | 在给定均值方差下，修正量 $\Delta a$ 取某个具体值的密度 |

**用人话读**："RL head 的策略就是一个高斯分布，中心和宽度都由 Actor 网络根据当前场景算出来。"

**为什么是这个形式**：高斯分布的 $\log$ 概率有解析公式，能直接支持 PPO 需要的重要性采样比值计算，不像扩散策略那样要处理不可解的积分。
:::

**log-probability 的精确计算**：

$$
\log \pi_{\text{RL}}(\Delta a | h) = -\frac{1}{2}\sum_{i=1}^{d}\left[\frac{(\Delta a_i - \mu_i)^2}{\sigma_i^2} + \log(2\pi\sigma_i^2)\right]
$$

**这个公式在做什么**：把高斯分布的密度函数展开成对数形式——这是计算 $r_t$（新旧策略概率比）时实际会用到的公式，逐维累加后就是整个动作修正量的对数概率。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $d=7$ | **动作维度数** | 修正量的维度，如 xyz + rotation + gripper |
| $(\Delta a_i-\mu_i)^2/\sigma_i^2$ | **归一化偏差** | 第 $i$ 维实际采样值离均值多远，用方差归一化 |
| $\log(2\pi\sigma_i^2)$ | **分布宽度的惩罚项** | 方差越大，这一项越大，压低整体的 log 概率 |
| $\sum_{i=1}^d$ | **逐维累加** | 各维度独立同分布假设下，联合对数概率就是各维对数概率之和 |

**用人话读**："把每一维修正量的'离中心有多远'和'分布本身有多宽'都算进去，加总起来就是这个动作修正量的对数概率。"

**为什么是这个形式**：这是多维独立高斯分布 log-likelihood 的标准公式，直接决定了 $r_t = \exp(\log\pi_{\text{RL}} - \log\pi_{\text{RL,old}})$ 怎么算，是 PPO 训练里唯一需要对 RL head 求梯度的地方。
:::

**代入数字**：假设第 1 维（x 方向），$\Delta a_1 = 0.01$，$\mu_1 = 0.008$，$\sigma_1 = 0.02$：

$$
\log \pi_1 = -\frac{1}{2}\left[\frac{(0.01-0.008)^2}{0.02^2} + \log(2\pi \times 0.02^2)\right] = -\frac{1}{2}\left[0.01 + (-3.07)\right] = 1.53
$$

**这个公式在做什么**：把一组具体数值代入上面的对数概率公式，验证单维情况下的计算过程。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $(0.01-0.008)^2/0.02^2=0.01$ | **归一化偏差的数值** | 采样值离均值只差 0.002，除以方差后得到 0.01 |
| $\log(2\pi\times0.02^2)=-3.07$ | **宽度惩罚的数值** | 标准差较小（0.02），这一项为负 |
| $1.53$ | **最终对数概率** | 两项相加取负一半得到的具体值 |

**用人话读**："把具体的采样值、均值、标准差代进公式，一步步算出这一维的对数概率是 1.53。"

**为什么是这个形式**：展示抽象公式在真实数值下如何计算，帮助确认对每一维单独计算、再求和得到整个 $\log\pi_{\text{RL}}(\Delta a|h)$ 的过程。
:::

### 2.5 Anchoring 正则化

为了防止 RL head 偏离 VLA 太远，加入 anchoring 损失：

$$
\mathcal{L}_{\text{anchor}} = \lambda \cdot \|\Delta a_{\text{RL}}\|^2
$$

**这个公式在做什么**：给修正量的大小加一个惩罚，修正量越大扣分越多——逼着 RL head 只做小幅微调，不去大幅覆盖 VLA 原本的决策。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\|\Delta a_{\text{RL}}\|^2$ | **修正幅度的体检分** | 修正量各维度平方求和，幅度越大这个值越大 |
| $\lambda$ | **约束力度旋钮** | 人工设定的权重，$\lambda$ 越大越限制修正量的大小 |
| $\mathcal{L}_{\text{anchor}}$ | **额外扣分项** | 加进总损失里，和 PPO 的 actor loss 一起优化 |

**用人话读**："修正量越大，额外扣的分就越多，这逼着 RL head 只敢做小幅调整。"

**为什么是这个形式**：平方惩罚（类似 L2 正则）对大幅修正的惩罚呈非线性增长，能有效抑制"RL head 完全无视 VLA、自己重新决策"这种退化情况——消融实验显示去掉这项后性能暴跌 13.5%（4.4 节）。
:::

**作用**：惩罚过大的修正量，确保 RL head 只做"微调"而不是"完全覆盖"VLA 的决策。

$\lambda = 0.1$ 时的效果：修正量通常限制在 VLA 原始动作幅度的 10-20% 以内。

---

## 三、为什么冻结 VLA + 小 head 能工作

### 3.1 VLA Hidden States 的信息量

论文通过 probing experiment 证明：VLA backbone 的 hidden states 已经包含了做好决策所需要的几乎所有信息。

**Probing 实验**：在冻结 VLA 的 hidden states 上训练一个 linear probe 预测"当前离目标还有多远"：
- R² = 0.92——hidden states 几乎完美编码了任务进度信息
- 这意味着 VLA 不是"不知道"怎么做，而是"最后一步解码"没做好

### 3.2 VLA 的瓶颈在哪

```mermaid
flowchart TB
    subgraph 理解层["VLA Backbone（足够好）"]
        A["视觉理解 ✓"] --> B["语言理解 ✓"]
        B --> C["运动规划 ✓"]
        C --> D["Hidden States ✓"]
    end
    subgraph 执行层["动作解码（瓶颈！）"]
        D --> E["动作 token 生成"]
        E --> F["256-bin 量化"]
        F --> G["最终动作"]
    end
    style E fill:#ff6b6b,color:#fff
    style F fill:#ff6b6b,color:#fff
```

VLA 的瓶颈主要在**动作解码层**：
1. **量化误差**：256 bin 的分辨率约为 0.8% 误差，在精密操作中会累积
2. **自回归误差累积**：7 维动作逐维预测，前面维度的误差影响后面
3. **SFT 学到的是"平均行为"**：多条示教的平均会模糊最优动作

BootRL 的 RL head 直接在连续空间输出修正量（不受量化限制），可以修补这些瓶颈。

### 3.3 泛化能力保持

**核心论点**：因为 VLA backbone 完全冻结，它的泛化能力**零损失**。

| 方法 | 目标任务提升 | 其他任务退化 | 净收益 |
|------|------------|------------|--------|
| VLA-RL (LoRA) | +15% | -8% | +7% |
| RIPT-VLA (Full FT) | +20% | -12% | +8% |
| **BootRL (冻结+head)** | **+25%** | **0%** | **+25%** |

**为什么 BootRL 不退化**：
- VLA 参数没有任何改变 → 在所有任务上的表现完全不变
- RL head 是额外加的模块 → 只在目标任务上激活
- 不需要目标任务时，可以直接移除 RL head，回到原始 VLA

---

## 四、实验结果

### 4.1 LIBERO Benchmark

| 方法 | 可训练参数 | LIBERO 平均 | 训练时间 | 泛化保持 |
|------|-----------|------------|---------|---------|
| SFT baseline | — | 76.5% | — | 100% |
| VLA-RL (LoRA) | 50M | 81.0% | 48h | 85% |
| RIPT-VLA | 7B | 93.6% | 24h | 78% |
| **BootRL** | **100M** | **91.8%** | **8h** | **100%** |

**关键发现**：
- BootRL 用 1/70 的训练参数达到了接近最强方法的性能
- 训练时间仅 8 小时——因为只需要反向传播 100M 参数
- **泛化能力完全保持**——这是其他方法做不到的

### 4.2 训练效率对比

| 指标 | BootRL | VLA-RL | RIPT-VLA |
|------|--------|--------|----------|
| 每 iteration 训练时间 | 3 分钟 | 15 分钟 | 8 分钟 |
| 达到 85% 成功率 | 15 iterations | 60 iterations | 25 iterations |
| 总 GPU 小时 | 8h | 48h | 24h |
| 显存需求 | 24GB（推理）+ 8GB（head） | 80GB+ | 48GB+ |

**BootRL 的训练极其轻量**——VLA backbone 只需要做 forward pass（推理），不需要存储梯度。反向传播只涉及 100M 的 RL head。

### 4.3 泛化实验

在 LIBERO 的一个子集上训练 RL，然后测试在**未见过的任务**上的表现：

| 方法 | 训练任务成功率 | 未见任务成功率 | 泛化保持率 |
|------|-------------|-------------|----------|
| VLA-RL (LoRA) | 90.2% | 72.1% | 80% |
| RIPT-VLA | 93.6% | 69.8% | 75% |
| **BootRL** | 91.8% | **76.5%** | **100%** |

BootRL 在未见任务上的表现等于 SFT baseline（76.5%）——因为 backbone 没变！其他方法都有不同程度的退化。

### 4.4 消融实验

| 配置 | LIBERO 平均成功率 |
|------|-----------------|
| **完整 BootRL** | **91.8%** |
| 去掉 anchoring（RL head 独立预测） | 78.3%（-13.5%） |
| 去掉 RL Token（用最后一个 token） | 87.2%（-4.6%） |
| Head 太小（10M 参数） | 84.5%（-7.3%） |
| Head 太大（500M 参数） | 90.1%（-1.7%） |
| 不冻结 backbone（全量微调） | 93.6%（+1.8%，但泛化退化） |
| Anchoring $\lambda = 0$（无正则） | 82.1%（-9.7%） |
| Anchoring $\lambda = 1.0$（过强正则） | 85.4%（-6.4%） |

**关键观察**：
- **Anchoring 是最关键的设计**：去掉后性能暴跌 13.5%——RL head 单独从零学习太难了
- **RL Token 有帮助**：专门位置比复用最后 token 好 4.6%
- **Head 大小有最优点**：~100M 是性价比最高的选择

---

## 五、技术细节

### 5.1 RL Token 的实现

RL Token 是一个可学习的特殊 embedding，插入到 VLA 输入序列中：

```
输入序列: [IMG_1] [IMG_2] ... [IMG_N] [INST_1] ... [INST_M] [RL_TOKEN] [ACT_1] ... [ACT_7]
```

VLA backbone 在 forward pass 中正常处理整个序列（attention 机制让 RL Token 能"看到"前面所有图像和指令 token），然后在 RL Token 位置提取 hidden state：

$$
h_{\text{RL}} = \text{TransformerOutput}[\text{pos}=N+M+1] \in \mathbb{R}^{4096}
$$

**这个公式在做什么**：明确指出 RL head 的输入具体是从哪里取出来的——整个输入序列经过 Transformer 处理后，只取 RL Token 那一个位置（第 $N+M+1$ 位）的输出向量。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $N$ | **图像 token 数量** | 输入序列里 IMG token 的个数 |
| $M$ | **指令 token 数量** | 输入序列里指令 token 的个数 |
| $\text{pos}=N+M+1$ | **RL Token 的位置** | 图像和指令 token 之后紧跟的那个特殊位置 |
| $\text{TransformerOutput}[\cdot]$ | **取值操作** | 从 Transformer 完整输出序列里，只挑出这一个位置对应的向量 |
| $h_{\text{RL}}\in\mathbb{R}^{4096}$ | **提取出的摘要向量** | 就是前面反复用到的 RL head 输入 |

**用人话读**："输入序列过完整个 Transformer 之后，只把 RL Token 所在那个位置的输出向量拿出来，作为 RL head 的输入。"

**为什么是这个形式**：因为 attention 机制让 RL Token 位置能"看到"前面所有图像和指令 token（2.1 节），所以这一个位置的输出天然浓缩了对整个场景的理解，不需要额外设计聚合方式。
:::

RL Token 的 embedding 本身是可训练的（唯一一个在 backbone 中可训练的参数），但只有 4096 个参数——可以忽略不计。

### 5.2 训练稳定性技巧

**技巧 1：Value function warmup**

Critic head 在正式 RL 训练前先用 Monte Carlo return 做 5 epoch warmup：

$$
\mathcal{L}_{\text{warmup}} = \mathbb{E}\left[(V_\phi(s_t) - G_t)^2\right], \quad G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

**这个公式在做什么**：在正式开始 PPO 训练之前，先单独用蒙特卡洛（Monte Carlo, MC——直接用轨迹跑完后的真实回报）算出的回报 $G_t$，把 Critic 网络预热到一个靠谱的起点，避免训练初期 Actor 拿到一个乱打分的 Critic 反馈。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $G_t=\sum_{k=0}^{T-t}\gamma^k r_{t+k}$ | **真实回报账本** | 从时刻 $t$ 到轨迹结束，把实际拿到的奖励按折扣加总，是最直接的"事后真值" |
| $V_\phi(s_t)$ | **Critic 的预热猜测** | 训练初期 Critic 对状态 $s_t$ 的估值，此时还没见过真实回报的分布 |
| $(\cdot)^2$ | **偏差评分** | Critic 猜测和真实回报的平方误差 |
| $\mathbb{E}[\cdot]$ | **样本平均** | 对采集到的 warmup 数据取平均 |

**用人话读**："在正式 PPO 训练开始前，先让 Critic 老老实实用轨迹跑完后算出来的真实回报练习打分，练 5 个 epoch 再上场。"

**为什么是这个形式**：如果 Critic 一开始就乱打分，Actor 用错误的 advantage 估计更新会走偏方向，warmup 用最直接无偏但方差较大的 MC 回报（而非 GAE）先把 Critic 校准到大致正确的量级，之后正式训练再切换到更高效的估计方式。
:::

**技巧 2：Action clipping**

RL head 的修正量被 clip 到合理范围：

$$
\Delta a_{\text{RL}} = \text{clip}(\Delta a_{\text{raw}}, -\delta_{\max}, \delta_{\max})
$$

**这个公式在做什么**：把 Actor 网络采样出的原始修正量强行卡在一个安全范围内，防止某次采样抽出一个离谱的大修正值，破坏 VLA 原本合理的动作。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\Delta a_{\text{raw}}$ | **采样出的原始修正量** | 直接从 $\mathcal{N}(\mu_{\text{RL}},\sigma_{\text{RL}}^2)$ 采出来、还没做任何限制的值 |
| $\delta_{\max}$ | **安全边界** | 允许的最大修正幅度，$\delta_{\max}=0.05$ 表示每步最多修正 5cm/5度 |
| $\text{clip}(\cdot,-\delta_{\max},\delta_{\max})$ | **硬限位器** | 把值截断在 $[-\delta_{\max},\delta_{\max}]$ 区间内，超出就拉回边界 |

**用人话读**："不管采样抽出多大的修正量，最终真正执行时都不会超过 5cm/5 度这个硬上限。"

**为什么是这个形式**：这是和 2.5 节 anchoring 正则化（软惩罚）互补的硬约束——正则化让网络"倾向于"输出小修正量，clip 则保证即使某次采样出现极端值，也有一个绝对不会突破的物理安全边界。
:::

$\delta_{\max} = 0.05$（限制每步最多修正 5cm/5度）。

**技巧 3：渐进式 $\sigma$ 衰减**

训练初期允许大探索（$\sigma$ 初始化较大），后期逐渐减小：

$$
\sigma_{\text{init}} = 0.05, \quad \sigma_{\min} = 0.005
$$

**这个公式在做什么**：设定探索噪声的起始值和最终下限——训练刚开始时标准差大一些，鼓励 RL head 多探索不同的修正量，随着训练进行逐渐收窄到一个较小但不为零的下限。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\sigma_{\text{init}}=0.05$ | **探索起点** | 训练刚开始时 Actor 输出标准差的初始值，允许较大范围的随机探索 |
| $\sigma_{\min}=0.005$ | **探索下限** | 标准差衰减的最终下限，保留一点点随机性，不会完全变成确定性策略 |

**用人话读**："刚开始训练时修正量的随机波动范围比较大，方便探索；训练越往后波动范围越小，但始终保留一点点随机性。"

**为什么是这个形式**：训练早期需要大方差去探索哪种修正方向能提高奖励；训练后期策略已经找到较好的修正方向，缩小方差能让动作更稳定、减少不必要的抖动，但不衰减到 0 是为了避免策略过早变成完全确定性、丧失继续改进的能力。
:::

### 5.3 和 SAC 的对比

论文也尝试了用 [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) 替代 PPO 训练 RL head：

| RL 算法 | LIBERO 成功率 | 训练稳定性 | Sample Efficiency |
|---------|-------------|-----------|-------------------|
| PPO | 91.8% | 高 | 中等 |
| SAC | 89.5% | 中（偶尔崩） | 高 |
| TD3 | 88.2% | 中 | 高 |

PPO 因为是 on-policy 的，和 VLA 的 rollout 流程更自然匹配。SAC 虽然 sample-efficient 但需要 replay buffer，在 VLA 的多模态输入下存储成本高。

---

## 六、和其他方法的深度对比

### 6.1 全方位对比

| 维度 | BootRL | VLA-RL | RIPT-VLA | SRPO |
|------|--------|--------|----------|------|
| VLA backbone | 冻结 | LoRA 微调 | 全量/LoRA | 全量/LoRA |
| 可训练参数 | 100M | 50M | 7B | 50M |
| 泛化保持 | 100% | ~85% | ~75% | ~80% |
| 最终性能 | 91.8% | 81.0% | 93.6% | 99.2% |
| 训练时间 | 8h | 48h | 24h | 60h |
| 显存 | 32GB | 80GB+ | 48GB+ | 80GB+ |
| 工程复杂度 | 低 | 高 | 中 | 高 |

### 6.2 BootRL 的独特优势

1. **即插即用**：RL head 可以随时加上/移除，不影响原 VLA
2. **多任务适配**：可以为不同任务训练不同的 RL head，共享同一个 VLA backbone
3. **低资源友好**：单卡 32GB 就能训练
4. **快速迭代**：8 小时就能完成一次完整训练

### 6.3 BootRL 的劣势

1. **性能天花板**：冻结 backbone 意味着无法学到需要深层表示变化的行为
2. **依赖 VLA 质量**：如果 VLA backbone 本身表示差（如在完全陌生的域），RL head 也无能为力
3. **Residual 限制**：只能做"微调"级别的修正，无法发现全新的行为模式（如 pushcut）

---

## 七、局限性与讨论

### 7.1 什么时候 BootRL 不够用

当任务需要**根本性的策略改变**（而非微调精度）时，BootRL 可能不够：
- 需要发现全新操作策略（如 pushcut）→ 用 SimpleVLA-RL
- VLA 在目标域完全不工作（成功率 < 10%）→ 用全量 RL 微调
- 需要极致性能（99%+）→ 用 SRPO

### 7.2 和 Residual RL 的关系

BootRL 的思路和 Residual RL 类似（参见 [PLD](./015_PLD_Residual_RL自改进VLA)），但有关键区别：
- **BootRL**：在 VLA latent space 做 residual（修正的是 hidden state 到动作的映射）
- **Residual RL**：在动作空间做 residual（修正的是最终动作值）
- **BootRL**：用 VLA 的 RL Token hidden state 作为输入
- **Residual RL**：用原始观测作为输入

### 7.3 可扩展性

BootRL 天然支持多任务学习——不同任务使用不同的 RL head，但共享同一个冻结 VLA backbone。这在部署时非常高效：
- 存储：一个 7B 的 backbone + N 个 100M 的 head
- 推理：只需一次 backbone forward + 对应 head forward
- 切换任务：只需换 head，不需要重新加载 backbone

---

## 八、个人评价

### 8.1 核心贡献

BootRL 提出了一种极其务实的 VLA RL 方案。在学术界追求极致性能的背景下，BootRL 关注的是**实用性**：低成本、快速、保持泛化。这对实际部署非常有价值。

### 8.2 实践建议

- 如果你需要**快速**验证 RL 能否帮助你的 VLA：先试 BootRL（8 小时就有结论）
- 如果 BootRL 提升不够：再考虑全模型 RL（如 RIPT-VLA、SRPO）
- 如果你需要**多任务部署**：BootRL 的 per-task head 是最优选择

### 8.3 学术意义

BootRL 证明了一个重要的结论：**VLA 的预训练表示已经足够好——RL 微调的主要价值在于修正最后的动作解码，而非改变深层理解**。这为 VLA 的设计提供了指导——投入更多资源在预训练上（提升表示质量），RL 只需要轻量化即可。

---

## 延伸阅读

- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) ← BootRL 使用的 RL 算法
- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) ← 备选 RL 算法
- [VLA-RL 精读](./006_VLA_RL_PPO直接训练自回归VLA) ← 对比：全模型 RL 训练
- [PLD 精读](./015_PLD_Residual_RL自改进VLA) ← 对比：动作空间的 Residual RL
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) ← 全景图
