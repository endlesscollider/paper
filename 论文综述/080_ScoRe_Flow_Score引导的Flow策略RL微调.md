---
title: ScoRe-Flow：Score 引导的 Flow 策略 RL 微调
order: 280
tags: [强化学习, Flow Matching, Score Function, PPO, On-Policy]
category: 精读
star: 4
---

# ScoRe-Flow：Score 引导的 Flow 策略 RL 微调 深度精读

> **论文标题**: ScoRe-Flow: Complete Distributional Control via Score-Based Reinforcement Learning for Flow Matching  
> **作者**: Xiaotian Qiu, Lukai Chen, Jinhao Li, Qi Sun, Cheng Zhuo, Guohao Dai  
> **发表**: arXiv:2604.10962, ICML 2026  

**知识链接**：
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — Flow Matching 的基本原理（ODE 生成、速度场学习）
- [Score Function（密度梯度）与 Score Matching](/前置知识/001t_前置知识_Score_Function密度梯度与Score_Matching) — ⚠️ **必读**：Score function $\nabla_x \log p(x)$ 的定义、直觉和在生成模型中的角色
- [ReinFlow：Flow 策略的噪声注入 RL 微调](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) — ⚠️ **必读**：ScoRe-Flow 的 baseline，理解"只加噪声不加引导"的局限性
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 算法
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — Flow/Diffusion 的 log-prob 困难
- [SAC-Flow：用 SAC 直接训练 Flow 策略](./079_SAC_Flow_用SAC直接训练Flow策略) — Off-policy 路线对比
- [FlowRL：Flow VLA 的在线 RL 微调](./018_FlowRL_Flow_VLA的在线RL微调) — 大规模 VLA 的 Flow RL

---

## 相关阅读（开始前必须了解的内容）

本文在三个前置知识之上构建。如果你跳过它们直接读本文，会在第一个公式处卡住：

| 前置知识 | 你需要从中了解什么 |
|----------|------------------|
| [Flow Matching](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) | 速度场 $v_\theta$ 是什么、ODE 推理过程、为什么比 DDPM 快 |
| [Score Function（密度梯度）](/前置知识/001t_前置知识_Score_Function密度梯度与Score_Matching) | $s(x) = \nabla_x \log p(x)$ 的定义和含义——"指向概率高的方向" |
| [ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) | 如何给 Flow 加噪声来获得 log-prob、这种方法的局限性（只能控制 variance 不能控制 mean） |

---

## 贯穿全文的例子

> **场景**：一个 7-DOF 机械臂做桌面物体操作（pick-and-place）。动作 $a \in \mathbb{R}^7$ 是关节位移。策略是一个 Flow Matching 模型，用 4 步 ODE 从高斯噪声生成动作。
>
> - **预训练**（Flow Matching）：用示范数据训练，成功率 78%
> - **目标**：用 RL（PPO）把成功率提升到 95%+
> - **baseline（ReinFlow）**：给 ODE 加噪声后用 PPO → 成功率提升到 85%（但收敛慢）
> - **本文方法（ScoRe-Flow）**：在 ReinFlow 基础上加 score drift 引导 → 成功率 95%+，收敛快 2.4 倍
>
> **核心改进的直觉**：ReinFlow 只是在正确路线附近"随机抖动"来探索，而 ScoRe-Flow 不仅抖动，还能主动调整 flow 的行进方向——让粒子朝着"好动作聚集的方向"偏移。

---

## 一、问题定位：ReinFlow 的局限

### 1.1 回顾 ReinFlow 的做法

[ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) 的核心是给确定性 ODE 加噪声：

$$
a_{k+1} = a_k + v_\theta(a_k, t_k, s) \cdot \Delta t + \sigma_\phi(t_k) \cdot \epsilon_k
$$

**这个公式在做什么**：在确定性 ODE 的每一步更新上叠加一个高斯噪声项，把原本"走哪都固定"的 flow 变成有随机性、可以探索的策略。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $a_k$ | **当前位置** | flow 离散化后第 $k$ 步的粒子位置（即中间动作） |
| $v_\theta(a_k, t_k, s) \cdot \Delta t$ | **原定路线** | 预训练速度场给出的确定性前进方向，乘以步长后就是"没有噪声时该走多远" |
| $\sigma_\phi(t_k) \cdot \epsilon_k$ | **随机抖动** | $\epsilon_k \sim \mathcal{N}(0, I)$，乘上可学习的噪声强度 $\sigma_\phi(t_k)$，让每一步都带一点随机偏移 |
| $a_{k+1}$ | **下一步位置** | 原定路线加上随机抖动后落到的新位置 |

**用人话读**："每一步先按照预训练模型指的方向走一段，再随机抖一下，抖动幅度由一个可学习的网络控制。"

**为什么是这个形式**：只加各向同性噪声、不改变 drift 方向，是最小改动就能让确定性 ODE 变成可探索、log-prob 可精确计算的随机策略——但这也是它的局限：drift 方向完全由预训练网络决定，噪声再怎么调都无法修正一个"跑偏"的大方向。
:::

这解决了两个问题：(1) 让策略有了随机性用于探索；(2) 让 log-prob 可以精确计算（高斯累加）。

**但局限在于**：噪声 $\sigma_\phi \cdot \epsilon$ 是**各向同性的**——每个方向等概率地随机抖动，不改变 drift $v_\theta$ 的方向。

### 1.2 类比理解局限性

想象你在一个大型停车场找自己的车：

- **ReinFlow 的策略**：你记得车大概在"东侧某处"（drift 方向），所以你往东走，但每一步都随机偏左或偏右一点（噪声）。如果你的大方向记对了，最终能找到车——但如果大方向偏了（比如车其实在东南角），光靠随机左右抖动很难修正。

- **ScoRe-Flow 的策略**：你不仅随机偏移，还有一个"信号源"（score function）告诉你"车的密集区域在偏南方向"——这个额外的方向信号让你每步都能稍微修正大方向，更快找到车。

**数学上的对应**：

| 停车场类比 | 数学对应 |
|-----------|---------|
| "往东走"（大方向） | Drift $v_\theta$ |
| "每步随机偏移" | 噪声 $\sigma \cdot \epsilon$ |
| "信号源告诉你车在东南" | Score function $s_t(a)$（[定义见前置知识](/前置知识/001t_前置知识_Score_Function密度梯度与Score_Matching)） |
| "修正大方向" | Score drift $\alpha \cdot s_t$ |

### 1.3 形式化问题

用连续时间 SDE 的语言，ReinFlow 控制的是策略分布的**方差**（variance）——通过 $\sigma_\phi$ 调节探索幅度。但它无法控制**均值**（mean）——drift 方向完全由预训练的 $v_\theta$ 决定。

**ScoRe-Flow 的核心目标**：同时控制 mean（通过 score drift）和 variance（通过噪声），实现对策略分布的**完整分布控制**（Complete Distributional Control）。

---

## 二、Score Function 在 Flow 中的闭式表达

### 2.1 什么是这里的 Score Function

本文的 score function 是指**时刻 $t$ 的中间分布 $\rho_t(a)$ 的密度梯度**：

$$
s_t(a) = \nabla_a \log \rho_t(a)
$$

**这个公式在做什么**：定义"score"为中间分布 $\rho_t$ 的对数密度对位置 $a$ 的梯度——它指向"概率密度上升最快"的方向，也就是同伴粒子聚集得更密的地方。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\rho_t(a)$ | **第 $t$ 步的人口密度地图** | flow 在时刻 $t$ 时，粒子在空间中的分布密度 |
| $\log \rho_t(a)$ | **对数密度地形图** | 取对数是为了让梯度计算更稳定，且和后续 score matching 理论一致 |
| $\nabla_a(\cdot)$ | **指南针** | 对位置 $a$ 求梯度，指向密度上升最快的方向 |
| $s_t(a)$ | **方向引导信号** | 综合起来就是"站在 $a$ 这个位置，往哪走能最快挤进人群密集区" |

**用人话读**："在 flow 第 $t$ 步，score 就是告诉粒子 $a$'同伴们主要聚在哪个方向'的指南针。"

**为什么是这个形式**：这是概率论里 score function 的标准定义（详见 [Score Function 前置知识](/前置知识/001t_前置知识_Score_Function密度梯度与Score_Matching)），选择"对数密度的梯度"而不是"密度本身的梯度"，是因为对数梯度不受密度整体缩放的影响，且和已有的 score matching 理论直接兼容。
:::

**关键区分**：这里的 score function 和 RL 中常见的"策略梯度 score function"（$\nabla_\theta \log \pi_\theta$）完全不同——前者是对**数据** $a$ 求梯度，后者是对**参数** $\theta$ 求梯度。详细辨析见 [Score Function 前置知识的辨析表](/前置知识/001t_前置知识_Score_Function密度梯度与Score_Matching#⚠️-两种-score-function-辨析-极其重要)。

### 2.2 关键发现：Score 有闭式解

一般情况下，计算 $\nabla_a \log \rho_t(a)$ 需要知道 $\rho_t$ 的解析形式或额外训练一个 score 网络。但本文证明了——**对于线性 Flow Matching 路径，score 可以直接从速度场 $v_\theta$ 计算出来**：

$$
s_t(a) = \frac{t \cdot v_\theta(t, a, s) - a}{1 - t}
$$

**这个公式在做什么**：证明 score 不需要额外训练一个网络来估计，而是可以直接从预训练好的速度场 $v_\theta$ 做一次代数变换免费算出来。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $v_\theta(t, a, s)$ | **速度预测** | "粒子应该往哪走"，速度场网络的原始输出 |
| $t \cdot v_\theta(t, a, s) - a$ | **偏差向量** | 时间加权后的目标位置估计减去当前位置，代表"我离主流还差多远" |
| $1 - t$ | **剩余时间** | 越接近终点这个值越小，让 score 的放大倍数越大 |
| $s_t(a)$ | **免费的方向引导信号** | 偏差向量除以剩余时间，就是这个位置处的 score |

**用人话读**："score 就是把速度场按时间加权后减去当前位置，再除以剩余时间——不需要新网络，纯代数变换就能从已有的速度场里'榨'出方向引导信号。"

**为什么是这个形式**：这个闭式解来自线性 Flow Matching 路径下条件分布的性质（推导见下方"推导直觉"），关键价值在于零额外参数、零额外前向传播——如果 score 需要单独训练网络，ScoRe-Flow 就要付出双网络、双训练的代价。

**数值代入**（$d=2$，$t=0.6$，机械臂处于中间状态 $a=[0.3,-0.1]$）：

- 速度场预测：$v_\theta = [1.5, 2.0]$（网络认为这个粒子应该往右上方运动）
- 分子：$t \cdot v_\theta - a = 0.6 \times [1.5, 2.0] - [0.3, -0.1] = [0.9, 1.2] - [0.3, -0.1] = [0.6, 1.3]$
- 分母：$1 - t = 0.4$
- Score：$s_{0.6}(a) = [0.6, 1.3] / 0.4 = [1.5, 3.25]$

含义：score 指向右上方 $[1.5, 3.25]$，说明"大部分粒子同伴"在当前位置的右上方。沿 score 方向给粒子一个推力，它就会被"拉"向同伴们聚集的区域——即高概率区域。

**注意 $t \to 1$ 时的发散**：当 $t$ 接近 1 时，$1-t \to 0$，score 趋于无穷大。这是因为 $t=1$ 时分布退化为 delta 函数（粒子们都到达了各自的终点），"山"变成了无限陡的"针尖"。这个发散问题在下面第三节通过 $(1-t)$ 衰减因子解决。
:::

### 2.3 推导直觉（为什么 score 有这个形式）

线性 Flow Matching 的路径是：$a_t = (1-t) \cdot a_0 + t \cdot a_1$，其中 $a_0 \sim \mathcal{N}(0, I)$，$a_1$ 是数据。

在时刻 $t$，给定位置 $a_t$，条件速度场就是 $u_t = a_1 - a_0$。而边际速度场 $v_\theta$ 近似于对所有可能的 $(a_0, a_1)$ 配对做加权平均。

从 $a_t = (1-t)a_0 + t \cdot a_1$ 可以反解 $a_0 = (a_t - t \cdot a_1) / (1-t)$。由于 $a_0 \sim \mathcal{N}(0, I)$，$a_t$ 给定 $a_1$ 的条件分布是：

$$
a_t | a_1 \sim \mathcal{N}(t \cdot a_1, (1-t)^2 I)
$$

**这个公式在做什么**：写出"给定最终数据 $a_1$"时，flow 路径上时刻 $t$ 的中间点 $a_t$ 服从的条件分布——这是推导 score 闭式解的关键中间结果。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $a_t \mid a_1$ | **条件位置** | 已知最终会落在数据点 $a_1$ 的前提下，第 $t$ 步中间位置的分布 |
| $t \cdot a_1$ | **条件均值** | 随时间线性地从 0 移向 $a_1$，$t$ 越大越靠近数据点 |
| $(1-t)^2 I$ | **条件方差** | 随时间线性收缩，$t\to1$ 时方差趋于 0，说明终点处所有路径都精确收拢到 $a_1$ |

**用人话读**："如果我们已经知道粒子最终会到达 $a_1$，那么它在中间时刻 $t$ 的位置就是一个以 $t\cdot a_1$ 为中心、方差随时间收缩的高斯分布。"

**为什么是这个形式**：这是线性插值路径 $a_t=(1-t)a_0+t\cdot a_1$（$a_0\sim\mathcal N(0,I)$）在固定 $a_1$ 时的直接结果——反解 $a_0=(a_t-t\cdot a_1)/(1-t)$ 后代入高斯分布的线性变换公式即可得到，这一步是后续推出 score 闭式解 $s_t(a)=(t\cdot v_\theta - a)/(1-t)$ 的数学基础。
:::

其条件 score 是 $\nabla_{a_t} \log p(a_t | a_1) = -(a_t - t \cdot a_1) / (1-t)^2$。做适当的边际化和速度场替换后，得到上面的闭式公式。

---

## 三、方法：ScoRe-Flow 的完整 SDE

### 3.1 在 ReinFlow 基础上加一项

回忆 [ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) 的 SDE 只有两项（drift + noise）。ScoRe-Flow 加入第三项——score drift 修正：

$$
\mathrm{d}a_t = \Big[\underbrace{v_\theta(t, a_t, s)}_{\text{① 预训练速度场}} + \underbrace{\alpha_\psi^{\text{scaled}}(t) \cdot s_t(a_t)}_{\text{② Score drift 修正（新增）}}\Big]\mathrm{d}t + \underbrace{\sigma_\phi(t, a_t, s)}_{\text{③ 学习的噪声}}\,\mathrm{d}W_t
$$

**这个公式在做什么**：ScoRe-Flow 的核心 SDE——在 ReinFlow 的"原路线 + 随机抖动"基础上，多加一项能主动把粒子推向高概率区域的 score drift 修正。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $v_\theta(t, a_t, s)$（①） | **原定路线** | 预训练速度场给出的基础前进方向，和 ReinFlow 完全一样 |
| $\alpha_\psi^{\text{scaled}}(t) \cdot s_t(a_t)$（②，新增） | **方向修正推手** | 学习出的强度 $\alpha_\psi^{\text{scaled}}$ 乘上 score，把粒子主动拉向高概率区域，控制分布的 mean |
| $\sigma_\phi(t, a_t, s)\,\mathrm{d}W_t$（③） | **随机探索** | 可学习噪声强度乘上维纳过程增量，负责探索，控制分布的 variance |
| $\mathrm{d}a_t$ | **本步位移** | ①+②的确定性方向乘以 $\mathrm{d}t$，加上③的随机扰动，就是这一微小时间步内粒子的总位移 |

**用人话读**："粒子的每一步移动 = 原来的既定方向，加上一个把它推向高概率区域的额外修正，再加上一份随机抖动用于探索。"

**为什么是这个形式**：ReinFlow 只有 ①+③，只能调探索幅度（variance）却调不了方向（mean）；加入②后，drift 方向本身可以被学习调整，实现对策略分布 mean 和 variance 的完整控制，而且②③分别由独立的网络学习，互不绑定。
:::

**关键设计洞察**：② 和 ③ 是**解耦的**——可以独立控制"往哪偏"和"探索多少"。之前的方法（如 Score-SDE 消融）把两者绑定为 $\sigma = \sqrt{2\alpha}$，灵活性不够。

**和 ReinFlow 对比**：

| | ReinFlow | ScoRe-Flow |
|---|---|---|
| Drift | $v_\theta$（固定） | $v_\theta + \alpha \cdot s_t$（可修正方向） |
| Noise | $\sigma_\phi$（学习） | $\sigma_\phi$（学习） |
| 能控制 | 只有 variance | mean + variance |
| 类比 | 在固定铁轨上抖动 | 铁轨方向也能调 |

### 3.2 $(1-t)$ Time-decay 稳定性约束

上一节提到 score 在 $t \to 1$ 时会发散：$|s_t| = O((1-t)^{-1})$。如果直接把发散的 score 乘上去，drift 修正会无穷大，训练崩溃。

解决方案——给 $\alpha_\psi$ 乘一个 $(1-t)$ 因子：

$$
\alpha_\psi^{\text{scaled}}(t) = (1-t) \cdot \alpha_\psi(t)
$$

**这个公式在做什么**：给学习出来的引导强度 $\alpha_\psi(t)$ 再乘上一个 $(1-t)$ 衰减因子，抵消 score 在 $t\to1$ 时的发散，防止训练崩溃。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\alpha_\psi(t)$ | **学习出的基础强度** | 一个小 MLP 的输出，代表"未经修正的引导强度" |
| $(1-t)$ | **镇静剂** | 越接近终点（$t\to1$）这个因子越小，正好抵消 score 的 $(1-t)^{-1}$ 发散 |
| $\alpha_\psi^{\text{scaled}}(t)$ | **实际生效的引导强度** | 两者相乘后代入 SDE 的 drift 修正项，全程保持有界 |

**用人话读**："学出来的引导强度要先乘上一个随时间递减的镇静剂，才能真正用到 SDE 里——否则接近终点时 score 发散会把训练炸掉。"

**为什么是这个形式**：score 的量级是 $O((1-t)^{-1})$，只要用 $(1-t)$ 这个恰好互为倒数关系的因子相乘，乘积 $\alpha_\psi^{\text{scaled}}\cdot s_t$ 就能在整个 $[0,1]$ 区间内保持有界，这是消融实验证实的必要约束，不加就会训练崩溃。
:::

**数值验证**：

| $t$ | $|s_t| \sim \frac{1}{1-t}$ | $(1-t)$ 因子 | 乘积 $\sim 1$ |
|-----|---------------------------|-------------|--------------|
| 0.0 | 1.0 | 1.0 | 1.0 |
| 0.5 | 2.0 | 0.5 | 1.0 |
| 0.9 | 10.0 | 0.1 | 1.0 |
| 0.99 | 100.0 | 0.01 | 1.0 |

→ 无论 $t$ 在哪，实际的 drift 修正幅度始终受控（大约恒定量级）。

**消融实验确认**：如果不加 $(1-t)$ 衰减（即固定 $\alpha = 1$），训练在 $t$ 接近 1 时直接崩溃（loss 爆炸）。这不是可选的——是方法能 work 的必要条件。

---

## 四、两个可学习组件的设计

### 4.1 组件概览

ScoRe-Flow 在预训练的 flow 网络 $v_\theta$ 之外，引入两个轻量级的可学习组件：

| 组件 | 名称 | 输入 | 输出 | 架构 | 参数量 |
|------|------|------|------|------|--------|
| Score scheduler $\alpha_\psi$ | "方向引导强度调节器" | 标量时间 $t$ | 标量 $\alpha > 0$ | 2 层 MLP + Softplus | ~几百 |
| Variance predictor $\sigma_\phi$ | "探索幅度控制器" | $(a_t, t, s)$ | 标量 $\sigma \in [\sigma_{\min}, \sigma_{\max}]$ | MLP + bounded Tanh | ~几千 |

**为什么这样设计**：

- **$\alpha_\psi$ 只依赖时间 $t$**：作者发现 score 引导的最优强度主要和"当前处于 flow 的哪个阶段"有关——开头需要强引导（粒子离目标远），结尾需要弱引导（粒子已接近目标）。状态和位置的影响很小，所以简单的 $t \mapsto \alpha$ 映射就够了。
- **$\sigma_\phi$ 依赖 $(a_t, t, s)$**：噪声强度需要更细粒度的控制——在某些状态下需要更多探索（新情况），在某些位置需要更少探索（已接近好动作区域）。

### 4.2 Score Scheduler $\alpha_\psi$ 的细节

$\alpha_\psi$ 的设计目标是让网络自动学会"在 flow 的不同阶段，score 引导应该多强"。

**实现**：
```python
class ScoreScheduler(nn.Module):
    """学习 score drift 的强度随时间变化的模式"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),   # 输入：标量 t
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()               # 保证输出 > 0
        )
    
    def forward(self, t):
        alpha = self.net(t.unsqueeze(-1))  # 学习的基础强度
        return (1 - t) * alpha             # 乘以 (1-t) 衰减因子
```

上面这段代码实现了两个功能：(1) 一个小 MLP 学习 $\alpha_\psi(t)$，输出始终为正（Softplus 保证）；(2) 乘以 $(1-t)$ 得到最终的 $\alpha_\psi^{\text{scaled}}(t)$，防止 score 发散。

### 4.3 Variance Predictor $\sigma_\phi$ 的细节

**实现**：
```python
class VariancePredictor(nn.Module):
    """学习每步的噪声强度（探索幅度）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128,
                 sigma_min=0.01, sigma_max=0.5):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.net = nn.Sequential(
            nn.Linear(action_dim + obs_dim + 1, hidden_dim),  # 输入: a_t, s, t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 输出在 [-1, 1]
        )
    
    def forward(self, a_t, t, obs):
        x = torch.cat([a_t, obs, t.unsqueeze(-1)], dim=-1)
        raw = self.net(x)  # [-1, 1]
        # 映射到 [sigma_min, sigma_max]
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (raw + 1) / 2
        return sigma
```

**设计要点**：
- **输出有界** $[\sigma_{\min}, \sigma_{\max}]$：Tanh 的输出映射到固定范围。$\sigma_{\min}=0.01$ 防止噪声为零（必须有探索）；$\sigma_{\max}=0.5$ 防止噪声太大（轨迹崩溃）。
- **输入包含 $(a_t, t, s)$**：噪声强度可以根据当前位置、时间和观测自适应调节。比如在安全关键状态下可以自动降低探索。

---

## 五、Log-Probability 计算与 PPO 训练

### 5.1 离散化后每步是高斯转移

和 [ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) 的处理方式相同，SDE 离散化后每步转移是一个高斯分布：

$$
p(a_{k+1} | a_k, s) = \mathcal{N}\Big(a_{k+1};\; a_k + \big[v_\theta + \alpha_\psi^{\text{scaled}} \cdot s_k\big] \Delta t,\; \sigma_\phi^2 \Delta t \cdot I\Big)
$$

**这个公式在做什么**：说明 SDE 离散化后每一步转移仍然是一个高斯分布，只是均值中心从 ReinFlow 的 $v_\theta$ 变成了加了 score drift 修正后的方向——这保证了 log-prob 依然可以用解析公式直接算出来，供 PPO 使用。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $a_k + [v_\theta + \alpha_\psi^{\text{scaled}} \cdot s_k]\Delta t$ | **新的高斯中心** | 如果没有随机噪声，下一步粒子理应到达的位置，比 ReinFlow 多了 score drift 的偏移 |
| $\sigma_\phi^2 \Delta t \cdot I$ | **围绕中心的扩散范围** | 各维度独立、方差相同的高斯协方差，决定了随机抖动的幅度 |
| $\mathcal{N}(a_{k+1}; \mu, \Sigma)$ | **转移概率密度** | 下一步位置 $a_{k+1}$ 落在均值 $\mu$、协方差 $\Sigma$ 的高斯分布上的概率密度 |

**用人话读**："每一步的下一个位置仍然服从高斯分布，只是这个高斯分布的中心点比 ReinFlow 多偏移了一段 score drift 修正。"

**为什么是这个形式**：PPO 训练需要计算 log-prob 并求梯度，高斯分布的 log-prob 有闭式解，只要保证每步转移仍是高斯（哪怕均值变了），整套 PPO 流程就可以直接复用，不需要任何额外近似。
:::

$s_k = s_{t_k}(a_k)$，用第二节的闭式公式从 $v_\theta$ 直接算出。

### 5.2 轨迹 Log-Prob

和 ReinFlow 一样，整条轨迹的 log-prob 是各步之和：

$$
\log\pi(a_K|s) = \log p_0(a_0) + \sum_{k=0}^{K-1} \log p(a_{k+1}|a_k, s)
$$

**这个公式在做什么**：把整条轨迹（从初始噪声 $a_0$ 到最终动作 $a_K$）的对数概率，拆成初始分布的 log-prob 加上每一步转移的 log-prob 之和，让 PPO 能用整条轨迹的 log-prob 计算重要性比率。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\log p_0(a_0)$ | **起点概率** | 初始噪声 $a_0 \sim \mathcal N(0,I)$ 的 log-density，这部分不含可学习参数 |
| $\log p(a_{k+1}\mid a_k, s)$ | **单步转移概率** | 每一步高斯转移的 log-density，取自上一条公式 |
| $\sum_{k=0}^{K-1}(\cdot)$ | **逐步累加** | 把 $K$ 步的转移概率取对数后相加（对数乘法变加法） |
| $\log\pi(a_K\mid s)$ | **整条轨迹的 log-prob** | PPO 计算重要性采样比率时需要的最终动作 $a_K$ 的对数概率 |

**用人话读**："整条轨迹的对数概率，等于起点的对数概率加上每一步转移的对数概率，一步步加起来。"

**为什么是这个形式**：因为整条 SDE 轨迹是一个马尔可夫链，联合概率等于各步条件概率相乘，取对数后就变成了求和——这是 PPO 能在多步生成过程上直接算梯度的关键，全程不需要任何近似。
:::

每一项是标准的多维高斯 log-density：

$$
\log p(a_{k+1}|a_k, s) = -\frac{d}{2}\log(2\pi\sigma_\phi^2\Delta t) - \frac{\|a_{k+1} - \mu_k\|^2}{2\sigma_\phi^2\Delta t}
$$

**这个公式在做什么**：把上一条公式里"单步转移的 log-prob"展开成具体的多维高斯对数密度公式——前半项是归一化常数，后半项是"实际落点离均值有多远"的惩罚。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $-\frac{d}{2}\log(2\pi\sigma_\phi^2\Delta t)$ | **归一化常数项** | 只依赖动作维度 $d$ 和方差 $\sigma_\phi^2\Delta t$，保证概率密度积分为 1，和实际落点 $a_{k+1}$ 无关 |
| $\frac{\|a_{k+1}-\mu_k\|^2}{2\sigma_\phi^2\Delta t}$ | **偏离惩罚** | 实际落点 $a_{k+1}$ 距离均值 $\mu_k$ 的欧氏距离平方，除以方差做归一化——偏得越远这一项越大 |
| $\mu_k$ | **本步预期落点** | 即 5.1 节公式里的均值 $a_k + [v_\theta + \alpha_\psi^{\text{scaled}}\cdot s_k]\Delta t$ |
| $-(\cdot)$ | **符号翻转** | 惩罚项前面的负号让"偏离越大、log-prob 越小"，符合概率密度的直觉 |

**用人话读**："这一步的对数概率 = 一个只和方差有关的常数，减去实际落点偏离预期中心的程度（偏离越大，概率越低）。"

**为什么是这个形式**：这是多维独立高斯分布 log-density 的标准闭式解，直接由高斯 PDF 取对数得到，不需要数值积分或近似，是 PPO 能够对整条 SDE 轨迹求梯度的基础。
:::

**这个公式在做什么**：把上一条公式里"单步转移概率"具体展开成多维高斯分布的标准 log-density 公式，包含一个归一化常数项和一个平方误差项。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $-\frac{d}{2}\log(2\pi\sigma_\phi^2\Delta t)$ | **归一化常数** | 高斯分布 log-density 公式里固定的归一化项，只依赖维度 $d$ 和方差，不依赖实际取值 |
| $\|a_{k+1} - \mu_k\|^2$ | **偏差平方** | 实际下一步位置和高斯均值 $\mu_k$ 之间的欧氏距离平方，偏差越大概率越低 |
| $2\sigma_\phi^2\Delta t$ | **方差缩放** | 分母里的方差项，方差越大，同样的偏差对应的概率密度下降得越慢 |
| $\log p(a_{k+1}\mid a_k, s)$ | **单步 log-density** | 综合起来就是"这一步实际走到的位置，在预期高斯分布下有多'合理'" |

**用人话读**："这一步走到的地方离预期中心差多远（平方误差），除以方差再减去一个和维度、方差有关的常数，就是这一步的对数概率。"

**为什么是这个形式**：这是多维独立高斯分布对数密度的标准解析公式，之所以能直接套用，是因为 ScoRe-Flow 保证了每一步转移仍然是高斯——只是均值 $\mu_k$ 里多包含了 score drift 修正这一项。
:::

其中 $\mu_k = a_k + [v_\theta(t_k, a_k, s) + \alpha_\psi^{\text{scaled}}(t_k) \cdot s_k] \Delta t$。

### 5.3 PPO 训练

有了 log-prob，标准 PPO 直接使用。训练联合优化三组参数：

| 参数 | 网络 | PPO 中的作用 |
|------|------|-------------|
| $\theta$ | 速度场 $v_\theta$ | 基础策略（同时影响 drift 和 score） |
| $\psi$ | Score scheduler $\alpha_\psi$ | 控制方向引导强度 |
| $\phi$ | Variance predictor $\sigma_\phi$ | 控制探索幅度 |

**训练流程**：

```mermaid
flowchart TD
    A["环境交互：用 SDE 采样动作轨迹"] --> B["记录每步的<br/>(a_k, v_θ, s_k, σ_φ, ε_k)"]
    B --> C["计算 log-prob<br/>（高斯累加）"]
    C --> D["计算 GAE Advantage"]
    D --> E["PPO clip 更新<br/>θ, ψ, φ 联合优化"]
    E --> A
```

**关键细节**：
- $v_\theta$ 同时参与 drift 和 score 的计算（score = $f(v_\theta)$），所以优化 $\theta$ 会同时改善"方向"和"引导"
- 初始化时 $\alpha_\psi$ 较小（接近 ReinFlow），训练过程中逐渐增大
- Critic 网络独立训练（和标准 PPO 一样）

---

## 六、和其他方法的核心对比

### 6.1 方法定位图

```mermaid
flowchart TD
    subgraph "Flow 策略 RL 微调方法"
        R["ReinFlow<br/>只加噪声"] --> S["ScoRe-Flow<br/>+score drift（本文）"]
        R --> SSDE["Score-SDE<br/>drift和noise绑定"]
        F["FlowRL<br/>似然近似+PPO"] 
        SAC["SAC-Flow<br/>改网络结构+SAC"]
    end
    style S fill:#e1f5fe
```

### 6.2 详细对比表

| 方法 | 改了 Flow 的什么 | Drift 修正？ | Variance 独立学？ | RL 算法 | 对预训练模型友好？ |
|------|----------------|------------|-----------------|--------|----------------|
| [ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) | 只加噪声 $\sigma$ | ❌ | ✅ | PPO | ✅（不改网络） |
| Score-SDE（消融） | Score drift + 绑定噪声 | ✅ | ❌（$\sigma = \sqrt{2\alpha}$） | PPO | ✅ |
| **ScoRe-Flow（本文）** | Score drift + 独立噪声 | ✅ | ✅ | PPO | ✅（不改网络） |
| [SAC-Flow](./079_SAC_Flow_用SAC直接训练Flow策略) | 重新参数化速度网络 | ❌（直接改 $v_\theta$） | N/A | SAC | ❌（需改网络结构） |
| [FlowRL](./018_FlowRL_Flow_VLA的在线RL微调) | 不改 Flow 生成过程 | ❌ | N/A | PPO | ✅（近似 log-prob） |

### 6.3 ScoRe-Flow vs SAC-Flow 的本质区别

这两个方法代表 Flow RL 微调的两条完全不同的路线：

| 维度 | ScoRe-Flow（本文） | [SAC-Flow](./079_SAC_Flow_用SAC直接训练Flow策略) |
|------|-------------------|-----------|
| **哲学** | "在采样时引导方向" | "直接改 flow 内部结构" |
| **修改位置** | 采样过程（加 drift + noise） | 网络结构（GRU/Transformer） |
| **RL 算法** | On-policy PPO | Off-policy SAC |
| **样本效率** | 较低（on-policy 必须丢弃旧数据） | 较高（off-policy 复用旧数据） |
| **对预训练模型兼容性** | ✅ 完全兼容（不改 $v_\theta$ 结构） | ❌ 需要重新设计 $v_\theta$ 结构 |
| **适用场景** | 有大型预训练 flow（如 π₀），不想改结构 | 从头训练或小模型 |

**互补性**：SAC-Flow 样本效率高但需要改网络；ScoRe-Flow 不改网络结构，对已有预训练 flow model（如 π₀）更友好。

### 6.4 ScoRe-Flow vs Score-SDE（为什么解耦很重要）

Score-SDE（论文中的消融实验）把 score 强度和噪声强度绑定为 $\sigma = \sqrt{2\alpha}$：

| | Score-SDE（绑定） | ScoRe-Flow（解耦） |
|---|---|---|
| 关系 | 增大 score 引导 → 噪声自动增大 | 两者独立调节 |
| 问题 | 想加强引导时被迫加大噪声 → 训练不稳定 | 可以强引导 + 小噪声 |
| 初期 | 收敛快（score 引导强） | 收敛快 |
| 后期 | 性能受限（噪声无法单独减小） | 性能更高 |

**消融结论**：解耦后最终性能提升约 3-5%（在操作任务上差距更大）。

---

## 七、实验结果

### 7.1 收敛速度

在 D4RL 运动任务（HalfCheetah, Hopper, Walker2d）上：

| 对比 | ScoRe-Flow 的优势 |
|------|------------------|
| vs ReinFlow（相同步数 $K=4$） | 快 **2.4 倍**到达 90% 最终性能 |
| vs DPPO（扩散策略 + PPO，$K=50$） | 快 **21.9 倍** |

收敛快的原因：score drift 让策略在每个 PPO 更新后更快地"找到"好动作区域，而不是靠随机噪声碰运气。

### 7.2 操作任务

| 任务 | ReinFlow-S | DPPO | **ScoRe-Flow** |
|------|-----------|------|-----------|
| Robomimic PickPlaceCan | 91.7% | 96.5% | **98.3%** |
| Robomimic Square | 77.3% | 78.3% | **84.7%** |
| Robomimic Transport | 88.7% | 53.0% | **94.4%** |
| Kitchen Complete（满分 4） | 3.9 | 3.8 | **4.0** |

**关键观察**：
- 在最难的 Transport 任务上（双臂协作），ScoRe-Flow 超出 ReinFlow **5.7%**——score 引导在高维复杂任务上优势更明显
- ScoRe-Flow 在所有任务上都达到或接近满分

### 7.3 消融：Score 各组件的作用

| 配置 | 效果 |
|------|------|
| 完整 ScoRe-Flow | 最佳 |
| 去掉 score（$\alpha=0$） | 退化为 ReinFlow（只有噪声），收敛变慢 |
| Score 强度固定为 1（$\alpha=1$，不学习） | Score 在 $t\to1$ 爆炸，训练崩溃 |
| 不加 $(1-t)$ 衰减 | 同上：训练崩溃 |
| Score-SDE（drift 和 variance 绑定） | 初期收敛快，但最终性能低于 ScoRe-Flow |
| 只学 $\alpha$，固定 $\sigma$ | 性能中等（不能自适应调节探索） |

**最重要的消融结论**：
1. Score drift 本身贡献了收敛速度的主要提升（vs ReinFlow）
2. $(1-t)$ 衰减是必需的（没有它训练直接崩）
3. $\alpha$ 和 $\sigma$ 解耦贡献了最终性能的额外提升

---

## 八、关键 Takeaway

### 8.1 核心贡献总结

1. **Score = 免费的方向引导**。从 flow 的速度场做一个简单代数变换 $s_t = (tv_\theta - a)/(1-t)$ 就能得到 score——不需要额外网络，不需要额外训练。这让 drift 修正变成了"免费午餐"。

2. **解耦 mean 和 variance 控制**。之前的方法把"往哪走"（drift）和"探索多少"（noise）绑在一起。ScoRe-Flow 解耦了它们——可以独立控制"策略朝哪偏"和"探索幅度多大"。

3. **On-policy PPO，不需要改 flow 网络结构**。和 SAC-Flow 需要重新参数化速度网络不同，ScoRe-Flow 的 flow 网络结构完全不变——只在采样时加了 score drift 和学习噪声。这对已有的大型预训练 flow model（如 π₀）更友好。

4. **$(1-t)$ 衰减是关键**。Score 在 $t \to 1$ 时发散，不加约束会导致训练崩溃。$(1-t)$ 因子的 hard constraint 是这个方法能 work 的必要条件。

### 8.2 适用场景

| 场景 | ScoRe-Flow 是否合适 | 原因 |
|------|-------------------|------|
| 有大型预训练 Flow（如 π₀）+ 想用 RL 提升 | ✅ 最佳选择 | 不改网络结构，直接在采样时加引导 |
| 从头训练 Flow + RL | ⚠️ 可以但不一定最优 | SAC-Flow 可能样本效率更高 |
| 离线 RL 设置 | ❌ 不适用 | On-policy 方法，需要在线交互 |
| Diffusion Policy（DDPM）+ RL | ❌ 不适用 | 本方法专为 Flow Matching 设计 |

---

## 延伸阅读

- [Score Function（密度梯度）与 Score Matching](/前置知识/001t_前置知识_Score_Function密度梯度与Score_Matching) ← Score function 是什么、为什么能从 $v_\theta$ 免费算出来
- [ReinFlow：Flow 策略的噪声注入 RL 微调](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) ← ScoRe-Flow 的 baseline
- [SAC-Flow：用 SAC 直接训练 Flow 策略](./079_SAC_Flow_用SAC直接训练Flow策略) ← Off-policy 路线：改网络结构让梯度稳定
- [FlowRL：Flow VLA 的在线 RL 微调](./018_FlowRL_Flow_VLA的在线RL微调) ← 大规模 VLA 上的 Flow RL
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) ← Flow Matching 基础
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) ← PPO 的完整原理

**原始论文**：Qiu et al., "ScoRe-Flow: Complete Distributional Control via Score-Based Reinforcement Learning for Flow Matching", ICML 2026
