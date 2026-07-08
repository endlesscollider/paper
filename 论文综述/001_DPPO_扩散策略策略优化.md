---
title: DPPO：扩散策略策略优化
order: 101
tags: [强化学习, 扩散模型]
category: 精读
star: 4
---

# DPPO：Diffusion Policy Policy Optimization 深度精读

> **论文标题**: Diffusion Policy Policy Optimization  
> **作者**: Allen Z. Ren, Justin Lidard, Lars L. Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, Max Simchowitz  
> **机构**: Princeton University, MIT, Toyota Research Institute, CMU  
> **发表**: NeurIPS 2024 (arXiv:2409.00588)  
> **代码**: https://diffusion-ppo.github.io/

**标签**: `#扩散策略` `#强化学习微调` `#PPO` `#机器人操作` `#Sim2Real` `#策略梯度` `#行为克隆` `#长horizon操作`

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO clip 的核心机制
- [扩散模型 DDPM](/前置知识/000b_前置知识_扩散模型DDPM) — 去噪步的高斯结构
- [Diffusion Policy](/前置知识/000c_前置知识_Diffusion_Policy) — 被微调的基础模型
- [对数似然与变分下界](/前置知识/000e_前置知识_对数似然与变分下界) — 理解每步 log-likelihood
- [为什么扩散策略难以 RL 微调](/前置知识/000f_前置知识_为什么扩散策略难以RL微调) — DPPO 的动机深度解析
- [D²PPO：解决表示坍塌](./004_D2PPO_解决表示坍塌) — DPPO 的改进方案
- [Online DPRL 综述](./003_Online_DPRL_综述_扩散策略与在线RL) — 在四大家族中的定位

---

## 一、背景与动机

### 1.1 现状：机器人策略学习的主流 Pipeline

2024 年的机器人学习已经形成了一个清晰的范式：

1. 人类遥操作采集示教数据（通常 50–300 条轨迹）
2. 用行为克隆（BC）训练策略（Diffusion Policy 是当前最强的 BC 方法）
3. 部署后发现性能有天花板（因为示教数据不完美/覆盖不足）

BC 的天花板在哪？
- 人类示教本身有误差（手抖、反应慢）
- 数据只覆盖了有限的状态空间
- 没有闭环反馈——策略不知道自己执行得对不对

### 1.2 为什么想用 RL 微调

RL 的核心优势：**通过和环境交互，策略可以超越示教数据的水平。**

- **BC 策略**："模仿人类怎么做" → 最好和人类一样
- **RL 微调**："在模仿基础上自己探索优化" → 可以超越人类

实际例子：BC 训练后插入任务成功率 65%（和人类示教的成功率差不多）；RL 微调后插入任务成功率 95%（通过自己练习超越了人类示教的精度）。

### 1.3 扩散策略 + RL 微调的核心难题

Diffusion Policy 是当前最强的 BC 方法（多模态、高维、稳定），但它有一个致命问题让 RL 微调很难做：

**策略梯度需要 $\log \pi(\mathbf{a}|\mathbf{s})$，但扩散策略没有显式的动作概率密度。**

为什么没有？回忆扩散策略的采样过程：

$$
\mathbf{a}_K \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \leftarrow \text{从纯噪声开始}
$$

$$
\mathbf{a}_{k-1} = \text{denoise}(\mathbf{a}_k, \mathbf{s}, k), \quad k = K, K-1, \ldots, 1
$$

$$
\mathbf{a}_0 = \text{最终动作} \quad \leftarrow \text{经过 } K \text{ 步去噪得到}
$$

要算 $\pi(\mathbf{a}_0|\mathbf{s})$，就要对所有可能的中间路径 $\{\mathbf{a}_1, \ldots, \mathbf{a}_K\}$ 积分：

$$
\pi(\mathbf{a}_0|\mathbf{s}) = \int \cdots \int p(\mathbf{a}_K) \prod_{k=1}^K p_\theta(\mathbf{a}_{k-1}|\mathbf{a}_k, \mathbf{s}) \, d\mathbf{a}_1 \cdots d\mathbf{a}_K
$$

这是一个 $K \times \text{action\_dim}$ 维度的积分，不可能解析计算。

### 1.4 之前的尝试和它们的问题

在 DPPO 之前，人们尝试了几种绕过这个问题的方法：

| 方法 | 思路 | 问题 |
|------|------|------|
| DQL | 学 Q 函数，梯度穿过去噪链 | 稀疏奖励 + 长 action chunk 时训练不稳定 |
| IDQL | Q 函数给动作打分选择 | 本质还是监督学习，改进有限 |
| DIPO | Q 梯度"推"buffer 里的动作 | 间接更新，信息损失大 |
| QSM | 对齐 score 和 Q 梯度 | 理论优美但实践中复杂任务崩溃 |

**这些方法的共性问题**：都依赖 off-policy Q-function 学习。Q 函数在高维动作空间 + 稀疏奖励 + 大 action chunk 的组合下极不稳定。

### 1.5 DPPO 的核心贡献声明

论文提出了一个"违反直觉"的发现：

> 对于扩散策略的微调，最简单的策略梯度方法（PPO）反而比花哨的 Q-learning 方法效果更好。

之前大家猜测 PG 对扩散策略不行（因为去噪步太多 → 等效 horizon 长 → 梯度方差大）。DPPO 证明这个猜测是错的。

---

## 二、方法：两层扩散策略 MDP

这是论文的核心技术贡献。

### 2.1 思路概述

既然不能算整体的 $\log \pi(\mathbf{a}_0|\mathbf{s})$，那就**别算它**。把去噪过程本身展开成一个 MDP，让 PPO 在这个展开的 MDP 上工作。

原始问题的结构如下：

- **环境 MDP**：$s_0 \to a_0 \to s_1 \to a_1 \to \cdots$
- **每个 $a_t$ 的生成**：$\mathbf{a}_K \to \mathbf{a}_{K-1} \to \cdots \to \mathbf{a}_0$（$K$ 步去噪）

DPPO 的视角：把两层嵌套在一起，形成一个大的 MDP。大 MDP 的每一"步"对应去噪的一步；大 MDP 的状态 = (环境状态, 当前去噪中间结果)；大 MDP 的动作 = 去噪一步；大 MDP 的奖励 = 只在去噪完成时给（即 $k=0$ 时给环境奖励）。

```mermaid
flowchart LR
    subgraph 环境MDP
        S0["s₀"] --> A0["a₀"] --> S1["s₁"] --> A1["a₁"] --> S2["s₂"]
    end
    subgraph 去噪链生成a₀
        NK["a_K ~ N(0,I)"] --> NK1["a_{K-1}"] --> dots["..."] --> N0["a_0"]
    end
    A0 -.- N0
```

### 2.2 形式化定义

定义展开后的 MDP（叫 $\bar{\mathcal{M}}$）如下：

**索引系统**：用 $\bar{t}(t, k) = t \times K + (K - k - 1)$ 把 (环境时间步 $t$, 去噪步 $k$) 映射到一个线性索引。注意 $k$ 是递减的（$K \to 0$），因为去噪是从高噪声到低噪声。

**状态**：

$$
\bar{s}_{\bar{t}(t,k)} = (s_t,\; \mathbf{a}_t^{k+1})
$$

含义：$s_t$ = 当前环境状态，$\mathbf{a}_t^{k+1}$ = 上一步去噪后的动作（当前要继续去噪的输入）。

**动作**：

$$
\bar{a}_{\bar{t}(t,k)} = \mathbf{a}_t^k
$$

含义：去噪一步，从 $\mathbf{a}_t^{k+1}$ 变成 $\mathbf{a}_t^k$。

**奖励**：

$$
\bar{R}_{\bar{t}(t,k)} = \begin{cases} 0 & \text{如果 } k > 0 \quad \leftarrow \text{去噪中间步没有环境奖励} \\ R(s_t, \mathbf{a}_t^0) & \text{如果 } k = 0 \quad \leftarrow \text{去噪完成，动作给环境，获得奖励} \end{cases}
$$

**转移**：

$$
\bar{P}(\bar{s}_{\bar{t}+1} | \bar{s}_{\bar{t}}, \bar{a}_{\bar{t}}) = \begin{cases} \delta(s_t, \mathbf{a}_t^k) & \text{如果 } k > 0 \quad \leftarrow \text{确定性：环境状态不变} \\ P(s_{t+1}|s_t, \mathbf{a}_t^0) \otimes \mathcal{N}(\mathbf{0}, \mathbf{I}) & \text{如果 } k = 0 \quad \leftarrow \text{环境推进 + 采样下一个纯噪声} \end{cases}
$$

**策略**：

$$
\bar{\pi}_\theta(\bar{a} | \bar{s}) = p_\theta(\mathbf{a}_t^k | \mathbf{a}_t^{k+1}, s_t) = \mathcal{N}\!\left(\mathbf{a}_t^k;\; \mu_\theta(\mathbf{a}_t^{k+1}, k+1, s_t),\; \sigma_{k+1}^2 \mathbf{I}\right)
$$

**关键**：这是一个高斯分布！有显式的概率密度！可以直接算 log-likelihood！

$$
\log \bar{\pi}_\theta(\bar{a} | \bar{s}) = -\frac{\|\mathbf{a}_t^k - \mu_\theta(\mathbf{a}_t^{k+1}, k+1, s_t)\|^2}{2\sigma_{k+1}^2} + \text{const}
$$

### 2.3 策略梯度公式

对展开的 MDP 用标准策略梯度：

$$
\nabla_\theta \bar{J}(\bar{\pi}_\theta) = \mathbb{E}_{\bar{\pi}_\theta}\!\left[\sum_{\bar{t} \geq 0} \nabla_\theta \log \bar{\pi}_\theta(\bar{a}_{\bar{t}} | \bar{s}_{\bar{t}}) \cdot \bar{r}(\bar{s}_{\bar{t}}, \bar{a}_{\bar{t}})\right]
$$

其中 $\bar{r}$ 是从 $\bar{t}$ 开始的折扣累积奖励。

**采样这个梯度**只需要正常地跑 Diffusion Policy 的采样过程——从纯噪声开始去噪 $K$ 步，中间记录每一步的输入输出。这本来就是生成动作时会做的事情，不需要额外开销。

### 2.4 PPO 更新的具体形式

用 PPO 的 clip 目标函数：

$$
L_{\text{CLIP}} = \mathbb{E}\!\left[\min\!\left(\bar{r}(\theta) \cdot \hat{A},\; \text{clip}(\bar{r}(\theta),\, 1-\varepsilon,\, 1+\varepsilon) \cdot \hat{A}\right)\right]
$$

其中概率比为：

$$
\bar{r}(\theta) = \frac{\bar{\pi}_\theta(\bar{a}|\bar{s})}{\bar{\pi}_{\theta_{\text{old}}}(\bar{a}|\bar{s})}
$$

**这里每一步去噪都有自己的 clip 约束。** 不是对整条去噪链做一次 clip，而是每步都 clip。这保证了每一步去噪的变化幅度都被控制住。

### 2.5 Advantage 估计的设计

这是一个关键设计选择。论文对比了三种方案：

**方案 A（DPPO 选择的）：只用环境状态估计 value**

$$
\hat{V}(\bar{s}_{\bar{t}(t,0)}) := \tilde{V}(s_t)
$$

$$
\hat{A}(\bar{s}, \bar{a}) = \gamma_{\text{denoise}}^k \cdot (\bar{r} - \tilde{V}(s_t))
$$

$\gamma_{\text{denoise}}$ 是去噪折扣因子（通常 0.8–0.99），$k$ 是当前去噪步编号。

**方案 B：value 同时依赖环境状态和去噪结果**

$$
\hat{V} = \tilde{V}(s_t, \mathbf{a}_t^{k=1})
$$

**方案 C：学 Q 函数**

$$
\tilde{Q}(s_t, \mathbf{a}_t^{k=0})
$$

**为什么方案 A 最好？**

实验发现方案 B 和 C 在复杂任务上不如 A。论文推测原因是：

1. 扩散策略的动作高度随机（每次采样 $\mathbf{a}_K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 都不同），给 value 函数的输入加入动作维度后，value 估计方差急剧增大
2. Action chunk 使动作维度更高（$T_a \times \text{action\_dim}$），进一步加剧了这个问题
3. 只用环境状态 $s_t$ 估计 value，相当于在所有可能的去噪路径上平均，天然降低了方差

**$\gamma_{\text{denoise}}$ 的作用**：

$\gamma_{\text{denoise}}^k$ 随 $k$ 增大而指数衰减：

- $k=0$（最后一步去噪，接近最终动作）：权重最大
- $k=K-1$（第一步去噪，接近纯噪声）：权重最小

直觉：越接近纯噪声的步骤，对最终动作的影响越间接，梯度信号也越嘈杂，应该给更小的权重。这类似于 DDPM 完整 ELBO 中对不同时间步的权重分配。

### 2.6 只微调最后几步去噪

预训练时用 $K=20$ 或 $K=100$ 步去噪（更好地拟合复杂数据分布）。

微调时只更新最后 $K'$ 步的参数（$K'$ 通常 5–10）：

- 步骤 $K \to K-K'+1$：使用冻结的预训练权重 $\theta_{\text{frozen}}$
- 步骤 $K' \to 1$：使用可微调的权重 $\theta_{\text{ft}}$（初始化为预训练权重的副本）

**为什么可以只调最后几步？**

- 前几步去噪：从纯噪声变成粗略形状 → BC 已经学得很好了，不需要改
- 后几步去噪：从粗略形状精细调整成最终动作 → RL 需要在这里做精细调整

类比：画画时先画轮廓（前几步去噪）→ 不需要改；最后上色/细节（后几步去噪）→ RL 微调这里。

**实验验证**：$K'=10$ 就能达到全部微调的效果，但节省了大量 GPU 显存和计算时间。

### 2.7 DDIM 加速微调

对于像素输入或长 horizon 任务，可以用 DDIM 把去噪步压缩到 $K_{\text{DDIM}}=5$：

- **预训练**：DDPM，$K=100$ 步去噪
- **微调**：DDIM，$K_{\text{DDIM}}=5$ 步去噪

训练时 $\eta=1$（DDIM 公式中加噪声，等效于 DDPM，提供探索）；评估时 $\eta=0$（确定性采样，性能最好）。

### 2.8 噪声调度 trick

DDPM 原始的余弦调度在 $k=0$ 附近 $\sigma_k \approx 10^{-4}$，非常小。但在 RL 微调中：

- 太小的 $\sigma$ → 探索不足 → 策略困在局部最优
- 太大的 $\sigma$ → 动作噪声太大 → 训练不稳定

**DPPO 的做法**：

$$
\sigma_{\min}^{\text{exp}} = 0.01 \sim 0.1 \quad \leftarrow \text{采样动作时的最小噪声（保证探索）}
$$

$$
\sigma_{\min}^{\text{prob}} = 0.1 \quad \leftarrow \text{计算 log-likelihood 时的最小噪声（避免概率爆炸）}
$$

为什么 $\sigma_{\min}^{\text{prob}}$ 要单独设？因为如果 $\sigma$ 太小，$\log \mathcal{N}(\mathbf{a}; \mu, \sigma^2)$ 的绝对值会非常大（分母 $\sigma^2$ 在指数里），导致 PPO 的概率比 $r(\theta)$ 数值不稳定。

---

## 三、为什么 DPPO 比其他方法强——机制分析

论文用一个精心设计的 2D 避障任务（D3IL benchmark 的 Avoid 环境）做了深入的可视化分析。

### 3.1 实验设置

- **任务**：机械臂需要从桌子一侧移动到另一侧，中间有障碍物
- **动作空间**：2D（末端执行器位置）
- **示教数据**：有两种模式（左绕和右绕）
- **RL 目标**：学会从上方（最快路径）到达目标线

三组数据：M1（两种较近的路径）、M2（两种较远的路径）、M3（两种中等距离的路径）。

### 3.2 优势一：结构化的在流形上探索

微调开始时（第一次 PPO iteration），观察策略采样的轨迹分布：

**DPPO**：
- 轨迹分布广泛地围绕预训练数据的两个模式
- 但始终在"合理路径"的流形附近
- 覆盖了两个模式之间的空间（可能发现更好的中间路径）

**Gaussian (PPO)**：
- 加了各向同性的高斯噪声
- 探索缺乏结构，尤其在 M2（两模式相距较远）时
- 噪声可能把轨迹推到完全不合理的区域

**GMM (PPO)**：
- 每个混合分量内部的探索范围较窄
- 覆盖不如 DPPO 广

**为什么 DPPO 的探索更好？**

扩散模型的去噪过程天然有"把采样拉回数据流形"的效果。即使中间加了大噪声，多步去噪会逐步把它修正回合理的动作空间。这和高斯策略"加完噪声就直接执行"完全不同。

$$
\text{Gaussian}: \quad \mu + \text{noise} \;\to\; \text{直接执行（noise 可能把动作推到不合理的地方）}
$$

$$
\text{DPPO}: \quad \mathbf{a}_K\text{(纯噪声)} \;\to\; \text{去噪} \;\to\; \cdots \;\to\; \mathbf{a}_0\text{（多步修正确保在流形上）}
$$

### 3.3 优势二：多步去噪带来的训练稳定性

**实验 1：往动作里注入外部噪声**

模拟真实机器人的不完美执行器（控制信号有随机误差）：在微调第 5 个 iteration 后，往采样的动作上加 $\text{Uniform}(0.1, 0.2)$ 的噪声。

结果：

| 方法 | 表现 |
|------|------|
| Gaussian | 成功率直接崩到 0 |
| GMM | 成功率直接崩到 0 |
| DPPO ($K' \geq 4$ 步) | 成功率基本不受影响 |
| DPPO ($K' = 2$ 步) | 有下降但不至于崩溃 |

**为什么？** 多步去噪相当于多次"修正"的机会。外部噪声加在最终动作上，但如果还有去噪步可以调整，策略就能"恢复"。

**实验 2：大 action chunk**

把 action chunk size $T_a$ 从 4 增大到 16：

| 方法 | 表现 |
|------|------|
| Gaussian (PPO) | $T_a=8$ 开始性能下降，$T_a=16$ 几乎不收敛 |
| GMM (PPO) | $T_a=8$ 开始明显下降 |
| DPPO | $T_a=16$ 仍然稳定收敛 |

大 action chunk → 动作空间维度高 → 单步更新很难精确控制整条 chunk。DPPO 的迭代去噪天然地在 chunk 的时间维度上进行"逐步精细化"。

### 3.4 优势三：微调后策略的鲁棒性

微调**完成后**测试策略的鲁棒性：

**测试 1：动作噪声扰动**

往微调好的策略的输出动作上加不同幅度的噪声：

| 策略 | 噪声 0.1 | 噪声 0.2 |
|------|-----------|-----------|
| DPPO | 100% → 90% | 100% → 75% |
| Gaussian | 95% → 40% | 95% → 10% |

**测试 2：初始状态分布变化**

- 训练时初始状态：固定的一个点
- 测试时初始状态：扩大范围（从更多位置开始）

DPPO 能从大范围的初始状态收敛到最优路径；Gaussian 只在训练时的初始状态附近有效。

**解释**：扩散策略天然具有"去卷积"能力——给它一个带噪的输入，它能恢复出合理的输出。这个能力在 RL 微调后被保留下来了。

---

## 四、实验详情

### 4.1 实验环境总览

论文在三个层次的 benchmark 上验证：

| Benchmark | 任务 | 难度 | 关键挑战 |
|-----------|------|------|---------|
| OpenAI Gym | Hopper, Walker2D, HalfCheetah | 低 | 密集奖励，连续运动 |
| Robomimic | Lift, Can, Square, Transport | 中–高 | 稀疏奖励，操作精度，双臂 |
| FurnitureBench | One-leg, Lamp, Round-table | 很高 | 多阶段、长 horizon、真机 |

### 4.2 和其他扩散 RL 方法的对比（Section 5.1）

**设置**：用相同的 Diffusion Policy BC 预训练，然后用不同 RL 方法微调。

**Gym 任务（密集奖励，简单）**：

| 方法 | Hopper-v2 | Walker2D-v2 |
|------|-----------|-------------|
| DPPO | ~3600 | ~5200 |
| IDQL | ~3500 | ~5000 |
| DAWR | ~3400 | — |
| DQL | ~3200 | — |
| DIPO | ~3000 | — |
| QSM | ~2800（训练不稳定） | — |

Gym 任务相对简单，差距不算很大。但 DPPO 训练曲线最平滑。HalfCheetah 上所有方法都能收敛，差距不大。

**Robomimic 任务（稀疏奖励，操作）——差距拉开的地方**：

| 任务 | DPPO | DRWR | Gaussian | DQL | DIPO | QSM |
|------|------|------|----------|-----|------|-----|
| Lift（简单） | ~100% | ~100% | ~100% | ~100% | ~100% | ~100% |
| Can（中等） | ~100% | ~95% | — | 70–90% | — | — |
| Square（困难） | ~100% | ~85% | ~60% | 更低 | 更低 | 更低 |
| Transport（双臂） | >90% | ~40% | — | ~10% | ~5% | ≈0% |

Transport 任务是双臂操作（14 维动作），需要两个机械臂协调取出 bin 盖、拿放物体、传递锤子。Episode 长度 800 步，奖励完全稀疏。

**为什么其他方法在 Transport 崩了？**

- Off-policy 方法（DQL, DIPO, QSM）：需要学 $Q(s, a)$。$a$ 是 $14 \times T_a = 112$ 维的 action chunk。在如此高维的动作空间学 Q 函数，加上稀疏奖励，几乎不可能。
- DRWR/DAWR：加权回归类，对 reward signal 的利用效率低于直接策略梯度。

### 4.3 和其他策略参数化的对比（Section 5.2）

**关键问题**：DPPO 比其他方法强，到底是因为用了 PPO，还是因为用了扩散模型参数化？

**对比**：同样用 PPO，但策略用不同的参数化：Gaussian-MLP、Gaussian-Transformer、GMM-MLP、GMM-Transformer、DPPO-MLP（扩散模型，MLP 网络）、DPPO-UNet（扩散模型，UNet 网络）。

**State 输入结果（Robomimic）**：

| 方法 | Square | Transport |
|------|--------|-----------|
| DPPO-MLP | ~100%（200 iter） | >90% |
| DPPO-UNet | ~100%（250 iter） | >90% |
| Gaussian-MLP | ~60% | ~50%（训练不稳定） |
| Gaussian-Transformer | ~55% | — |
| GMM-MLP | ~30% | ≈0% |
| GMM-Transformer | ~25% | — |

**Pixel 输入结果（Robomimic）**：

| 方法 | Transport (pixel) |
|------|-------------------|
| DPPO-ViT-MLP | ~70%（很难的任务，pixel 输入还能做到这个水平） |
| Gaussian-ViT-MLP | 0%（完全无法从 pixel 微调成功） |

**结论**：不仅是 PPO vs 其他 RL 算法的优势，扩散模型参数化本身在 PPO 微调中也比 Gaussian/GMM 好。两个优势叠加。

### 4.4 FurnitureBench：长 horizon 操作 + Sim-to-Real

**任务复杂度**：

- **One-leg**：组装桌子的一条腿（抓桌面放固定架 → 拿起桌腿 → 对齐 → 插入 → 拧紧），约 700 步
- **Lamp**：组装台灯（固定底座 → 拧灯泡 → 放灯罩），约 1000 步
- **Round-table**：组装圆桌（放桌面 → 插+拧桌腿 → 插+拧底座），约 1000 步

每个任务有两种初始随机度：Low 和 Med（Medium 的初始物体位姿随机范围更大）

**仿真结果**：

| 任务 | DPPO | Gaussian |
|------|------|----------|
| One-leg Low | ~95% | ~80%（3 seeds 中 1 个崩） |
| One-leg Med | ~70% | 0%（所有种子崩） |
| Lamp Low | ~85% | ~90%（唯一 Gaussian 略好的场景） |
| Lamp Med | ~60% | 0% |
| Round-table Low | ~75% | 0% |
| Round-table Med | ~50% | 0% |

**关键观察**：Gaussian 策略在 Medium randomness 下几乎全部策略坍塌（成功率崩到 0）。DPPO 在所有 6 个设置中都没有坍塌。

### 4.5 Sim-to-Real 迁移（最重要的结果）

**设置**：
- 在 IsaacGym 中用 1000 个并行环境训练 One-leg 任务
- 训练完直接部署到真实的 Franka Emika Panda 机械臂
- Zero-shot，不用任何真实数据微调
- 20 次 trials，单盲评估

**结果**：

| 方法 | 仿真成功率 | 真实成功率 |
|------|-----------|-----------|
| Diffusion BC only | 65% | 45% |
| DPPO | 95% | 80%（16/20） |
| Gaussian (PPO) | 88% | 0%（0/20） |
| Gaussian + BC reg | 53% | 50%（10/20） |

**为什么 Gaussian 仿真 88% 但真机 0%？**

视频分析显示 Gaussian 微调后的策略动作非常 "jittery"（高频抖动）。在仿真的完美执行器模型下可以 work，但真实执行器有带宽限制，高频命令会被滤掉，导致实际运动和预期完全不同。

**为什么 DPPO 不抖？**

多步去噪本身就是一个 low-pass filter。高频噪声会在去噪过程中被平滑掉。所以 DPPO 天然输出低频、平滑的动作序列。

**DPPO 微调后的行为变化**：

论文展示了代表性的硬件 rollout：
- **预训练策略**：成功但精度有限，偶尔在插入阶段因为对不齐而失败
- **DPPO 策略**：展现出**纠正行为（corrective behavior）**——如果第一次插入没对齐，会主动后退微调再重试

这种纠正行为在人类示教里几乎不存在（人类一般一次对齐就成功了）。这证明 RL 微调确实让策略学到了超越示教数据的能力。

---

## 五、消融实验细节

### 5.1 Advantage 估计方法对比

| 任务 | Value of $s_t$（DPPO） | Value of $(s_t, \mathbf{a}_t^1)$ | $Q(s_t, \mathbf{a}_t^0)$ |
|------|------------------------|----------------------------------|---------------------------|
| Hopper-v2（简单） | 第二 | **最好**（任务简单，Q 估计容易） | 第三 |
| Can（中等） | **最好** | 略差 | 略差 |
| Square（困难） | **明显最好** | 性能明显下降 | 性能更差 |

**结论**：任务越复杂、动作维度越高，"只用 $s_t$ 做 value"的优势越明显。

### 5.2 去噪折扣因子 $\gamma_{\text{denoise}}$ 的影响

| 任务 | 最佳 $\gamma_{\text{denoise}}$ | 解释 |
|------|-------------------------------|------|
| Hopper（简单） | 0.8 最好，0.5 太小学得慢 | 需要精确调节每一步 → $\gamma$ 大一点好 |
| Can（中等） | 0.5–1.0 差不多 | — |
| Square（困难） | 0.5 略好 | 早期去噪步噪声太大，梯度信号差 → $\gamma$ 小一点好 |

$\gamma_{\text{denoise}}$ 可以理解为"对早期去噪步的梯度削减"。

### 5.3 微调去噪步数 $K'$ 的影响

| $K'$ | Can 最终成功率 | Wall-clock time |
|------|---------------|-----------------|
| 3 | 85%（差） | 最快（约 30s/iter） |
| 5 | 95% | — |
| 10 | 100%（最好） | 中等（约 43s/iter）← 最佳性价比 |
| 20 | 100%（和 $K'=10$ 差不多） | 最慢（约 55s/iter） |

### 5.4 噪声调度 $\sigma_{\min}^{\text{exp}}$ 的影响

| 任务 | 最佳设置 | 原因 |
|------|---------|------|
| Can | $\sigma_{\min}^{\text{exp}} = 0.001 \sim 0.1$ 差不多 | 任务较简单 |
| Square | $\sigma_{\min}^{\text{exp}} = 0.1$ 必须设这么高 | 任务更复杂，需要更强的探索噪声才能找到改进方向。如果噪声太小，策略会 over-optimize 已有的少量成功样本，状态覆盖不够，最终陷入局部最优。 |

### 5.5 预训练数据量的影响

| 示教条数 | DPPO 最终成功率 | Gaussian 最终成功率 |
|---------|----------------|-------------------|
| 10 条 | 60% | 20% |
| 25 条 | 80% | 50% |
| 50 条 | 95% | 80% |

观察：
- DPPO 在少数据时优势更大（10 条时差 40%）
- 更多数据缩小了差距但 DPPO 始终领先
- DPPO 用 10 条数据的效果比 Gaussian 用 50 条还好

---

## 六、实现细节（对复现非常重要）

### 6.1 训练超参数

**通用设置**：

| 超参数 | 值 |
|--------|-----|
| $\gamma_{\text{env}}$ | 0.99（Gym）/ 0.999（Robomimic, Furniture） |
| GAE $\lambda$ | 0.95 |
| PPO epochs per iteration | 5–10 |
| Mini-batches | 4（Gym） |
| Batch size | 5000–10000（按 sample 数） |

**DPPO 特有**：

| 超参数 | 值 |
|--------|-----|
| $\gamma_{\text{denoise}}$ | 0.99（Gym）/ 0.9（操作任务） |
| $\sigma_{\min}^{\text{exp}}$ | 0.1（大部分）/ 0.04（Furniture） |
| $\sigma_{\min}^{\text{prob}}$ | 0.1 |
| $K$（去噪步数） | 20 步（state）/ 100 步（pixel，用 DDIM 5 步微调） |
| $K'$（微调步数） | 10 步（state）/ 5 步 DDIM（pixel） |

**PPO clip ratio $\varepsilon$**：

- 范围：0.01（操作任务）~ 0.1（Gym）
- DPPO 的 trick：对不同去噪步用不同的 $\varepsilon$
  - $\varepsilon_{k=K-1} = 0.1 \times \varepsilon_{k=0}$（早期去噪步允许更大变化）
  - 中间步：指数插值
- 选择标准：让 clipping fraction 维持在 10%–20%

### 6.2 网络架构

**State-based（Robomimic）**：

| 组件 | 结构 |
|------|------|
| Actor | 观测 encoder MLP + 时间步 encoder + 去噪 MLP（2 层残差, 512/1024 维） |
| Critic | 独立的 MLP（256×3） |
| 总参数 | 约 2.3M（DPPO-MLP） |

**Pixel-based（Robomimic）**：

| 组件 | 结构 |
|------|------|
| Vision encoder | ViT（patch size 8, 每个 camera 独立编码） |
| Actor head | MLP |
| 总参数 | 约 1M |

**FurnitureBench**：

| 组件 | 结构 |
|------|------|
| Actor | UNet（允许变 chunk size） |
| 参数 | 约 6.8M |
| Gaussian baseline | 需要 10.6M 参数的大 MLP 才能拟合数据（扩散更参数高效） |

### 6.3 Domain Randomization（Sim-to-Real）

- **观测噪声**：测量真实硬件的传感器噪声范围，在仿真中加相同幅度
  - 物体高度估计噪声特别大（AprilTag 在某些角度不准）
  - 添加对应的随机扰动
- **动作噪声**：对 DPPO 采样的动作加 $\mathcal{N}(0, 0.03)$ 随机扰动
  - 模拟不完美的底层控制器
  - DPPO 能扛住这个噪声
  - Gaussian 策略加了同样的噪声后成功率归零
- **初始状态**：Med randomness（物体初始位置随机化）

### 6.4 Wall-clock 时间对比

**Robomimic State（50 并行环境, L40 GPU）**：

| 方法 | Lift | Can | Square | Transport |
|------|------|-----|--------|-----------|
| DRWR | 32.5s | 39.5s | 59.8s | 346.1s |
| DPPO-MLP | 35.2s | 42.0s | 65.6s | 350.3s |
| DPPO-UNet | 83.6s | 92.7s | 130.4s | 431.1s |
| QSM | 64.4s | 72.4s | 107.6s | 391.2s |
| Gaussian-MLP | 27.7s | 35.7s | 56.2s | 255.6s |

DPPO-MLP 比 Gaussian-MLP 慢约 24%（因为多步去噪采样的开销），但比大部分 off-policy 方法（QSM, DIPO）快。

**Pixel（50 并行环境）**：

| 方法 | 每 iteration 时间 |
|------|-------------------|
| DPPO-ViT-MLP | 194.9s |
| Gaussian-ViT-MLP | 153.6s |

差距缩小到 14%（渲染开销是瓶颈，去噪计算占比小了）。

---

## 七、深入理解：为什么 DPPO 在 Sim-to-Real 中特别强

### 7.1 普通 Gaussian PPO 微调后的问题

**现象**：Gaussian 策略在仿真中 88% 成功，真机 0%。视频观察显示动作高频抖动（jittery）。

**原因分析**：

1. PPO 微调让策略学到了很精确但"尖锐"的动作分布
2. 仿真中没有执行器延迟 → 高频动作能被完美执行
3. 真机有带宽限制 → 高频命令被低通滤波 → 实际轨迹偏离预期
4. 偏离后策略没有纠正能力 → 失败

### 7.2 DPPO 天然的平滑性

扩散模型的去噪过程本身就是一个迭代平滑过程：

$$
\mathbf{a}_K \text{（纯噪声, 高频成分多）} \;\to\; \mathbf{a}_{K-1} \text{（去掉一些高频）} \;\to\; \cdots \;\to\; \mathbf{a}_0 \text{（平滑的最终动作）}
$$

类比：去噪就像反复对信号做 low-pass filtering。

### 7.3 DPPO 的纠正行为

BC 预训练的策略是开环的——按照记住的动作模式执行，不会根据反馈调整。

DPPO 微调后，策略学会了闭环的纠正行为：

**场景**：桌腿对准桌面的孔，准备插入

- **BC 策略**：尝试 1 没对齐 → 推了一下 → 失败 → 继续按记忆执行 → 又失败
- **DPPO 策略**：尝试 1 没对齐 → 感知到偏差 → 后退 → 重新调整角度 → 对齐 → 插入成功

这种行为不在人类示教中（人类示教通常是"理想"的一次性操作），是 RL 微调通过自己探索和奖励反馈学到的。

### 7.4 域随机化的配合

DPPO 能有效利用 domain randomization 的训练：

仿真中的 domain randomization 包括：物体质量 ±20%、摩擦系数 ±30%、执行器延迟 ±5ms、观测噪声。

- **DPPO 策略**：因为多步去噪的鲁棒性，能在这些随机化条件下稳定训练 → 学到的策略天然对这些变化鲁棒 → sim-to-real gap 更小
- **Gaussian 策略**：域随机化 + RL 微调 = 双重不稳定性 → 训练经常崩溃 → 不得不减少随机化范围 → 学到的策略对真实世界变化不鲁棒

---

## 八、局限性与展望

### 8.1 当前局限

1. **推理速度**：需要多步去噪，比 Gaussian 策略慢约 2–5x。对于需要 >100Hz 控制频率的任务可能不够快。

2. **从零训练效率**：如果没有 BC 预训练，纯 DPPO 从零开始比 Gaussian PPO 慢约 6x（因为等效 horizon 长了 $K$ 倍）。

3. **探索可能太保守**：结构化的在流形上探索是优点，但对于需要"跳出当前数据分布"的任务（如发现全新的策略模式），可能限制了探索范围。论文中 Lamp Low 任务 Gaussian 略好于 DPPO，可能就是这个原因。

4. **需要好的预训练**：如果 BC 预训练本身很差（数据太少或质量太低），DPPO 能改进的空间也有限。

### 8.2 论文提出的未来方向

1. **大规模预训练 + DPPO 微调**：用大量多任务数据预训练一个通用 Diffusion Policy，然后用 DPPO 针对具体任务微调。预训练的多样性会提供更好的"数据流形"给 DPPO 探索。

2. **视觉策略的 Sim-to-Real**：当前只在 state-based 策略上做了真机实验。pixel-based + DPPO + sim-to-real 是下一步。

3. **和 model-based planning 结合**：用世界模型做高层规划，用 DPPO 微调的 Diffusion Policy 做底层执行。

4. **超越机器人的应用**：药物设计（分子生成 + RL 搜索）、扩散语言模型（文本生成 + 人类反馈）。

---

## 九、和其他工作的定位关系

**时间线**：

| 年份 | 工作 | 定位 |
|------|------|------|
| 2022 | Diffusion Policy 原始论文 | BC 训练 |
| 2023 | DQL, IDQL, DIPO, QSM | Q-learning 类微调尝试 |
| 2024.09 | DPPO（本文） | PPO 微调，证明策略梯度更好 |
| 2025 | D²PPO | 解决 DPPO 的表示坍塌问题 |
| 2025.01 | Online DPRL 综述 | 统一对比确认 DPPO 路线正确 |

**定位**：Diffusion Policy 提供了"预训练基座"，DPPO 提供了"微调方法"，合起来 = 机器人学习的预训练+微调完整 pipeline。

类比 LLM：GPT 预训练 ≈ Diffusion Policy BC；RLHF (PPO) ≈ DPPO。

---

## 十、个人评价与读后感

### 10.1 这篇文章的重要性

这是扩散策略 RL 方向的**范式确立**之作。在它之前，没人知道 PG 方法对扩散策略到底行不行。大家都在折腾复杂的 Q-learning 变体。DPPO 一巴掌打过来说"你们搞复杂了，直接 PPO 就是最好的"。

### 10.2 技术洞察

最深刻的 insight 是：**去噪过程的"多余计算"不是负担，而是资产。** 多步去噪带来了结构化探索、训练稳定性和策略鲁棒性。这三个好处在 RL 微调的场景下价值巨大。

### 10.3 工程价值

对于实际做机器人的团队：如果你已经有一个 Diffusion Policy 的 BC 模型，DPPO 是目前最可靠的提升方式。尤其是需要 sim-to-real 的场景——DPPO 的平滑性和鲁棒性直接转化为真机成功率。

### 10.4 遗留问题

- 推理速度是硬伤，需要 consistency model 或 flow matching 来解决
- 对于没有好预训练的场景（exploration from scratch），DPPO 帮助有限
- PPO clip ratio 的调参经验（尤其是对不同去噪步用不同 $\varepsilon$）需要进一步系统化

---

## 延伸阅读

- **000f_前置知识_为什么扩散策略难以RL微调.md** ← 深度解析 DPPO 的动机：为什么 log π 算不出来、为什么 Q-Learning 路线全部失败、为什么展开去噪链是正确答案
- **000e_前置知识_对数似然与变分下界.md** ← 理解为什么"每步去噪有 log-likelihood"
- **000a_前置知识_策略梯度与PPO.md** ← PPO 的 clip 机制细节
- **000c_前置知识_Diffusion_Policy.md** ← Diffusion Policy 的 action chunk 和采样
- **003_Online_DPRL_综述** ← 把 DPPO 放到更大的算法版图中看
- D²PPO (2025) ← 解决 DPPO 的 representation collapse 问题
- Flow Policy (2025–2026) ← 更快的替代方案
