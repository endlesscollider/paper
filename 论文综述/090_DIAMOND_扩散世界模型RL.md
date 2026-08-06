---
title: DIAMOND：扩散模型作为世界模型
order: 290
tags: [世界模型, 强化学习, 扩散模型, Atari, NeurIPS, 像素预测, Model-Based RL]
category: 精读
star: 4
---

# DIAMOND：Diffusion for World Modeling 深度精读

> **论文标题**: Diffusion for World Modeling: Visual Details Matter in Atari  
> **作者**: Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, François Fleuret  
> **机构**: University of Geneva, Microsoft Research, University of Edinburgh  
> **发表**: NeurIPS 2024 (Spotlight)  
> **代码**: https://github.com/eloialonso/diamond

**标签**: `#世界模型` `#扩散模型` `#Model-Based RL` `#Atari` `#像素预测` `#NeurIPS2024`

**知识链接**：
- [世界模型基础](/前置知识/000t_前置知识_世界模型基础) — 世界模型概念入门
- [扩散模型 DDPM](/前置知识/000b_前置知识_扩散模型DDPM) — 扩散模型的核心原理
- [世界模型强化学习综述](/论文综述/S10_世界模型强化学习综述) — 全景对比
- [DreamerV3 精读](/论文综述/089_DreamerV3_通用世界模型RL) — 主要对比方法
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — Actor-Critic 基础

---

## 一、背景与动机

### 1.1 世界模型 RL 的"视觉质量"问题

DreamerV3 和 IRIS 都取得了很好的成绩，但它们的世界模型有一个共同缺陷：**生成的想象画面不够清晰。**

为什么这是个问题？因为 RL agent 的决策依赖**视觉细节**：

| 游戏 | 关键视觉细节 | 模糊的后果 |
|------|-------------|-----------|
| Breakout | 球的精确位置（1-2 像素） | 挡板错位 → 失球 |
| Pong | 球的速度和方向（像素级） | 接不住球 |
| Boxing | 对手的手臂位置 | 判断不了何时出拳 |
| Freeway | 多辆车的精确横坐标 | 过马路被撞 |

**DreamerV3 的问题**：在潜空间操作 → 解码器重建画面时会模糊化细节。

**IRIS 的问题**：VQ-VAE 的 16 个 token 承载不了全部像素信息 → 量化误差丢失细节。

### 1.2 DIAMOND 的核心洞察

> **Visual details matter.** 扩散模型是目前图像生成质量最高的方法——为什么不直接用它来建模环境转移？

DIAMOND 的思路极其简洁：

$$
\text{下一帧} = \text{Diffusion}(\text{噪声}; \text{条件} = [\text{过去几帧}, \text{动作}])
$$

直接在**像素空间**做条件生成，不经过任何中间表示。

### 1.3 贯穿全文的例子

> **场景**：Atari Breakout。当前状态：球正以 45° 向右下方运动，即将撞到右侧墙壁；挡板在中间偏左。
>
> - **DreamerV3 想象**：知道"球大概在右边"，但具体碰墙后的反弹角度可能有 2-3 像素误差
> - **IRIS 想象**：球的位置被量化到 4×4 网格中的某个 token，丢失了 sub-token 级别的位置信息
> - **DIAMOND 想象**：直接用扩散模型在像素级生成下一帧，球碰墙后精确反弹到对称位置

---

## 二、方法详解

### 2.1 问题建模

环境转移建模为条件概率：

$$
p_\theta(o_{t+1} \mid o_{t-k:t}, a_{t-k:t})
$$

> 一句话：给定过去 $k+1$ 帧观测和对应的动作，预测下一帧的像素分布。

**为什么需要多帧历史？**

单帧不包含"运动信息"——你看一张 Breakout 截图，无法判断球是向上还是向下运动。用 4 帧历史（$k=3$）就能隐式编码速度和加速度信息。

### 2.2 条件扩散模型

DIAMOND 用 DDPM（去噪扩散概率模型）来参数化上述条件概率。

**前向加噪过程**（训练时用）：

$$
q(o_{t+1}^{(n)} \mid o_{t+1}^{(0)}) = \mathcal{N}(o_{t+1}^{(n)}; \sqrt{\bar{\alpha}_n} \cdot o_{t+1}^{(0)}, (1 - \bar{\alpha}_n) \mathbf{I})
$$

> 一句话：对真实的下一帧逐步加高斯噪声，加 $n$ 步后得到不同噪声程度的版本。

**反向去噪过程**（推理时用）：

从纯噪声 $o_{t+1}^{(N)} \sim \mathcal{N}(0, \mathbf{I})$ 出发，逐步去噪得到清晰的下一帧：

$$
o_{t+1}^{(n-1)} = \frac{1}{\sqrt{\alpha_n}} \left( o_{t+1}^{(n)} - \frac{1-\alpha_n}{\sqrt{1-\bar{\alpha}_n}} \epsilon_\theta(o_{t+1}^{(n)}, n, c) \right) + \sigma_n \mathbf{z}
$$

其中 $c = [o_{t-k:t}, a_{t-k:t}]$ 是条件信息，$\epsilon_\theta$ 是 U-Net 噪声预测网络。

**逐项拆解**：
- $o_{t+1}^{(n)}$：第 $n$ 步的带噪图像
- $\epsilon_\theta(o_{t+1}^{(n)}, n, c)$：U-Net 预测"这一步加了什么噪声"
- $\frac{1-\alpha_n}{\sqrt{1-\bar{\alpha}_n}}$：噪声缩放系数，确保正确的去噪步长
- $\sigma_n \mathbf{z}$：采样噪声，保持随机性（环境本身可能是随机的）

**训练目标**（简化的噪声预测损失）：

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{n, o_{t+1}^{(0)}, \epsilon} \left[ \| \epsilon - \epsilon_\theta(o_{t+1}^{(n)}, n, [o_{t-k:t}, a_{t-k:t}]) \|^2 \right]
$$

> 一句话：给 U-Net 看一张加了噪声的图片 + 历史帧和动作，让它猜出加的是什么噪声。猜得越准，说明对环境动态理解得越好。

**逐项拆解**：
- $n \sim \text{Uniform}(1, N)$：随机选一个噪声级别
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：真正加的噪声（训练时已知）
- $\epsilon_\theta(\cdots)$：网络预测的噪声
- 期望对噪声级别 $n$、真实下一帧 $o_{t+1}^{(0)}$、以及噪声 $\epsilon$ 三者求平均。实际用 mini-batch 近似

**数值例子**（Breakout 中的一次训练步）：

1. 从 replay buffer 采一个 transition：4 帧历史 + 动作"左移" + 下一帧（球碰墙反弹后）
2. 随机选 $n = 500$（总共 1000 步中的中间位置）
3. 给下一帧加 $n=500$ 步的噪声：画面变成"略微能看出有球但很模糊"
4. U-Net 接收：这张模糊图片 + 4 帧历史 + 动作 → 输出它认为加的噪声
5. 计算 MSE loss：如果预测的噪声和真正加的噪声接近 → loss 小 → 模型学到了"球碰墙会反弹"

### 2.3 U-Net 架构细节

DIAMOND 使用标准的 2D U-Net，但有以下适配：

| 组件 | 配置 |
|------|------|
| 输入 | 带噪下一帧 64×64×3 |
| 条件注入 | 历史帧（4×3=12 通道）+ 动作（one-hot 嵌入） |
| 分辨率层级 | 64→32→16→8 |
| 每层 ResBlock 数 | 2 |
| 注意力层 | 在 16×16 和 8×8 分辨率加 self-attention |
| 时间步嵌入 | 正弦位置编码 + MLP 注入每个 ResBlock |
| 动作嵌入 | one-hot → MLP → 和时间步嵌入拼接 |
| 总参数 | ~100M |

**条件注入方式**：历史帧直接在通道维度和带噪图片拼接（3+12=15 通道输入）。动作通过 FiLM 调制（类似 [AdaLayerNorm](/前置知识/001f_前置知识_AdaLayerNorm条件化归一化) 的机制）注入。

### 2.4 推理加速：DDIM

训练用 1000 步 DDPM，推理用 **DDIM 采样 10 步**：

$$
o_{t+1}^{(n-\Delta)} = \sqrt{\bar{\alpha}_{n-\Delta}} \cdot \hat{o}_{t+1}^{(0)} + \sqrt{1 - \bar{\alpha}_{n-\Delta}} \cdot \epsilon_\theta(o_{t+1}^{(n)}, n, c)
$$

10 步 DDIM 而不是 1000 步 DDPM → 推理速度提升 **100 倍**，质量损失很小（FID 从 2.1 升到 3.4）。

### 2.5 奖励和终止预测

DIAMOND 除了预测下一帧，还需要预测奖励和 episode 是否结束：

- **奖励预测器**：一个小 CNN，输入当前帧 $o_t$，输出 $\hat{r}_t \in \mathbb{R}$
- **终止预测器**：一个小 CNN，输入当前帧 $o_t$，输出 $\hat{d}_t \in \{0, 1\}$

这两个网络独立训练，和扩散世界模型分开。

### 2.6 在想象中训练 Agent

DIAMOND 的策略训练流程与 DreamerV3 类似，但有一个关键区别：**想象是在像素空间进行的**。

```python
# 伪代码：DIAMOND 的想象 rollout
imagined_obs = initial_obs_stack  # 4 帧历史（从 buffer 采样）
for t in range(imagine_horizon):
    action = actor(encode(imagined_obs))  # 策略选动作
    next_frame = diffusion_sample(imagined_obs, action, steps=10)  # 10 步 DDIM
    reward = reward_predictor(next_frame)
    done = termination_predictor(next_frame)
    imagined_obs = concat(imagined_obs[1:], next_frame)  # 滑动窗口更新历史
```

**注意**：每一步想象需要 10 次 U-Net 前传（DDIM 10 步）。这比 DreamerV3（1 次 GRU 前传）慢约 10 倍。但每一步的预测质量远高于 DreamerV3。

### 2.7 Actor-Critic 训练

在想象轨迹上用标准的 Actor-Critic 更新：

- Actor：ResNet 编码当前帧 → 输出动作分布
- Critic：ResNet 编码当前帧 → 输出 V 值
- 使用 λ-回报作为 TD 目标
- 策略梯度 + 熵正则

---

## 三、实验结果

### 3.1 Atari 100k Benchmark

DIAMOND 在 26 个 Atari 游戏上的表现：

| 方法 | 类型 | 平均 HNS | 中位 HNS |
|------|------|----------|----------|
| 人类 | — | 1.00 | 1.00 |
| SimPLe (2019) | 世界模型 | 0.44 | 0.14 |
| IRIS (2023) | 世界模型 | 1.05 | 0.50 |
| DreamerV3 (2023) | 世界模型 | 1.21 | 0.67 |
| **DIAMOND (2024)** | **世界模型** | **1.46** | **0.89** |
| EfficientZero (2022) | 世界模型+搜索 | 1.16 | 0.74 |

**关键观察**：
- DIAMOND 比 DreamerV3 高 20%（1.46 vs 1.21）
- 中位数提升更大（0.89 vs 0.67）→ 说明 DIAMOND 在更多游戏上表现一致
- 这是纯世界模型方法（不用额外搜索）的新 SOTA

### 3.2 哪些游戏受益最大

DIAMOND 相对 DreamerV3 提升最大的游戏：

| 游戏 | DreamerV3 HNS | DIAMOND HNS | 提升原因 |
|------|--------------|-------------|---------|
| Boxing | 0.8 | 2.1 | 需要精确判断对手位置 |
| Breakout | 1.5 | 3.2 | 球的像素级位置决定得分 |
| Pong | 1.2 | 1.8 | 精确的碰撞判断 |
| Freeway | 0.3 | 0.9 | 多辆车的精确间距判断 |

**共同特点**：这些游戏都需要**像素级精确的空间推理**。

### 3.3 CS:GO 神经游戏引擎

DIAMOND 的一个额外实验：在 CS:GO 的静态游戏画面上训练扩散世界模型，然后生成交互式的"神经游戏"。

- 训练数据：~16 小时的 CS:GO 游戏录像
- 输入：玩家的键鼠操作
- 输出：逐帧生成第一人称画面

虽然不完美（偶尔有闪烁和不一致），但展示了扩散世界模型作为"通用模拟器"的潜力。

---

## 四、与 DreamerV3 的深入对比

| 维度 | DreamerV3 | DIAMOND |
|------|-----------|---------|
| 操作空间 | 潜空间 (1024-d) | 像素空间 (64×64×3) |
| 生成质量 | 中（VAE 重建模糊） | 高（扩散模型细节清晰） |
| 单步想象速度 | 快（1 次 GRU） | 慢（10 次 U-Net） |
| 长 horizon 稳定性 | 中（15 步后漂移） | 高（像素级预测更准） |
| 适用范围 | 离散+连续，所有观测 | 目前主要验证在离散动作+视觉 |
| 模型大小 | ~30M | ~100M |
| 训练计算量 | 1× | ~3× |
| 推理计算量 | 1× | ~10× |

**什么时候选 DIAMOND？**
- 任务需要像素级精确决策（视觉游戏、精细操作）
- 不在乎推理速度（离线训练场景）
- 观测是图像且分辨率不太高

**什么时候选 DreamerV3？**
- 需要快速推理（实时控制）
- 任务不需要精确视觉细节
- 需要处理多种观测类型（状态向量 + 图像）
- 需要连续动作空间

---

## 五、局限与未来方向

### 5.1 当前局限

1. **推理速度**：每步 10 次 U-Net 前传，在 GPU 上约 50ms/步。对需要 30Hz 实时控制的机器人来说太慢
2. **只验证了离散动作**：连续动作空间的扩展尚未充分验证
3. **分辨率限制**：目前只在 64×64 上验证。更高分辨率需要更大模型
4. **不显式建模状态**：没有像 DreamerV3 那样的潜空间 → 难以做层次化规划

### 5.2 未来方向

1. **潜空间扩散**：在学习的潜空间中做扩散，兼顾质量和速度
2. **一步蒸馏**：用一致性蒸馏把 10 步 DDIM 压缩为 1 步
3. **视频扩散**：一次预测多帧而非逐帧预测
4. **结合规划**：加入 MCTS 或 MPC 做显式多步规划
5. **扩展到机器人**：高保真的像素预测对机器人操作场景尤其有价值

---

## 六、总结

| 维度 | DIAMOND |
|------|---------|
| 核心问题 | 世界模型的视觉质量不够，影响 RL 决策 |
| 核心方案 | 用扩散模型直接在像素空间建模环境转移 |
| 关键贡献 | 首次证明扩散世界模型在 Atari 100k 上 SOTA |
| 意外发现 | 可以作为"神经游戏引擎" |
| 核心代价 | 推理速度约 DreamerV3 的 1/10 |

**最深刻的 insight**：在世界模型 RL 中，视觉生成质量不是"锦上添花"——它直接决定了 agent 的决策质量。用更好的生成模型做世界模型，即使计算更贵，最终性能也更好。

---

## 延伸阅读

- [扩散模型 DDPM](/前置知识/000b_前置知识_扩散模型DDPM) — DIAMOND 的底层技术
- [DreamerV3 精读](/论文综述/089_DreamerV3_通用世界模型RL) — 主要对比方法
- [世界模型强化学习综述](/论文综述/S10_世界模型强化学习综述) — 全景对比五大方法
- [世界模型基础](/前置知识/000t_前置知识_世界模型基础) — 概念入门
- [向量量化与离散表示学习](/前置知识/001q_前置知识_向量量化与离散表示学习) — IRIS (对比方法) 的基础
