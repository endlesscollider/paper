---
title: Diffusion Policy
order: 3
tags: [扩散模型, 机器人, 模仿学习]
category: 前置知识
---

# 前置知识：Diffusion Policy（扩散策略）从原理到工程

> **为什么要读这篇**：Diffusion Policy 是 DPPO 要微调的对象、Online DPRL 综述讨论的核心。不懂它，后面的"去噪 MDP"、"action chunk"、"temporal ensemble"都没法理解。  
> **原始论文**: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion  
> **作者**: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song  
> **机构**: Columbia University, MIT, Toyota Research Institute  
> **发表**: IJRR 2024 (arXiv:2303.04137)

**标签**: `#前置知识` `#Diffusion-Policy` `#行为克隆` `#机器人操作` `#动作块` `#多模态策略` `#visuomotor`

---

## 第一部分：Diffusion Policy 要解决什么问题

### 1.1 行为克隆的策略表示问题

行为克隆（BC）的框架很简单：给定状态 s，预测动作 a。问题在于 **用什么函数来表示 π(a|s)**。

$$
选择 1: 确定性策略 a = f_θ(s)
  → 只能输出一个动作
  → 多模态示教数据（同一状态有多种合理动作）→ 学出平均值 → 糟糕

选择 2: 高斯策略 π(a|s) = N(μ_θ(s), σ²)
  → 单峰分布，只能表示一种模式
  → 同上问题

选择 3: GMM π(a|s) = Σᵢ wᵢ × N(μᵢ(s), σᵢ²)
  → 可以多模态！但模式数量要预设
  → 训练不如高斯稳定
  → 高维动作空间中 GMM 表示能力也有限

选择 4: Diffusion Policy π(a|s) = 从噪声去噪 K 步得到 a
  → 可以表示任意复杂的多模态分布
  → 训练稳定（loss 就是 MSE）
  → 高维动作空间中仍然有效
$$

### 1.2 多模态动作分布为什么重要

```text
场景 1: 绕障碍物
  人类示教: 有人走左边，有人走右边
  高斯策略: 学出均值 → 走中间 → 撞障碍

场景 2: 抓取物体
  人类示教: 有人从上方抓，有人从侧面抓
  高斯策略: 学出中间角度 → 抓不到

场景 3: 插入任务
  人类示教: 有人微调后再插，有人直接对齐
  高斯策略: 学出一个"既不微调也不直接"的动作 → 失败

关键: 平均两个正确答案不等于正确答案。
     只有多模态策略才能"选一个模式走到底"。
```

### 1.3 Diffusion Policy 的核心 idea

```text
不预测一个动作点，而是学习动作的整个分布。
用扩散模型的去噪过程来"采样"这个分布。

类比:
  高斯策略: 指着一个点说 "去这里"
  Diffusion Policy: 画出整个可行区域说 "这片区域都行，随便选一个"
```

---

## 第二部分：Action Chunking（动作块预测）

### 2.1 什么是 Action Chunk

传统 BC：每个时间步预测当前一步的动作 a_t。

Diffusion Policy：**一次预测未来 T_a 步的整段动作序列**。

```text
传统: s_t → a_t                          (单步)
DP:   s_t → [a_t, a_{t+1}, ..., a_{t+T_a-1}]  (一段 chunk)

例子 (action_dim=7, T_a=8):
  输入: 当前观测 s_t
  输出: shape [8, 7] 的矩阵 → 未来 8 个控制步的 7 维动作
```

### 2.2 为什么用 Action Chunk

**好处 1：时间一致性**

$$
单步预测:
  t=0: 预测 a₀（基于 s₀）
  t=1: 预测 a₁（基于 s₁）→ 可能和 a₀ 不连贯

chunk 预测:
  t=0: 预测 [a₀, a₁, a₂, ..., a₇] → 一整段是连贯的
  因为网络一次性考虑了整段时间 → 动作序列内部平滑
$$

**好处 2：减少决策频率**

$$
控制频率: 50Hz（每 20ms 一个动作）
chunk size: 8
→ 只需要每 160ms 做一次推理
→ 即使推理慢也不影响实时性

对扩散策略特别重要:
  20 步去噪 × 0.2ms/步 = 4ms → 每次推理需要 4ms
  如果每步都推理: 4ms/20ms = 20% 计算时间用在推理上
  如果 chunk=8: 4ms/160ms = 2.5% → 压力小多了
$$

**好处 3：隐式短期规划**

```text
单步: 只看"当前这一瞬间做什么" → 短视
chunk: 要考虑"未来 8 步怎么走" → 被迫做短期规划

例子:
  要把杯子从 A 移到 B:
  单步策略: 每帧独立决定方向 → 可能走弯路
  chunk 策略: 一次规划 8 步轨迹 → 更直接到达目标
```

### 2.3 Observation Horizon（观测视野）

Diffusion Policy 不仅预测未来 T_a 步动作，还可以接收过去 T_o 帧的观测：

$$
输入: [o_{t-T_o+1}, ..., o_{t-1}, o_t]   → T_o 帧历史观测
输出: [a_t, a_{t+1}, ..., a_{t+T_a-1}]   → T_a 步未来动作

典型值:
  T_o = 2~5（历史观测帧数）
  T_a = 4~16（预测动作步数）
$$

历史观测的作用：帮助推断运动趋势（速度方向、加速度等物理量不在单帧中体现）。

### 2.4 执行策略

预测了 T_a 步的动作块后，怎么用？

**策略 A：全部执行（Open-Loop Chunking）**

$$
t=0:  预测 chunk [a₀, a₁, ..., a₇] → 执行全部 8 步
t=8:  预测新 chunk [a₈, a₉, ..., a₁₅] → 执行全部 8 步
...

优点: 计算最少（每 T_a 步才推理一次）
缺点: 如果中途状态偏了（外部扰动、执行误差），不会纠正
$$

**策略 B：部分执行 + 重新规划（Receding Horizon）**

$$
t=0: 预测 [a₀, a₁, ..., a₇] → 只执行前 T_exec 步（比如前 4 步）
t=4: 预测 [a₄, a₅, ..., a₁₁] → 只执行前 4 步
...

类似 MPC（模型预测控制）的思路:
  规划远一点 → 只执行近一点 → 重新规划

优点: 有机会纠正偏差
缺点: 推理频率更高
$$

**策略 C：Temporal Ensemble（时间集成）**

$$
每步都重新推理，然后对同一时刻的多个预测做加权平均:

t=0: 预测 [a₀, a₁, a₂, ...]
t=1: 预测     [â₁, â₂, â₃, ...]
t=2: 预测         [ã₂, ã₃, ã₄, ...]

对于时刻 t=2 的动作，有三个预测:
  来自 t=0 的预测: a₂
  来自 t=1 的预测: â₂
  来自 t=2 的预测: ã₂

加权融合: a₂_final = w₀×a₂ + w₁×â₂ + w₂×ã₂
权重: 通常用指数衰减，越新的预测权重越大

优点: 最平滑（多次预测的平均减少了随机波动）
缺点: 计算量最大（每步都推理）
$$


---

## 第三部分：Diffusion Policy 的训练

### 3.1 数据来源

```text
人类遥操作采集:
  操作者通过 SpaceMouse / VR 手柄 / 示教器控制机器人
  同步记录: 观测（图像 + 关节角度）和 动作（末端位移或关节指令）
  每条轨迹: 几百到几千步
  数据量: 通常 50-300 条轨迹（相比 LLM 的万亿 token 少得多）
```

### 3.2 训练样本的构造

$$
从轨迹中滑动窗口切出训练样本:

轨迹: o₀, a₀, o₁, a₁, o₂, a₂, ..., o_N, a_N

样本 1: 观测=[o₀, o₁], 动作块=[a₁, a₂, ..., a₈]     (T_o=2, T_a=8)
样本 2: 观测=[o₁, o₂], 动作块=[a₂, a₃, ..., a₉]
样本 3: 观测=[o₂, o₃], 动作块=[a₃, a₄, ..., a₁₀]
...

数据预处理:
  - 观测和动作归一化到 [0,1] 或 [-1,1]（用 min/max statistics）
  - 图像: resize + 可选的 random crop augmentation
$$

### 3.3 训练过程（和标准 DDPM 一样）

$$
Diffusion Policy Training:
  repeat:
    (观测, 动作块) ~ 数据集           # 采样一条训练样本
    k ~ Uniform(1, ..., K)           # 随机选时间步
    ε ~ N(0, I)                       # 采样噪声（和动作块同维度）
    a_k = √ᾱ_k × 动作块 + √(1-ᾱ_k) × ε    # 给动作块加噪
    ε̂ = ε_θ(a_k, k, 观测)            # 网络预测噪声（条件是观测）
    Loss = MSE(ε, ε̂)                 # 均方误差
    θ ← θ - lr × ∇_θ Loss
  until converged
$$

**训练时间参考**：
```text
Robomimic (state input, 300 demos):  8000 epochs, 几个小时
Robomimic (pixel input, 100 demos):  8000 epochs, 十几个小时
FurnitureBench (50 demos):           8000 epochs, 几个小时
```

### 3.4 推理过程

$$
Diffusion Policy Inference:
  输入: 当前观测窗口 [o_{t-T_o+1}, ..., o_t]
  
  a_K ~ N(0, I)   # shape = [T_a, action_dim]，从纯噪声开始
  
  for k = K, K-1, ..., 1:
    ε̂ = ε_θ(a_k, k, 观测)     # 条件去噪一步
    μ = (1/√α_k) × (a_k - β_k/√(1-ᾱ_k) × ε̂)
    if k > 1:
      a_{k-1} = μ + σ_k × z,  z~N(0,I)
    else:
      a_{k-1} = μ
  
  输出: a₀ = [a_t, a_{t+1}, ..., a_{t+T_a-1}]  # 未来 T_a 步动作
$$

---

## 第四部分：为什么 Diffusion Policy 比其他 BC 方法强

### 4.1 论文的实验结果

在 12 个任务、4 个 benchmark 上对比：

```text
Benchmark 1: Robomimic (4 任务: Lift, Can, Square, Transport)
Benchmark 2: Push-T (一个物体推到 T 形区域)
Benchmark 3: Block Pushing (多模态推物体)
Benchmark 4: Kitchen (多阶段厨房操作)

对比方法:
  - Gaussian BC (MSE loss 回归)
  - GMM BC (高斯混合模型)
  - IBC (Implicit BC，能量模型)
  - BET (Behavior Transformer)
  - Diffusion Policy (本文)

结果摘要:
  平均提升: 46.9%（相对最强 baseline）
  
  特别是多模态任务（Push-T, Block Pushing）:
    Gaussian BC: ~40% 成功率
    Diffusion Policy: ~85% 成功率
    → 多模态优势巨大
  
  精细操作任务（Square Nut Assembly）:
    之前方法: 30-50%
    Diffusion Policy: 85%+
```

### 4.2 为什么性能提升这么大

```text
原因 1: 多模态处理
  同一个观测 → 扩散模型能从分布中采样不同模式
  → 不会出现 "平均两种模式" 的问题

原因 2: 高维建模能力
  action chunk 是高维的（如 7×16=112 维）
  高斯/GMM 在这个维度下假设独立性 → 丢失维度间关系
  扩散的去噪过程天然建模维度间的依赖

原因 3: 训练稳定性
  MSE loss 对噪声预测 → 没有对抗训练的不稳定性
  → 可以用少量数据（50-300 条）稳定训练

原因 4: Action chunk + 多步去噪的协同
  chunk 提供时间一致性
  多步去噪提供迭代精化
  两者结合 → 高精度 + 高平滑度的动作输出
```

### 4.3 Diffusion Policy vs ACT（Action Chunking Transformer）

$$
ACT 的策略表示: CVAE (Conditional VAE)
  - 用 latent z 来建模多模态性
  - 训练: encoder 从真实动作推断 z，decoder 从 z+观测 生成动作
  - 推理: z 从先验采样 → decoder 生成动作

对比:
  多模态能力: ACT 靠 z 的不同取值表示不同模式
              DP 靠去噪过程的随机性表示不同模式
  
  训练 loss: ACT = 重建 loss + KL(q(z|action) || p(z))
             DP = 纯 MSE（更简单）
  
  推理速度: ACT 快（一次前向传播）
            DP 慢（K 次前向传播）
  
  表达力: DP > ACT（扩散模型理论上可以表示任意分布）
  
  RL 微调: ACT 几乎不可能（CVAE 的似然更难处理）
           DP 困难但可行（DPPO 解决了这个问题）

实践中选择:
  推理速度要求高 + 数据量少 → ACT
  需要最强表达力 + 后续想 RL 微调 → Diffusion Policy
$$

---

## 第五部分：实现细节与工程考量

### 5.1 关键超参数

```text
动作块大小 T_a:
  太小(1-2): 没有时间一致性的好处
  太大(32+): 预测太远的未来不准确
  典型值: 4-16
  
  任务越精细（插入）→ 用更大的 T_a（需要更长的规划）
  任务越简单（推物体）→ 较小的 T_a 就够

观测视野 T_o:
  典型值: 2-5
  太大: 网络输入过长，训练慢
  太小: 可能丢失运动趋势信息

去噪步数 K:
  state input: 20 步通常够
  pixel input: 100 步（更复杂的分布需要更多步）
  
  DPPO 微调时可以用 DDIM 压缩到 5 步
```

### 5.2 数据归一化

$$
非常重要！扩散模型假设数据分布在一个有界区间内。

归一化方式:
  min-max: x_norm = (x - x_min) / (x_max - x_min)
  → 映射到 [0, 1]

  或者映射到 [-1, 1]:
  x_norm = 2 × (x - x_min) / (x_max - x_min) - 1

统计量来源:
  从训练数据集计算 min/max
  保存下来，推理时用同样的参数反归一化输出

不归一化会怎样:
  - 如果动作量级差异大（有的维度 ±0.01，有的 ±10）
  - 扩散模型的固定噪声调度对所有维度一视同仁
  - 小量级维度会被噪声淹没 → 学不好
$$

### 5.3 EMA（指数移动平均）

```text
训练时维护一个 EMA 版本的网络参数:
  θ_ema ← 0.995 × θ_ema + 0.005 × θ

推理时用 θ_ema 而不是 θ:
  EMA 平滑了训练中的参数波动 → 推理更稳定
  
这在扩散模型训练中是标准做法（从 DDPM 原文就有）。
```

### 5.4 Pixel Input 的处理

$$
图像 → CNN/ViT 编码器 → 特征向量 → 和 proprioception 拼接 → 作为 condition

典型 pipeline:
  相机图像 [84×84×3 或 96×96×3]
  → ResNet-18 / ViT (patch=8)
  → spatial softmax / learned embedding (降维到 ~64-256 维)
  → concat with proprioception (关节角度等)
  → 作为 condition 注入去噪网络

数据增强:
  Random Shift: 随机平移图像几个像素 → 对平移不变性很重要
  Color Jitter: 可选，帮助 Sim-to-Real
  
  注意: 预训练和微调要用相同的增强（否则分布不匹配）
$$

---

## 第六部分：Diffusion Policy 的局限性

### 6.1 推理速度

$$
核心瓶颈: 每次出动作需要 K 步去噪

以 K=20, 网络推理 0.2ms/步 为例:
  总推理时间: 4ms
  
  50Hz 控制 → 每 20ms 需要一个动作
  4ms / 20ms = 20% 的时间在推理 → 还行

  但如果 K=100:
  总推理时间: 20ms → 刚好用完一个控制周期 → 太紧了

  加上 action chunk (T_a=8) 缓解了这个问题:
  每 8 步推理一次 → 有效推理间隔 = 8×20ms = 160ms
  20ms/160ms = 12.5% → OK

缓解措施:
  - DDIM 压缩步数
  - 更小的网络（MLP vs UNet）
  - Action chunk 减少推理频率
  - 未来: Consistency Model (1步), Flow Matching (4-10步)
$$

### 6.2 无法直接做 RL 微调

$$
策略梯度需要 log π(a|s):
  高斯策略: 直接有公式 ✓
  Diffusion Policy: 边际概率 ∫...da₁...da_K 不可解析 ✗

2023 年之前的状态:
  "Diffusion Policy 只能做 BC，不能做 RL"

2024.09 DPPO:
  "展开去噪链为 MDP，每步有高斯似然，可以做 PPO"
  → 解决了这个问题
$$

### 6.3 开环执行的脆弱性

```text
Diffusion Policy 预测的是开环动作块:
  "未来 8 步应该这样走"

如果执行中途发生意外（物体滑动、外部碰撞）:
  策略不知道出事了 → 继续执行原来的计划 → 失败

缓解:
  - Temporal ensemble（每步重新预测 → 隐式闭环）
  - Receding horizon（只执行前几步 → 经常重新规划）
  - DPPO 微调后的策略学会了 corrective behavior（纠正行为）
```

### 6.4 数据质量依赖

```text
BC 的天花板 = 示教数据的质量

如果示教数据:
  - 人类操作者技术差 → 策略也学得差
  - 覆盖不全（只在特定初始条件下采集）→ 泛化差
  - 有人类的坏习惯（犹豫、多余动作）→ 策略也会犹豫

Diffusion Policy 不能超越数据质量上限。
只有 RL 微调（DPPO）才能突破这个天花板。
```

---

## 第七部分：和后续工作的关系

$$
Diffusion Policy (2023) → DPPO (2024):
  DP 是 BC 基座 → DPPO 在上面做 RL 微调 → 超越示教数据
  
Diffusion Policy → 3D Diffusion Policy (2024):
  从 2D 图像 → 3D 点云作为输入 → 更好的空间理解

Diffusion Policy → Scaling (2024):
  网络从 ~2M 参数 → 1B 参数 → 性能随规模提升

Diffusion Policy → Flow Matching Policy (2025-2026):
  用 Flow Matching 替代 DDPM → 更快的推理 + 更简洁的理论

Diffusion Policy + VLA (2025-2026):
  大型 Vision-Language 模型作为理解层
  Diffusion Policy 作为动作生成层
  → 语义理解 + 精细控制 的结合
$$

---

## 第八部分：思考题

1. 如果 T_a=1（chunk size 为 1），Diffusion Policy 退化成什么？还有多模态优势吗？
   → 还有！多模态来自去噪过程的随机性，不是来自 chunk。
   → 但失去了时间一致性的优势。

2. Temporal ensemble 和 MPC 有什么异同？
   → 都是"规划远一点，只执行近一点"
   → 区别: MPC 有显式的 dynamics model，DP 的 "模型" 是隐式的（在去噪网络中）

3. 为什么 Diffusion Policy 用 MSE 预测噪声就能学到多模态分布？
   → 关键: 同一个 x_k 在不同 x₀ 下对应不同的 ε。
   → 网络学到的是 E[ε|x_k]，即条件期望。
   → 在高噪声（k大）时，条件期望趋于 0 → 不提供模式信息。
   → 在低噪声（k小）时，条件期望指向最近的模式。
   → 所以去噪从"没有方向"逐步收敛到"某一个模式"。

4. 为什么 DPPO 论文说"微调 5 步 DDIM 就够"？
   → BC 预训练已经学好了动作分布的大结构
   → 微调只需要在这个结构上做小调整
   → 5 步足以表达从预训练分布到微调分布的偏移

5. 如果没有 action chunk（T_a=1），DPPO 的两层 MDP 会怎样？
   → 还是 work，但失去了 chunk 的时间一致性
   → DPPO 论文中 T_a=1 的从零训练实验确认了这一点
