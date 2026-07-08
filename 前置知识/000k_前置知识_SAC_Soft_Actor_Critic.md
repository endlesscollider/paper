---
title: SAC (Soft Actor-Critic)
order: 11
tags: [强化学习]
category: 前置知识
---

# 前置知识：SAC (Soft Actor-Critic)——连续动作空间的 Off-Policy RL

> **一句话**：SAC 是连续动作空间中最常用的 off-policy 强化学习算法。它在标准 RL 目标（最大化累积奖励）上加了一个"最大熵"正则项，让策略在学习最优行为的同时保持适当的随机性，兼顾探索与利用。

**前置概念**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — On-Policy 路线的对比

---

## 贯穿全文的例子

> 一个 7 自由度的机械臂需要学习从桌面抓起一个方块并放到目标位置。
>
> - **状态** $s \in \mathbb{R}^{20}$：7 个关节角度 + 7 个关节速度 + 方块 3D 位置 + 方块 3D 朝向
> - **动作** $a \in \mathbb{R}^7$：7 个关节的力矩指令，每个维度范围 $[-1, 1]$
> - **奖励**：$r = -\|p_{\text{hand}} - p_{\text{block}}\| + 10 \cdot \mathbb{1}[\text{抓住}] + 20 \cdot \mathbb{1}[\text{放到目标}]$
>
> 我们用 SAC 来训练一个小型策略网络（3 层 MLP，256 维隐藏层），让它学会完成这个抓取-放置任务。

---

## 一、SAC 在 RL 算法家族中的位置

### 1.1 On-Policy vs Off-Policy

RL 算法分为两大家族：

| 类型 | 代表算法 | 数据使用方式 | 样本效率 | 稳定性 | 并行性 |
|------|---------|------------|---------|--------|--------|
| On-Policy | PPO, TRPO, A2C | 只能用当前策略采的数据，用完即弃 | 低 | 高 | 好（天然可并行） |
| Off-Policy | **SAC**, TD3, DDPG | 可以复用历史所有数据（Replay Buffer） | **高** | 中-高 | 一般 |

**On-Policy 的浪费**：PPO 每次收集一批数据 → 更新一次策略 → 旧数据全部作废。这在环境交互昂贵时非常浪费。

**Off-Policy 的节约**：SAC 把所有历史数据存在 Replay Buffer 中，反复抽样训练。一次环境交互可以被利用几十甚至上百次。

**类比**：
- On-Policy 像"背了就忘"的考试——每次考完把题目扔了，下次重新背新题
- Off-Policy 像"建错题本"——过去做过的每道题都存着，随时可以翻出来复习

### 1.2 SAC 相对于其他 Off-Policy 方法的优势

在 SAC 之前，连续动作空间的 off-policy 方法主要是：
- **DDPG** (2015)：确定性策略 + OU 噪声探索 → 探索不充分，对超参敏感（详见 [DDPG 前置知识](/前置知识/000p_前置知识_DDPG_确定性策略梯度)）
- **TD3** (2018)：修复了 DDPG 的过估计问题 → 更稳定，但探索仍靠外加噪声（详见 [TD3 前置知识](/前置知识/000q_前置知识_TD3)）

SAC 的核心创新是**把探索内建到目标函数中**——通过最大化策略的熵，让探索成为优化目标的一部分，而不是靠手动加噪声。

---

## 二、最大熵强化学习：SAC 的核心思想

### 2.1 标准 RL 目标的局限

标准 RL 最大化累积奖励：

$$
J_{\text{standard}}(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T \gamma^t r_t\right]
$$

问题：策略可能**过早收敛到一个确定性的局部最优**，不再探索其他可能。

**在我们的例子中**：
- 机械臂发现"从方块正上方垂直下压"能成功抓取
- 策略迅速收敛到这个模式，每次都输出相同的关节力矩序列
- 但如果方块位置稍有变化（偏了 2cm），这个固定模式就失败了
- 策略从未探索过"先侧移再下压"、"从侧面推到手下再抓"等替代方案

### 2.2 最大熵目标

SAC 在目标中加入策略熵作为奖励：

$$
J_{\text{SAC}}(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T \gamma^t \Big(r_t + \alpha \cdot \mathcal{H}\big(\pi(\cdot|s_t)\big)\Big)\right]
$$

**为什么需要这个公式**：我们希望策略不仅能获得高奖励，还要保持"行为多样性"——在同样好的多种策略中，选择最随机的那个。这样既保证了性能，又保留了应对新情况的灵活性。

**逐项拆解**：
- $r_t$：环境给的即时奖励
- $\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s_t)]$：策略在状态 $s_t$ 的熵（随机性的度量）
- $\alpha > 0$：温度系数，控制"获得高奖励"和"保持随机性"之间的权衡
- 熵越大 → 策略越随机（各方向动作概率均匀）→ 被奖励
- 熵越小 → 策略越确定（只输出一种动作）→ 被惩罚

**代入数字**：

状态：方块在机械臂正下方。有两种策略：

策略 A（确定性）：
- 每次都输出 $a = [0.5, 0.3, -0.8, 0.1, -0.2, 0.4, 0.0]$
- 期望 reward = 8.0
- 熵 $\mathcal{H} \approx 0$（几乎没有随机性）
- SAC 目标（$\alpha=0.2$）：$8.0 + 0.2 \times 0 = 8.0$

策略 B（随机性）：
- 输出以 $[0.5, 0.3, -0.8, 0.1, -0.2, 0.4, 0.0]$ 为均值、标准差 0.1 的高斯分布
- 期望 reward = 7.6（因为有时随机偏离最优轨迹，略低）
- 熵 $\mathcal{H} = \frac{7}{2}(1 + \ln(2\pi \times 0.01)) \approx 2.8$
- SAC 目标：$7.6 + 0.2 \times 2.8 = 8.16$

**策略 B 的 SAC 目标更高！** 虽然它平均奖励略低（7.6 vs 8.0），但它保持了探索能力。SAC 会选择策略 B，因为它在"够好"的同时最大化了行为多样性。

### 2.3 最大熵的三个好处

1. **持续探索**：策略不会过早坍缩为确定性行为，始终保持对新状态的适应能力
2. **多模态行为**：如果"左绕"和"右绕"都能绕过障碍物到达目标，最大熵策略会保持两种模式（而非只记住一种）
3. **鲁棒性**：训练中保持的随机性相当于一种隐式的 domain randomization，让策略对状态扰动更鲁棒

---

## 三、SAC 的完整算法

### 3.1 五个网络组件

SAC 需要维护以下网络：

| 网络 | 符号 | 输入 → 输出 | 作用 |
|------|------|-----------|------|
| Actor（策略） | $\pi_\theta(a|s)$ | $s \to (\mu, \sigma)$ | 决定做什么动作（输出高斯分布参数） |
| Critic 1 | $Q_{\phi_1}(s, a)$ | $(s, a) \to \mathbb{R}$ | 评估"在状态 $s$ 做动作 $a$ 有多好" |
| Critic 2 | $Q_{\phi_2}(s, a)$ | $(s, a) \to \mathbb{R}$ | 同上（双 Q，取最小值防止过估计） |
| Target Critic 1 | $Q_{\bar{\phi}_1}$ | 同上 | Critic 1 的慢更新副本（稳定训练用） |
| Target Critic 2 | $Q_{\bar{\phi}_2}$ | 同上 | Critic 2 的慢更新副本 |

**为什么要两个 Critic？**

Q-learning 有系统性的**过估计偏差**——因为 Bellman 目标中取了 max/期望操作，噪声会被放大为正偏差。双 Q 取最小值可以有效缓解：

$$
Q_{\text{target}} = \min(Q_{\bar{\phi}_1}(s', a'), Q_{\bar{\phi}_2}(s', a'))
$$

这个 trick 来自 TD3 算法（Fujimoto et al., 2018）。

### 3.2 Actor：高斯策略 + 重参数化

SAC 的 Actor 输出一个对角高斯分布：

$$
\pi_\theta(a|s) = \mathcal{N}\big(\mu_\theta(s), \;\text{diag}(\sigma_\theta(s))^2\big)
$$

**采样动作**时使用重参数化技巧（reparameterization trick）：

$$
a = \tanh\Big(\mu_\theta(s) + \sigma_\theta(s) \odot \epsilon\Big), \quad \epsilon \sim \mathcal{N}(0, I)
$$

**为什么需要重参数化**：直接从分布中"随机采样"这个操作不可微分——梯度无法穿过随机采样。重参数化把随机性挪到了外部噪声 $\epsilon$，让输出 $a$ 关于网络参数 $\theta$ 可微，从而可以用梯度下降优化。

**为什么用 $\tanh$**：把动作压缩到 $(-1, 1)$ 范围内，匹配物理系统的有界动作空间。

**在我们的例子中**：
- 网络输入：$s \in \mathbb{R}^{20}$（关节状态 + 方块位姿）
- 网络输出：$\mu \in \mathbb{R}^7$（均值）、$\log\sigma \in \mathbb{R}^7$（对数标准差）
- 采样：$a = \tanh(\mu + e^\sigma \odot \epsilon)$，得到 7 维关节力矩

### 3.3 Critic 更新：Soft Bellman 方程

标准 Bellman 方程：
$$Q(s, a) = r + \gamma \cdot Q(s', a')$$

**Soft Bellman 方程**（加了熵项）：

$$
Q(s, a) = r + \gamma \cdot \mathbb{E}_{a' \sim \pi}\Big[Q(s', a') - \alpha \log \pi(a'|s')\Big]
$$

**为什么加 $-\alpha \log \pi$**：回忆 SAC 的目标是最大化 "奖励 + 熵"。所以在计算未来价值时，不仅考虑奖励还要考虑未来状态的熵。$-\log\pi(a'|s')$ 就是熵的"采样估计"——如果 $a'$ 的概率越低（$\log\pi$ 越负），说明策略越随机，未来的熵越大，value 越高。

**Critic 的训练 Loss**：

$$
L_Q(\phi_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[\Big(Q_{\phi_i}(s,a) - y\Big)^2\right]
$$

其中 target 为：

$$
y = r + \gamma \Big(\min\big(Q_{\bar{\phi}_1}(s', \tilde{a}'),\; Q_{\bar{\phi}_2}(s', \tilde{a}')\big) - \alpha \log \pi_\theta(\tilde{a}'|s')\Big), \quad \tilde{a}' \sim \pi_\theta(\cdot|s')
$$

**代入数字**：
- 当前状态 $s$：手在方块上方 5cm
- 动作 $a$：向下运动（关节力矩使末端下降）
- 奖励 $r = -0.05$（距离惩罚：还没抓到）
- 下一状态 $s'$：手在方块上方 2cm（更近了）
- 下一动作 $\tilde{a}' \sim \pi_\theta(\cdot|s')$
- $\min(Q_{\bar{\phi}_1}(s', \tilde{a}'), Q_{\bar{\phi}_2}(s', \tilde{a}')) = 15.0$
- $-\alpha\log\pi_\theta(\tilde{a}'|s') = 0.4$
- $y = -0.05 + 0.99 \times (15.0 + 0.4) = -0.05 + 15.25 = 15.20$
- 如果当前 $Q_\phi(s, a) = 12.0$，则 loss = $(12.0 - 15.20)^2 = 10.24$ → 推动 $Q$ 上升

### 3.4 Actor 更新：最大化 Q + 熵

Actor 的目标是找到使 "Q 值 + 熵" 最大的动作分布：

$$
L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{B}}\left[\mathbb{E}_{a \sim \pi_\theta}\Big[\alpha \log \pi_\theta(a|s) - Q_\phi(s, a)\Big]\right]
$$

**为什么是这个形式**：
- 最小化 $\alpha\log\pi$ → 最大化熵（让策略更随机）
- 最小化 $-Q$ → 最大化 Q 值（让动作更好）
- 两者加起来 → 找到"既好又随机"的策略

通过重参数化，梯度可以直接反传：

$$
\nabla_\theta L_\pi \approx \nabla_\theta\Big[\alpha\log\pi_\theta(a|s) - Q_\phi(s, a)\Big]\Big|_{a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon)}
$$

### 3.5 温度系数 $\alpha$ 的自动调节

手动设 $\alpha$ 很难——不同任务、不同训练阶段需要不同的值。SAC 通过以下目标自动调节：

$$
L(\alpha) = \mathbb{E}_{a \sim \pi_\theta}\left[-\alpha \Big(\log\pi_\theta(a|s) + \bar{\mathcal{H}}\Big)\right]
$$

其中 $\bar{\mathcal{H}}$ 是目标熵，通常设为 $-\dim(\mathcal{A})$（负动作维度）。

**直觉**：
- 如果当前熵 $> |\bar{\mathcal{H}}|$（策略太随机）→ 减小 $\alpha$（减少熵奖励，让策略更确定）
- 如果当前熵 $< |\bar{\mathcal{H}}|$（策略太确定）→ 增大 $\alpha$（增加熵奖励，鼓励探索）

**在我们的例子中**：$\dim(\mathcal{A}) = 7$，所以 $\bar{\mathcal{H}} = -7$。如果策略的每步熵远低于 7 nats，$\alpha$ 会自动增大逼迫策略多探索。

---

## 四、完整训练循环

```
初始化：
  Actor π_θ, Critic Q_φ1, Q_φ2
  Target Critics Q̄_φ1 ← Q_φ1, Q̄_φ2 ← Q_φ2
  Replay Buffer B（空）
  温度 α（可学习）

循环（每个环境步）:
  1. 采样动作: a = tanh(μ_θ(s) + σ_θ(s) · ε), ε ~ N(0,I)
  2. 执行: s', r, done = env.step(a)
  3. 存入 Buffer: B.push(s, a, r, s', done)
  4. 如果 Buffer 中数据 > 最小数量:
     从 B 中随机抽 mini-batch {(s_i, a_i, r_i, s'_i, d_i)}_{i=1}^{256}
     a. 更新 Critic 1, 2（最小化 Soft Bellman error）
     b. 更新 Actor（最大化 Q - α·log_π）
     c. 更新 α（让熵趋近目标值）
     d. 软更新 Target: φ̄ ← 0.995·φ̄ + 0.005·φ
  5. s ← s'
```

**Replay Buffer 的作用**：

Buffer 存储所有历史 $(s, a, r, s', \text{done})$ 元组（通常容量 100K-1M）。每次训练从中**均匀随机**采 mini-batch。

这就是 off-policy 的核心：**训练用的数据来自过去各种策略版本**。旧策略采的数据照样能用来改善当前策略。

**样本效率对比**：
- PPO：需要 500K 环境步（每步只用一次）
- SAC：需要 50K 环境步（每步平均被重复用 ~10 次）
- 结论：SAC 对环境交互次数的需求低约 10 倍

---

## 五、SAC vs PPO：什么时候用哪个

| 维度 | SAC | PPO |
|------|-----|-----|
| 动作空间 | 连续（必须） | 连续 / 离散 都行 |
| 样本效率 | **高**（off-policy 复用数据） | 低（on-policy 用完即弃） |
| 计算效率 | 中（每步需要 gradient update） | 可以攒一批再更新 |
| 并行化 | 较难 | **好**（天然适合大规模并行仿真） |
| 训练稳定性 | 中-高 | **高** |
| 超参数敏感度 | 中（自动调 α 很关键） | 较低 |
| 适合大模型 | 不太适合 | **适合**（并行采样匹配大模型更新） |
| 适合小网络 | **非常适合** | 过杀（小网络不需要那么多样本） |
| 适合昂贵环境 | **非常适合**（样本效率高） | 不太适合（需要太多交互） |

**经验法则**：
- 策略网络小（< 10M 参数）+ 环境交互贵 → **SAC**
- 策略网络大（> 1B 参数）+ 仿真并行度高 → **PPO**

---

## 六、SAC 的常见超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| 学习率（Actor） | 3e-4 | Adam optimizer |
| 学习率（Critic） | 3e-4 | 通常和 Actor 相同 |
| 折扣因子 $\gamma$ | 0.99 | 越接近 1，越关注长期收益 |
| Soft update $\tau$ | 0.005 | Target network 的更新速度 |
| Replay Buffer 大小 | 1M | 太小会忘记重要经验 |
| Batch size | 256 | 每次更新从 Buffer 抽多少 |
| 初始探索步数 | 10K | 开始训练前的纯随机探索 |
| 目标熵 $\bar{\mathcal{H}}$ | $-\dim(\mathcal{A})$ | 动作维度的负数 |

---

## 七、总结

| 概念 | 说明 |
|------|------|
| SAC 是什么 | 连续动作空间的 off-policy Actor-Critic 算法 |
| 核心创新 | 最大熵目标：优化 "奖励 + α × 策略熵" |
| 关键组件 | Actor(高斯策略) + 双Critic + 自动α |
| 样本效率 | 高（Replay Buffer 复用数据） |
| 适用场景 | 连续动作 + 小/中型网络 + 样本昂贵 |
| 不适用场景 | 离散动作空间、需要大规模并行训练的超大模型 |

---

## 延伸阅读

- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — On-Policy 路线对比
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — 熵和 KL 散度的数学关系
