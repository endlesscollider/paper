---
title: TD-MPC2：隐式世界模型的可扩展连续控制
order: 291
tags: [世界模型, 强化学习, Model-Based RL, MPC, 连续控制, 隐式模型, 多任务]
category: 精读
star: 4
---

# TD-MPC2：Scalable, Robust World Models for Continuous Control 深度精读

> **论文标题**: TD-MPC2: Scalable, Robust World Models for Continuous Control  
> **作者**: Nicklas Hansen, Hao Su, Xiaolong Wang  
> **机构**: UC San Diego  
> **发表**: ICLR 2024  
> **代码**: https://github.com/nicklashansen/tdmpc2  
> **项目页**: https://tdmpc2.com

**标签**: `#世界模型` `#Model-Based RL` `#MPC` `#连续控制` `#隐式模型` `#多任务` `#ICLR2024`

**知识链接**：
- [世界模型基础](/前置知识/000t_前置知识_世界模型基础) — 世界模型概念入门
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Q 学习核心
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — Actor-Critic 对比
- [世界模型强化学习综述](/论文综述/S10_世界模型强化学习综述) — 全景对比
- [DreamerV3 精读](/论文综述/089_DreamerV3_通用世界模型RL) — 主要对比方法

---

## 一、背景与动机

### 1.1 连续控制的挑战

机器人操作、运动控制等任务的动作空间是**连续**的（如关节角度、力矩）。与 Atari 的离散 4-18 个动作不同，连续控制面临：

- 动作维度高（人形机器人 22 维）
- 需要精确的连续值（不是选择离散选项）
- 物理动力学复杂（接触、摩擦、关节耦合）

DreamerV3 在连续控制上表现不错但不是最强。DIAMOND 和 IRIS 主要在离散动作的 Atari 上验证。

### 1.2 TD-MPC2 的核心思路

TD-MPC2 提出一个激进的简化：**世界模型不需要重建观测，只需要预测回报。**

传统世界模型（如 DreamerV3）：
```
观测 → 编码 → 潜状态 → 解码重建观测（保证潜状态有信息量）
                    └→ 预测奖励
```

TD-MPC2：
```
观测 → 编码 → 潜状态 → 预测奖励（直接优化回报预测）
                    └→ 预测下一个潜状态
                    └→ 预测 Q 值
                    (不解码！不重建！)
```

**为什么可以不重建？** 因为世界模型的最终目的是帮助做决策——而决策只需要知道"哪个动作的回报高"，不需要知道"下一帧长什么样"。

### 1.3 贯穿全文的例子

> **场景**：一个 6 自由度机械臂学习"插入圆柱体"任务。
>
> - 观测：6 个关节角度 + 末端位姿 + 圆柱体位姿 = 约 30 维向量
> - 动作：6 个关节力矩（连续值）
> - 奖励：圆柱体越接近目标槽越高
>
> TD-MPC2 的做法：
> 1. 把 30 维观测编码为 256 维潜状态
> 2. 在潜空间中模拟"如果接下来 5 步这样转动关节，最终插入得有多近？"
> 3. 尝试 512 种不同的力矩序列，选最好的那个执行

---

## 二、方法详解

### 2.1 五个核心组件

TD-MPC2 学习五个神经网络：

$$
\begin{aligned}
\text{编码器:} \quad & z_t = h(o_t) \\
\text{动态模型:} \quad & z_{t+1} = d(z_t, a_t) \\
\text{奖励模型:} \quad & \hat{r}_t = R(z_t, a_t) \\
\text{Q 函数:} \quad & Q(z_t, a_t) \\
\text{策略先验:} \quad & a_t = \pi(z_t) + \epsilon
\end{aligned}
$$

> 一句话：编码器压缩观测，动态模型预测下一步状态，奖励模型和 Q 函数评估动作好坏，策略先验给规划提供初始猜测。

**关键区别**：没有解码器 $\hat{o}_t = g(z_t)$。

**这意味着什么？** 潜状态 $z_t$ 不需要包含"能还原观测所有细节"的信息——它只需要包含"对预测奖励和下一步有用"的信息。这是一种**面向任务的表示学习**。

### 2.2 联合训练目标

所有组件通过一个联合损失优化：

$$
\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \mathcal{B}} \left[ \sum_{t=0}^{H-1} \left( \underbrace{c_1 \cdot \|d(z_t, a_t) - \text{sg}(h(o_{t+1}))\|^2}_{\text{动态一致性}} + \underbrace{c_2 \cdot \|R(z_t, a_t) - r_t\|^2}_{\text{奖励预测}} + \underbrace{c_3 \cdot (Q(z_t, a_t) - y_t)^2}_{\text{TD 损失}} \right) \right]
$$

> 一句话：在 replay buffer 中采样真实轨迹，让动态模型预测的下一步潜状态接近真实下一步的编码，同时让奖励预测和 Q 值预测都准确。

**逐项拆解**：

**(1) 动态一致性损失**：$\|d(z_t, a_t) - \text{sg}(h(o_{t+1}))\|^2$

- 动态模型从 $(z_t, a_t)$ 预测 $\hat{z}_{t+1}$
- 目标是真实下一步观测的编码 $h(o_{t+1})$，但加了 **stop-gradient**
- 为什么 stop-gradient？防止编码器"坍缩"——如果不加，编码器可能学到把所有观测都映射到同一个点，这样动态预测永远是"正确的"（但没有信息量）
- stop-gradient 迫使动态模型去主动追踪编码器的变化，而不是反过来

**(2) 奖励预测损失**：$\|R(z_t, a_t) - r_t\|^2$

- 标准 MSE。让潜空间编码"什么动作在什么状态下能得高分"。

**(3) TD 损失**：$(Q(z_t, a_t) - y_t)^2$

TD 目标：
$$
y_t = r_t + \gamma \cdot Q_{\text{target}}(h(o_{t+1}), \pi(h(o_{t+1})))
$$

- 使用 target 网络（EMA 更新）稳定训练
- 策略先验 $\pi$ 提供下一步动作用于计算 target Q

**数值例子**（机械臂插入任务）：

假设当前编码 $z_t$ = [0.3, -0.1, 0.7, ...]（256 维），动作 $a_t$ = [0.1, -0.2, ...]（6 维力矩）：
- 动态模型预测 $\hat{z}_{t+1}$ = [0.28, -0.08, 0.72, ...]
- 真实下一步编码 $h(o_{t+1})$ = [0.29, -0.09, 0.71, ...]
- 动态一致性 loss = $\|[0.28, -0.08, 0.72, ...] - [0.29, -0.09, 0.71, ...]\|^2$ ≈ 0.001（很小→预测准）

### 2.3 MPC 规划：MPPI 算法

TD-MPC2 在决策时用 **MPPI（Model Predictive Path Integral）** 在线搜索：

$$
a_0^* = \arg\max_{a_{0:H-1}} \left[ \sum_{t=0}^{H-1} \gamma^t R(z_t, a_t) + \gamma^H Q(z_H, \pi(z_H)) \right]
$$

> 一句话：尝试很多动作序列，选使"短期奖励+长期 Q 值"最高的那组。

**MPPI 的具体步骤**：

```python
# 伪代码：TD-MPC2 决策过程
def select_action(obs):
    z = encoder(obs)  # 编码当前观测
    
    # 初始化：用策略先验生成 512 条候选动作序列（每条 H=5 步）
    action_sequences = policy_prior(z, horizon=5) + noise  # [512, 5, action_dim]
    
    for iteration in range(6):  # MPPI 迭代 6 轮
        # 评估每条序列的总价值
        values = []
        for seq in action_sequences:
            z_curr = z
            total_reward = 0
            for t, a in enumerate(seq):
                total_reward += gamma**t * reward_model(z_curr, a)
                z_curr = dynamics_model(z_curr, a)
            total_reward += gamma**H * Q(z_curr, policy_prior(z_curr))
            values.append(total_reward)
        
        # 加权重采样：高价值序列的权重更大
        weights = softmax(values / temperature)
        mean = weighted_average(action_sequences, weights)
        action_sequences = mean + noise  # 重新采样
    
    return mean[0]  # 返回第一步动作
```

**关键设计**：
- H=5 步短 horizon → 动态模型误差不会累积太多
- Q 函数兜底 → 5 步之后的长期回报由 Q 值提供，不需要展开更长
- 策略先验提供好的初始化 → 512 条候选序列不是完全随机的，而是围绕策略先验采样

### 2.4 为什么 TD-MPC2 适合连续控制

| 优势 | 原因 |
|------|------|
| 精确动作 | MPPI 在连续空间中搜索最优动作序列 |
| 鲁棒性 | 每步重新规划→对模型误差有在线纠错能力 |
| 不需要像素重建 | 连续控制的观测通常是低维状态向量 |
| 利用 Q 函数 | 短 horizon 模型 + 长 horizon Q 的组合避免累积误差 |

---

## 三、规模化：单模型 80 任务

### 3.1 多任务训练

TD-MPC2 的一个关键贡献：训练**一个** 317M 参数的模型，同时解决 80 个不同任务。

任务覆盖：
- DMControl：walker、humanoid、dog 等 30 个任务
- MetaWorld：50 个操作任务
- 跨域：不同机器人形态、不同动作维度

**训练方式**：所有任务的数据混合训练，模型内部用 task embedding 区分不同任务。

### 3.2 与单任务专家对比

| 设置 | 平均得分（归一化） |
|------|-------------------|
| 单任务 SAC（每个任务独立训） | 0.78 |
| 单任务 TD-MPC2 | 0.92 |
| **多任务 TD-MPC2（1 个模型 80 任务）** | **0.89** |

多任务模型只比单任务略低 3%——但只需要训练一个模型！这证明了隐式世界模型的可扩展性。

---

## 四、与其他方法的对比

### 4.1 性能对比（DMControl Suite）

| 方法 | 类型 | 1M 步平均分 | 计算量 |
|------|------|------------|--------|
| SAC | Model-Free | 780 | 1× |
| DreamerV3 | MBRL (Actor-Critic) | 890 | 1.5× |
| **TD-MPC2** | **MBRL (MPC)** | **910** | 2× |
| TD-MPC (v1) | MBRL (MPC) | 850 | 2× |

### 4.2 TD-MPC2 vs DreamerV3

| 维度 | DreamerV3 | TD-MPC2 |
|------|-----------|---------|
| 推理方式 | 前传一次出动作 | 每步 MPC 搜索 |
| 推理延迟 | ~1ms | ~20ms |
| 连续控制性能 | 890 | 910 |
| Atari 性能 | 1.21 HNS | 不适用（离散动作） |
| 多任务规模化 | 未充分验证 | 317M 单模型 80 任务 |
| 解码器 | 有（重建观测） | 无 |
| 状态观测支持 | 需要图像 | 原生支持状态向量 |

---

## 五、局限与未来方向

### 5.1 当前局限

1. **推理速度**：MPC 每步需要约 20ms（6 轮 MPPI × 512 候选 × 5 步展开）。对于需要 1000Hz 控制频率的精细操作不够快
2. **不支持离散动作**：MPPI 天然在连续空间搜索，对 Atari 类离散问题不适用
3. **没有长程规划**：H=5 步 + Q 值兜底，对需要几十步规划的长 horizon 任务可能不够
4. **视觉观测支持有限**：虽然有编码器，但主要在状态向量观测上验证

### 5.2 未来方向

1. **加速 MPC**：用策略蒸馏把 MPC 的搜索结果"学进"一个快速策略网络
2. **扩展到视觉**：结合图像编码器，在像素观测任务上验证
3. **更大规模**：1000+ 任务的 foundation model for control
4. **层次化 MPC**：高层长 horizon 规划 + 低层 TD-MPC2 精细控制

---

## 六、总结

| 维度 | TD-MPC2 |
|------|---------|
| 核心问题 | 连续控制需要精确动作 + 鲁棒规划 |
| 核心方案 | 隐式世界模型（无解码器） + MPPI 在线规划 + TD 学习 |
| 关键创新 | "不需要重建观测"的极简设计 + 多任务规模化 |
| 最大突破 | 317M 单模型解 80 个连续控制任务 |
| 核心代价 | MPC 推理慢；不支持离散动作 |

**最深刻的 insight**：世界模型不一定要"看得见未来"（重建像素/观测），只要"算得准回报"（预测 reward/Q 值）就够了。这个极简思路反而在连续控制上取得了最好的结果。

---

## 延伸阅读

- [世界模型基础](/前置知识/000t_前置知识_世界模型基础) — 概念入门
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — TD-MPC2 的核心组件
- [DreamerV3 精读](/论文综述/089_DreamerV3_通用世界模型RL) — Actor-Critic 路线的对比
- [DIAMOND 精读](/论文综述/090_DIAMOND_扩散世界模型RL) — 像素级世界模型的对比
- [世界模型强化学习综述](/论文综述/S10_世界模型强化学习综述) — 全景对比
- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 连续控制的 Model-Free 基线
