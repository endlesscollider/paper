---
title: Replay Buffer（经验回放）
order: 18
tags: [强化学习]
category: 前置知识
---

# 前置知识：Replay Buffer（经验回放）

> **一句话**：Replay Buffer 是一个存储历史交互经验的数据结构，让 off-policy RL 算法能够反复利用过去的数据进行训练，大幅提升样本效率。

**前置概念**：
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Replay Buffer 中的数据用来训练 Q 函数

---

## 贯穿全文的例子

> 一个机械臂学习抓取。每次和环境交互产生一条经验 $(s_t, a_t, r_t, s_{t+1}, \text{done})$。
> - $s_t$：当前观测
> - $a_t$：执行的动作
> - $r_t$：获得的奖励
> - $s_{t+1}$：下一个状态
> - done：是否结束

---

## 一、为什么需要 Replay Buffer

### 1.1 On-Policy 的浪费

On-policy 算法（如 [PPO](/前置知识/000a_前置知识_策略梯度与PPO)）要求训练数据必须来自当前策略。策略一更新，旧数据就"过期"了：

```
收集 1000 步数据 → 更新策略 → 旧数据作废 → 重新收集……
```

每条数据只用一次就扔掉。如果环境交互很昂贵（物理机器人每步要几百毫秒），这种浪费是不可接受的。

### 1.2 Off-Policy 的复用

Off-policy 算法（如 [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)、[DDPG](/前置知识/000p_前置知识_DDPG_确定性策略梯度)、[TD3](/前置知识/000q_前置知识_TD3)）允许用**任何策略产生的数据**来训练当前策略。Replay Buffer 就是存储这些数据的地方：

```
每步：存入 Buffer → 从 Buffer 随机抽 batch → 训练
同一条数据可以被抽到多次 → 每条数据的利用率 ×10~×100
```

### 1.3 打破数据相关性

即使没有样本效率的考虑，Replay Buffer 还有一个关键作用：**打破连续数据之间的时间相关性**。

连续时间步的 $(s_t, a_t, r_t, s_{t+1})$ 高度相关——$s_{t+1}$ 和 $s_t$ 只差一步。如果直接用连续数据训练神经网络，梯度方向会高度偏向最近的经验，导致学习不稳定。

从大 Buffer 中随机抽样，打乱了时间顺序，每个 mini-batch 中的数据来自不同时间点，近似 i.i.d.。

---

## 二、基本实现

### 2.1 数据结构

最简单的 Replay Buffer 是一个固定大小的**环形队列**：

| 属性 | 典型值 |
|------|--------|
| 容量 | 100K ~ 1M 条 transition |
| 存储内容 | $(s, a, r, s', \text{done})$ 五元组 |
| 写入方式 | 新数据覆盖最旧的数据（FIFO） |
| 采样方式 | 均匀随机抽 mini-batch |

### 2.2 使用流程

```
1. 初始化空 Buffer（容量 N = 1,000,000）
2. 训练循环:
   a. 从环境获取 (s, a, r, s', done)
   b. Buffer.push(s, a, r, s', done)  ← 如果满了就覆盖最旧的
   c. 如果 Buffer.size() > min_size（如 10000）:
      batch = Buffer.sample(256)  ← 随机抽 256 条
      用 batch 更新 Critic（和 Actor）
```

### 2.3 代入数字

训练 [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) 做机械臂抓取：
- Buffer 容量：1M
- 每步存一条数据：$(s_t, a_t, r_t, s_{t+1}, \text{done})$
- Mini-batch 大小：256
- 训练了 50K 步环境交互
- 每步从 Buffer 中随机抽 256 条数据训练一次
- 每条数据平均被抽到 $256 \times 50K / 50K = 256$ 次（但由于 Buffer 逐渐增大，实际分布不均匀）

**样本效率对比**：
- PPO 50K 步，每条数据用 ~5 次（3-10 个 epoch）
- SAC 50K 步，每条数据用 ~50-200 次（从 Buffer 反复抽）

---

## 三、变体

### 3.1 Prioritized Replay Buffer

不均匀采样——给 TD-error 大的数据更高的采样概率。"学错最多的经验最值得复习。"

### 3.2 Hindsight Experience Replay (HER)

把失败的经验"重标记"——如果机械臂没抓到目标 A 但到达了位置 B，就把这条经验标记为"目标是 B"的成功经验。适合 goal-conditioned RL。

---

## 四、总结

| 维度 | 说明 |
|------|------|
| 本质 | 存储历史交互数据的 FIFO 队列 |
| 核心作用 | 复用数据（↑样本效率）+ 打破相关性（↑训练稳定性）|
| 适用算法 | 所有 off-policy 方法（SAC、DDPG、TD3、DQN） |
| 不适用 | On-policy 方法（PPO、TRPO）——它们需要当前策略的新鲜数据 |

---

## 延伸阅读

- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 使用 Replay Buffer 的主流算法
- [DDPG](/前置知识/000p_前置知识_DDPG_确定性策略梯度) — 第一个使用 Replay Buffer 的连续控制算法
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — Buffer 中的数据用来逼近 Q 函数
