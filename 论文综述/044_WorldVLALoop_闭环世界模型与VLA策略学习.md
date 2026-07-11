---
title: World-VLA-Loop：闭环世界模型与 VLA 策略协同学习
order: 244
tags: [强化学习, VLA, 世界模型, 闭环学习, 视频扩散, 状态感知]
category: 精读
---

# World-VLA-Loop：闭环协同学习深度精读

> **论文标题**: Closed-Loop Learning of Video World Model and VLA Policy
> **作者**: Xiaokang Liu, et al.
> **机构**: ShowLab, National University of Singapore
> **发表**: arXiv:2602.06508, 2025
> **项目页**: https://showlab.github.io/World-VLA-Loop/

**标签**: `#VLA` `#强化学习` `#世界模型` `#闭环学习` `#视频扩散` `#状态感知`

**知识链接**：
- [世界模型基础](/前置知识/000t_前置知识_世界模型基础) — World Model 概念
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — RL 算法
- [扩散模型 DDPM](/前置知识/000b_前置知识_扩散模型DDPM) — 视频扩散基础
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — 全景概览
- [World-Env 精读](./024_WorldEnv_世界模型虚拟环境VLA后训练) — 对比：单向学习
- [WoVR 精读](./036_WoVR_可靠世界模型RL后训练VLA) — 对比：防幻觉方案
- [World-Gymnast 精读](./038_WorldGymnast_视频世界模型RL训练机器人) — 对比：迭代改进

---

## 一、背景与动机

### 1.1 现有世界模型 RL 的单向问题

现有方法（World-Env、World-Gymnast）的世界模型和策略是**单向关系**：

```
World Model (固定或偶尔更新) → RL 训练 → Policy 改进
```

但世界模型有一个关键缺陷：**对"几乎成功"的关键决策点预测不准**。

- 训练数据中成功和失败轨迹混杂
- 在"差一点就成功/失败"的边界状态，世界模型不确定要生成什么
- 恰恰这些状态对策略学习最关键

### 1.2 World-VLA-Loop 的核心贡献

**闭环设计**：世界模型和策略**互相改进**，形成正反馈循环。

```mermaid
flowchart LR
    WM["World Model"] -->|"提供训练环境"| P["VLA Policy"]
    P -->|"生成新轨迹数据"| WM
    WM -->|"接收 Sans 数据改进"| WM
```

**关键创新**：引入 **Sans Dataset**（Near-Success Trajectory Dataset）——专门包含"几乎成功但最终失败"的轨迹，用来训练世界模型在关键边界状态的预测能力。

---

## 贯穿全文的例子

> **场景**：机器人执行 "place the cup on the saucer"。
>
> - **边界状态**：杯子已经在碟子上方 2mm，但角度偏了 5°
>   - 如果继续放 → 滑落（失败）
>   - 如果微调角度 → 成功
> - **普通世界模型**：对这种边界状态预测不准（有时生成成功，有时生成失败）
> - **World-VLA-Loop**：
>   1. 策略在世界模型中探索，产生大量"几乎成功"的轨迹
>   2. 用这些 Near-Success 数据改进世界模型 → 准确预测"5° 偏角 → 滑落"
>   3. 策略从改进的世界模型中学到"必须先微调角度"
>   4. 循环迭代，成功率持续上升

---

## 二、方法详解

### 2.1 State-Aware Video World Model

世界模型不只预测视觉帧，还同时预测**状态/奖励信号**：

$$
\hat{o}_{t+1}, \hat{r}_t, \hat{s}_{t+1} = \text{WM}(o_t, a_t)
$$

**状态感知**的好处：
- 世界模型知道"什么状态变化会导致成功/失败"
- 可以直接输出 reward 信号，不需要额外的 VLM 评估
- 对关键状态转变更敏感

### 2.2 Sans Dataset（Near-Success Trajectories）

**数据收集**：在策略训练过程中，收集"进度 > 80% 但最终失败"的轨迹：

$$
\text{Sans} = \{ \tau : \text{progress}(\tau) > 0.8 \text{ AND } \text{success}(\tau) = 0 \}
$$

**为什么重要**：
- 这些轨迹包含了关键的"失败边界"信息
- 世界模型学习这些数据后，能准确预测"什么操作会导致最后时刻失败"
- 策略就能学会"避免那些导致失败的微小偏差"

### 2.3 闭环训练流程

```python
for loop_iter in range(max_loops):
    # Step 1: Policy RL in World Model
    for rl_step in range(200):
        imagined_rollout = world_model.rollout(policy)
        reward = world_model.predict_reward(imagined_rollout)
        policy = ppo_update(policy, imagined_rollout, reward)

    # Step 2: Collect Sans data from improved policy
    real_rollouts = env.rollout(policy, n=100)
    near_success = filter_sans(real_rollouts)

    # Step 3: Improve World Model with Sans data
    world_model = finetune(world_model, near_success + success_data)

    # Step 4: Verify improvement
    eval_success = evaluate(policy, env)
    print(f"Loop {loop_iter}: {eval_success}%")
```

### 2.4 关键设计：奖励信号联合预测

世界模型同时预测视频帧和奖励，使用多任务学习：

$$
\mathcal{L}_{\text{WM}} = \underbrace{\mathcal{L}_{\text{video}}}_{\text{帧预测}} + \lambda_r \underbrace{\mathcal{L}_{\text{reward}}}_{\text{奖励预测}} + \lambda_s \underbrace{\mathcal{L}_{\text{state}}}_{\text{状态预测}}
$$

这保证世界模型不仅"画面好看"，而且"物理语义正确"。

---

## 三、实验结果

### 3.1 与单向方法对比

| 方法 | 世界模型更新？ | 用 Sans 数据？ | 成功率 |
|------|-------------|-------------|--------|
| World-Env (单向) | ❌ | ❌ | 65% |
| World-Gymnast (偶尔更新) | 偶尔 | ❌ | 78% |
| WoVR (防幻觉) | ✅ | ❌ | 80% |
| **World-VLA-Loop** | **✅ 每轮** | **✅** | **88%** |

### 3.2 Sans Dataset 的效果

| 配置 | 成功率 |
|------|--------|
| Loop without Sans | 82% |
| Loop with random failure data | 83% |
| **Loop with Sans (near-success)** | **88%** |

Near-Success 数据比随机失败数据有效得多（+5%），因为它精确描述了失败边界。

### 3.3 真实机器人

| 任务 | World-Env | World-VLA-Loop | 提升 |
|------|----------|---------------|------|
| Precise placement | 48% | 72% | +24% |
| Multi-step assembly | 30% | 55% | +25% |
| Deformable object | 35% | 58% | +23% |

在精密操作上优势特别大——因为这些任务的"成功/失败边界"特别窄。

---

## 四、总结

| 维度 | World-VLA-Loop |
|------|---------------|
| 核心问题 | 世界模型在关键决策边界预测不准 |
| 核心方案 | 闭环协同学习 + Sans (Near-Success) Dataset |
| 独特贡献 | 策略和世界模型互相改进的正反馈机制 |
| 关键数据 | Near-Success 轨迹（进度>80% 但失败的） |
| vs 单向方法 | +8-23% 成功率提升 |
| 适用场景 | 精密操作（窄成功边界） |

---

## 延伸阅读

- [World-Env：世界模型虚拟环境](./024_WorldEnv_世界模型虚拟环境VLA后训练) — 单向方法
- [WoVR：可靠世界模型](./036_WoVR_可靠世界模型RL后训练VLA) — 防幻觉方案
- [World-Gymnast：迭代改进](./038_WorldGymnast_视频世界模型RL训练机器人) — 偶尔更新世界模型
- [世界模型基础](/前置知识/000t_前置知识_世界模型基础) — 概念入门
