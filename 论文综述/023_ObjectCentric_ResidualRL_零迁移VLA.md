---
title: Object-Centric Residual RL：零迁移 Sim-to-Real VLA 增强
order: 223
tags: [强化学习, VLA, Residual RL, Sim-to-Real, 物体中心, SAC, 零迁移]
category: 精读
star: 3
---

# Object-Centric Residual RL：零迁移增强 VLA 深度精读

> **论文标题**: Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement  
> **作者**: Jiafei Duan, Liang Peng, et al.  
> **机构**: Microsoft Research  
> **发表**: arXiv:2606.18953, 2025  
> **项目页**: https://www.microsoft.com/en-us/research/articles/object-centric-residual-rl/

**标签**: `#VLA` `#强化学习` `#SAC` `#ResidualRL` `#Sim-to-Real` `#物体中心` `#零迁移`

**知识链接**：
- [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — Residual Policy 训练算法
- [Replay Buffer](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — Off-policy 数据存储
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — Residual RL 的上下文
- [Sim-to-Real 迁移综述](/论文综述/S04_Sim_to_Real迁移综述) — Sim-to-Real 迁移方法论
- [PLD 精读](./015_PLD_Residual_RL自改进VLA) — 对比：另一种 Residual RL 方案
- [BootRL 精读](./013_BootRL_冻结VLA加RL_Head) — 对比：冻结 VLA + 小 head
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — VLA + RL 全景图

---

## 一、背景与动机

### 1.1 Residual RL 的 Sim-to-Real 困境

Residual RL 的思路很简洁：**冻结 VLA base policy，训练一个轻量的 residual policy 来修正动作**。

$$
a_{\text{final}} = a_{\text{VLA}} + a_{\text{residual}}
$$

但是 Residual RL 在 Sim-to-Real 上面临一个三难困境：

| 方案 | 问题 |
|------|------|
| 仿真中用特权状态训练 residual | 部署时没有特权状态，需要额外蒸馏步骤 |
| 仿真中用图像训练 residual | 仿真图像 vs 真实图像有视觉 domain gap |
| 真实环境中训练 residual | 昂贵、不安全、样本效率低 |

### 1.2 核心洞察：物体位姿是天然的"零迁移"表示

本文的关键观察：**物体 6D 位姿**（position + orientation）是一种天然跨域的紧凑表示：

- 仿真中：直接从物理引擎获取
- 真实世界：用现成的位姿估计器（如 FoundationPose）获取
- 两者的数值空间**完全一致**——不存在 domain gap！

**类比**：就像 GPS 坐标在模拟地图和真实世界中是一样的，物体位姿在仿真和现实中也是一样的数字。

---

## 贯穿全文的例子

> **场景**：VLA 模型（OpenVLA）执行 "pick up the mug by its handle"。
>
> - VLA 的动作精度不够：经常抓到杯身而非手柄
> - **Object-Centric 输入**：杯子的 $(x, y, z, r_x, r_y, r_z)$ + 手柄的相对位置
> - **Residual Policy**：一个 3 层 MLP（~100K 参数），输入物体位姿，输出动作修正量
> - **训练**：纯仿真，用 SAC 训练 1 小时
> - **部署**：零迁移到真实机器人（VLA + FoundationPose + residual MLP）

---

## 二、方法详解

### 2.1 整体框架

```mermaid
flowchart LR
    subgraph Real["真实环境"]
        R1["Camera"] --> R2["FoundationPose"]
        R2 --> R3["Object Pose"]
    end
    
    subgraph Sim["仿真环境（训练）"]
        S1["Physics Engine"] --> S2["Ground Truth Pose"]
    end
    
    subgraph Policy["策略组合"]
        P1["Frozen VLA"] --> P2["Base Action a_VLA"]
        R3 --> P3["Residual MLP"]
        P3 --> P4["Correction Δa"]
        P2 --> P5["a_final = a_VLA + Δa"]
        P4 --> P5
    end
```

### 2.2 Object-Centric State Space

Residual policy 的输入是一个紧凑的物体中心状态向量：

$$
s_{\text{obj}} = [p_{\text{target}}, q_{\text{target}}, p_{\text{ee}}, q_{\text{ee}}, p_{\text{target}} - p_{\text{ee}}]
$$

**逐项拆解**：
- $p_{\text{target}} \in \mathbb{R}^3$ — 目标物体的 3D 位置
- $q_{\text{target}} \in \mathbb{R}^4$ — 目标物体的姿态（四元数）
- $p_{\text{ee}} \in \mathbb{R}^3$ — 机械臂末端的位置
- $q_{\text{ee}} \in \mathbb{R}^4$ — 末端的姿态
- $p_{\text{target}} - p_{\text{ee}} \in \mathbb{R}^3$ — 相对位置差

总维度：$3+4+3+4+3 = 17$ 维。极其紧凑。

**为什么这样设计**：
1. **维度低** → MLP 只需 3 层就能学好
2. **语义明确** → 每一维都有清晰的物理含义
3. **跨域一致** → 仿真和真实的数字格式完全相同

### 2.3 Residual Policy 训练

Residual policy $\pi_{\text{res}}$ 是一个小 MLP：

$$
\Delta a = \pi_{\text{res}}(s_{\text{obj}}; \phi) \in [-\epsilon, \epsilon]^7
$$

**约束**：输出被 clip 到 $[-\epsilon, \epsilon]$，限制修正幅度（如 $\epsilon = 0.05$，即最多修正 5cm）。

训练使用 [SAC](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)：
- **Off-policy**：样本效率高，约 100K 环境步即收敛
- **最大熵**：鼓励多样性探索，避免过拟合到单一修正模式
- **连续动作空间**：天然适合 7-DoF 修正量

**代入数字**：
- VLA 输出 $a_{\text{VLA}} = [0.32, 0.15, 0.08, 0, 0, 0.1, 1]$（位置增量 + 姿态增量 + 夹爪）
- Residual 输出 $\Delta a = [-0.02, +0.03, -0.01, 0, 0, 0, 0]$（微调末端位置）
- 最终动作 $a_{\text{final}} = [0.30, 0.18, 0.07, 0, 0, 0.1, 1]$

### 2.4 零迁移部署

部署时的pipeline：

1. 摄像头拍摄真实场景
2. VLA 接收图像 + 语言指令 → 输出 $a_{\text{VLA}}$
3. FoundationPose 从图像中估计物体 6D 位姿 → $s_{\text{obj}}$
4. Residual MLP 接收 $s_{\text{obj}}$ → 输出 $\Delta a$
5. 执行 $a_{\text{VLA}} + \Delta a$

**为什么能零迁移**：因为 residual policy 的输入是物体位姿（数字），不是图像。仿真中位姿是 $(0.3, 0.2, 0.1)$，真实中也是 $(0.3, 0.2, 0.1)$ — 没有 domain gap。

---

## 三、实验结果

### 3.1 仿真实验

| 方法 | 成功率 | 训练耗时 | 需要真实数据? |
|------|--------|---------|-------------|
| VLA alone | 62% | - | ❌ |
| Image-based Residual RL | 71% | 8h | ❌ |
| Privileged → Distill | 78% | 12h | ❌ |
| **Object-Centric Residual** | **82%** | **1h** | ❌ |
| Real-world Residual RL | 80% | 20h | ✅ |

### 3.2 真实机器人零迁移

| 任务 | VLA only | + Object-Centric Residual | 提升 |
|------|----------|--------------------------|------|
| Mug grasping | 55% | 80% | +25% |
| Precise placement | 40% | 72% | +32% |
| Drawer opening | 65% | 85% | +20% |

**关键发现**：仿真训练 1 小时，零迁移到真实机器人，接近甚至超过了需要 20 小时真实环境训练的 baseline。

---

## 四、核心优势与局限

### 优势

1. **极轻量**：Residual MLP 只有 ~100K 参数，推理几乎无延迟
2. **零迁移**：仿真训练直接部署，无需 domain adaptation
3. **模块化**：VLA 完全冻结，residual 可独立替换
4. **训练快速**：SAC + 低维状态 = 1 小时收敛

### 局限

1. **依赖位姿估计器**：FoundationPose 的精度直接影响 residual 质量
2. **单物体假设**：当前方法主要处理单目标物体场景
3. **修正幅度有限**：如果 VLA 的 base action 差得太远（超出 $\epsilon$ 范围），residual 无法救回

---

## 五、总结

| 维度 | Object-Centric Residual RL |
|------|---------------------------|
| 核心创新 | 物体位姿作为跨域不变表示，实现零迁移 Residual RL |
| RL 算法 | SAC |
| Residual 规模 | ~100K 参数的 3 层 MLP |
| 训练环境 | 纯仿真（1h） |
| 部署方式 | 零迁移到真实机器人 |
| 关键依赖 | 6D 位姿估计器（FoundationPose） |

---

## 延伸阅读

- [PLD：Residual RL 自改进 VLA](./015_PLD_Residual_RL自改进VLA) — 另一种 Residual RL 方案（但需要蒸馏回 VLA）
- [BootRL：冻结 VLA 加 RL Head](./013_BootRL_冻结VLA加RL_Head) — 类似的"不改 VLA"思路
- [Sim-to-Real 迁移综述](/论文综述/S04_Sim_to_Real迁移综述) — Sim-to-Real 的系统性介绍
