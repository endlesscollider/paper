---
title: RIPT-VLA：无 Critic 的 VLA 交互式后训练
order: 207
tags: [强化学习, VLA, GRPO, RLOO, 机器人操作]
category: 精读
star: 3
---

# RIPT-VLA：无 Critic 的 VLA 交互式后训练 深度精读

> **论文标题**: Interactive Post-Training for Vision-Language-Action Models  
> **作者**: Shuhan Tan, Kairan Dou, Yue Zhao, Philipp Krähenbühl  
> **机构**: UT Austin, Nankai University  
> **发表**: arXiv:2505.17016, 2025  
> **代码**: https://ariostgx.github.io/ript_vla/

**标签**: `#VLA` `#强化学习` `#RLOO` `#无Critic` `#few-shot` `#LIBERO` `#MetaWorld`

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO clip 机制
- [GRPO](/前置知识/000m_前置知识_GRPO_Group_Relative_Policy_Optimization) — GRPO 的组内比较思想
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — 先 BC 再 RL 的思路
- [动作 Token 化与自回归策略](/前置知识/000l_前置知识_动作Token化与自回归策略) — 动作表示基础
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) — VLA + RL 的全景图
- [VLA-RL 精读](./006_VLA_RL_PPO直接训练自回归VLA) — 对比：PPO + Critic 路线

---

## 一、背景与动机

### 1.1 VLA 训练的两阶段局限

当前 VLA 模型的标准训练流程：
1. **Stage 1 预训练**：在大规模多样化数据上学习通用视觉-运动技能
2. **Stage 2 SFT**：在少量任务特定数据上微调

这两个阶段都是**离线监督学习**——策略从未真正和环境交互过。这导致：
- 无法处理复合误差（一步偏了，后面全错）
- 对 SFT 数据量高度依赖（数据少时性能急剧下降）
- 不知道自己的动作执行后环境会怎样

### 1.2 RIPT-VLA 的核心提议：第三阶段

RIPT-VLA 提出 VLA 训练应该有第三个阶段：

$$
\text{预训练} \to \text{SFT} \to \textbf{强化学习交互式后训练 (RIPT)}
$$

**类比 LLM**：
- GPT 预训练 ≈ VLA 预训练
- GPT SFT ≈ VLA SFT
- GPT RLHF ≈ **RIPT-VLA**

### 1.3 为什么要无 Critic

[VLA-RL](./006_VLA_RL_PPO直接训练自回归VLA) 用的是标准 PPO（有 Critic）。RIPT-VLA 选择了无 Critic 路线，原因：

1. **7B VLA 的 Critic 太贵**：即使共享 backbone，Critic head 仍需要额外参数和计算
2. **稀疏奖励下 Critic 不准**：0/1 reward + 长轨迹 → Critic 的 TD 目标极其嘈杂
3. **工程简单**：无 Critic 意味着只需要一个模型，训练调参都更容易
4. **灵感来源**：DeepSeek-R1 证明无 Critic 的 GRPO 在 LLM 上效果很好

---

## 二、方法：LOOP + Dynamic Rejection

### 2.1 核心算法：Leave-One-Out Proximal Policy Optimization (LOOP)

RIPT-VLA 的优化算法叫 LOOP，是 RLOO advantage estimation + PPO clip 的组合。

**Step 1：Group Sampling**

对同一个初始 context $\mathbf{c} = (o_1, g)$（初始观测 + 语言指令），用当前策略 $\pi_\psi$ 采样 $K$ 条轨迹：

$$
\{\mathbf{a}_k \sim \pi_\psi(\cdot | \mathbf{c})\}_{k=1}^{K}
$$

每条轨迹获得 binary reward：$R_k = R(\mathbf{c}, \mathbf{a}_k) \in \{0, 1\}$

**Step 2：Leave-One-Out Advantage**

对每条轨迹 $k$，计算其 advantage（详见 [GRPO 前置知识](/前置知识/000m_前置知识_GRPO_Group_Relative_Policy_Optimization)）：

$$
b_k = \frac{1}{K-1}\sum_{j \neq k} R_j, \quad A_k = R_k - b_k
$$

**逐项拆解**：
- $b_k$：去掉第 $k$ 条轨迹后，其余 $K-1$ 条的平均 reward，作为 baseline
- $A_k$：第 $k$ 条轨迹相对于"同组其他成员"的优势
- 如果 $R_k = 1$（成功）且大部分同组失败，则 $A_k > 0$（正 advantage）
- 如果 $R_k = 0$（失败）且有部分同组成功，则 $A_k < 0$（负 advantage）

**为什么用 Leave-One-Out 而不是标准 GRPO？**

标准 GRPO 的 baseline 是所有样本的均值（包括自己）：$b = \frac{1}{K}\sum_{j=1}^K R_j$

Leave-One-Out 去掉了自己，降低了 bias。直觉：如果你考了 100 分，班级均分的计算里包含你自己的 100 分，会拉高 baseline，低估你的真实优势。去掉自己后的均分更公平。

**Step 3：PPO Clip 更新**

$$
\mathcal{L}_{\text{PPO}} = -\min\left(r_i \cdot A_i, \; \text{clip}(r_i, 1-\epsilon, 1+\epsilon) \cdot A_i\right)
$$

其中概率比 $r_i = \frac{\pi_\theta(\mathbf{a}_i | \mathbf{c}_i)}{\pi_\psi(\mathbf{a}_i | \mathbf{c}_i)}$。

**代入数字的例子**：

假设 $K=8$，执行 "pick up butter" 任务，结果为 [0, 0, 1, 0, 0, 1, 0, 0]（第 3、6 条成功）：

对第 3 条轨迹（成功）：
- $b_3 = (0+0+0+0+1+0+0)/7 = 1/7 \approx 0.143$
- $A_3 = 1 - 0.143 = 0.857$（正 advantage：做得比同组好！）

对第 1 条轨迹（失败）：
- $b_1 = (0+1+0+0+1+0+0)/7 = 2/7 \approx 0.286$
- $A_1 = 0 - 0.286 = -0.286$（负 advantage：做得比同组差）

PPO 会增大第 3 条轨迹中动作的概率，减小第 1 条轨迹中动作的概率。

### 2.2 Dynamic Rejection：解决全零 Advantage 问题

这是 RIPT-VLA 对比标准 GRPO 的核心创新。

**问题**：当 $K$ 条轨迹的 reward 全部相同时（全成功或全失败），$A_k = 0$ for all $k$——没有任何梯度信号。

**情况 1：全部失败**（任务太难，当前策略完全做不到）
- $R_1 = R_2 = \cdots = R_K = 0$
- $A_k = 0 - 0 = 0$，对所有 $k$
- 结果：这 $K$ 条数据完全浪费了

**情况 2：全部成功**（任务已掌握，不需要继续训了）
- $R_1 = R_2 = \cdots = R_K = 1$
- $A_k = 1 - 1 = 0$，对所有 $k$
- 结果：同样浪费

**Dynamic Rejection 策略**：

```
while |D_rollout| < B do
    采样 context c ~ D_context
    生成 K 条轨迹，获得 rewards {R_k}
    if all R_k are identical then
        丢弃这个 context，重新采样  ← 核心！
    else
        计算 advantages，加入 D_rollout
    end if
end while
```

**为什么这很重要？**

随着训练进行，策略越来越好，越来越多的任务被"完全解决"（全部成功）。如果不做 rejection：
- 后期大部分 batch 都是全零 advantage
- 有效梯度信号被稀释
- 训练变慢甚至停滞

Dynamic rejection 保证**每个 batch 中所有数据都有非零 advantage**，等效 batch size 恒定，梯度信号密度恒定。

**消融实验**：动态拒绝带来 +3.3% 的绝对成功率提升。

### 2.3 适配不同动作表示

RIPT-VLA 不只适用于 token 化动作，还兼容连续动作（回归头）。

**Token 化动作头**（如 QueST）：
- 动作是离散 token → log-prob 直接从 softmax 得到
- 采样：从分类分布中随机采样

**回归动作头**（如 OpenVLA-OFT）：
- 原始模型只输出均值 $\mu_\theta$（MSE/L1 训练），没有概率分布
- RIPT-VLA 的解决方案：**添加一个轻量的 scale prediction head**
  - 额外训练一个小网络预测 $\sigma_\theta$
  - 假设动作分布为 Laplace（对应 L1 loss）或 Gaussian（对应 MSE）
  - 这样就有了 $\log \pi_\theta(a_t) = \log \text{Laplace}(a_t; \mu_\theta, \sigma_\theta)$

$$
\log \pi_\theta(a_t | s_t) = -\frac{|a_t - \mu_\theta(s_t)|}{\sigma_\theta(s_t)} - \log(2\sigma_\theta(s_t))
$$

---

## 三、实验结果

### 3.1 标准多任务（50 示教/任务）

在 LIBERO 4 个套件上的结果：

**大模型（Stage-1 预训练 + Stage-2 SFT）**：

| 方法 | Goal | Spatial | Object | Long | 平均 |
|------|------|---------|--------|------|------|
| OpenVLA-OFT (SFT) | 97.9 | 97.6 | 98.4 | 92.9 | 96.7 |
| **OpenVLA-OFT + RIPT** | **99.0** | **98.6** | **98.6** | **93.8** | **97.5** |
| 提升 | +1.1 | +1.0 | +0.2 | +0.9 | **+0.8** |

**小模型（只有 Stage-2 SFT）**：

| 方法 | Goal | Spatial | Object | Long | 平均 |
|------|------|---------|--------|------|------|
| QueST (SFT) | 80.8 | 87.4 | 93.6 | 68.8 | 82.7 |
| **QueST + RIPT** | **92.7** | **95.6** | **98.4** | **87.5** | **93.6** |
| 提升 | +11.9 | +8.2 | +4.8 | **+18.7** | **+10.9** |

**关键发现**：
- 对于已经很强的大模型（96.7%），RIPT-VLA 仍能将失败率从 3.3% 降到 2.5%
- 对于较弱的小模型（82.7%），RIPT-VLA 带来 +10.9% 的巨大提升
- **最大提升在 LIBERO-Long（+18.7%）**——长序列任务最受益于在线交互学习

### 3.2 Few-shot 学习（极少示教）

这是 RIPT-VLA 最亮眼的结果。

**1-shot 设置**（每个任务只有 1 条示教）：

| 方法 | LIBERO-Long 成功率 |
|------|-------------------|
| QueST SFT (1-shot) | 3.5%（几乎不工作） |
| **QueST + RIPT (1-shot)** | **97.2%** |
| 提升 | **+93.7%** |

**震撼的结论**：
- SFT 只有 1 条示教时几乎完全不能用（成功率 3.5%）
- 但 RIPT-VLA 可以从这个几乎失败的起点出发，通过 15 次 RL iteration 把成功率拉到 97%！
- 这说明 VLA 预训练已经学到了丰富的运动技能，只是 1 条 SFT 数据不足以"激活"它们
- RL 交互式后训练能够高效地"唤醒"这些潜在技能

**不同示教数量的对比**：

| 示教数 | SFT 成功率 | + RIPT 成功率 | 提升 |
|--------|-----------|-------------|------|
| 1 | 3.5% | 97.2% | +93.7% |
| 2 | 15.3% | 97.8% | +82.5% |
| 5 | 50.2% | 71.4% | +21.2% |
| 10 | 68.8% | 87.5% | +18.7% |
| 50 | 68.8% | 87.5% | +18.7% |

**观察**：示教越少，RIPT-VLA 的相对提升越大。1 条示教时的 +93.7% 是惊人的。

### 3.3 跨场景泛化

**设置**：在场景 A 用 50 条示教预训练，在场景 B 用 1-5 条示教 SFT，然后 RIPT-VLA 在场景 B 交互优化。

结果：RIPT-VLA 在 1-shot 跨场景时成功率从 ~5%（SFT）提升到 ~95%。

### 3.4 跨目标泛化

**设置**：在"把红色杯子放到右边盘子"上预训练，用少量示教 SFT 到"把红色杯子放到左边盘子"。

结果：3 条示教 + RIPT → 从 SFT 的 0.7% 提升到 59.7%；10 条 + RIPT → 从 29.4% 到 79.7%。

---

## 四、为什么 RIPT-VLA 如此 Sample-Efficient

### 4.1 预训练知识的"唤醒"效应

RIPT-VLA 的高效率不是因为 RL 算法多好，而是因为**VLA 预训练已经学到了丰富的视觉-运动技能**。

类比 LLM：
- GPT-4 预训练后"知道"怎么写代码，但不知道用户想要什么风格
- RLHF 不是教 GPT "怎么写代码"，而是告诉它"用户偏好什么"

同样：
- QueST 预训练后"知道"怎么抓取、移动、放置（通用运动原语）
- RIPT-VLA 不是教它"怎么运动"，而是告诉它"在这个具体场景中，把这些原语组合起来执行什么任务"

### 4.2 Binary reward 的信息足够了

对于一个已经"会做"的模型，binary success/fail 信号包含的信息其实很丰富：
- 成功 = "刚才那个动作序列的组合是对的"
- 失败 = "刚才某些动作需要调整"

8 次尝试中有 2 次成功，模型就能通过对比知道"成功时我做了什么不同的事"。

### 4.3 Group Size 的作用

$K=8$ 或 $K=16$ 的 group sampling 相当于：
- 对每个场景做 8-16 次尝试
- 从中找到"什么 works，什么不 works"
- 这比人类练习一个技能的方式类似——反复尝试，总结规律

---

## 五、局限性与讨论

### 5.1 和 VLA-RL (PPO) 的对比

| 维度 | RIPT-VLA (RLOO) | VLA-RL (PPO) |
|------|----------------|--------------|
| Critic | 不需要 | 需要（共享 backbone） |
| 显存 | 较少 | 较多 |
| 稀疏奖励处理 | 纯靠 group 内比较 | RPRM 密集化 |
| Credit assignment | 粗糙（轨迹级） | 精细（step-level via GAE） |
| 训练稳定性 | Dynamic rejection 保证 | Critic warmup 保证 |
| 最终性能 | 在大模型上略低 | 在大模型上略高 |
| 工程复杂度 | 低 | 高 |

根据 "What Can RL Bring to VLA Generalization?" 的实验，**PPO 在稀疏奖励下系统性优于 GRPO/RLOO**。但 RIPT-VLA 的优势在于简洁和高效——尤其是 few-shot 场景下效果惊人。

### 5.2 当前局限

1. **高成功率时信号减弱**：当策略已经 >95% 成功时，大部分 group 全部成功，即使 dynamic rejection 也很难找到有效的训练信号
2. **只有 binary reward**：不区分"差一点成功"和"完全失败"——SRPO 的 progress reward 是对此的改进
3. **只在仿真验证**：真实机器人的 RL 交互更贵且有安全问题

---

## 六、个人评价

### 6.1 这篇文章的重要性

RIPT-VLA 提出了一个极其简洁有效的第三阶段训练范式。最让人印象深刻的是 **1-shot → 97% 的结果**——这说明 VLA 预训练的价值远超我们之前的认知。SFT 的数据效率低不是因为模型不行，而是因为监督学习无法充分"激活"已有的知识。

### 6.2 技术洞察

Dynamic rejection 虽然简单，但解决了 GRPO 在 VLA 场景中的根本缺陷。它保证了有效训练信号的恒定密度——这对长期稳定训练至关重要。

### 6.3 实践建议

- 如果资源有限（单卡 24G），优先考虑 RIPT-VLA（无 Critic，轻量）
- 如果追求极致性能（4× A100），考虑 VLA-RL + RPRM（更精细的信号）
- 如果只有极少示教数据，RIPT-VLA 是当前最佳选择

---

## 延伸阅读

- [VLA-RL 精读](./006_VLA_RL_PPO直接训练自回归VLA) ← PPO + Critic 路线的对比
- [GRPO 前置知识](/前置知识/000m_前置知识_GRPO_Group_Relative_Policy_Optimization) ← RLOO/GRPO 的详细原理
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) ← 完整方法对比
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) ← PPO clip 机制细节
