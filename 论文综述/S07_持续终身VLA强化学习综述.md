---
title: 持续终身VLA强化学习综述
order: 7
tags: [强化学习, VLA, 持续学习, 终身学习, 灾难性遗忘, Replay, LoRA]
category: 综述
star: 5
---

# 持续/终身 VLA 强化学习综述：顺序学习 50 个任务时，早期经验如何不被忘记

> **检索截止日期**：2026-07-11
> **核心问题**：VLA 按顺序学习大量机器人任务时，旧任务的交互经验逐渐被 replay buffer 稀释或淘汰，如何同时保持旧技能、学习新技能和控制存储成本？
> **一句话结论**：先用“大型预训练 VLA + LoRA + on-policy RL 顺序微调”作为最强简单基线；如果仍遗忘，再加入**按任务均衡、有固定总预算的小型回放**，而不是一个全局 FIFO buffer。

---

## 一、先把问题定义对

“训练 50 个任务”可能对应三个不同问题，它们的正确解法不同。

| 设定 | 数据是否同时可用 | 主要问题 | 关键词 |
|---|---:|---|---|
| 多任务学习 | 50 个任务始终都可以采样 | 任务不平衡、梯度冲突 | Multi-task RL |
| 持续/终身学习 | 任务依次到来，旧环境不再或很少可访问 | 灾难性遗忘、稳定性-可塑性 | Continual/Lifelong RL |
| 非平稳 RL | 没有清晰任务边界 | 环境、奖励或机器人本体持续变化 | Non-stationary RL |

用户描述的“学到第 50 个任务时，第 1 个任务的 RL 经验所剩无几”，属于**有限记忆预算下的 continual VLA-RL**。

### 1.1 为什么全局 FIFO buffer 会失败

设 buffer 最多存 $B$ 条 transition，每个任务生成 $N$ 条。对全局 FIFO，当后续任务产生的数据超过 $B$ 后，任务 1 的数据会被全部淘汰。即使用全局 reservoir sampling，任务 1 的期望占比也只有约 $1/T$，学到 50 个任务时只剩 2%。

这不只是“经验少”，还会导致：

1. 旧任务几乎不再产生梯度。
2. critic 被当前任务的奖励尺度和状态分布覆盖。
3. 旧轨迹由旧策略产生，直接用于 PPO/GRPO 会产生严重 off-policy 偏差。
4. 早期任务的成功率下降后，又更难采到新的成功轨迹，形成恶性循环。

---

## 二、最新结论：不要默认“没有大 replay 就会忘”

2026 年的两组 VLA 实验给出了表面上相反的结论，实际上它们的训练目标不同。

### 2.1 Continual VLA-RL：可以几乎不用旧经验

[Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://arxiv.org/abs/2603.11653) 是目前最直接、最系统的 continual VLA-RL 研究。它在 LIBERO、RoboCasa 和 ManiSkill 等环境中比较了：

- Sequential Fine-Tuning
- EWC
- Expert Replay
- Dark Experience Replay
- Dynamic Weight Expansion
- SLCA
- RETAIN
- Multi-task oracle

其主要发现是：

1. **大型预训练 VLA + LoRA + on-policy GRPO** 直接按顺序学习，平均遗忘很小。
2. 在论文的三个主 LIBERO 设定中，Seq. FT 的 NBT 约在 $-2.4\%$ 到 $1.0\%$ 之间。
3. 该结论被扩展到 **30 个顺序任务**，但还没有验证 50 个真实机器人任务。
4. 去掉任一关键条件都可能显著变差：全参数微调、小模型、或将 on-policy RL 换成 SFT。
5. 传统 continual learning 方法不一定增益：它们常通过限制更新来保留旧知识，但也可能损害新任务的可塑性。

这与 [RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/abs/2509.04259) 的解释一致：on-policy RL 在能够解决新任务的多个策略中，倾向找到相对原策略 KL 变化更小的解。[Retaining by Doing](https://arxiv.org/abs/2510.18874) 进一步将关键因素归结为 **on-policy data**，而不只是显式 KL penalty 或 advantage 形式。

### 2.2 Continual VLA 模仿学习：小 replay 仍然很重要

[Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning](https://arxiv.org/abs/2603.03818) 研究的是示教数据微调，而不是纯 on-policy RL。该工作比较 $\pi_0$、GR00T N1.5 和从头训练的小策略，发现：

1. 预训练 VLA 配合简单 Experience Replay 即可强力抗遗忘。
2. 其主实验按每个旧任务保留 $M$ 个 transition，而不是让所有任务争抢一个 FIFO buffer。
3. 当每任务只保留约 **2%（100 个样本）**时，预训练 VLA 的遗忘明显小于非预训练策略。
4. 当只剩约 **0.2%（10 个样本）**时，所有方法都开始明显遗忘；因此“大模型不需要 replay”不能直接外推到 SFT/BC。
5. 表面上已忘记的任务往往仍保留可快速恢复的潜在知识，说明“当前成功率下降”不完全等于“表示已被抹除”。

### 2.3 两组结论为什么不矛盾

| 维度 | Continual VLA-RL | Continual VLA-BC/SFT |
|---|---|---|
| 主要新数据 | 当前策略自己的 on-policy rollout | 固定专家示教 |
| 更新信号 | 优势加权，只强化高回报行为 | 对所有示教 token/action 做拟合 |
| 对原策略的偏移 | 通常较小 | 可能较大 |
| 旧数据需求 | 特定条件下可以很少或不需要 | 小而均衡的 replay 通常必要 |
| 主要风险 | 采样成本、稀疏奖励、策略崩溃 | 分布偏移、旧任务被新数据覆盖 |

因此，设计方案前必须先问：**旧经验是用来做旧策略的 RL 更新，还是只用来做 BC/KL 锚定？** 后者通常更稳妥。

---

## 三、文献地图：哪些文章真正解决了这个问题

### 3.1 基准和问题定义

| 工作 | 作用 | 与 50 任务问题的关系 |
|---|---|---|
| [Towards Continual Reinforcement Learning: A Review and Perspectives](https://arxiv.org/abs/2012.13490) | 给出 continual RL 的非平稳性分类、方法谱系和评价指标 | 理论入口，适合写 related work |
| [Continual World](https://arxiv.org/abs/2105.10919) | 基于 Meta-World 的持续机器人 RL 基准 | 强调不能只看遗忘，还要看 forward transfer |
| [LIBERO](https://arxiv.org/abs/2306.03310) | 4 个 task suite，共 130 个操作任务 | 目前 VLA 持续学习最常用基准，可组成 10/30/50 任务序列 |
| [Disentangling Transfer in Continual Reinforcement Learning](https://arxiv.org/abs/2209.13900) | 分析 SAC 的 actor、critic、探索和数据对迁移的作用，提出 ClonEx-SAC | 对 off-policy 多任务 RL 和 critic 设计很有参考价值 |

### 3.2 经验回放和有限记忆

| 工作 | 核心方法 | 对 VLA 的启示 |
|---|---|---|
| [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682) | 将当前 on-policy 数据与旧数据的 off-policy/BC 约束混合；内存受限时随机丢弃 | 旧轨迹不一定要直接做 policy gradient，可用 BC 保持行为 |
| [Dark Experience Replay](https://arxiv.org/abs/2004.07211) | 同时回放样本和历史 logits，用知识蒸馏保持旧函数 | VLA 可存旧 action distribution/动作 token logits，而不只存硬动作标签 |
| [Forget Me Not / Pretrained VLAs are Surprisingly Resistant...](https://arxiv.org/abs/2603.03818) | 比较 0.2%、2%、20% 的 per-task replay | 直接证明“预训练 VLA + 小 replay”比小模型更能抗遗忘 |
| [Stellar VLA](https://arxiv.org/abs/2511.18085) | 任务-技能知识空间、语义路由，只用 1% replay | 经验少时不只保存样本，还可显式建模任务和技能关系 |

### 3.3 参数隔离、Adapter 和技能库

| 工作 | 核心方法 | 局限 |
|---|---|---|
| [EWC](https://arxiv.org/abs/1612.00796) | 用 Fisher 重要性限制关键参数的变化 | 任务很多时可塑性下降，对大 VLA 的 Fisher 估计成本高 |
| [PackNet](https://arxiv.org/abs/1711.05769) | 迭代剪枝并为新任务分配空闲参数 | 保留强，但容量会耗尽，任务路由也是问题 |
| [TAIL](https://arxiv.org/abs/2310.05905) | 为大型预训练策略比较 Adapter、P-Tuning 和 LoRA | LoRA 只训约 1% 参数即可兼顾适应与保留，但主要是模仿学习 |
| [LOTUS](https://arxiv.org/abs/2311.02058) | 从未分段示教中持续发现技能，建立可增长 skill library，用 meta-controller 组合 | 系统复杂度高，不是端到端 VLA-RL |
| [TOPIC](https://arxiv.org/abs/2504.15517) | Task-Specific Prompt + 任务关系图持续演化 | 适合 few-shot action-incremental learning，尚非通用 on-policy VLA-RL |
| [MergeVLA](https://arxiv.org/abs/2511.18810) | 稀疏 LoRA task mask + 可组合 action expert + 测试时路由 | 更像多专家合并；存储和路由开销随技能数增加 |

### 3.4 直接面向 VLA 持续 RL/RFT

| 工作 | 设定与贡献 | 需注意的边界 |
|---|---|---|
| [Reinforcement Fine-Tuning Naturally Mitigates Forgetting](https://arxiv.org/abs/2507.05386) | 在多模态基础模型上系统比较 SFT 和 RFT，提出 RIF-RFT | 不是机器人 VLA 为主，但提供了 RFT 选择性更新的解释 |
| [LifeLong-RFT](https://arxiv.org/abs/2602.10503) | chunk-level on-policy RFT；QACR + CTAR + FCR 可验证过程奖励；LIBERO 中先学 6 任务，再顺序学 4 任务 | 它仍为每个旧任务保留 5 条示教做 ER；远未证明 50 任务规模 |
| [Simple Recipe Works](https://arxiv.org/abs/2603.11653) | 最系统的 continual VLA-RL 方法比较；强调大预训练模型、LoRA 和 on-policy RL 的协同 | 最长实验为 30 任务；对更大域偏移、真机长期磨损和奖励变化仍缺证据 |
| [Preserving and Combining Knowledge in Robotic Lifelong RL](https://doi.org/10.1038/s42256-025-00983-2) | 用贝叶斯非参数知识空间积累、组合一次性任务知识 | 不是当代端到端 VLA 训练配方，但“技能组合而非样本堆积”的思路很重要 |

---

## 四、50 任务时的可落地训练方案

### 4.1 先建立最强简单基线

如果使用 OpenVLA-OFT、$\pi_0$ 或类似的大型预训练 VLA，建议先跑：

1. 冻结大部分 VLM/backbone，仅训 LoRA 和必要的 action head。
2. 使用 on-policy PPO/GRPO/RLOO，一次只训当前任务。
3. 每学完一个任务，评估所有已学任务和一组 held-out 任务。
4. 不要在看到实际遗忘之前就加入复杂 EWC、独立 expert 或无限 replay。

这一基线直接对应 Simple Recipe Works，也是后续论文比较必须包含的 baseline。

### 4.2 如果遗忘，使用固定预算的 task-balanced memory

设总记忆预算为 $B$，已学任务数为 $t$，最简单的配额是：

$$
|\mathcal{M}_j| = \left\lfloor \frac{B}{t} \right\rfloor, \qquad j=1,\ldots,t.
$$

当新任务到来时，对每个旧任务的 memory 重做 reservoir/herding，而不是只删最早数据。训练 batch 按两层分层采样：

$$
P(\text{current})=\rho, \qquad
P(\text{old task}=j)=\frac{1-\rho}{t-1}.
$$

可先用 $\rho=0.5$ 作为基线。对 50 个任务和 $B=5000$ 条 transition，最终每任务约 100 条，正好对应 Forget Me Not 中的 2% replay 量级；**但该论文只在 10 任务 suite 上验证，50 任务是需要新实验验证的外推。**

### 4.3 旧经验不要直接当作 on-policy PPO 数据

对当前任务做 RL，对旧任务做 BC/蒸馏锚定，是更稳定的混合目标：

$$
\mathcal{L}=
\mathcal{L}_{\text{RL}}^{\text{current}}
+\lambda_{\text{BC}}\,
\mathbb{E}_{(o,l,a)\sim\mathcal{M}}
[-\log \pi_\theta(a\mid o,l)]
+\beta\,
\mathbb{E}_{(o,l)\sim\mathcal{M}}
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\text{stored}}).
$$

原因是旧轨迹来自旧策略，不满足 PPO/GRPO 的近似 on-policy 假设。若必须对旧轨迹做 RL，需要 importance sampling、V-trace/Retrace 或专门的 off-policy actor-critic，不能直接混入 PPO batch。

### 4.4 应保存什么

对视觉机器人轨迹，存所有 RGB 帧成本很高。推荐按优先级保留：

1. 每任务少量成功轨迹，覆盖不同初始状态。
2. 失败分界附近的 recovery 轨迹，而不是大量重复简单成功样本。
3. 任务指令、本体信息、奖励版本和 success predicate，否则旧数据可能无法重现。
4. 旧策略的 action logits/分布参数，用于 Dark Experience Replay 式蒸馏。
5. 经过版本化的视觉 latent 可用于压缩，但 encoder 变化后会出现 latent staleness，需要冻结 encoder 或定期重编码。

### 4.5 什么时候该用独立 Adapter/技能库

当任务之间域差很大，例如不同机器人本体、不同动作维度或完全不同的场景，共享一个 LoRA 可能开始冲突。此时可考虑：

- 按任务族而不是单任务分配 LoRA。
- 通过语言指令和初始观测做 adapter/expert routing。
- 将技能拆成可重用子技能，用 meta-controller 组合，参考 LOTUS 和 Stellar VLA。

但不建议默认“每任务一个 LoRA”。50 个完全独立 adapter 能防止遗忘，却会让参数、路由错误和部署复杂度线性增长，也失去跨任务正迁移。

---

## 五、实验必须怎么评估

仅报“50 任务最终平均成功率”不足以证明 continual learning 有效。应在学完第 $i$ 个任务后，评估任务 $j$ 并记录成功率 $S_{i,j}$，形成完整的 $50\times50$ 矩阵。

至少报告：

| 指标 | 回答的问题 |
|---|---|
| Final AVG | 学完所有任务后的总体能力 |
| NBT / Forgetting | 旧任务相对刚学完时掉了多少 |
| FWT | 旧知识是否让新任务学得更快 |
| Held-out zero-shot | 是否破坏 VLA 原有泛化能力 |
| AUC / learning speed | 每个新任务需要多少 rollout 才学会 |
| Memory and compute | 每任务保留多少帧/轨迹/参数，训练时间是否随任务数增长 |
| Worst-task success | 平均数是否掩盖某些任务完全遗忘 |

同时至少测试 3 种 task order，并分开报告：同场景任务、跨场景任务、跨本体任务。LIBERO 中的多个任务共享视觉和动作结构，不能代表所有真实部署分布偏移。

---

## 六、目前文献还没有解决什么

### 6.1 50 任务仍是有价值的研究空缺

截至检索日期，尚没有一篇工作同时完成：

- 大型 VLA；
- 真正 on-policy RL；
- 50 个以上顺序任务；
- 固定总内存，而非 per-task memory 无限增长；
- 同时包含场景、技能、奖励和 embodiment 变化；
- 长期真机验证。

Simple Recipe Works 已做到 30 任务的仿真验证；Forget Me Not 主要使用 10 任务 LIBERO suite；LifeLong-RFT 的 continual 阶段是 4 个新任务。因此，“固定记忆预算下的 50-task continual VLA-RL”本身就能形成清晰论文问题。

### 6.2 可直接发展成论文的选题

#### 选题 A：Budgeted Continual VLA-RL

**问题**：总 buffer 固定为 $B$，任务从 10 增长到 30/50/100 时，应如何分配经验？

**方法候选**：任务均衡 reservoir + 任务内 k-center 多样性选择 + 基于遗忘风险的自适应配额。

**关键对比**：Seq-LoRA-GRPO 无 replay、全局 reservoir、per-task 均分、风险自适应 replay。

#### 选题 B：Replay-free 的适用边界

**问题**：Simple Recipe Works 的结论在什么时候失效？

**自变量**：模型规模、预训练数据覆盖、LoRA rank、SFT/RL 比例、任务相似度、奖励噪声、任务序列长度。

**贡献形式**：给出“什么时候不用 replay，什么时候必须用”的经验 scaling law 或判定器。

#### 选题 C：Current-task RL + Old-task Behavioral Anchor

**问题**：如何不把过时旧轨迹错当成 on-policy 数据，又利用它们防遗忘？

**方法**：当前任务用 GRPO/PPO，旧 memory 用 BC + action-distribution distillation + 局部 KL anchor，并自适应调整 $\lambda_{BC}$。

**新意**：直接解决 VLA-RL 中 replay 的 off-policy 不匹配，比“单纯增大 buffer”更具算法贡献。

#### 选题 D：Continual Critic for VLA

**问题**：actor 可能在 LoRA 下不忘，但 critic 是否已忘记旧任务的奖励尺度和状态价值？

**方法**：任务条件化 critic、多头 critic、分布式 value normalization、旧任务 value distillation。

**价值**：当前多数 VLA continual 工作聚焦 policy 遗忘，对 critic 演化的系统研究不足。

#### 选题 E：Skill-aware Memory Compression

**问题**：固定内存下，应保存完整 RGB 轨迹、关键帧、latent、action chunk，还是技能 prototype？

**方法**：用技能分割将长轨迹压缩为关键状态 + action primitive + 旧策略分布，只在检测到遗忘时解压回放。

**参考**：LOTUS 的 skill library、Stellar VLA 的任务-技能知识空间和 Dark Experience Replay 的函数蒸馏。

---

## 七、一套可发表的实验矩阵

建议至少包含以下基线：

1. Joint multi-task oracle。
2. Sequential full fine-tuning + SFT。
3. Sequential LoRA + SFT。
4. Sequential LoRA + on-policy GRPO，无 replay。
5. 第 4 项 + task-balanced ER（0.2% / 2% / 20%）。
6. EWC。
7. Per-task/task-family LoRA + router。
8. 你提出的自适应 memory 或 behavioral anchor 方法。

实验轴：

| 轴 | 建议取值 |
|---|---|
| 任务数 | 10 / 30 / 50 |
| 任务顺序 | 随机 3 个 seed + 由易到难 + 由难到易 |
| 记忆预算 | 0 / 10 / 100 / 1000 samples per task，以及固定总预算 |
| 更新参数 | full FT / LoRA rank 8, 32, 128 / action-head only |
| 训练范式 | SFT / on-policy RL / RL + old-task BC anchor |
| 分布偏移 | 同本体同场景 / 跨场景 / 跨本体 |
| 评估 | AVG, NBT, FWT, ZS, worst-task, AUC, memory, rollout cost |

如果算力有限，先用 LIBERO 组成 10/30/50 任务序列，完成全部 ablation；再在 RoboCasa/ManiSkill 做跨域复现；最后只选 5-10 个代表任务做真机验证。这比一开始就尝试 50 个真机任务更容易得到可归因的结果。

---

## 八、结论

对“学 50 个任务后，第 1 个任务经验所剩无几”，不应只用“把 buffer 变大”回答。目前更可靠的决策顺序是：

1. 先判断是联合多任务还是顺序 continual learning。
2. 对大型预训练 VLA，先测试 LoRA + on-policy RL 的 replay-free 基线。
3. 对 SFT/BC 或已观测到遗忘的 RL，使用按任务均衡、总预算固定的小 replay。
4. 当前任务用 on-policy RL，旧轨迹优先用于 BC/蒸馏锚定，避免过时数据污染 PPO/GRPO。
5. 如果任务冲突仍很强，再引入 task-family adapter、技能库或路由机制。
6. 用完整成功率矩阵、NBT、FWT、held-out 泛化和固定内存成本评价，而不是只看最终平均分。

目前最值得优先精读的三篇是：

1. [Simple Recipe Works](https://arxiv.org/abs/2603.11653)：直接回答 continual VLA-RL 应该怎么做。
2. [Pretrained VLAs are Surprisingly Resistant to Forgetting](https://arxiv.org/abs/2603.03818)：直接回答 replay 需要多大。
3. [LIBERO](https://arxiv.org/abs/2306.03310)：定义实验、任务序列和指标的基础。

