---
title: LifeLong-RFT：VLA 持续学习 RL 微调
order: 225
tags: [强化学习, VLA, 持续学习, Process Reward, Action Chunking, GRPO]
category: 精读
star: 4
---

# LifeLong-RFT：用可验证过程奖励做 VLA 持续强化微调

> **论文**: *Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning*<br>
> **作者**: Yuan Liu, Haoran Li, Shuai Tian, Yuxing Qin, Yuhui Chen, Yupeng Zheng, Yongzhen Huang, Dongbin Zhao<br>
> **版本**: arXiv:2602.10503v2, 2026-05-16<br>
> **项目页**: https://yuan-liu-lifelong-rft.github.io/<br>
> **综述上下文**: [持续/终身 VLA 强化学习综述](./S07_持续终身VLA强化学习综述)

---

## 一、它解决什么问题

VLA 通常通过专家示教做 SFT。当机器人需要用少量新示教持续学习新任务时，SFT 存在两个问题：

1. 对新任务的数据需求高。
2. 新任务的监督梯度可能覆盖旧任务能力。

LifeLong-RFT 将专家示教中的 observation/instruction 当作起点，从当前 VLA 采样多个 action chunk，再根据它们与参考动作的一致性构造可验证奖励，用 GRPO 做 chunk-level reinforcement fine-tuning。

需要准确理解的是：

- 它的奖励来自**专家动作标签与格式检查**，不是机器人在环境中完成任务后的 online success reward。
- 它不需要在训练期间与环境交互，也不需要额外预训练 reward model。
- 它仍然依赖已标注的参考轨迹，因此更接近“用 RL 目标改造的示教后训练”，不是无示教的纯在线 RL。

---

## 二、Chunk-level GRPO

对一个 observation $o$ 和语言指令 $l$，从旧策略采样 $G$ 个 action chunk：

$$
\{a_i\}_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot\mid o,l).
$$

每个 chunk 得到可验证奖励 $r_i$，组内标准化得到 advantage：

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}.
$$

优化目标采用 GRPO/PPO 式 clipping，并加入参考策略 KL 正则，限制策略过度偏移。不训练显式 critic，因此比大型 VLA 的 actor-critic PPO 更省显存。

---

## 三、三维可验证过程奖励

论文 v2 中的三个奖励是 **QACR、CTAR 和 FCR**，而不是环境进度、安全性和效率奖励。

### 3.1 QACR：量化动作一致性

Quantized Action Consistency Reward 比较预测动作 token $a$ 与参考 token $\tilde a$ 在各个位置是否匹配：

$$
R_{\mathrm{QACR}}=
\frac{\sum_{k=1}^{\min(U,V)}\mathbb{I}(a_k=\tilde a_k)}
{\max(U,V)}.
$$

无法通过 Fast+ tokenizer 格式检查的输出直接得 0 分。

### 3.2 CTAR：连续轨迹对齐

Continuous Trajectory Alignment Reward 先将 action token 解码回连续 action chunk，然后同时比较 pose 和 gripper：

$$
d_t=\frac{1}{D}\lVert y_t^{pose}-\tilde y_t^{pose}\rVert_1,
\qquad r_t^{pose}=\exp(-\alpha d_t),
$$

$$
r_t^{grip}=\mathbb{I}(y_t^{grip}=\tilde y_t^{grip}),
$$

$$
R_{\mathrm{CTAR}}=\frac{1}{H}\sum_{t=1}^{H}
\left[\beta r_t^{pose}+(1-\beta)r_t^{grip}\right].
$$

这一项是论文最重要的奖励。消融实验中去掉 CTAR 后，LIBERO 平均成功率从 95.6% 降到 4.7%。

### 3.3 FCR：格式合规

Format Compliance Reward 是二值奖励：

$$
R_{\mathrm{FCR}}=\mathbb{I}[\text{action chunk 尺寸和动作维度合法}].
$$

它防止自回归 VLA 生成不可解码的动作 token 序列。

### 3.4 总奖励

$$
r=\omega R_{\mathrm{QACR}}
+(1-\omega)R_{\mathrm{CTAR}}
+\lambda R_{\mathrm{FCR}}.
$$

论文使用 $\omega=0.7$ 和 $\lambda=0.1$。

---

## 四、持续学习设置

LifeLong-RFT 的 LIBERO continual learning 实验遵循 LOTUS 设定：

1. 在每个 LIBERO suite 的前 6 个任务上做 base-task 训练，每任务 50 条示教。
2. 后 4 个任务依次到来，每个新任务只用 10 条示教。
3. 对每个已学任务保留 5 条示教做 Experience Replay。
4. 每学完一个新任务，评估所有已学任务，计算 FWT、NBT 和 AUC。

因此，LifeLong-RFT 并不是 replay-free continual learning，也没有在 50 个顺序任务上验证。它解决的是“有少量旧示教和少量新示教时，如何用 RFT 提高适应效率并减少遗忘”。

---

## 五、实验结论

### 5.1 联合多任务学习

以 NORA-Long 为 backbone，LifeLong-RFT 在 LIBERO Object/Spatial/Goal/Long 上的成功率分别为 99.2%、98.2%、95.8% 和 89.0%，平均 95.6%；对应 SFT baseline 平均为 91.8%。

### 5.2 持续学习

在四个 LIBERO suite 的 6+4 设定中，LifeLong-RFT 相对 NORA-Long SFT 同时改善了 FWT、NBT 和 AUC。其中 LIBERO-Goal 的 AUC 提升 35.9 个点。

论文摘要中的“平均成功率提升 22%”是指 continual LIBERO 实验中相对 SFT 的总体改善，不应解读为已在 50 任务上证明了 22% 收益。

### 5.3 真实机器人

在 Pick Banana、Pick Bread、Pull Drawer 和 Hang Chinese Knot 四个任务上，RFT 平均成功率为 87.5%，对应 NORA-Long SFT 为 78.8%。持续设定中，新任务使用 20 条示教，每个旧任务保留 5 条示教。

---

## 六、优点、局限与正确定位

| 维度 | 评价 |
|---|---|
| 核心贡献 | 把动作 token 正确性、连续控制误差和格式合法性统一为 chunk-level 可验证奖励 |
| 数据效率 | 对新任务仅使用标准 SFT 数据量的 20% |
| 训练成本 | 无 critic，无训练期环境交互，但论文实验仍使用 8 张 NVIDIA H20 |
| 防遗忘机制 | 小量 Experience Replay + 参考策略 KL + RFT 的选择性更新 |
| 主要局限 | 奖励依赖参考动作；continual 阶段只有 4 个新任务；不能替代真正环境奖励的在线自我改进 |

LifeLong-RFT 最适合的场景是：已有预训练离散动作 VLA 和少量示教，希望在不重新访问环境的条件下，比普通 SFT 更有效地适应新任务。如果研究问题是 50 个任务的长期 on-policy RL，它应作为有限 replay 基线，而不是已完成的最终解法。
