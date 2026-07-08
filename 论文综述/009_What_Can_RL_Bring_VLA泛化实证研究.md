---
title: What Can RL Bring to VLA Generalization? 实证研究
order: 209
tags: [强化学习, VLA, PPO, GRPO, DPO, 泛化性]
category: 精读
star: 4
---

# What Can RL Bring to VLA Generalization? 实证研究精读

> **论文标题**: What Can RL Bring to VLA Generalization? An Empirical Study  
> **作者**: Yanjiang Guo, Jianke Zhang, Xiaoyu Chen, Xiang Ji, Yen-Jen Wang, Yucheng Hu, Jianyu Chen  
> **机构**: Tsinghua University, Shanghai AI Lab  
> **发表**: NeurIPS 2025 (arXiv:2505.19789)  
> **代码**: https://rlvla.github.io/

**标签**: `#VLA` `#强化学习` `#PPO` `#GRPO` `#DPO` `#泛化性` `#OOD` `#实证研究`

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 算法基础
- [GRPO](/前置知识/000m_前置知识_GRPO_Group_Relative_Policy_Optimization) — GRPO 的原理
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — SFT vs RL 的基本对比
- [KL 散度与策略约束](/前置知识/000j_前置知识_KL散度与策略约束) — DPO 和 KL 约束
- [动作 Token 化与自回归策略](/前置知识/000l_前置知识_动作Token化与自回归策略) — 自回归 VLA 基础
- [VLA-RL 精读](./006_VLA_RL_PPO直接训练自回归VLA) — PPO 训 VLA 的具体方案
- [RIPT-VLA 精读](./007_RIPT_VLA_无Critic的VLA后训练) — GRPO 路线的代表

---

## 一、核心问题

这篇论文不是提出新方法，而是回答一个基本问题：

> **RL 微调到底能给 VLA 带来什么样的泛化能力提升？相比 SFT，RL 的优势具体体现在哪些维度？**

这是 NeurIPS 2025 上第一篇系统性的 VLA RL 实证研究。

### 1.1 为什么需要这项研究

2025 年上半年出现了一批 VLA + RL 的工作（VLA-RL、RIPT-VLA、SimpleVLA-RL 等），但它们都各自用不同的 benchmark、不同的 baseline、不同的评估指标。缺少一个**统一框架下的公平对比**来回答：

1. RL 比 SFT 好在哪里？（不只是"成功率高几个百分点"，而是泛化的哪个维度）
2. 不同 RL 算法（PPO vs GRPO vs DPO）之间到底谁更好？
3. RL 的好处在什么条件下最显著？什么条件下没用？

### 1.2 三个泛化维度

论文定义了 VLA 泛化性的三个正交维度：

| 维度 | 含义 | 具体扰动 |
|------|------|---------|
| **语义泛化** (Semantic) | 理解不同表述的指令 | 同义指令（"grab the cup" vs "pick up the mug"） |
| **执行泛化** (Execution) | 处理初始状态变化 | 物体位置偏移、机器人起始位姿变化 |
| **视觉泛化** (Visual) | 处理视觉外观变化 | 光照变化、相机角度变化、背景改变、干扰物 |

---

## 二、实验设置

### 2.1 统一框架

所有方法使用：
- **同一个 base model**：OpenVLA-7B（LoRA 微调）
- **同一个 benchmark**：LIBERO（40 个操作任务）
- **同一组超参数搜索空间**
- **同一套评估协议**

### 2.2 对比方法

| 方法 | 类型 | 是否需要环境交互 | 是否需要 Critic |
|------|------|----------------|---------------|
| SFT | 监督学习 | 否 | 否 |
| DPO | 偏好学习 | 否（用离线数据） | 否 |
| GRPO | 在线 RL | 是 | 否 |
| **PPO** | 在线 RL | 是 | **是** |

### 2.3 评估协议

**In-Domain (ID) 测试**：和训练时相同的任务条件

**Out-of-Distribution (OOD) 测试**：分三个维度分别施加扰动：
- Semantic OOD：用 GPT-4 生成语义等价但措辞不同的指令
- Execution OOD：在测试时把物体位置随机偏移 ±3-5cm
- Visual OOD：改变光照方向、添加背景纹理、微调相机视角

---

## 三、核心发现

### 3.1 发现一：PPO 显著优于 GRPO 和 DPO

| 方法 | ID 成功率 | OOD-Semantic | OOD-Execution | OOD-Visual | OOD 平均 |
|------|----------|-------------|---------------|-----------|---------|
| SFT | 76.5% | 68.2% | 52.3% | 70.1% | 63.5% |
| DPO | 79.2% | 71.4% | 55.0% | 72.8% | 66.4% |
| GRPO | 80.1% | 72.0% | 58.7% | 71.5% | 67.4% |
| **PPO** | **81.0%** | **75.3%** | **63.8%** | **71.2%** | **70.1%** |

**关键观察**：
- PPO 在 ID 和 OOD 上都是最好的
- PPO 在 Execution OOD 上优势最大（+5.1% vs GRPO，+11.5% vs SFT）
- 在 Visual OOD 上所有方法差距不大（RL 对视觉泛化帮助有限）

### 3.2 发现二：RL 的泛化收益不均匀

论文最重要的发现是：**RL 对三个维度的泛化提升是不对称的**。

| 泛化维度 | RL vs SFT 的提升幅度 | 解释 |
|---------|--------------------|----|
| **Execution（执行）** | 大幅提升（+11.5%） | RL 的在线交互让策略学会了"纠错" |
| **Semantic（语义）** | 中等提升（+7.1%） | RL 训练中策略对指令理解更鲁棒 |
| **Visual（视觉）** | 几乎无提升（+1.1%） | 视觉表征主要由预训练决定，RL 不改善视觉编码器 |

**为什么 Execution 提升最大？**

SFT 策略是"开环"的——按记忆执行，遇到偏差不会纠正。RL 训练中策略必须处理各种初始状态的变化，天然学到了闭环纠错能力。

**代入例子**：
- SFT 策略：记住"向右移 10cm 抓取" → 如果杯子偏左了 3cm → 抓空
- PPO 策略：学会"先感知杯子在哪，再决定移多少" → 杯子偏了也能抓到

**为什么 Visual 提升很小？**

VLA 的视觉理解能力主要来自预训练的视觉编码器（SigLIP + DinoV2）。RL 微调时通常冻结视觉编码器（否则训练不稳定），所以视觉泛化不会被 RL 改善。

### 3.3 发现三：PPO 比 GRPO 好的原因

论文做了深入分析来解释为什么 PPO > GRPO：

**原因 1：Credit Assignment**

- PPO 有 Critic，可以把终端奖励通过 GAE 分配到每一步 → 步级别的梯度信号
- GRPO 把整条轨迹的 reward 均匀分配给所有 token → 信号极度稀释

**代入数字**：一条 50 步的轨迹，每步 7 个 token = 350 个 token 决策。
- PPO：Critic 可以识别"第 23 步的动作特别关键"，给它更大的 advantage
- GRPO：350 个 token 共享同一个 advantage 值，无法区分重要性

**原因 2：稀疏奖励的利用效率**

- PPO 的 Critic 可以从失败轨迹中学到"这个状态的 value 很低" → 间接信号
- GRPO 对失败轨迹只有一个全局的负 advantage → 没有状态级别的区分

**原因 3：训练稳定性**

PPO 的 Critic warmup 和 GAE 组合提供了低方差的梯度估计。GRPO 的梯度方差更大，需要更大的 batch size 才能稳定。

### 3.4 发现四：VLA warmup 的关键作用

论文验证了一个重要的工程细节：**在开始 RL 前，先用 SFT 目标 warmup 几步至关重要**。

| 设置 | 最终 ID 成功率 |
|------|--------------|
| 直接开始 PPO（无 warmup） | 72.3%（甚至比 SFT 差） |
| warmup 500 步后开 PPO | 78.5% |
| warmup 2000 步后开 PPO | **81.0%** |

**原因**：如果 Critic 初始估计太差，GAE 计算出的 advantage 是噪声，反而误导策略更新。warmup 让 Critic 先有了合理的 value landscape。

### 3.5 发现五：PPO epoch 数不宜太大

| PPO epochs/batch | 最终性能 |
|-----------------|---------|
| 1 | 79.2% |
| **2** | **81.0%** |
| 4 | 78.1% |
| 8 | 74.3%（过拟合） |

**和 LLM RLHF 的区别**：LLM 通常用 4 个 PPO epoch。VLA 只能用 1-2 个——因为机器人任务的数据量更小、分布更窄，多次重用同一 batch 容易过拟合。

---

## 四、实践建议总结

论文在最后给出了一系列实践建议：

### 4.1 算法选择

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 有充足 GPU（4×A100 80G） | PPO | 性能最好 |
| GPU 有限（2×A100 40G） | GRPO | 无 Critic，省一半显存 |
| 无法在线交互（只有离线数据） | DPO | 完全离线 |
| 极少示教（1-5 条） | PPO 或 GRPO | RL 的探索弥补数据不足 |

### 4.2 关键超参数

| 超参数 | 推荐值 | 敏感度 |
|--------|--------|--------|
| 学习率 | 2e-5 (LoRA) | **高**（大了会崩） |
| PPO clip ε | 0.2 | 中 |
| PPO epochs | 1-2 | **高**（多了会过拟合） |
| KL penalty β | 0.01-0.05 | 中 |
| Critic warmup 步数 | 1000-2000 | **高** |
| 采样温度 | 1.0-1.5 | 中 |
| GAE λ | 0.95 | 低 |
| 折扣因子 γ | 0.99 | 低 |

### 4.3 提升视觉泛化的建议

论文指出 RL 无法提升视觉泛化，但给出了替代方案：
- 在 RL 训练环境中加入 visual domain randomization（光照、纹理等随机化）
- 使用更强的预训练视觉编码器
- 考虑 vision augmentation 技术

---

## 五、和其他工作的关系

### 5.1 验证了 VLA-RL 的结论

VLA-RL 声称 PPO 是训练自回归 VLA 的最佳算法——本文用更严格的实验验证了这一点。

### 5.2 解释了 RIPT-VLA 的成功与局限

RIPT-VLA 用 RLOO（GRPO 变体）取得了好结果——但本文指出在稀疏奖励下 PPO 更好。RIPT-VLA 的成功更多归因于 dynamic rejection 策略和 few-shot 场景下 VLA 预训练知识的激活，而不是 GRPO 算法本身的优势。

### 5.3 指明了未来方向

- RL 无法提升视觉泛化 → 需要新方法（如 RAPT 的 robustness-aware 训练）
- Execution 泛化是 RL 的主要贡献维度 → 应该在评估中更重视这一点
- PPO > GRPO 但成本更高 → 需要更高效的 Actor-Critic 架构

---

## 六、个人评价

### 6.1 贡献

这是 VLA + RL 方向急需的一篇"整理性"工作。在众多声称"我的方法最好"的论文中，一篇公平、系统的实证比较极有价值。NeurIPS 2025 接收说明了社区对此类工作的需求。

### 6.2 核心洞察

最深刻的发现是 **"RL 的泛化收益是不对称的"**：
- Execution 泛化 ↑↑↑
- Semantic 泛化 ↑↑
- Visual 泛化 ≈ 0

这为后续研究指明了方向——如果要提升视觉鲁棒性，RL 不是答案，需要在视觉表征层面做工作。

### 6.3 实践价值

这篇论文的超参数推荐和工程建议对于想实践 VLA + RL 的团队非常有价值。尤其是 "PPO epochs 不超过 2" 和 "Critic warmup 是必须的" 这两个结论，可以帮助很多人避免踩坑。

---

## 延伸阅读

- [VLA-RL 精读](./006_VLA_RL_PPO直接训练自回归VLA) ← PPO 训 VLA 的详细方案
- [RIPT-VLA 精读](./007_RIPT_VLA_无Critic的VLA后训练) ← GRPO 路线的代表
- [GRPO 前置知识](/前置知识/000m_前置知识_GRPO_Group_Relative_Policy_Optimization) ← GRPO 算法详解
- [VLA 模型的 RL 后训练综述](/论文综述/S06_VLA模型的RL后训练综述) ← 方法全景图
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) ← PPO 的完整推导
