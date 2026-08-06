---
title: InvariantStock：不变特征学习选股
order: 185
tags: [量化交易, 不变学习, 因果推断, 分布漂移]
category: 精读
star: 4
---

# InvariantStock：跨环境不变特征学习 深度精读

> **论文标题**: InvariantStock: Learning Invariant Features for Mastering the Shifting Market  
> **作者**: Haiyao Cao, Jinan Zou, Yuhang Liu, et al.  
> **机构**: University of Electronic Science and Technology of China (UESTC)  
> **发表**: TMLR 2024 (Transactions on Machine Learning Research)  
> **链接**: https://openreview.net/forum?id=dtNEvUOZmA

**知识链接**：
- [分布漂移与在线适配](/前置知识/001z_前置知识_分布漂移与在线适配) — InvariantStock 解决的核心问题
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — 评估指标
- [Walk-Forward 滚动回测](/前置知识/001x_前置知识_Walk_Forward滚动回测) — 回测框架
- [DoubleAdapt 精读](/论文综述/084_DoubleAdapt_分布漂移双重适配) — 另一条解决漂移的路线（适配 vs 不变）

---

## 贯穿全文的例子

> 2019 年 A 股：核心资产行情。高 ROE + 低波动 + 大市值的股票大涨。
> 2022 年 A 股：小微盘行情。小市值 + 高波动 + 低关注度的股票大涨。
> 
> 问题："高 ROE → 涨"在 2019 年成立，在 2022 年不成立——这是环境依赖的"虚假相关"。
> 
> InvariantStock 的目标：找到在 2019 和 2022 年**都**预示上涨的特征模式——这些才是"因果性"的不变特征。比如"相对同行业的盈利增速"可能在任何环境下都和超额收益正相关。

---

## 一、核心思想：不变风险最小化（IRM）

### 1.1 从因果推断到选股

因果推断有一个关键概念：**虚假相关（spurious correlation）** vs **因果关系（causal relation）**。

- 虚假相关：冰淇淋销量和溺水人数正相关——不是因为吃冰淇淋导致溺水，而是都和"夏天"有关。改变季节（环境），相关性消失。
- 因果关系：温度升高 → 游泳人数增加 → 溺水增加。无论在哪个城市（环境），这个关系都成立。

**映射到选股**：
- 虚假相关："高 ROE → 涨"在 2019-2021 成立，因为那时资金偏好蓝筹。2022 年资金转向小盘后，这个相关消失。
- 因果关系："盈利超预期 → 涨"在任何市场环境中都倾向于成立。

### 1.2 IRM 的数学框架

设 $\mathcal{E} = \{e_1, e_2, \ldots, e_K\}$ 为一组环境（如按年份/市场状态划分）。

标准 ERM（经验风险最小化）：

$$
\min_\Phi \sum_{e \in \mathcal{E}} L^e(\Phi)
$$

> 在所有环境的数据上最小化总损失——问题是它会利用虚假相关（在某些环境中有效但在其他环境中失效的模式）。

IRM（不变风险最小化）增加一个约束：

$$
\min_\Phi \sum_{e \in \mathcal{E}} L^e(\Phi) + \lambda \sum_{e \in \mathcal{E}} \left\|\nabla_{w|w=1.0} L^e(w \cdot \Phi)\right\|^2
$$

> 第二项要求：对于每个环境 $e$，最优的 classifier（线性头）都是 $w=1.0$。如果 $\Phi$ 只学了不变特征，那么同一个线性头在所有环境中都最优；如果 $\Phi$ 学了虚假相关，在某些环境中 $w=1.0$ 就不是最优的。

**逐项拆解**：
- $\Phi$：特征提取器（如 GRU/Transformer backbone），把原始因子映射到高维表示
- $L^e(\Phi)$：在环境 $e$ 中，用 $\Phi$ 提取的特征做预测的损失
- $w$：一个标量缩放因子，初始化为 1.0
- $\nabla_{w|w=1.0} L^e(w \cdot \Phi)$：在 $w=1.0$ 处的梯度。如果 $w=1.0$ 是最优解，这个梯度为 0
- $\lambda$：惩罚强度。越大越强调"不变性"，但可能牺牲整体性能

**直觉**：如果 $\Phi$ 学到了一个"只在环境 $e_1$ 有用"的特征，那么在 $e_1$ 中 $w=1.0$ 的梯度为 0，但在 $e_2$ 中梯度不为 0（因为这个特征在 $e_2$ 无用，最优 $w$ 不是 1.0）。惩罚项会抑制这种环境特异性特征。

### 1.3 数值例子

假设 $\Phi$ 提取了 2 维特征 $z = (z_1, z_2)$：
- $z_1$："相对行业盈利增速"——在所有环境中都和超额收益正相关
- $z_2$："绝对 ROE 高低"——只在蓝筹行情中和超额收益正相关

**在牛市环境 $e_1$**：预测 = $w_1 z_1 + w_2 z_2$，最优 $(w_1, w_2) = (1.0, 1.0)$  
**在熊市环境 $e_2$**：预测 = $w_1 z_1 + w_2 z_2$，最优 $(w_1, w_2) = (1.0, -0.5)$

$z_2$ 在两个环境中最优权重不同（1.0 vs -0.5）→ IRM 惩罚项会"杀掉" $z_2$，只保留 $z_1$。

---

## 二、InvariantStock 的具体设计

### 2.1 环境划分

InvariantStock 需要定义"环境"。论文使用了三种划分方式：

1. **时间划分**：按年份或按季度切分（如 2019/2020/2021/2022 各为一个环境）
2. **市场状态划分**：根据指数收益率/波动率聚类为"牛市/熊市/震荡市"
3. **混合划分**：时间 × 市场状态的交叉

**为什么环境划分很关键？** 如果环境划分太粗（只分 2 个），IRM 约束太弱；如果太细（每周一个环境），每个环境样本太少。论文推荐 4-8 个环境。

### 2.2 模型结构

InvariantStock 的网络分为三部分：

$$
\hat{y}_i = h(\Phi(\mathbf{x}_i))
$$

- **特征提取器 $\Phi$**：GRU 或 Transformer，把原始特征 $\mathbf{x}_i \in \mathbb{R}^{T \times d}$ 映射到 $\mathbf{z}_i \in \mathbb{R}^{d_z}$
- **不变性约束**：施加在 $\mathbf{z}_i$ 上，确保 $\mathbf{z}$ 在所有环境中都有效
- **预测头 $h$**：一个共享的线性层，从 $\mathbf{z}$ 预测收益

### 2.3 完整损失函数

$$
L = \underbrace{\sum_{e \in \mathcal{E}} \frac{1}{|D^e|} \sum_{(x,y) \in D^e} \ell(h(\Phi(x)), y)}_{\text{预测损失（所有环境）}} + \lambda \underbrace{\sum_{e \in \mathcal{E}} \left\|\nabla_{w|w=1.0} L^e(w \cdot \Phi)\right\|^2}_{\text{不变性正则}}
$$

> 第一项让模型在所有环境中都能预测好；第二项让模型只使用跨环境一致有效的特征。

**逐项拆解**：
- $D^e$：属于环境 $e$ 的数据集
- $\ell(\cdot, \cdot)$：MSE 或 Ranking Loss
- $\lambda$：权衡预测性能和不变性的超参数。论文中搜索范围 $[10^{-4}, 10^2]$

### 2.4 训练细节

InvariantStock 还加入了两个辅助技巧：

**1. 环境对比损失**（让不变特征在不同环境间对齐）：
$$
L_{\text{contrast}} = -\log \frac{\exp(\text{sim}(z_i^{e_1}, z_j^{e_2}) / \tau)}{\sum_k \exp(\text{sim}(z_i^{e_1}, z_k^{e_2}) / \tau)}
$$

同一只股票在不同环境下的表示应该相似（正对），不同股票的表示应该不同（负对）。

**2. 环境感知的数据增强**：通过对因子添加环境特异性噪声来模拟更多的环境变体。

---

## 三、实验结果

### 3.1 CSI300 主实验（2020-2022 测试期）

| 模型 | 年化收益 | Sharpe | 最大回撤 | 换手率 |
|------|----------|--------|----------|--------|
| LSTM | 21.35% | 1.12 | -28.42% | 45% |
| GRU | 24.67% | 1.28 | -25.61% | 42% |
| Transformer | 18.92% | 0.95 | -31.05% | 48% |
| HIST | 32.41% | 1.67 | -22.18% | 38% |
| MASTER | 38.56% | 1.93 | -19.87% | 35% |
| **InvariantStock** | **83.15%** | **3.72** | **-18.78%** | **32%** |

测试条件：单边 15bps 交易成本，Top-30 等权多头策略。

### 3.2 结果分析

InvariantStock 的结果**异常突出**——年化 83.15% 和 Sharpe 3.72 远超所有其他方法。需要注意：

| 关注点 | 说明 |
|--------|------|
| **测试期特殊性** | 2020-2022 恰好是 A 股风格剧烈切换期，不变特征学习的优势最大化 |
| **环境划分可能有事后偏见** | 如果环境是事后定义的（知道 2022 是熊市），不变性约束可能"偷看"了信息 |
| **样本外泛化未验证** | 论文只测了一个时间段，没有多个滚动期的稳健性验证 |
| **但确实发表在正规期刊** | TMLR 有同行评审，结果至少通过了审稿人的检查 |

**结论**：InvariantStock 的思路很有价值，但报告的绝对数字需要独立复现来验证。在自己的数据上复现时，如果能达到其他方法的 1.5-2 倍就算成功。

### 3.3 消融实验

| 变体 | 年化收益 | 说明 |
|------|----------|------|
| 完整 InvariantStock | 83.15% | 全部组件 |
| 去掉 IRM 正则 | 45.23% | 退化为普通训练 |
| 去掉环境对比损失 | 62.41% | 不变性约束减弱 |
| 随机环境划分 | 38.67% | 环境划分无意义 |
| ERM baseline（无环境） | 24.67% | 最基础的 GRU |

**关键发现**：IRM 正则贡献最大（从 45% 到 83%），环境划分的质量也极其重要（随机划分只有 38%）。

---

## 四、InvariantStock vs DoubleAdapt

| 维度 | InvariantStock | DoubleAdapt |
|------|---------------|-------------|
| **解决漂移的思路** | 学习"不变特征"，根本不受漂移影响 | 学习"如何适配"，跟着漂移走 |
| **需要在线更新？** | 不需要（训练时一次性学好不变表示） | 需要（每天用适配器调整） |
| **训练时需要环境标注** | **是**（关键依赖） | 不需要 |
| **推理复杂度** | 与普通模型相同 | 额外 +12% |
| **适应剧烈漂移** | 强（已经是不变特征） | 中（适配器有容量限制） |
| **适应缓慢渐变** | 中（不变特征可能过于保守） | 强（持续微调） |
| **可复现性** | 较难（环境划分敏感） | 较容易 |

**建议**：
- 如果能合理定义环境 → 先试 InvariantStock
- 如果环境定义不明确 → 用 DoubleAdapt 更安全
- 最优方案可能是：InvariantStock 的特征提取器 + DoubleAdapt 的在线适配

---

## 五、实操建议

### 5.1 环境划分的实用方法

```python
import numpy as np
from sklearn.cluster import KMeans

def define_environments(market_returns, n_envs=4):
    """根据市场指数收益率聚类来定义环境"""
    # 用 20 天滚动收益 + 波动率作为聚类特征
    features = np.column_stack([
        market_returns.rolling(20).mean(),     # 趋势
        market_returns.rolling(20).std(),       # 波动
        market_returns.rolling(60).mean(),      # 长期趋势
    ]).dropna()
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_envs, random_state=42)
    env_labels = kmeans.fit_predict(features)
    
    # 每个聚类就是一个"环境"
    return env_labels
```

### 5.2 IRM 正则项的实现

```python
import torch

def irm_penalty(losses_by_env):
    """
    计算 IRM 惩罚项
    losses_by_env: list of (loss_tensor, env_predictions)
    """
    penalty = 0.0
    for loss_e in losses_by_env:
        # 创建一个 dummy scalar w=1.0
        w = torch.tensor(1.0, requires_grad=True)
        # 计算 w * loss 对 w 的梯度
        grad = torch.autograd.grad(
            w * loss_e, w, create_graph=True
        )[0]
        # 惩罚项 = 梯度的平方
        penalty += grad ** 2
    return penalty
```

---

## 六、总结

| 要点 | 内容 |
|------|------|
| 核心思想 | 学习跨市场环境不变的特征表示 |
| 理论基础 | 不变风险最小化（IRM），来自因果推断 |
| 环境划分 | 按时间/市场状态聚类定义 4-8 个环境 |
| 报告结果 | CSI300 2020-2022：年化 83.15%, Sharpe 3.72（需独立复现） |
| 关键依赖 | 环境划分的质量——随机划分效果骤降 |
| 建议 | 思路很好，但结果过于突出，必须在自己的数据上独立验证 |

---

## 延伸阅读

- [分布漂移与在线适配](/前置知识/001z_前置知识_分布漂移与在线适配) — 漂移的全面介绍
- [DoubleAdapt 精读](/论文综述/084_DoubleAdapt_分布漂移双重适配) — 适配路线的代表方法
- [MASTER 精读](/论文综述/083_MASTER_市场感知截面选股Transformer) — 截面关系建模的强基座
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — 评估指标体系
