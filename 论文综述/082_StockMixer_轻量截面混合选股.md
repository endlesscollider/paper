---
title: StockMixer：轻量截面混合选股
order: 182
tags: [量化交易, 截面模型, MLP-Mixer]
category: 精读
star: 4
---

# StockMixer：Perceiving Market Dynamics via MLP-Mixer 深度精读

> **论文标题**: StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price Prediction  
> **作者**: Jinyong Kim, Junseok Lee, Yongjae Lee  
> **机构**: UNIST (蔚山科学技术院)  
> **发表**: AAAI 2024  
> **DOI**: https://doi.org/10.1609/aaai.v38i8.28681

**知识链接**：
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — IC/RankIC 的定义与截面建模思路
- [Walk-Forward 滚动回测](/前置知识/001x_前置知识_Walk_Forward滚动回测) — StockMixer 的评估框架
- [LightGBM 在量化选股中的应用](/前置知识/001y_前置知识_LightGBM在量化选股中的应用) — StockMixer 需要超越的基线
- [MASTER 精读](/论文综述/083_MASTER_市场感知截面选股Transformer) — 更复杂的截面关系建模方案（对比）

---

## 贯穿全文的例子

> 美股市场中有 N=487 只股票（NASDAQ 指数成分股）。
> 每只股票有 $d=8$ 维特征（开盘/最高/最低/收盘/成交量/5日MA/10日MA/20日MA），观察过去 $T=10$ 天。
> 
> 输入是一个三维张量：$\mathbf{X} \in \mathbb{R}^{N \times T \times d} = \mathbb{R}^{487 \times 10 \times 8}$
> 
> 目标：预测每只股票未来 1 天的收益率排序，选 Top-K 买入。

---

## 一、背景与动机

### 1.1 现有方法的问题

2024 年之前的截面选股深度模型（如 GNN、图 Transformer）有一个共同缺陷：

1. **复杂但没必要**：用 attention/GNN 建模股票关系，参数多、训练慢、容易过拟合
2. **图结构依赖先验**：需要预定义股票关系图（行业、供应链），图结构不准会误导模型
3. **时间建模 + 截面建模耦合**：同时学两种模式，优化困难

StockMixer 的核心观察：**MLP-Mixer 那种"分维度混合"的简单架构就够了**。

### 1.2 MLP-Mixer 的启示

MLP-Mixer（Google 2021）在图像领域证明：不需要 attention，只用 MLP 沿不同维度交替混合就能达到 ViT 级别的性能。核心操作：
- Token-Mixing MLP：不同 patch 之间交换信息
- Channel-Mixing MLP：同一个 patch 内部不同通道交换信息

StockMixer 把这个思想搬到股票预测：
- **Stock-Mixing**：不同股票之间交换信息 → 建模截面关系
- **Time-Mixing**：同一只股票不同时间步交换信息 → 建模时序动态
- **Indicator-Mixing**：同一时间步不同特征交换信息 → 建模因子交互

---

## 二、模型架构

### 2.1 输入表示

输入张量 $\mathbf{X} \in \mathbb{R}^{N \times T \times d}$ 有三个维度：
- $N$：股票数量（截面维度）
- $T$：时间步数（时序维度）
- $d$：特征数量（因子维度）

### 2.2 三种 Mixing 操作

StockMixer 的核心是三个 MLP 层，分别沿三个维度做信息混合：

**Stock-Mixing**（沿第 1 维混合）：

$$
\mathbf{Z}^{(s)} = \sigma\left(\mathbf{X}^{\top_N} \mathbf{W}_s + \mathbf{b}_s\right)^{\top_N}
$$

这里的操作是：把张量转置使得"股票"维度变成最后一维，然后用一个 $N \times N$ 的线性层让不同股票之间交换信息，最后转置回来。

> Stock-Mixing 让模型知道"A 股票今天涨了，这对 B 股票意味着什么"——不需要预定义关系图，直接从数据中学习股票间的联动模式。

**Time-Mixing**（沿第 2 维混合）：

$$
\mathbf{Z}^{(t)} = \sigma\left(\mathbf{Z}^{(s) \top_T} \mathbf{W}_t + \mathbf{b}_t\right)^{\top_T}
$$

> Time-Mixing 让模型学到"过去 10 天中哪几天的信息对预测最重要"——比如前一天的收益率权重可能最大。

**Indicator-Mixing**（沿第 3 维混合）：

$$
\mathbf{Z}^{(d)} = \sigma\left(\mathbf{Z}^{(t)} \mathbf{W}_d + \mathbf{b}_d\right)
$$

> Indicator-Mixing 让模型学到因子间的交互——比如"高成交量 + 价格突破 MA20 = 强信号"。

### 2.3 完整前向传播

三次 mixing 之后，用一个 prediction head 输出每只股票的预测分：

$$
\hat{\mathbf{y}} = \text{Flatten}(\mathbf{Z}^{(d)}) \cdot \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \in \mathbb{R}^N
$$

整个模型就是**三个线性层 + 激活函数 + 一个输出层**——参数量极少。

### 2.4 数值维度追踪

用贯穿例子的数据追踪维度变化：

| 步骤 | 输入维度 | 操作 | 输出维度 | 参数量 |
|------|----------|------|----------|--------|
| 输入 | $487 \times 10 \times 8$ | — | — | — |
| Stock-Mixing | $(10 \times 8) \times 487$ | $\mathbf{W}_s \in \mathbb{R}^{487 \times 487}$ | $(10 \times 8) \times 487$ | 237K |
| Time-Mixing | $(487 \times 8) \times 10$ | $\mathbf{W}_t \in \mathbb{R}^{10 \times 10}$ | $(487 \times 8) \times 10$ | 100 |
| Indicator-Mixing | $487 \times 10 \times 8$ | $\mathbf{W}_d \in \mathbb{R}^{8 \times 8}$ | $487 \times 10 \times 8$ | 64 |
| Flatten + Output | $487 \times 80$ | $\mathbf{W}_{\text{out}} \in \mathbb{R}^{80 \times 1}$ | $487$ | 80 |
| **总计** | | | | **~238K** |

注意 Stock-Mixing 的参数最多（因为 $N=487$），但整个模型仍然很小。

---

## 三、损失函数

### 3.1 收益率回归 + 排名正则

StockMixer 使用混合损失：

$$
L = L_{\text{MSE}} + \alpha \cdot L_{\text{rank}}
$$

**MSE Loss**：
$$
L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

**排名损失**（三元组形式）：
$$
L_{\text{rank}} = \frac{1}{|P|}\sum_{(i,j) \in P} \max\left(0, -(\hat{y}_i - \hat{y}_j)(y_i - y_j) + m\right)
$$

> 对于每一对实际收益 $y_i > y_j$ 的股票对 $(i,j)$，要求预测值也满足 $\hat{y}_i > \hat{y}_j$，且差距至少为 margin $m$。

**为什么需要排名损失？** 因为最终策略是选 Top-K，只关心排序正确性。MSE 优化绝对值精度，可能在排序上不是最优（比如把所有预测值压缩到一个很小的范围内，MSE 很小但排序信息丢失）。

### 3.2 数值例子

假设 3 只股票的情况：
- 预测 $\hat{y} = (0.5, 0.3, 0.8)$，实际 $y = (0.02, -0.01, 0.03)$
- 正确对：$(C, A), (C, B), (A, B)$（实际排序 C > A > B）
- 检查对 $(C, A)$：$(\hat{y}_C - \hat{y}_A)(y_C - y_A) = (0.8-0.5)(0.03-0.02) = 0.003 > 0$ ✓
- 检查对 $(A, B)$：$(\hat{y}_A - \hat{y}_B)(y_A - y_B) = (0.5-0.3)(0.02-(-0.01)) = 0.006 > 0$ ✓

所有对都正确排序 → $L_{\text{rank}} = 0$。

---

## 四、实验结果

### 4.1 主实验（美股三大指数）

| 指标 | StockMixer | LSTM | GRU | Transformer | GAT |
|------|------------|------|-----|-------------|-----|
| NASDAQ Sharpe | **1.465** | 0.872 | 0.931 | 0.756 | 1.021 |
| NYSE Sharpe | **1.454** | 0.812 | 0.867 | 0.698 | 0.923 |
| S&P500 Sharpe | **1.586** | 0.901 | 0.956 | 0.812 | 1.102 |

**关键发现**：
1. StockMixer 以极简架构超越了所有复杂方法（GNN、Attention、图网络）
2. Transformer 表现最差——再次验证了"Transformer 不是量化默认最优解"
3. Sharpe > 1.4 意味着风险调整后的收益非常可观

### 4.2 消融实验

| 变体 | Sharpe | 说明 |
|------|--------|------|
| 完整 StockMixer | 1.465 | 三种 mixing 全部使用 |
| 去掉 Stock-Mixing | 1.102 | 退化为独立时序模型 |
| 去掉 Time-Mixing | 1.215 | 只看最后一天的截面 |
| 去掉 Indicator-Mixing | 1.312 | 因子不交互 |
| 去掉排名损失 | 1.298 | 只用 MSE |

**结论**：Stock-Mixing 贡献最大——截面股票间的关系建模是性能提升的核心来源。

---

## 五、为什么 StockMixer 适合作为第一个复现目标

| 优势 | 说明 |
|------|------|
| **结构极简** | 三个线性层 + 一个输出层，50 行代码能实现 |
| **参数极少** | ~200K 参数，CPU 几分钟训练完 |
| **不需要外部数据** | 不需要行业分类、知识图谱等额外数据 |
| **效果已验证** | AAAI 2024 发表，三个美股指数上一致超越 baseline |
| **DataLoader 简单** | 输入就是一个 $N \times T \times d$ 张量 |
| **可渐进增强** | 实现后可以逐步加入 DoubleAdapt、市场状态门控 |

---

## 六、工程实现要点

### 6.1 核心模型代码

StockMixer 的前向传播可以用不到 40 行 PyTorch 实现：

```python
import torch
import torch.nn as nn

class StockMixer(nn.Module):
    """StockMixer: 三维 MLP-Mixer 用于截面选股"""
    
    def __init__(self, n_stocks: int, n_timesteps: int, n_features: int):
        super().__init__()
        # 三个 mixing 层
        self.stock_mixer = nn.Sequential(
            nn.Linear(n_stocks, n_stocks),
            nn.GELU(),
            nn.Linear(n_stocks, n_stocks),
        )
        self.time_mixer = nn.Sequential(
            nn.Linear(n_timesteps, n_timesteps),
            nn.GELU(),
            nn.Linear(n_timesteps, n_timesteps),
        )
        self.indicator_mixer = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.GELU(),
            nn.Linear(n_features, n_features),
        )
        # 输出层
        self.head = nn.Linear(n_timesteps * n_features, 1)
    
    def forward(self, x):
        """
        x: (batch, N_stocks, T_timesteps, d_features)
        对于截面模型，batch=1（一个截面）
        """
        # Stock-Mixing: 沿股票维度混合
        # x: (1, N, T, d) → 转置为 (1, T, d, N) → MLP → 转回
        x = x.squeeze(0)  # (N, T, d)
        x_s = x.permute(1, 2, 0)  # (T, d, N)
        x_s = self.stock_mixer(x_s)  # (T, d, N)
        x = x_s.permute(2, 0, 1)  # (N, T, d)
        
        # Time-Mixing: 沿时间维度混合
        x_t = x.permute(0, 2, 1)  # (N, d, T)
        x_t = self.time_mixer(x_t)  # (N, d, T)
        x = x_t.permute(0, 2, 1)  # (N, T, d)
        
        # Indicator-Mixing: 沿特征维度混合
        x = self.indicator_mixer(x)  # (N, T, d)
        
        # Flatten + 预测
        x = x.reshape(x.size(0), -1)  # (N, T*d)
        scores = self.head(x).squeeze(-1)  # (N,)
        return scores
```

**代码关键点**：
- `permute` 操作是核心——通过转置让 MLP 作用在不同维度上
- 每个 mixer 用两层 MLP + GELU 激活（论文中也测试过单层，效果略差）
- `head` 把每只股票的 $T \times d$ 维特征压缩为 1 个分数

### 6.2 DataLoader 设计

```python
class CrossSectionalDataset(torch.utils.data.Dataset):
    """截面数据集：每个样本是一个完整的日期截面"""
    
    def __init__(self, features, returns, dates, lookback=10):
        """
        features: dict[date] → (N_stocks, d_features) 当日特征
        returns: dict[date] → (N_stocks,) 次日收益率
        lookback: 使用过去几天的数据
        """
        self.features = features
        self.returns = returns
        self.dates = dates
        self.lookback = lookback
    
    def __getitem__(self, idx):
        target_date = self.dates[idx]
        # 取过去 lookback 天的数据堆叠
        past_dates = self.dates[idx - self.lookback : idx]
        
        # (T, N, d) → (N, T, d)
        x = torch.stack([self.features[d] for d in past_dates])  # (T, N, d)
        x = x.permute(1, 0, 2)  # (N, T, d)
        
        y = self.returns[target_date]  # (N,)
        return x, y
```

### 6.3 注意事项

1. **股票数量 N 不固定**：不同天可能有不同数量的股票在交易（停牌、新上市等）。需要用 padding + mask 处理
2. **Stock-Mixing 的 $N \times N$ 矩阵**：如果 $N=3000$（全 A 股），这个矩阵有 900 万参数——可能需要降维或分组
3. **归一化**：论文中在每个 mixer 前后加了 LayerNorm，实际实现中这很重要

---

## 七、对比与启示

### 7.1 StockMixer vs MASTER

| 维度 | StockMixer | MASTER |
|------|-----------|--------|
| 截面关系建模 | 线性层（固定 $N \times N$） | Attention（动态权重） |
| 市场信息 | 无 | 有（引入市场级特征） |
| 参数量 | ~200K | ~2M |
| 训练时间 | 分钟级 | 小时级 |
| 效果 | NASDAQ Sharpe 1.47 | CSI300 RankIC 0.076 |
| 适用场景 | 固定成分股（成分不常变） | 成分股变动大的市场 |

### 7.2 何时选 StockMixer

- ✅ 作为**第一个实现**：结构简单、快速验证数据管道是否正确
- ✅ 成分股**相对固定**的指数（如 S&P500、CSI300）
- ✅ **计算资源有限**：不需要 GPU
- ❌ 全市场选股（3000+ 只）：$N \times N$ 矩阵太大
- ❌ 需要动态适配的场景：不含在线学习机制

---

## 八、总结

| 要点 | 内容 |
|------|------|
| 核心创新 | 把 MLP-Mixer 的"分维度混合"搬到股票预测 |
| 三种混合 | Stock/Time/Indicator Mixing |
| 参数量 | ~200K（极轻量） |
| 效果 | NASDAQ/NYSE/S&P500 Sharpe 1.4-1.6，超越 GNN/Transformer |
| 适合 | 第一轮复现目标，验证截面建模的价值 |
| 不足 | 无动态适配、无市场状态感知 |

---

## 延伸阅读

- [MASTER 精读](/论文综述/083_MASTER_市场感知截面选股Transformer) — 更强但更复杂的截面方案
- [DoubleAdapt 精读](/论文综述/084_DoubleAdapt_分布漂移双重适配) — 给 StockMixer 加上在线适配能力
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — 评估方法论
