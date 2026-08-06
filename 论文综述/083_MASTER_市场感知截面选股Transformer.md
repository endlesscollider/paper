---
title: MASTER：市场感知截面选股 Transformer
order: 183
tags: [量化交易, 截面模型, Transformer, 注意力机制]
category: 精读
star: 5
---

# MASTER：Market-Aware Stock Transformer 深度精读

> **论文标题**: MASTER: Market-Guided Stock Transformer for Stock Price Forecasting  
> **作者**: Tong Li, Zhaoyang Liu, Yanyan Shen, et al.  
> **机构**: Shanghai Jiao Tong University  
> **发表**: AAAI 2024  
> **DOI**: https://ojs.aaai.org/index.php/AAAI/article/view/27767

**知识链接**：
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — IC/ICIR/RankIC 的定义
- [Walk-Forward 滚动回测](/前置知识/001x_前置知识_Walk_Forward滚动回测) — MASTER 的评估框架
- [LightGBM 在量化选股中的应用](/前置知识/001y_前置知识_LightGBM在量化选股中的应用) — 必须超越的基线
- [分布漂移与在线适配](/前置知识/001z_前置知识_分布漂移与在线适配) — MASTER 面临的核心挑战
- [StockMixer 精读](/论文综述/082_StockMixer_轻量截面混合选股) — 更轻量的截面方案（对比）

---

## 贯穿全文的例子

> CSI300（沪深 300）成分股，日频预测。
> - 每只股票有 $d=6$ 维日频特征：(开盘/最高/最低/收盘/成交量/换手率) 的过去 $T=20$ 天
> - 输入：$N=300$ 只股票 × $T=20$ 天 × $d=6$ 维 = $\mathbf{X} \in \mathbb{R}^{300 \times 20 \times 6}$
> - 另外还有一个"市场级"特征：沪深 300 指数本身的收益率、波动率、情绪指标等
> - 目标：预测每只股票未来 1 天的超额收益率排序

---

## 一、MASTER 解决什么问题

### 1.1 现有截面模型的三个缺陷

| 缺陷 | 说明 | MASTER 的解法 |
|------|------|--------------|
| **1. 股票关系是静态的** | GNN 方法用固定图（行业/供应链）建模关系，但关系会随时间变化 | 用 Attention 动态计算股票间关系权重 |
| **2. 忽略市场整体状态** | 现有方法只看个股特征，不知道"当前是牛市还是熊市" | 引入 Market-Guided Attention |
| **3. 时序建模与截面建模分离** | 先用 LSTM 提取时序特征，再用 GNN 建模截面——两步解耦可能丢信息 | Intra-stock Transformer 统一处理 |

### 1.2 核心创新

MASTER 的关键贡献是 **Market-Guided Attention**：

> 不是让股票和股票直接做 attention，而是让"市场状态"来**引导**哪些股票之间应该关注。在牛市中，高 beta 股票之间的关系更重要；在熊市中，防御型股票之间的关系更重要。

---

## 二、模型架构

### 2.1 总体流程

```mermaid
flowchart TD
    A["个股特征 X ∈ ℝ^(N×T×d)"] --> B["Intra-Stock Transformer<br>（每只股票独立的时序建模）"]
    B --> C["股票嵌入 H ∈ ℝ^(N×d_model)"]
    
    M["市场级特征 m ∈ ℝ^(d_m)"] --> D["Market Encoder"]
    D --> E["市场嵌入 h_m ∈ ℝ^(d_model)"]
    
    C --> F["Market-Guided<br>Inter-Stock Attention"]
    E --> F
    F --> G["增强后的股票嵌入 H' ∈ ℝ^(N×d_model)"]
    G --> H["Prediction Head"]
    H --> I["预测分数 ŷ ∈ ℝ^N"]
```

### 2.2 Intra-Stock Transformer（个股时序建模）

对每只股票 $i$，取其过去 $T$ 天的特征 $\mathbf{x}_i \in \mathbb{R}^{T \times d}$，用一个 Transformer Encoder 提取时序模式：

$$
\mathbf{h}_i = \text{TransformerEncoder}(\mathbf{x}_i) \in \mathbb{R}^{d_{\text{model}}}
$$

这里的 Transformer 就是标准的多头自注意力 + FFN，让每天的特征关注其他天的特征。最后取 [CLS] token 或 average pooling 得到该股票的表示 $\mathbf{h}_i$。

> 为什么用 Transformer 而不是 LSTM？ Transformer 能直接关注任意两天的关系（如"第 1 天的放量和第 10 天的突破"），不受"遗忘门"的限制。

所有 $N$ 只股票共享同一个 Transformer 参数——这保证了模型可以泛化到新股票。

### 2.3 Market-Guided Inter-Stock Attention（核心创新）

这是 MASTER 的核心模块。标准 attention 是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

MASTER 的 Market-Guided Attention 在此基础上加了市场信息的调制：

$$
\text{MG-Attention}(H, h_m) = \text{softmax}\left(\frac{(H \cdot W_Q)(H \cdot W_K)^\top}{\sqrt{d_k}} + \mathbf{B}(h_m)\right) \cdot (H \cdot W_V)
$$

> 在标准 QK 相似度上加了一个**市场条件偏置** $\mathbf{B}(h_m)$——它根据当前市场状态告诉 attention"哪些股票对之间的关系此刻更重要"。

**逐项拆解**：
- $H \in \mathbb{R}^{N \times d_{\text{model}}}$：所有 $N$ 只股票的嵌入矩阵
- $W_Q, W_K, W_V$：标准的 Query/Key/Value 投影矩阵
- $\frac{QK^\top}{\sqrt{d_k}}$：标准 attention 分数——基于股票特征本身的相似度
- $\mathbf{B}(h_m) \in \mathbb{R}^{N \times N}$：市场引导偏置矩阵，由市场嵌入 $h_m$ 生成
- $h_m$：编码了"当前是什么市场状态"的向量

**$\mathbf{B}(h_m)$ 的生成方式**：

$$
\mathbf{B}(h_m) = \text{MLP}(h_m) \in \mathbb{R}^{N \times N}
$$

或者更实际的实现（因为 $N \times N$ 太大）：

$$
\mathbf{B}(h_m) = \mathbf{q}_m \cdot \mathbf{k}_m^\top, \quad \mathbf{q}_m = W_{mq} \cdot h_m \in \mathbb{R}^{N}, \quad \mathbf{k}_m = W_{mk} \cdot h_m \in \mathbb{R}^{N}
$$

通过低秩分解避免直接生成 $N \times N$ 矩阵。

### 2.4 数值例子

假设 $N=3$ 只股票（简化），$d_k = 2$：

标准 attention 分数 $\frac{QK^\top}{\sqrt{d_k}}$：
```
      Stock_A  Stock_B  Stock_C
A  [   0.5      0.8      0.2  ]
B  [   0.8      0.5      0.3  ]  ← A 和 B 相似度高
C  [   0.2      0.3      0.5  ]
```

市场偏置 $\mathbf{B}(h_m)$（假设当前是"成长股行情"，A、B 是成长股）：
```
      Stock_A  Stock_B  Stock_C
A  [   0.0      0.5      -0.3 ]
B  [   0.5      0.0      -0.3 ]  ← 市场状态加强了 A-B 之间的注意力
C  [  -0.3     -0.3       0.0 ]
```

最终 attention 分数 = 标准分数 + 市场偏置：
```
      Stock_A  Stock_B  Stock_C
A  [   0.5      1.3     -0.1  ]
B  [   1.3      0.5      0.0  ]  ← A-B 关系被进一步增强
C  [  -0.1      0.0      0.5  ]  ← C 被"孤立"
```

**直觉**：在成长股行情中，成长股之间互相关注更紧密（它们的命运与共），而价值股被边缘化。这正是"市场引导注意力"的效果。

### 2.5 市场特征编码

市场级特征 $\mathbf{m}$ 通常包括：
- 市场指数的收益率（近 1/5/20 天）
- 市场波动率（VIX 或 realized vol）
- 涨跌比（上涨股数/下跌股数）
- 成交量异动
- 市场情绪指标（融资余额、换手率均值等）

这些被一个 MLP 编码为 $h_m \in \mathbb{R}^{d_{\text{model}}}$。

---

## 三、训练细节

### 3.1 损失函数

MASTER 使用 **IC 最大化 + MSE 回归** 的混合损失：

$$
L = L_{\text{MSE}} - \lambda \cdot \text{IC}(\hat{\mathbf{y}}, \mathbf{y})
$$

直接把 IC（Pearson 相关系数）作为正则项加入损失——因为最终目标就是最大化 IC。

**为什么 IC 可以作为损失？** Pearson 相关系数是可微的（对预测值的梯度可以解析计算），所以可以直接用 SGD 优化。

### 3.2 训练设置

| 设置 | 值 |
|------|-----|
| 数据集 | CSI300 / CSI800 |
| 训练期 | 2007-2019 |
| 验证期 | 2019Q1-2020Q2 |
| 测试期 | 2020Q3-2022Q4 |
| 滚动方式 | 按季度重训 |
| 随机种子 | 5 个 |
| 学习率 | 1e-4 |
| Batch | 1 个截面/batch |
| 优化器 | Adam |

---

## 四、实验结果

### 4.1 CSI300 主实验

| 模型 | IC↑ | ICIR↑ | RankIC↑ | RankICIR↑ | 超额年化↑ | IR↑ |
|------|-----|------|---------|-----------|-----------|------|
| LightGBM | 0.042 | 0.43 | 0.051 | 0.52 | 12.4% | 1.23 |
| LSTM | 0.038 | 0.38 | 0.046 | 0.47 | 9.8% | 0.97 |
| GRU | 0.041 | 0.41 | 0.050 | 0.50 | 11.2% | 1.12 |
| Transformer | 0.035 | 0.33 | 0.042 | 0.40 | 7.5% | 0.72 |
| HIST | 0.052 | 0.53 | 0.062 | 0.64 | 18.3% | 1.87 |
| **MASTER** | **0.064** | **0.67** | **0.076** | **0.79** | **27.1%** | **2.40** |

**关键发现**：
1. MASTER 在所有指标上都显著领先
2. IC 0.064、RankIC 0.076——在 CSI300 上这是极强的数字
3. IR 2.40 意味着超额收益极其稳定（几乎每个月都能跑赢基准）
4. 普通 Transformer 又是最差的——关键不是 attention 架构本身，而是有没有合理的归纳偏置

### 4.2 消融实验

| 变体 | IC | 相对完整版的变化 |
|------|-----|-----------------|
| 完整 MASTER | 0.064 | — |
| 去掉 Market-Guided | 0.055 | -14% |
| 去掉 Inter-Stock Attention | 0.048 | -25% |
| 去掉 Intra-Stock Transformer | 0.051 | -20% |
| 用固定图替代 attention | 0.053 | -17% |

**结论**：
- Inter-Stock Attention（截面关系建模）贡献最大
- Market-Guided 偏置提供了显著增益（IC +0.009）
- 动态 attention 比固定图（行业/供应链）更有效

---

## 五、与 HIST、TRA 的对比

### 5.1 HIST（Hidden Industry Shared Transformer）

HIST 的思路是：用隐式的"概念"（行业、主题）来组织股票关系。

- 预定义概念矩阵（如"新能源"概念包含 30 只相关股票）
- 每只股票对每个概念的关联度通过 attention 计算
- 概念层面做信息聚合，再投射回个股

Qlib Alpha360 上的表现：年化 9.87%，IR 1.37。

### 5.2 TRA（Temporal Routing Adaptor）

TRA 的思路是：不同时期需要不同的"专家模型"。

- 维护多个专家头（如 3 个 MLP head）
- 一个路由网络根据当前时间特征选择哪个专家
- 本质是条件计算（Mixture of Experts 的时间版本）

Qlib Alpha360 上的表现：年化 9.20%，IR 1.28。

### 5.3 对比总结

| 方法 | 核心思想 | Qlib 年化 | IR | 适用场景 |
|------|----------|-----------|-----|----------|
| HIST | 隐式概念组织股票 | 9.87% | 1.37 | 有行业/概念数据 |
| TRA | 时间路由切换专家 | 9.20% | 1.28 | 市场状态频繁切换 |
| MASTER | 市场引导动态注意力 | 27.1%* | 2.40* | 通用截面场景 |

*注：MASTER 使用自己的测试设置（2020Q3-2022Q4），和 Qlib 标准设置不完全可比。

---

## 六、MASTER 的局限与后续方向

| 局限 | 说明 | 可能的改进 |
|------|------|----------|
| 计算复杂度 | $O(N^2)$ attention 在全市场（3000+股）上很贵 | 线性 attention / 分组 attention |
| 无在线适配 | 按季度重训，季度内不更新 | 加入 DoubleAdapt |
| 市场特征人工选择 | 需要手动定义市场级因子 | 可以自动从截面特征聚合 |
| 数据需求大 | Transformer 结构需要更多数据 | 对小市场可能不如 StockMixer |

---

## 七、总结

| 要点 | 内容 |
|------|------|
| 核心创新 | Market-Guided Attention——用市场状态调制股票间注意力 |
| 三模块 | Intra-Stock Transformer + Market Encoder + MG Inter-Stock Attention |
| CSI300 效果 | IC 0.064, RankIC 0.076, 超额年化 27%, IR 2.40（5 种子） |
| vs LightGBM | IC 高 52%、超额年化高 119% |
| 适合 | 中等规模指数（300-800 只）的日频选股 |
| 代价 | 参数量 ~2M，需要 GPU，需要市场级特征 |
| 建议顺序 | StockMixer → MASTER → MASTER + DoubleAdapt |

---

## 延伸阅读

- [StockMixer 精读](/论文综述/082_StockMixer_轻量截面混合选股) — 更轻量的截面方案
- [DoubleAdapt 精读](/论文综述/084_DoubleAdapt_分布漂移双重适配) — 给 MASTER 加上漂移适配
- [InvariantStock 精读](/论文综述/085_InvariantStock_不变特征学习选股) — 另一条路：学习不变特征
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — IC/ICIR 的基础定义
