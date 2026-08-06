---
title: HIST 与 TRA：概念/行业关系建模选股
order: 186
tags: [量化交易, 截面模型, 概念图, 时间路由]
category: 精读
star: 3
---

# HIST 与 TRA：概念关系与时间路由 深度精读

> **HIST**: Modeling Shared and Specific Information of Individual Stocks (ACM MM 2022)  
> **TRA**: Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor (KDD 2021)  
> **代码**: https://github.com/microsoft/qlib (官方实现)

**知识链接**：
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — 评估指标
- [LightGBM 在量化选股中的应用](/前置知识/001y_前置知识_LightGBM在量化选股中的应用) — 基线对比
- [MASTER 精读](/论文综述/083_MASTER_市场感知截面选股Transformer) — 更新的截面关系方法
- [分布漂移与在线适配](/前置知识/001z_前置知识_分布漂移与在线适配) — TRA 的设计动机

---

## 贯穿全文的例子

> CSI300 成分股中：
> - 贵州茅台、五粮液、泸州老窖属于"白酒"概念
> - 宁德时代、比亚迪属于"新能源车"概念
> - 贵州茅台还和五粮液同属"消费龙头"概念
> 
> 当"白酒"板块集体涨停时，HIST 能利用这个概念关系预测尚未涨停的白酒股也会跟涨。
> 而 TRA 则根据"当前是板块轮动行情"选择"动量追踪"专家，而非"均值回归"专家。

---

## 一、HIST：隐式概念图建模

### 1.1 核心思想

每只股票的收益可以分解为两部分：
- **共享部分**：来自它所属的"概念"（行业、主题）的整体运动
- **个体部分**：股票自身独有的驱动力

$$
y_i = \underbrace{y_{\text{shared},i}}_{\text{概念级信号}} + \underbrace{y_{\text{specific},i}}_{\text{个股级信号}}
$$

> HIST 先用"概念"聚合相似股票的信息，再分离出每只股票的独有信息。

### 1.2 概念矩阵

HIST 使用一个预定义的**概念-股票关联矩阵** $C \in \mathbb{R}^{N \times K}$：

| 股票 | 白酒 | 新能源 | 消费龙头 | 科技 |
|------|------|--------|----------|------|
| 贵州茅台 | 1 | 0 | 1 | 0 |
| 五粮液 | 1 | 0 | 1 | 0 |
| 宁德时代 | 0 | 1 | 0 | 0 |
| 比亚迪 | 0 | 1 | 0 | 0 |
| 腾讯 | 0 | 0 | 0 | 1 |

$C_{ik} = 1$ 表示股票 $i$ 属于概念 $k$。

### 1.3 共享信息提取

**Step 1**：对每个概念 $k$，聚合所属股票的表示：

$$
\mathbf{g}_k = \frac{1}{|S_k|}\sum_{i \in S_k} \mathbf{h}_i
$$

其中 $S_k$ 是属于概念 $k$ 的股票集合，$\mathbf{h}_i$ 是股票 $i$ 的 GRU 嵌入。

**Step 2**：概念表示通过 attention 分配回个股：

$$
\mathbf{h}_{\text{shared},i} = \sum_{k: C_{ik}=1} \alpha_{ik} \cdot \mathbf{g}_k, \quad \alpha_{ik} = \text{softmax}_k(\mathbf{h}_i^\top \mathbf{g}_k)
$$

> 股票 $i$ 从它所属的多个概念中，根据当前状态选择性地接收信息。如果茅台今天的走势更像"消费龙头"而非"白酒"，attention 会给"消费龙头"更高权重。

**Step 3**：个体信息 = 总表示 - 共享部分：

$$
\mathbf{h}_{\text{specific},i} = \mathbf{h}_i - \mathbf{h}_{\text{shared},i}
$$

最终预测：$\hat{y}_i = \text{MLP}([\mathbf{h}_{\text{shared},i}; \mathbf{h}_{\text{specific},i}])$

### 1.4 HIST 的 Qlib 表现

在 Alpha360 因子集上（20 随机种子）：

| 指标 | HIST | GRU | LightGBM | Transformer |
|------|------|-----|----------|-------------|
| 年化超额 | 9.87% | 6.44% | 8.25% | 2.73% |
| IR | 1.37 | 0.87 | 1.06 | 0.36 |
| IC | 0.048 | 0.035 | 0.042 | 0.025 |

HIST 超越 LightGBM 1.62 个百分点年化超额——概念信息确实有用。

---

## 二、TRA：时间路由适配器

### 2.1 核心思想

不同时期，市场运行的"模式"不同：
- 动量期：涨的继续涨（追涨策略有效）
- 反转期：涨多的回落（逆向策略有效）
- 波动期：大幅震荡（短线低吸高抛有效）

TRA 维护多个"专家"模型，用一个路由网络根据当前市场状态选择合适的专家。

### 2.2 模型结构

$$
\hat{y}_i = \sum_{m=1}^M w_m(c_t) \cdot f_m(\mathbf{x}_i)
$$

**逐项拆解**：
- $M$：专家数量（论文中 $M=3$）
- $f_m(\mathbf{x}_i)$：第 $m$ 个专家对股票 $i$ 的预测——每个专家是一个独立的 MLP head
- $c_t$：第 $t$ 天的时间 context 特征（如最近 20 天的市场收益率、波动率等）
- $w_m(c_t)$：路由网络输出的权重——$\sum_m w_m = 1$，表示当前时刻各专家的重要性

> 路由网络看了最近的市场状态后说"现在是动量行情，给动量专家权重 0.7，给反转专家权重 0.2，给波动专家权重 0.1"。

### 2.3 路由网络

$$
\mathbf{w}(c_t) = \text{softmax}\left(\text{MLP}_{\text{router}}(c_t)\right) \in \mathbb{R}^M
$$

路由网络的输入 $c_t$ 通常包括：
- 市场指数近 5/10/20 天收益率
- 市场波动率
- 涨跌比
- 成交额变化率

### 2.4 训练

TRA 的训练有一个技巧：**不强制指定哪个专家对应哪种市场状态**——让模型自动学习分工。

损失 = 加权预测损失 + 专家多样性正则：

$$
L = \sum_t \ell\left(\sum_m w_m(c_t) f_m(x_t), y_t\right) + \beta \cdot \text{Diversity}(\{f_m\})
$$

多样性正则确保不同专家不会学到相同的模式（否则退化为单模型）。

### 2.5 TRA 的 Qlib 表现

| 指标 | TRA | GRU | HIST |
|------|-----|-----|------|
| 年化超额 | 9.20% | 6.44% | 9.87% |
| IR | 1.28 | 0.87 | 1.37 |
| IC | 0.046 | 0.035 | 0.048 |

TRA 略逊于 HIST，但仍显著优于 GRU baseline。两者可以结合——HIST 建模截面关系，TRA 做时间路由。

---

## 三、HIST vs TRA 对比

| 维度 | HIST | TRA |
|------|------|-----|
| 建模目标 | 股票之间的关系（空间） | 不同时间模式的切换（时间） |
| 额外输入 | 概念/行业归属矩阵 | 市场 context 特征 |
| 参数开销 | 中（attention 层） | 低（多个 MLP head） |
| 依赖先验知识 | 是（需要概念定义） | 否（自动学习） |
| 适用场景 | 有明确行业/概念结构 | 市场状态频繁切换 |
| Qlib 表现 | 年化 9.87%, IR 1.37 | 年化 9.20%, IR 1.28 |
| 与 MASTER 的关系 | MASTER 用 attention 替代了固定概念图 | MASTER 用市场引导替代了路由 |

---

## 四、工程实现要点

### 4.1 HIST 的概念矩阵获取

```python
# A 股概念数据获取（通过 Tushare 或 AKShare）
import akshare as ak

def get_concept_matrix(stock_codes):
    """获取股票-概念关联矩阵"""
    # 获取所有概念板块成分
    concepts = {}
    concept_list = ak.stock_board_concept_name_em()
    
    for _, row in concept_list.iterrows():
        concept_name = row['板块名称']
        members = ak.stock_board_concept_cons_em(symbol=concept_name)
        concepts[concept_name] = set(members['代码'].tolist())
    
    # 构建关联矩阵
    concept_names = list(concepts.keys())
    matrix = np.zeros((len(stock_codes), len(concept_names)))
    
    for j, concept in enumerate(concept_names):
        for i, code in enumerate(stock_codes):
            if code in concepts[concept]:
                matrix[i, j] = 1.0
    
    return matrix, concept_names
```

### 4.2 TRA 的路由网络

```python
class TemporalRoutingAdaptor(nn.Module):
    """TRA: 时间路由适配器"""
    
    def __init__(self, input_dim, hidden_dim, n_experts=3, context_dim=10):
        super().__init__()
        # 多个专家 head
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(n_experts)
        ])
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, stock_features, market_context):
        """
        stock_features: (N, input_dim) 每只股票的嵌入
        market_context: (context_dim,) 当天的市场状态特征
        """
        # 路由权重
        weights = self.router(market_context)  # (n_experts,)
        
        # 各专家预测
        expert_preds = torch.stack([
            expert(stock_features).squeeze(-1)  # (N,)
            for expert in self.experts
        ], dim=-1)  # (N, n_experts)
        
        # 加权汇总
        output = (expert_preds * weights.unsqueeze(0)).sum(dim=-1)  # (N,)
        return output
```

---

## 五、适合作为基准的原因

HIST 和 TRA 在 Qlib 中有**成熟的官方实现**、**20 个种子的可靠数据**、**标准化的评估流程**：

1. 可直接在 Qlib 框架下运行，不需要自己写 DataLoader
2. 结果可与其他 Qlib 模型直接对比
3. 作为"截面关系模型"的代表性基准，适合用来验证新方法的增量价值

---

## 六、总结

| 方法 | 核心贡献 | Qlib 年化/IR | 适合场景 |
|------|----------|-------------|----------|
| HIST | 概念驱动的共享/个体信息分离 | 9.87% / 1.37 | 有行业/概念先验 |
| TRA | 时间路由选择专家 | 9.20% / 1.28 | 市场状态频繁切换 |
| 两者结合 | 空间关系 + 时间路由 | 预期 >10% | 通用场景 |

---

## 延伸阅读

- [MASTER 精读](/论文综述/083_MASTER_市场感知截面选股Transformer) — HIST 的进化版（动态 attention 替代固定概念图）
- [分布漂移与在线适配](/前置知识/001z_前置知识_分布漂移与在线适配) — TRA 的路由本质是在适配漂移
- [LightGBM 在量化选股中的应用](/前置知识/001y_前置知识_LightGBM在量化选股中的应用) — 必须超过的基线
