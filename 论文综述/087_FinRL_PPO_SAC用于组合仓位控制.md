---
title: FinRL：PPO/SAC 用于组合仓位控制
order: 187
tags: [量化交易, 强化学习, PPO, SAC, 仓位管理]
category: 精读
star: 4
---

# FinRL：深度强化学习量化交易框架 深度精读

> **论文标题**: FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance  
> **竞赛**: FinRL Contests 2023-2025  
> **代码**: https://github.com/AI4Finance-Foundation/FinRL  
> **竞赛论文**: https://arxiv.org/abs/2504.02281

**知识链接**：
- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — FinRL 使用的核心 RL 算法
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — FinRL 的另一核心算法
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — 与监督学习选股的对比
- [Walk-Forward 滚动回测](/前置知识/001x_前置知识_Walk_Forward滚动回测) — FinRL 的评估框架

---

## 贯穿全文的例子

> 管理一个 DJ30（道琼斯 30）成分股组合。
> - **状态** $s$：30 只股票的当前持仓量 + 价格 + 技术指标 + 账户余额
> - **动作** $a \in [-1, 1]^{30}$：每只股票的买卖比例（+1=全仓买入，-1=全部卖出）
> - **奖励** $r$：单步组合收益率
> - **约束**：每笔交易扣除 0.1% 手续费；不能做空（$a_i \geq 0$ for long-only）

---

## 一、RL 在量化交易中的定位

### 1.1 监督学习 vs 强化学习

| 维度 | 监督学习选股 | RL 交易 |
|------|-------------|---------|
| **输出** | 股票排名分数 | 具体仓位/交易动作 |
| **目标** | 最大化 IC（预测准确性） | 最大化累计收益/Sharpe |
| **考虑交易成本** | 通常不考虑（后期加） | 天然内嵌在奖励函数中 |
| **考虑持仓约束** | 不考虑 | 天然内嵌在状态/动作空间中 |
| **序列决策** | 每天独立决策 | 今天的交易影响明天的状态 |
| **难度** | 中（有标签） | 高（稀疏奖励、不稳定） |

### 1.2 RL 适合做什么

根据 FinRL 竞赛和学术实验的经验，RL 适合做**低维度控制**，不适合做**高维度选股**：

**适合 RL 的任务**：
- 给定 Top-30 股票（由监督模型选出），RL 决定每只的仓位比例
- 决定何时加仓/减仓（择时）
- 决定组合的整体风险暴露（杠杆/对冲比例）
- 决定换手频率（交易成本控制）

**不适合 RL 的任务**：
- 从 3000 只股票中直接选股（动作空间太大，$3000$ 维连续动作）
- 端到端预测收益率（有标签的监督学习更直接）

### 1.3 推荐的 Pipeline

```mermaid
flowchart LR
    A["监督模型<br>（StockMixer/MASTER）"] -->|"输出 Top-K 排名"| B["RL 控制器<br>（PPO/SAC）"]
    B -->|"输出仓位权重"| C["交易执行"]
    C -->|"环境反馈"| B
```

监督模型负责"选什么"，RL 负责"买多少、什么时候买"。

---

## 二、FinRL 的环境设计

### 2.1 状态空间

$$
s_t = [\underbrace{b_t}_{\text{余额}}, \; \underbrace{\mathbf{p}_t \in \mathbb{R}^N}_{\text{股价}}, \; \underbrace{\mathbf{h}_t \in \mathbb{R}^N}_{\text{持仓}}, \; \underbrace{\mathbf{f}_t \in \mathbb{R}^{N \times d}}_{\text{技术指标}}]
$$

**逐项拆解**：
- $b_t$：当前现金余额（标量）
- $\mathbf{p}_t$：每只股票的当前价格
- $\mathbf{h}_t$：每只股票的当前持仓股数
- $\mathbf{f}_t$：技术指标（MACD、RSI、CCI、布林带等）

对于 DJ30：$\dim(s) = 1 + 30 + 30 + 30 \times 5 = 211$

### 2.2 动作空间

$$
a_t \in [-K, K]^N
$$

每只股票的动作表示"买卖多少股"。$a_i > 0$ 买入，$a_i < 0$ 卖出。$K$ 是每步最大交易量。

**实际处理**：
- 离散化：$a_i \in \{-K, ..., -1, 0, 1, ..., K\}$
- 或连续化：$a_i \in [-1, 1]$，再乘以最大可交易量

### 2.3 奖励函数

$$
r_t = \frac{V_{t+1} - V_t}{V_t}
$$

其中 $V_t = b_t + \sum_{i=1}^N h_{i,t} \cdot p_{i,t}$ 是总资产价值。

**带交易成本的版本**：
$$
r_t = \frac{V_{t+1} - V_t - \text{cost}_t}{V_t}, \quad \text{cost}_t = c \cdot \sum_{i=1}^N |a_{i,t}| \cdot p_{i,t}
$$

其中 $c = 0.001$（单边 0.1%）。

### 2.4 数值例子

初始余额 $b_0 = 100000$，持有 10 股苹果（$p=150$），5 股微软（$p=300$）。

$$
V_0 = 100000 + 10 \times 150 + 5 \times 300 = 103000
$$

RL 决定：买入 5 股苹果，卖出 2 股微软。

交易成本 = $0.001 \times (5 \times 150 + 2 \times 300) = 0.001 \times 1350 = 1.35$

次日苹果涨到 155，微软涨到 305：

$$
V_1 = (100000 - 5 \times 150 + 2 \times 300 - 1.35) + 15 \times 155 + 3 \times 305 = 99848.65 + 2325 + 915 = 103088.65
$$

$r_0 = (103088.65 - 103000) / 103000 = 0.086\%$

---

## 三、FinRL 竞赛的滚动设置

### 3.1 短周期滚动

FinRL 2023-2025 竞赛采用极短的滚动周期：

| 设置 | 值 |
|------|-----|
| 训练期 | 30 个交易日 |
| 验证期 | 5 个交易日 |
| 测试期 | 5 个交易日 |
| 滚动步长 | 5 天 |
| 交易成本 | 单边 0.1% |
| 标的 | DJ30 |

### 3.2 FinRL 竞赛结果（DJ30）

PPO 在竞赛的标准设置下：

| 指标 | PPO | DJIA（被动持有） |
|------|-----|----------------|
| 累计收益 | 63.37% | 18.95% |
| 年化收益 | 18.41% | 6.32% |
| Sharpe Ratio | 1.55 | 0.47 |
| 最大回撤 | -9.96% | -21.53% |
| 胜率（月度） | 68% | 54% |

### 3.3 重要警告

**竞赛结果 vs 真实后市表现**：

在 FinRL 竞赛的"赛后评估"（真实未来数据，非历史回测）中，RL 队伍**普遍没有跑赢 DJIA 的收益**。这说明：

1. 历史 Walk-Forward 表现好 ≠ 真实未来表现好
2. 短窗口训练的 RL 对市场状态切换极其脆弱
3. PPO 的 Sharpe 1.55 可能有过拟合成分——在相对平稳的历史期间表现好，但遇到从未见过的市场状态就失效

**结论**：RL 适合做仓位控制的"辅助决策"，不应作为唯一信号源。

---

## 四、PPO vs SAC 在交易中的选择

### 4.1 对比

| 维度 | PPO | SAC |
|------|-----|-----|
| 策略类型 | On-Policy（随机策略） | Off-Policy（最大熵随机策略） |
| 数据效率 | 低（每步数据只用一次） | 高（Replay Buffer 反复使用） |
| 训练稳定性 | 高（clip 保证小步更新） | 中（需要仔细调 alpha） |
| 探索能力 | 中（靠策略噪声） | 高（最大熵鼓励探索） |
| 适合交易场景 | 仓位调整频率低、决策保守 | 需要探索不同策略组合 |
| FinRL 默认 | ✅ 推荐首选 | 备选 |

### 4.2 实操建议

1. **先用 PPO**：更稳定、超参少、不容易崩溃
2. **动作空间设计**：不要让 RL 直接输出 30 维连续动作——维度太高。改为输出 3-5 个"宏观控制变量"：
   - 整体仓位比例（0~1）
   - Top-N 的 N 值（10/20/30/50）
   - 权重方式（等权/市值加权/IC 加权）
   - 止损阈值（-3%/-5%/-8%）
3. **奖励工程**：不要只用收益率。加入 Sharpe 惩罚和回撤惩罚：

$$
r_t = \underbrace{\frac{V_{t+1} - V_t}{V_t}}_{\text{收益}} - \underbrace{\lambda_1 \cdot \text{DD}_t}_{\text{回撤惩罚}} - \underbrace{\lambda_2 \cdot \text{turnover}_t}_{\text{换手惩罚}}
$$

---

## 五、现有仓库的 RL 设计评估

根据用户提到的代码结构：

### 5.1 `src/train/rl_stock/deep_signal_environment.py:89`

这个环境让 PPO 选择仓位、TopN、权重方式和阈值——**这是正确方向**。RL 不直接选股，而是做低维度控制。

### 5.2 `config/rl_stock.yaml:2` 的风险

如果配置中让 RL 对全市场 4000 只股票直接输出动作权重，问题是：
- 4000 维连续动作空间，PPO 几乎无法有效探索
- 每次决策的组合空间是指数级的
- 奖励信号极其稀疏（一天的组合收益率受太多股票影响）

**建议**：保持低维度控制器设计，不要尝试端到端全市场 RL。

---

## 六、代码示例：低维度 RL 控制器

```python
import gymnasium as gym
import numpy as np

class PortfolioControlEnv(gym.Env):
    """
    低维度 RL 环境：不直接选股，只控制仓位和风险参数
    """
    
    def __init__(self, signal_model, market_data):
        super().__init__()
        self.signal_model = signal_model  # 预训练的截面选股模型
        self.market_data = market_data
        
        # 状态：市场特征 + 当前持仓状态
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,)
        )
        
        # 动作：4 个低维控制变量
        # [仓位比例, TopN选择, 止损阈值, 换仓频率]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
    
    def _decode_action(self, action):
        """把连续动作解码为离散的交易参数"""
        position_pct = action[0]            # 0~100% 仓位
        top_n = int(action[1] * 40) + 10    # Top 10~50
        stop_loss = -0.03 - action[2] * 0.07  # -3% ~ -10%
        hold_days = int(action[3] * 4) + 1   # 持有 1~5 天
        return position_pct, top_n, stop_loss, hold_days
    
    def step(self, action):
        position_pct, top_n, stop_loss, hold_days = self._decode_action(action)
        
        # 1. 用监督模型获取股票排名
        scores = self.signal_model.predict(self.current_features)
        top_stocks = np.argsort(scores)[-top_n:]
        
        # 2. 等权分配仓位
        weight_per_stock = position_pct / top_n
        
        # 3. 模拟交易 hold_days 天
        portfolio_return = self._simulate_trade(
            top_stocks, weight_per_stock, hold_days, stop_loss
        )
        
        # 4. 计算奖励（含交易成本）
        reward = portfolio_return - self.transaction_cost
        
        return self._get_obs(), reward, done, truncated, info
```

---

## 七、总结

| 要点 | 内容 |
|------|------|
| RL 的正确定位 | 做低维仓位/风险控制，不做高维选股 |
| 推荐算法 | PPO 首选（稳定），SAC 备选（高数据效率） |
| 动作空间 | 3-5 维宏观控制变量，不要 N 维股票权重 |
| 奖励设计 | 收益 - 回撤惩罚 - 换手惩罚 |
| FinRL 结果 | DJ30 历史 Sharpe 1.55，但赛后真实期不稳定 |
| 关键警告 | RL 跨市场状态泛化不稳定，不应作为唯一信号源 |
| 正确 Pipeline | 监督模型选股 → RL 控制仓位/风险 |

---

## 延伸阅读

- [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO) — PPO 算法的完整推导
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 的最大熵框架
- [MacroHFT 精读](/论文综述/088_MacroHFT_分层市场状态路由RL交易) — 分层 RL + 市场状态路由
- [截面选股模型与评价指标](/前置知识/001w_前置知识_截面选股模型与评价指标) — 监督选股模型基础
