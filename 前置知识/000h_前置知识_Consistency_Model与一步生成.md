---
title: Consistency Model 与一步生成
order: 8
tags: [扩散模型, 深度学习]
category: 前置知识
---

# 前置知识：Consistency Model 与一步生成

> **为什么要读这篇**：Consistency Model 是把扩散模型的多步推理压缩到 1–2 步的极端加速方案。在 Online DPRL 综述中被多次提及为"让 BPTT 方法完全可行"的关键技术。理解它才能判断什么时候该用多步扩散、什么时候该用一步一致性模型。
> **前置要求**：读完 000b（DDPM）、000g（Flow Matching）

**标签**: `#前置知识` `#Consistency Model` `#一步生成` `#蒸馏` `#自一致性` `#加速推理` `#机器人策略`

**知识链接**：
- [扩散模型 DDPM](./000b_前置知识_扩散模型DDPM) — Consistency Model 要加速的对象
- [Flow Matching](./000g_前置知识_Flow_Matching与连续归一化流) — 另一种加速方案的对比
- [为什么扩散策略难以 RL 微调](./000f_前置知识_为什么扩散策略难以RL微调) — 一步生成如何彻底解决梯度链问题
- [Online DPRL 综述](/论文综述/003_Online_DPRL_综述_扩散策略与在线RL) — 在算法家族中的定位

---

## 一、核心想法：自一致性约束

### 1.1 直觉

考虑一条从噪声到数据的 PF-ODE（Probability Flow ODE）轨迹：

```mermaid
graph LR
    X0["x₀<br/>(噪声)"] --> X02["x₀.₂"] --> X05["x₀.₅"] --> X08["x₀.₈"] --> X1["x₁<br/>(数据)"]
    style X1 fill:#c8e6c9
```

普通扩散模型需要沿轨迹**逐步走**。Consistency Model 的想法是：

> **不管你在轨迹上的哪个位置，都能一步跳到终点 $\mathbf{x}_1$。**

### 1.2 形式化定义

定义一致性函数 $f_\theta : (\mathbf{x}_t, t) \to \mathbf{x}_1$，满足：

$$
\boxed{f_\theta(\mathbf{x}_t,\, t) = f_\theta(\mathbf{x}_{t'},\, t') \quad \forall\; (\mathbf{x}_t, t),\, (\mathbf{x}_{t'}, t') \;\text{在同一条 PF-ODE 轨迹上}}
$$

加上边界条件：

$$
f_\theta(\mathbf{x}_1,\, 1) = \mathbf{x}_1 \quad \text{（在数据端是恒等映射）}
$$

这两个条件保证了：从轨迹上任何点出发，输出都是同一个终点 $\mathbf{x}_1$。

```mermaid
flowchart LR
    A["x₀ (噪声)"] -->|"f_θ(x₀, 0)"| D["x₁"]
    B["x₀.₅ (中间)"] -->|"f_θ(x₀.₅, 0.5)"| D
    C["x₀.₈ (接近数据)"] -->|"f_θ(x₀.₈, 0.8)"| D
    style D fill:#c8e6c9
```

### 1.3 推理——极度简化

$$
\text{采样}\; \mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \;\;\xrightarrow{\;f_\theta(\mathbf{x}_0,\, 0)\;}\;\; \mathbf{x}_1 = \text{最终动作}
$$

**一次前向传播！** 对比 DDPM 的 20–100 次和 Flow 的 4–10 次。

---

## 二、两种训练方式

### 2.1 Consistency Distillation (CD)

**前提**：已有一个训练好的扩散模型作为 teacher。

**核心 loss**：

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}_{t,\,\mathbf{x}}\left\| f_\theta(\mathbf{x}_{t+\Delta t},\; t{+}\Delta t) - f_{\theta^-}(\hat{\mathbf{x}}_t,\; t) \right\|^2
$$

其中：
- $\hat{\mathbf{x}}_t$ 是用 teacher 的 ODE 从 $\mathbf{x}_{t+\Delta t}$ 走一步到 $t$ 得到的
- $\theta^-$ 是 $\theta$ 的 EMA（指数移动平均）副本

**直觉**：相邻两点在同一条轨迹上 → 它们的 $f_\theta$ 输出应该一致。

```mermaid
flowchart TD
    A["x_{t+Δt}"] -->|"f_θ"| C["终点预测 A"]
    A -->|"Teacher ODE 一步"| B["x̂_t"]
    B -->|"f_{θ⁻}"| D["终点预测 B"]
    C -.-|"应该相等"| D
    style C fill:#e3f2fd
    style D fill:#e3f2fd
```

### 2.2 Consistency Training (CT)

不需要 teacher，直接从数据训练：

$$
\mathcal{L}_{\text{CT}} = \mathbb{E}_{t,\,\mathbf{x}_0,\,\mathbf{x}_1}\left\| f_\theta(\mathbf{x}_{t+\Delta t},\; t{+}\Delta t) - f_{\theta^-}(\mathbf{x}_t,\; t) \right\|^2
$$

其中 $\mathbf{x}_t, \mathbf{x}_{t+\Delta t}$ 通过线性插值直接构造（不需要跑 ODE）。

配合 $\Delta t$ 的退火调度：训练早期 $\Delta t$ 大（粗略一致性），后期 $\Delta t$ 小（精细一致性）。

### 2.3 对比

|  | CD（蒸馏） | CT（从头训练） |
|---|---|---|
| 需要 teacher | ✓ | ✗ |
| 生成质量 | 较高 | 略低 |
| 典型场景 | 已有 Diffusion Policy | 全新训练 |

---

## 三、网络架构：满足边界条件

一致性函数需要 $f_\theta(\mathbf{x}_1, 1) = \mathbf{x}_1$。通过 skip connection 参数化：

$$
f_\theta(\mathbf{x}, t) = c_{\text{skip}}(t) \cdot \mathbf{x} + c_{\text{out}}(t) \cdot F_\theta(\mathbf{x}, t)
$$

其中系数满足：

$$
c_{\text{skip}}(1) = 1,\; c_{\text{out}}(1) = 0 \quad \Rightarrow \quad f_\theta(\mathbf{x}_1, 1) = \mathbf{x}_1 \;\checkmark
$$

$$
c_{\text{skip}}(0) = 0,\; c_{\text{out}}(0) = 1 \quad \Rightarrow \quad f_\theta(\mathbf{x}_0, 0) = F_\theta(\mathbf{x}_0, 0) \;\text{（完全由网络决定）}
$$

$F_\theta$ 的架构和 DDPM 的去噪网络**完全相同**（UNet / MLP / Transformer），只是外面多了一层 skip connection。

---

## 四、多步一致性采样

### 4.1 为什么 1 步不够完美

理论上 1 步就能生成，但实际网络不可能完美满足一致性约束。2–3 步采样能大幅提升质量：

### 4.2 多步算法（$N=2$ 为例）

$$
\begin{aligned}
&\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&\hat{\mathbf{x}}_1 = f_\theta(\mathbf{x}_0,\; 0) &\text{（第一步：粗略预测终点）}\\
&\mathbf{x}_{0.5} = 0.5\,\boldsymbol{\epsilon} + 0.5\,\hat{\mathbf{x}}_1,\quad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}) &\text{（在中间时间重新注入噪声）}\\
&\hat{\mathbf{x}}_1 = f_\theta(\mathbf{x}_{0.5},\; 0.5) &\text{（第二步：精细化预测）}
\end{aligned}
$$

```mermaid
flowchart LR
    A["x₀ ~ N(0,I)"] -->|"f_θ(x₀, 0)"| B["x̂₁ (粗略)"]
    B -->|"加噪到 t=0.5"| C["x₀.₅"]
    C -->|"f_θ(x₀.₅, 0.5)"| D["x̂₁ (精细)"]
    style D fill:#c8e6c9
```

直觉：先画草图 → 加一点随机扰动 → 重新精细化。

---

## 五、对 RL 微调的影响

### 5.1 $N=1$ 时的策略梯度

当策略只有 1 步时，生成过程为：

$$
\mathbf{a} = f_\theta(\mathbf{x}_0,\; 0,\; \mathbf{s}), \quad \mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

这在形式上和**确定性策略 + 噪声输入**一样。$\log \pi(\mathbf{a}|\mathbf{s})$ 可以通过变量替换公式计算：

$$
\log \pi(\mathbf{a}|\mathbf{s}) = \log p(\mathbf{x}_0) - \log \left|\det\frac{\partial f_\theta}{\partial \mathbf{x}_0}\right|
$$

> Jacobian 行列式计算 $O(d^2)$，或用 Hutchinson estimator 近似到 $O(d)$。

更重要的是——**BPTT 完全无压力**：

$$
\nabla_\theta Q(\mathbf{s}, \mathbf{a})\Big|_{\mathbf{a}=f_\theta(\mathbf{x}_0, 0, \mathbf{s})} \quad \text{梯度只穿过 1 层网络}
$$

不存在梯度消失/爆炸。和训练普通神经网络策略一样简单。

### 5.2 和各方法的 RL 友好度对比

| 生成模型 | 推理步数 | BPTT 可行性 | 需要 DPPO 展开 | 直接 PG |
|---|---|---|---|---|
| DDPM 20步 | 20 | ✗ (梯度崩) | ✓ (必须) | ✗ |
| Flow 5步 | 5 | ✓ | 可选 | △ (方差) |
| Consistency 2步 | 2 | ✓✓ | 不需要 | △ |
| Consistency 1步 | 1 | ✓✓✓ | 不需要 | ✓ |

### 5.3 Trade-off：表达力 vs 推理速度 vs RL 友好度

```mermaid
graph TD
    A["DDPM 20步<br/>表达力 ★★★★★<br/>速度 ★★<br/>RL难度 高"]
    B["Flow 5步<br/>表达力 ★★★★<br/>速度 ★★★★<br/>RL难度 中"]
    C["Consistency 2步<br/>表达力 ★★★<br/>速度 ★★★★★<br/>RL难度 低"]
    D["Consistency 1步<br/>表达力 ★★<br/>速度 ★★★★★★<br/>RL难度 极低"]
    A --> B --> C --> D
```

> RL 微调本身会提升策略质量。即使初始表达力略低，微调后可能追平多步扩散。"Consistency Policy + 简单 PG" 是否能匹配 "Diffusion Policy + DPPO"？2026 年仍是开放问题。

---

## 六、和 Flow Matching 的对比

|  | Flow Matching | Consistency Model |
|---|---|---|
| 推理步数 | 4–10 | 1–2 |
| 路径定义 | 连续 ODE 轨迹 | 轨迹上任意点→终点 |
| 训练方式 | 向量场回归 | 一致性约束/蒸馏 |
| 表达力 | 高 | 中 |
| RL 微调方式 | BPTT / Proximity | 直接 PG / BPTT |
| 适用场景 | 速度和质量平衡 | 极致速度优先 |

---

## 七、总结

### 核心公式回顾

$$
\underbrace{f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t')}_{\text{一致性约束}} \qquad \underbrace{f_\theta(\mathbf{x}_1, 1) = \mathbf{x}_1}_{\text{边界条件}}
$$

### 什么时候选 Consistency Model

| 选它 | 不选它 |
|---|---|
| 控制频率 > 100Hz | 数据极其复杂多模态 |
| 边缘设备部署 | 不缺推理时间 |
| 需要简单 PG 做 RL | DPPO 已效果很好 |
| 已有 Diffusion Policy 可蒸馏 | — |

---

## 延伸阅读

- Song et al. (2023) "Consistency Models" ← 原始论文
- Song & Dhariwal (2024) "Improved Consistency Training"
- Prasad et al. (2024) "Consistency Policy" ← 用在机器人上
- [Flow Matching](./000g_前置知识_Flow_Matching与连续归一化流) ← 另一种加速方案
- [DPPO](/论文综述/001_DPPO_扩散策略策略优化) ← 多步扩散的 RL 微调方案
- [Online DPRL 综述](/论文综述/003_Online_DPRL_综述_扩散策略与在线RL) ← 步数对算法选择的影响
