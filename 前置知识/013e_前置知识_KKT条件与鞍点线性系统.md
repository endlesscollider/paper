---
title: KKT 条件与鞍点线性系统
order: 121
tags: [数值优化, KKT, 鞍点系统, 约束优化, 流固耦合]
category: 前置知识
star: 3
---

# KKT 条件与鞍点线性系统

> **前置知识**：拉格朗日乘子法基础，线性代数（矩阵方程的概念）。
> 本文是 [Cosserat 杆断裂与 MPM 流固耦合综述](/论文综述/S34_Cosserat_MPM_DEM_断裂流固耦合综述) 的前置知识，重点支撑 DIQ-MPM 的 monolithic 流固耦合求解。

---

## 问题：带等式约束的最小化

很多工程问题需要在满足约束的前提下最小化某个目标：

$$
\min_{\mathbf{x}} \; f(\mathbf{x}) \quad \text{subject to} \quad \mathbf{G}\mathbf{x} = \mathbf{b}
$$

**这个公式在做什么**：在满足一组线性等式约束的前提下，找使目标函数 $f$ 最小的解——约束把可行解限制在 $\mathbf{G}\mathbf{x} = \mathbf{b}$ 定义的一个仿射子空间上，最优解就是这个子空间里的最低点。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mathbf{x}$ | **待求变量** | 优化对象；在物理仿真中通常是速度场或位移场 |
| $f(\mathbf{x})$ | **目标函数** | 要最小化的量，如动能、弹性势能或数据拟合误差 |
| $\mathbf{G}$ | **约束矩阵** | 线性算子，把 $\mathbf{x}$ 映射到约束空间，每一行对应一条约束 |
| $\mathbf{b}$ | **约束右端项** | 约束的目标值；取零表示"约束量必须相等"这类无源条件 |

**为什么约束取线性形式**：线性约束保证可行集是凸的（一个仿射子空间），此时 KKT 条件不仅是必要条件也是充分条件，整个问题可以转化为线性方程组直接求解，不需要非线性优化的迭代过程。这也是流固耦合界面无穿透约束能写成 $\mathbf{G}(\mathbf{v}_f - \mathbf{v}_s) = 0$ 的原因。
:::

最直接的处理方式是引入**拉格朗日乘子** $\boldsymbol{\lambda}$，把约束"软化"进目标，构造无约束问题的极值条件。对光滑凸问题，极值点必须满足 KKT（Karush-Kuhn-Tucker）条件：梯度为零、约束满足。

---

## 从 KKT 条件到鞍点方程

对上面的问题，拉格朗日函数为 $L(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \boldsymbol{\lambda}^\top(\mathbf{G}\mathbf{x} - \mathbf{b})$。

对 $\mathbf{x}$ 和 $\boldsymbol{\lambda}$ 分别求偏导并令其为零，得到 KKT 一阶必要条件：

$$
\nabla_\mathbf{x} f(\mathbf{x}) + \mathbf{G}^\top \boldsymbol{\lambda} = 0, \qquad \mathbf{G}\mathbf{x} = \mathbf{b}
$$

**这个公式在做什么**：给出带线性约束问题的两个一阶必要条件——第一式说明最优点处目标梯度必须被约束力 $\mathbf{G}^\top\boldsymbol{\lambda}$ 精确抵消（否则还能沿约束面继续下降），第二式说明解必须落在约束面上。两者联立就是 KKT 条件。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\nabla_\mathbf{x} f(\mathbf{x})$ | **目标梯度** | 目标函数对 $\mathbf{x}$ 的导数，指向 $f$ 上升最快的方向 |
| $\mathbf{G}^\top \boldsymbol{\lambda}$ | **约束力** | 约束矩阵转置与乘子的乘积，方向垂直于约束面，负责把梯度"顶回去" |
| $\boldsymbol{\lambda}$ | **拉格朗日乘子** | 每条约束一个分量，物理上就是该约束对应的约束力大小 |
| $\mathbf{G}\mathbf{x} = \mathbf{b}$ | **原始可行性** | 强制解满足约束，这是 KKT 的第二组条件 |

**第一式的几何含义**：在最优解处，目标函数的梯度（想往外走的趋势）必须完全落在约束面的法空间里被约束力抵消。如果梯度有沿约束面的切向分量，说明沿约束面还能继续下降，当前点就不是最优的。这就是"梯度与约束法空间共线"的经典结论，$\boldsymbol{\lambda}$ 正是那个共线系数。
:::

当 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{c}^\top \mathbf{x}$（二次目标）时，上式化简为一个**鞍点线性系统**：

$$
\begin{bmatrix} \mathbf{A} & \mathbf{G}^\top \\ \mathbf{G} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{c} \\ \mathbf{b} \end{bmatrix}
$$

**这个公式在做什么**：把"最小化二次目标同时满足线性约束"这个问题，转换为一个联立线性方程组——同时解出最优解 $\mathbf{x}$ 和对应的约束力 $\boldsymbol{\lambda}$，一步到位，无需迭代。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mathbf{A}$ | **目标函数的 Hessian** | 通常是质量矩阵或刚度矩阵，正定或半正定 |
| $\mathbf{G}^\top \boldsymbol{\lambda}$ | **约束力** | 约束矩阵的转置作用于乘子，给出满足约束所需的"外力" |
| $\mathbf{G}\mathbf{x} = \mathbf{b}$ | **约束方程** | 强制解必须满足的线性约束，如界面法向速度连续 |
| $\boldsymbol{\lambda}$ | **拉格朗日乘子** | 约束对应的"影子价格"——在物理仿真中通常就是约束力/接触力 |
| 左下角 $\mathbf{0}$ | **零块** | 对应 $\partial L / \partial \boldsymbol{\lambda}$ 方程，强制约束满足，没有 $\boldsymbol{\lambda}$ 的自身动力学 |

**为什么叫"鞍点"系统**：左端矩阵不是正定的（因为有零块），其极值不是最小值而是鞍点（对 $\mathbf{x}$ 是最小，对 $\boldsymbol{\lambda}$ 是最大）。这让标准的 CG 等正定求解器失效，需要专门的鞍点求解器或预条件方法。
:::

---

## 为什么鞍点系统比迭代更好

在流固耦合中，另一种常见做法是"分离迭代"：先解流体、再解固体、再更新界面力，重复直到收敛。这类迭代法有两个缺陷：

1. **收敛条件苛刻**：当流固密度比很大（如水和钢）时，迭代可能不收敛或收敛极慢。
2. **数值断裂**：迭代不收敛时界面约束未被精确满足，出现物理上不合理的流固穿透或真空间隙。

鞍点系统把两个未知量 $\mathbf{x}$（速度场）和 $\boldsymbol{\lambda}$（界面力）**联立求解**，约束在数学上精确满足——断裂从根源上被消除。

---

## DIQ-MPM 中的鞍点系统

DIQ-MPM 的 monolithic 求解把流体速度 $\mathbf{v}_f$、固体速度 $\mathbf{v}_s$、界面接触力 $\boldsymbol{\lambda}$ 写进一个鞍点系统：

$$
\begin{bmatrix} \mathbf{A}_f & 0 & \mathbf{G}^\top \\ 0 & \mathbf{A}_s & -\mathbf{G}^\top \\ \mathbf{G} & -\mathbf{G} & 0 \end{bmatrix} \begin{bmatrix} \mathbf{v}_f \\ \mathbf{v}_s \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{f}_f \\ \mathbf{f}_s \\ 0 \end{bmatrix}
$$

**这个公式在做什么**：把流体速度 $\mathbf{v}_f$、固体速度 $\mathbf{v}_s$ 和界面接触力 $\boldsymbol{\lambda}$ 放进同一个鞍点系统联立求解，一次线性求解同时得到三者的值，界面约束在数学上被精确满足。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\mathbf{A}_f$，$\mathbf{A}_s$ | **流体/固体系统矩阵** | 各自的隐式动量方程矩阵；对角块，因为流体和固体内部互不耦合 |
| $\mathbf{G}^\top$，$-\mathbf{G}^\top$ | **约束力耦合块** | 把界面力 $\boldsymbol{\lambda}$ 按相反符号分配到流体和固体方程（作用力与反作用力） |
| $\mathbf{G}$，$-\mathbf{G}$ | **约束方程块** | 提取界面两侧的法向速度，相减得到速度差 |
| 右端第三行的 $0$ | **无穿透约束** | 强制 $\mathbf{G}\mathbf{v}_f - \mathbf{G}\mathbf{v}_s = 0$，即法向速度连续 |
| $\boldsymbol{\lambda}$ | **界面接触力** | 系统求出的未知量，正值表示界面处的排斥力 |

**为什么两个耦合块符号相反**：牛顿第三定律的直接体现——界面力对流体和固体的作用方向相反。取相反符号后，两者在界面上传递的动量总量守恒，不会凭空产生或消失动量。
:::

右端第三行为零表示法向无穿透约束：$\mathbf{G}\mathbf{v}_f = \mathbf{G}\mathbf{v}_s$，即流固界面两侧法向速度相等。$\boldsymbol{\lambda}$ 求解出来就是界面接触力，无需单独建模。

---

## 不等式约束与互补条件

当约束是不等式（如接触：只能推、不能拉）时，KKT 条件扩展为互补条件：

$$
\boldsymbol{\lambda} \geq 0, \quad \mathbf{G}\mathbf{x} - \mathbf{b} \geq 0, \quad \boldsymbol{\lambda}^\top(\mathbf{G}\mathbf{x} - \mathbf{b}) = 0
$$

**这个公式在做什么**：给出不等式约束下的最优性条件——约束力非负、约束不被违反、且两者不能同时为正，把"接触时有力、分离时无力"这一物理直觉写成三个数学条件。

::: details 📐 公式详解（点击展开）

| 子表达式 | 它是谁 | 它在干嘛 |
|---------|--------|---------|
| $\boldsymbol{\lambda} \geq 0$ | **对偶可行性** | 约束力只能推不能拉——接触面不能把物体吸住 |
| $\mathbf{G}\mathbf{x} - \mathbf{b} \geq 0$ | **原始可行性** | 约束不被违反，如物体之间不能相互穿透 |
| $\boldsymbol{\lambda}^\top(\mathbf{G}\mathbf{x} - \mathbf{b}) = 0$ | **互补松弛** | 两者至少有一个为零：要么接触（距离 0，力非零），要么分离（力 0，距离非零） |

**互补条件的物理读法**：接触力和间隙距离是一对"此消彼长"的量。物体分开时（间隙为正），接触力必须为零；接触力非零时，间隙必须恰为零。这个条件是非光滑的（两个分支的切换点不可微），所以这类问题不能直接用牛顿法求解，需要 LCP 或半光滑牛顿等专门方法。
:::
