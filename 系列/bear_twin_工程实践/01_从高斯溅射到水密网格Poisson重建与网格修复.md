---
title: "从高斯溅射到水密网格：Poisson 重建与网格修复"
order: 971
tags: [3DGS, Poisson重建, 网格修复, Open3D, trimesh]
category: 系列
series:
  id: bear_twin_工程实践
  index: 1
---

# 从高斯溅射到水密网格：Poisson 重建与网格修复

> **输入**：3DGS 训练产出的 PLY 文件（32,704 个高斯核，每个含位置、不透明度、球谐系数）。
> **输出**：一个水密（watertight）三角网格（~9,500 面片，体积 ~0.79 L），后续所有粒子采样和弹簧构建都以它为起点。

**知识链接**：
- [3D Gaussian Splatting](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建) — 输入 PLY 的来源
- [Poisson 表面重建](/前置知识/020a_前置知识_Poisson表面重建从点云到水密网格) — 本章核心算法的原理
- [PhysTwin 论文精读](/论文综述/169_PhysTwin_从稀疏视频构建可交互软体数字孪生) — 为什么需要水密网格

---

## 贯穿全文的例子

> 你手上有一个毛绒熊的 3DGS 点云（`bear_object.ply`，2.2 MB），里面存了 32,704 个高斯核。每个高斯核有 xyz 坐标、不透明度 opacity、45 个球谐系数（用于编码视角依赖的颜色）。现在要把这团"带颜色的椭球云"变成一个封闭的三角网格——后续的体积采样需要判断"这个点在不在物体内部"，而这个判断只有水密网格才能回答。

---

## 一、为什么不能直接用高斯点云

3DGS 的输出不是传统点云——它是一组带协方差的椭球基元，专为渲染优化，不保证表面连续性：

| 属性 | 高斯核 | 三角网格 |
|------|--------|----------|
| 拓扑 | 无连接关系 | 顶点-边-面的流形结构 |
| 体积概念 | 无（只有渲染权重） | 封闭表面围出的体积 |
| 法向一致性 | 无 | 法向朝外，表面连续 |
| "内部/外部"判断 | 不可能 | 射线检测或 winding number |

弹簧质点系统需要在物体**内部**撒点（interior sampling），这要求网格是水密的——没有破洞、没有自交。所以第一步必须把高斯云转成水密网格。

---

## 二、完整管线：九个阶段

```mermaid
flowchart LR
    A["PLY 加载\n32704 高斯"] --> B["不透明度过滤\nsigmoid > 0.12"]
    B --> C["统计离群移除\nk=30, σ=2.0"]
    C --> D["法向估计\nr=10mm, k=30"]
    D --> E["Poisson 重建\ndepth=9"]
    E --> F["密度裁剪\n底 5%"]
    F --> G["最大连通分量"]
    G --> H["空洞填充\ntrimesh + O3D"]
    H --> I["水密网格 ✓\n~9500 面片"]
```

> **看什么**：每个阶段都在做减法（过滤噪声）或加法（补全信息），目标是让 Poisson 重建拿到干净、法向一致的输入，再修补它遗留的缺陷。

---

## 三、PLY 加载与球谐→RGB 转换

3DGS PLY 文件的颜色不是直接的 RGB，而是球谐系数。零阶球谐系数 `f_dc_0/1/2` 编码了视角无关的基色。转换公式：

$$
\text{RGB} = \text{sigmoid}\left(\frac{f_{\text{dc}}}{\sqrt{4\pi}} + 0.5\right)
$$

**这个公式在做什么**：零阶球谐系数除以归一化常数 $\sqrt{4\pi}$（球谐函数 $Y_0^0$ 的模）得到辐射率，加 0.5 的偏移再过 sigmoid 压到 [0,1]。这让"最暗的高斯"映射到接近 0、"最亮的"映射到接近 1。

::: details 📐 公式详解（点击展开）

| 子表达式 | 含义 |
|---------|------|
| $f_{\text{dc}}$ | PLY 中 `f_dc_0/1/2` 三个字段 |
| $\sqrt{4\pi} \approx 3.545$ | 零阶球谐的归一化常数 |
| $+0.5$ | 偏移，让零辐射对应 sigmoid(0.5) ≈ 0.62 而不是 0.5 |
| sigmoid | 压缩到 [0,1]，保证合法 RGB |

:::

实际代码中 `step1_reconstruct.py` 提取 xyz 和 opacity 后，用 SH 系数计算每个点的颜色，用于后续可视化。

---

## 四、不透明度过滤

高斯核的不透明度以 logit 形式存储（训练时的参数化方式），需要过 sigmoid 才是真实不透明度：

$$
\alpha = \sigma(\text{opacity\_raw}) = \frac{1}{1 + e^{-\text{opacity\_raw}}}
$$

**阈值选择**：`step1_reconstruct.py` 用 `opacity_raw > -2.0` 过滤，对应 $\alpha > \sigma(-2.0) \approx 0.12$。这是一个很宽松的阈值——只丢弃几乎完全透明的点。

| 阈值 | sigmoid 值 | 保留比例 | 效果 |
|------|-----------|---------|------|
| -4.0 | 0.018 | ~99% | 几乎不过滤 |
| **-2.0** | **0.12** | **~88%** | 去掉最透明的 12% |
| 0.0 | 0.50 | ~60% | 激进，可能丢细节 |

Bear 数据：32,704 → 28,834 点（过滤掉 3,870 个低不透明度高斯）。

---

## 五、统计离群点移除（SOR）

过滤不透明度后，仍有空间上的离群点（训练时飘到远处的高斯）。Open3D 的 `remove_statistical_outlier` 计算每个点到其 $k$ 个最近邻的平均距离，如果平均距离超过全局均值加 $n$ 倍标准差，就视为离群点：

$$
\bar{d}_i > \mu_{\bar{d}} + n \cdot \sigma_{\bar{d}} \implies \text{移除}
$$

**参数**：$k=30$，$n=2.0$。Bear 数据：28,834 → 25,089 点。

**为什么是 k=30 而不是 k=10**：高斯核在物体表面分布不均匀（纹理丰富的区域密集、平坦区域稀疏），小 $k$ 只看最近的几个邻居，容易把稀疏区域的正常点也判为离群。$k=30$ 提供更稳定的局部密度估计。

---

## 六、法向估计与一致性朝向

接下来的核心步骤是 **Poisson 表面重建**（详见 [前置知识：Poisson 表面重建](/前置知识/020a_前置知识_Poisson表面重建从点云到水密网格)）——从带法向点云生成水密三角网格。Poisson 重建通过求解一个全局 Poisson 方程，把离散的点法向"补全"成连续的内外指示函数，再提取等值面得到网格。

Poisson 重建的输入要求每个点带法向，且法向方向一致（都朝外）。Open3D 分两步完成：

1. **局部法向估计**：对每个点取半径 10mm 内的邻域（最多 30 点），拟合局部切平面，切平面的法向就是该点的法向。数学上是对邻域点的协方差矩阵做特征值分解，取最小特征值对应的特征向量。

2. **全局一致性朝向**：局部法向有 180° 歧义（朝内和朝外都是合法法向）。`orient_normals_consistent_tangent_plane(k=30)` 用传播算法，从一个种子点出发，沿最小生成树传播法向朝向，确保相邻点的法向方向一致。

**为什么法向一致性很重要**：Poisson 重建把法向当作"表面的内外梯度场"来求解。如果法向随机朝内朝外，重建出来的表面会像翻过来的袜子——到处是自交和破洞。

---

## 七、Poisson 表面重建

> 本节介绍 Bear 管线中 Poisson 重建的具体参数和用法。Poisson 重建的完整原理（Poisson 方程推导、八叉树结构、为什么能保证水密性）见 [前置知识：Poisson 表面重建](/前置知识/020a_前置知识_Poisson表面重建从点云到水密网格)。

这是管线的核心步骤。Poisson 重建把"带法向的点云"转换为"三角网格"。

**直觉**：把每个点的法向想象成表面处的梯度——法向指向"从内到外"的方向。如果存在一个标量场 $\chi$，它在物体内部为 1、外部为 0，那么 $\chi$ 的梯度场应该和这些法向一致。Poisson 重建就是求解这个标量场：

$$
\nabla^2 \chi = \nabla \cdot \mathbf{V}
$$

**这个公式在做什么**：一个 Poisson 方程——左边是标量场 $\chi$ 的拉普拉斯算子（二阶空间导数），右边是法向场 $\mathbf{V}$ 的散度。求解这个方程找到一个"从内到外平滑过渡"的标量场，取其等值面就是重建的表面。

::: details 📐 公式详解（点击展开）

| 子表达式 | 含义 |
|---------|------|
| $\chi$ | 指示函数：物体内部 ≈ 1，外部 ≈ 0 |
| $\nabla^2$ | 拉普拉斯算子 $\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$ |
| $\mathbf{V}$ | 输入的法向场（定义在点云位置） |
| $\nabla \cdot \mathbf{V}$ | 法向场的散度 |

**为什么用 Poisson 方程**：直接求"梯度等于法向"（$\nabla \chi = \mathbf{V}$）通常无解（法向场不一定是某个标量场的梯度），但最小二乘意义下的最优解恰好满足 Poisson 方程。这是一个全局优化问题，天然产生光滑表面。

:::

**参数**：`depth=9` 控制八叉树的精度（$2^9 = 512$ 格子边长），`scale=1.1` 让八叉树略大于点云包围盒（留边距）。

重建输出的原始网格包含大量多余三角面，需要后续裁剪。

---

## 八、密度裁剪与连通分量提取

Poisson 重建会在点稀疏的区域"脑补"表面——在包围盒边角产生虚假的膨胀。Open3D 同时输出每个顶点的**密度值**（该位置有多少输入点支持），用密度裁剪去掉低密度区域：

```python
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
```

**底 5% 分位数**：去掉密度最低的 5% 顶点。这是一个保守的阈值——只移除最明显的虚假膨胀。

裁剪后可能产生多个不连通的碎片，用 `mesh.cluster_connected_triangles()` 提取最大连通分量，丢弃碎片。

Bear 数据：Poisson 输出 → 密度裁剪 → 最大连通分量 → ~9,500 三角面片。

---

## 九、空洞填充：两种策略

密度裁剪和连通分量提取后，网格可能仍有破洞。`step1_fix.py` 实现了两种修补策略：

### 策略一：trimesh 自动填充

```python
mesh_tm = trimesh.Trimesh(vertices, faces)
trimesh.repair.fill_holes(mesh_tm)
```

trimesh 的 `fill_holes` 检测边界环（boundary loop），用三角化扇面补上。优点是简单快速，缺点是大洞补出的面片可能不平滑。

### 策略二：Open3D 拆分 + 逐块修复

当 trimesh 无法完全封闭时（`is_watertight` 仍为 False），`step1_fix.py` 转入 Open3D 备用路径：

1. `cluster_connected_triangles` 拆分所有连通分量
2. 取最大分量
3. `subdivide_midpoint` 细分一次（增加面片密度）
4. 尝试再次 `fill_holes`

### 诊断指标

`step1_fix.py` 会检查：

| 指标 | 含义 | 合格标准 |
|------|------|---------|
| boundary edges | 属于只有一个面片的边 | 0（水密） |
| non-manifold edges | 属于 3 个以上面片的边 | 0 |
| is_watertight | trimesh 判定 | True |
| volume | 网格围出的体积 | > 0 且合理 |

Bear 最终网格：0 boundary edges、0 non-manifold edges、volume ≈ 0.79 L（合理——毛绒熊约 8.8×10.9×18.4 cm）。

---

## 十、两个脚本的分工

| | step1_reconstruct.py | step1_fix.py |
|-|---------------------|-------------|
| 定位 | 主管线：从 PLY 到网格 | 修补：诊断 + 修复 |
| 输入 | bear_object.ply | SOR 过滤后点云 + Poisson 网格 |
| 输出 | 07_bear_mesh.ply | 08_bear_mesh_clean.ply |
| 空洞处理 | trimesh 一轮 | trimesh + Open3D 双保险 |
| 适用场景 | 首次运行 | Poisson 结果有缺陷时 |

---

## 输出检查清单

```bash
# 验证水密性
python3 -c "
import trimesh
m = trimesh.load('artifacts/07_bear_mesh.ply')
print(f'watertight: {m.is_watertight}')
print(f'faces: {len(m.faces)}')
print(f'volume: {m.volume:.4f} L')
print(f'bounds: {m.bounds.ptp(axis=0)}')
"
# 预期：watertight=True, faces≈9500, volume≈0.79, bounds≈[0.088, 0.109, 0.184]
```
