---
title: "粒子三层采样：Object / Surface / Interior 的生成与去重"
order: 972
tags: [FPS, 体积采样, 去重, 粒子系统]
category: 系列
series:
  id: bear_twin_engineering
  index: 2
---

# 粒子三层采样：Object / Surface / Interior 的生成与去重

> **输入**：第一章的水密网格（9,500 面片，0.79 L 体积）。
> **输出**：三组粒子 + 优先级去重 → 4,074 个唯一粒子，作为弹簧质点系统的节点。

**知识链接**：
- [质点弹簧模型](/前置知识/006e_前置知识_质点弹簧模型与绳索仿真基础) — 为什么需要三层粒子
- [Chamfer 距离](/前置知识/011b_前置知识_Chamfer距离_点云形状相似性度量) — object points 参与的误差计算

---

## 贯穿全文的例子

> 网格是一个封闭的壳，但弹簧质点系统需要"充实"的粒子——表面点让外形对齐、内部点提供体积和质量、object points 是后续要跟踪的标记点。你要在网格上撒 1,024 个表面点、体积内撒 10,000 个内部点、再用最远点采样（FPS）在表面选 1,024 个均匀分布的 object 点，最后用 5mm 体素网格去重，确保没有两个粒子挤在一起。

---

## 一、为什么需要三类粒子

PhysTwin 等弹簧质点仿真方法的粒子系统不是一层均匀粒子，而是三层有角色分工的：

| 类型 | 生成方式 | 数量目标 | 作用 |
|------|---------|---------|------|
| **Object points** | FPS 均匀采样网格顶点 | 1,024 | 跟踪标记点，用于 Chamfer 距离计算；CoTracker 输入 |
| **Surface points** | 随机表面采样 | 1,024 | 补充表面覆盖，确保外形完整 |
| **Interior points** | 体积内随机采样 | 10,000 | 提供质量和体积约束，防止物体塌陷 |

**三者的地位差异**：
- Object 点是"观测点"——它们对应 CoTracker 追踪的 2D 像素，物理优化时计算 Chamfer 距离；
- Surface 点是"辅助点"——补充表面覆盖，避免 FPS 采样不足导致表面漏点；
- Interior 点是"支撑点"——纯物理作用，不参与误差计算。

**为什么不直接用网格顶点**：网格顶点分布不均（Poisson 重建在细节区域密集、平坦区域稀疏），且顶点数过多（9,500 点仿真太慢）或过少（100 点无法表达变形）。需要一个采样策略来平衡覆盖度和计算量。

---

## 二、Object 点：最远点采样（FPS）

FPS 是贪心算法——从一个种子点开始，每次选"离已选集合最远的点"，保证采样点在空间中尽可能均匀。

```mermaid
flowchart LR
    A["随机种子点 p₀"] --> B["找离 p₀ 最远的 p₁"]
    B --> C["找离 {p₀,p₁} 最远的 p₂"]
    C --> D["...重复 N 次"]
    D --> E["N 个均匀分布点"]
```

**算法伪代码**：
```
S = 网格所有顶点
选中 = [随机一个顶点]
for i in 1..N:
    距离 = [min(||p - q|| for q in 选中) for p in S]
    选中.append(S[argmax(距离)])
```

**为什么用 FPS 而不是均匀网格**：均匀网格（如体素中心）不适应物体形状——细长的绳子会在某个方向上只有一两个点。FPS 自适应物体的空间分布，细长方向自动采样更密。

**Bear 数据**：从 9,500 顶点 FPS 采样 1,024 个 object 点。计算量 $O(N \cdot M)$（$N$ 是目标点数，$M$ 是候选点数），Bear 数据约 2 秒。

---

## 三、Surface 点：随机表面采样

trimesh 的 `sample_surface(count=N)` 按面积加权随机采样：

1. 计算每个三角面片的面积
2. 按面积比例分配采样点数（大面片多采、小面片少采）
3. 在每个面片内用重心坐标随机采样

**为什么需要 Surface 点**：FPS 只选网格顶点，而顶点之间的表面可能有 5mm+ 的间隔（对毛绒熊这种 10cm 尺度的物体来说不算小）。Surface 点填补这些空白。

**Bear 数据**：随机表面采样 1,024 点。

---

## 四、Interior 点：体积内随机采样

trimesh 的 `sample_volume(count=N)` 用拒绝采样：

1. 生成包围盒内的随机点
2. 用 `contains` 检查点是否在网格内部（winding number 或 ray casting）
3. 保留内部点、拒绝外部点，直到凑够 $N$ 个

**为什么需要这么多 interior 点**：物体受外力时，如果只有表面点，表面容易"塌陷"进内部（没有任何东西抵抗）。Interior 点通过弹簧连接提供"体积弹性"——你压缩表面，内部弹簧被压缩产生反弹力。

**Bear 数据**：体积采样目标 10,000 点（实际生成可能略多，后续去重时裁掉）。

---

## 五、5mm 体素去重：优先级与分组

三组点加起来 12,000+ 个，但很多点在空间中距离很近（<5mm），如果都保留会导致：
1. 弹簧质量矩阵条件数爆炸（两个点几乎重合，弹簧刚度趋于无穷）
2. 仿真不稳定、数值误差累积

**去重策略**：用 5mm 体素网格，每个体素只保留一个点，**优先级规则**：

$$
\text{object} > \text{surface} > \text{interior}
$$

**这个规则在做什么**：同一个 5mm×5mm×5mm 的立方体里如果有多个点，优先保留 object 点（因为它参与误差计算），其次 surface，最后 interior。Interior 点最容易被牺牲——它们是"填充物"，损失几百个不影响外形。

**实现**：
```python
voxel_size = 0.005  # 5mm
coords = np.floor(all_points / voxel_size).astype(int)
seen = {}
for i, (coord, label) in enumerate(zip(coords, labels)):
    key = tuple(coord)
    if key not in seen or priority[label] > priority[seen[key][1]]:
        seen[key] = (i, label)
kept = [i for i, _ in seen.values()]
```

**Bear 数据**：12,048 原始点 → 4,074 去重后唯一点（保留率 33.8%）。

---

## 六、最终统计与分布

去重后三组点的最终数量：

| 组 | 去重前 | 去重后 | 保留率 |
|----|-------|-------|--------|
| Object | 1,024 | 1,024 | 100% |
| Surface | 1,024 | 827 | 80.8% |
| Interior | 10,000 | 2,223 | 22.2% |
| **总计** | **12,048** | **4,074** | **33.8%** |

**为什么 object 保留率 100%**：FPS 本身已经保证点之间距离较远（算法的贪心性质），基本不会出现两个 object 点落在同一个 5mm 体素的情况。

**为什么 interior 保留率最低**：体积采样是完全随机的，点密度不均匀，大量点聚集在局部区域，去重时被优先级更高的 surface/object 点替代。

---

## 七、为什么不在去重后再分类

Bear Twin 的实现顺序是：**先生成三组 → 去重 → 保留分组标签**。能否反过来：**先合并所有候选点 → 去重 → 事后按位置分类**？

不能。原因是 object 点的身份很重要——后续 CoTracker 要追踪这些点在视频中的轨迹，如果去重时某个 object 点被替换成附近的 surface 点，这个"替身"的初始位置和真正的 object 点有偏差，追踪起始帧就对不上。

保留分组标签 → 后续 `final_data.pkl` 知道"前 1,724 个点是 object、接下来 827 个是 surface、最后 2,223 个是 interior"。

---

## 八、对弹簧连接的预告

`step2_particles.py` 只生成粒子，不生成弹簧。下一章（Chapter 3）用 KDTree 在这 4,074 个点之间连弹簧——半径 12mm、每个点最多连 10 个邻居 → 最终 ~13,400 根弹簧。

**粒子数和弹簧数的关系**：平均每个粒子约 6.6 根弹簧（13,400 / 4,074 / 2，因为每根弹簧连两个点）。这是一个稀疏连接——比完全图（$\binom{4074}{2} \approx 8$ 百万）小三个数量级。

---

## 输出检查清单

```bash
# 验证粒子数量和范围
python3 -c "
import numpy as np
pts = np.load('artifacts/13_voxel_points_5mm.npy')
print(f'n_particles: {len(pts)}')
print(f'bounds: {pts.min(axis=0)} to {pts.max(axis=0)}')
print(f'mean_spacing: {np.linalg.norm(pts[1:] - pts[:-1], axis=1).mean():.4f} m')
"
# 预期：n_particles=4074, bounds≈[-0.04, 0.05, 0]~[0.05, 0.1, 0.18], mean_spacing≈0.01m
```