---
title: "ShapeNet Part：16 类点云部件标注"
order: 3
category: CAD 合成
tags: [数据集, 部件分割, 点云, ShapeNet, 基线评测]
star: 4
date: "2016-10-01"
---

# ShapeNet Part：16 类点云部件标注

**论文**：Yi et al., "A Scalable Active Framework for Region Annotation in 3D Shape Collections", SIGGRAPH Asia 2016  
**官网**：[https://shapenet.org](https://shapenet.org)  
**许可证**：ShapeNet Terms of Use（学术免费）  
**规模**：31,963 个模型，16 个类别，50 个部件类型

## 核心特点

ShapeNet Part 是历史最悠久、引用最广泛的部件分割基准，PointNet、PointNet++、DGCNN 等点云方法都在这里评测。与 PartNet 相比，粒度更粗（只有 1-6 个部件/物体），但格式更简单，每个点直接带一个部件 ID，无需解析层次结构。

涵盖的 16 个类别：

```
飞机 Airplane · 自行车 Motorbike · 椅子 Chair · 桌子 Table
汽车 Car · 吉他 Guitar · 灯具 Lamp · 摩托车 Motorcycle
杯子 Mug · 手枪 Pistol · 火箭 Rocket · 滑板 Skateboard
勺子 Earphone · 刀 Knife · 笔记本 Laptop · 花盆 Bag
```

其中 Motorbike（摩托车）有轮子、车架、车把、灯等部件标注，是与"轮子分离"最接近的类别。

## 标注格式

每个物体对应一个 `.pts` 点云文件和一个 `.seg` 标签文件：

```
# model_0001.pts  — 每行是一个点的 x y z
0.1234 -0.5678 0.9012
...

# model_0001.seg  — 每行是对应点的部件 ID（从1开始）
3
3
1
2
...
```

部件 ID 到语义的映射在类别元数据中给出。这种格式可以直接用 numpy 读取，不需要额外的解析库。

## 与 PartNet 的对比

| 维度 | ShapeNet Part | PartNet |
|------|--------------|---------|
| 类别数 | 16 | 24 |
| 标注粒度 | 粗（1-6 部件） | 细（层次化 3 级） |
| 格式 | 点云 + 标签 | Mesh + JSON |
| 铰接信息 | ❌ | ❌ |
| 使用门槛 | 极低 | 中等 |
| 主流用途 | 点云分割基线 | 层次化分割研究 |

## 在 3DGS 中的使用方式

ShapeNet Part 没有多视角图像，需要先将点云渲染成多视角 RGB 图，或借用 ShapeNet 的完整 mesh 渲染。常见做法是用 ShapeNet 的 mesh 配合 Blender 渲染多视角图，再用 ShapeNet Part 的点云标签作为 3D GT。

由于粒度较粗，ShapeNet Part 更适合做**物体级分割的上界估计**，而不是细粒度零件分割研究。如果需要更细的部件，应使用 PartNet。

**知识链接**

- [PartNet：更细粒度的层次化版本](/数据集/001_PartNet_层次化部件分割数据集)
