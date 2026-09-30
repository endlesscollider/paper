---
title: TRELLIS.2 深度解析：从稀疏体素到全材质 3D 资产生成
description: 从 SLat 表示到 O-Voxel，从 Rectified Flow Transformer 到三阶段推理流水线，完整解析微软 TRELLIS.2 4B 参数图像转 3D 模型的每一个设计决策。
order: 313
category: 系列
series:
  id: trellis2_deep_dive
  totalChapters: 9
  dir: 系列/trellis2_deep_dive
---

# TRELLIS.2 深度解析：从稀疏体素到全材质 3D 资产生成

给一张图片，10 秒后得到一个带 PBR 材质的完整三维网格——这是 TRELLIS.2 能做到的事。它是微软研究院的一个 4B 参数的图像转 3D 生成模型，论文发表于 2025 年（arXiv 2512.14692）。

TRELLIS.2 的核心贡献有两个：**SLat（结构化潜表示）** 和 **O-Voxel（开放体素）**。SLat 是一个稀疏体素网格，每个被激活的体素携带一个局部潜向量——它同时编码了形状和材质，是整个生成流水线的枢纽。O-Voxel 是 TRELLIS.2 相比原版 TRELLIS 最关键的改进：它把网格直接存进体素格子，彻底绕开了 SDF 等值面提取，不再丢失开放曲面和非流形拓扑。

生成过程分三个串联的 Rectified Flow Transformer 阶段：先生成稀疏结构（哪些体素是活跃的），再在这些体素上生成形状潜量，最后叠加纹理潜量，得到同时携带几何和 PBR 材质的完整 SLat，解码为可用的 GLB 网格。

本系列从最底层的动机出发，逐步拆解每个设计决策，包含代码级别的工程实现分析。

---

## 目录

| 章节 | 内容 |
|------|------|
| [第一章 为什么 3D 生成这么难](./01_为什么3D生成这么难_已有表示方法与设计动机.md) | 现有 3D 表示的局限，统一潜空间的设计动机 |
| [第二章 SLat 结构化潜表示](./02_SLat结构化潜表示_稀疏体素与局部潜量设计.md) | 稀疏体素网格、局部潜向量、为什么这个设计能支持区域编辑 |
| [第三章 O-Voxel 原生三维表示](./03_O-Voxel_告别等值面的原生三维表示.md) | TRELLIS.2 的核心创新：field-free 体素表示，任意拓扑，PBR 材质 |
| [第四章 编码器：SC-VAE 与 DINOv3](./04_编码器_SC-VAE与DINOv3特征聚合.md) | 多视角特征聚合，稀疏卷积 VAE，形状与纹理的分离编码 |
| [第五章 生成骨干：Rectified Flow Transformer](./05_生成骨干_Rectified_Flow_Transformer架构.md) | 整流流、稀疏 DiT、adaLN、跨注意力图像条件 |
| [第六章 三阶段推理流水线](./06_三阶段推理流水线_从图像到完整三维资产.md) | 结构流 → 形状流 → 纹理流，级联上采样，CFG |
| [第七章 解码与导出](./07_解码与导出_O-Voxel转GLB网格.md) | O-Voxel 转 GLB 全流程，UV 展开，PBR 纹理烘焙 |
| [第八章 训练流程](./08_训练流程_数据准备与流匹配目标.md) | Objaverse-XL 数据管线，SC-VAE 训练，流匹配 CFG 目标 |
| [第九章 代码实战](./09_代码实战_从图像到3D的完整链路.md) | example.py 到推理链路，纹理化 pipeline，工程细节 |

---

## 前置知识

读这个系列，你需要了解以下概念。项目里已有对应文章的，点链接直接读：

- [3D Gaussian Splatting](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建) — 理解 3DGS 的高斯表示和可微渲染
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — 理解整流流（Rectified Flow）的数学基础
- [向量量化与离散表示学习](/前置知识/001q_前置知识_向量量化与离散表示学习) — VAE 和 VQ-VAE 的基本概念
- [常微分方程 ODE](/前置知识/001b_前置知识_常微分方程ODE直觉与数值求解) — 流匹配推理时 ODE 求解的直觉

---

## 参考资料

- 论文（原版 TRELLIS）：[Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506)，CVPR 2025 Spotlight
- 论文（TRELLIS.2）：[Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692)
- 代码仓库：`/home/flageval/projects/TRELLIS.2`
- 项目页面：[microsoft.github.io/TRELLIS](https://microsoft.github.io/TRELLIS/)
