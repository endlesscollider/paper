---
title: "TRELLIS 深度解析：从一张图到三维资产的完整链路"
order: 910
tags: [3D生成, TRELLIS, SLAT, Rectified Flow, 3D Gaussian Splatting, VAE, 稀疏表示, 微软, 系列]
category: 系列
star: 5
series:
  id: trellis_deep_dive
  totalChapters: 7
  dir: /系列/trellis_deep_dive
---

# TRELLIS 深度解析：从一张图到三维资产的完整链路

> 输入一张图片，输出可以直接用于游戏、仿真或机器人场景的三维资产——高斯、网格、辐射场，三种格式随选随用。本系列从设计动机出发，把 TRELLIS 每一个组件的结构和工作原理拆解清楚。

## 系列简介

TRELLIS 是微软研究院发布的大规模三维资产生成模型，核心贡献是一种叫做 **SLAT（Structured LATent，结构化潜变量）** 的统一三维表示。传统方法要么只生成网格，要么只生成辐射场，而 SLAT 把三维空间结构和外观信息编码在同一个稀疏体素潜空间里，解码时才按需输出格式——同一份 latent，接上不同的解码头，就能分别得到三维高斯、网格、或辐射场。

模型的整体生成流程分为两阶段：先用 Rectified Flow 扩散模型生成稀疏占据结构（哪些体素是"实心的"），再在这个稀疏结构上用第二个 Rectified Flow 生成每个体素的 SLAT 特征。整个过程以 DINOv2 编码的图像特征为条件，骨干网络是专门为稀疏结构改造的 Sparse Transformer。

最大模型有 **2B 参数**，在包含 50 万个三维物体的 TRELLIS-500K 数据集上训练。

**适合读者**：
- 对三维生成感兴趣、想理解 SLAT 为什么比点云/网格/NeRF 更适合做统一表示的研究者
- 使用 3D Gaussian Splatting 做仿真或机器人场景、想了解如何用生成模型自动建模的工程师
- 熟悉扩散模型或 Flow Matching，想看它在三维空间上如何运作的读者
- 想读懂 TRELLIS 源码、理解 SparseTensor 和稀疏 Transformer 怎么工作的开发者

**你将获得**：
- 对 SLAT 表示设计的完整理解：为什么用稀疏体素、为什么多视角特征能编码几何和外观
- 对两阶段生成流程的机制认知：占据结构 → SLAT 特征的两次 Rectified Flow 采样
- 对 SparseStructureVAE 和 SLatVAE 编码器/解码器架构的代码级认识
- 对三种解码头（高斯/网格/辐射场）各自输出格式和设计取舍的理解
- 对图像条件如何注入（DINOv2 patch token → cross-attention）的完整链路
- 读懂 `TrellisImageTo3DPipeline.run()` 每一行代码的能力

**前置知识**：
- [3D Gaussian Splatting：用一堆椭球把真实场景搬进电脑](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建) — 理解高斯解码头的输出格式
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — 理解 Rectified Flow 生成过程

## 章节目录

| 章节 | 标题 | 简介 |
|------|------|------|
| **第一部分：全局认知** | | |
| 01 | [全景图：TRELLIS 在解决什么问题？](./01_全景图_TRELLIS在解决什么问题) | 三维生成的挑战、SLAT 的核心思路、两阶段流程鸟瞰 |
| 02 | [SLAT 表示设计：为什么是稀疏体素 + 多视角特征](./02_SLAT表示设计_稀疏体素与多视角特征) | SLAT 是什么、和点云/NeRF/网格相比的优势、数据格式 |
| **第二部分：生成主干** | | |
| 03 | [第一阶段：稀疏占据结构的 Rectified Flow 生成](./03_稀疏占据结构生成_第一阶段Flow) | SparseStructureFlowModel 架构、稀疏结构如何从噪声采样出来 |
| 04 | [第二阶段：SLAT 特征的 Rectified Flow 生成](./04_SLAT特征生成_第二阶段Flow) | SLatFlowModel、稀疏 Transformer、SLAT VAE 编解码 |
| **第三部分：解码头与输出** | | |
| 05 | [三种解码头：高斯、网格、辐射场的输出机制](./05_三种解码头_高斯网格辐射场) | 每个解码头的结构、输出参数、为什么同一 SLAT 能解出三种格式 |
| **第四部分：图像条件与完整流水线** | | |
| 06 | [图像条件注入：DINOv2 特征如何驱动生成](./06_DINOv2图像条件注入) | 图像预处理、DINOv2 编码、patch token 如何进入 cross-attention |
| 07 | [完整 Pipeline：从一张图到 GLB 文件的每一步](./07_完整Pipeline走读与参数调优) | `run()` 方法完整走读、采样参数调优、多图条件与变体生成 |
