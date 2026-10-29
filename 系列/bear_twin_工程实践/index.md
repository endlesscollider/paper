---
title: "Bear Twin 工程实践：从点云到弹簧质点系统的构建流程"
description: "从 3D Gaussian Splatting 点云到水密网格、粒子采样、弹簧连接——构建可仿真弹簧质点系统的通用工程流程"
series_id: bear_twin_engineering
totalChapters: 3
order: 970
tags: [3DGS, Poisson重建, 网格修复, 粒子采样, 弹簧连接, 点云处理]
category: 系列
---

# Bear Twin 工程实践

Bear Twin 项目的前半段是一套通用的 3D 几何处理流程——从 3D Gaussian Splatting 点云出发，经过 Poisson 表面重建、网格修复、粒子采样、弹簧连接，最终构建出一个可仿真的弹簧质点系统。这一套流程不依赖任何具体的物理仿真论文方法，是从静态重建到物理仿真的"标准件"。

---

## 项目定位

本系列聚焦**工程实现**：如何把一团高斯点云变成一个拓扑合理、密度均匀、能跑物理仿真的弹簧质点系统。不涉及视频感知、参数优化等下游任务——那些属于具体的仿真方法（如 PhysTwin）。

**输入**：3DGS 训练产出的 PLY 文件（32,704 个高斯核）

**输出**：弹簧质点系统的几何定义——4,074 个粒子 + 13,228 根弹簧 + 每根弹簧的静止长度

```mermaid
flowchart LR
    A["3DGS PLY\n32704 高斯"] --> B["Ch1: Poisson 重建\n水密网格"]
    B --> C["Ch2: 三层采样\n4074 粒子"]
    C --> D["Ch3: 弹簧构建\n13228 弹簧"]
    D --> E["几何定义完成\n可导入任意仿真器"]
```

---

## 章节目录

| 章 | 标题 | 对应脚本 | 关键输出 |
|----|------|---------|---------|
| 1 | [从高斯溅射到水密网格：Poisson 重建与网格修复](./01_从高斯溅射到水密网格Poisson重建与网格修复) | step1_reconstruct, step1_fix | 水密网格 ~9,500 面片 |
| 2 | [粒子三层采样：Object / Surface / Interior 的生成与去重](./02_粒子三层采样Object_Surface_Interior的生成与去重) | step2_particles | 4,074 个粒子 |
| 3 | [弹簧连接图构建：KDTree 邻域搜索与拓扑参数权衡](./03_弹簧连接图构建KDTree邻域搜索与拓扑参数权衡) | step2b_springs | 13,400 根弹簧 |

---

## 前置知识

- [3D Gaussian Splatting](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建) — 输入 PLY 的来源
- [Poisson 表面重建](/前置知识/020a_前置知识_Poisson表面重建从点云到水密网格) — 第一章的核心算法
- [质点弹簧模型](/前置知识/006e_前置知识_质点弹簧模型与绳索仿真基础) — 弹簧力公式与系统组成

---

## 下游应用

本系列的输出（粒子 + 弹簧）可以直接导入任何弹簧质点仿真器：
- [PhysTwin 深度解析系列](/系列/phystwin_deep_dive/) — 从视频校准物理参数的完整方法
- NVIDIA Warp、MuJoCo、Taichi 等通用仿真引擎

---

## 工程特点

1. **水密性保证**：Poisson 重建 + 空洞填充双保险，确保网格无破洞、无自交
2. **三层粒子设计**：Object（追踪标记点）、Surface（外形覆盖）、Interior（体积支撑），各有分工
3. **拓扑参数权衡**：弹簧半径 12mm、max_nb=10，在计算量和连接性之间取得平衡
4. **5mm 体素去重**：避免粒子过密导致数值不稳定

---

## 源代码

Bear Twin 项目源码位于 `/home/flageval/projects/bear_twin/`，本系列覆盖其中 step1–step3 的脚本。
