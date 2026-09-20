---
title: "π₀.₇-inspired G1 VLA：从遥操作数据到 Flow Matching 部署"
order: 309
tags: [VLA, 机器人学习, Unitree G1, π₀.₇, OpenPI, Flow Matching, 系列]
category: 系列
star: 5
series:
  id: pi07_flow_matching_deep_dive
  totalChapters: 9
  dir: /系列/pi07_flow_matching_deep_dive
---

# π₀.₇-inspired G1 VLA：从遥操作数据到 Flow Matching 部署

> 一套面向 Unitree G1 43 DoF 人形机器人的工程实现：把 XR 遥操作、ROS2 记录、LeRobot 数据、π₀.₅ Flow Matching、FAST 规划分支、RTC 和安全门串成一条可检查的链路。

## 系列定位

这个系列解析的是 [Pi0.7-Flow-matching](https://github.com/SILVIO-ZHENG/Pi0.7-Flow-matching) 项目，而不是 Physical Intelligence 官方发布的 π₀.₇。项目 README 已经明确说明：可执行的模型路径基于公开 OpenPI π₀.₅ PyTorch Flow Matching 实现，π₀.₇ 是项目对自身整条系统的命名。文章因此把“官方 OpenPI 已有能力”和“这个项目新增的 G1、训练与部署扩展”分开标记。

已有 [OpenPI 深度解析系列](/系列/openpi_deep_dive/) 负责 π₀、π₀.₅ 的基础架构、数据变换和模型代码。本系列不重复解释那些通用模块，而是回答一个更工程化的问题：**一台拥有 43 个关节、三路相机和两只 Dex3-1 灵巧手的 G1，怎样从一次 XR 示教走到安全执行的 32 维动作块？**

## 全链路

图中的每一条边都是一个需要保持数据契约的边界；任何一个边界的形状、时间戳或关节顺序错位，都会在模型训练或真机控制时暴露出来。

```mermaid
flowchart LR
    A["XR 手腕位姿 + 21x3 手部关键点"] -->|标定与重定向| B["双臂 IK + Dex3 目标"]
    B -->|43 DoF 候选动作| C["ROS2 状态 / 动作 / 相机流"]
    C -->|动作时间戳对齐| D["MCAP + Parquet + MP4"]
    D -->|episode QC 与分割| E["LeRobot V3 数据集"]
    E -->|q01/q99 + H=50 + mask| F["32 维动作块"]
    F -->|FAST CE + Flow Matching| G["π₀.₅ backbone + Action Expert"]
    G -->|异步请求 + RTC| H["滚动动作缓存"]
    H -->|关节限位 / 急停 / 新鲜度| I["G1 安全门与执行器"]
```

> 读图时重点看三次维度变化：完整机器人状态保留 43 维，策略实际控制 28 维，模型固定接收 32 维；最后 4 维不是新增执行器，而是被 mask 掉的模型 padding。

## 章节目录

| 章节 | 标题 | 解决的问题 |
|------|------|------------|
| 01 | [项目定位与系统全链路](./01_项目定位与系统全链路) | π₀.₇-inspired 到底指什么，代码边界在哪里 |
| 02 | [G1 43 DoF 关节契约](./02_G1_43DoF关节契约) | 43、28、32 三种表示如何无损转换 |
| 03 | [XR 遥操作与双臂动作生成](./03_XR遥操作与双臂动作生成) | 手腕位姿、IK、手部关键点怎样变成训练动作 |
| 04 | [动作中心的时间对齐与记录](./04_动作中心的时间对齐与记录) | 为什么用动作时间戳作为多传感器共同坐标 |
| 05 | [数据质检 LeRobot 与 q01 q99](./05_数据质检LeRobot与q01q99) | 失败轨迹、episode 尾部 padding 和归一化怎样处理 |
| 06 | [π₀.₅ Flow Action Expert](./06_π05_Flow_Action_Expert) | 视觉语言前缀如何条件化连续 50 步动作 |
| 07 | [FAST Flow 联合训练与知识隔离](./07_FAST_Flow联合训练与知识隔离) | 为什么用两个目标，以及梯度如何被隔离 |
| 08 | [RTC 异步推理与滚动动作块](./08_RTC异步推理与滚动动作块) | 远程推理延迟如何变成可训练的硬前缀 |
| 09 | [安全门与端到端复现实验](./09_安全门与端到端复现实验) | 怎样从离线回放逐步推进到真机命令 |

## 阅读前提

- 线性代数、反向传播和基本概率知识。
- [什么是 VLA？](/系列/openpi_deep_dive/01_什么是VLA) 与 [π₀ 一句话做了什么？](/系列/openpi_deep_dive/02_pi0一句话做了什么)。
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流)。
- [LeRobot 数据格式与 VLA 数据管线](/系列/openpi_deep_dive/05_数据格式入门)。

ROS2、MoveIt2、Unitree SDK2、CUDA 和预训练权重属于运行环境，不是阅读前提。系列中的代码路径全部以克隆到 `.pi07_src/` 的提交 `ebaeac0` 为依据，源码目录不纳入文章构建。

## 贯穿例子

全文使用一个固定任务：**G1 用双手拿起桌上的箱子，再把它放到目标区域**。这个任务同时需要手腕空间运动、手指闭合、当前关节状态、三路图像、阶段性 subtask 和连续动作块，因此能暴露整条链路最容易被忽略的边界。

## 延伸阅读

- [OpenPI 项目地图](/系列/openpi_deep_dive/03_OpenPI项目地图)：理解官方代码的模块划分。
- [π₀：通用机器人基础模型](/论文综述/014_Pi0_通用机器人基础模型)：阅读原始 π₀ 方法的研究背景。
- [Flow Matching 策略的强化学习方法全景综述](/论文综述/S22_Flow_Matching强化学习方法综述)：理解连续动作策略与强化学习的后续结合。
