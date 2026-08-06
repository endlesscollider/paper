---
title: "Fast-WAM 深度解析：从 World-Action Model 到实时机器人控制"
order: 600
tags: [WAM, 机器人学习, Flow Matching, DiT, MoT, 视频生成, 扩散模型, 系列]
category: 系列
star: 5
series:
  id: fastwam_deep_dive
  totalChapters: 21
  dir: /系列/fastwam_deep_dive
---

# Fast-WAM 深度解析：从 World-Action Model 到实时机器人控制

逐层拆解 Fast-WAM 的每一个网络组件、每一个设计决策、每一条数据流。

## 系列简介

Fast-WAM 基于 Mixture-of-Transformers (MoT) 架构，在 Wan2.2 视频生成模型上加入轻量 Action Expert，共享注意力联合训练。核心创新：训练时联合去噪获得物理理解，推理时 KV-Cache 跳过视频生成，190ms 延迟实时出动作。

## 章节目录

### 第一部分：设计哲学与全局架构

| 章节 | 标题 |
|------|------|
| 01 | [WAM 范式：为什么机器人需要想象未来](./01_WAM范式) |
| 02 | [Fast-WAM 设计哲学：训练时想象推理时跳过](./02_设计哲学) |
| 03 | [全局架构鸟瞰：五大组件与数据流](./03_全局架构鸟瞰) |

### 第二部分：3D-VAE 与视频表示

| 章节 | 标题 |
|------|------|
| 04 | [Wan2.2 3D-VAE：视频的 latent 表示](./04_3DVAE视频表示) |

### 第三部分：Video Expert (WanVideoDiT)

| 章节 | 标题 |
|------|------|
| 05 | [Video Expert 总览](./05_VideoExpert总览) |
| 06 | [DiTBlock 逐层拆解](./06_DiTBlock逐层拆解) |
| 07 | [Separated Timestep 与 3D RoPE](./07_SeparatedTimestep与3DRoPE) |
| 08 | [Video Attention Mask 与 Head 输出](./08_VideoAttentionMask与Head) |

### 第四部分：Action Expert (ActionDiT)

| 章节 | 标题 |
|------|------|
| 09 | [ActionDiT 网络结构与维度对齐](./09_ActionDiT网络结构) |
| 10 | [Backbone 权重初始化：线性插值](./10_Backbone权重初始化) |

### 第五部分：MoT (Mixture-of-Transformers)

| 章节 | 标题 |
|------|------|
| 11 | [MoT 联合注意力机制](./11_MoT联合注意力) |
| 12 | [MoT Attention Mask 三种模式详解](./12_MoT_AttentionMask详解) |

### 第六部分：Flow Matching 训练原理

| 章节 | 标题 |
|------|------|
| 13 | [Continuous Flow Matching 数学原理](./13_FlowMatching数学原理) |
| 14 | [Scheduler 实现详解](./14_Scheduler实现详解) |

### 第七部分：训练系统

| 章节 | 标题 |
|------|------|
| 15 | [数据管道：RobotVideoDataset](./15_数据管道) |
| 16 | [FastWAMProcessor：归一化与数据变换](./16_Processor数据变换) |
| 17 | [T5 预计算与条件注入](./17_T5预计算与条件注入) |
| 18 | [training_loss 完整流程](./18_TrainingLoss完整流程) |
| 19 | [Wan22Trainer 训练循环与分布式](./19_Trainer训练循环) |

### 第八部分：推理架构

| 章节 | 标题 |
|------|------|
| 20 | [infer_action：KV-Cache Prefill-Decode](./20_infer_action推理流程) |
| 21 | [infer_joint 与 IDM 推理模式](./21_infer_joint与IDM推理) |

## 前置知识

- Transformer Self-Attention / Cross-Attention
- 扩散模型基础：前向加噪、反向去噪
- 机器人模仿学习：observation, policy, action

## 论文信息

- 标题：Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- 作者：Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao
- 链接：[arXiv 2603.16666](https://arxiv.org/abs/2603.16666)
- 代码：[GitHub](https://github.com/yuantianyuan01/FastWAM)
- 模型：[HuggingFace](https://huggingface.co/yuanty/fastwam)
