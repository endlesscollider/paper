---
title: "fast-WAM 人类数据 World Model 实验全记录：从 Wan2.2 到人类数据 Pretrain"
order: 700
tags: [WAM, World Model, 视频生成, Wan2.2, EgoVLA, GR1, 人类数据, Pretrain, Flow Matching, 系列]
category: 系列
star: 5
series:
  id: fastwam_human_pretrain
  totalChapters: 13
  dir: /系列/fastwam_human_pretrain
---

# fast-WAM 人类数据 World Model 实验全记录：从 Wan2.2 到人类数据 Pretrain

> 完整记录 fast-WAM 模型在基于人类数据构建 World Model 过程中的每一步实验：从 Wan2.2 base model 的能力评估，到 EgoVLA/GR1 仿真微调，到数据配比/训练策略的系统性消融，再到 70D 统一动作空间设计与人类数据 pretrain 的过拟合分析。

## 系列简介

**fast-WAM (fast World-Action Model)** 是一种基于视频扩散模型的机器人策略学习方法。它的核心理念是：**用大规模视频生成模型作为 backbone，同时预测未来视频和机器人动作**——视频分支负责学习物理世界的动态规律，动作分支借助视频的物理先验来提高操控精度。

本系列完整记录了该模型从"选定 base model"到"尝试用人类 ego 视角数据进行 pretrain"这一过程中的所有实验。**不是论文精读，而是一份真实的工程实验记录**——包含大量闭环仿真结果、失败分析、消融实验、以及最终得到的核心结论。

本系列适合：
- 想了解"视频生成 + 动作预测"联合训练范式的研究者
- 对 Wan2.2 视频模型在机器人场景适配感兴趣的工程师
- 想学习如何系统性地做模型迭代、消融实验、瓶颈诊断的同学
- 对多机体统一动作空间设计感兴趣的开发者

## 章节目录

| 章节 | 标题 | 简介 |
|------|------|------|
| 01 | [Wan2.2 视频生成模型：整体架构与设计哲学](./01_Wan2.2整体架构与设计哲学) | Wan 系列演进、完整 pipeline、参数规模、T2V vs TI2V 的差异 |
| 02 | [Wan2.2 的 3D-VAE 与视频 Token 化](./02_Wan2.2的3D-VAE与视频Token化) | Wan-VAE 架构、时空压缩策略、latent 维度计算、分辨率敏感性的根源 |
| 03 | [Wan2.2 的 DiT 骨干与 Flow Matching](./03_Wan2.2的DiT骨干与FlowMatching) | DiT 逐层结构、3D RoPE、文本 cross-attention、CFG 策略、采样器对比 |
| 04 | [Wan2.2 在机器人场景的适配：LoRA 微调与推理对齐](./04_Wan2.2机器人场景适配) | OOD 表现、VITRA 数据微调、推理差异排查、8 卡训练效果 |
| 05 | [fast-WAM 在 EgoVLA 的首次闭环验证](./05_EgoVLA首次闭环验证) | 仿真环境与 12 任务、数据处理、首批训练配置、长任务全部失败的解读 |
| 06 | [fast-WAM 在 GR1 人形机器人上的微调](./06_GR1人形机器人微调) | 44D/29D 动作空间、video-to-action 模式、闭环 10% 的原因分析 |
| 07 | [数据配比实验：成功率到底由什么决定？](./07_数据配比实验) | 7 种方案对比、采样率 vs 成功率、seed 消融、12 任务稳定性分层 |
| 08 | [State Drop 与 History：排除 Shortcut 假设](./08_StateDrop与History实验) | state drop 消融、训练曲线不变 → 排除 shortcut、history 无效 |
| 09 | [瓶颈诊断：视频生成质量、物理失真与空间泛化](./09_瓶颈诊断) | 推理视频可视化、Wan2.2 OOD 测试、分辨率消融、空间泛化实验 |
| 10 | [优化探索：Video Pretrain、A2V、光流 Condition](./10_优化探索) | 仿真视频 pretrain、giga-policy 式 a2v、光流代理、任务描述修正 |
| 11 | [70D 统一手部动作空间：设计、可逆映射与验证](./11_70D统一动作空间) | 空间定义、MANO vs Inspire 分布对比、可逆编解码、归一化 |
| 12 | [人类数据 Pretrain 与过拟合全面分析](./12_人类数据Pretrain与过拟合) | EgoVerse 数据、效率评估、train/val 劈叉、过拟合根因分析 |
| 13 | [阶段总结与后续方向](./13_阶段总结与后续方向) | 核心结论汇总、瓶颈定位、后续计划 |

## 前置知识

阅读本系列前建议了解：
- Transformer Self-Attention / Cross-Attention 基本机制
- 扩散模型 / Flow Matching 的去噪直觉（本系列会讲，但有基础更好）
- 基本的模仿学习概念（state, action, policy, 闭环仿真）
- VAE（变分自编码器）的基本思想

## 学习建议

1. **第 1-4 章**：理解 Wan2.2 这个 base model 的架构和能力边界——这是后续所有实验的基础
2. **第 5-6 章**：看 fast-WAM 在两个仿真平台的初始表现——建立"当前能力"的概念
3. **第 7-9 章**：跟着实验一起诊断问题——为什么性能上不去？瓶颈在哪？
4. **第 10-12 章**：看各种优化尝试的效果——什么 work 什么不 work
5. **第 13 章**：总结全局认知，思考后续方向
