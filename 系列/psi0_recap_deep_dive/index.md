---
title: "Ψ₀-Recap 代码深度拆解：从 rollout 数据到 advantage-conditioned VLA 微调"
order: 316
tags: [强化学习, VLA, RECAP, advantage conditioning, 分布式值函数, Ψ₀, 工程实践, 开源复现, 系列]
category: 系列
star: 4
series:
  id: psi0_recap_deep_dive
  totalChapters: 5
  dir: /系列/psi0_recap_deep_dive
---

# Ψ₀-Recap 代码深度拆解：从 rollout 数据到 advantage-conditioned VLA 微调

> Stanford CS224R 课程项目 [Psi0-Recap](https://github.com/ashah1002/Psi0-Recap) 是目前公开的最完整的 RECAP 开源复现。本系列逐文件拆解它的实现，对照 π\*₀.₆ 原文的设计意图，看一个学术复现在哪些地方做了简化、在哪些地方忠实还原了原始 recipe。

## 系列简介

[RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）](/论文综述/016_RECAP_从真实部署经验中RL学习)是 Physical Intelligence 提出的、让 VLA 模型通过真实部署经验做 RL 改进的方法。核心思路是：训练一个 Critic → 算出优势 → 二值化 → 作为文本条件输入训练 VLA。原论文没有开源代码。

[Psi0-Recap](https://github.com/ashah1002/Psi0-Recap) 是 Stanford CS224R (Spring 2026) 的期末项目，三位学生基于开源的 [Ψ₀ 人形 VLA](https://github.com/physical-superintelligence-lab/Psi0) 实现了完整的 RECAP pipeline，在 SIMPLE 仿真的 bendpick 任务上验证。

**本系列的价值**：

- 这是目前唯一一个完整的、可运行的 RECAP 开源实现
- 代码量小（核心 RECAP 模块约 500 行），适合逐行阅读
- 每个 pipeline 步骤都有独立的脚本入口，方便理解和复用
- 与原文的设计差异清晰可辨，能帮助读者分辨"哪些是 RECAP 的核心不可省"和"哪些是 PI 原文特有的工程选择"

**本系列不讲什么**：

- 不重复 [RECAP 原理篇](/论文综述/016_RECAP_从真实部署经验中RL学习) 已经讲过的公式推导和动机
- 不讲 Ψ₀ 模型本身的预训练/后训练流程（那是上游的基座，本系列只关心 RECAP 叠加在基座之上的部分）
- 不讲 SIMPLE 仿真环境的搭建细节

## 章节目录

| 章节 | 标题 | 简介 |
|------|------|------|
| 01 | [仓库全景与 Pipeline 总览](./01_仓库全景与Pipeline总览) | 目录结构、数据流向、五个脚本的执行顺序 |
| 02 | [分布式价值函数：网络结构与训练](./02_分布式价值函数_网络结构与训练) | ResNet-50 + MLP → 201 bins 分类、return 计算、episode 级分层验证 |
| 03 | [优势计算与标签生成](./03_优势计算与标签生成) | Monte Carlo advantage、百分位阈值二值化、Parquet 标签文件格式 |
| 04 | [Advantage Conditioning 接入 Ψ₀ 训练](./04_Advantage_Conditioning接入训练) | SimpleRepackTransform 的 prefix 注入、AdvantageLabelStore 查表、微调脚本参数 |
| 05 | [与原文的设计差异总结](./05_与原文的设计差异总结) | 简化了什么、保留了什么、哪些地方值得二次开发 |

## 前置知识

阅读本系列前建议了解：

- [RECAP：从真实部署经验中 RL 学习](/论文综述/016_RECAP_从真实部署经验中RL学习) — 原理篇，讲清楚整套方法的设计动机
- [RECAP 工程实践：训练流程、数据管道与 Loss 实现](/工程实践/RECAP_训练流程与Loss工程实现) — 论文级别的工程描述
- [Advantage Conditioning：优势条件化策略提取](/前置知识/002r_前置知识_Advantage_Conditioning优势条件化策略提取) — 数学推导
- [分布式值函数与类别化回报预测](/前置知识/002q_前置知识_分布式值函数与类别化回报预测) — Critic 训练原理

## 仓库信息

| 维度 | 信息 |
|------|------|
| GitHub | [ashah1002/Psi0-Recap](https://github.com/ashah1002/Psi0-Recap) |
| 上游 | [physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0) |
| 作者 | Jonathan Lu, Karthik Pythireddi, Aadi Shah (Stanford CS224R Spring 2026) |
| 许可证 | Apache 2.0 |
| 验证环境 | SIMPLE bendpick 任务（MuJoCo + Isaac Sim 渲染） |
| 基座模型 | Ψ₀（Qwen3-VL-2B + 500M Flow Matching action expert） |
| RECAP 核心代码 | `src/psi/recap/`（约 500 行）+ `scripts/recap/`（约 400 行） |
