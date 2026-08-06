---
title: "从示教数据到一次成功抓取：miGenRL 的 BC 训练与 Rollout 全链路解析"
order: 320
tags: [机器人学习, 双臂协作, ACT, 行为克隆, rot6d, PCA, 系列]
category: 系列
star: 5
series:
  id: stack_can_drawer_migenrl_deep_dive
  totalChapters: 9
  dir: /系列/stack_can_drawer_migenrl_deep_dive
---

# 从示教数据到一次成功抓取：miGenRL 的 BC 训练与 Rollout 全链路解析

> 一批 HDF5 示教轨迹，怎么变成一个能在仿真里稳定抓取、放置的策略网络？这中间要经过配置继承、六维旋转表征、参考系动态切换、PCA降维、关键帧采样、DETR-VAE网络、加权损失函数、时间聚合推理——这个系列把 `miGenRL` 这一个子系统从头拆到尾。

## 系列简介

`miGenRL` 是 embodied-arena 里专门负责"用示教数据训练操作策略、再让策略控制仿真环境"的子系统。这个系列以一份真实的训练配置为线索：

```
config/migenrl/stack_can_drawer_bimanual_base.yaml
```

**系列的边界很明确：只讲 `miGenRL` 自己的代码和配置——数据表征转换、数据管线、网络架构、BC训练循环、Rollout推理循环。不讲仿真环境怎么搭建、机器人怎么定义、任务怎么判成功——那些是 embodied-arena 环境侧的工作，`miGenRL` 只是这个环境的一个"数据进、动作出"的消费者。**

读完这个系列，你应该能独立地在 `miGenRL` 这套框架里改配置、调参数、排查训练和推理链路上的问题，并且知道每一处修改会通过哪条路径影响最终结果。

## 贯穿全系列的例子

全系列使用同一个任务的配置数字，反复对照：

```text
配置       config/migenrl/stack_can_drawer_bimanual_base.yaml
观测       3 路相机 depth_mask + 双臂末端位姿 + 手指关节
动作表征   rot6d_processed_actions，target_object_frame 参考系
动作维度   62 维原始 → PCA 降维后 30 维（双臂各 15 维：3 位置 + 6 旋转 + 6 手部隐变量）
策略       ACT（DETR-VAE + 双臂解码头 + 手部交叉注意力解码器）
训练       workflow.stages: [bc]，240 epoch，keyframe 采样，early stopping patience 40
评估       6 个 rollout episode，temporal_aggregation，decay 0.05，query_frequency 1
```

每一章都会把这组具体数字代入公式里演算一遍。

## 章节目录

| 章节 | 标题 | 一句话简介 |
|------|------|------------|
| 01 | [全链路总览](./01_全链路总览) | `miGenRL` 内部"配置→数据→训练→推理"的完整模块地图 |
| 02 | [配置继承链](./02_配置继承链) | `miGenRL` 自己的三层 YAML `base:` 合并机制、`deep_merge` 的具体规则 |
| 03 | [数据表征：rot6d 与参考系](./03_数据表征_rot6d与参考系) | 为什么用 6D 旋转表征而不是四元数或欧拉角，`target_object_frame` 参考系的构造公式 |
| 04 | [参考系切换与手部降维](./04_hand_synergy_PCA降维) | `policy_frame_id` 参考系切换机制，`hand_synergy` PCA 如何把 22 维手指关节压成 6 维 |
| 05 | [数据管线：从 HDF5 到训练 batch](./05_数据管线) | keyframe 采样策略的评分算法、窗口展开逻辑，为什么长 horizon 任务不能均匀采样 |
| 06 | [ACT 策略网络架构](./06_ACT策略网络架构) | DETR-VAE 双臂解码、手部交叉注意力解码器，输入输出的完整 tensor shape 走读 |
| 07 | [BC 训练循环](./07_BC训练循环) | 加权 L1 + KL 的损失函数、early stopping 判定、多套 checkpoint 策略 |
| 08 | [Rollout 推理循环](./08_Rollout推理循环) | temporal aggregation 时间聚合的指数衰减公式、动作后处理管线、成功率怎么算 |
| 09 | [实战调参与排错](./09_实战调参与排错) | 参数调优表、排错决策树、改配置前必查的坑位清单 |

## 前置知识

建议按需查阅，不必读完再开始：

- [矩阵的秩与低秩近似](/前置知识/000z_前置知识_矩阵的秩与低秩近似) — 第 04 章 PCA 降维的数学基础
- [主成分分析 PCA](/前置知识/002j_前置知识_主成分分析PCA) — 第 04 章 hand_synergy 的完整数学原理
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式) — BC 训练的范式定位
- [对数似然与变分下界](/前置知识/000e_前置知识_对数似然与变分下界) — 第 06 章 CVAE 的 ELBO 来源
- [重参数化技巧](/前置知识/002e_前置知识_重参数化技巧) — 第 06 章 CVAE 采样的可导性
- [Cross Attention 与交替注意力机制](/前置知识/001e_前置知识_Cross_Attention与交替注意力机制) — 第 06 章手部解码器
- [动作平滑性正则化 CAPS](/前置知识/000i_前置知识_动作平滑性正则化CAPS) — 与第 08 章 temporal aggregation 的动机类似（另一种解决动作抖动的思路）

相关工程实践文章：

- [ACT Decoder 架构详解](/工程实践/ACT_Decoder架构详解) — 第 06 章会大量引用这篇的图示
- [条件约束的 ACT 模型](/工程实践/条件约束的ACT模型)
- [InterACT 与 ACT 的区别解析](/工程实践/InterACT与ACT的区别解析)
- [双臂任务训练方法研究](/工程实践/双臂任务训练方法研究)
- [从 ACT 到 PerAct2：双臂协调教程](/工程实践/从ACT到PerAct2_双臂协调教程)

## 阅读建议

**只想知道 `miGenRL` 这条链路长什么样**：读 01 章，20 分钟。

**要调训练配置**：读 02 → 05 → 07，再配合 09 章的手册。

**要理解动作是怎么从示教数据流到网络、又从网络流回仿真的**：读 03 → 04，这两章是全系列最"硬核"的坐标变换部分。

**要理解评估时动作为什么会抖或者不抖**：读 08 章。

---

开始读：[第 01 章 全链路总览](./01_全链路总览)
