---
title: "LingBot-VLA 2.0 代码深度解析：从统一动作表示到 MoE 动作专家与双查询蒸馏"
order: 308
tags: [VLA, LingBot, Qwen3-VL, MoE, 双查询蒸馏, Flow Matching, 统一动作表示, Muon, 系列]
category: 系列
star: 5
series:
  id: lingbot_vla2_deep_dive
  totalChapters: 8
  dir: /系列/lingbot_vla2_deep_dive
---

# LingBot-VLA 2.0 代码深度解析：从统一动作表示到 MoE 动作专家与双查询蒸馏

> 从"20 种机器人怎么塞进一个 55 维动作向量"，到"VLM 和动作专家怎么共享一套注意力逐层交替算"，再到"两个视觉教师怎么把几何和时序预判蒸进模型"，完整拆解 Robbyant 开源的 6B 参数 VLA 基础模型 LingBot-VLA 2.0 的代码实现、设计动机与训练全流程。

## 系列简介

LingBot-VLA 2.0 是 Robbyant（蚂蚁技术团队）开源的一个 6B 参数视觉-语言-动作（VLA）基础模型（[arXiv 2607.06403](https://arxiv.org/abs/2607.06403)，[GitHub](https://github.com/robbyant/lingbot-vla-v2)）。如果说 [LingBot-VLA 2.0 论文精读](/论文综述/139_LingBotVLA2_从基础到应用的实用VLA) 讲的是"它做了什么、为什么这么做"，那这个系列讲的是"代码到底怎么写的"——每一个组件、每一个张量维度、每一段训练循环都过一遍。

它的整体范式和 GR00T、π₀ 系列一脉相承——用一个视觉语言模型（VLM）理解场景和指令，再用一个 Flow Matching 的动作生成模块把理解转成连续动作——但在工程实现上有几处很有代表性的设计，也是这个系列的重点：

- **VLM + 动作专家双塔逐层交替**：不是"VLM 算完再交给动作头"，而是两个 Transformer **每一层都把各自的 Q/K/V 拼在一起做一次联合注意力**，然后各走各的 FFN。VLM 骨架用 Qwen3-VL-4B，动作专家是一个 36 层的 Qwen2 结构。
- **55 维统一动作表示**：把单臂、双臂、人形、带灵巧手/移动底盘的 20 种机器人，全部映射进同一个 55 维的"槽位向量"，用 robot config 的 feature mapping 做切片映射，用不到的槽位补零。
- **MoE 动作专家**：动作专家的部分层把 FFN 换成 token-level 稀疏 MoE，用 sigmoid 路由 + 共享/路由专家 + 无辅助损失负载均衡（DeepSeek-V3 风格）。
- **双查询蒸馏**：在 VLM 的输入序列里额外插入"当前深度 / 未来深度 / 未来视频"三类可学习查询 token，训练时用 LingBot-Depth（几何教师）和 DINO-Video（因果时序教师）去蒸馏它们，把几何感知和未来预判能力灌进模型。
- **Flow Matching 动作生成**：直线插值 + 速度场回归，推理时欧拉法几步积分，配合 KV-Cache 复用 VLM 的 prefix。
- **务实的训练工程**：三种可切换的 MoE 路由训练模式（loss-free bias hook / sequence-wise 辅助损失 / router z-loss）、可选 Muon 优化器、在线 RunningStats 算归一化统计、按 token 预算的动态 batching、FSDP2 分布式。

**适合读者**：
- 已经了解 VLA 基本范式（比如读过 [GR00T N1.7 系列](/系列/groot_n1d7_deep_dive/) 或 [XR0 系列](/系列/xr0_deep_dive/)），想看一个把"跨具身统一表示 + MoE + 蒸馏"三件事都落到代码的实现
- 想理解"VLM 和动作专家逐层共享注意力"这种双塔结构到底怎么写的工程师
- 想在自己的机器人上后训练 LingBot-VLA 2.0、需要理解 robot config 和数据格式的开发者
- 对 MoE 无辅助损失负载均衡、Flow Matching、知识蒸馏在 VLA 中的落地感兴趣的研究者

**你将获得**：
- 对 LingBot-VLA 2.0 双塔架构（Qwen3-VL + Qwen2 动作专家 + 逐层交替注意力）的代码级理解
- 对 55 维统一动作表示如何通过 robot config 承接 20 种机器人的完整认知
- 对 token-level MoE 动作专家（sigmoid 路由、共享/路由专家、fused kernel、loss-free 均衡）的逐行拆解
- 对双查询蒸馏（三类查询 token 的插入、两个教师、蒸馏 loss 与注意力屏蔽）的完整理解
- 对 Flow Matching 训练目标与欧拉采样、以及 KV-Cache 复用的数学与实现认知
- 对训练循环、多路 loss 组合、三种 MoE 路由训练模式、Muon 优化器、归一化统计计算的实操能力

## 章节目录

| 章节 | 标题 | 简介 |
|------|------|------|
| **第一部分：全局认知与架构总览** | | |
| 01 | [全景图：LingBot-VLA 2.0 在解决什么问题](./01_全景图_LingBotVLA2在解决什么问题) | 三大缺口与三项改进的对应、与 π₀.₅/GR00T 的定位，代码仓库地图 |
| 02 | [双塔架构总览：VLM + 动作专家逐层交替注意力](./02_双塔架构总览_VLM与动作专家逐层交替) | 一图看懂 token 怎么组装、两个 Transformer 怎么共享注意力各走 FFN |
| **第二部分：数据侧的创新** | | |
| 03 | [55 维统一动作表示与 Robot Config 特征映射](./03_统一动作表示与RobotConfig特征映射) | 20 种机器人怎么塞进一个向量、feature mapping 怎么切片、相对动作 |
| 04 | [归一化与数据管线：RunningStats、动态 batching、指令构造](./04_归一化与数据管线) | 三种归一化怎么算、compute_norm_stats、按 token 预算的动态 batching |
| **第三部分：模型核心组件** | | |
| 05 | [MoE 动作专家：token 路由、共享/路由专家与融合 kernel](./05_MoE动作专家_token路由与融合kernel) | Qwen2TokenMoeBlock 逐行拆解、sigmoid 路由、fused_moe、loss-free 偏置 |
| 06 | [双查询蒸馏：三类查询 token、两个教师与蒸馏 loss](./06_双查询蒸馏_查询token与两个教师) | current/future depth + future video 查询怎么插、两个教师怎么蒸、注意力屏蔽 |
| 07 | [Flow Matching 动作生成：训练目标、欧拉采样与 KV-Cache 复用](./07_FlowMatching动作生成与KVCache复用) | x_t 直线插值、速度场回归、推理几步积分、prefix 缓存 |
| **第四部分：训练方法** | | |
| 08 | [训练循环与 MoE 三种路由训练模式、Muon 优化器](./08_训练循环与MoE路由训练模式) | 多路 loss 组合、loss-free vs 辅助损失 vs z-loss、Muon、FSDP2 |

## 前置知识

阅读本系列前建议了解：

- [视觉-语言-动作模型（VLA）综述](/论文综述/S03_视觉语言动作模型VLA综述) — VLA 基本范式
- [稀疏专家混合（MoE）](/前置知识/005n_前置知识_MoE稀疏专家混合) — 第 5、8 章的 MoE 动作专家依赖此
- [知识蒸馏基础](/前置知识/000v_前置知识_知识蒸馏基础) — 第 6 章双查询蒸馏的底座
- [FFN 前馈网络](/前置知识/005m_前置知识_FFN前馈网络与Transformer的两阶段设计) — MoE 替换的模块
- [Muon：正交化动量优化器](/论文综述/106_Muon_正交化动量优化器) — 第 8 章的可选优化器

## 学习建议

- **想快速建立整体认知**：读第 01、02 章即可，能看懂数据怎么从图像/指令流到动作。
- **关注数据侧创新（跨具身、统一表示）**：重点读第 03、04 章。
- **关注模型结构（MoE、蒸馏、Flow）**：重点读第 05、06、07 章，这三章是本系列的技术核心。
- **想自己后训练/复现**：第 03、04、08 章覆盖了 robot config、归一化、训练配置的全部实操细节。

每一章都尽量自包含，跳读某一章不会因为缺上下文而完全看不懂；但第 02 章的架构总览是理解后续所有章节的地图，建议先读。

## 关联阅读

- [LingBot-VLA 2.0 论文精读](/论文综述/139_LingBotVLA2_从基础到应用的实用VLA) — 从论文视角看"做了什么、为什么"，与本系列（代码视角）互补
