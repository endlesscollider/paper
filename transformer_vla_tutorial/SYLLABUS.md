# 课程大纲：从 Transformer 到 ACT / VLA

## Part A：机器学习与序列建模基础

### 0. 总地图：为什么 Transformer 能连接语言、视觉与动作
- 序列、token、embedding 的统一视角
- 机器人控制为什么也能变成序列预测
- 三条主线：语言 Transformer、ACT、VLA

### 1. 新手最小基础
- 什么是模型、参数、损失函数、训练、推理
- 监督学习与行为克隆
- 向量、矩阵、线性层、softmax
- batch、sequence length、hidden dimension

### 2. Attention 直觉与手算
- “根据当前问题，去查资料，再汇总答案”
- Q/K/V 类比
- scaled dot-product attention
- mask、multi-head attention
- 练习：手算 3 个 token 的 attention

### 3. Transformer 架构
- Token embedding + positional encoding
- Self-attention block
- MLP / FFN
- residual connection 与 layer norm
- Encoder、Decoder、Encoder-Decoder
- ViT：图像 patch 也是 token

## Part B：从序列模型到机器人动作

### 4. 控制问题如何变成 Transformer 问题
- 观测 token：图像、关节状态、语言
- 动作表示：连续动作、离散动作、tokenized action
- 行为克隆：从专家数据学习策略
- 闭环控制与开环动作块

### 5. ACT：Action Chunking Transformer
- 普通行为克隆为什么容易抖动
- 动作块 action chunk 的意义
- temporal ensembling 为什么能让控制更平滑
- CVAE latent 处理多模态示范
- ACT 中 Transformer 编码什么、解码什么
- 双臂任务中 action vector 如何组织

### 6. VLA：Vision-Language-Action 中的 Transformer
- VLA 的统一形式：输入图像与语言，输出动作
- RT-1：机器人数据上的 Transformer policy
- RT-2：把 VLM 知识迁移到动作输出
- OpenVLA：开源 VLA 的典型接口
- Octo：generalist robot policy 的数据与条件控制
- π/GR00T 类路线：VLM backbone + action expert
- 为什么很多新模型不只用“纯 Transformer head”

## Part C：实践与研究判断

### 7. 从零写一个小 Transformer
- NumPy attention
- PyTorch mini Transformer
- toy imitation learning

### 8. 读论文的方法
- 看输入输出，不先看公式
- 画 token 流图
- 找 loss、dataset、action representation
- 复现实验前先复现张量形状

### 9. ACT / VLA 选型指南
- 数据少、固定任务、低成本双臂：ACT 优先
- 跨任务、语言泛化、开放世界：VLA 优先
- 高精细连续控制：考虑 diffusion/flow action head
- 部署约束：频率、延迟、观测同步、安全

### 10. 进阶专题
- Perceiver / PerAct
- Diffusion Policy
- tokenized action vs continuous action head
- hierarchical policy
- sim-to-real 与数据混合
