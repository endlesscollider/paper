# 专题笔记

针对具体技术方向的深度研究笔记，覆盖 ACT 架构变体、双臂协调、RL 微调工程实践等主题。

## ACT 系列

- [ACT Decoder 架构详解](/工程实践/ACT_Decoder架构详解) — CVAE Encoder + Decoder 的完整拆解
- [InterACT 与 ACT 的区别解析](/工程实践/InterACT与ACT的区别解析) — interaction residual 分支的设计与实现
- [条件约束的 ACT 模型](/工程实践/条件约束的ACT模型) — phase / arm role / frame 等结构化条件注入
- [GR00T 与 π 系列对比 ACT](/工程实践/GR00T与π系列对比ACT) — 基础模型路线与 ACT 的工程对比

## 双臂协调

- [从 ACT 到 PerAct2：双臂协调教程](/工程实践/从ACT到PerAct2_双臂协调教程) — 为什么双臂需要显式 Coordination
- [双臂任务训练方法研究](/工程实践/双臂任务训练方法研究) — 从 naive 拼接到耦合约束建模
- [双臂动作扰动与数据增强调研](/工程实践/双臂动作扰动与数据增强调研) — action/trajectory 扰动提升闭环成功率

## MiGenRL RL 微调

- [MiGenRL RL 训练流程](/工程实践/MiGenRL_RL训练流程) — 统一 RL 入口与 RLInf 架构
- [MiGenRL RLPD 专家回放实现](/工程实践/MiGenRL_RLPD专家回放实现) — RLPD 第一版落地实现
- [MiGenRL RL 微调实现深度剖析](/工程实践/MiGenRL_RL微调实现深度剖析) — 网络架构、权重更新、分布式后端
