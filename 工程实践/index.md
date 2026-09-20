# 工程实践

本栏目记录机器人学习工程实践中遇到的具体问题、架构设计和实现细节。

## 文章列表

### ACT 系列

- [ACT Decoder 架构详解](./ACT_Decoder架构详解) — ACT 模型 Decoder 端的完整结构拆解
- [InterACT 与 ACT 的区别解析](./InterACT与ACT的区别解析) — 从直觉到源码的深度对比
- [条件约束的 ACT 模型](./条件约束的ACT模型) — language-conditioned ACT 到结构化条件约束

### 模型对比

- [GR00T 与 π 系列对比 ACT](./GR00T与π系列对比ACT) — VLA 基础模型 vs 专用模仿学习策略
- [从 ACT 到 PerAct2：双臂协调教程](./从ACT到PerAct2_双臂协调教程) — 为什么双臂需要显式 Coordination

### RL 后训练

- [RLinf BC 到 RL 的 ACT 后训练架构](./RLinf_BC到RL的ACT后训练架构) — 从 BC 预训练到 PPO 微调的完整架构、Loss 设计与工程实践
- [GR00T N1.7 四种 RL 方案全景对比](./GR00T_N1d7_四种RL方案全景对比) — PPO / QC / SAC Flow-G / ConRFT 的核心差异、选型与兼容性
- [GR00T N1.7 Chunk-SAC 四种 Actor 目标详解](./GR00T_N1d7_ChunkSAC四种Actor目标详解) — SAC Flow-G 内部的 awr_flow / direct_q / sac_flow_g / awr 拆解
- [RECAP 工程实践：训练流程、数据管道与 Loss 实现](./RECAP_训练流程与Loss工程实现) — π0.6\* 的完整迭代流程、价值函数/策略训练伪代码、数据配比与超参数
- [SAC_FLOW_G 完全解剖：GR00T VLA 在线强化学习链路的工程实现](./SAC_FLOW_G完全解剖_GR00T_VLA在线强化学习链路的工程实现) — Critic 离线预热、Flow-G 门控 Actor、熵温度自适应五组件拆解

### 双臂操作

- [双臂任务训练方法研究](./双臂任务训练方法研究) — 双臂协作模型训练的方法论
- [双臂动作扰动与数据增强调研](./双臂动作扰动与数据增强调研) — action/trajectory 扰动提升闭环成功率

### 仿真与资产转换

- [URDF 转 USD 完整工程流程详解](./URDF转USD完整工程流程详解) — Mesh 缩放、URDF 修复、格式转换、材质绑定、物理参数、传感器注入全流程
- [Isaac Lab 仿真对齐与参数调优：从相机内参到关节 PD 增益](./IsaacLab仿真对齐与参数调优) — 相机镜头几何标定接入、关节 PD 增益系统辨识、物理材质对齐、域随机化范围设定
