# 前置知识

本系列整理机器人策略学习所需的核心前置概念。建议按顺序阅读，后续论文和综述会反复引用这些内容。

## 目录

- [策略梯度与 PPO](./000a_前置知识_策略梯度与PPO) — REINFORCE → PPO 的完整推导
- [扩散模型 DDPM](./000b_前置知识_扩散模型DDPM) — 前向/逆向过程、噪声调度
- [Diffusion Policy](./000c_前置知识_Diffusion_Policy) — 把扩散模型当策略用
- [行为克隆与 RL 微调范式](./000d_前置知识_行为克隆与RL微调范式) — 为什么先 BC 再 RL
- [对数似然与变分下界](./000e_前置知识_对数似然与变分下界) — ELBO 推导
- [为什么扩散策略难以 RL 微调](./000f_前置知识_为什么扩散策略难以RL微调) — DPPO 的动机
- [Flow Matching 与连续归一化流](./000g_前置知识_Flow_Matching与连续归一化流) — 更快的生成范式
- [Consistency Model 与一步生成](./000h_前置知识_Consistency_Model与一步生成) — 极致加速推理
- [动作平滑性正则化 CAPS / Grad-CAPS](./000i_前置知识_动作平滑性正则化CAPS) — 抑制策略输出的高频振荡
- [KL 散度与策略约束](./000j_前置知识_KL散度与策略约束) — 防止 RL 微调偏离太远
- [SAC (Soft Actor-Critic)](./000k_前置知识_SAC_Soft_Actor_Critic) — 最大化熵的 off-policy 方法
- [动作 Token 化与自回归策略](./000l_前置知识_动作Token化与自回归策略) — 自回归 VLA 的动作表示核心
- [GRPO (Group Relative Policy Optimization)](./000m_前置知识_GRPO_Group_Relative_Policy_Optimization) — 无 Critic 的组内比较策略优化
- [Process Reward Model](./000n_前置知识_Process_Reward_Model) — 为每步提供进度信号的过程奖励模型
- [Q 函数与 Value 函数](./000o_前置知识_Q函数与Value函数) — RL 中"好坏"的数学定义
- [DDPG（确定性策略梯度）](./000p_前置知识_DDPG_确定性策略梯度) — 连续动作空间 Deep RL 的开山之作
- [TD3（Twin Delayed DDPG）](./000q_前置知识_TD3) — 双 Q 取最小值 + 延迟更新
- [Replay Buffer（经验回放）](./000r_前置知识_Replay_Buffer_经验回放) — Off-Policy 算法的数据复用核心
- [数据并行与 AllReduce 基础](./001h_前置知识_数据并行与AllReduce基础) — 多卡训练的地基：梯度怎么对齐
- [FSDP：全分片数据并行](./001i_前置知识_FSDP全分片数据并行) — 把参数/梯度/优化器状态也切开存
- [张量并行与流水线并行：Megatron 核心思想](./001j_前置知识_张量并行与流水线并行_Megatron核心思想) — 切开层内计算 / 切开不同层
- [TD 学习与 n 步回报的偏差问题](./001k_前置知识_TD学习与n步回报的偏差问题) — n 步回报为什么在 off-policy 数据上有偏，以及 Q-chunking 如何解决
- [行为约束策略优化](./001l_前置知识_行为约束策略优化) — "约束"具体怎么通过训练损失起作用，以及策略表达能力如何决定约束的效果
- [群表示论与不可约表示](./001m_前置知识_群表示论与不可约表示) — 群、表示、不可约表示、Clebsch-Gordan 系数：对称性的数学语言
- [等变神经网络与不变性、等变性](./001n_前置知识_等变神经网络与不变性等变性) — 让对称性"长在"网络结构里，而不是靠训练数据"学出来"
- [Behler-Parrinello 局部能量框架与近视性原理](./001o_前置知识_Behler_Parrinello局部能量框架与近视性原理) — 为什么机器学习力场可以随系统尺寸线性扩展
- [FQL：Flow Q-Learning](./001p_前置知识_FQL_Flow_Q_Learning) — 教师-学生分工：多步 Flow 网络负责表达力，单步网络负责速度，靠蒸馏连接两者
