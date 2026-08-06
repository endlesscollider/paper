---
title: "GR00T N1.7 × RLinf 的 ConRFT 全链路解析：从离线 Cal-QL 到在线异步 actor-learner"
order: 304
tags: [强化学习, GR00T, ConRFT, RLinf, Cal-QL, CQL, Flow Matching, 异步训练, 离线到在线, 工程实践, 系列]
category: 系列
star: 5
series:
  id: groot_conrft_rlinf_deep_dive
  totalChapters: 14
  dir: /系列/groot_conrft_rlinf_deep_dive
---

# GR00T N1.7 × RLinf 的 ConRFT 全链路解析

> 一条真实在跑的 VLA 后训练链路，两个阶段、两个入口、两套目标函数。这个系列把它从 `ssh` 那一行命令开始，一路拆到 `torch.logsumexp` 那一行代码。

## 系列简介

这个系列讲的是一件很具体的事：**怎么把 ConRFT 这个论文方法，落到 GR00T N1.7 这个 Flow Matching VLA 模型上，跑在 RLinf 这个分布式 RL 框架里。**

链路的形状是这样的：

```text
起点   GR00T N1.7 BC checkpoint-500000（多任务双臂，62 维动作，16 步 chunk）
任务   open_laptop（打开笔记本电脑），来自 embodied-arena 的 Isaac Sim 仿真
数据   一批 BC 演示（open_laptop_ego2_ok）

阶段一 离线 Cal-QL critic warmup      同步、无环境、800 次更新、8 卡 H20 FSDP
       └─ 入口 train_embodied_agent.py + miarena_r1_conrft_critic_warmup_gr00t_n1d7

阶段二 在线异步 actor-learner          64 个 4090 环境边采边训、50 步 × 35 次更新
       └─ 入口 train_async.py + miarena_r1_conrft_gr00t_n1d7
```

它和这个知识库里已有的另一条链路是**兄弟关系**，共享同一套底层设施：

- [LeRobot-VLARL 工程实现](/系列/lerobot_vlarl_deep_dive/) — ConRFT 的另一个实现（LeRobot 侧），本链路的目标函数就是对齐它

**本系列的独特之处**在于它讲的是"同一个 Chunk-SAC 骨架上，换成 ConRFT 的目标函数会发生什么"：

- Critic 侧多了一个 **Cal-QL 保守项**（12 个候选动作 + MC 回报下界），而且只在离线阶段开
- Actor 侧变成 **BC + 0.01·Q** 的两项（离线）和 **BC + 0.01·Q + 1.0·W2** 的三项（在线）
- same-state pair TD 被**明确禁用**——这是配置里的硬门禁，不是"没配"
- 阶段二用的是 RLinf 的**异步** runner，环境采集不等 learner 更新

## 这个系列会回答的问题

1. 为什么 ConRFT 必须分成两个**独立进程、独立配置、独立 run 目录**的阶段，而不能在一个脚本里跑完
2. 一个 Hydra 配置为什么要继承五层，每一层各自负责什么，最终生效的值是哪个
3. GR00T 的 Flow Matching 去噪链怎么变成一个可以求梯度的随机策略 $\pi(a|s)$，`log_pi` 到底是什么
4. `bc_reference_actions` 这个"冻结 BC 参考动作"是怎么在**同一次前向**里用同一份初始噪声跑出来的，为什么这是 W2 项的关键
5. Cal-QL 的 12 个候选动作分别从哪来，为什么要包含 next-state 的策略采样
6. `cql_scale` 那个 `if` 分支到底在防什么（以及为什么在本配置下它恰好等于 1）
7. 在线阶段 `bc : q : w2 = 1 : 0.01 : 1` 这组权重意味着什么，成功率不涨时该动哪一个
8. 异步 runner 里"环境采集"、"learner 更新"、"权重发布"三条时间线是怎么错开的
9. 从阶段一的 checkpoint resume 到阶段二时，`update_step` 为什么必须是 800 而不是 0
10. TensorBoard 里 `conrft/*` 那一堆指标各自在监控什么，哪几个是"训练已经坏了"的早期信号

## 📌 先读第 01 章第 9 节

第 01 章第 9 节记录了对这条链路做静态审查时发现的**全部问题**，一共 14 条，按 P0/P1/P2 分级，每条都带文件、符号、具体数值构成的证据链。

**其中绝大多数已经在 2026-07-29 修掉了**（[修复状态总表](./01_全链路总览#9.15-修复状态总表)），但这一节仍然是整个系列最该先读的部分，因为它回答的是"每个配置项为什么是现在这个值"。举几个例子：

| 曾经的问题 | 现在的配置 | 不知道来龙去脉会怎样 |
|------------|------------|----------------------|
| BC 项走 velocity loss，梯度恒为 0 | BC 项改成动作空间 masked MSE | 会觉得"和论文不一致"而改回去 |
| resume 时 Flow-G gate 被清成恒等 | `preserve_stage1_flow_g_adapter: true` | 会把它当成冗余配置删掉 |
| `critic_warmup_updates` 与 `update_step` 耦合 | 保持 `800` 不变，另加硬门禁 | 会"顺手"把它设成 0，直接让阶段交接失效 |
| critic clamp 造成 Q 梯度死区 | `straight_through_action_clip: true` | 会以为它只是个数值技巧 |

还有两条是**设计取舍而非缺陷**，需要长期监控：`terminal_success` 下 Cal-QL 下界的有效性（看 `calql_positive_bound_fraction`），以及 ConRFT 的 actor 更新次数是 Flow-G16 的 16 倍（做跨方法对比时必须说明）。

## 章节目录

| 章节 | 标题 | 一句话简介 |
|------|------|------------|
| 01 | [全链路总览](./01_全链路总览) | 两个阶段的完整数据流图、ConRFT 在 Chunk-SAC 家族里的位置，以及**全部已识别问题的清单与修复状态**（14 条，每条带证据链 + 修法） |
| 02 | [从 ConRFT 论文到本链路](./02_论文到实现的映射) | 一致性策略 → Flow Matching、单步动作 → 16 步 chunk、人类干预 → 仿真自动 reset |
| 03 | [启动层：三层脚本与前置门禁](./03_启动层与门禁) | `conrft.env` / `run_remote.sh` / `run_miarena_groot_chunk_sac.sh` 各管什么 |
| 04 | [配置继承链与硬门禁](./04_配置继承链) | 五层 Hydra defaults 的合并结果，以及 `config.py` 里的 8 条断言 |
| 05 | [模型层：三个 ForwardType](./05_模型层三个ForwardType) | `SAC` / `SAC_Q` / `SFT`，以及双轨去噪产出 BC 参考动作 |
| 06 | [Critic 架构与 chunk 级 TD target](./06_Critic架构与TD目标) | `flat_absolute` twin-Q、$\gamma^{16}$ bootstrap、target 网络 EMA |
| 07 | [阶段一数据侧：BC replay 与 MC 回报](./07_阶段一数据侧) | `terminal_success` 奖励怎么定义、$G_t$ 怎么算、为什么必须按 episode 存 |
| 08 | [阶段一 Critic：Cal-QL 保守项](./08_阶段一CalQL保守项) | 12 个候选的完整来源、logsumexp 归一化、`cql_scale` 的真相 |
| 09 | [阶段一 Actor：BC 加轻量 Q](./09_阶段一Actor目标) | 为什么离线阶段 actor 训在 expert batch 上，W2 项为什么是 0 |
| 10 | [阶段二异步 actor-learner 架构](./10_阶段二异步架构) | Channel 拓扑、后台 drain 线程、50 秒权重发布、跨机 placement |
| 11 | [阶段二在线目标：三项博弈](./11_阶段二在线三项目标) | BC 项、Q 项、W2 项各自往哪拉，`1 : 0.01 : 1` 的后果 |
| 12 | [Checkpoint、resume 与阶段交接](./12_Checkpoint与阶段交接) | `conrft_components`、`resume_config_hash`、`update_step=800` 的语义 |
| 13 | [指标手册](./13_指标手册) | `conrft/*` 与 `chunk_sac/*` 全部关键指标的读法与告警阈值 |
| 14 | [风险清单与调参优先级](./14_风险清单与调参) | 已识别的设计风险、各自的证据、以及按优先级排序的整改建议 |

## 前置知识

这条链路踩在很多概念上。**强烈建议先读完这三篇**，否则第 06、08 章会很吃力：

- [SAC Soft Actor-Critic](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — 整个 critic/actor 交替更新的骨架
- [CQL 保守 Q 学习](/前置知识/002g_前置知识_CQL保守Q学习) — 第 08 章的保守项就是它
- [Cal-QL 校准保守 Q 学习](/前置知识/002h_前置知识_CalQL校准保守Q学习) — MC 回报下界的原理

按需查阅：

- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — GR00T 动作头的生成范式
- [随机微分方程 SDE 与扩散模型的联系](/前置知识/001c_前置知识_随机微分方程SDE直觉与扩散模型的联系) — 第 05 章 `log_pi` 的来源
- [重参数化技巧](/前置知识/002e_前置知识_重参数化技巧) — 为什么动作可以对策略参数求梯度
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数)
- [TD 学习与 n 步回报的偏差问题](/前置知识/001k_前置知识_TD学习与n步回报的偏差问题)
- [Replay Buffer 经验回放](/前置知识/000r_前置知识_Replay_Buffer_经验回放)
- [FSDP 全分片数据并行](/前置知识/001i_前置知识_FSDP全分片数据并行) — 8 卡训练的分片方式
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式)

相关论文与系列：

- [ConRFT 论文精读](/论文综述/010_ConRFT_一致性策略RL微调VLA) — 方法的出处
- [Q-Chunking：RL 与动作分块](/论文综述/071_QChunking_RL与动作分块) — chunk 级 MDP 的理论来源
- [RLPD：高效在线 RL 利用离线数据](/论文综述/075_RLPD_高效在线RL利用离线数据) — 另一条离线到在线路线
- [GR00T N1.7 深度解析](/系列/groot_n1d7_deep_dive/) — 模型本身
- [RLinf 深度解析](/系列/rlinf_deep_dive/) — 框架本身

## 阅读建议

**如果你要决定这次实验能不能开**：只读 [第 01 章第 9 节](./01_全链路总览#9.-问题-风险与实验安排隐患清单)，10 分钟。

**如果你只想知道这条链路在干什么**：读 01 → 02 → 11，大约 30 分钟。

**如果你要接手运维这条链路**：读 01 → 03 → 04 → 12 → 13 → 14。这几章覆盖了"怎么启动、配置从哪来、checkpoint 怎么交接、指标怎么看、哪里容易坏"。

**如果你要改算法**：全读，重点是 05（模型接口契约）→ 06（critic）→ 08（Cal-QL）→ 09/11（两阶段 actor 目标）。改任何一处之前先看 04 章的门禁清单，很多"看起来能改"的参数其实会被断言拦住。

**如果你在做类似的工作但用别的框架**：02 章的映射表和 14 章的风险清单最有价值——它们是"把一个论文方法落到真实链路上"这件事本身的经验。

## 贯穿全系列的例子

全系列使用同一个任务和同一组数字，方便前后对照。

```text
任务         open_laptop（打开笔记本电脑）
配置         config/dexbench/humanoid/eval/open_laptop.yaml
指令         "open the laptop"
动作         62 维（双臂 + 双手）
动作块       16 步，replan 周期也是 16（GR00T N1.7 原生 chunk）
episode      最长 350 个物理步
奖励         terminal_success —— 仅成功 episode 的最后一步给 +1，其余全 0
γ            0.99
阶段一       800 次离线更新，每 100 次存一个 checkpoint
阶段二       50 个 runner step × 35 次更新 = 1750 次，64 个训练环境，每 5 step 评测一次
BC 数据      open_laptop_ego2_ok
基座         GR00T N1.7 checkpoint-500000 + Cosmos-Reason2-2B backbone
硬件         H20 head 8 卡（rollout 0-3 / actor 4-7）+ RTX 4090 节点 4 卡（env）
```

每当出现一个公式，我们就把这组数字代进去算一遍。比如 $\gamma^{16} = 0.99^{16} = 0.851$ 这个数会在第 06 章反复出现；$\tau\log 13 = 2.565$ 这个数会在第 08 章反复出现。

---

开始读：[第 01 章 全链路总览](./01_全链路总览)
