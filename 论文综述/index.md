# 论文综述

本系列对深度学习、强化学习与机器人学习领域的核心方向进行系统性综述。每篇综述覆盖该方向的发展脉络、代表方法对比与未来趋势。

## 综述文章

- [深度强化学习方法综述](./S01_深度强化学习方法综述) — 从 DQN 到 PPO/SAC 的算法全景
- [机器人模仿学习综述](./S02_机器人模仿学习综述) — 从行为克隆到扩散策略的范式演进
- [视觉-语言-动作模型 VLA 综述](./S03_视觉语言动作模型VLA综述) — RT-2/Octo/π₀ 大模型操控路线
- [Sim-to-Real 迁移综述](./S04_Sim_to_Real迁移综述) — 域随机化、系统辨识、对抗训练
- [扩散模型在决策与控制中的应用综述](./S05_扩散模型在决策与控制中的应用综述) — Diffuser/Diffusion Policy/DPPO

## 论文精读

以下是对单篇论文的逐段深度解读：

### 扩散策略 + RL

- [DPPO：扩散策略策略优化](./001_DPPO_扩散策略策略优化) — NeurIPS 2024，PPO 微调扩散策略
- [AGILE：人形机器人 RL 工作流](./002_AGILE_人形机器人RL工作流) — 人形全身操作的完整 pipeline
- [Online DPRL 综述](./003_Online_DPRL_综述_扩散策略与在线RL) — 把所有方法放到统一框架对比
- [D²PPO：解决表示坍塌](./004_D2PPO_解决表示坍塌) — 防止 DPPO 长期微调的退化
- [IDQL：隐式扩散 Q 学习](./005_IDQL_隐式扩散Q学习) — Off-policy 路线的代表

### RL + VLA（自回归/通用 VLA 的 RL 后训练）

- [VLA-RL：PPO 直接训练自回归 VLA](./006_VLA_RL_PPO直接训练自回归VLA) — PPO + Process Reward Model 训练 7B VLA
- [RIPT-VLA：无 Critic 的 VLA 交互式后训练](./007_RIPT_VLA_无Critic的VLA后训练) — RLOO + Dynamic Rejection，1-shot → 97%
- [RLDG：RL 专家蒸馏到通用 VLA](./008_RLDG_RL专家蒸馏到VLA) — 小模型跑 RL，大模型做泛化
- [What Can RL Bring to VLA Generalization?](./009_What_Can_RL_Bring_VLA泛化实证研究) — NeurIPS 2025 实证研究，PPO > GRPO > DPO
- [ConRFT：一致性策略 RL 微调 VLA](./010_ConRFT_一致性策略RL微调VLA) — 真实机器人 RL，离线+在线两阶段
- [SRPO：自参考策略优化](./011_SRPO_自参考策略优化) — V-JEPA 2 隐空间做 progress reward，99.2% LIBERO
- [SimpleVLA-RL：可扩展 VLA RL 训练](./012_SimpleVLA_RL_可扩展VLA_RL训练) — 基于 veRL 框架，发现 pushcut 新行为
- [BootRL：冻结 VLA + 轻量 RL Head](./013_BootRL_冻结VLA加RL_Head) — 100M RL head，泛化零损失
- [RobustVLA：鲁棒性感知 RL 后训练](./014_RobustVLA_鲁棒性感知RL后训练) — CEM 对抗环境搜索，最坏情况优化
- [PLD：Residual RL 自改进 VLA](./015_PLD_Residual_RL自改进VLA) — Probe-Learn-Distill 迭代循环
- [RECAP：从真实部署经验中 RL 学习](./016_RECAP_从真实部署经验中RL学习) — Advantage Conditioning，零架构修改
- [VLA-RFT：世界模型验证奖励 RL 微调](./017_VLA_RFT_世界模型验证奖励RL微调) — 视频预测做仿真器，400 步即超越 SFT
- [FlowRL：Flow VLA 的在线 RL 微调](./018_FlowRL_Flow_VLA的在线RL微调) — 解决 Flow Matching 的 log-prob 不可算问题
- [TGRPO：轨迹级 GRPO 微调 VLA](./019_TGRPO_轨迹级GRPO微调VLA) — 里程碑密集奖励 + 课程学习
- [GRAPE：偏好对齐 VLA 泛化](./020_GRAPE_偏好对齐VLA泛化) — DPO + GPT-4V 生成 cost function，泛化最强

### 机器人数据与预训练模型

- [Open X-Embodiment：大规模跨体机器人数据集与 RT-X 模型](./011_OpenX_大规模跨体机器人数据集与RTX模型) — ICRA 2024，百万级跨体态数据集
- [Octo：开源通用机器人策略](./012_Octo_开源通用机器人策略) — CoRL 2024，800k 轨迹预训练的开源策略
- [DROID：大规模真实世界操作数据集](./013_DROID_大规模真实世界操作数据集) — RSS 2024，76k 轨迹 564 场景的极致多样性
- [π₀：通用机器人基础模型](./014_Pi0_通用机器人基础模型) — 2024，VLM + Flow Matching 的工业级方案
- [OpenVLA：开源视觉-语言-动作模型](./015_OpenVLA_开源视觉语言动作模型) — CoRL 2024，7B 开源 VLA 标杆
- [HPT：异构预训练 Transformer](./016_HPT_异构预训练Transformer) — NeurIPS 2024，跨体态策略的 scaling law
- [CrossFormer：跨体通用策略](./017_CrossFormer_跨体通用策略) — CoRL 2024，一个策略控制操作+导航+运动+飞行
- [RT-2：视觉-语言-动作模型](./018_RT2_视觉语言动作模型) — CoRL 2023，定义 VLA 范式的里程碑
- [GR00T N1：人形机器人基础模型](./019_GR00T_N1_人形机器人基础模型) — 2025，NVIDIA 双系统人形基础模型
- [LeRobot：开源端到端机器人学习库](./020_LeRobot_开源端到端机器人学习库) — 2025，Hugging Face 的全链路开源库
