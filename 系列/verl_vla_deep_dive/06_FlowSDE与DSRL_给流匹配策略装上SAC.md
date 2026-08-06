---
title: "Flow-SDE 与 DSRL：给流匹配策略装上 SAC"
series:
  id: verl_vla_deep_dive
  chapter: 6
order: 6
---

# 第 06 章 Flow-SDE 与 DSRL：给流匹配策略装上 SAC

> 前情提要：第 05 章讲了 `SupportSACTraining` 契约要求模型实现 `sac_forward_actor` 返回动作和 log_prob。这一章讲清楚 Pi0.5 和 GR00T 这类流匹配（Flow Matching）策略，本身是确定性 ODE 积分，怎么被改造出这个 log_prob，以及另一条完全不同的路线 DSRL 是怎么工作的。

## 知识链接

- 上一章：[模型集成契约：三个接口统一 ACT / Pi0.5 / GR00T](./05_模型集成契约_三个接口统一ACT_Pi0_GR00T)
- 下一章：[环境集成与人机协同：BaseEnv、Recorder、Teleop](./07_环境集成与人机协同_BaseEnv_Recorder_Teleop)
- [系列目录](./index)
- [Flow Matching 与连续归一化流](/前置知识/000g_前置知识_Flow_Matching与连续归一化流) — 理解流匹配采样本身
- [随机插值与 Flow-SDE：给确定性 ODE 注入可算密度的噪声](/前置知识/002i_前置知识_随机插值与Flow_SDE转移分布构造) — **必读**，本章 Flow-SDE 部分的完整数学推导在这里
- [DSRL：在噪声空间对扩散/流策略做强化学习](/前置知识/002t_前置知识_DSRL_扩散策略的噪声空间强化学习) — **必读**，本章 DSRL 部分的完整原理讲解在这里
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)

---

## 1. 问题：流匹配采样没有 log_prob

Pi0.5 和 GR00T N1.6 生成动作的方式都是流匹配：从一个标准高斯噪声出发，经过若干步 ODE 积分（每步用网络预测的速度场 $v_\theta$ 更新一次位置），走到最终动作。这个过程是**确定性**的——同样的起点噪声，同样的网络参数，永远得到同样的动作。

标准 SAC 的最大熵目标（[SAC 前置知识](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic)）需要知道策略在给定状态下选择某个动作的概率密度 $\log\pi(a|s)$，用来计算熵正则项和 importance ratio。确定性映射没有"概率密度"这个概念——你要么必须知道整条 ODE 积分路径的完整 Jacobian（计算量随维度爆炸），要么用别的办法绕开这个问题。

verl-vla 里 Pi0.5 和 GR00T 共用的解法是 **Flow-SDE**（对应前置知识 002i）：把确定性积分改写成一个随机微分方程，让每一步都变成一个可以直接算出 log_prob 的高斯转移。

## 2. Flow-SDE 在代码里的具体实现

`pi0_torch/trainable_model.py` 里，每一步去噪不再是简单的 Euler 外推，而是显式构造一个高斯转移并从中采样：

```python
x0_pred = x_t - v_t * t_cur_exp          # 用当前速度反推出"预测的最终动作"(t=0侧)
x1_pred = x_t + v_t * (1.0 - t_cur_exp)  # 用当前速度反推出"预测的纯噪声"(t=1侧)
if noise_scale > 0:
    sigma = noise_levels * noise_scale * torch.sqrt(t_cur_safe / (1.0 - t_cur_safe))
    x_mean = x0_pred * x0_weight + x1_pred * x1_weight   # 两端预测的加权组合作为均值
    x_prev = x_mean + sigma_t * eps                        # 加噪声采样
    step_log_probs.append(self._gaussian_log_prob(x_prev, x_mean, sigma_t))
```

**这段代码在做什么**：原本 Euler 积分只是"用当前速度往前推一小步"，现在多了一步——先用当前的速度预测反推出这一步"如果不加噪声会走到哪里"（$x_0$ 侧和 $x_1$ 侧的两个预测），把它们加权平均得到一个均值 $x_\text{mean}$，再围绕这个均值加一个标准差为 $\sigma_t$ 的高斯噪声采样出真正的下一步位置。因为知道了均值和标准差，这一步转移的高斯 log_prob 就能直接写出闭式解并算出数值，累加起 $K$ 步的 log_prob 就得到整条生成路径的对数概率。

$\sigma_t$ 的公式（`sqrt(t/(1-t))` 形状）来自布朗桥理论下"两端确定、中间自由、且要保证边缘分布尽量贴近原始确定性 ODE"的约束，完整数学推导在 [Flow-SDE 前置知识](/前置知识/002i_前置知识_随机插值与Flow_SDE转移分布构造#33-flow-sde-实际用的每步标准差公式) 里有——本章不重复推导过程，只讲工程落地。

GR00T 的实现（`gr00t_n1d6/trainable_model.py`）用的是同一套思路（作者注释直接写"Aligns with pi0_torch.PI0TrainableModel"），方向相反（GR00T 是从 $t=0$ 噪声走向 $t=1$ 动作，跟 Pi0.5 相反，但机制完全对称），噪声退火用 cosine schedule（`beta_schedule(step, beta0, beta_min, T)`）而不是 Pi0.5 的 `ScheduledScalar` 类，效果类似——训练早期噪声大鼓励探索，后期噪声小提高稳定性。

GR00T 还额外支持一个 `flow_sde_std_head`：不用固定公式算标准差，而是让一个小网络从 DiT 的隐藏特征直接预测每一步的噪声标准差——把"噪声该有多大"这件事也变成可学习的，而不是完全依赖解析公式。

### 噪声按 task 分层调度、只对部分动作维度探索

Pi0.5 支持 `flow_sde_task_noise_level`：不同任务用不同的噪声强度调度，因为不同任务对探索的容忍度不一样（精细操作任务需要小噪声避免破坏动作精度，粗动作任务可以容忍更大噪声换取更快探索）。

GR00T 支持 `sac_action_train_mask`：只对动作向量的某些维度（比如夹爪开合）加噪声做探索训练，其余维度（比如手臂大关节角）锁定为无噪声版本的确定性输出：

```python
x_train = self._denoise(..., add_noise=True)
x_base = self._denoise(..., add_noise=False)
x_final = torch.where(mask, x_train, x_base)
```

这个设计对应一个实际场景：机器人任务里往往只有部分动作维度（比如"要不要闭合夹爪"这种离散决策倾向的维度）是策略容易出错、需要强化学习纠正的地方，其余维度（连续的空间移动）预训练阶段已经学得足够好，不需要额外冒险探索。

## 3. Critic 挂载点：共享 backbone 输出，独立池化

第 05 章讲过 critic 是外挂的独立参数，这里补充它具体挂在流匹配模型的哪个位置。`sac_forward_state_features` 对 Pi0.5 来说就是跑一次 `self.policy.embed_prefix(...)`（视觉语言 backbone 的前向），拿到的 `prefix_embs`（图像+语言 token 编码序列）同时喂给：

- **actor 侧**：走 Flow-SDE 积分（本章第 2 节），产出动作和 log_prob。
- **critic 侧**：走 cross-attention 池化（第 05 章第 7 节讲过的 `CrossAttentionCriticGroup`），压成固定维表示后接 MLP head 输出 Q 值。

两者共享同一份 `prefix_embs`，不重复跑视觉塔——这是第 05 章"backbone 只算一次"原则在流匹配模型上的具体体现。critic 挂载的位置是**VLM backbone 输出之后、动作专家（action expert）之前**，也就是说 critic 消费的信息跟 actor 一样多（都能看到完整的图像和语言编码），只是各自用不同的方式（积分 vs 池化）把这份信息转成不同的输出（动作 vs 价值）。

## 4. DSRL：完全不同的第二条路线

Flow-SDE 是"改造原策略让它天生带随机性"，DSRL（[前置知识](/前置知识/002t_前置知识_DSRL_扩散策略的噪声空间强化学习)）走的是完全相反的路：**把整个 VLA（backbone + 流匹配动作头）彻底冻结**，只训练一个独立的、小得多的 SAC actor，这个 actor 的"动作"不是机器人动作，而是喂给流匹配采样器的**初始噪声**。

```python
class DSRLNoiseActor(nn.Module):
    def sample(self, features, state, deterministic=False, noise_scale=None):
        mean, log_std = self(features, state)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = normal.rsample()                                    # 重参数化采样
        squashed = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - squashed.pow(2) + _TANH_EPS)  # tanh 修正
        noise_flat = squashed * self.noise_bound
        # 整个动作块共享同一个噪声向量,不是逐步独立采样
        noise = noise_flat.unsqueeze(1).expand(batch_size, self.noise_horizon, self.noise_dim)
        return noise, log_prob.sum(dim=-1)
```

这段代码是标准的 tanh-Gaussian SAC actor（重参数化采样 + tanh 边界修正，[对应前置知识第三节](/前置知识/002t_前置知识_DSRL_扩散策略的噪声空间强化学习#三-actor-的具体形式tanh-gaussian-重参数化)有完整的公式推导），唯一 VLA 特有的地方是最后把噪声向量沿时间维广播——**整个动作块（chunk）内所有时间步共享同一份噪声**，而不是每个时间步独立采样。这是刻意的设计选择：机器人动作块通常需要在时间上保持连贯，如果每个时间步的探索噪声互相独立，解码出来的动作序列会在时间上抖动。

`head init` 用极小的初始化增益（`gain=0.01`），让训练刚开始时 actor 输出接近标准正态噪声（对应第 05 章讲的那条原则——新模块刚接入时不能大幅改变系统原有行为）——这正是流匹配模型训练时假设的噪声先验，保证冷启动阶段不会因为噪声分布偏移太远而破坏预训练策略原本的行为。

`DSRLSteering`（`steering.py`）是挂到 Pi0.5/GR00T 模型上的组合模块，负责在 rollout 时把 `DSRLNoiseActor` 采样出的噪声接入到冻结的流匹配解码流程里，并且 `select_critic_noise` 确保 critic 打分的对象也是这个噪声（而不是最终解码出来的动作）——保证 actor 和 critic 优化的是同一个空间。

DSRL 与 Flow-SDE 在代码里是**显式互斥**的：

```python
if self.flow_sde_enable:
    raise ValueError("DSRL and Flow-SDE cannot both be enabled.")
```

原因很直观：两者都是"给流匹配采样引入随机性"的机制，但作用层级不同——Flow-SDE 是让积分轨迹的每一步都带噪声并逐步累计 log_prob，DSRL 是训练一个独立小网络专门决定起点噪声、中间积分步骤保持确定性。同时启用会让"随机性到底在哪一层"变得混乱，且两者对应的梯度更新目标（更新整个 backbone vs 只更新一个小 actor）也互相冲突。

## 5. 两条路线的选型对比

| 维度 | Flow-SDE | DSRL |
|---|---|---|
| 谁参与梯度更新 | 整个动作头（甚至 backbone）都可能更新 | 完全冻结原策略，只训练一个独立小网络 |
| 训练成本 | 高——每步积分都要跑一次去噪网络前向 | 低——backbone 只需前向一次拿特征，噪声 actor 本身很小 |
| 探索灵活度 | 更高，可以改变整条生成路径 | 受限于"能被冻结解码器映射成有意义动作"的噪声子空间 |
| 是否需要重新训练 backbone | 是（通常） | 否 |
| 适用场景 | 有充足计算资源、希望策略本身持续改进 | 计算/显存受限、只想在现有策略基础上做轻量在线纠偏 |

在 verl-vla 的配置系统里，这两者是模型构造时通过配置项二选一的能力，Trainer 侧（第 08 章的 SAC trainer）的训练循环代码完全不需要关心当前用的是哪一种——因为两者都通过同一套 `SupportSACTraining` 契约（第 05 章）暴露给外部，`sac_forward_actor` 返回动作和 log_prob，训练循环只管调用这个接口。

## 小结

| 概念 | 要点 |
|---|---|
| 流匹配采样的困境 | 确定性 ODE 积分没有天然的概率密度，无法直接套用 SAC |
| Flow-SDE | 把确定性积分改写成带噪声的高斯转移，每步都能算 log_prob，逐步累加得到整条路径的对数概率 |
| 噪声调度细节 | 时间上递减（早探索晚收敛）、可按 task 分层、可只对部分动作维度启用 |
| Critic 挂载点 | backbone 输出之后、动作专家之前，跟 actor 共享同一份编码，不重复跑视觉塔 |
| DSRL | 完全冻结原策略，只在"初始噪声空间"训练一个独立小 SAC actor |
| 两者互斥 | 都是"给流匹配注入随机性"，但作用层级不同，配置上强制二选一 |

## 下章预告

[第 07 章](./07_环境集成与人机协同_BaseEnv_Recorder_Teleop) 转到环境侧——看 `BaseEnv` 怎么用一套统一接口封装 LIBERO 仿真、Isaac Lab Arena 大规模仿真、Piper 真机机械臂之间的巨大差异，动作块（action chunk）是怎么被逐步执行并支持人类实时接管的，以及数据最终是怎么被录制成 LeRobot 数据集的。
