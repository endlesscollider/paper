---
title: "SAC 训练循环：EpisodeBuffer、ReplayPool 与 RLPD"
series:
  id: verl_vla_deep_dive
  chapter: 8
order: 8
---

# 第 08 章 SAC 训练循环：EpisodeBuffer、ReplayPool 与 RLPD

> 前情提要：第 07 章讲了环境怎么产出带 `next.reward`/`next.terminated` 等字段的原始交互数据。这一章讲这些碎片化的原始数据怎么被收集成完整 episode、存进 replay pool、最终喂给 critic/actor 更新——也就是 SAC trainer 完整的 off-policy 训练循环。

## 知识链接

- 上一章：[环境集成与人机协同：BaseEnv、Recorder、Teleop](./07_环境集成与人机协同_BaseEnv_Recorder_Teleop)
- 下一章：[RECAP 工作流：六阶段自我提升闭环](./09_RECAP工作流_六阶段自我提升闭环)
- [系列目录](./index)
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — **必读**，本章 Bellman target 公式的理论基础
- [Replay Buffer 经验回放](/前置知识/000r_前置知识_Replay_Buffer_经验回放) — **必读**，理解 replay pool 存在的意义
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数)
- [TD3](/前置知识/000q_前置知识_TD3) — target 网络软更新与延迟策略更新的来源
- [CQL 保守 Q 学习](/前置知识/002g_前置知识_CQL保守Q学习) — critic loss 里可选的 CQL 正则项

---

## 1. 从"一批碎片步"到"一条完整 episode"：EpisodeBuffer

第 03 章讲过 `EnvLoop` 每次 rollout 只跑固定的 `max_interactions` 步（一个"窗口"）。一条真实的机器人 episode 可能跨越好几个窗口才结束，也可能一个窗口内包含好几条完整 episode（如果 episode 很短）。SAC 训练需要的是"完整的一条episode"（这样才能正确判断哪一步是终止步、reward 该怎么累计），所以需要一层缓冲把碎片重新拼接：

```python
class EpisodeBuffer:
    def ingest(self, rollout: DataProto) -> list[DataProto]:
        done = terminated | truncated                 # [num_lanes, num_steps]
        episodes = []
        for lane in range(num_lanes):
            buffer = self._lanes[lane]
            for step in range(num_steps):
                buffer.append(self._select_step(rollout, lane, step))
                if done[lane, step]:
                    if len(buffer) > 1:
                        episodes.append(self._concat_steps(buffer))
                    buffer.clear()
        return episodes
```

**这段代码在做什么**：`lane` 指一个并行环境实例的时间序列。每来一批新的 rollout 窗口数据，就把每个 lane 里逐步产生的新数据追加到该 lane 对应的缓冲区里，一旦某一步出现 `done`（终止或截断），就把该缓冲区里累积的全部步骤拼成一条完整 episode 吐出来，然后清空缓冲区继续等下一条。`auto_reset=True` 时（环境自动重置继续跑），未结束的步会跨多次 rollout 调用持续累积在 `self._lanes` 里，直到真正遇到 done；`auto_reset=False` 时要求每个窗口内每个 lane 都必须出现一次 done，否则直接报错——这是一种"契约检查"，防止配置错误导致某些 episode 永远无法被正确切分。

## 2. 完整 episode → SAC 需要的 transition 格式

拿到一条完整 episode 之后，`prepare_sac_actor_input` 把它转成 SAC 训练需要的 `t0`（当前时刻）/`t1`（下一时刻）transition 对：

```python
def prepare_sac_actor_input(episode, *, trainer_config, global_steps):
    done_substeps = terminated_substeps | truncated_substeps
    valid_reward_substeps = (done_substeps.cumsum(dim=2) - done_substeps.long()) == 0
    reward_steps = (reward_substeps * valid_reward_substeps).sum(dim=2)
    ...
    episode.batch["info.rewards"] = (reward_steps - float(trainer_config.step_penalty)) * valid_mask.float()
    return flatten_trajectories(add_transition_prefixes(episode, transition_boundary_mask=done_steps))
```

**这段代码在做什么**：一个"训练步"（step）在动作块视角下实际包含若干个"子步"（substep，块内的每一个物理执行步），`valid_reward_substeps` 保证只统计"这个训练步的动作块真正执行完之前"产生的奖励，块内提前终止之后的子步奖励不会被错误地计入。`step_penalty` 是一个固定的每步惩罚，鼓励策略尽快完成任务（拖得越久扣得越多）。最后 `add_transition_prefixes` 把每个时间步的观测/动作打上 `t0.*`（当前）和 `t1.*`（下一时刻，用错位一步获得）前缀，`flatten_trajectories` 把 `[批次, 时间]` 的二维结构展平成一维的独立 transition 集合——这一步之后，每一行数据就是标准的 $(s, a, r, s', \text{done})$ 五元组，可以直接喂进任何 off-policy 算法。

## 3. Off-policy 训练循环的核心节奏：一次采集，多次复用

SAC trainer 的主循环把训练步划分成若干个"rollout window"，每个 window 内只在前几步做真正的环境交互，剩下的步全部复用已经存在 replay pool 里的历史数据训练：

```python
for rollout_window in range(total_rollout_windows):
    for training_step in range(rollout_interval):
        need_rollout = (training_step < rollout_times) or self.global_steps < warm_rollout_steps
        if warm_rollout_steps <= self.global_steps < actor_config.critic.warmup_steps:
            need_rollout = False    # critic 预热阶段完全不采集,纯粹从 replay pool 训练
        if need_rollout:
            rollout_output, _, rollout_metrics = self.cluster.rollout(...)
            actor_input = self._prepare_actor_input(rollout_output)   # EpisodeBuffer + prepare_sac_actor_input
        actor_output = self.cluster.train(actor_input or empty_batch, ...)
```

**这就是 SAC 作为 off-policy 算法的样本效率来源**：每个 window 只花 `rollout_times` 次去真实环境采集新数据（代价高，因为要跑物理仿真甚至真机），剩余的 `rollout_interval - rollout_times` 次训练全部从已经攒起来的 replay pool 里反复采样——同一批经验数据可以被使用很多次，不需要每次梯度更新都对应一次新的环境交互。`critic.warmup_steps` 阶段甚至完全暂停采集，只用现有数据训练 critic 到一个稳定水平，之后才开始交替采集和训练。

## 4. Critic loss：完整推导

SAC 的 critic 通过最小化 TD error 来学习 $Q$ 值，`_calculate_critic_loss`（`workers/engine/sac/training_worker.py`）实现了标准的 soft Bellman target：

**Step 1：这个公式在做什么**

$$
y = r + \gamma (1 - d) \big(Q^{\text{target}}(s', a') - \alpha \log\pi(a'|s')\big)
$$

这个公式算出的是"critic 应该被训练成输出的目标值"。它回答的问题是：给定当前这一步的即时奖励 $r$，以及"下一个状态下，按照当前策略继续走下去，长期能拿到多少期望回报"（用 target 网络估计），critic 对 $(s,a)$ 的估值应该等于两者之和。

> **一句话直觉**：这一步拿到的真实奖励，加上"接下来按当前策略继续走，target 网络觉得能拿多少分——但要扣掉一点因为随机探索而'应该打折'的分数"。

**逐符号拆解**：

| 符号 | 数学含义 | 具体是什么 | 典型值/维度 |
|---|---|---|---|
| $y$ | Bellman target，critic 训练的回归目标 | 每个样本一个标量（多个 critic head 时会 broadcast） | `[B]` |
| $r$ | 即时奖励 | `info.rewards`，环境给的 reward 减去 step_penalty | 标量 |
| $\gamma$ | 折扣因子 | `actor_config.critic.gamma`，典型值 0.99 | 常数 |
| $d$ | 是否终止 | `info.terminateds`，1 表示这条 episode 到这一步已经终止 | 0 或 1 |
| $Q^{\text{target}}(s',a')$ | target 网络对下一状态-动作的估值 | 由 target critic 网络前向算出，`use_target_network=True` | 标量 |
| $a'$ | 下一状态下重新采样的动作 | 用当前（在线）actor 在 $s'$ 上采样得到 | 与动作空间同维 |
| $\alpha$ | 熵温度系数 | 固定值或自动调节的可学习标量 | 典型 0.01~1.0 |
| $\log\pi(a'|s')$ | 策略在 $a'$ 上的 log 概率 | actor 采样时同时输出 | 标量 |

**梯度方向**：如果当前 critic 对 $(s,a)$ 的估计 $Q_\theta(s,a)$ 小于 target $y$，MSE 梯度会把 $Q_\theta(s,a)$ 往上推；反之往下压。这正是 TD learning 的标准更新方向——让当前估值逐步逼近"用一步真实奖励加上下一步估值"算出来的目标。

**为什么要减 $\alpha\log\pi(a'|s')$**：这是 SAC 相对标准 Q-learning 的核心改动——最大熵目标不仅要最大化累积奖励，还要最大化策略的熵（鼓励策略保持一定随机性、避免过早收敛到局部最优）。$-\log\pi(a'|s')$ 就是熵的定义本身（概率越低的动作，$-\log\pi$ 越大），加进 target 里意味着"如果 target 网络认为下一步应该选一个'不那么确定'的动作分布，就要在估值上多给一点鼓励"。

**数值代入**：取 $r=0.3$（这一步拿到不错的奖励），$\gamma=0.99$，$d=0$（还没终止），$Q^{\text{target}}(s',a')=8.5$，$\alpha=0.1$，$\log\pi(a'|s')=-2.0$（对应一个概率密度不算太高的探索动作）：

$$
y = 0.3 + 0.99 \times (1-0) \times \big(8.5 - 0.1 \times (-2.0)\big) = 0.3 + 0.99 \times (8.5 + 0.2) = 0.3 + 0.99 \times 8.7 = 0.3 + 8.613 = 8.913
$$

代码里对应实现（配合上一行的 `q_target` 是 `use_target_network=True, method="min"` 算出的双 Q 取小值）：

```python
y = (rewards + gamma * (1.0 - terminateds) * q_target if next_log_prob is None
     else rewards + gamma * (1.0 - terminateds) * (q_target - alpha * next_log_prob))
mse = F.mse_loss(q_predict, y.unsqueeze(1).expand_as(q_predict), reduction="none")
td_loss = ((mse * valid_mask).sum(dim=0) / valid_mask.sum().clamp_min(1.0)).mean()
```

`q_target` 本身是用 `method="min"` 从多个 critic head（双 Q 或更多）里取最小值算出来的——这是抑制价值高估的标准 SAC 技巧，对应第 05 章讲过的 `sac_forward_critic` 接口。`(1.0 - terminateds)` 这一项是"到达终止状态就不再 bootstrap 未来价值"的标准处理——如果 episode 已经结束，target 就只剩即时奖励 $r$，不应该再加上"未来"的估值（因为已经没有未来了）。

**为什么这个 target 值意味着什么**：假设当前 critic 对这个 $(s,a)$ 输出 $Q_\theta = 7.2$，target 是 $8.913$，说明 critic 目前低估了这个动作的价值，MSE 梯度会推动 $Q_\theta$ 往 $8.913$ 靠近，这个方向的更新会让 critic 更倾向于认为这个动作（以及类似的状态-动作对）是有价值的。

## 5. 可选正则项：CQL

critic loss 里还可以叠加一个 CQL（保守 Q 学习，[前置知识见此](/前置知识/002g_前置知识_CQL保守Q学习)）正则项，用于抑制离线数据之外的动作被过度高估：

```python
if self.cql_enabled and q_policy is not None:
    q_candidates = torch.stack([q_predict, q_policy], dim=0)
    cql_per_critic = torch.logsumexp(q_candidates / self.cql_temperature, dim=0) * self.cql_temperature - q_predict
    cql_loss = self.cql_alpha * cql_per_critic.mean()
```

这一项和主 TD loss 直接相加（`total_loss = td_loss + cql_loss`），完整的 CQL 数学原理已经在前置知识里讲透，这里不重复推导——只强调它在 verl-vla 里是一个**可选开关**，`cql_enabled=False` 时这部分代码完全不参与计算。

## 6. Actor 更新：延迟策略更新 + 可选 BC 正则

Actor 不是每个训练步都更新，而是受两个条件门控：

```python
update_actor = (
    global_steps >= self.actor_config.critic.warmup_steps
    and global_steps % self.actor_config.actor_update_interval == 0
    and not critic_only_update
)
```

**这是标准的延迟策略更新**（对应 [TD3 前置知识](/前置知识/000q_前置知识_TD3) 里讲过的思想）——critic 需要先训练到一定稳定程度（`warmup_steps`）才开始用它的输出更新 actor，且 actor 更新频率通常比 critic 更新更低（`actor_update_interval`），因为 critic 提供的价值信号如果本身还在剧烈波动，用来指导 actor 更新方向会不稳定。

actor loss 本身是标准 SAC 形式，可选叠加行为克隆正则项（TD3+BC 风格）：

```python
def _calculate_actor_loss(self, log_probs, q_values, valids):
    alpha = self._get_alpha()
    loss = -q_values if log_probs is None else alpha * log_probs - q_values
    return (loss * valids).sum() / valids.sum().clamp_min(1.0)
```

$\alpha\log\pi(a|s) - Q(s,a)$ 越小越好——梯度会把策略推向"更高 $Q$ 值"和"更低确定性（更高熵）"之间取得平衡的方向。当 `log_probs is None`（比如第 05 章讲的 ACT 模型，直接加噪声没有 log_prob）时退化成纯粹的 $-Q(s,a)$，即"确定性策略梯度"（DDPG/TD3 风格）而不是严格的最大熵 SAC。

若启用 `td3_enabled`，额外加一项行为克隆 loss（复用模型的 `sft_loss` 接口）：`actor_loss = sac_loss + td3_bc_alpha * bc_loss`——这是防止 SAC 训练把策略带偏离示教数据分布太远的一种简单有效手段，尤其在训练早期 critic 还不够准确时能起到稳定作用。

## 7. Target 网络更新：Polyak 平均与 warmup 期加速

```python
critic_target_tau = (1.0 if force_tau_one_in_warmup and global_steps < warmup_steps else float(tau))
if not skip_critic_update:
    self.engine.module.sac_update_target_network(critic_target_tau)
```

Target 网络软更新公式：

$$
\theta^{\text{target}} \leftarrow \tau \cdot \theta^{\text{online}} + (1-\tau) \cdot \theta^{\text{target}}
$$

> **一句话直觉**：target 网络永远只朝在线网络的方向"慢慢挪"一点点，不会瞬间变成在线网络的完全复制品，这样 Bellman target 里的 $Q^{\text{target}}$ 不会因为在线网络每一步的剧烈更新而剧烈波动。

`warmup_steps` 阶段可以把 $\tau$ 强制设为 1.0——此时 target 网络直接等于在线网络（相当于没有软更新，硬拷贝），因为训练早期在线网络本身变化剧烈，早点让 target 网络跟上反而有助于收敛速度；等过了 warmup 阶段再切回配置里的小 $\tau$（比如 0.005）做正常的软更新，此时更看重 target 网络的稳定性。

Actor 侧还额外维护一份 EMA（指数滑动平均）影子权重，训练每一步后用平滑后的值覆盖回真实参数，作为稳定 rollout 采样行为的额外手段——跟 target 网络的思路是同一类工程技巧（用平滑代替突变）在不同位置的应用。

## 8. ReplayPool：按 task 分组的双池采样

`SACReplayPool`（`utils/replay_pool.py`）是承载所有 transition 的核心数据结构，设计上有两个值得强调的点。

**第一，按任务分组**：每个任务（`task_id`）有自己独立的一组 pool，采样时先在任务之间均匀分配采样数量，再在每个任务内部随机取样。这避免了"某个任务的数据量特别大，训练时几乎全是这个任务的样本"的不均衡问题——在多任务训练场景下尤其重要。

**第二，正负样本分池**：每个任务下又分成 `positive_pool`（对应成功轨迹）和 `negative_pool`（对应失败轨迹），采样时按配置的比例（`positive_sample_ratio`）从两个池分别取样再拼接：

```python
target_positive = int(round(batch_size * positive_sample_ratio))
target_negative = batch_size - target_positive
sampled_positive = min(target_positive, self.positive_size)
sampled_negative = min(target_negative, self.negative_size)
# 某一类数据不够时,用另一类补齐 deficit
```

**为什么要显式控制正负样本比例**：机器人任务的成功率在训练初期通常很低,如果完全按数据自然分布采样,critic 几乎全部在学习"失败长什么样",很难获得足够的成功案例信号来学习"到底怎样才能成功"。显式设置一个较高的正样本采样比例(比如 0.5,即让一半训练样本来自成功轨迹),即使实际收集到的成功数据很少,也能保证每次训练都有足够的正例信号,这是应对稀疏成功率、类别不均衡问题的常见手段。

物理存储上用固定容量的**环形缓冲区**（FIFO 覆盖式写入，`(position + idx) % single_pool_capacity`），最新数据不断覆盖最旧数据，保证 pool 里始终是"最近一段时间"的经验，避免过时的、由早期弱策略产生的经验一直占据训练资源。

## 9. RLPD：混合离线示教数据与在线经验

RLPD（Reinforcement Learning with Prior Data）在 verl-vla 里的实现是"两个 replay pool 并行存在"——除了在线 rollout 产生的 `replay_pool`，还有一个从离线 LeRobot 数据集预填充的 `offline_replay_pool`：

```python
def _prefill_replay_pool_from_rlpd(self):
    if not self.trainer_config.rlpd.enable:
        return
    for prefill_batch in iter_rlpd_replay_prefill_batches(rlpd_config, global_steps=self.global_steps):
        self._submit_rlpd_prefill_batch(prefill_batch)   # 标记 add_to_offline_replay_only=True
```

训练时从两个池分别按比例采样再拼接成一个 mini-batch，喂给 critic/actor 更新：

```python
def _sample_rlpd_batch(self, positive_sample_ratio):
    online_batch, ... = self.replay_pool.sample_batch(self.online_replay_sample_batch_size, positive_sample_ratio, ...)
    offline_batch, ... = self.offline_replay_pool.sample_batch(self.offline_replay_sample_batch_size, positive_sample_ratio, ...)
    return [b for b in (online_batch, offline_batch) if b is not None], sample_info
```

**为什么要这样混合**：纯在线 SAC 在训练初期，策略几乎不会成功，在线 pool 里几乎全是失败经验，critic 很难学到有用的信号。RLPD 的思路是让**离线专家演示数据**（人类遥操作录制的成功示范）从训练一开始就持续参与 critic/actor 的更新，弥补在线数据早期质量低的问题，同时又不完全依赖离线数据（保留在线探索发现新解法的能力）。这正是第 07 章讲的人机协同数据采集能直接服务于 SAC 训练的关键桥梁——遥操作录制出的 LeRobot 数据集，可以直接作为 RLPD 的离线先验数据源接进训练循环。

## 小结

| 概念 | 要点 |
|---|---|
| EpisodeBuffer | 按 lane 缓存 rollout 碎片步，遇到 done 才吐出完整 episode，处理跨窗口/窗口内多 episode 两种情况 |
| off-policy 节奏 | 一次采集 + 多次复用：window 内只在前几步跑真实环境交互，其余步全部从 replay pool 训练 |
| Bellman target | $y = r+\gamma(1-d)(Q^{\text{target}}(s',a')-\alpha\log\pi(a'\|s'))$，双 Q 取最小值抑制高估，终止时不 bootstrap |
| Actor loss | $\alpha\log\pi(a\|s)-Q(s,a)$，延迟更新 + 可选 BC 正则防止偏离示教分布太远 |
| Target 网络 | Polyak 软更新，warmup 期可强制 tau=1 加速收敛 |
| ReplayPool | 按 task 分组 + 正负样本双池采样，环形缓冲保留最近经验 |
| RLPD | 离线示教数据独立预填充一个 pool，训练时和在线 pool 按比例混合采样 |

## 下章预告

[第 09 章](./09_RECAP工作流_六阶段自我提升闭环) 讲 RECAP——一个不依赖显式 reward function、完全基于 episode 长度和成功/失败标签构造 shaped return 的迭代式自我提升工作流，看它的"评估→采集→打分→训练value→推理advantage→训练policy"六个阶段具体怎么串联，以及 return/advantage/indicator 三个核心量的精确计算方式。
