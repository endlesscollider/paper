---
title: "算法实现：PPO 配置详解"
series:
  id: rlinf_deep_dive
  chapter: 8
order: 8
---

# 算法实现：PPO 配置详解

> 前情提要：上一章讲了 Runner 怎么协调 Worker 的执行节奏。本章往下钻一层：Actor 训练那一步里，PPO 算法具体在算什么、每个配置参数改了会发生什么。

> 如果对 PPO 的 clip 机制、GAE 优势估计还不熟悉，建议先读 [策略梯度与 PPO](/前置知识/000a_前置知识_策略梯度与PPO)——本章只讲 RLinf 怎么把这些概念落成配置和代码，不重复推导公式。

## 一、PPO 训练一步要算什么

回顾 [第 07 章](./07_Runner训练循环) 的流程：Runner 采集完一批轨迹后，会依次调用 Actor 的 `compute_advantages_and_returns()` 和 `run_training()`。这两步分别对应 PPO 的两个核心计算：

1. 用收集到的 reward，结合 Critic 的价值估计，算出每一步动作的"优势"（这个动作比平均水平好多少）
2. 用优势去更新策略网络，同时限制更新幅度不要太大（这就是 PPO 名字里 "Clip" 的来源）

这两步涉及的参数分散在配置文件的好几个地方，下面按"先讲这个参数解决什么问题，再看它在代码里怎么生效"的顺序逐一拆开。全部放在一起看，一个典型的 PPO 配置（以具身 RL 任务为例）大致是这样：

```yaml
algorithm:
  adv_type: gae
  loss_type: actor_critic
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  normalize_advantages: True
  huber_delta: 10.0
  reward_type: action_level
  bootstrap_type: always
  kl_beta: 0.0
  group_size: 1
```

接下来一节一节地讲。

## 二、GAE：怎么从"最终奖励"算出"每一步的优势"

### 2.1 问题：一条轨迹的 reward 怎么分给每一步

具身 RL 里，一条轨迹（比如机器人抓取一个物体）有很多步动作，最后可能只在完成任务那一刻给一次奖励。PPO 需要知道"这一步的动作到底好不好"，而不只是"这条轨迹整体好不好"——这正是 [前置知识里 GAE](/前置知识/000a_前置知识_策略梯度与PPO) 要解决的问题：结合 Critic 对每一步"未来能拿多少奖励"的预测，把最终的 reward 逐步往前"折算"到每一个时间步上。

配置里对应的是 `adv_type: gae`，RLinf 用一个装饰器注册的方式管理不同的优势估计算法，`gae` 只是其中一种可选值（还有 GRPO 用的 `grpo`）：

```python
@register_advantage("gae")
def compute_gae_advantages_and_returns(rewards, gamma, gae_lambda, values, dones, ...):
    T = rewards.shape[0]
    gae = 0
    for step in reversed(range(T)):
        delta = rewards[step] + gamma * values[step+1] * (~dones[step+1]) - values[step]
        gae = delta + gamma * gae_lambda * (~dones[step+1]) * gae
        returns[step] = gae + values[step]
    advantages = returns[:] - values[:-1]
```

这个循环是**从最后一步往前算**（`reversed(range(T))`）——因为第 $t$ 步的优势依赖第 $t+1$ 步算出的结果，必须先有"未来"才能推出"现在"。两个配置参数控制这个递推的行为：

- `gamma=0.99`：折扣因子，未来的奖励打折算到现在。值越接近 1，越看重长远的奖励；值越小，越只关心眼前的奖励。
- `gae_lambda=0.95`：控制这个递推是偏向"只看一步就停"（TD，方差小但有偏）还是"一直往后看到底"（Monte Carlo，无偏但方差大）。`gae_lambda=1` 等价于纯 Monte Carlo，`gae_lambda=0` 等价于只看一步的 TD(0)，`0.95` 是两者之间常用的折中值。

### 2.2 优势标准化：为什么要归一化

算出 `advantages` 后，配置里的 `normalize_advantages: True` 会把这批优势值减去均值、除以标准差，变成一个均值 0、方差 1 的分布。这一步的动机很直接：一个 batch 里不同轨迹的奖励尺度可能差异很大（比如有的任务奖励设计是 0/1，有的是累积距离），如果不归一化，梯度更新的幅度会被"奖励尺度大的那批数据"主导，训练不稳定。归一化之后，每个 batch 里"好动作"和"差动作"的相对差异被保留下来，但绝对尺度统一了。

## 三、PPO Clip：限制策略一步能走多远

### 3.1 为什么需要限制更新幅度

有了每一步的优势估计，最直接的做法是"优势为正就提高这个动作的概率，优势为负就降低"。但如果每次更新都朝着这个方向走一大步,策略可能一下子跳到一个从未探索过的、表现很差的区域——因为优势是基于*旧*策略估计出来的,新策略如果变化太大,这个估计就不再准确了。PPO 的解法是**限制新旧策略的概率比值**，不让它偏离 1 太多。

这个比值就是 `ratio = exp(logprobs - old_logprobs)`——新策略对这个动作的对数概率减去旧策略的,再取指数,还原成概率的比值。`ratio > 1` 说明新策略更倾向于选这个动作,`< 1` 说明更不倾向。PPO 的做法是把这个比值裁剪到一个范围内,裁剪后的比值和裁剪前的比值分别算一次 loss,取更保守（数值更大，因为这里是负的 loss）的那个:

```python
ratio = torch.exp(logprobs - old_logprobs)
clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

policy_loss1 = -advantages * ratio
policy_loss2 = -advantages * clipped_ratio
policy_loss = torch.max(policy_loss1, policy_loss2)
```

对应两个配置项：`clip_ratio_high=0.2` 控制比值最多能比 1 大多少（新策略最多比旧策略"更倾向"这个动作 20%），`clip_ratio_low=0.2` 控制最多能比 1 小多少。两者相等时就是标准 PPO 的对称裁剪；如果想让"提高好动作概率"和"降低差动作概率"用不同的容忍度，可以把两者设成不同的值（比如 DAPO 这类改进方法就用了非对称裁剪）。

### 3.2 Dual Clip：优势为负时的额外保护

标准 clip 只管住了"优势为正"时 ratio 太大的情况,但当优势为负（这是个差动作）时,如果 ratio 变得非常大（策略突然极度偏爱这个差动作),`-advantages * ratio` 反而会变得非常大（因为负乘以大的正数),PPO 的 clip 机制在这种情况下不起作用——这是标准 PPO 的一个已知问题。

`clip_ratio_c=3.0` 引入的 Dual Clip 就是为了补上这个漏洞,在标准 clip 之外再加一层保护,给 loss 一个绝对的上界：

```python
if clip_ratio_c is not None:
    policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
    policy_loss = torch.min(policy_loss, policy_loss3)
```

效果是：当优势为负时,不管 ratio 变得多大,loss 都不会超过 `3.0 * |advantage|`——相当于给策略更新的破坏力设了一个天花板,防止极端情况下策略被带偏太远。

## 四、Critic：value_clip 和 huber_delta

### 4.1 为什么 Critic 也需要 clip

PPO 是 Actor-Critic 方法（`loss_type: actor_critic`），除了训练策略网络，还要同时训练一个 Critic（价值头）去预测"这个状态未来能拿多少奖励"，Critic 的预测越准，第二节里的 GAE 优势估计就越可靠。和策略更新一样，Critic 的价值预测如果一步跳得太远，也会让训练变得不稳定——这次的解法思路完全一样：对价值预测的变化幅度做裁剪。

```python
value_pred_clipped = prev_values + torch.clamp(values - prev_values, -value_clip, value_clip)
value_loss_original = huber_loss(returns - values, huber_delta)
value_loss_clipped = huber_loss(returns - value_pred_clipped, huber_delta)
value_loss = torch.max(value_loss_original, value_loss_clipped)
```

`value_clip=0.2` 限制了这一步价值预测相对上一步最多能变化多少,取"裁剪前"和"裁剪后"两个 loss 中更大的那个（同样是保守策略，逻辑和 3.1 节的 policy clip 完全对称）。

### 4.2 为什么用 Huber loss 而不是 MSE

价值预测的误差用什么损失函数衡量也有讲究。普通的 MSE（均方误差）对大误差的惩罚是平方增长的——如果某一步价值预测严重偏离（比如遇到一个训练早期极端的样本），MSE 会产生一个巨大的梯度，把整个训练带偏。Huber loss 的做法是：误差小的时候和 MSE 一样（保持对小误差的敏感度），误差大的时候切换成线性增长（像 MAE 一样，不会被极端样本带偏）：

$$
\text{Huber}(x, \delta) = \begin{cases} \frac{1}{2}x^2 & |x| \le \delta \\ \delta(|x| - \frac{1}{2}\delta) & |x| > \delta \end{cases}
$$

`huber_delta=10.0` 就是这个切换点——误差绝对值小于 10 时表现像 MSE，大于 10 时表现像 MAE。这个值需要结合具体任务的 reward 尺度来定：如果 reward 设计成 0~1 之间的稀疏值，10 这个阈值可能永远不会触发（等价于纯 MSE）；如果 reward 是累积距离这类数值较大的信号，就需要调小这个阈值才能真正起到抑制极端梯度的作用。

## 五、bootstrap_type：Episode 结束时该不该"预测未来"

### 5.1 两种结束方式的区别

GAE 递推需要知道每一步的"下一个状态的价值" `values[step+1]`，但 episode 结束时没有下一个状态了怎么办？这里要先区分两种不同的"结束"：

- **termination**：任务真正完成了（比如机器人成功抓到了物体），之后不会再有奖励，不需要预测未来
- **truncation**：只是因为步数用完被强制截断，任务本身还没完成，理论上还有未来的奖励没被算进去

`bootstrap_type` 控制这两种情况下是否用 Critic 去"猜"一个 `values[step+1]` 填进递推公式（这就是"bootstrap"这个词的含义——用模型自己的预测去补全缺失的未来）：

- `standard`：只在 truncation 时 bootstrap（因为任务还没完成，未来确实存在），termination 时不 bootstrap（视为未来价值是 0）
- `always`：termination 和 truncation 都 bootstrap

### 5.2 为什么具身 RL 通常用 always

具身 RL 里"任务完成"不代表"这条轨迹后面没有价值可学"——比如机器人抓取任务，抓到物体后可能还有几十步的仿真在跑（等环境判定 episode 该结束），如果直接把这些步的未来价值设为 0，会人为地压低了成功轨迹后半段的优势估计，让"成功"这件事在训练信号里显得不够有分量。所以具身 RL 场景几乎都用 `always`，配合 `env.train.ignore_terminations: True`（[后面会讲到](#七env-配置auto-reset-与-ignore-terminations)）：不把 termination 当成真正的边界，让数据收集持续到步数上限，训练信号更连续。

## 六、Actor 配置：batch size 和梯度累积

### 6.1 micro_batch_size 与 global_batch_size 的关系

[第 06 章](./06_训练后端_FSDP与Megatron#五微批量训练梯度累积与-before_micro_batch)讲过 FSDP 训练需要把一个大 batch 拆成多个 micro-batch 依次算、梯度本地累积。这里的拆分方式由两个配置项共同决定：

```yaml
actor:
  micro_batch_size: 20     # 单次前向+反向处理的样本数（受显存限制）
  global_batch_size: 160   # 一次参数更新总共用到的样本数（决定了梯度的统计意义）
```

两者的关系是：

$$
\text{梯度累积步数} = \frac{\text{global\_batch\_size}}{\text{micro\_batch\_size} \times \text{actor\_world\_size}}
$$

比如 `global_batch_size=160`，`micro_batch_size=20`，Actor 用 2 张卡（`actor_world_size=2`），累积步数就是 `160 / (20 * 2) = 4`——每张卡要连续处理 4 个 micro-batch，把梯度累积起来，才做一次真正的参数更新。`micro_batch_size` 主要受显存限制（越大显存占用越高），`global_batch_size` 主要影响梯度的统计稳定性（越大梯度估计越准，但每一步训练变慢）——这是显存和训练稳定性之间的权衡。

### 6.2 critic_warmup_steps：让 Critic 先学会"看懂"环境

PPO 训练刚开始时，Critic 的价值预测是完全随机的，如果这时候就用它来算优势、指导 Actor 更新，得到的优势估计毫无意义，反而会把 Actor 引向错误的方向。`critic_warmup_steps` 提供了一个缓冲期：在这段步数内，只更新 Critic 的参数，Actor 的 loss 被强制置零，不产生梯度：

```python
if self.optimizer_steps < self.critic_warmup_steps:
    policy_loss = torch.tensor(0.0)
```

等 Critic 训得差不多能看懂"什么状态值多少分"了，再让 Actor 开始基于这些价值估计做真正的策略更新。默认值是 0（不做预热），只有在观察到训练早期 Actor loss 震荡剧烈时才需要调大这个值。

## 七、Env 配置：auto_reset 与 ignore_terminations

前面提到的 `bootstrap_type: always` 需要配合两个 Env 配置项一起工作,它们共同决定了"episode 结束"这件事在数据层面是怎么被处理的：

```yaml
env:
  train:
    auto_reset: True           # episode 结束后环境自动重置,继续收集
    ignore_terminations: True  # 不把 termination 当作真正的边界
    max_episode_steps: 80
```

| auto_reset | ignore_terminations | 行为 |
|---|---|---|
| True | True | 结束自动重置，不停止收集，GAE 计算时忽略 done（对应 `always` bootstrap） |
| True | False | 结束自动重置，但 GAE 计算时在 done 处截断（对应 `standard` bootstrap） |
| False | False | 结束不重置，需要显式管理，支持 loss_mask 过滤无效步 |

推荐组合是 `auto_reset: True` + `ignore_terminations: True`，和第五节的 `bootstrap_type: always` 保持一致——三个配置项实际上是在描述同一个设计决策的三个方面，改一个通常需要三个一起改。

## 八、Rollout 配置：pipeline_stage_num 怎么"隐藏"推理延迟

### 8.1 问题：Env 和 Rollout 交替执行天然有空等

Env 要等 Rollout 给动作才能往下走一步仿真，Rollout 要等 Env 给观测才能推理——如果严格串行，两者永远有一方在等待对方。`pipeline_stage_num` 提供了一种缓解方式：把整批环境拆成几个 stage，不同 stage 交错执行，让"Rollout 推理 stage A 的动作"和"Env 执行 stage B 上一步的动作"同时发生：

```yaml
rollout:
  pipeline_stage_num: 2
```

效果类似流水线：当 Rollout 在给 stage 1 的环境算下一步动作时，stage 2 的环境正好可以利用这段时间执行上一步已经算好的动作，两部分工作交叠起来，谁都不用完全空等对方。`pipeline_stage_num=2` 通常已经能把大部分推理延迟藏起来；再往上增加 stage 数收益会递减，还会增加调度和内存管理的复杂度。

## 九、总结

| 参数 | 解决的问题 | 关键权衡 |
|------|-----------|---------|
| `gamma` / `gae_lambda` | 怎么把最终奖励折算到每一步 | 长远 vs 眼前，偏差 vs 方差 |
| `normalize_advantages` | 不同 batch 间奖励尺度不一致 | 统一尺度，保留相对差异 |
| `clip_ratio_high/low` | 防止策略更新一步跨太远 | 更新幅度 vs 学习速度 |
| `clip_ratio_c`（Dual Clip） | 优势为负时标准 clip 失效的漏洞 | 极端情况下的额外保护 |
| `value_clip` / `huber_delta` | Critic 预测突变 / 极端误差带偏训练 | 稳定性 vs 拟合速度 |
| `bootstrap_type` | episode 结束时怎么处理"未来价值" | 任务语义 vs 训练信号连续性 |
| `micro_batch_size` / `global_batch_size` | 显存限制下怎么维持梯度统计稳定性 | 显存 vs 训练稳定性 |
| `critic_warmup_steps` | Critic 没学好时不该指导 Actor | 训练前期稳定性 vs 训练时长 |
| `pipeline_stage_num` | Env/Rollout 交替执行的空等 | 延迟隐藏 vs 调度复杂度 |

## 下一章预告

[第 09 章](./09_算法实现_GRPO配置详解) 将对比 GRPO 与 PPO 的差异，逐一讲解 `group_size`、`filter_rewards`、`rollout_epoch` 等 GRPO 特有配置。
