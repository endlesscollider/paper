---
title: "算法实现：GRPO 配置详解"
series:
  id: rlinf_deep_dive
  chapter: 9
order: 9
---

# 算法实现：GRPO 配置详解

> 前情提要：上一章逐参数拆解了 PPO——核心是用 Critic 预测每一步的价值，再用 GAE 把最终奖励折算成每一步的优势。本章看 GRPO 怎么用一个完全不同的思路，绕开 Critic 这一整套东西。

> 如果对 GRPO 的分组采样、组内标准化思想还不熟悉，建议先读 [GRPO 前置知识](/前置知识/000m_前置知识_GRPO_Group_Relative_Policy_Optimization)——本章只讲 RLinf 怎么把这个思想落成配置和代码。

## 一、GRPO 要解决什么问题

PPO 的优势估计依赖一个训练好的 Critic——但 Critic 本身也需要训练，如果任务的 reward 非常稀疏（比如整条轨迹只有最后一刻的成功/失败），Critic 很难学出准确的价值预测，优势估计的质量就跟着变差。GRPO（Group Relative Policy Optimization）换了一个思路：不训练 Critic，而是对同一个任务**多跑几次**，让这几次的结果互相比较，谁的 reward 比这组的平均水平高就是好的、比平均水平低就是差的。这个"和组内同伴比"的相对排名，就直接充当了优势估计，完全不需要 Critic 参与。

这个思路上的差异，直接决定了配置和代码层面 GRPO 和 PPO 的核心区别：

| 维度 | PPO | GRPO |
|------|-----|------|
| 是否需要 Critic | 需要 value head | 不需要 |
| 优势怎么来 | GAE，依赖 $V(s)$ | 组内相对排名（和同伴比） |
| `group_size` | 1（每个任务跑一次） | 通常 4~8（每个任务跑多次） |
| 显存占用 | 较高（多存一个 value head + 对应优化器状态） | 较低 |
| 适用场景 | reward 稠密，Critic 容易学 | reward 稀疏（只有最终 success/fail） |

## 二、group_size：怎么让"多跑几次"产生优势信号

### 2.1 直觉：把结果排出名次

`group_size=8` 的意思是：给同一个任务描述（比如"把杯子放到盘子上"）连续跑 8 次 rollout，得到 8 个不同的结果（因为策略采样带随机性，8 次的轨迹和结局都可能不一样）：

```
任务 "把杯子放到盘子上":
  rollout 1: 成功 → reward = 1.0
  rollout 2: 失败 → reward = 0.0
  rollout 3: 成功 → reward = 1.0
  ...（共 8 次，假设 4 次成功 4 次失败）

组内平均 mean = 0.5，组内标准差 std = 0.5

标准化后的优势：
  成功的那几次: (1.0 - 0.5) / 0.5 = +1.0  → 训练时强化这次的动作
  失败的那几次: (0.0 - 0.5) / 0.5 = -1.0  → 训练时抑制这次的动作
```

**为什么要减均值再除标准差**：减均值让"优势"变成相对量——不是看这次 reward 绝对是多少，而是看比这组的平均水平好还是差。除标准差是为了把不同任务之间的优势值统一到差不多的尺度上（有些任务 8 次里 7 次成功，方差很小；有些任务 4 次成功 4 次失败，方差较大，直接用未标准化的差值会导致方差小的任务对梯度贡献过小）。

### 2.2 代码：分组、算均值方差、标准化

这个逻辑在 `compute_grpo_advantages` 里，注册方式和上一章的 GAE 一致（用装饰器接入统一的 advantage 接口），只是内部计算完全不同：

```python
@register_advantage("grpo")
def compute_grpo_advantages(rewards, loss_mask, group_size, **kwargs):
    # 把展平的 reward 重新按组分开：[num_prompts * group_size] -> [num_prompts, group_size]
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True)

    advantages = (grouped_rewards - grouped_reward_mean) / (grouped_reward_std + 1e-6)
    advantages = advantages.view(1, -1) * loss_mask   # 展平回去，配合 loss_mask 过滤无效步
    return advantages, None
```

`rewards.view(-1, group_size)` 这一步是整个函数的关键——它假设传进来的 reward 是按"同一任务的 `group_size` 个 rollout 排在一起"组织的，reshape 之后第一维就是"任务数"，第二维就是"组内第几次尝试"，`mean`/`std` 沿着组内维度（`dim=-1`）算，天然就是"组内比较"。分母加 `1e-6` 是防止某组标准差恰好是 0（比如全成功或全失败）时除零——但这种情况本身也不该产生有效的训练信号，这就引出了下一节的过滤机制。

## 三、filter_rewards：踢掉"没有区分度"的组

### 3.1 问题：全成功或全失败的组，优势毫无意义

如果一组 8 次 rollout 全部成功（或全部失败），组内标准差是 0，第二节公式里除以 `(std + 1e-6)` 会得到一个极大或没有意义的数值——这组数据不但没有提供有效的训练信号，反而可能因为数值不稳定污染梯度。更根本的问题是：全成功说明这个任务对当前策略"太容易"，全失败说明"太难"，两种情况下策略都学不到有区分度的东西——真正有用的信号来自"部分成功部分失败"的组，这才能告诉策略"哪种做法更好"。

`filter_rewards` 就是用来把这类没有区分度的组直接踢出训练的：

```yaml
algorithm:
  filter_rewards: True
  rewards_lower_bound: 0.1   # 组平均 reward 低于这个值，说明接近全失败，过滤掉
  rewards_upper_bound: 0.9   # 组平均 reward 高于这个值，说明接近全成功，过滤掉
```

实现上是算出每组的平均 reward，只保留落在 `[lower_bound, upper_bound]` 区间内的组，通过更新 `loss_mask` 让被过滤的组不参与梯度计算（而不是真的删除数据，这样张量形状保持一致，只是这部分的 loss 贡献变成 0）：

```python
mean_reward_in_group = reward_matrix.mean(dim=1)  # [num_prompts]
reward_filter_mask = (
    (mean_reward_in_group >= rewards_lower_bound) &
    (mean_reward_in_group <= rewards_upper_bound)
)
rollout_batch["loss_mask"] = reward_filter_mask & rollout_batch["loss_mask"]
```

`0.1` 和 `0.9` 这两个阈值不是精确的"全成功/全失败"判定（`group_size=8` 时真正的全失败均值是 0，全成功是 1），留出一点缓冲区间的原因是：即使 7 比 1 这种接近极端的组，区分度也已经很弱，同样值得过滤掉，把训练预算留给更有信息量的中间地带的组。

## 四、rollout_epoch 与 group_size 如何共同决定数据量

`group_size` 决定"同一个任务跑几次"，`rollout_epoch` 决定"一次训练步收集几轮数据"，两者是相乘的关系，直接影响每一步用到的训练数据总量：

```yaml
algorithm:
  group_size: 8
  rollout_epoch: 8
env:
  train:
    total_num_envs: 64
```

一步训练用到的 chunk 总数是：

$$
\text{总 chunk 数} = \text{total\_num\_envs} \times \text{rollout\_epoch} \times \text{n\_chunk\_steps}
$$

代入数字，假设每个 episode 跑 240 步、每个 action chunk 包含 5 步（`n_chunk_steps = 240/5 = 48`）：

$$
64 \times 8 \times 48 = 24576 \text{ 个 chunk}
$$

这里 `total_num_envs=64`、`group_size=8` 意味着实际同时在跑的**不同任务数**是 `64 / 8 = 8` 个——64 个并行环境里，每 8 个环境跑的是同一个任务的 8 个副本，这样才能让第二节的"组内比较"有意义（组内的 8 个环境必须是同一个任务，否则比较就没有意义）。

## 五、GRPO 的 loss：仍然是 PPO 的 clip，只是没有 Critic loss

前几节讲的都是"优势怎么算"这一半，优势算出来之后，用它去更新策略网络的方式，GRPO 和 PPO 完全一样——都是 [上一章](./08_算法实现_PPO配置详解#三ppo-clip限制策略一步能走多远)讲过的 ratio clip 机制。区别只体现在配置的 `loss_type` 上：

```yaml
algorithm:
  loss_type: actor   # 而不是 actor_critic
```

对应的注册函数直接复用了上一章的 `compute_ppo_actor_loss`，只是不再额外计算 critic loss：

```python
@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs):
    actor_loss, actor_metrics = compute_ppo_actor_loss(**kwargs)
    return actor_loss, actor_metrics   # 没有 critic_loss，因为没有 Critic
```

这也是为什么 GRPO 配置里 `model.add_value_head: False`、`critic.use_critic_model: False`——没有 Critic，就不需要额外的一套参数和优化器，`micro_batch_size` 因此可以设得比 PPO 更大（比如 128 对比 PPO 常见的 20），因为省下了 value head 占用的那部分显存。同理，GRPO 完全不涉及"未来价值"这个概念，`gamma`、`gae_lambda`、`bootstrap_type` 这些和时序折算相关的参数在 GRPO 里都不生效。

## 六、update_epoch：同一批数据能不能多训几轮

GRPO 和 PPO 都支持对同一批采集到的数据做多轮参数更新（而不是收集一批数据只更新一次就扔掉）：

```yaml
algorithm:
  update_epoch: 2   # 对同一批 rollout 数据训练 2 轮，每轮重新 shuffle
```

**为什么能这样做而不会"过拟合"到这批旧数据**：这正是 PPO clip 机制存在的意义——每一轮更新，`ratio = exp(logprobs - old_logprobs)` 里的 `old_logprobs` 始终是采集数据那一刻的策略,不会随着 `update_epoch` 的轮次变化,clip 机制保证了不管重复训练几轮，新策略和采集时的旧策略之间的差异始终被限制在 `clip_ratio` 允许的范围内——这也是为什么 GRPO 虽然抛弃了 Critic，仍然完整保留了 PPO 的 clip 部分：clip 提供的"更新幅度保护"和"是否用 Critic 估计优势"是两个独立的设计决策，可以自由组合。

## 七、和 PPO 配置的完整对照

| 配置字段 | PPO 典型值 | GRPO 典型值 | 原因 |
|---------|--------|---------|------|
| `adv_type` | `gae` | `grpo` | 优势来源不同：Critic 估计 vs 组内排名 |
| `loss_type` | `actor_critic` | `actor` | GRPO 没有 Critic loss |
| `group_size` | 1 | 4~8 | GRPO 需要同任务多次 rollout 才能比较 |
| `model.add_value_head` | True | False | 省去 value head 的参数和显存 |
| `micro_batch_size` | 较小（如 20） | 较大（如 128） | 没有 value head，显存腾出来了 |
| `gamma` / `gae_lambda` / `bootstrap_type` | 生效 | 不生效 | GRPO 不涉及时序价值折算 |
| `filter_rewards` | 不需要 | 建议开启 | 过滤组内无区分度的数据 |

## 下一章预告

[第 10 章](./10_算法实现_SAC与其他算法) 将讲解 SAC、DAgger、DSRL、NFT 等其他算法在 RLinf 中的实现差异。
