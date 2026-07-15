---
title: "算法实现：GRPO 配置详解"
series:
  id: rlinf_deep_dive
  chapter: 9
order: 9
---

# 算法实现：GRPO 配置详解

> 前情提要：上一章逐参数拆解了 PPO 配置。本章对比 GRPO 与 PPO 的差异。

## GRPO vs PPO：核心区别

| 维度 | PPO | GRPO |
|------|-----|------|
| 是否需要 Critic | ✅ 需要 value head | ❌ 不需要 |
| 优势估计方式 | GAE（需要 V(s)） | 组内相对排名 |
| group_size | 1 | 通常 4-8 |
| 显存占用 | 较高（存 value head + optimizer） | 较低 |
| 训练稳定性 | 更稳定（有 baseline） | 需要更多 rollout 来降低方差 |
| 典型场景 | 环境奖励稠密 | 环境奖励稀疏（只有最终 success/fail） |

## GRPO 典型配置

以 `libero_spatial_grpo_openpi.yaml` 为例（对比 PPO 变化的部分）：

```yaml
algorithm:
  # ===== 与 PPO 不同的核心参数 =====
  adv_type: grpo              # 改为 grpo（不是 gae）
  loss_type: actor            # 改为 actor（不是 actor_critic，无 critic loss）
  group_size: 8               # 每个任务做 8 次 rollout，组内排名
  rollout_epoch: 8            # 每个训练步收集 8 轮数据
  update_epoch: 2             # 每批数据训练 2 轮
  
  # ===== GRPO 特有参数 =====
  filter_rewards: True        # 过滤全成功/全失败的组
  rewards_lower_bound: 0.1    # 组平均奖励下界
  rewards_upper_bound: 0.9    # 组平均奖励上界
  
  # ===== 粒度通常为 chunk_level =====
  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level
  
  # ===== 不需要 GAE 参数 =====
  # gamma 和 gae_lambda 在 GRPO 中不使用

actor:
  micro_batch_size: 128       # 通常比 PPO 大（无 value head 占显存）
  global_batch_size: 2048     # 更大的 batch
  model:
    add_value_head: False     # ❌ 不加 value head
    num_action_chunks: 5

critic:
  use_critic_model: False     # ❌ 不用 Critic

env:
  train:
    total_num_envs: 64
    group_size: ${algorithm.group_size}  # 环境数 = 实际任务数 × group_size
```

## group_size 的工作机制

`group_size=8` 意味着：每个任务描述（prompt）会被执行 8 次，得到 8 个 rollout 结果。

```
任务 "把杯子放到盘子上":
  rollout 1: 成功 → reward = 1.0
  rollout 2: 失败 → reward = 0.0
  rollout 3: 成功 → reward = 1.0
  rollout 4: 失败 → reward = 0.0
  rollout 5: 成功 → reward = 1.0
  rollout 6: 失败 → reward = 0.0
  rollout 7: 成功 → reward = 1.0
  rollout 8: 失败 → reward = 0.0

组内平均: mean = 0.5, std = 0.5

标准化后 advantages:
  成功的 rollout: (1.0 - 0.5) / 0.5 = +1.0  → 强化
  失败的 rollout: (0.0 - 0.5) / 0.5 = -1.0  → 抑制
```

### GRPO 优势计算源码

```python
@register_advantage("grpo")
def compute_grpo_advantages(rewards, loss_mask, group_size, **kwargs):
    grouped_rewards = rewards.view(-1, group_size)  # [num_prompts, group_size]
    
    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True)
    
    advantages = (grouped_rewards - grouped_reward_mean) / (grouped_reward_std + 1e-6)
    
    # 广播到 loss_mask 的形状
    advantages = advantages.view(1, -1) * loss_mask
    return advantages, None
```

## filter_rewards：奖励过滤

当一个组内全部成功或全部失败时，组内标准差为 0，advantages 无意义。RLinf 用 `filter_rewards` 过滤掉这些组：

```python
if self.cfg.algorithm.get("filter_rewards", False):
    # 计算每个组的平均奖励
    mean_reward_in_group = reward_matrix.mean(dim=1)  # [num_prompts]
    
    # 只保留平均奖励在 [lower, upper] 之间的组
    reward_filter_mask = (
        mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
    ) & (
        mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
    )
    
    # 更新 loss_mask
    rollout_batch["loss_mask"] = reward_filter_mask & rollout_batch["loss_mask"]
```

- `rewards_lower_bound: 0.1`：全失败的组（mean ≈ 0）被过滤
- `rewards_upper_bound: 0.9`：全成功的组（mean ≈ 1）被过滤
- 只有有区分度的组参与训练

## rollout_epoch 与 group_size 的关系

- `rollout_epoch=8`：环境会连续跑 8 轮 rollout
- `group_size=8`：每轮用不同的随机种子，产出同一任务的 8 个变体

数据量计算：
```
每步训练数据量 = total_num_envs × rollout_epoch × n_chunk_steps
             = 64 × 8 × (240/5)
             = 64 × 8 × 48
             = 24576 个 chunk
```

## update_epoch：PPO 式多轮更新

GRPO 也支持对同一批数据多轮更新（PPO-style epochs）：

```yaml
algorithm:
  update_epoch: 2   # 对同一批数据训练 2 轮
```

每轮会重新 shuffle 数据。这利用了 PPO clip 的保护——即使多轮更新，clip 机制确保策略不会偏离太远。

## GRPO 的 loss_type: actor

与 PPO 的 `actor_critic` 不同，GRPO 只有 Actor loss：

```python
@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs):
    actor_loss, actor_metrics = compute_ppo_actor_loss(**kwargs)
    return actor_loss, actor_metrics  # 没有 critic_loss
```

底层仍然用 PPO 的 clip 机制计算策略梯度——GRPO 的"GRPO 部分"体现在 advantage 的计算方式，loss 函数与 PPO 完全相同。

## 环境端的 group_size 配置

```yaml
env:
  train:
    total_num_envs: 64
    group_size: ${algorithm.group_size}  # 引用 algorithm 的值
```

Env Worker 用 `group_size` 来决定如何给每组环境分配任务描述。64 个环境 / group_size 8 = 8 个不同任务同时在跑。

## GRPO vs PPO 配置差异对照表

| 配置字段 | PPO 值 | GRPO 值 | 原因 |
|---------|--------|---------|------|
| `adv_type` | `gae` | `grpo` | 不同的 advantage 计算 |
| `loss_type` | `actor_critic` | `actor` | GRPO 无 Critic |
| `group_size` | `1` | `4-8` | GRPO 需要多个 rollout 对比 |
| `rollout_epoch` | `1` | `4-8` | 配合 group_size |
| `model.add_value_head` | `True` | `False` | GRPO 不需要 value head |
| `reward_type` | `action_level` | `chunk_level` | GRPO 通常用 chunk 级别 |
| `micro_batch_size` | 20 | 128 | 无 value head，显存更多 |
| `gamma/gae_lambda` | 有意义 | 不使用 | GRPO 不做时序 discount |
| `filter_rewards` | 不需要 | `True` | 过滤无区分度的组 |

## 下一章预告

[第 10 章](./10_算法实现_SAC与其他算法) 将讲解 SAC、DAgger、DSRL、NFT 等其他算法在 RLinf 中的实现差异。
