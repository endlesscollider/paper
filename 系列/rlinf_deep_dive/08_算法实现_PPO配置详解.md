---
title: "算法实现：PPO 配置详解"
series:
  id: rlinf_deep_dive
  chapter: 8
order: 8
---

# 算法实现：PPO 配置详解

> 前情提要：上一章详解了 Runner 的同步和异步模式。本章进入算法层面，逐参数拆解 PPO 的配置。

## PPO 在 RLinf 中的典型配置

以 `maniskill_ppo_openvla_quickstart.yaml` 为例：

```yaml
algorithm:
  # ===== 核心算法选择 =====
  adv_type: gae              # 优势估计方式：gae / grpo / reinpp / raw
  loss_type: actor_critic    # 损失类型：actor_critic（PPO）/ actor（GRPO）
  loss_agg_func: "token-mean" # 损失聚合方式
  
  # ===== PPO Clip =====
  clip_ratio_high: 0.2       # ratio 上界裁剪
  clip_ratio_low: 0.2        # ratio 下界裁剪
  clip_ratio_c: 3.0          # dual clip 系数（可选）
  value_clip: 0.2            # value function clip
  
  # ===== GAE 参数 =====
  gamma: 0.99                # 折扣因子
  gae_lambda: 0.95           # GAE λ 参数
  normalize_advantages: True # 优势标准化
  
  # ===== Critic =====
  huber_delta: 10.0          # Critic loss 的 Huber delta
  
  # ===== 奖励/logprob 粒度 =====
  reward_type: action_level   # action_level / chunk_level / token_level
  logprob_type: action_level  # 同上
  entropy_type: action_level  # 同上
  
  # ===== Bootstrap =====
  bootstrap_type: always      # standard / always
  
  # ===== 正则化 =====
  kl_beta: 0.0               # KL 惩罚系数
  entropy_bonus: 0            # 熵正则系数
  
  # ===== Rollout 参数 =====
  group_size: 1               # 每个 prompt 的 rollout 数（PPO=1，GRPO>1）
  rollout_epoch: 1            # 每个训练步收集几轮数据
  
  # ===== 采样参数 =====
  sampling_params:
    do_sample: True
    temperature_train: 1.0
    temperature_eval: 0.6
    top_k: 50
    top_p: 1.0
```

## 逐参数详解

### adv_type: gae

选择优势估计算法。对于 PPO，使用 GAE（Generalized Advantage Estimation）：

```python
@register_advantage("gae")
def compute_gae_advantages_and_returns(rewards, gamma, gae_lambda, values, dones, ...):
    T = rewards.shape[0]
    gae = 0
    for step in reversed(range(T)):
        delta = rewards[step] + gamma * values[step+1] * (~dones[step+1]) - values[step]
        gae = delta + gamma * gae_lambda * (~dones[step+1]) * gae
        returns[step] = gae + values[step]
    advantages = returns - values[:-1]
```

- `gamma=0.99`：每步奖励按 0.99 衰减。越远的奖励权重越低。
- `gae_lambda=0.95`：平衡偏差和方差。=1 时等价于 Monte Carlo；=0 时等价于 TD(0)。

### loss_type: actor_critic

注册的损失函数 `compute_ppo_actor_critic_loss`：

```python
@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs):
    actor_loss, actor_metrics = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics = compute_ppo_critic_loss(**kwargs)
    return actor_loss + critic_loss, {**actor_metrics, **critic_metrics}
```

即 PPO 同时训练 Actor（策略头）和 Critic（价值头）。

### clip_ratio_high / clip_ratio_low

PPO 的核心裁剪机制：

```python
ratio = exp(logprobs - old_logprobs)  # 新旧策略概率比
clipped_ratio = clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)

policy_loss1 = -advantages * ratio
policy_loss2 = -advantages * clipped_ratio
policy_loss = max(policy_loss1, policy_loss2)  # 取较保守的那个
```

- `clip_ratio_high=0.2`：新策略最多比旧策略"好" 20%
- `clip_ratio_low=0.2`：新策略最多比旧策略"差" 20%
- 通常两者相等。设不等可以做非对称裁剪（如 DAPO）。

### clip_ratio_c: 3.0（Dual Clip）

Dual clip 防止 advantages 为负时 ratio 过大：

```python
if clip_ratio_c is not None:
    policy_loss3 = sign(advantages) * clip_ratio_c * advantages
    policy_loss = min(policy_loss, policy_loss3)  # 给一个额外上界
```

当 `advantages < 0`（坏动作）时，即使 ratio 很大，loss 也不会超过 `3.0 * |advantage|`。

### value_clip: 0.2

对 Critic 的价值预测做裁剪，防止价值函数突变：

```python
value_pred_clipped = prev_values + clamp(values - prev_values, -value_clip, value_clip)
value_loss = max(huber(returns - values), huber(returns - value_pred_clipped))
```

### huber_delta: 10.0

Critic loss 使用 Huber loss 代替 MSE，对大误差更鲁棒：

```python
def huber_loss(x, delta):
    if |x| <= delta: return 0.5 * x^2
    else: return delta * (|x| - 0.5 * delta)
```

`delta=10.0` 意味着误差 < 10 时用 MSE，> 10 时用 MAE。

### reward_type / logprob_type

控制奖励和 logprob 的计算粒度：

| 类型 | 含义 | 张量形状 |
|------|------|---------|
| `action_level` | 每个 action chunk 一个值 | `[B, num_chunks, action_dim]` |
| `chunk_level` | 每个 chunk 一个值 | `[B, num_chunks, 1]` |
| `token_level` | 每个 token 一个值（VLA 自回归） | `[B, seq_len]` |

PPO with VLA 通常用 `action_level`；GRPO 通常用 `chunk_level`。

### bootstrap_type

Episode 结束时是否用 V(s') 做 bootstrap：

- `standard`：只在 truncation（超时截断）时 bootstrap，termination（任务完成）时不 bootstrap
- `always`：termination 和 truncation 都 bootstrap

具身 RL 通常用 `always`，因为任务成功不代表学习应该停止。

### loss_agg_func

损失的聚合方式：

| 值 | 含义 |
|----|------|
| `"mean"` | 对所有有效 token 取平均 |
| `"token-mean"` | 按 token 数加权平均（等价于 mean） |
| `"episode-mean"` | 按 episode 长度加权平均 |

## Actor 配置

```yaml
actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  
  micro_batch_size: 20        # 每个 micro-batch 的样本数
  global_batch_size: 160      # 全局 batch size
  seed: 1234
  enable_offload: True        # GPU 显存不够时开启
  
  model:
    model_path: "/path/to/model"
    add_value_head: True      # 加 value head（PPO 需要）
    is_lora: True             # 使用 LoRA
    max_prompt_length: 30     # prompt token 数上限
    num_action_chunks: 5      # 动作 chunk 数
  
  optim:
    lr: 1.0e-4                # 策略学习率
    value_lr: 3.0e-3          # 价值头学习率（通常更高）
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0            # 梯度裁剪阈值
    critic_warmup_steps: 0    # Critic 预热步数（此期间 actor loss = 0）
```

### micro_batch_size vs global_batch_size

- `global_batch_size`：一个训练步使用的总样本数
- `micro_batch_size`：每次前向+反向的样本数
- 梯度累积步数 = `global_batch_size / (micro_batch_size * actor_world_size)`

例如：`global_batch_size=160`，`micro_batch_size=20`，`actor_world_size=2` → 累积 4 步。

### critic_warmup_steps

让 Critic 先学几步后再让 Actor 开始训练：

```python
if self.optimizer_steps < self.critic_warmup_steps:
    policy_loss = torch.tensor(0.0)  # Actor loss 归零
```

这有助于 Critic 先建立稳定的价值估计，再用来指导 Actor。

## Env 配置

```yaml
env:
  train:
    total_num_envs: 32        # 训练用并行环境总数
    auto_reset: True          # episode 结束自动 reset
    ignore_terminations: True # 忽略 termination 信号（always bootstrap）
    max_episode_steps: 80     # 最大步数（truncation 阈值）
    max_steps_per_rollout_epoch: 80  # 每个 rollout epoch 的最大步数
    enable_offload: True      # 环境 GPU 渲染完后 offload
  eval:
    total_num_envs: 32
    video_cfg:
      save_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval
```

### auto_reset 与 ignore_terminations

| auto_reset | ignore_terminations | 行为 |
|---|---|---|
| True | True | Episode 结束自动 reset，不停止收集。GAE 计算忽略 done（always bootstrap）。 |
| True | False | Episode 结束自动 reset，GAE 计算时 done 处截断。 |
| False | False | Episode 结束不 reset，需要显式管理。支持 loss_mask。 |

推荐配置：`auto_reset: True` + `ignore_terminations: True`（与 `bootstrap_type: always` 搭配）。

## Rollout 配置

```yaml
rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"       # huggingface / sglang / vllm
  gpu_memory_utilization: 0.5  # Rollout 模型的 GPU 显存占比
  enforce_eager: True          # 禁用 CUDA graph（调试用）
  enable_offload: True         # 推理完 offload
  pipeline_stage_num: 2        # 流水线 stage 数
  
  model:
    model_path: "/path/to/model"
    precision: ${actor.model.precision}
```

### pipeline_stage_num

将 Env-Rollout 交互切分为多个 stage，流水线化执行：

```
Stage 1: Env 发送 obs → Rollout 推理 → Env step
Stage 2: Env 发送 obs → Rollout 推理 → Env step  (与 Stage 1 的下一步重叠)
```

`pipeline_stage_num=2` 几乎可以把推理延迟藏起来。

## 下一章预告

[第 09 章](./09_算法实现_GRPO配置详解) 将对比 GRPO 与 PPO 的差异，逐一讲解 `group_size`、`filter_rewards`、`rollout_epoch` 等 GRPO 特有配置。
