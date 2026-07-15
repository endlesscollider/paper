---
title: "Runner 训练循环"
series:
  id: rlinf_deep_dive
  chapter: 7
order: 7
---

# Runner 训练循环

> 前情提要：上一章详解了 FSDP 训练后端。本章聚焦 Runner 层——训练主循环的两种模式。

## Runner 的定位

Runner 是 RLinf 的"指挥官"，不参与具体计算，只负责：
1. 协调 Worker Group 的执行顺序
2. 控制权重同步频率
3. 管理评估和 Checkpoint
4. 收集和记录 metrics

## EmbodiedRunner：同步模式

### 构造时初始化

```python
class EmbodiedRunner:
    def __init__(self, cfg, actor, rollout, env, reward=None, critic=None):
        # Channel 创建
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        self.reward_channel = Channel.create("Reward") if reward else None
        
        # 步数控制
        self.global_step = 0
        self.weight_sync_interval = cfg.runner.weight_sync_interval
        
        # 计时器
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        
        # Metric logger (TensorBoard / WandB / SwanLab)
        self.metric_logger = MetricLogger(cfg)
```

### 主循环详解

```python
def run(self):
    for step in range(start_step, self.max_steps):
        self.actor.set_global_step(self.global_step)
        self.rollout.set_global_step(self.global_step)
        
        with self.timer("step"):
            # ========== 阶段 1：权重同步 ==========
            with self.timer("sync_weights"):
                if step % self.weight_sync_interval == 0:
                    self.update_rollout_weights()
            
            # ========== 阶段 2：数据采集 ==========
            with self.timer("generate_rollouts"):
                env_handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    actor_channel=self.actor_channel,
                    reward_channel=self.reward_channel,
                )
                rollout_handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                if self.reward:
                    reward_handle = self.reward.compute_rewards(...)
                
                env_handle.wait()
                rollout_handle.wait()
            
            # ========== 阶段 3：Actor 训练 ==========
            with self.timer("training"):
                actor_handle = self.actor.run_training()
                actor_result = actor_handle.wait()
        
        self.global_step += 1
        
        # ========== 阶段 4：评估与保存 ==========
        run_val, save_model, _ = check_progress(
            self.global_step, self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
        )
        if save_model: self._save_checkpoint()
        if run_val: eval_metrics = self.evaluate()
        
        # ========== 阶段 5：Metrics 记录 ==========
        self._log_metrics(training_metrics, time_metrics, eval_metrics)
```

### weight_sync_interval

```yaml
runner:
  weight_sync_interval: 1   # 每 N 步同步一次权重（默认 1 = 每步都同步）
```

设为 > 1 可以减少同步开销，但引入 off-policy 偏差。

### 评估流程

```python
def evaluate(self):
    # 用 eval Channel 让 Env 和 Rollout 协作跑评估
    env_handle = self.env.evaluate(
        input_channel=self.env_channel,
        rollout_channel=self.rollout_channel,
    )
    rollout_handle = self.rollout.evaluate(
        input_channel=self.rollout_channel,
        output_channel=self.env_channel,
    )
    env_results = env_handle.wait()
    rollout_handle.wait()
    return compute_evaluate_metrics(env_results)
```

评估时使用 `temperature_eval`（通常比训练低，如 0.6），环境使用 `eval` 配置（可能有不同的 `total_num_envs`）。

## AsyncEmbodiedRunner：异步模式

### 与同步模式的核心区别

| | 同步模式 | 异步模式 |
|---|---------|---------|
| 执行方式 | 串行：采集 → 训练 → 采集 → 训练 | 并行：采集和训练同时进行 |
| GPU 利用率 | 低（一半时间在等） | 高（几乎无空闲） |
| 数据新鲜度 | 总是最新策略产的 | 可能有 1-2 步延迟 |
| 代码复杂度 | 低 | 高（需处理 off-policy 修正） |

### 异步主循环

```python
class AsyncEmbodiedRunner(EmbodiedRunner):
    def run(self):
        # 启动长驻 Worker（不会返回）
        env_handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            actor_channel=self.actor_channel,
            metric_channel=self.env_metric_channel,
        )
        rollout_handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            metric_channel=self.rollout_metric_channel,
        )
        actor_handle = self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel
        )
        
        # Actor 训练循环
        while self.global_step < self.max_steps:
            actor_training_handle = self.actor.run_training()
            actor_result = actor_training_handle.wait()
            
            if not actor_result[0]:  # 数据还没到，跳过
                time.sleep(1.0)
                continue
            
            self.global_step += 1
            if self.global_step % self.weight_sync_interval == 0:
                self.update_rollout_weights(no_wait=self.sync_weight_no_wait)
```

### no_wait 权重同步

```python
def update_rollout_weights(self, no_wait=False):
    if not no_wait:
        # 同步方式：等发送完成
        return super().update_rollout_weights()
    
    # 异步方式：发送后不等待
    if not self._cleanup_pending_rollout_weight_sync(no_wait):
        # 上次发送还没完，合并（跳过本次）
        self._weight_sync_coalesced_total += 1
        return
    
    rollout_handle = self.rollout.request_actor_sync_model()
    actor_handle = self.actor.sync_model_to_rollout()
    self._pending_rollout_weight_sync = (rollout_handle, actor_handle)
```

权重同步合并（coalescing）：如果 Actor 训练太快、上一次同步还没完成，就跳过本次同步。这避免了同步堆积。

### 版本控制

异步模式下，Rollout 产出的数据可能来自旧版策略。通过 `versions` 字段追踪：

```python
# Rollout 产出时标记当前版本
versions = torch.full_like(prev_logprobs, float(self.version))

# Actor 使用时，version 用于 importance sampling 修正
# Decoupled PPO 中的 proximal policy 计算
version_diff = current_version - behav_version
alpha = (current_version - 1 - behav_version) / version_diff
proximal_logprobs = old_logprobs + alpha * (logprobs - old_logprobs)
```

## max_steps 计算

```python
def set_max_steps(self):
    self.num_steps_per_epoch = 1  # 一个 epoch = 一个 training step
    self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs
    
    if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
        self.max_steps = min(self.max_steps, max_steps)
```

配置 `max_epochs: 1000` + `max_steps: -1` → 训练 1000 步。

## Checkpoint 管理

```python
def _save_checkpoint(self):
    checkpoint_root = os.path.join(log_path, experiment_name)
    base_output_dir = os.path.join(
        checkpoint_root, f"checkpoints/global_step_{self.global_step}"
    )
    actor_save_path = os.path.join(base_output_dir, "actor")
    os.makedirs(actor_save_path, exist_ok=True)
    self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
```

产出目录结构：
```
results/experiment_name/
├── checkpoints/
│   ├── global_step_40/actor/model.pt
│   ├── global_step_80/actor/model.pt
│   └── ...
├── tensorboard/
└── video/eval/
```

## MetricLogger 支持的后端

```yaml
runner:
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "my_experiment"
    logger_backends: ["tensorboard"]  # 可选: wandb, swanlab, tensorboard
```

## 下一章预告

[第 08 章](./08_算法实现_PPO配置详解) 将进入算法层面，逐参数拆解 PPO 在 RLinf 中的配置和实现。
