---
title: "训练后端：FSDP 与 Megatron"
series:
  id: rlinf_deep_dive
  chapter: 6
order: 6
---

# 训练后端：FSDP 与 Megatron

> 前情提要：上一章追踪了数据从环境到梯度更新的完整路径。本章深入训练后端——Actor Worker 内部如何管理模型、优化器和分布式训练。

## 训练后端选择

```yaml
actor:
  training_backend: "fsdp"      # 或 "megatron"
```

| | FSDP | Megatron |
|---|------|----------|
| 模型加载 | HuggingFace `AutoModel` | Megatron 格式 checkpoint |
| 并行策略 | 纯数据并行（FSDP sharding） | Tensor + Pipeline + Data 并行 |
| 适用模型规模 | < 30B | 任意规模 |
| 配置复杂度 | 低 | 高 |
| 推理后端 | HuggingFace | SGLang / vLLM |
| 典型场景 | 具身 RL（VLA 3B-7B） | 智能体 RL（LLM 70B+） |

本章重点讲 FSDP 后端（具身 RL 的主流选择）。

## FSDPModelManager：核心管理器

`rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` 是 FSDP 后端的核心类。

### 初始化流程

```python
class FSDPModelManager:
    def __init__(self, cfg, world_size, rank):
        # 1. 解析精度
        self.torch_dtype = torch_dtype_from_precision(cfg.model.precision)
        
        # 2. 创建 DeviceMesh（支持 DDP + FSDP 混合）
        self._device_mesh = create_device_mesh(world_size, cfg.fsdp_config.get("fsdp_size", -1))
        
        # 3. 创建 FSDP Strategy（工厂方法）
        self._strategy = FSDPStrategyBase.create(cfg, world_size, dp_group, logger)
        
        # 4. 创建 AMP context
        self.amp_context = self._create_amp_context()
```

### FSDP Strategy 工厂

```python
class FSDPStrategyBase:
    @classmethod
    def create(cls, cfg, world_size, dp_group, logger):
        strategy = cfg.fsdp_config.get("strategy", "fsdp2").lower()
        match strategy:
            case "fsdp":   return FSDPStrategy(...)   # PyTorch FSDP1
            case "fsdp2":  return FSDP2Strategy(...)  # PyTorch FSDP2 (fully_shard)
```

两个策略的区别：

| | FSDP1 | FSDP2 |
|---|-------|-------|
| API | `torch.distributed.fsdp.FullyShardedDataParallel` | `torch.distributed.fsdp2.fully_shard()` |
| 包装方式 | 包装整个模型为一个 FSDP 模块 | 对每层调用 `fully_shard()` |
| 灵活性 | 较低 | 高（可以混合 FSDP + DDP） |
| 默认选择 | 旧版兼容 | RLinf 推荐 |

### setup_model_and_optimizer()

Actor Worker 初始化时调用：

```python
def setup_model_and_optimizer(self):
    # 1. 加载模型
    model = self.model_provider_func()
    
    # 2. FSDP wrap
    self.model = self._strategy.wrap_model(model, self._device_mesh)
    
    # 3. 创建优化器
    self.optimizer = self._create_optimizer()
    
    # 4. 创建 LR scheduler
    self.lr_scheduler = get_lr_scheduler(self.optimizer, self._cfg)
    
    # 5. 可选：GradScaler (fp16)
    if self._cfg.fsdp_config.grad_scaler.enabled:
        self.grad_scaler = ShardedGradScaler(...)
```

## FSDP 配置详解

```yaml
actor:
  fsdp_config:
    strategy: "fsdp"               # "fsdp" 或 "fsdp2"
    sharding_strategy: "full_shard" # full_shard / shard_grad_op / no_shard
    gradient_checkpointing: False   # 是否使用梯度检查点（省显存）
    forward_prefetch: False         # 前向预取下一层参数
    limit_all_gathers: False        # 限制 all-gather 并发数
    backward_prefetch: null         # 反向预取："pre" 或 "post"
    use_orig_params: False          # 使用原始参数引用
    use_liger_kernel: False         # 使用 Liger Kernel 优化
    cpu_offload: False              # 参数 CPU offload
    reshard_after_forward: True     # 前向后重新 shard
    
    mixed_precision:
      param_dtype: "bf16"           # 参数精度
      reduce_dtype: "bf16"          # 通信精度
      buffer_dtype: "bf16"          # buffer 精度
    
    amp_autocast:
      enabled: False                # AMP 自动混合精度
      precision: "bf16"
    
    grad_scaler:
      enabled: False                # 梯度缩放（fp16 用）
```

### sharding_strategy 对比

| 策略 | 行为 | 显存占用 | 通信量 |
|------|------|---------|--------|
| `full_shard` | 参数、梯度、优化器状态全部 shard | 最低 | 最高（allgather + reduce-scatter） |
| `shard_grad_op` | 只 shard 梯度和优化器 | 中等 | 中等 |
| `no_shard` | 不 shard（等价于 DDP） | 最高 | 最低（只 allreduce 梯度） |

**具身 RL 实践**：大部分 VLA 模型（3B-7B）在单机 8 卡时用 `no_shard` 就够了（显存足够），配合 `gradient_checkpointing: True` 进一步省显存。

## 微批量训练流程

Actor Worker 的 `run_training()` 内部：

```python
def run_training(self):
    # 1. 计算 advantages
    advantages, returns = calculate_adv_and_returns(...)
    
    # 2. 多 epoch 更新
    for epoch in range(update_epoch):
        # 3. 打乱数据
        shuffle_ids = torch.randperm(total_samples)
        
        # 4. 切分为 micro-batch
        micro_batches = get_iterator_k_split(batch, n_mini_batches)
        
        for i, micro_batch in enumerate(micro_batches):
            # 5. 梯度累积控制
            is_last_micro_batch = (i == n_mini_batches - 1)
            sync_context = self._strategy.before_micro_batch(
                model, is_last_micro_batch
            )
            
            with sync_context:
                with self.amp_context:
                    # 6. 前向：计算 logprobs
                    logprobs = self.model.forward(micro_batch)
                    
                    # 7. 计算 loss
                    loss, metrics = policy_loss(
                        logprobs=logprobs,
                        old_logprobs=micro_batch["prev_logprobs"],
                        advantages=micro_batch["advantages"],
                        clip_ratio_high=...,
                        clip_ratio_low=...,
                    )
                
                # 8. 反向传播
                loss.backward()
        
        # 9. 梯度裁剪 + 优化器步进
        grad_norm = self._strategy.clip_grad_norm_(model, clip_grad)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
```

### before_micro_batch() 的作用

```python
def before_micro_batch(self, model, is_last_micro_batch):
    if is_last_micro_batch:
        return nullcontext()  # 最后一个 micro-batch：正常同步梯度
    else:
        return model.no_sync()  # 非最后一个：禁用梯度同步，累积梯度
```

这避免了每个 micro-batch 都做 allreduce，只在最后一个做一次。

## Offload 机制

### 参数 Offload

```python
def offload_param_and_grad(self):
    """把模型参数和梯度搬到 CPU，释放 GPU 显存"""
    for param in self.model.parameters():
        param.data = param.data.to("cpu")
        if param.grad is not None:
            param.grad = param.grad.to("cpu")
    self.is_weight_offloaded = True

def load_param_and_grad(self, device):
    """把模型参数搬回 GPU"""
    for param in self.model.parameters():
        param.data = param.data.to(device)
    self.is_weight_offloaded = False
```

### 优化器 Offload

```python
def offload_optimizer(self):
    """把优化器状态搬到 CPU"""
    for state in self.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()
    self.is_optimizer_offloaded = True
```

### Offload 时序

```
初始化：加载模型 → FSDP wrap → 创建优化器 → offload 全部到 CPU

训练时：
  load_param_and_grad(GPU)    ← 搬回
  load_optimizer(GPU)          ← 搬回
  ... 训练 step ...
  offload_optimizer()          ← 搬走
  offload_param_and_grad()     ← 搬走

权重同步时：
  load_param_and_grad(GPU)    ← 搬回
  get_model_state_dict()       ← 提取
  send() 给 Rollout            ← 发送
  offload_param_and_grad()     ← 搬走
```

## Checkpoint 保存与恢复

### 保存

```python
def save_checkpoint(self, save_path, global_step):
    state_dict = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
    torch.save(state_dict, os.path.join(save_path, "model.pt"))
    # 可选：保存优化器状态
```

### 恢复

```yaml
runner:
  resume_dir: "checkpoints/global_step_100"  # 从此目录恢复
```

恢复时 `init_workers()` 会在模型加载后调用 `load_checkpoint()`。

## LoRA 支持

```yaml
actor:
  model:
    is_lora: True              # 启用 LoRA
    lora_r: 16                 # LoRA 秩
    lora_alpha: 32             # LoRA alpha
    lora_target_modules: ["q_proj", "v_proj"]
```

启用 LoRA 后，只有 LoRA 参数参与训练和权重同步，大幅减少通信量和显存占用。

## 下一章预告

[第 07 章](./07_Runner训练循环) 将详解 Runner 的训练循环——同步模式和异步模式的完整执行流程、评估、Checkpoint、metrics 记录。
