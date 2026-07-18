---
title: "训练后端：FSDP 与 Megatron"
series:
  id: rlinf_deep_dive
  chapter: 6
order: 6
---

# 训练后端：FSDP 与 Megatron

> 前情提要：上一章追踪了数据从环境到梯度更新的完整路径。本章深入训练后端——Actor Worker 内部到底怎么管理模型、优化器和分布式训练。

> 如果你对"多卡训练时梯度怎么对齐"还没有直觉，建议先读 [数据并行与 AllReduce 基础](/前置知识/001h_前置知识_数据并行与AllReduce基础)、[FSDP：全分片数据并行](/前置知识/001i_前置知识_FSDP全分片数据并行)、[张量并行与流水线并行：Megatron 核心思想](/前置知识/001j_前置知识_张量并行与流水线并行_Megatron核心思想) 这三篇前置知识——本章会大量引用它们里面的概念（shard、All-Gather、Reduce-Scatter、no_sync 等），不重复讲原理，只讲 RLinf 怎么把这些原理落地成代码。

## 一、为什么需要两套训练后端

RLinf 的 Actor Worker 负责训练策略网络，但"训练"这件事在小模型和大模型上完全不是一个量级的工程问题：

- VLA 模型通常是 3B~7B 参数，用 [FSDP](/前置知识/001i_前置知识_FSDP全分片数据并行) 分片一下，单机 8 卡就能训得动，配置也简单——按前置知识里的说法，这属于"纯数据并行方向"的显存优化，不需要动模型内部结构。
- LLM Agent 类任务的模型可能是 70B 甚至更大，一层的参数就大到一张卡塞不下，这时候单靠 FSDP 不够，需要 [张量并行和流水线并行](/前置知识/001j_前置知识_张量并行与流水线并行_Megatron核心思想) 把层内计算、层间计算都拆开——这正是 Megatron-LM 的强项。

所以 RLinf 提供了两套可切换的训练后端，通过一行配置决定用哪个：

```yaml
actor:
  training_backend: "fsdp"      # 或 "megatron"
```

两者的定位差异：

| | FSDP | Megatron |
|---|------|----------|
| 模型加载 | HuggingFace `AutoModel` | Megatron 格式 checkpoint |
| 并行策略 | 数据并行 + FSDP 分片 | 张量并行 + 流水线并行 + 数据并行 |
| 适用模型规模 | < 30B | 任意规模，尤其是 70B+ |
| 配置复杂度 | 低 | 高（需要指定 TP/PP size） |
| 推理后端 | HuggingFace | SGLang / vLLM |
| 典型场景 | 具身 RL（VLA 3B-7B） | 智能体 RL（LLM 70B+） |

具身 RL 场景几乎全部使用 FSDP（RLinf 里所有 VLA 示例配置的 `training_backend` 都是 `"fsdp"`），Megatron 主要留给超大模型的 LLM Agent 场景（比如代码生成的在线 RL 任务，RLinf 甚至在配置校验里强制要求这类任务必须用 Megatron）。本章重点讲 FSDP 后端，最后一节简要对比 Megatron 后端的关键差异。

## 二、FSDPModelManager：把"分片"这件事管起来

在 [FSDP 前置知识](/前置知识/001i_前置知识_FSDP全分片数据并行) 里我们讲过，FSDP 的核心动作是"用哪种 shard 策略切、什么时候拼、拼完什么时候还回去"。这些动作不是散落在训练代码各处的，RLinf 把它们全部收进一个类：`rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` 里的 `FSDPModelManager`。可以把它理解成"FSDP 相关操作的总管家"——加载模型、包装分片、创建优化器、做 offload、存/读 checkpoint，全部通过这个类完成，Actor Worker 只需要调用它提供的接口，不用关心底层 PyTorch FSDP API 的细节。

### 2.1 初始化时要确定的三件事

`FSDPModelManager.__init__` 在真正加载模型之前，需要先把"怎么分片"这件事定下来。这里涉及两个问题：分片粒度是什么（DeviceMesh），用哪个版本的 FSDP API（Strategy）。

**DeviceMesh：决定"卡怎么分组"**。前置知识里提到过，FSDP 和普通 DDP 可以混合使用——比如 16 张卡，可以分成 4 组，组内 4 张卡做 FSDP 分片，4 组之间做普通数据并行。这个"分组方式"由 `create_device_mesh` 决定：

```python
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 所有卡都参与 FSDP 分片，不额外分组
        device_mesh = init_device_mesh(
            device_type, mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
    else:
        # 分成 (world_size // fsdp_size) 组，组内 fsdp_size 张卡做分片
        device_mesh = init_device_mesh(
            device_type,
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"],
        )
    return device_mesh
```

`fsdp_size` 默认是 `-1`，也就是最常见的情况：全部卡都参与分片，不做额外的 DDP 分组。只有当模型不大、卡数很多时，才会考虑把卡分组（比如 32 张卡，模型本身 4 卡就能分片装下，剩下的用普通 DDP 复制），这样能减少不必要的 All-Gather 通信范围。

**Strategy：决定"用 FSDP1 还是 FSDP2"**。PyTorch 的 FSDP 有新旧两代 API，RLinf 用一个工厂方法根据配置选择：

```python
class FSDPStrategyBase:
    @classmethod
    def create(cls, cfg, world_size, dp_group, logger):
        strategy = cfg.fsdp_config.get("strategy", "fsdp2").lower()
        match strategy:
            case "fsdp":   return FSDPStrategy(...)   # 老版 FullyShardedDataParallel
            case "fsdp2":  return FSDP2Strategy(...)  # 新版 fully_shard()
```

两者的区别，本质上对应前置知识里"以子模块为单位切分"的两种实现方式：

| | FSDP1 | FSDP2 |
|---|-------|-------|
| API | `FullyShardedDataParallel`（包装整个模型） | `fully_shard()`（逐层调用） |
| 灵活性 | 较低 | 较高，可以混合 FSDP + DDP |
| RLinf 默认选择 | 旧版兼容 | 推荐 |

选定 Strategy 之后，`FSDPModelManager` 还会准备好 AMP（自动混合精度）的上下文管理器——如果配置里没开启 AMP，就用一个空的 `nullcontext()`，训练代码不用为"是否开 AMP"写两套分支。

### 2.2 setup_model_and_optimizer：从"裸模型"到"能训练的模型"

模型真正被创建、分片、配好优化器，是在 Actor Worker 初始化时调用的 `setup_model_and_optimizer()`。这一步要做四件事，顺序是固定的：先加载原始的 HuggingFace 模型，再决定是否开梯度检查点，然后按 2.1 节确定的 Strategy 把模型包装成分片版本，最后基于分片后的模型创建优化器和学习率调度器——顺序不能反，因为优化器必须管理的是"分片后"的参数（每张卡上的那一部分），而不是完整模型的参数。

```python
def setup_model_and_optimizer(self) -> None:
    module = self.model_provider_func()          # 1. 加载 HuggingFace 模型

    if self._cfg.fsdp_config.get("gradient_checkpointing", False):
        module.gradient_checkpointing_enable()    # 2. 可选：梯度检查点省显存

    self.model = self._strategy.wrap_model(        # 3. FSDP/FSDP2 分片包装
        model=module, device_mesh=self._device_mesh
    )
    self.optimizer = self.build_optimizer(self.model)  # 4. 基于分片后的模型建优化器
    self.lr_scheduler = self.build_lr_scheduler(self.optimizer, self._cfg.optim)
```

其中 `wrap_model` 是真正调用 PyTorch FSDP API 的地方，我们在下一节详细看它做了什么。

## 三、wrap_model：把前置知识里的"切分"落到代码上

前置知识提到，FSDP2 用 `fully_shard()` 逐层包装，Column/Row 那种切分方式在这里不涉及（那是张量并行的事），FSDP 只做"参数分片存储"这一件事。RLinf 里 FSDP2 的 `wrap_model` 实现：

```python
def wrap_model(self, model, device_mesh):
    # 混合精度策略：对应前置知识里的 param_dtype / reduce_dtype
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch_dtype_from_precision(mp_cfg.param_dtype),
        reduce_dtype=torch_dtype_from_precision(mp_cfg.reduce_dtype),
        cast_forward_inputs=True,
    )

    offload_policy = (
        CPUOffloadPolicy(pin_memory=cfg.offload_pin_memory)
        if cfg.cpu_offload else OffloadPolicy()
    )

    # 关键一步：只对 Transformer Layer 逐层调用 fully_shard，
    # 而不是把整个模型当成一个黑盒切分
    fsdp2_model = apply_fsdp2_to_model(
        module=model, config=cfg, device_mesh=device_mesh,
        mp_policy=mp_policy, offload_policy=offload_policy,
        reshard_after_forward=cfg.reshard_after_forward,
    )
    return fsdp2_model
```

`apply_fsdp2_to_model` 内部会先找出模型里"哪些子模块要按层切"（通常是 Transformer 的 `DecoderLayer` 类，靠模型自带的 `_no_split_modules` 属性识别），对每一个这样的子模块单独调用 `fully_shard()`，最后再对整个模型调用一次 `fully_shard()`（这次 `reshard_after_forward` 强制设为 `False`，原因见 [前置知识 4 节](/前置知识/001i_前置知识_FSDP全分片数据并行#四-reshard-after-forward-一个关键的权衡开关)：最外层反向传播马上就要用到，没必要立刻释放）。

```python
for name, submodule in module.named_modules():
    if submodule.__class__.__name__ in transformer_layer_cls_to_wrap:
        fully_shard(submodule, mesh=device_mesh, mp_policy=mp_policy, ...)

# 最外层整体再包一次，reshard_after_forward 固定为 False
return fully_shard(module, mesh=device_mesh, mp_policy=mp_policy,
                    offload_policy=offload_policy, reshard_after_forward=False)
```

这就是"以层为单位切分"在真实代码里的样子——不是切开整个模型的参数向量，而是一层一层地决定"这一层该不该分片、分片后前向完要不要立刻释放"。

## 四、sharding_strategy：FSDP1 独有的额外选项

FSDP2 用 `reshard_after_forward` 这个二元开关控制切分粒度，而老版 FSDP1 提供了更细的 `sharding_strategy` 配置，直接对应前置知识里讲的三档选择：

```yaml
actor:
  fsdp_config:
    sharding_strategy: "full_shard"  # full_shard / shard_grad_op / no_shard
```

| 策略 | 行为 | 显存占用 | 通信量 |
|------|------|---------|--------|
| `full_shard` | 参数、梯度、优化器状态全部分片 | 最低 | 最高 |
| `shard_grad_op` | 参数常驻完整，只分片梯度和优化器状态 | 中等 | 中等 |
| `no_shard` | 不分片，退化为普通 DDP | 最高 | 最低 |

**具身 RL 的实践经验**：大部分 VLA 模型（3B-7B）在单机 8 卡时显存是够用的,直接用 `no_shard`（等价于普通 DDP，省去 All-Gather 的通信开销）配合 `gradient_checkpointing: True`（省显存）就足够了——这正好呼应了前置知识里提到的"如果模型本身不大，切分反而是多余开销"的结论。只有显存紧张（比如同时跑 SAC 的双 Q 网络，或者用更大的 GR00T 模型）才会切到 `full_shard`。

## 五、微批量训练：梯度累积与 before_micro_batch

Actor Worker 的训练循环要处理一个常见约束：想要的 `global_batch_size` 往往比单卡显存能承受的 batch 更大，只能拆成多个 micro-batch 依次算，梯度在本地累积，最后统一更新一次参数。这正是 [前置知识里 no_sync() 的应用场景](/前置知识/001h_前置知识_数据并行与AllReduce基础#41-no-sync-暂停梯度同步)：非最后一个 micro-batch 不应该触发梯度同步，否则通信次数会白白增加几倍。

RLinf 用 `before_micro_batch` 这个上下文管理器统一处理这件事,FSDP1 和 FSDP2 的底层机制不同(`no_sync()` vs `set_requires_gradient_sync`),但接口一致,训练代码不需要关心用的是哪个版本:

```python
# FSDP2 版本的实现
def before_micro_batch(self, model, is_last_micro_batch):
    if not self.cfg.fsdp_config.enable_gradient_accumulation:
        return nullcontext()
    if is_last_micro_batch:
        model.set_requires_gradient_sync(True)   # 最后一个:开启同步
    else:
        model.set_requires_gradient_sync(False)  # 非最后一个:暂停同步,只本地累积
    return nullcontext()
```

真实的训练循环（`EmbodiedFSDPActor.run_training`，简化后）就是把这个上下文管理器套在每个 micro-batch 的前向+反向外面：

```python
self.optimizer.zero_grad()
for idx, batch in enumerate(train_micro_batch):
    backward_ctx = self.before_micro_batch(
        self.model, is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
    )
    with self.amp_context:
        output_dict = self.model(forward_inputs=batch["forward_inputs"], ...)
    loss, metrics = policy_loss(logprobs=output_dict["log_probs"], ...)
    loss.backward()

# 循环结束后统一做一次梯度裁剪 + 参数更新
grad_norm = self._strategy.clip_grad_norm_(self.model)
self.optimizer.step()
self.lr_scheduler.step()
```

注意梯度裁剪（`clip_grad_norm_`）和参数更新都是在所有 micro-batch 处理完之后才做一次——这正好对应前置知识里"只在最后一步做一次 Reduce-Scatter"的设计,避免了每个 micro-batch 都触发一次全量通信。

## 六、Offload：训练用完就把显存还回去

具身 RL 常见的部署方式是 Actor（训练）和 Rollout（推理）共享同一批 GPU（[第 03 章](./03_Scheduler调度系统#共享-gpu-场景)提到过这一点）。既然两个角色轮流用 GPU，闲下来的那个就应该把显存让出来——这是 Offload 机制存在的原因。

`FSDPModelManager` 提供了参数/梯度和优化器状态两组独立的 Offload 接口,原因是这两类数据的搬运时机往往不同(比如同步权重给 Rollout 时只需要搬回参数,不需要搬优化器状态):

```python
def offload_param_and_grad(self, offload_grad=False):
    """把模型参数（以及可选的梯度）搬到 CPU"""
    self._strategy.offload_param_and_grad(self.model, offload_grad)
    self.is_weight_offloaded = True

def offload_optimizer(self):
    """把优化器状态（如 Adam 的动量）搬到 CPU"""
    self._strategy.offload_optimizer(self.optimizer)
    self.is_optimizer_offloaded = True
```

对应的搬回 GPU 操作是 `load_param_and_grad` / `load_optimizer`，逻辑对称，不再重复贴代码。真正需要注意的是**什么时候搬、什么时候还**——这决定了 GPU 显存在训练全程的占用曲线：

```
初始化：加载模型 → FSDP 分片包装 → 创建优化器 → 全部 offload 到 CPU（GPU 空出来给 Rollout 用）

每次训练前：
  load_param_and_grad(GPU)   ← 训练要用，先搬回来
  load_optimizer(GPU)         ← 训练要用，先搬回来
  ... 执行若干个 training step ...
  offload_optimizer()         ← 训完立刻搬走，让出显存
  offload_param_and_grad()    ← 训完立刻搬走

同步权重给 Rollout 时：
  load_param_and_grad(GPU)    ← 只需要参数，不需要优化器状态
  get_model_state_dict()       ← 从分片状态还原出完整权重
  send() 发给 Rollout Worker
  offload_param_and_grad()     ← 发完立刻搬走
```

可以看到一个规律：**任何时候，只要不是"正在被用"，就应该待在 CPU**。Actor 训练时 Rollout 的模型在 CPU，Rollout 推理时 Actor 的模型在 CPU，两者从不同时占用完整的 GPU 显存——这就是共享 GPU 部署能跑起来的关键。

## 七、Checkpoint：分片存 vs 完整存

保存 checkpoint 时会遇到一个和前面完全一样的问题：每张卡手里只有分片后的那一部分参数，怎么存？RLinf 提供两种存法，分别对应不同的使用场景：

- **分片存（DCP，Distributed Checkpoint）**：每张卡只存自己手里的那部分 shard，速度快、不需要临时拼出完整模型。缺点是恢复训练时也必须用完全相同的卡数和分片方式加载,不能跨卡数直接用。
- **完整存（full_state_dict）**：先用 All-Gather 把所有卡的 shard 拼成完整的模型权重，只让 rank 0 存一份完整的 `.pt` 文件。这份文件可以脱离分布式环境直接用（比如给 Rollout Worker 单卡加载,或者上传做后续分析）。

```python
def save_checkpoint(self, save_path, step=0):
    # 保存前先确保参数/优化器都在 GPU 上（可能之前被 offload 走了）
    if self.is_weight_offloaded:
        self.load_param_and_grad(self.device)
    if self.is_optimizer_offloaded:
        self.load_optimizer(self.device)
    self._strategy.save_checkpoint(self.model, self.optimizer, self.lr_scheduler, save_path)
```

`_strategy.save_checkpoint` 内部默认两种都存：先用 `torch.distributed.checkpoint` 存分片版本（用于快速恢复训练），再额外汇总一份完整权重文件（用于分享或单独加载）。这是"训练可续跑"和"权重可复用"两个需求的折中方案——只存一种都会牺牲另一种场景。

## 八、LoRA：只同步、只训练一小部分参数

如果开启 LoRA 微调（配置里 `is_lora: True`），FSDP 的分片对象和训练目标都会发生变化——只有 LoRA 注入的低秩矩阵会被标记为可训练，其余原始权重全部冻结：

```yaml
actor:
  model:
    is_lora: True
    lora_r: 16
    lora_alpha: 32
    lora_target_modules: ["q_proj", "v_proj"]
```

这对 FSDP 分片策略也有影响：`get_fsdp_wrap_policy` 里专门有一段逻辑，把满足条件（叶子模块、有 `weight` 且 `requires_grad=True`）的 LoRA 参数单独用 `lambda_auto_wrap_policy` 分一组。效果是训练时反向传播只需要给这一小部分参数算梯度、只在权重同步时传输这一小部分参数——对应到前面 All-Gather/Reduce-Scatter 的通信量,直接从"完整模型大小"降到"LoRA 参数大小"（通常不到原模型的 1%），这也是为什么大模型 + 少量显存的场景经常搭配 LoRA 一起用。

## 九、Megatron 后端：当模型大到 FSDP 也扛不住

FSDP 解决的始终是"数据并行方向"的显存冗余问题——不管怎么分片,单层的完整参数在计算那一刻还是要临时拼在一张卡上。如果模型大到连"一层"都塞不进一张卡（比如 70B 参数的 LLM，一个 Transformer 层就有几十亿参数），就必须用 [张量并行和流水线并行](/前置知识/001j_前置知识_张量并行与流水线并行_Megatron核心思想) 把层内、层间的计算都切开——这正是 Megatron 后端存在的原因。

Megatron 后端在 RLinf 里的配置维度明显更多，需要显式指定切分方式：

```yaml
actor:
  training_backend: "megatron"
  model:
    tensor_model_parallel_size: 4     # 层内切分成几份（对应前置知识的 TP）
    pipeline_model_parallel_size: 2   # 模型按层切成几段（对应前置知识的 PP）
    context_parallel_size: 1
```

对应的 Worker 实现是 `MegatronActor`（`rlinf/workers/actor/megatron_actor_worker.py`），和 `FSDPActor` 相比最核心的差异是前向传播的写法。FSDP 里模型前向就是普通的 `self.model(...)` 调用；Megatron 里因为模型被切成了多段（流水线并行）、每层内部也被切开（张量并行），前向传播必须用 Megatron-Core 提供的调度器来驱动，写成一个"给定一个 batch，怎么算出 loss"的回调函数：

```python
def get_forward_step_func(self):
    def forward_output_and_loss_func(dataloader_iter, model):
        batch = next(dataloader_iter)
        # custom_forward 内部会根据 tensor_model_parallel_size 自动处理
        # 词表切分、序列并行等细节，训练代码不需要手写通信逻辑
        output = self.custom_forward(model, batch["input_ids"], ...)

        def loss_func(output):
            loss, metrics = policy_loss(logprobs=output["log_probs"], ...)
            return loss, metrics
        return output, loss_func
    return forward_output_and_loss_func
```

这个回调会被 Megatron-Core 的流水线调度器反复调用——调度器负责把 micro-batch 依次送入流水线的不同 stage，处理前面提到的"流水线气泡"问题，这部分逻辑完全由 Megatron-Core 库管好，RLinf 只需要提供"怎么算 loss"这一个函数。

权重同步给 Rollout 时也多了一步 RLinf 自己实现的**resharding**（重新分片）：Actor 端可能用 TP=4 训练，但 Rollout 端推理用 TP=2 更高效，两边的切分方式不一致，直接传输 shard 会对不上。`MegatronCoreWeightReshard` 负责把 Actor 的分片重新组合、按 Rollout 需要的方式再切一遍，再发送出去——这是 FSDP 后端不需要处理的额外复杂度（FSDP 端到端只有一种切分方式），也是"配置复杂度：高"这一栏的具体体现。

## 十、总结

| 环节 | FSDP 做法 | 对应前置知识 |
|------|----------|------------|
| 初始化 | DeviceMesh 决定分组 + Strategy 决定 FSDP1/FSDP2 | 数据并行分组 |
| 分片粒度 | 逐个 Transformer Layer 调用 `fully_shard()` | FSDP 以层为单位切分 |
| 训练循环 | micro-batch 累积梯度，`before_micro_batch` 控制同步时机 | no_sync() / 梯度累积 |
| 显存管理 | Offload 参数/梯度/优化器状态到 CPU，用时才搬回 | —— |
| Checkpoint | 分片存（快速续训）+ 完整存（可移植） | 分片存储 |
| 大模型场景 | 切换到 Megatron，用 TP+PP 处理单层过大的问题 | 张量并行 / 流水线并行 |

## 下一章预告

[第 07 章](./07_Runner训练循环) 将详解 Runner 的训练循环——同步模式和异步模式的完整执行流程、评估、Checkpoint、metrics 记录。
