---
title: "Worker 体系：从 EnvWorker 到 FSDP 训练引擎"
series:
  id: verl_vla_deep_dive
  chapter: 4
order: 4
---

# 第 04 章 Worker 体系：从 EnvWorker 到 FSDP 训练引擎

> 前情提要：第 03 章讲了 EnvLoop 怎么让模型推理和环境交互重叠执行。这一章下探到 Worker 层——EnvWorker 怎么隔离仿真器、Actor/Rollout Worker 怎么用同一个类支持两种部署形态、FSDP 引擎怎么在训练态和推理态之间切换权重。

## 知识链接

- 上一章：[EnvLoop 流水线：异步 rollout 与部署形态](./03_EnvLoop流水线_异步rollout与部署形态)
- 下一章：[模型集成契约：三个接口统一 ACT / Pi0.5 / GR00T](./05_模型集成契约_三个接口统一ACT_Pi0_GR00T)
- [系列目录](./index)
- 如果对 FSDP（全分片数据并行）不熟悉，建议先读 [FSDP 全分片数据并行](/前置知识/001i_前置知识_FSDP全分片数据并行)

---

## 1. EnvWorker：按字符串分派构造仿真器，仿真器跑在独立进程里

`EnvWorker.init_worker()` 是一段朴素的 if/elif 分派，按配置里的 `simulator_type` 字符串选择对应的环境类：

```python
def init_worker(self):
    if self.simulator_type == "libero":
        from verl_vla.envs.libero.libero_env import LiberoEnv
        self.simulator_list.append(EnvManager(self.simulator_cfg, ..., env_cls=LiberoEnv, stage_id=stage_id, ...))
    elif self.simulator_type == "arena":
        from verl_vla.envs.arena.arena_env import IsaacLabArenaEnv
        ...
    elif self.simulator_type == "piper":
        from verl_vla.envs.piper.piper_env import PiperEnv
        ...
    else:
        raise NotImplementedError(f"Simulator type {self.simulator_type} not implemented")

    for simulator in self.simulator_list:
        simulator.start_simulator()
```

**为什么不用注册表模式**：模型集成层（第 05 章会看到）用的是显式注册表，而环境集成这里用的是最朴素的 if/elif。这不是疏漏，而是因为环境种类少（目前 5 种）且每种都需要单独的懒加载导入（避免没装 Isaac Sim 的环境也要求装 Isaac 依赖），显式分支反而更直观。加一种新仿真器只需要加一个分支，同时要在 `SimulatorConfig` 里加类型校验。

`stage_num`（来自上一章的 `pipeline_stage_num`）决定每个 EnvWorker 内部起几个 `EnvManager` 实例——这是流水线并行的物理落地点。

### `EnvManager`：一个跨进程的"透明"仿真器代理

真正跑仿真器的地方不是 `EnvWorker` 所在的进程，而是 `EnvManager` 用 `torch.multiprocessing.spawn` 起的**独立子进程**：

```python
def start_simulator(self):
    self.context = mp.get_context("spawn")
    self.command_queue = self.context.Queue()
    self.result_queue = self.context.Queue()
    self.process = self.context.Process(
        target=_simulator_worker,
        args=(self.cfg, self.rank, self.world_size, self.stage_id, self.stage_num, self.env_cls, ...),
    )
    self.process.start()
```

方法调用通过 `__getattr__` 拦截，动态转发成跨进程消息：

```python
def __getattr__(self, name):
    def method_proxy(*args, **kwargs):
        self.command_queue.put({"method": name, "args": args, "kwargs": kwargs})
        result = self.result_queue.get()
        if result["status"] == "error":
            raise Exception(result["error"])
        return result["data"]
    return method_proxy
```

**这段代码在做什么**：调用方（`EnvWorker`）眼里 `EnvManager` 就是一个普通对象，调它任何方法（比如 `simulator.step(actions)`）实际发生的是——把方法名和参数打包成一条消息扔进队列，子进程收到消息后反射调用真正的环境实例，把结果通过另一个队列传回来。对调用方而言完全透明。

**为什么要这么设计**：很多物理仿真器（Isaac Sim、MuJoCo 等）有全局单例、独占的 GL/渲染上下文，和 PyTorch/CUDA 的上下文管理很容易冲突；仿真器崩溃时独立进程可以被单独重启而不拖垮整个训练进程。代码里还能看到 `set_process_numa_affinity`——给每个 GPU rank 对应的仿真子进程绑定同 NUMA node 的 CPU，这是大规模并行仿真常见的性能优化。

## 2. VLAActorRolloutRefWorker：一个类同时扮演训练和推理角色

`workers/engine/engine_workers.py::VLAActorRolloutRefWorker` 是本章最关键的类——它证明了"训练"和"推理"在代码层面可以是**同一个 worker 的两种能力**，而不是两种进程实现：

```python
class VLAActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config, role: str, **kwargs):
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
```

`role` 字符串决定这个 worker 实例到底具备哪些能力。两个瘦子类固定了角色，用于 disaggregated 部署：

```python
class VLAActorWorker(VLAActorRolloutRefWorker):
    def __init__(self, config, role=None):
        super().__init__(config, role="actor")

class VLARolloutWorker(VLAActorRolloutRefWorker):
    def __init__(self, config, role=None):
        super().__init__(config, role="rollout")
```

`init_model()` 里有一处代码是权重共享/独立的分岔点：

```python
if "rollout" in self.role:
    self.rollout = rollout_cls(
        config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh,
        engine=self.actor.engine if "actor" in self.role else None,   # 关键分支
        ...
    )
```

- `role="actor_rollout"`（colocate）：`self.actor` 已经构建好，`engine=self.actor.engine` 被传给 rollout 构造器。rollout 内部（`HFRollout.__init__`）会做 `self.module = engine.module`——**rollout 和 actor 直接共享同一个 `nn.Module` 对象的内存**，权重更新即时可见，完全不需要任何同步操作。
- `role="rollout"`（disaggregated，独立的 `VLARolloutWorker`）：`"actor" in self.role` 为假，`engine=None`，`HFRollout` 走独立分支，用 `build_vla_model` 单独加载一份权重，此后靠 checkpoint engine 异步同步。

这正是第 02、03 章反复提到的"colocate 权重零同步开销、disaggregated 需要显式广播"在代码里的具体落点。

`switch_to_rollout()`/`switch_to_train()` 是 colocate 场景专用的接口，本质是转发到 FSDP 引擎：

```python
def _require_fsdp_rollout_engine(self) -> VLAFSDPEngine:
    if self.config.actor.strategy not in {"fsdp", "fsdp2"}:
        raise RuntimeError("switch_to_rollout/switch_to_train are only supported when actor.strategy is fsdp or fsdp2.")
    return self.actor.engine

def switch_to_rollout(self):
    self._require_fsdp_rollout_engine().switch_to_rollout()

def switch_to_train(self):
    self._require_fsdp_rollout_engine().switch_to_train()
```

`update_weights()` 根据角色三分支路由（推理端接收权重 / 训练端发送权重 / colocate 走父类默认逻辑），对应 disaggregated 部署下真正的权重广播实现：

```python
async def update_weights(self, global_steps=None):
    if self._is_rollout and not self._is_actor:
        weights = self.checkpoint_engine.receive_weights()
        await self.rollout.update_weights(weights, global_steps=global_steps)
    elif self._is_actor and not self._is_rollout:
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)
    else:
        await super().update_weights(global_steps=global_steps)
```

## 3. VLAFSDPEngine：FSDP2 权重切分与训练/推理态切换

`workers/engine/fsdp/vla_impl.py::VLAFSDPEngine` 继承 verl 通用的 `FSDPEngine`，注册为 `model_type="vla_model"` 的后端。核心职责一：构造模型时绕开 Transformers 的 AutoModel 机制：

```python
@EngineRegistry.register(model_type="vla_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class VLAFSDPEngine(FSDPEngine):
    def _build_module(self):
        from verl_vla.models import build_vla_model
        module = build_vla_model(self.model_config, torch_dtype=...)
        module.to(torch_dtype)
        return module
```

`build_vla_model` 就是第 05 章要讲的模型集成分派入口——这里只是它的调用方之一。

核心职责二：FSDP2 的 wrap 目标选择。这里有个不那么显然的坑——通用 FSDP 包装策略会自动把 `nn.Embedding`/`lm_head` 之类的层单独包一层，但很多原生 VLA 策略（比如 LeRobot 的 ACT）代码里直接用 `.weight` 属性访问 embedding 权重而不走 `forward()`，通用包装会导致 DTensor（FSDP 分片张量）和普通 Tensor 混用报错：

```python
def _select_fsdp2_wrap_targets(model, transformer_layer_cls_to_wrap):
    # 特意排除通用的 nn.Embedding/lm_head 目标
    return [module for module in model.modules() if module.__class__.__name__ in transformer_layer_cls_to_wrap]
```

核心职责三：`switch_to_rollout()`/`switch_to_train()` 的具体实现——在同一份 FSDP 分片权重上切换物理布局：

```python
def switch_to_rollout(self):
    self._rollout_eval_ctx = self.eval_mode()
    self._rollout_eval_ctx.__enter__()
    aggressive_empty_cache(force_sync=True)
    self._rollout_rng_state = get_torch_device().get_rng_state()   # 保存训练态的随机数状态
    get_torch_device().set_rng_state(self._train_rng_state)

    if fsdp_version(self.module) == 2:
        self.module.unshard()                          # 把分片权重聚合成完整权重
        for module in self.module.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        self.module.set_reshard_after_forward(False)    # 推理阶段不要每次前向后自动重新分片
```

**为什么需要保存/恢复 RNG 状态**：推理阶段的动作采样（比如带探索噪声）会消耗随机数生成器的状态，如果不特意保存训练态时的 RNG 状态并在切回训练前恢复，训练的随机性（数据 shuffle、dropout 等）会被推理阶段的采样悄悄污染，导致同样的随机种子在不同 rollout 频率下训练结果不可复现。这是一个容易被忽略但对实验可复现性很关键的细节。

`unshard()` 把 FSDP 分片状态下"每个 rank 只存一部分权重"的布局，聚合成"每个 rank 都有完整权重"的布局，这样才能直接做完整的前向推理；训练时则需要切回分片状态以节省显存。`set_reshard_after_forward(False)` 避免推理阶段每次前向后自动触发一次重新分片（默认训练时的行为），因为连续多步推理之间没必要每步都重新分片再聚合。

核心职责四（disaggregated 专用）：把训练侧权重导出成扁平化的 tensor 流，供 checkpoint engine 传输，兼容 LoRA 场景（需要先合并 adapter）：

```python
def get_per_tensor_param(self, ...):
    if self._is_lora:
        with merged_lora_context(self.module, backup_adapters=True):
            params = normalize_peft_param_name(self.module.state_dict())
            params = {name: param.clone() for name, param in params.items()}
    per_tensor_param = (
        (name, param.full_tensor().to(torch.bfloat16) if isinstance(param, DTensor) else param)
        for name, param in params.items()
    )
    return per_tensor_param, None
```

`param.full_tensor()` 把 FSDP 分片的 `DTensor` 重新聚合成完整张量再转 bfloat16 传输，减少跨进程通信量。

## 4. VLARolloutReplica：撮合 VLA rollout 接入 verl 通用权重广播机制

`vla_replica.py::VLARolloutReplica` 是一个"适配器"，继承 verl 通用的 `RolloutReplica`（原本是为 vLLM/SGLang 这类需要"启服务、睡眠/唤醒、清 KV cache"的重量级推理引擎设计的抽象）。VLA 场景里几乎所有这些生命周期方法都是空操作：

```python
class VLARolloutReplica(RolloutReplica):
    async def launch_servers(self): pass
    async def wake_up(self): pass
    async def sleep(self): pass
    async def clear_kv_cache(self): pass   # VLA 推理没有 KV cache

    def execute_checkpoint_engine(self, method, *args, **kwargs):
        return [worker.execute_checkpoint_engine.remote(method, *args, **kwargs) for worker in self.workers]
```

它不是"独立的模型副本"本身，真正的独立模型副本是 `VLARolloutWorker`（Ray actor）。`VLARolloutReplica` 只是持有这些 worker 的引用，把权重更新请求代理转发过去，让 `CheckpointEngineManager` 能用统一接口驱动权重广播，不需要为 VLA 场景重写整套广播逻辑。

## 5. SACTrainingWorker / SFTTrainingWorker：算法差异在这里体现

`ACTOR_WORKER_REGISTRY` 是一张按配置类型选 TrainingWorker 实现的映射表：

```python
ACTOR_WORKER_REGISTRY = {
    "verl_vla.workers.config.ActorConfig": (ActorConfig, SACTrainingWorker),
    "verl_vla.workers.config.SFTActorConfig": (SFTActorConfig, SFTTrainingWorker),
}
```

`SACTrainingWorker` 和 `SFTTrainingWorker` 是真正实现"这一步训练具体怎么算 loss、怎么更新参数"的地方——`SACTrainingWorker` 涉及 replay pool 采样、critic/actor 交替更新、target 网络软更新等 off-policy 训练的全部细节，第 08 章整章展开；`SFTTrainingWorker` 相对简单，就是标准的监督 loss 反向传播。两者的共同边界都是调用模型自己实现的 `sft_loss`/`sac_forward_actor` 等方法（第 05 章的模型契约）——**训练 worker 只负责"什么时候调用、怎么调度梯度更新"，具体的 loss 计算逻辑下沉到模型层**。

## 小结

| 概念 | 要点 |
|---|---|
| EnvManager | 用独立子进程隔离仿真器，方法调用通过队列跨进程转发，对调用方透明 |
| VLAActorRolloutRefWorker | 一个类通过 role 参数同时支持 actor/rollout/两者兼具三种角色 |
| 权重共享分岔点 | `engine=self.actor.engine if "actor" in self.role else None`——colocate 传引用共享，disaggregated 传 None 独立加载 |
| VLAFSDPEngine | 排除 Embedding/lm_head 的通用 wrap 目标，避免原生 VLA 策略直接访问 .weight 报错 |
| switch_to_rollout | unshard 权重、保存/恢复 RNG 状态、关闭自动 reshard，实现同一份权重的训练态↔推理态切换 |
| get_per_tensor_param | disaggregated 权重导出，兼容 LoRA 合并，转 bfloat16 减少传输量 |
| VLARolloutReplica | 把 VLA rollout worker 接入 verl 通用权重广播机制的适配层，绝大多数生命周期方法是空实现 |

## 下章预告

[第 05 章](./05_模型集成契约_三个接口统一ACT_Pi0_GR00T) 转到模型侧——看 `TrainableVLAModelBase`/`SupportSFTTraining`/`SupportSACTraining` 三个契约怎么把 ACT、Pi0.5、GR00T N1.6 三种架构迥异的原生策略统一接入同一套训练/推理接口，同时保证原生 checkpoint 格式不被破坏。
