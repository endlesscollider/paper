---
title: "启动流程：从 YAML 到分布式集群"
series:
  id: rlinf_deep_dive
  chapter: 2
order: 2
---

# 启动流程：从 YAML 到分布式集群

> 前情提要：上一章介绍了 RLinf 的五大 Worker 角色和两种执行模式。本章将从入口脚本的第一行代码开始，逐步追踪整个启动过程。

## 入口文件：`train_embodied_agent.py`

这是所有具身 RL 训练的入口，整个文件不到 100 行，但串联了 RLinf 的核心启动链路：

```python
@hydra.main(version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft")
def main(cfg) -> None:
    # 第一步：验证配置
    cfg = validate_cfg(cfg)
    
    # 第二步：创建集群
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    
    # 第三步：解析 Placement
    component_placement = HybridComponentPlacement(cfg, cluster)
    
    # 第四步：创建 Worker Group 并启动
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    
    # 第五步：创建 Runner 并运行
    runner = EmbodiedRunner(cfg=cfg, actor=actor_group, rollout=rollout_group, env=env_group, reward=reward_group)
    runner.init_workers()
    runner.run()
```

下面逐步拆解每个环节。

## 第一步：Hydra 配置加载与验证

### Hydra 如何组合配置

RLinf 使用 [Hydra](https://hydra.cc/) 框架管理配置。每个 YAML 文件头部用 `defaults` 引入子配置：

```yaml
defaults:
  - env/libero_spatial@env.train        # 加载 env/libero_spatial.yaml 到 env.train 节点
  - env/libero_spatial@env.eval         # 同一份环境配置复用到 eval
  - model/pi0@actor.model               # 加载 model/pi0.yaml 到 actor.model 节点
  - training_backend/fsdp@actor.fsdp_config  # 加载 FSDP 配置
```

`@` 符号表示"把这个子配置插入到哪个路径"。最终所有子配置会被合并成一个完整的 `DictConfig` 对象。

### validate_cfg() 做了什么

`rlinf/config.py` 中的 `validate_cfg()` 负责：

1. **检查必填字段**：model_path、precision 等必须指定
2. **填充默认值**：比如 `rollout_backend` 默认 `"sglang"`、`gae_lambda` 默认 `0.95`
3. **类型转换**：把 `precision: "bf16"` 转成 `torch.bfloat16`
4. **互斥检查**：FSDP mixed_precision 和 AMP autocast 不能同时开启
5. **根据 HuggingFace config 自动填充模型参数**（Megatron 后端）：`num_layers`、`hidden_size` 等

关键实现：对 FSDP 后端调用 `validate_fsdp_cfg()`，对 Megatron 后端调用 `validate_megatron_cfg()`。

## 第二步：创建 Cluster

```python
cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
```

`Cluster` 是 Ray 集群的抽象层，位于 `rlinf/scheduler/cluster/cluster.py`。它做的事：

1. **初始化 Ray**（如果还没有）：`ray.init(address="auto", namespace=...)`
2. **发现节点信息**：自动检测集群中有多少节点、每个节点有多少 GPU
3. **构建 NodeGroup**：把同类节点分组（如一组 A800 节点、一组 4090 节点）

配置示例：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: 0-7    # 这三个 Worker Group 共享 GPU 0-7
```

## 第三步：解析 Placement

```python
component_placement = HybridComponentPlacement(cfg, cluster)
```

Placement 是 RLinf 调度系统的核心——它决定了"哪个 Worker 运行在哪个 GPU 上"。

### component_placement 配置语法

最简形式：

```yaml
cluster:
  component_placement:
    actor,env,rollout: 0-7       # 三个组共享 GPU 0-7，每个组各占 8 个进程
```

进阶形式（多节点异构）：

```yaml
cluster:
  component_placement:
    actor:
      node_group: a800           # 只在 A800 节点上运行
      placement: 0-8             # 占用 GPU 0-8
    rollout:
      node_group: 4090           # 在 4090 节点上运行推理
      placement: 0-8
    env:
      node_group: robot          # 在有机器人硬件的节点上运行
      placement: 0-3:0-7         # 4 个机器人，每个开 2 个进程
```

### 语法规则详解

`resource_ranks:process_ranks` 的含义：

- `resource_ranks`：硬件资源编号（通常是 GPU 编号）
- `process_ranks`：进程编号
- 如果只写 `0-7`，等价于 `0-7:0-7`（一 GPU 一进程）
- `0-3:0-7` 表示 4 个 GPU 承载 8 个进程（每 GPU 2 个进程）
- `0-7:0-3` 表示 8 个 GPU 承载 4 个进程（每进程 2 个 GPU）
- `all` 表示使用所有可用资源

### HybridComponentPlacement 的解析过程

```python
class HybridComponentPlacement:
    def __init__(self, cfg, cluster):
        # 1. 解析每个 component 的放置策略
        for component_names in placement_config.keys():
            self._parse_component_placement(cluster, component_placement, component_names)
        
    def get_strategy(self, component_name: str) -> PlacementStrategy:
        # 返回具体 Worker Group 的 PlacementStrategy 对象
        ...
```

最终每个 Worker Group 获得一个 `FlexiblePlacementStrategy` 或 `NodePlacementStrategy`，告诉 Ray：这组 Worker 需要几个进程、每个进程分配哪些 GPU。

## 第四步：创建 Worker Group 并启动

### create_group() → WorkerGroup

```python
actor_group = EmbodiedFSDPActor.create_group(cfg)
```

`Worker.create_group()` 是类方法，返回一个 `WorkerGroup` 对象。此时 Worker 还没有被实际创建——只是记录了"要创建什么类、传什么参数"。

### launch() → 真正启动 Ray Actors

```python
actor_group = actor_group.launch(
    cluster,
    name="ActorGroup",
    placement_strategy=actor_placement
)
```

`launch()` 内部：

1. 根据 `PlacementStrategy` 计算出需要创建几个 Ray Actor
2. 为每个 Actor 配置资源需求（GPU 数、CPU 数、自定义资源）
3. 调用 `ray.remote(WorkerClass).options(resources=...).remote(cfg)` 创建 Actor
4. 等待所有 Actor 就绪

每个 Worker Actor 启动时会自动：
- 设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 环境变量
- 选择对应的 GPU（`CUDA_VISIBLE_DEVICES`）
- 初始化 `torch.distributed` 进程组
- 注册到 Scheduler Manager

### Actor 类的选择逻辑

根据 `algorithm.loss_type` 自动选择不同的 Actor Worker 类：

```python
if cfg.algorithm.loss_type == "embodied_sac":
    actor_worker_cls = EmbodiedSACFSDPPolicy    # SAC 专用
elif cfg.algorithm.loss_type == "embodied_dagger":
    actor_worker_cls = EmbodiedDAGGERFSDPPolicy # DAgger 专用
elif cfg.algorithm.loss_type == "embodied_nft":
    actor_worker_cls = EmbodiedNFTFSDPPolicy    # NFT 专用
else:
    actor_worker_cls = EmbodiedFSDPActor        # PPO/GRPO 通用
```

## 第五步：Runner 初始化与运行

### init_workers()

```python
runner.init_workers()
```

按照**严格顺序**初始化各 Worker：

```python
def init_workers(self):
    self.actor.init_worker().wait()      # 先初始化 Actor（加载模型、建 FSDP）
    self.rollout.init_worker().wait()    # 再初始化 Rollout（加载推理模型）
    self.env.init_worker().wait()        # 最后初始化 Env（启动仿真器）
    if self.reward is not None:
        self.reward.init_worker().wait()
```

为什么要按顺序？因为 Worker 初始化要加载大模型，如果同时加载会 OOM。先让 Actor 加载完可以 offload 到 CPU，再让 Rollout 加载。

### Channel 的创建

Runner 构造时创建了 4 个 Channel 用于 Worker 间通信：

```python
self.env_channel = Channel.create("Env")          # Env ↔ Rollout 的观测/动作交换
self.rollout_channel = Channel.create("Rollout")  # Runner → Rollout 的指令
self.actor_channel = Channel.create("Actor")      # Env → Actor 的轨迹传输
self.reward_channel = Channel.create("Reward")    # Env ↔ Reward 的奖励计算
```

### run() 的整体流程

同步模式（EmbodiedRunner）的主循环：

```python
def run(self):
    for step in range(max_steps):
        # 1. 同步权重：Actor → Rollout
        self.update_rollout_weights()
        
        # 2. 环境交互 + 策略推理（通过 Channel 协作）
        env_handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            actor_channel=self.actor_channel,
        )
        rollout_handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_handle.wait()
        rollout_handle.wait()
        
        # 3. Actor 训练
        actor_handle = self.actor.run_training()
        actor_result = actor_handle.wait()
        
        # 4. 评估 & 保存
        if should_eval: self.evaluate()
        if should_save: self._save_checkpoint()
```

## 完整启动时序图

```mermaid
sequenceDiagram
    participant User as 用户脚本
    participant Hydra as Hydra 配置
    participant Cluster as Cluster
    participant Placement as Placement
    participant Actor as Actor Group
    participant Rollout as Rollout Group
    participant Env as Env Group
    participant Runner as Runner
    
    User->>Hydra: 加载 YAML 配置
    Hydra->>User: 合并后的 DictConfig
    User->>User: validate_cfg()
    User->>Cluster: Cluster(cfg.cluster)
    Cluster->>Cluster: ray.init() + 发现节点
    User->>Placement: HybridComponentPlacement(cfg, cluster)
    Placement->>Placement: 解析 component_placement
    
    User->>Actor: create_group(cfg).launch(cluster, placement)
    Actor->>Actor: 创建 N 个 Ray Actor
    User->>Rollout: create_group(cfg).launch(cluster, placement)
    Rollout->>Rollout: 创建 M 个 Ray Actor
    User->>Env: create_group(cfg).launch(cluster, placement)
    Env->>Env: 创建 K 个 Ray Actor
    
    User->>Runner: EmbodiedRunner(cfg, actor, rollout, env)
    Runner->>Runner: 创建 Channel
    
    Runner->>Actor: init_worker() — 加载模型、建 FSDP
    Runner->>Rollout: init_worker() — 加载推理模型
    Runner->>Env: init_worker() — 启动仿真器
    
    Runner->>Runner: run() — 进入训练循环
```

## 关键源码文件

| 文件 | 职责 |
|------|------|
| `examples/embodiment/train_embodied_agent.py` | 入口脚本 |
| `rlinf/config.py` | 配置验证与默认值填充 |
| `rlinf/scheduler/cluster/cluster.py` | Cluster 抽象 |
| `rlinf/scheduler/placement/placement.py` | ComponentPlacement 解析 |
| `rlinf/utils/placement.py` | HybridComponentPlacement |
| `rlinf/scheduler/worker/worker.py` | Worker 基类 |
| `rlinf/scheduler/worker/worker_group.py` | WorkerGroup.launch() |
| `rlinf/runners/embodied_runner.py` | 同步 Runner |
| `rlinf/runners/async_embodied_runner.py` | 异步 Runner |

## 下一章预告

[第 03 章](./03_Scheduler调度系统) 将深入 `rlinf/scheduler/` 目录，详解 Cluster、Placement、Channel、Collective 四大子系统的内部实现。
