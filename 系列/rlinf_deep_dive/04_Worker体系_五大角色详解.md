---
title: "Worker 体系：五大角色详解"
series:
  id: rlinf_deep_dive
  chapter: 4
order: 4
---

# Worker 体系：五大角色详解

> 前情提要：上一章详解了 Scheduler 的四大子系统。本章深入每个 Worker 的具体实现。

## Actor Worker：策略训练

Actor 是 RLinf 中最复杂的 Worker，负责接收轨迹数据、计算优势、执行策略梯度更新。

### 类层次

```
Worker (scheduler/worker/worker.py)
└── FSDPModelManager (hybrid_engines/fsdp/fsdp_model_manager.py)
    └── FSDPActor (workers/actor/fsdp_actor_worker.py)
        └── EmbodiedFSDPActor (同文件)      — PPO/GRPO 通用
        └── EmbodiedSACFSDPPolicy            — SAC 专用
        └── EmbodiedDAGGERFSDPPolicy         — DAgger 专用
        └── EmbodiedNFTFSDPPolicy            — NFT 专用
```

### init_worker() 做了什么

```python
def init_worker(self):
    # 1. 构建模型 + FSDP wrap + 优化器
    self.setup_model_and_optimizer()
    
    # 2. 如果开启 offload，把模型参数搬到 CPU
    if self.enable_offload:
        self.offload_param_and_grad()
        self.offload_optimizer()
    
    # 3. 计算 Actor → Rollout 的权重同步目标 rank
    self._setup_rollout_weight_dst_ranks()
```

### 权重同步目标计算

当 Actor 有 M 个进程、Rollout 有 N 个进程时，每个 Actor rank 负责向 `ceil(N/M)` 个 Rollout rank 发送权重：

```python
def _setup_rollout_weight_dst_ranks(self):
    rollout_world_size = self._component_placement.get_world_size("rollout")
    actor_world_size = self._world_size
    rank = self._rank
    self._weight_dst_rank_in_rollout = []
    rollout_ranks_per_actor = (rollout_world_size + actor_world_size - 1) // actor_world_size
    for i in range(rollout_ranks_per_actor):
        if i * actor_world_size + rank < rollout_world_size:
            self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)
```

例如 2 个 Actor + 4 个 Rollout：Actor 0 → Rollout 0, 2；Actor 1 → Rollout 1, 3。

### sync_model_to_rollout()

```python
async def sync_model_to_rollout(self):
    # 1. 如果 offload 了，先把参数搬回 GPU
    if self.enable_offload and self.is_weight_offloaded:
        self.load_param_and_grad(self.device)
    
    # 2. 获取完整 state_dict
    state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
    
    # 3. 异步发送给所有目标 Rollout rank
    handles = []
    for rank in self._weight_dst_rank_in_rollout:
        handles.append(self.send(state_dict, self._rollout_group_name, rank, async_op=True))
    for handle in handles:
        await handle.async_wait()
    
    # 4. 如果需要，再 offload 回 CPU
    if self.enable_offload:
        self.offload_param_and_grad()
```

### recv_rollout_trajectories()

Actor 从 Channel 接收 Env Worker 发来的轨迹数据：

```python
async def recv_rollout_trajectories(self, input_channel: Channel):
    # 计算需要接收几份数据
    send_num = env_world_size * stage_num
    recv_num = actor_world_size
    split_num = compute_split_num(send_num, recv_num)
    
    # 逐份接收 Trajectory
    recv_list = []
    for _ in range(split_num):
        trajectory = await input_channel.get(async_op=True).async_wait()
        recv_list.append(trajectory)
    
    # 合并为一个大 batch
    self.rollout_batch = convert_trajectories_to_batch(recv_list)
```

### run_training()

PPO/GRPO 的训练步骤：

1. **计算优势**：调用 `calculate_adv_and_returns()`（GAE 或 GRPO）
2. **Shuffle 数据**：随机打乱 batch
3. **多 epoch 更新**：`update_epoch` 轮（GRPO 通常 2 轮）
4. **Micro-batch 前向+反向**：
   - 前向得到 logprobs
   - 计算 policy loss（clip ratio）
   - 反向传播
   - 梯度累积
5. **优化器 step**：梯度裁剪 + Adam 更新

## Rollout Worker：策略推理

### 核心职责

从 Env 接收观测 → 用策略模型做推理 → 返回动作和 logprobs。

### 类定义

```python
class MultiStepRolloutWorker(Worker):
    """多步推理 Worker，一次生成 num_action_chunks 步动作"""
```

### init_worker() 关键步骤

```python
def init_worker(self):
    # 1. 加载 HuggingFace 模型（和 Actor 同架构，但独立副本）
    self.hf_model = get_model(rollout_model_config)
    
    # 2. 可选：加载 expert 模型（DAgger 用）
    if self.cfg.rollout.get("expert_model", None):
        self.expert_model = get_model(expert_model_config)
    
    # 3. 设置推理模式
    self.hf_model.eval()
    
    # 4. 可选优化
    if enable_torch_compile: self.hf_model.enable_torch_compile()
    if enable_cuda_graph: self.hf_model.capture_cuda_graph(...)
    
    # 5. 计算通信 rank 映射
    self.dst_ranks = self._setup_dst_ranks(...)
    self.src_ranks = self._setup_src_ranks(...)
    
    # 6. 如果 offload，把模型搬到 CPU
    if self.enable_offload: self.offload_model()
```

### generate_one_epoch()

一个 rollout epoch 的推理循环：

```python
async def generate_one_epoch(self, input_channel, output_channel):
    for _ in range(self.n_train_chunk_steps):
        for _ in range(self.num_pipeline_stages):
            # 1. 从 Channel 接收环境观测
            env_output = await self.recv_env_output(input_channel)
            
            # 2. 模型推理
            actions, result = self.predict(env_output["obs"])
            
            # 3. 构建 RolloutResult
            rollout_result = RolloutResult(
                actions=actions,
                prev_logprobs=result["prev_logprobs"],
                prev_values=result["prev_values"],
                bootstrap_values=self.get_bootstrap_values(env_output.get("final_obs")),
                versions=torch.full_like(..., float(self.version)),
            )
            
            # 4. 发送回 Env
            self.send_rollout_result(output_channel, rollout_result, mode="train")
```

### sync_model_from_actor()

从 Actor 接收新权重：

```python
async def sync_model_from_actor(self):
    param_state_dict = await self.recv(
        self.actor_group_name,
        src_rank=self.actor_weight_src_rank,
        async_op=True,
    ).async_wait()
    self.hf_model.load_state_dict(param_state_dict)
```

### Pipeline Stage

`pipeline_stage_num` 控制 Env-Rollout 之间的流水线并行度。当 `pipeline_stage_num=2` 时，Env 和 Rollout 各自会把数据拆成 2 份，交替处理，减少等待时间。

## Env Worker：环境交互

### 核心职责

管理仿真器实例，执行环境 step，收集观测和奖励。

### 类定义

```python
class EnvWorker(Worker):
    """环境 Worker，管理仿真器并与 Rollout Worker 协作完成数据采集"""
```

### init_worker() 关键步骤

```python
def init_worker(self):
    # 1. 获取环境类
    train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
    
    # 2. 创建环境实例（每个 stage 一个）
    self.env_list = self._setup_env_and_wrappers(
        env_cls=train_env_cls,
        num_envs_per_stage=self.train_num_envs_per_stage,
    )
    
    # 3. 初始化环境（reset）
    self._init_env()
```

### 环境数量计算

```python
# 每个 Env Worker 的每个 pipeline stage 管理的环境数
train_num_envs_per_stage = total_num_envs // env_world_size // stage_num
```

例如：`total_num_envs=64`，`env_world_size=4`，`stage_num=2` → 每个 Worker 每 stage 管理 8 个环境。

### interact() 主循环

```python
async def interact(self, input_channel, rollout_channel, actor_channel, ...):
    for rollout_epoch in range(self.rollout_epoch):
        for chunk_step in range(self.n_train_chunk_steps):
            for stage in range(self.stage_num):
                # 1. 把当前观测发送给 Rollout
                env_output = self._get_env_output(stage)
                rollout_channel.put(env_output)
                
                # 2. 等待 Rollout 返回动作
                rollout_result = await input_channel.get(async_op=True).async_wait()
                
                # 3. 执行环境 step
                obs, rewards, dones, infos = self.env_list[stage].step(actions)
                
                # 4. 记录 chunk step 数据
                self._record_chunk_step(rollout_result, rewards, dones, ...)
        
        # 5. 一个 rollout epoch 结束，发送轨迹给 Actor
        trajectory = self._build_trajectory()
        actor_channel.put(trajectory)
```

### 支持的环境类型

`rlinf/envs/__init__.py` 中的 `get_env_cls()` 注册了所有支持的环境：

| env_type | 类 | 仿真器 |
|---------|-----|--------|
| `maniskill` | `ManiskillEnv` | ManiSkill3 |
| `libero` | `LiberoEnv` | LIBERO |
| `robotwin` | `RoboTwinEnv` | RoboTwin |
| `isaaclab` | 注册表 | IsaacLab |
| `metaworld` | `MetaWorldEnv` | MetaWorld |
| `calvin` | `CalvinEnv` | CALVIN |
| `robocasa` | `RobocasaEnv` | RoboCasa |
| `realworld` | `RealWorldEnv` | 真机 |
| `frankasim` | `FrankaSimEnv` | Franka 仿真 |
| `opensora_wm` | `OpenSoraEnv` | OpenSora 世界模型 |
| `wan_wm` | `WanEnv` | Wan 世界模型 |

## Reward Worker：奖励计算

### 两种模式

1. **规则奖励**（默认）：环境直接返回奖励，Reward Worker 不启动
2. **模型奖励**：用奖励模型对轨迹打分（如 VLM 评估任务完成度）

### 配置开关

```yaml
reward:
  use_reward_model: False  # True 时启动 Reward Worker
```

### EmbodiedRewardWorker

当使用模型奖励时：

```python
class EmbodiedRewardWorker(Worker):
    async def compute_rewards_async(self, input_channel, output_channel):
        while True:
            data = await input_channel.get(async_op=True).async_wait()
            # 用奖励模型计算 reward
            reward = self.reward_model(data["images"], data["states"])
            output_channel.put({"reward": reward, "env_id": data["env_id"]})
```

### 混合奖励

支持环境奖励 + 模型奖励的加权组合：

```yaml
reward:
  use_reward_model: True
  reward_weight: 1.0       # 模型奖励权重
  env_reward_weight: 0.5   # 环境奖励权重
```

## Critic Worker（可选）

### 何时使用

只有 PPO（`loss_type: actor_critic`）需要独立的 Critic。GRPO 不用 Critic。

### 两种实现方式

1. **共享模型**：Actor 模型加一个 `value_head`，Critic 和 Actor 共用 backbone
   ```yaml
   actor:
     model:
       add_value_head: True    # 在模型上加 value head
   critic:
     use_critic_model: False   # 不用独立 Critic
   ```

2. **独立模型**：单独部署一个 Critic Worker（Megatron 后端用）
   ```yaml
   critic:
     use_critic_model: True
     group_name: "CriticGroup"
   ```

## Worker 间的 Offload 协作

当 Actor 和 Rollout 共享 GPU 时（`actor,rollout: 0-7`），通过 offload 机制避免 OOM：

```mermaid
sequenceDiagram
    participant GPU as GPU 显存
    participant Actor as Actor Worker
    participant Rollout as Rollout Worker
    
    Note over GPU: Rollout 模型在 GPU 上
    Rollout->>Rollout: 推理完成
    Rollout->>GPU: offload_model() → 模型搬到 CPU
    Note over GPU: GPU 空闲
    
    Actor->>GPU: load_param_and_grad() → 模型搬回 GPU
    Note over GPU: Actor 模型在 GPU 上
    Actor->>Actor: 训练 step
    Actor->>GPU: offload_param_and_grad() → 模型搬到 CPU
    Note over GPU: GPU 空闲
    
    Rollout->>GPU: load_model() → 模型搬回 GPU
    Note over GPU: Rollout 模型在 GPU 上
    Rollout->>Rollout: 开始推理
```

配置：
```yaml
actor:
  enable_offload: True
rollout:
  enable_offload: True
```

## 下一章预告

[第 05 章](./05_数据流与通信机制) 将追踪数据从环境产生到梯度更新的完整流转过程，详解 EnvOutput、RolloutResult、Trajectory 等核心数据结构。
