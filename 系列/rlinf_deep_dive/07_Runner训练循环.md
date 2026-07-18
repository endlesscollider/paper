---
title: "Runner 训练循环"
series:
  id: rlinf_deep_dive
  chapter: 7
order: 7
---

# Runner 训练循环

> 前情提要：上一章详解了 FSDP 训练后端——Actor Worker 内部怎么管理模型和显存。本章往上一层看：谁在指挥 Actor、Rollout、Env 这几个 Worker 按什么顺序、什么节奏配合工作？答案是 Runner。

## 一、Runner 要解决什么问题

前几章讲的 Worker（Actor/Rollout/Env/Reward）各自只知道怎么做自己的事：Env 知道怎么和仿真器交互，Rollout 知道怎么推理出动作，Actor 知道怎么训练。但没有一个 Worker 知道"现在应该干什么"——现在是该采集数据了，还是该同步权重了，还是该存 checkpoint 了？

这正是 Runner 的职责：它自己不做任何计算，只是站在最上层，按固定的节奏依次调用各个 Worker 的方法。可以把 Runner 类比成乐队的指挥——指挥不演奏任何乐器，但决定了"现在小提琴进，现在鼓点停"。具体来说 Runner 要管四件事：

1. 控制权重什么时候从 Actor 同步给 Rollout
2. 协调数据采集（Env + Rollout）和策略训练（Actor）的执行顺序
3. 按周期触发评估和 Checkpoint 保存
4. 汇总各个 Worker 报回来的 metrics，写入日志

RLinf 提供两种 Runner，对应两种完全不同的协调节奏：`EmbodiedRunner`（同步模式）和 `AsyncEmbodiedRunner`（异步模式）。本章先讲同步模式建立基本框架，再看异步模式在此基础上做了什么改动。

## 二、同步模式：先采集，再训练，严格交替

同步模式的核心思路非常直接：**在任意时刻，系统里只发生一件事**——要么所有资源都在采集数据，要么所有资源都在训练。这是最容易理解、最不容易出 bug 的调度方式，代价是 GPU 有一部分时间在空等（比如 Actor 训练时，Rollout 和 Env 的资源是闲置的）。

### 2.1 一步（step）里发生的四个阶段

`EmbodiedRunner.run()` 的主循环，每一轮迭代对应"用当前策略采一批数据，训练一步"。按发生顺序拆开看：

**阶段一：同步权重**。每一步开始前，先检查是否需要把 Actor 最新的参数同步给 Rollout——不是每一步都同步，而是按 `weight_sync_interval` 控制频率：

```python
if _step % self.weight_sync_interval == 0:
    self.update_rollout_weights()
```

`weight_sync_interval` 默认是 1（每步都同步）。调大这个值可以省下同步权重的通信开销，但代价是 Rollout 会用稍微过时的策略采集好几步的数据——这是训练速度和数据"新鲜度"之间的权衡，配置在 `runner.weight_sync_interval` 里。

**阶段二：采集数据**。Env 和 Rollout 交替工作（Env 给观测，Rollout 出动作，Env 再执行动作），这个过程在 [第 05 章](./05_数据流与通信机制#阶段-3env-积累轨迹) 已经详细讲过数据怎么流转，这里 Runner 要做的只是把两个 Worker 启动起来、等它们都跑完：

```python
env_handle = self.env.interact(input_channel=self.env_channel, ...)
rollout_handle = self.rollout.generate(input_channel=self.rollout_channel, ...)
self.actor.recv_rollout_trajectories(input_channel=self.actor_channel).wait()
rollout_handle.wait()
```

注意这里 Runner 等待的是 `self.actor.recv_rollout_trajectories(...)` 而不是直接等 `env_handle`——因为 Actor 要等到收完一整轮采集出的所有轨迹才算数据采集阶段真正结束，`env_handle` 本身会在后面统一收集 metrics 时再 wait。

**阶段三：计算 Advantage，然后训练**。数据采集完，先调用 `compute_advantages_and_returns()` 算出 GAE（[第 08 章](./08_算法实现_PPO配置详解) 会细讲这一步），再调用 `run_training()` 真正做梯度更新：

```python
actor_rollout_metrics = self.actor.compute_advantages_and_returns().wait()
actor_training_handle = self.actor.run_training()
actor_training_metrics = actor_training_handle.wait()
self.global_step += 1
```

**阶段四：周期性的评估和保存**。每一步结束后检查是否到了评估或保存 checkpoint 的时机（由 `check_progress` 根据 `val_check_interval` / `save_interval` 判断），如果需要评估，还会先做一次权重同步（保证评估用的是最新策略）：

```python
if run_val:
    self.update_rollout_weights()
    eval_metrics = self.evaluate()
if save_model:
    self._save_checkpoint()
```

四个阶段严格顺序执行，任何一步没完成，下一步都不会开始——这就是"同步"的含义。

### 2.2 评估复用了训练的 Channel

评估阶段调用的 `evaluate()` 内部逻辑和采集数据阶段几乎一样，都是让 Env 和 Rollout 互相配合跑几轮：

```python
def evaluate(self):
    env_handle = self.env.evaluate(input_channel=self.env_channel, rollout_channel=self.rollout_channel)
    rollout_handle = self.rollout.evaluate(input_channel=self.rollout_channel, output_channel=self.env_channel)
    env_results = env_handle.wait()
    rollout_handle.wait()
    return compute_evaluate_metrics([r for r in env_results if r is not None])
```

区别在于配置上：评估时 Env 用的是 `env.eval` 配置（可能环境数量、任务难度都和训练不一样），采样温度也通常调低（比训练温度更"贪心"，减少探索带来的方差，让评估结果更能反映策略的真实水平）。

## 三、异步模式：不再互相等待

同步模式最大的浪费在于：Actor 训练的时候，Rollout 和 Env 的 GPU/CPU 资源完全闲置；反过来采数据的时候，Actor 那部分资源也在闲置。如果硬件条件允许两组资源同时跑（比如它们本来就分布在不同的 GPU 上），完全可以让"采集"和"训练"**同时进行**——这就是 `AsyncEmbodiedRunner` 要做的事。

### 3.1 核心改动：Worker 变成长驻服务

同步模式里，`env.interact()` 和 `rollout.generate()` 每一步都要重新调用一次，跑完这一批数据就返回。异步模式把这两个方法改成了**长期运行、不返回**的服务——在 `run()` 一开始就启动，之后一直在后台跑，源源不断地往 `actor_channel` 里产出轨迹：

```python
env_handle = self.env.interact(..., metric_channel=self.env_metric_channel)
rollout_handle = self.rollout.generate(..., metric_channel=self.rollout_metric_channel)
actor_handle = self.actor.recv_rollout_trajectories(input_channel=self.actor_channel)
```

这三行代码执行完，采集流程就已经在后台跑起来了，`run()` 后面进入的 `while` 循环里，Actor 只管不断尝试训练，不用管数据是不是"刚好采集完"这件事——它直接问 Channel 里有没有数据可用：

```python
while self.global_step < self.max_steps:
    actor_training_handle = self.actor.run_training()
    actor_result = actor_training_handle.wait()
    if not actor_result[0]:
        # channel 里数据还不够一个 batch，这一步先跳过，睡一下再试
        time.sleep(1.0)
        continue
    self.global_step += 1
```

`actor_result[0]` 是一个布尔值，表示"这次调用是否真的拿到了足够数据完成了一次训练"——如果 Rollout 还没攒够一个 batch 的轨迹，`run_training()` 会直接返回"没训成"，Runner 睡一秒再重试，不会阻塞在这里空等。

### 3.2 权重同步也要变成"不等待"，否则白做了异步

如果异步模式下权重同步还是像同步模式那样"发出去就等它传完"，Actor 会在等待同步的那几秒里被迫停下——这就把好不容易解耦开的"采集"和"训练"又耦合到了一起。所以异步模式引入了 `no_wait` 参数：

```python
def update_rollout_weights(self, no_wait=False):
    if not no_wait:
        return super().update_rollout_weights()  # 老实等，退化为同步行为

    if not self._cleanup_pending_rollout_weight_sync(no_wait):
        # 上一次的同步还没完成，这次就不发了，直接跳过
        self._weight_sync_coalesced_total += 1
        return

    rollout_handle = self.rollout.request_actor_sync_model()
    actor_handle = self.actor.sync_model_to_rollout()
    self._pending_rollout_weight_sync = (rollout_handle, actor_handle)  # 记下来，下次检查
```

**为什么需要"合并跳过"这个逻辑**：假设 Actor 训练速度很快，每隔几百毫秒就想同步一次权重，但一次权重同步（传输几 GB 的模型参数）需要好几秒才能传完。如果不做任何限制，新的同步请求会不断堆积，网络带宽被同步请求占满，训练反而被拖慢。`_cleanup_pending_rollout_weight_sync` 检查上一次同步是否已经做完（`rollout_handle.done()` 和 `actor_handle.done()`），如果没做完就直接放弃这次同步——相当于"忙的时候就不打扫了，等空下来再打扫"，用旧权重多跑几步比等一个还没传完的新权重更划算。

### 3.3 数据是旧策略产出的，怎么修正

异步模式带来一个新问题：Rollout 拿到的策略权重可能已经不是 Actor 当前最新的版本了（因为权重同步是"尽力而为"、可能被跳过的）。用这种"过时"数据直接算策略梯度，理论上是有偏的——这是 off-policy 场景的经典问题。

RLinf 的解法是给每条数据打上版本号 `versions`（表示这条数据是第几版策略产出的），Actor 训练时对比"数据的版本"和"当前策略的版本"，用 [PPO 前置知识](/前置知识/000a_前置知识_策略梯度与PPO)里 importance sampling 的思路做修正。修正逻辑在损失函数里（`rlinf/algorithms/losses.py`），先算出数据落后了几个版本：

```python
version_diff = current_version - v_behav          # 当前版本 - 数据产出时的版本
version_gap = (current_version - 1) - v_behav      # proximal 版本 - 数据产出时的版本
alpha = torch.clamp(version_gap / version_diff, 0.0, 1.0)
```

**为什么要算这个 `alpha`**：如果数据是最新策略产出的（`version_diff` 很小甚至是 0），几乎不需要修正；如果数据落后了好几个版本，需要在"旧策略的 logprob"和"当前策略的 logprob"之间插值出一个"过渡版本"（proximal policy）作为重要性采样的基准，`alpha` 就是这个插值系数——数据越旧，`alpha` 越接近 1，插值点越靠近当前策略。这一步的目的是让重要性采样比值不会因为版本差太多而剧烈震荡，训练更稳定。具体的插值和 clip 逻辑属于算法细节，后面 PPO 章节还会再展开，这里只需要知道：**版本号是异步训练下"数据不新鲜"问题的追踪手段，插值修正是用来抵消这种不新鲜带来的偏差**。

## 四、两种模式该怎么选

| | 同步模式 | 异步模式 |
|---|---------|---------|
| 执行方式 | 采集和训练严格交替，谁都不能抢跑 | 采集和训练并行跑，互不等待 |
| 资源利用率 | 较低（各自等待对方时段是闲置的） | 较高 |
| 数据新鲜度 | 永远是当前最新策略产出的 | 可能落后几个版本，需要额外的修正逻辑 |
| 实现复杂度 | 低 | 高（引入版本追踪、合并跳过等机制） |
| 适用场景 | 调试阶段、Actor 和 Rollout 共享同一批 GPU 时 | 生产训练、Actor 和 Rollout 分布在独立 GPU 时 |

一个直观的判断标准：如果 Actor 和 Rollout 本来就要抢同一批 GPU（[第 03 章](./03_Scheduler调度系统#共享-gpu-场景)提到的共享部署），异步模式带来的"并行"其实是假的（两者本来就不能同时用 GPU），这时候用同步模式反而更简单、更不容易出 bug。只有当 Actor 和 Rollout 分布在不同的 GPU 上，异步模式的并行才有实际意义。

## 五、Checkpoint 存到哪、存什么

保存 checkpoint 由 `_save_checkpoint()` 触发，按当前的 `global_step` 生成一个独立目录，实际的存储动作委托给 [上一章](./06_训练后端_FSDP与Megatron#七checkpoint分片存-vs-完整存) 讲过的 Actor 的 `save_checkpoint` 接口：

```python
base_output_dir = os.path.join(log_path, experiment_name, f"checkpoints/global_step_{self.global_step}")
self.actor.save_checkpoint(os.path.join(base_output_dir, "actor"), self.global_step).wait()
```

产出的目录结构大致是：

```
results/experiment_name/
├── checkpoints/
│   ├── global_step_40/actor/
│   ├── global_step_80/actor/
│   └── ...
├── tensorboard/
└── video/eval/
```

每个 `global_step_N` 目录都是一份独立的、可以用来恢复训练或单独加载权重做推理的完整快照。

## 六、metrics 怎么收集和记录

Runner 每一步都要从多个 Worker 那里收集不同类型的数据，统一喂给 `MetricLogger`：

- **时间指标**（`time/*`）：每个阶段（同步权重、采集数据、训练）各花了多久，来自 `self.timer` 和各 Worker Handle 的 `consume_durations()`
- **环境指标**（`env/*`）：任务成功率、episode 长度等，来自 `compute_evaluate_metrics`
- **训练指标**（`train/*`）：loss、梯度范数、KL 散度等，来自 Actor 的 `run_training()` 返回值
- **评估指标**（`eval/*`）：只在评估步产生

`MetricLogger` 支持同时写到多个后端，配置里指定用哪些：

```yaml
runner:
  logger:
    log_path: "../results"
    experiment_name: "my_experiment"
    logger_backends: ["tensorboard"]  # 可选还有 wandb、swanlab
```

写日志走的是独立的后台线程（`self.log_thread`），不会阻塞主训练循环——把"记日志"这种慢操作和"训练"这种关键路径解耦，也是保证训练效率的一个小细节。

## 七、总结

| 环节 | 同步模式 | 异步模式 |
|------|---------|---------|
| 主循环结构 | for 循环，四阶段严格顺序执行 | while 循环，Actor 持续尝试训练，采集在后台跑 |
| 权重同步 | 同步等待发送完成 | `no_wait` + 合并跳过机制 |
| 数据一致性 | 永远最新 | 需要版本号 + importance sampling 修正 |
| Checkpoint / 评估 | 按 step 周期触发，逻辑一致 | 同左 |

## 下一章预告

[第 08 章](./08_算法实现_PPO配置详解) 将进入算法层面，逐参数拆解 PPO 在 RLinf 中的配置和实现。
