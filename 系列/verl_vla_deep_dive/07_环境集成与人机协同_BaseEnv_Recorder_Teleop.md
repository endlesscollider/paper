---
title: "环境集成与人机协同：BaseEnv、Recorder、Teleop"
series:
  id: verl_vla_deep_dive
  chapter: 7
order: 7
---

# 第 07 章 环境集成与人机协同：BaseEnv、Recorder、Teleop

> 前情提要：第 04 章讲了 EnvWorker 用独立子进程隔离仿真器，第 05、06 章讲完了模型侧的接入方式。这一章讲环境侧统一接口——LIBERO 仿真、Isaac Lab Arena 大规模仿真、Piper 真机机械臂，三种截然不同的运行时怎么被 `BaseEnv` 统一成同一套 API，动作块怎么执行、人类怎么实时接管、数据怎么被录制下来。

## 知识链接

- 上一章：[Flow-SDE 与 DSRL：给流匹配策略装上 SAC](./06_FlowSDE与DSRL_给流匹配策略装上SAC)
- 下一章：[SAC 训练循环：EpisodeBuffer、ReplayPool 与 RLPD](./08_SAC训练循环_EpisodeBuffer_ReplayPool与RLPD)
- [系列目录](./index)
- [行为克隆与 RL 微调范式](/前置知识/000d_前置知识_行为克隆与RL微调范式)

---

## 1. 统一契约：`env_step` 返回什么，比"怎么实现"更重要

`BaseEnv`（`envs/base.py`）是所有环境集成的公共基类。它的核心设计思路是：**Gym 风格的 `reset`/`step`/`close` 全部由基类实现好，子类只需要实现四个更底层的钩子**——`env_init`、`env_reset`、`env_step`、`env_close`。子类唯一需要保证的是 `env_step` 返回一个固定格式的字典：

```python
{
    "observation": [{"observation.images.image": ..., "observation.state": ...}, ...],  # 每个环境一份
    "task": [...],           # 语言指令
    "task_id": np.array(...),
    "next.reward": ...,
    "next.terminated": ...,  # 自然终止（比如任务成功/失败）
    "next.truncated": ...,   # 外部截断（比如达到最大步数）
    "next.success": ...,     # 评估用的成功信号，独立于 reward
}
```

**为什么要固定这个格式**：只要子类的 `env_step` 遵守这个契约，`BaseEnv` 提供的所有能力——动作块执行、人类干预覆盖、录制、遥操作面板发布——就能对任何新接入的环境自动生效，不需要为每种环境单独写一份"怎么执行动作块""怎么处理人类接管"的逻辑。这正是第 01 章讲的"环境集成代码只关心环境本身，不关心训练/评估/遥操作的编排逻辑"这条原则的具体落地。

## 2. 动作块执行与实时人类接管

verl-vla 的动作不是单步的，而是"动作块"（action chunk）——策略一次推理输出未来若干步的动作序列 `[B, T, D]`（$T$ 是块长度）。`BaseEnv.step()` 负责把这个块拆成 $T$ 个单步，依次调用子类的 `env_step`：

```python
for step_idx in range(num_chunk_steps):
    step_actions = action[:, step_idx]
    merged_step_result, step_restart_episode, chunk_intervened = self.step_with_teleop_and_recording(
        step_actions, critic_value=step_values, chunk_intervened=chunk_intervened, merged_step_result=merged_step_result,
    )
```

**这段代码在做什么**：不是简单地把 $T$ 步动作一次性丢给环境执行，而是每执行一步就检查一次"这一步是不是有人类正在实时操控这个环境"。这是支持人机协同训练的关键设计——策略给出的整段动作块，可能在执行到第 3 步时被人类突然接管。

`step_with_teleop_and_recording` 内部是一个 `while` 循环，处理接管逻辑：

```python
while True:
    next_action, intervention_mask, step_manual_reward, step_restart_episode, stop_episode = self.apply_teleop_action(action)
    if not intervention_mask.any():
        break
    # 有环境被接管：先执行"接管前已经确定要走的那一步"，再让接管生效
    need_execute = is_intervened & intervention_mask
    if need_execute.any():
        step_result = self.mask_step(action, need_execute, is_intervention=intervention_mask, ...)
        ...
    active_mask = intervention_mask & ~done
    action[active_mask] = next_action[active_mask]   # 把这个环境的后续动作换成人类给出的动作
    is_intervened |= active_mask
```

**这段代码在做什么**：`apply_teleop_action` 检查每个并行环境是否有对应的遥操作设备正在"接管"状态。一旦某个环境被标记为接管，它接下来这个动作块**剩余的所有步**都会用人类实时给出的动作，而不是策略原本规划好的动作块内容——即使策略的动作块本来是要连续走 8 步，人类在第 3 步接管之后，第 4-8 步会实时读取人类当前的操作，不会回退去执行策略原来算好的动作。

这个设计的动机是：机器人策略在探索/训练早期经常会做出危险或明显错误的动作（比如要把机械臂撞向障碍物），允许操作员实时观察并立即接管纠正，比"等整个动作块执行完再让人判断要不要重来"安全得多。执行结果里会带上 `info.is_intervention` 标记，让下游训练/统计代码能区分"这一步是策略自主做的还是人类纠正的"。

## 3. `mask_step`：支持部分环境执行的核心方法

由于允许部分环境被接管、部分环境按策略正常执行，`BaseEnv` 还需要支持"只对一部分并行环境实例执行这一步"（`execute_mask` 参数），这也是为什么它叫 `mask_step`。这个方法内部依次做：调用子类 `env_step`（只对选中的环境）、应用手动奖励/强制截断的覆盖（人类可以按键手动标记"这次成功了"或"这次失败了，重来"）、把结果发布给遥操作面板、写入录制器、更新最新观测缓存。

## 4. 三种环境实现：统一契约下的巨大差异

### LIBERO（仿真基准）

`LiberoEnv`（`envs/libero/libero_env.py`）用 `SubprocVectorEnv`（多进程向量化环境，来自 LIBERO 官方库）并扩展出 `ReconfigureSubprocEnv`，支持在训练过程中动态切换任务（不同任务对应不同的 `.bddl` 场景文件，需要重新构造底层仿真器）。`LiberoResetStatePlanner` 管理"任务 id / 试验 id / reset 状态 id"三元组：训练时从合法组合里随机采样，评估时按固定顺序遍历保证每个测试用例被覆盖且不重复计数。

### Isaac Lab Arena（大规模仿真）

`IsaacLabArenaEnv`（`envs/arena/arena_env.py`）体量更大，支持多种机器人本体（Unitree G1 人形、Fourier GR1 人形、Franka 机械臂）。它把"这个机器人具体怎么控制"下沉到独立的 `ArenaEmbodiment` 适配器——`JointSpaceEmbodiment`（关节空间控制，覆盖 G1 和 GR1）、`TaskSpaceEmbodiment`（任务空间/末端位姿控制，覆盖 Franka）。这样 `IsaacLabArenaEnv` 本身完全不需要知道自己在控制哪种机器人，只需要调用 `self.embodiment.policy_to_sim_action(...)` 做策略动作到仿真器动作的转换。

Arena 环境有一个值得单独说明的细节——**分层的自动重置机制**：IsaacLab 底层的 `ManagerBasedRLEnv.step()` 会在环境终止/超时时**立刻**（intra-chunk，块内）把那个环境重置好，返回重置后的观测；但因为一个动作块可能还没执行完，`BaseEnv._reset_done_envs` 会在**动作块执行完毕的边界**再做一次二次重置，确保下一个动作块从一个干净的、刚重置好的状态开始，不会出现"块内已经被 IsaacLab 内部重置过一次，紧接着又用同一批本该结束的观测继续走剩余步数"的语义错乱。

### Piper（真机机械臂）

`PiperEnv`（`envs/piper/piper_env.py`）是三者里跟仿真环境差异最大的一个——但对外接口和仿真环境完全一致，这正是 `BaseEnv` 抽象价值的最好证明。真机环境几个本质区别：

- **强制 `num_envs=1`**：一台真机不能"并行跑多个环境实例"。
- **reward/success 永远是 0/False**：真机没有仿真器那种"任务成功检测"的先验知识，完全依赖操作员按键手动标记（`manual_reward`/`force_truncated`，跟第 2 节的遥操作接管机制走的是同一套底层通道）。
- **动作转换用数值逆运动学**：末端位姿的增量指令需要转成关节角增量，`_PiperDifferentialIK` 用数值雅可比（`_numerical_jacobian`，对每个关节施加微小扰动、观测末端位姿变化算出雅可比列）加带关节限位约束的最小二乘（`scipy.optimize.lsq_linear`）求解。
- **图像来自后台线程持续读取的摄像头**：`_PiperCameraStream` 用独立线程跑 `cv2.VideoCapture`，随时保留最新一帧供环境查询，避免主循环被摄像头 IO 阻塞。
- **reset 需要真实驱动机械臂运动**：`reset_to_initial_pose` 要下发运动指令并**轮询等待**机械臂真正到达目标关节角（带超时和容差判断），跟仿真器"reset 瞬间完成"完全不同。

尽管实现细节天差地别，`PiperEnv.env_step` 返回的字典 schema 跟 `LiberoEnv`/`IsaacLabArenaEnv` 完全一致——上层的动作块执行、人类接管、录制逻辑不需要为真机写任何特殊分支。

## 5. Recorder：怎么把环境交互写成 LeRobot 数据集

`MultiRecorder`（`recorder/recorder.py`）是录制侧的入口，把每一次环境 transition 同时分发给多个具体的 recorder 实现：

```python
class MultiRecorder(BaseRecorder):
    def record_once(self, **kwargs):
        for recorder in self.recorders:
            recorder.record_once(**kwargs)
```

内置两种实现：`LeRobotDatasetRecorder`（写训练可用的 LeRobot 数据集）和 `VideoRecorder`（写带标注的可视化视频）。两者共享同一份数据，互不干扰。

`LeRobotDatasetRecorder` 的关键设计是**先内存缓冲、episode 结束才真正落盘**：

```python
def record_once(self, *, env_id=0, observation, action, task, ...):
    self._pending_frames[env_id].append(self.strategy.make_frame(observation=..., action=..., task=...))

def save_episode(self, env_id=0):
    for pending_frame in self._pending_frames[env_id]:
        self.dataset.add_frame(dict(pending_frame))
    self.dataset.save_episode()
    self._pending_frames[env_id].clear()
```

**为什么要缓冲而不是每步都写盘**：如果一个 episode 中途因为人类按下"重来"键而被放弃，已经写入磁盘的帧就成了脏数据，需要额外清理逻辑。用内存缓冲的方式，只有当 `save_episode` 真正被调用（episode 正常终止/截断）时才写盘，中途放弃直接调用 `clear_episode` 清空缓冲区即可，磁盘上永远只有完整的 episode。

`BaseLeRobotStrategy` 是环境和 recorder 之间的"schema 翻译层"——每种环境实现自己的 `features()`（定义 LeRobot 数据集需要哪些字段、每个字段的类型/形状）和 `make_frame()`（把一步环境交互转成一条符合 schema 的记录）。新增一种环境类型的录制支持，只需要实现这一个策略类并注册到 `registry.py`，不需要改动 `MultiRecorder`/`LeRobotDatasetRecorder` 的核心代码——这是典型的策略模式，把"什么样的数据该怎么存"和"数据存储的通用流程"解耦。

`AsyncRecorder` 是一层可选的性能优化包装——把实际的写盘操作挪到后台线程，`record_once`/`save_episode` 立即返回不阻塞主训练循环，`pop_completed()`/`finalize()` 是明确的同步点（会等待所有排队中的写盘操作完成）。

## 6. Teleop：设备输入到机器人动作的转换链路

人机协同的另一半是遥操作输入怎么变成动作。verl-vla 把这条链路拆成三个独立职责：

- **Device**（`teleop/devices/`）：只做"把浏览器/硬件事件变成一份稳定的设备状态快照"，比如 `KeyboardDevice` 记录当前按下了哪些键。它**完全不知道**"W 键应该对应什么机器人动作"。
- **Intervention Strategy**（`teleop/strategies/`）：针对每个 `(环境类型, 设备类型)` 组合定义"设备状态怎么翻译成这个环境需要的动作格式"。比如 `LiberoKeyboardStrategy` 把按键状态翻译成末端位姿的增量（`pos(3) + rotvec(3) + gripper(1)`）。
- **TeleopController**（`teleop/teleop_controller.py`）：把设备和策略连接起来，同时负责向浏览器面板发布观测、把人类动作应用到环境执行流程里。

**为什么要这样分层**：同一个键盘设备，控制 LIBERO 里的 Franka 机械臂和控制 Piper 真机机械臂，"W 键该转换成什么动作"的语义是不同的（不同的坐标系、不同的灵敏度、不同的动作维度）。把这个语义翻译逻辑放在 Strategy 层而不是 Device 层，使得同一个 `KeyboardDevice` 类可以被任意环境复用，新增一种环境的遥操作支持只需要写一个新的 Strategy 类，不需要碰设备层代码。

Strategy 提供两条动作路径：`apply_action(action, device)` 在策略正在自主运行时，检测到接管就替换或修改策略给出的动作（对应第 2 节的接管机制）；`get_action(device)` 用于纯遥操作/纯人工演示录制场景（`BaseEnv.record()`），直接产出人类想要的动作，不需要一个"被覆盖的策略动作"作为输入。这两条路径共享同一套坐标转换逻辑，保证演示数据、干预纠正数据、和策略 rollout 数据用的是完全一致的动作语义。

## 小结

| 概念 | 要点 |
|---|---|
| BaseEnv 契约 | 子类只需实现 env_init/env_reset/env_step/env_close，返回固定 schema 字典，其余能力（动作块、接管、录制、遥操作发布）自动获得 |
| 动作块执行 | 策略输出一次多步动作块，BaseEnv 逐步拆解执行，每步检查是否被人类接管 |
| 实时接管 | 一旦某环境被标记接管，该动作块剩余步全部使用人类实时动作，不回退执行策略原有规划 |
| Arena 分层重置 | IsaacLab 内部块内即时重置 + BaseEnv 在块边界的二次重置，保证语义清晰 |
| Piper 真机 | num_envs=1、reward/success 靠人工按键、动作转换用数值 IK、返回 schema 与仿真环境完全一致 |
| Recorder 缓冲策略 | 内存缓冲直到 episode 正常结束才落盘，避免中途放弃产生脏数据 |
| Strategy 模式 | BaseLeRobotStrategy 让新环境接入录制只需写一个策略类 |
| Teleop 三层分工 | Device(纯输入)/Strategy(语义翻译)/Controller(编排)，同一设备可服务任意环境 |

## 下章预告

[第 08 章](./08_SAC训练循环_EpisodeBuffer_ReplayPool与RLPD) 从环境侧产出的原始轨迹数据出发，讲清楚 `EpisodeBuffer` 怎么把 rollout 输出的碎片步收集成完整 episode、`SACReplayPool` 的双池采样机制、RLPD 怎么混合离线示教数据、以及 critic 的 Bellman target 和 target 网络更新的具体实现。
