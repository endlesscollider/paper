---
title: "参考系切换与手部降维：policy_frame_id 与 hand_synergy PCA"
series:
  id: stack_can_drawer_migenrl_deep_dive
  chapter: 4
order: 4
---

# 第 04 章：参考系切换与手部降维：policy_frame_id 与 hand_synergy PCA

> 前情提要：第 03 章讲了 `target_object_frame` 用目标物体的初始位姿作参考系。但这个任务分几个阶段（先抓取、后放置），参考物体在不同阶段是不同的（先是被抓取的物体，后是承接的容器）——参考系怎么随任务推进切换？这是本章第一部分要讲的 `policy_frame_id` 机制。第二部分讲另一个独立的表征问题：22 维的手指关节怎么压缩成 6 维。

## 1. 参考系切换的问题

回顾配置里的这一行：

```yaml
action_context_change_source: policy_frame_id
```

第 03 章讲到，`target_object_frame` 需要一个"参考物体"来确定坐标系原点。但这个任务分几个阶段，每个阶段该参照的物体是不同的：

```text
阶段1：右手抓取物体A        → 参考物体应该是 A
阶段2：左手操作容器B（比如打开）  → 参考物体应该是 B
阶段3：右手把A放进B          → 参考物体应该是 B（A要往B里放，位置该以B为参照）
阶段4：左手收尾操作B         → 参考物体应该是 B
```

策略在训练和推理时都需要知道：**当前这一帧，该用哪个物体的初始位姿作为坐标原点**。这个信息不能是隐含的，必须显式编码成一个数字告诉网络，否则网络无法区分"手腕相对物体A的位置是 (0.1, 0, 0.05)"和"手腕相对物体B的位置是 (0.1, 0, 0.05)"——这是两个完全不同的物理状态，但如果不告诉网络参考系是什么，这两个数值在网络输入里看起来一模一样。

## 2. frame_id：把物体名字变成一个整数

`context_rules.py` 里定义了一个简单的映射规则——**按环境侧场景定义里物体出现的顺序，给每个物体分配一个整数 ID**（这份物体清单来自 `env.cfg_path` 指向的环境配置，`miGenRL` 只是读取这个清单的顺序，不关心清单本身怎么来的）：

```python
def scene_object_id_map(names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}
```

假设场景里的物体按声明顺序是 `[物体A, 物体B, 物体C]`，那么：

$$
\text{物体A} \to 0,\qquad \text{物体B} \to 1,\qquad \text{物体C} \to 2
$$

这个映射是根据 `env.cfg_path` 指向的环境配置静态推出的，训练和 rollout 两个阶段都要用同一份映射规则，否则数字和物体名字的对应关系会错位。

### 2.1 数据记录阶段：把 frame_id 存进 HDF5

采集示教数据时，每一帧都要记录"当前活跃任务的目标物体是谁"，进而转换成 frame_id：

```python
scene_object_ids = self._scene_object_ids()          # {"物体A": 0, "物体B": 1, "物体C": 2}
for env_id in recording_env_ids:
    task_index, task = task_by_env[env_id]
    target_name = self._task_object_names(task)        # 例如当前阶段的目标物体是"物体A"
    phase_index[env_id] = float(task_index)             # 任务在依赖链里的顺序编号
    frame_id[env_id] = float(scene_object_ids.get(target_name, -1))
```

这两个标量被记录进 HDF5 的 `geometry_obs/frame_id` 和 `geometry_obs/phase_index` 字段——每一帧一个数字，随着任务推进而改变。当环境判定当前阶段任务完成、进入下一阶段时，这一帧记录的 `frame_id` 就会从一个物体的编号跳到另一个物体的编号。

### 2.2 训练阶段：frame_id 变成网络输入的一部分

```python
def read_frame_id_policy_context(demo, frame_index, context_cfg):
    context = demo["geometry_obs/frame_id"][frame_index]
    context = np.asarray(float(context), dtype=np.float32).reshape(-1)   # shape (1,)
    if context_cfg.get("append_phase_index"):
        phase = demo["geometry_obs/phase_index"][frame_index]
        context = np.concatenate([context, phase.reshape(-1)])            # shape (2,)
    return context
```

配置里 `policy_context.append_phase_index: true`，所以最终喂给网络的 `policy_context` 是一个 **2 维向量** `[frame_id, phase_index]`。网络（第 06 章会讲到）把这个向量当作一个条件信号，通过 embedding 查表的方式注入到多个网络层里——本质上是告诉网络"现在是第几阶段、该用哪个物体作参照"，让同一个网络能够处理任务的不同阶段，而不需要给每个阶段单独训练一个模型。

### 2.3 Rollout 阶段：frame_id 驱动参考系实时切换

训练时 frame_id 是从示教数据里读出来的固定值，但在 rollout（策略实际控制仿真）时，frame_id 需要**实时根据当前任务进度动态确定**，这是一条独立的执行链：

```python
class ActionContext:
    @staticmethod
    def change_keys(action_context_by_env, context_cfg, rollout_cfg):
        source = rollout_cfg.get("action_context_change_source")   # "policy_frame_id"
        if source in {"policy_context", "frame_id", "policy_frame_id"}:
            frame_values = cls.policy_values(action_context_by_env, context_cfg)
            return [("policy_context", value) for value in frame_values]
```

检测到 frame_id 发生变化后，`FrameBinding` 把新的 frame_id 写入 rollout 配置：

```python
class FrameBinding:
    def apply_current(self, rollout_cfg):
        rollout_cfg["_active_frame_ids_by_env"] = self.current_frame_ids
```

最后，坐标转换函数根据这个实时更新的 frame_id，去查询对应物体的初始位姿：

```python
def active_frame_pose_from_ids(env_unwrapped, rollout_cfg, *, device, dtype):
    frame_ids = rollout_cfg.get("_active_frame_ids_by_env")
    for env_id in range(num_envs):
        frame_id = int(round(float(frame_ids[env_id])))
        object_name = object_names[frame_id]                # 0 → 某物体名, 1 → 另一物体名, ...
        poses[env_id] = initial_object_pose_w(env_unwrapped, object_name, ...)
    return poses
```

这条链路串起来是：**任务进度变化 → frame_id 变化 → 触发参考系重新绑定 → 后续动作的坐标转换改用新物体的初始位姿**。整个过程对策略网络是透明的——网络只需要接收 `policy_context` 向量作为输入的一部分、输出相对当前参考系的动作，具体"当前参考系是哪个物体的哪个初始位姿"这件事完全由 rollout 侧的坐标转换代码负责，网络不需要自己去做这个查表。

### 2.4 为什么这套机制是必要的

如果没有 `frame_id` 这套动态切换机制，会出现两个后果之一：要么整条轨迹被迫始终使用同一个物体（比如永远相对第一个抓取的物体的初始位置）作参考系——这在"放入容器"这类后续阶段完全不合理，因为这个阶段的动作模式应该相对容器描述，而不是相对已经被抓在手里、位置已经不再变化的物体；要么需要给每个任务阶段单独训练一个策略网络，人为切割成多个模型，丢失了不同阶段之间可能共享的运动模式（比如"手靠近一个目标物体"这个子动作在不同阶段可能有相似的运动学结构）。用一个显式的 `frame_id` 让单一网络在训练和推理时都能正确区分"现在该用哪个物体做参照"，是让一个模型端到端处理多阶段任务的关键设计。

## 3. hand_synergy：把 22 维手指关节压成 6 维

现在转向一个独立的问题。配置里另一行：

```yaml
hand_synergy_path: runs_base/migenrl/baselines/stack_can_drawer_fixed_baseline/hand_synergy_dim6.pkl
```

这个任务用的双臂机器人每只手有 22 个关节自由度。如果直接把 22 维手指关节角度作为网络的动作输出维度，会带来两个问题：一是网络要学习一个 22 维的精细协调输出，训练难度显著增加；二是人手在自然抓握动作中的实际自由度远小于 22——大部分手指弯曲是同步的（食指弯曲的同时中指、无名指往往一起弯曲），把这种冗余暴露给网络学习是不必要的负担。

`hand_synergy` 用 [PCA](/前置知识/002j_前置知识_主成分分析PCA) 从示教数据里学出这套"手指协同运动模式"，把 22 维压缩到 6 维。如果对 PCA 本身的数学原理不熟悉，建议先读上面链接的前置知识文章，本节只讲这套机制在这个具体项目里的工程实现。

### 3.1 拟合：从示教数据中学出主方向

```python
HAND_JOINT_DIM = 22
HAND_LATENT_DIM = 6

def _fit_pca(samples: np.ndarray, latent_dim: int) -> dict[str, np.ndarray]:
    mean = samples.mean(axis=0)                          # (22,)
    centered = samples - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:latent_dim]                           # (6, 22)
    variance = singular_values ** 2 / max(len(samples) - 1, 1)
    explained = variance[:latent_dim] / variance.sum()
    return {"mean": mean, "components": components, "explained_variance_ratio": explained}

def fit_hand_synergy(left_samples, right_samples) -> dict:
    synergy = {"type": "pca", "latent_dim": 6}
    synergy["left"] = _fit_pca(left_samples, 6)
    synergy["right"] = _fit_pca(right_samples, 6)
    return synergy
```

注意左右手是**分别独立拟合**的——左手的 22 维关节样本和右手的 22 维关节样本各自算出一组 `mean` 和 `components`。这是合理的，因为左右手虽然结构镜像对称，但具体抓握时的角度分布可能因为惯用手习惯、任务分工（比如右手负责精细抓取、左手负责操作容器，两者的动作模式本身就不同）而不完全对称。拟合结果被序列化成 `hand_synergy_dim6.pkl`，文件名里的 `dim6` 就是 `HAND_LATENT_DIM=6` 的直接体现。

### 3.2 编码与解码：训练数据处理和网络输出解码

**编码**（训练数据预处理阶段，把示教数据里记录的原始 22 维手指角度压缩成网络要学习的 6 维目标）：

```python
def _encode_torch(values: torch.Tensor, side: dict) -> torch.Tensor:
    mean = torch.as_tensor(side["mean"])              # (22,)
    components = torch.as_tensor(side["components"])  # (6, 22)
    return (values - mean) @ components.T             # (..., 22) @ (22, 6) = (..., 6)
```

**解码**（rollout 阶段，把网络输出的 6 维隐向量还原成 22 维完整关节角度，才能真正驱动仿真里的手指关节）：

```python
def _decode_torch(values: torch.Tensor, side: dict) -> torch.Tensor:
    mean = torch.as_tensor(side["mean"])
    components = torch.as_tensor(side["components"])
    return values @ components + mean                  # (..., 6) @ (6, 22) + (22,) = (..., 22)
```

这两个函数和前置知识文章里推导的编码/解码公式完全对应，唯一的区别是这里区分左右手各用一套独立的 `mean`/`components`。

### 3.3 对整条动作向量的影响：62 维变 30 维

回顾第 02 章推算出的 `state_dim=30`、`action_dim=30`，现在可以看到这个数字具体是怎么来的。动作向量的完整布局（`action_layout.py` 里定义的 `DUAL_ARM_PCA_ACTION_DIM`）：

| 切片 | 内容 | 维度 |
|---|---|---|
| `[0:3]` | 左臂末端位置 | 3 |
| `[3:9]` | 左臂末端旋转（rot6d） | 6 |
| `[9:12]` | 右臂末端位置 | 3 |
| `[12:18]` | 右臂末端旋转（rot6d） | 6 |
| `[18:24]` | 左手手指（hand_synergy 隐向量） | 6 |
| `[24:30]` | 右手手指（hand_synergy 隐向量） | 6 |
| **合计** | | **30** |

对比不启用 PCA 时的原始布局（`DUAL_ARM_ACTION_DIM=62`），差异只在最后两段——左右手指分别从 22 维压到 6 维，其余臂部位姿部分不受影响：

$$
\underbrace{3+6+3+6}_{\text{双臂位姿}=18} + \underbrace{22+22}_{\text{双手原始}=44} = 62 \quad\longrightarrow\quad 18 + \underbrace{6+6}_{\text{双手PCA后}=12} = 30
$$

Rollout 阶段解码的调用点在每个 `*_action_to_env_action` 函数的最开头：

```python
def decode_torch_by_layout(action: torch.Tensor, synergy: dict | None) -> torch.Tensor:
    if not synergy:
        return action
    latent_dim = int(synergy["latent_dim"])                 # 6
    if action.shape[-1] == 18 + latent_dim * 2:              # 30 == 18+12
        left_pose = action[..., :9]                           # 位姿部分不需要解码
        right_pose = action[..., 9:18]
        left_hand = _decode_torch(action[..., 18:24], synergy["left"])    # (B,6) → (B,22)
        right_hand = _decode_torch(action[..., 24:30], synergy["right"])  # (B,6) → (B,22)
        return torch.cat([left_pose, right_pose, left_hand, right_hand], dim=-1)  # (B, 62)
```

网络自始至终只在 30 维的压缩空间里预测和学习，只有在最后一步——真正要把动作写入仿真物理引擎之前——才解码回 62 维的完整关节角度。这个设计的边界很清晰：PCA 压缩只发生在"网络看到/输出什么"这一层，仿真引擎本身、IK 控制器、任务成功判定逻辑全部工作在原始的完整维度上，对它们而言 PCA 压缩完全透明。

## 4. 两套机制的分工

这一章讲的 `policy_frame_id` 和 `hand_synergy` 是两个独立、互不依赖的表征机制，容易被混淆，这里做一个清晰的切分：

| 机制 | 解决什么问题 | 作用范围 |
|---|---|---|
| `policy_frame_id` | 位姿应该相对哪个坐标系表达 | 影响双臂末端位置和旋转（位置 3 维 + 旋转 6 维部分） |
| `hand_synergy` | 手指关节自由度太多、有冗余 | 只影响手指关节维度（22 维 → 6 维） |

两者可以独立配置、独立启用或禁用——一个任务可能只需要 `target_object_frame` 而不需要手部降维（比如末端只是一个简单夹爪，没有 22 个关节），另一个任务可能是单臂固定基座、不需要参考系切换，只需要手部降维。本任务恰好两者都用到，是因为它既是多阶段任务（需要参考系切换）又用的是高自由度灵巧手（需要降维）。

## 5. 小结与下一章

到这里，第 03、04 两章把"网络看到什么、输出什么"这个问题从三个维度拆解完毕：

1. **旋转怎么表示**——rot6d，6 个数字表示朝向，满足连续映射所需的最小维度。
2. **位姿相对哪里表示**——`target_object_frame`，相对目标物体的初始位姿，且通过 `policy_frame_id` 机制支持任务推进时动态切换参考物体。
3. **手指关节怎么压缩**——`hand_synergy`，用 PCA 把 22 维手指关节压缩到 6 维，只在网络输入输出层生效，仿真和任务判定逻辑不受影响。

最终得到的 `state_dim=30`、`action_dim=30`，是这三个设计叠加后的结果——它们共同决定了策略网络实际要处理的数据形状。下一章从数据源头开始：这些示教数据是怎么存储在 HDF5 里的，训练时又是怎么从长达数百步的完整轨迹里采样出一个个训练样本的（keyframe 采样策略）。

---

上一章：[第 03 章 数据表征：rot6d 与参考系](./03_数据表征_rot6d与参考系) ｜ 下一章：[第 05 章 数据管线](./05_数据管线)
