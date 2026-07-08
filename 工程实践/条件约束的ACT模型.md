---
title: 条件约束的 ACT 模型
order: 7
tags: [模仿学习, ACT, 双臂]
category: 工程实践
star: 3
---

# 面向双手操作的 Conditioned ACT 设计文档：从 language-conditioned ACT 到结构化条件约束

**日期**：2026-06-28  
**场景**：MiGenRL / embodied-arena，双手 full-action policy 在 `stack_can_into_drawer_bimanual` 任务中出现 inactive arm 漂移、phase 切换后坐标系语义不清、ACT/InterACT train-layout debug rollout 成功率低的问题。  
**目标**：不是引入自然语言作为用户交互接口，而是借鉴 language-conditioned ACT / VLA / 多任务机器人 Transformer 中“高优先级条件如何进入模型”的架构，把自然语言替换成更严格、更可控、更接近控制系统的结构化条件：phase、arm role、reference frame、target object、action mask、hold target、goal-in-frame、frame-relative geometry 等。

---

## 0. 背景：为什么 `phase_index` 拼到 qpos 里是不够的

当前我们讨论的问题不是“ACT 和 InterACT 哪个更强”，而是“模型是否真的知道当前动作应该在什么任务条件、什么坐标系、什么手臂角色约束下产生”。已有实验显示，在 train-layout debug rollout 中，ACT 6 条全失败，InterACT 也只有 1/6；失败主要卡在 phase 1，phase 0 中左手仍然出现明显漂移。这说明问题不是某个 policy variant 特有的，而是双手 full-action policy 在数据表示和模型条件注入上没有把任务结构讲清楚。

原始 ACT 的核心思想来自 Action Chunking Transformer：模型从当前观测预测未来一段 action chunk，以缓解高频控制下的误差累积和非平稳示教问题。原始论文强调 ACT 学习 action sequence 的生成模型，并通过 action chunking 和 temporal ensembling 改善闭环表现。参考：ACT 原论文《Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware》指出 ACT 是一个学习动作序列生成模型的方法，用于高精度双手任务，并在真实机器人上完成开杯、插电池等细粒度操作任务：https://arxiv.org/abs/2304.13705 。

在本项目当前 ACT 实现中，关键路径在：

- `source/miGenRL/policy.py`：`ACTPolicy.forward(qpos, image, actions, is_pad)` 调用 `self.model(qpos, image, env_state, actions, is_pad)`。
- `source/miGenRL/detr/models/detr_vae.py`：`DETRVAE.forward()` 中，训练时用 `encoder_joint_proj(qpos)` 将 qpos 投影成一个 token，用 action sequence 编码出 CVAE latent；推理/解码时用 `input_proj_robot_state(qpos)` 得到 `proprio_input`，与 `latent_input` 一起作为额外 token 拼到视觉 tokens 前。
- `source/miGenRL/detr/models/transformer.py`：视觉 feature 被 flatten 成 transformer memory，`latent_input` 和 `proprio_input` 作为两个前缀 token 进入 encoder memory；decoder 以 `query_embed` 表示 action chunk 中每个未来步的 query，cross-attend encoder memory 后输出每个 chunk step 的 action。

如果 `phase_index` 只是 qpos 中的一个标量，那么它会被 `input_proj_robot_state = nn.Linear(state_dim, hidden_dim)` 与关节、末端位姿、夹爪状态等连续物理状态混在一起投影成一个 `proprio_token`。这会带来几个结构性问题：

1. **条件没有独立 token 身份**：Transformer 不能单独 attend 到“phase 条件”；它只能看到混合后的 proprio token。phase 的语义被线性层压缩，没有显式边界。
2. **条件没有类型信息**：phase、ref frame、active arm、action mask 是离散控制条件；joint/qpos 是连续状态。把离散控制条件当作普通连续数值会引入伪顺序和伪距离，例如 phase 2 不一定比 phase 1 “大一倍”。
3. **条件没有层级优先级**：在大模型中，重要 prompt 往往以独立 token、prefix memory、cross-attention memory、adapter、FiLM/AdaLN、router 或 decoding constraint 形式影响模型；不是把 system prompt 压成一个 float 拼到输入向量里。
4. **坐标系切换没有被建模**：phase 只是时间段编号，不等价于“当前动作应在 can frame / drawer handle frame / drawer frame 下解释”。如果动作 label 仍是 world/base frame 下的 full action，模型必须同时学习视觉识别、phase 语义、frame 选择、双手角色、动作分布切换，这对小数据 BC 非常困难。
5. **inactive arm 没有输出约束**：即使输入里有 phase，输出空间仍允许左右手每个维度自由变化。BC 的 L1 loss 会在有限数据、闭环偏移、动作多峰时给 inactive arm 产生漂移，执行时漂移再改变视觉和接触状态，造成后续 phase 更难。

因此，本文档不主张“用语言描述替代 phase”，而是主张：**把语言条件方法中有效的结构拿过来，把 text embedding 换成结构化控制条件，并进一步加入硬约束和 frame-aware action 表示**。下面分五种方法详细展开，每一种都说明借鉴自哪里、条件如何进入 Transformer、在原本 ACT 上怎么改、训练和 rollout 如何保持一致、优缺点和适合的落地顺序。

---

## 方法一：独立 Condition Tokens —— NL-ACT / Octo 风格的最小架构改造

### 1.1 方法思想：把重要条件从 qpos 中拆出来，变成 Transformer 可见的独立 token

最直接、最容易落地的方法，是把 `phase_index`、`active_arm`、`left_role`、`right_role`、`ref_frame`、`target_object`、`action_mask`、`goal_pose_in_ref`、`hold_pose_in_ref` 等条件从 qpos 中拿出来，经过专门的 embedding/MLP，形成一个或多个 condition tokens，然后作为 prefix tokens 与原始 ACT 的 latent token、proprio token、image tokens 一起进入 Transformer encoder memory。这类方法对应 language-conditioned ACT 和 generalist robot policy 中最常见的一类：任务条件不是输入状态向量中的一维，而是独立任务 token。

NL-ACT 是最贴近 ACT 的例子。NL-ACT 明确是原 ACT 的 fork，它将自然语言 instruction 用 sentence encoder 预先转成 embedding，再把 instruction embedding 整合进 ACT Transformer encoder 输入，用于多任务机器人操作。NL-ACT README 说它修改 ACT 以接收由 `all-mpnet-base-v2` 生成的任务 instruction embedding，并将其集成到 Transformer encoder input：https://github.com/krohling/nl-act 。这对我们很关键：这里“自然语言”不是核心，核心是“任务条件被独立编码，并进入 Transformer 的 token/memory 空间”。如果把自然语言 embedding 换成结构化 condition embedding，本质就是 structured-condition ACT。

Octo 也体现了同样思想。Octo 是开源通用机器人策略，它把任务定义通过特定 tokenizer 变成 task tokens，再和观测 tokens 一起进入 Transformer backbone。Octo 支持 language instruction 和 goal image 等任务形式，核心架构是 task tokens + observation tokens，而不是把 task id 塞到 proprio 的一维里。参考 Octo 论文页面：https://arxiv.org/abs/2405.12213 。对我们来说，`phase/ref_frame/arm_role/mask/goal` 就是 task tokens，只不过它们不是自然语言，而是更精确的控制条件。

这种方法的核心改变可以概括为：

$$
原本：
  qpos = [joint, ee, gripper, phase_index]
  proprio_token = Linear(qpos)
  encoder memory = [latent_token, proprio_token, image_tokens]

改造后：
  qpos = [joint, ee, gripper]              # 不再含 phase_index
  condition = {phase, ref_frame, roles, mask, goal, hold, geometry}
  condition_tokens = ConditionEncoder(condition)
  proprio_token = ProprioEncoder(qpos)
  encoder memory = [latent_token, condition_tokens..., proprio_token, image_tokens]
$$

这里最重要的是“条件独立化”。Transformer 的 attention 机制天然适合处理多个 token 之间的关系；只要条件有独立 token，action query 就可以在 decoder cross-attention 中直接 attend 到 `phase_token`、`frame_token`、`role_token`、`mask_token`，而不是从一个混合的 proprio token 中反推哪个维度代表 phase。

### 1.2 应该拆出哪些 condition token

不要只拆 `phase_token`。phase id 只是一个状态机编号，它并不等价于可执行控制约束。对 `stack_can_into_drawer_bimanual` 来说，至少要拆以下几类：

1. **phase token**：表示当前处于 grasp、open drawer、place、retreat 等阶段。它有助于区分动作分布，但不应该单独承担全部语义。
2. **reference frame token**：表示当前动作 label/goal/几何关系应在哪个 frame 下解释，例如 robot_base、world、can、drawer_handle、drawer_inside、drawer_front 等。这个 token 是解决“没有识别坐标系转换”的关键。
3. **arm role tokens**：左手/右手分别是什么角色，例如 inactive_hold、active_grasp、active_pull、active_place、support_hold_object、preposition 等。双手 full-action policy 的最大问题之一是左右手动作分布混杂；role token 可以显式告诉模型每只手的职责。
4. **target object token**：当前主要操作对象是谁，例如 can、bottle、drawer_handle、drawer_box、tabletop。对于多对象场景，phase 相同但 target 不同会导致完全不同的目标位置和动作方向。
5. **action mask token**：每个 action 维度是否允许自由预测。它既可以作为 soft condition 让模型知道输出空间，也可以在输出层做 hard projection。mask 不一定只有 0/1，也可以是连续权重，表示某些维度允许小幅运动。
6. **goal pose token**：当前 phase 的目标末端位姿或目标对象位姿，建议表示在 ref frame 下。比如 phase 1 左手应相对于 drawer_handle 往外拉，goal 可以是 pull axis 方向上的 delta 或目标 EE pose。
7. **hold pose token**：inactive arm 或 holding object 的保持目标。比如右手已经夹住 can，phase 1 中右手不是“原地不动”，而是“保持 gripper-can 相对位姿/保持 can 稳定”。这个 hold target 必须显式给模型。
8. **frame-relative geometry tokens**：左/右 EE 在 ref frame 下的 pose，target 在 ref frame 下的 pose，goal 与当前 EE 的相对误差。模型不应该只从 RGB-D 中隐式学习这些几何量；这些几何条件可以显著降低学习难度。

离散条件使用 `nn.Embedding`，连续条件使用小 MLP。比如：

```python
class StructuredConditionEncoder(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.phase_embed = nn.Embedding(num_phases, hidden_dim)
        self.frame_embed = nn.Embedding(num_frames, hidden_dim)
        self.left_role_embed = nn.Embedding(num_roles, hidden_dim)
        self.right_role_embed = nn.Embedding(num_roles, hidden_dim)
        self.target_embed = nn.Embedding(num_targets, hidden_dim)
        self.mask_proj = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.goal_proj = nn.Sequential(nn.Linear(goal_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.hold_proj = nn.Sequential(nn.Linear(hold_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.geom_proj = nn.Sequential(nn.Linear(geom_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, cond):
        tokens = [
            self.phase_embed(cond["phase_id"]),
            self.frame_embed(cond["ref_frame_id"]),
            self.left_role_embed(cond["left_role_id"]),
            self.right_role_embed(cond["right_role_id"]),
            self.target_embed(cond["target_object_id"]),
            self.mask_proj(cond["action_mask"]),
            self.goal_proj(cond["goal_pose_in_ref"]),
            self.hold_proj(cond["hold_pose_in_ref"]),
            self.geom_proj(cond["geometry_in_ref"]),
        ]
        return torch.stack(tokens, dim=1)  # [B, N_cond, H]
```

这里 condition token 的数量不需要很大，8 到 16 个 token 足够。关键是每类 token 的语义明确，便于 debug。我们可以在 rollout 时保存每一步 condition，后续视频和 action 曲线可以对齐分析“phase 1 时 ref_frame 是否真的是 drawer_handle、left role 是否 active_pull、right role 是否 hold_object”。

### 1.3 在当前 ACT 代码上怎么改

当前 `DETRVAE.forward(qpos, image, env_state, actions, is_pad, inference_latent)` 的签名没有 `condition`。第一步需要让 `ACTPolicy.forward()`、dataset batch、training loop、rollout policy 都支持可选 condition：

```python
def forward(self, qpos, image, actions=None, is_pad=None, condition=None, return_gate=False):
    model_output = self.model(qpos, image, env_state, actions, is_pad, condition=condition)
```

然后在 `DETRVAE.__init__()` 中新增：

```python
self.condition_encoder = StructuredConditionEncoder(hidden_dim, action_dim)
self.condition_pos_embed = nn.Embedding(num_condition_tokens, hidden_dim)
```

在视觉分支中，当前代码大致是：

```python
proprio_input = self.input_proj_robot_state(qpos)
src = torch.cat(all_cam_features, axis=3)
pos = torch.cat(all_cam_pos, axis=3)
hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
```

现有 `Transformer.forward()` 只支持两个 additional tokens：latent 和 proprio。因此有两种改法。

**改法 A：最小侵入，扩展 `additional_tokens` 参数**

把 `Transformer.forward()` 从：

```python
def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
```

改成：

```python
def forward(self, src, mask, query_embed, pos_embed, additional_tokens=None, additional_pos_embed=None):
```

其中 `additional_tokens` 是 `[B, K, H]`，进入 transformer 时转成 `[K, B, H]`，再与 image tokens 拼接：

```python
additional_tokens = additional_tokens.permute(1, 0, 2)
src = torch.cat([additional_tokens, image_tokens], dim=0)
pos_embed = torch.cat([additional_pos_embed, image_pos], dim=0)
```

`DETRVAE.forward()` 里构造：

```python
proprio_input = self.input_proj_robot_state(qpos).unsqueeze(1)
latent_token = latent_input.unsqueeze(1)
condition_tokens = self.condition_encoder(condition)
additional_tokens = torch.cat([latent_token, condition_tokens, proprio_input], dim=1)
```

position embedding 对应扩展为：

```python
# token order: latent, cond_0...cond_N-1, proprio
prefix_pos = torch.cat([
    self.latent_pos.weight.view(1, 1, -1).expand(bs, -1, -1),
    self.condition_pos_embed.weight.view(1, n_cond, -1).expand(bs, -1, -1),
    self.proprio_pos.weight.view(1, 1, -1).expand(bs, -1, -1),
], dim=1)
```

**改法 B：不动 transformer 签名，把 condition tokens 折进 src**

也可以在 `DETRVAE.forward()` 中直接把 image feature flatten 成 sequence，自己拼 condition，再调用 encoder/decoder。但这会改动更大，不如改法 A 简洁。

训练 CVAE encoder 也要同步。当前训练时 latent encoder 输入是：`[CLS, qpos_embed, action_embed...]`。如果 condition 会影响 action chunk 的分布，那么 CVAE posterior 也应该看到 condition，否则 posterior 在多 phase/多 frame 数据上会混淆。改法：

```python
qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
condition_summary = self.condition_latent_encoder(condition)  # [B, N_cond, H]
encoder_input = torch.cat([cls_embed, condition_summary, qpos_embed, action_embed], dim=1)
```

同时 `pos_table` 的长度要从 `1 + 1 + num_queries` 改成 `1 + n_cond + 1 + num_queries`。如果不想改固定 sinusoid table，也可以给 condition tokens 用 learned position embedding，再与 action positions 分开拼接。

### 1.4 Dataset 和 config 怎么改

训练数据需要从每个 timestep 读取 condition。HDF5/LeRobot 转换时建议保存两类字段：

```yaml
condition/phase_id: int64 [T]
condition/ref_frame_id: int64 [T]
condition/left_role_id: int64 [T]
condition/right_role_id: int64 [T]
condition/target_object_id: int64 [T]
condition/action_mask: float32 [T, action_dim]
condition/goal_pose_in_ref: float32 [T, goal_dim]
condition/hold_pose_in_ref: float32 [T, hold_dim]
condition/geometry_in_ref: float32 [T, geom_dim]
```

对于 ACT chunk，样本索引为 `t` 时，condition 可以取当前时刻 `condition[t]`，也可以取未来 chunk 对应的 `condition[t:t+num_queries]`。最小版本取当前 condition，假设一个 chunk 内 phase 不变；但我们的任务可能 phase 边界附近 chunk 会跨 phase，所以更稳的是两种策略：

1. **chunk 内 condition 序列化**：condition token 也带 query-step 维度，例如 `condition_seq[t:t+K]`。这更准确，但实现复杂。
2. **训练时避免跨 phase chunk**：采样 chunk 时，如果 chunk 跨 phase，则 truncate 到 phase 边界或把后半段 pad。这样一个 chunk 内 condition 一致，最符合 ACT “当前条件预测未来动作片段”的假设。

建议最小版本采用第 2 种。因为当前失败集中在 phase 0/1，先让每个 phase 内的动作分布清楚，比跨 phase 平滑更重要。之后再引入 condition sequence。

config 中新增：

```yaml
policy:
  condition_mode: tokens
  condition_token_fields:
    - phase_id
    - ref_frame_id
    - left_role_id
    - right_role_id
    - target_object_id
    - action_mask
    - goal_pose_in_ref
    - hold_pose_in_ref
    - geometry_in_ref
  remove_phase_from_qpos: true
  condition_num_tokens: 9
  condition_dropout: 0.05
```

`condition_dropout` 可以随机 drop 某些条件 token，以防模型过度依赖单个字段；但对当前 debug 任务，初期建议设为 0 或很小。

### 1.5 训练 loss 和 rollout 的一致性

仅加入 condition tokens 后，loss 可以先不改，仍然是原 ACT 的 action L1 + KL。但如果 action mask 已经作为 condition token 输入，建议至少记录 mask-wise 指标：

```python
active_l1 = (abs(pred - gt) * mask).sum() / mask.sum()
inactive_l1 = (abs(pred - gt) * (1 - mask)).sum() / (1 - mask).sum()
```

这能回答一个关键问题：condition token 是否真的让 inactive arm 更稳定。如果 inactive_l1 仍然很高，说明只靠输入 token 还不够，需要方法三/四/五。

rollout 时，policy 服务器或 `rollout_policy.py` 必须从环境当前状态构造同样的 condition。这个 condition 不能只来自 dataset，而要由 task phase manager / oracle metadata / frame geometry 实时生成。否则训练和部署不一致。比如 phase 1 时：

```python
condition = {
    "phase_id": PHASE_OPEN_DRAWER,
    "ref_frame_id": FRAME_DRAWER_HANDLE,
    "left_role_id": ROLE_ACTIVE_PULL,
    "right_role_id": ROLE_HOLD_OBJECT,
    "target_object_id": TARGET_DRAWER_HANDLE,
    "action_mask": make_phase1_mask(),
    "goal_pose_in_ref": compute_left_pull_goal_in_handle_frame(env),
    "hold_pose_in_ref": compute_right_hold_can_pose(env),
    "geometry_in_ref": compute_ee_target_geometry(env),
}
```

### 1.6 优点、局限和推荐落地顺序

优点：

- 改动最小，最接近 NL-ACT/Octo 的已验证思路。
- 不破坏原 ACT 的 decoder query/action chunk 结构。
- 容易 ablation：只加 phase token、再加 frame token、再加 role/mask/goal。
- debug 可解释性强，可以可视化 condition token 对 action query 的 attention。

局限：

- 它仍然是 soft conditioning。模型可以选择不听 condition，尤其在数据少、视觉误差大、action label 噪声大时。
- 如果 action 输出空间仍然允许 inactive arm 自由变化，左手漂移可能只降低但不会消失。
- 如果 action label 仍是 world/base frame，frame token 只能提示模型“应该换坐标系”，但没有从监督目标上减轻学习难度。

推荐作为第一步实现：**从 qpos 中移除 phase scalar，新增 condition tokens，并把 condition tokens 拼入 encoder memory**。这一步应当作为所有后续方法的基础，因为 FiLM、cross-attention、expert routing、hard mask 都需要一个统一的 `condition` 数据结构。

---

## 方法二：Prompt Memory / Cross-Attention —— VIMA 风格的条件记忆控制 action decoder

### 2.1 方法思想：不要只把 condition 放进 encoder，而要让 action decoder 每层强制读取 condition memory

方法一把 condition tokens 拼到 ACT encoder memory 中，action query 在 decoder cross-attention 时可以 attend 到这些 tokens。但在原 ACT 的 DETR-style Transformer 中，condition、latent、proprio、image tokens 全部混在同一个 memory 里，decoder 的 cross-attention 没有区分“任务条件 memory”和“视觉观测 memory”。如果图像 tokens 很多，而 condition tokens 很少，condition 在 attention 中可能被稀释。对于我们的双手任务，phase/ref_frame/role/mask 是高优先级约束，不应该只作为普通 memory token 被模型自己决定是否读取。

VIMA 提供了一个更强的思路。VIMA 的核心是 multimodal prompt：prompt 可以包含文字和视觉对象 token，策略不是简单接收一个 task id，而是用 prompt encoder 编码提示，再让 action decoder 通过 cross-attention 被 prompt conditioned。VIMA 项目主页称其为 “General Robot Manipulation with Multimodal Prompts”，强调 prompt 是机器人操作泛化的接口：https://vimalabs.github.io/ 。对我们而言，prompt 不需要是自然语言；可以是结构化 prompt：phase、frame、arm role、target、mask、goal、hold、geometry。VIMA 风格的关键是：**condition prompt 形成独立 memory，action decoder 对它做显式 cross-attention**。

原 ACT decoder 当前每层结构是：

```text
action queries self-attention
→ cross-attention to encoder memory `[latent, proprio, image]`
→ MLP
→ action head
```

VIMA 风格改造后可以变成：

```text
action queries self-attention
→ cross-attention to condition memory `[phase, frame, role, mask, goal, hold]`
→ cross-attention to observation memory `[latent, proprio, image]`
→ MLP
→ action head
```

或者反过来先 attend obs 再 attend condition。对强约束任务，我建议 condition cross-attention 放在 observation cross-attention 前后都可以，但更推荐：先 condition，后 observation，最后再用 condition 做一次门控或 AdaLN。因为 action query 应先知道“我现在要解决哪个控制子问题”，再去观测中寻找相关视觉/状态证据。

### 2.2 和方法一的区别

方法一：condition tokens 与 image tokens 平级，全部进入同一个 encoder memory。decoder 有一个 cross-attention：

```python
memory = encoder([latent, condition, proprio, image])
query = decoder(query, memory)
```

方法二：condition memory 是单独的，decoder 每层显式多一次 cross-attention：

```python
obs_memory = obs_encoder([latent, proprio, image])
cond_memory = cond_encoder(condition_tokens)
query = decoder_with_cond(query, obs_memory, cond_memory)
```

这个差异非常重要。第一种是“把条件告诉模型”；第二种是“每层动作解码都必须经过条件通道”。它更接近大模型中 tool schema / system prefix memory / retrieval memory 的结构，也更适合我们说的“条件约束”。

具体到当前任务，phase 0 的 query 在生成右手 grasp chunk 时，每一层都可以读取：`phase=grasp`、`ref_frame=can`、`right_role=active_grasp`、`left_role=hold`、`mask=right active left hold`、`goal=right pregrasp/grasp pose in can frame`。这样 decoder 不需要从混合 memory 里“猜”当前条件，而是在每层都有一条独立条件通路。

### 2.3 在原 ACT Transformer 上怎么改

当前 `source/miGenRL/detr/models/transformer.py` 中 `TransformerDecoderLayer` 只有一个 `multihead_attn`，用于 `tgt` 到 `memory` 的 cross-attention。要实现 condition cross-attention，可以新增一个 decoder layer：

```python
class ConditionedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, normalize_before):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cond_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.obs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        ...
```

forward 逻辑：

```python
def forward(self, tgt, obs_memory, cond_memory, obs_pos=None, cond_pos=None, query_pos=None):
    q = k = tgt + query_pos
    tgt = self.norm1(tgt + self.dropout(self.self_attn(q, k, tgt)[0]))

    tgt2 = self.cond_attn(
        query=tgt + query_pos,
        key=cond_memory + cond_pos,
        value=cond_memory,
    )[0]
    tgt = self.norm2(tgt + self.dropout(tgt2))

    tgt2 = self.obs_attn(
        query=tgt + query_pos,
        key=obs_memory + obs_pos,
        value=obs_memory,
    )[0]
    tgt = self.norm3(tgt + self.dropout(tgt2))

    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = self.norm4(tgt + self.dropout(tgt2))
    return tgt
```

然后新增 `ConditionedTransformerDecoder`，循环多层 decoder layer。`Transformer.forward()` 也需要拆成两套 memory：

```python
obs_src, obs_pos = build_obs_tokens(src, latent_input, proprio_input, image_pos)
cond_src, cond_pos = condition_tokens.permute(1,0,2), condition_pos.permute(1,0,2)
obs_memory = self.encoder(obs_src, pos=obs_pos)
cond_memory = self.condition_encoder(cond_src, pos=cond_pos)
hs = self.decoder(tgt, obs_memory, cond_memory, obs_pos=obs_pos, cond_pos=cond_pos, query_pos=query_embed)
```

是否需要单独 `condition_encoder`？建议需要，但可以很浅。比如 1-2 层 self-attention，让 phase/frame/role/mask/goal 之间先交互，形成条件上下文。例如 role token 可以结合 action mask，frame token 可以结合 goal pose。对于初版，也可以把 `cond_memory = condition_tokens` 直接给 decoder，不加 condition encoder。

### 2.4 如何处理 ACT 的 CVAE latent

ACT 训练时有一个 CVAE posterior：用 action sequence 和 qpos 编码 latent z，推理时 z 默认置零或采样。加入 condition memory 后，latent 的语义也要条件化。否则同一个 latent 需要同时解释不同 phase/role/frame 下的多峰动作，会增加 KL 和重建难度。

建议在 `DETRVAE.forward()` 训练分支中，将 condition tokens 加入 posterior encoder：

```python
cond_tokens = self.condition_encoder_for_latent(condition)  # [B, N, H]
encoder_input = torch.cat([cls_embed, cond_tokens, qpos_embed, action_embed], dim=1)
```

这样 posterior 学的是 `q(z | qpos, condition, action_chunk)`，decoder 学的是 `p(action_chunk | qpos, image, condition, z)`。这比原来的 `q(z | qpos, action_chunk)` 更适合多条件任务。推理时 z=0 也变成“在当前 condition 下的均值动作模式”，而不是所有 phase 混在一起的均值。

### 2.5 condition memory 应该放哪些 token，如何组织顺序

VIMA-style prompt 的关键是 prompt 结构清晰。建议 condition memory 按固定顺序组织：

```text
[phase_token]
[ref_frame_token]
[target_object_token]
[left_role_token]
[right_role_token]
[action_mask_token]
[left_hold_pose_token]
[right_hold_pose_token]
[left_goal_pose_token]
[right_goal_pose_token]
[left_ee_in_ref_token]
[right_ee_in_ref_token]
[target_in_ref_token]
[pull_axis_or_place_axis_token]
```

其中 pose token 应该用连续 MLP，不建议把 pose 离散化。旋转表示建议使用 rot6d 或项目现有 action 表示；如果使用四元数，必须遵循 Isaac Lab / Isaac Sim 当前规范 `[x, y, z, w]`，不要使用旧版 `[w, x, y, z]`。

condition memory 可以加入 token type embedding：

```python
cond_token = value_embedding + type_embedding + slot_position_embedding
```

type embedding 很重要，因为 `goal_pose_token` 和 `hold_pose_token` 可能都是 6D/9D 连续向量，模型需要知道它们的语义不同。

### 2.6 训练 loss 如何适配

方法二本身仍然是 soft condition。建议 loss 至少增加三类指标：

1. **active action loss**：mask=1 的维度按正常 L1。
2. **inactive hold loss**：mask=0 的维度对 hold action 或 zero delta 做 L1。
3. **condition attention entropy / attention mass 监控**：不是用于训练也可以用于 debug，查看 action queries 是否真的 attend 到 condition memory。

如果实现中可以拿到 decoder cross-attention weights，建议保存：

```text
attn_to_phase
attn_to_ref_frame
attn_to_left_role
attn_to_right_role
attn_to_mask
attn_to_goal
attn_to_hold
```

在 rollout 失败时，如果 phase 1 左手没有 attend 到 drawer_handle/goal/mask token，那说明条件通路还没被模型利用；如果 attend 了但仍失败，则可能是 action 表示或输出约束问题。

### 2.7 和本项目现有 InterACT 的关系

当前代码里已有 `interact_segment_layers`，它会在 action head 后加一个 residual：用 `state_token/left_query/right_query` 经过 `interact_encoder` 得到 left/right token，再用 `interact_residual_head` 修正 action。这是一个“手臂交互 residual”的思路，但它仍然没有显式 condition memory。它知道 left/right query，却不知道 phase、ref frame、role、mask、goal、hold target。因此 InterACT 的失败并不意外：它增强了左右手建模，但没有告诉模型什么时候哪只手应该主动、哪只手应该保持、动作应在哪个 frame 下解释。

方法二可以与 InterACT 结合：

$$
condition memory → decoder cross-attention → hs
hs + interact left/right tokens → residual head
$$

或者更干脆：把 left/right role token 放入 condition memory，让 action query 自己通过 cross-attention 读取，而不是额外 residual。建议先做 VIMA-style condition decoder，再决定是否保留 interact residual。

### 2.8 优点、局限和推荐落地顺序

优点：

- 比简单 condition token 更强，条件不会被大量 image tokens 稀释。
- action decoder 每层都显式读取 condition memory，接近 prompt-driven policy。
- 很适合 debug attention：能看出模型是否使用 phase/frame/role/mask。
- 与结构化 condition 完全兼容，不需要自然语言。

局限：

- 需要改 `TransformerDecoderLayer`，代码侵入比方法一大。
- 仍然是 soft constraint，不能保证 inactive arm 不动。
- 如果数据里的 condition 质量差，模型会学到错误 prompt-action 对应。
- 计算量略增，但 condition token 数量很少，影响不大。

推荐落地顺序：在方法一完成统一 condition 数据结构后，实现 `condition_cross_attention=true` 的 transformer variant。先只用 1-2 层 condition encoder，decoder 每层加 condition attention。跑 train-layout debug 时重点看：phase 0 左手 path/span 是否下降，phase 1 是否更稳定进入 drawer opening，action query 对 role/mask/goal token 的 attention 是否有规律。

---

## 方法三：FiLM / AdaLN 条件调制 —— RT-1 / BC-Z 风格的层级条件注入

### 3.1 方法思想：condition 不只是 token，而是调制网络每一层的计算

独立 token 和 cross-attention 仍有一个共同特点：它们把条件作为“可读取的信息”。但模型是否读取、读取多少，仍然由注意力学习决定。对于一些必须遵守的任务条件，尤其是 active arm / inactive arm / reference frame / action mask，仅靠 token 可能还是不够。更强的做法是让 condition 直接改变网络层的归一化、通道缩放、偏置或 MLP 激活，也就是 FiLM、AdaLN、conditional normalization 一类方法。

BC-Z 和 RT-1 都体现了这种思想。BC-Z 研究机器人模仿学习的零样本任务泛化，系统可以 conditioned on 多种任务信息，包括预训练自然语言 embedding 或人类视频 embedding。PMLR 页面摘要明确指出 BC-Z 的 imitation system 可以由不同形式的信息进行条件化，包括自然语言和人类视频：https://proceedings.mlr.press/v164/jang22a.html 。RT-1 则是大规模真实机器人 Transformer，项目页面介绍其用自然语言指令和图像历史等输入预测离散化动作：https://robotics-transformer1.github.io/ 。在这些方法中，语言条件不仅可以作为 token，也常被用来调制视觉或 policy backbone，使中间特征从一开始就是 task-aware。

迁移到 ACT，我们不需要语言 encoder，而是用 `ConditionEncoder` 产生一个全局条件向量 `cond_global`，然后在 Transformer encoder/decoder 的每一层通过 FiLM 或 AdaLN 改变 hidden state：

$$
cond_global = MLP([phase, frame, roles, mask, goal, hold, geometry])
for each transformer block:
    hidden = SelfAttention(hidden)
    hidden = AdaLN(hidden, cond_global)
    hidden = MLP(hidden)
    hidden = AdaLN(hidden, cond_global)
$$

这和方法一/二的区别是：condition 不再只是一些被 attention 读取的 memory，而是成为每层计算的参数生成器。它更像“系统模式开关”。phase 0、phase 1、phase 2 对应不同的归一化尺度和偏置，模型的特征空间会随条件改变。

### 3.2 FiLM 和 AdaLN 分别是什么，为什么适合 ACT

FiLM 的基本形式是：

```python
y = gamma(cond) * x + beta(cond)
```

其中 `x` 是中间特征，`gamma` 和 `beta` 由条件向量生成。FiLM 最常用于 CNN feature map 或 MLP hidden activation。对机器人策略来说，条件可以是 task embedding；对我们来说，条件可以是 phase/ref_frame/role/mask/goal。

AdaLN 是 Transformer 中更常见的形式，尤其在 diffusion transformer / conditional transformer 里常见。基本形式是：

```python
x_norm = LayerNorm(x)
y = gamma(cond) * x_norm + beta(cond)
```

也可以生成 residual gate：

```python
x = x + gate_attn(cond) * Attention(AdaLN(x, cond))
x = x + gate_mlp(cond) * MLP(AdaLN(x, cond))
```

对 ACT 来说，AdaLN 比普通 FiLM 更自然，因为当前 `TransformerEncoderLayer` 和 `TransformerDecoderLayer` 已经有 LayerNorm。只要把固定的 `nn.LayerNorm` 换成 condition-aware 的 `AdaptiveLayerNorm` 或在 norm 后加 scale/shift，就能让 condition 影响每层。

这特别适合解决两个问题：

1. **phase 导致动作分布变化**：phase 0 grasp、phase 1 pull drawer、phase 2 place can 的动作分布差异很大。AdaLN 可以让同一个 transformer backbone 在不同 phase 下使用不同的 feature scaling。
2. **ref_frame 导致几何解释变化**：当 ref_frame 从 can 切到 drawer_handle，EE pose/goal 的相对几何含义变化。AdaLN 可以让 frame condition 改变模型对 proprio/image feature 的解释方式。

### 3.3 在当前 ACT Transformer 上如何改

当前 `TransformerEncoderLayer` 是标准 post-norm/pre-norm 结构。以 post-norm 为例：

```python
src2 = self.self_attn(q, k, value=src)[0]
src = src + self.dropout1(src2)
src = self.norm1(src)
src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
src = src + self.dropout2(src2)
src = self.norm2(src)
```

可以新增一个模块：

```python
class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(0)
            shift = shift.unsqueeze(0)
        return self.norm(x) * (1 + scale) + shift
```

注意当前 transformer 使用 `[S, B, H]`，cond 是 `[B, Hc]`，所以 scale/shift 要 broadcast 到 `[1, B, H]`。如果使用 batch_first，则相反。

然后把 layer 改成：

```python
src = src + self.dropout1(src2)
src = self.ada_norm1(src, cond)
src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
src = src + self.dropout2(src2)
src = self.ada_norm2(src, cond)
```

Decoder layer 同样改。Decoder 有三处 norm：self-attn 后、cross-attn 后、MLP 后。都可以条件化：

```python
tgt = self.ada_norm1(tgt + self.dropout1(self_attn_out), cond)
tgt = self.ada_norm2(tgt + self.dropout2(cross_attn_out), cond)
tgt = self.ada_norm3(tgt + self.dropout3(ffn_out), cond)
```

如果采用方法二的 condition cross-attention，则 decoder 有四处 norm：self-attn、condition cross-attn、observation cross-attn、MLP。AdaLN 可以和 condition cross-attention 同时使用：cross-attention 负责读取细节 token，AdaLN 负责全局调制模式。

### 3.4 condition_global 如何构造

FiLM/AdaLN 需要一个全局向量，而不是 token sequence。可以直接把所有条件 concat 后 MLP，也可以从 condition tokens pooling 得到。推荐两级结构：

```python
condition_tokens = StructuredConditionEncoder(condition)  # [B, N, H]
cond_global = condition_pooler(condition_tokens)          # [B, H]
```

pooler 可以是：

1. mean pooling：简单但不区分 token 重要性。
2. CLS token pooling：加一个 learnable `cond_cls`，通过 1-2 层 condition transformer 得到 summary。
3. attention pooling：用 learnable query attend condition tokens。

推荐用 CLS pooling：

```python
cond_cls = self.cond_cls.weight.expand(bs, 1, -1)
cond_seq = torch.cat([cond_cls, condition_tokens], dim=1)
cond_seq = self.cond_transformer(cond_seq)
cond_global = cond_seq[:, 0]
```

这样 `cond_global` 同时包含 phase、frame、role、mask、goal、hold、geometry 的交互信息。比如 phase 1 + left active_pull + ref_frame drawer_handle + right hold_object，会形成一个专门的控制模式向量。

### 3.5 是否应该调制视觉 backbone

RT-1/BC-Z 类方法常把任务条件注入视觉 backbone，使视觉特征 task-aware。对我们的项目，是否要调制 CNN backbone 取决于当前图像输入形式：

- 如果 `image_is_feature=false`，ACT 内部用 backbone 提取 RGB/depth feature，可以考虑在 backbone 后的 feature map 上做 FiLM。
- 如果 `image_is_feature=true`，图像特征已经预计算，则只能调制 transformer 中的 tokens。

最小版本不建议改 backbone。先在 transformer encoder/decoder 做 AdaLN。原因是当前问题主要是 policy/action 语义不清，而不是视觉 backbone 没有识别目标。改 backbone 会影响预训练/缓存特征路径，工程复杂度高。后续如果发现视觉注意力无法聚焦 drawer handle/can，再考虑在 image tokens 上做 condition FiLM：

```python
cam_features = self.input_proj(features)
cam_features = film(cam_features, cond_global)
```

### 3.6 和 phase-specific behavior 的关系

AdaLN 的一个重要优点是，它不需要为每个 phase 建完全独立的模型，但可以让同一个模型在不同 phase 下表现得像不同子策略。可以理解为：

$$
shared parameters + condition-generated scale/shift = phase/frame/role-conditioned computation
$$

这比硬拆四个 policy 更数据高效，也比单独 phase token 更强。对于小数据任务，phase-specific head 或 MoE 可能容易过拟合；AdaLN 是中间方案。

### 3.7 训练稳定性和初始化

AdaLN/FiLM 改造要注意初始化。建议所有 condition-generated scale/shift 的最后一层 zero-init，使模型初始时等价于原 ACT：

```python
nn.init.zeros_(linear.weight)
nn.init.zeros_(linear.bias)
```

并使用 `(1 + scale)` 而不是 `scale`，这样初始 scale=0 时不改变特征。这样可以加载原 ACT checkpoint 做 finetune，也可以保证训练初期不因条件调制过强而崩。

学习率建议：

- condition encoder 和 AdaLN 参数可以用主学习率。
- 如果从已有 ACT checkpoint finetune，可对 backbone/transformer 用较小 lr，对 condition modules 用较大 lr。
- 初期不要同时引入太多 loss 和 hard mask，先验证 AdaLN 是否降低 inactive arm 漂移。

### 3.8 在本项目中的配置建议

新增 config：

```yaml
policy:
  condition_mode: tokens
  condition_adaln:
    enabled: true
    apply_to_encoder: true
    apply_to_decoder: true
    apply_to_visual_backbone: false
    cond_pooling: cls
    zero_init: true
```

如果代码上不想新增 transformer variant，可以在现有 `TransformerEncoderLayer` 和 `TransformerDecoderLayer` 中加可选 `cond` 参数；`build_transformer(args)` 根据 config 决定是否构建 conditioned layer。为了不破坏旧 config，所有参数默认 false。

### 3.9 优点、局限和推荐落地顺序

优点：

- condition 影响每层计算，比 token 更强。
- 可以保持参数共享，不必为每个 phase 训练独立模型。
- 适合从原 ACT checkpoint finetune，zero-init 后兼容旧行为。
- 与方法一/二兼容：condition tokens 提供信息，AdaLN 提供模式调制。

局限：

- 仍然不是 hard constraint。模型仍可能输出 inactive arm 大动作。
- 如果 condition 错误，AdaLN 会全局错误调制，可能比 token 错误更严重。
- 需要改 transformer layer forward 签名，工程复杂度中等。

推荐落地：方法一完成后，优先加 decoder AdaLN；如果有效，再加 encoder AdaLN；最后再考虑视觉 feature FiLM。对当前问题，decoder AdaLN 最直接，因为 action chunk 解码阶段最需要知道 active arm/ref frame/action mask。

---

## 方法四：Expert Embedding / MoE / Phase-Specific Head —— MAE-ACT 风格的任务专家化

### 4.1 方法思想：不同 phase/condition 的动作分布差异太大时，让 condition 选择专家表示或输出头

如果 phase 0 是右手抓罐子，phase 1 是左手拉抽屉，phase 2 是右手/双手把物体放入抽屉，那么这些子任务的动作分布、关键视觉区域、坐标系、控制维度、失败模式都不同。把所有 phase 混在一个 action head 中，用一个线性层直接从 `hs` 输出 full action，容易出现“均值化”：右手该动时左手也动，左手该拉时右手也漂，phase 切换处动作不稳定。此时可以引入 expert embedding、phase-specific head 或 Mixture-of-Experts。

MAE-ACT 是直接针对 ACT 多任务问题的相关工作。论文《Mixture of Action Expert Embeddings: Multi-Task ACT》提出把 Mixture of Action Expert Embeddings 集成到 ACT 中，用 task prototypes 和 action expert embeddings 在一个统一策略中获得任务特定表示。论文摘要中提到，原始 ACT 多任务在挑战性 insertion 任务上成功率只有 17%，而 MAE-ACT 达到 67%；它强调通过 shared 和 task-specific representations 改善多任务双手操作：https://aair-lab.github.io/genplan25/papers/43.pdf 。这和我们的问题非常贴近：我们现在不是跨完全不同任务，而是同一任务内部的不同 phase/condition，但从动作分布角度看，它们也像多个子任务。

Expert 方法的核心不是“多训练几个模型”，而是在一个共享 backbone 上引入条件选择的专家参数：

```text
shared visual/proprio/condition encoder
→ condition router / task prototype
→ expert embedding or expert head
→ action chunk
```

它可以有三种强度：

1. **Expert query embedding**：不同 phase/role 给 action query 加不同 expert embedding。
2. **Expert residual head**：共享 action head 输出基础动作，再由 phase/condition-specific expert head 输出 residual。
3. **Phase-specific action head / MoE head**：每个 phase 或 role 有独立 action head，router 按 condition 选择或加权。

### 4.2 Expert query embedding：最小专家化

原 ACT 有 `self.query_embed = nn.Embedding(num_queries, hidden_dim)`，表示 action chunk 中每个未来时间步的 decoder query。它只编码 chunk step index，不编码当前 phase/role/frame。可以改成：

```python
query_embed = base_query_embed + expert_query_embed(condition)
```

其中 `expert_query_embed(condition)` 可以是：

```python
phase_expert = self.phase_query_embed(phase_id)        # [B, H]
frame_expert = self.frame_query_embed(ref_frame_id)    # [B, H]
role_expert = self.role_query_proj(role_pair)          # [B, H]
expert = phase_expert + frame_expert + role_expert
query_embed = self.query_embed.weight.unsqueeze(1) + expert.unsqueeze(0)
```

当前 `Transformer.forward()` 里 `query_embed` 是 `[num_queries, H]`，再扩展到 `[num_queries, B, H]`。要支持 batch-specific expert query，需要允许 `query_embed` 传入 `[num_queries, B, H]`。改动不大：

```python
if query_embed.dim() == 2:
    query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
elif query_embed.dim() == 3:
    pass
```

这样 phase 0 的 action queries 和 phase 1 的 action queries 在进入 decoder 之前就不同。它比 condition token 更直接，因为 action query 本身已经携带“我要生成哪类动作 chunk”的先验。尤其对 ACT 来说，query 是输出动作序列的槽位；给 query 加 expert embedding，相当于改变 action generation 的起点。

### 4.3 Expert residual head：共享预测 + 条件残差

当前 `DETRVAE` 的 action head 是：

```python
self.action_head = nn.Linear(hidden_dim, action_dim)
a_hat = self.action_head(hs)
```

如果 `split_action_heads` 开启，则按 left/right pos/rot/gripper 分头输出。但这些 head 仍然不是 phase-specific。可以增加 expert residual：

```python
base_action = self.action_head(hs)
expert_context = self.expert_encoder(condition)
expert_residual = self.expert_residual_head(torch.cat([hs, expert_context], dim=-1))
a_hat = base_action + expert_residual
```

如果希望更接近 MAE-ACT，可以让 expert residual 由多个 expert embeddings 混合：

```python
expert_embeddings = self.action_expert_embed.weight  # [E, H]
gate = softmax(self.router(cond_global))             # [B, E]
expert = einsum("be,eh->bh", gate, expert_embeddings)
expert = expert.unsqueeze(1).expand(-1, num_queries, -1)
residual = self.expert_head(torch.cat([hs, expert], dim=-1))
a_hat = base_action + residual
```

这里 `E` 可以不等于 phase 数量。比如 4 个 phase 但 6 个 expert：grasp、hold-object、pull、place、stabilize、retreat。router 根据 condition soft-select。对小数据，先用 hard phase expert 更稳定：

```python
expert = self.phase_expert_embed(phase_id)
```

残差 head 的最后一层建议 zero-init，使初始行为等价于原 ACT：

```python
_zero_last_linear(self.expert_residual_head)
```

当前代码已有 `_zero_last_linear()`，可以复用。

### 4.4 Phase-specific / role-specific action head

更强的做法是按 phase 或 arm role 使用不同 action head：

```python
heads = nn.ModuleList([make_head(hidden_dim, action_dim) for _ in range(num_phases)])
phase_action = torch.stack([head(hs) for head in heads], dim=0)  # [P,B,K,A]
a_hat = phase_action[phase_id, batch]
```

为了 batch 中不同样本 phase 不同，可以写成：

```python
all_actions = torch.stack([head(hs) for head in self.phase_heads], dim=1)  # [B,P,K,A]
idx = phase_id.view(B,1,1,1).expand(-1,1,K,A)
a_hat = all_actions.gather(1, idx).squeeze(1)
```

也可以做 soft MoE：

```python
gate = softmax(router(cond_global))  # [B,P]
all_actions = torch.stack([head(hs) for head in heads], dim=1)  # [B,P,K,A]
a_hat = (all_actions * gate[:, :, None, None]).sum(dim=1)
```

对我们的任务，我更推荐 **role-specific split head + mask projection**，而不是只按 phase 分 head。因为 phase 1 中右手可能 hold object，左手 active pull；phase 2 可能右手 active place，左手 hold drawer 或辅助。按 role 拆更符合控制语义：

```text
left_head receives hs + left_role_embedding + frame_embedding
right_head receives hs + right_role_embedding + frame_embedding
```

如果 action layout 是 `[left_pos, left_rot6d, right_pos, right_rot6d, left_gripper, right_gripper]`，可以每只手一个 head：

```python
left_action = self.left_role_heads[left_role](hs)
right_action = self.right_role_heads[right_role](hs)
a_hat = pack(left_action, right_action)
```

当前已有 `split_action_heads`，但它只是按 action group 拆：left pos、left rot、right pos、right rot、left gripper、right gripper。可以在这个基础上增加 role-conditioned head：

```python
self.left_pos_heads = nn.ModuleDict({role: make_head(...) for role in roles})
self.right_pos_heads = nn.ModuleDict({role: make_head(...) for role in roles})
```

不过工程上 ModuleDict + batch gather 麻烦。最小版本可以先做 expert residual head，而不是完全独立 head。

### 4.5 Expert 方法如何处理 inactive arm

Expert 方法可以显著改善 inactive arm，因为它让 phase/role 的动作分布分开。但它仍然不能保证 inactive arm 不动。最好的做法是结合 action mask：

1. Expert head 负责预测 active arm 和合理 residual。
2. Output projection 负责把 inactive arm 维度 clamp/hold。

例如 phase 0：

```python
raw_action = expert_policy(...)
final_action = action_mask * raw_action + (1 - action_mask) * hold_action
```

但即使有 hard projection，expert head 仍然有训练意义：它可以学习 inactive arm 的 hold action、active arm 的 frame-aware motion，不会全部依赖规则。

Expert routing 也可以把 mask 作为输入。一个很好的设计是让 router 不只看 phase，而是看 `role_pair + ref_frame + target`：

```python
router_input = cond_global
expert_gate = softmax(router(router_input))
```

这样 phase 0 如果 target 是 can 和 bottle，可以用不同 expert；phase 1 如果 drawer handle 方向不同，也可以选择不同 pull expert。

### 4.6 和原 ACT CVAE 的关系

ACT 的 latent z 本来用于建模示教动作的多样性。如果加入 expert，latent 和 expert 的职责要分清：

- expert/condition：表示当前任务模式、phase、frame、role，是确定性的高层条件。
- latent z：表示在同一条件下的动作风格、多峰细节、示教变体。

不要让 latent 去承载 phase 信息。也就是说，训练 posterior 必须看到 condition；否则 latent 会被迫编码 phase，推理时 z=0 就会混合不同 phase 的均值。正确形式：

```text
q(z | qpos, condition, action_chunk)
p(action_chunk | image, qpos, condition, expert(condition), z)
```

### 4.7 在 config 和训练中如何做 ablation

建议分四个配置逐步验证：

```yaml
# A: expert query only
policy:
  condition_mode: tokens
  expert_query:
    enabled: true
    source: phase_frame_role

# B: expert residual head
policy:
  expert_residual:
    enabled: true
    num_experts: 4
    router: hard_phase
    zero_init: true

# C: soft MoE residual
policy:
  expert_residual:
    enabled: true
    num_experts: 6
    router: condition_mlp
    entropy_reg: 0.01

# D: phase-specific output heads
policy:
  phase_heads:
    enabled: true
    num_heads: 4
```

每个版本都只做 debug rollout，不立即做正式 rollout。评估指标：

- train-layout success rate。
- phase 0 left span/path。
- phase 1 right object-hold stability。
- phase transition 前后 action jump。
- expert gate 分布是否塌缩到单一 expert。
- 每个 phase 的 active/inactive L1。

### 4.8 优点、局限和推荐落地顺序

优点：

- 非常适合多 phase/多动作分布问题。
- 与 ACT 的 query/action head 结构天然兼容。
- 可以从轻量 expert query 开始，逐步增加 residual/head。
- MAE-ACT 已经说明对多任务 ACT 是合理方向。

局限：

- 数据少时，phase-specific head 容易过拟合。
- hard phase expert 如果 phase 标注有误，会直接选错专家。
- soft MoE 需要额外监控 gate，可能出现 expert collapse。
- 仍需结合 frame-aware action 和 hard mask，不能单独解决坐标系和 inactive arm 执行约束。

推荐落地：先做 **expert query embedding + expert residual head**，不要一开始做完全 phase-specific head。因为 query/residual 可以 zero-init，从现有 ACT checkpoint 平滑 finetune；完全独立 head 参数更多，debug 难度更大。等确认 phase/role expert 能降低 drift，再考虑 role-specific heads。

---

## 方法五：输出层 Hard Constraint / Frame-Aware Action Projection —— 类 constrained decoding 的严格控制层

### 5.1 方法思想：真正严格的条件不能只靠模型“学会听话”，必须限制输出空间

前四种方法都在讲 condition 如何进入 Transformer：token、cross-attention、AdaLN、expert。它们能让模型更清楚地知道任务条件，但都属于 soft conditioning。对于机器人闭环控制，尤其是 inactive arm hold、抓住物体后的相对位姿保持、drawer pull 轴向约束，仅靠 soft conditioning 很可能不够。大模型领域中，如果某些输出必须满足格式或规则，常常不只靠 prompt，而会使用 constrained decoding、grammar、schema、tool call validation。对应到 ACT，严格条件应该落在输出层：**模型输出 raw action，但执行前用 action mask、hold target、reference frame transform、物理/任务约束把 raw action 投影到允许动作空间**。

这不是否定学习，而是把学习和控制分工：

```text
policy 学习：在当前条件下，active 维度应该怎么动，hold 维度的目标是什么，动作 chunk 的细节是什么。
constraint layer 保证：不允许动的维度不会被执行，frame 表示会正确转换，hold/object-relative 关系不会因模型噪声被破坏。
```

这对当前失败最直接。phase 0 右手抓罐子时，左手乱晃不是“模型理解不够”这么简单；即使模型理解了，BC 小误差也会在 rollout 中积累。既然左手在 phase 0 的策略语义就是 hold，那么输出执行时应当直接投影：左手维度来自 hold target，而不是 raw prediction。

### 5.2 Action mask 不是 loss weight，而是执行投影

很多人会把 action mask 只用于 loss：active 维度权重大，inactive 维度权重小或监督 zero delta。这不够。应该同时用于训练和推理：

```python
final_action = action_mask * raw_action + (1 - action_mask) * hold_action
```

这里 `action_mask` 形状可以是 `[B, K, action_dim]`，支持 chunk 内每一步不同 mask。如果 chunk 不跨 phase，也可以 `[B, 1, action_dim]` broadcast 到 K。

`hold_action` 不是简单上一帧 action。它应该由任务条件决定：

1. 对 inactive free arm：hold 当前 EE pose 或当前 joint target。
2. 对 holding object 的手：保持 gripper-object 相对位姿，或保持当前 gripper closed + EE 随物体/目标稳定。
3. 对 drawer pull 中非主轴维度：保持 drawer_handle frame 下的横向/竖向/旋转分量，只允许 pull axis 方向 delta。
4. 对 gripper：在 grasp 后保持 closed，在 release phase 才允许打开。

因此 hold_action 需要和 ref_frame 一起定义，不是简单零向量。

### 5.3 Frame-aware action projection：模型输出 ref frame 下的 delta，执行前转换

当前我们怀疑 phase 限制太弱的核心之一，是模型没有识别坐标系转换。解决这个问题，不能只加 `ref_frame_token`，还应该改变 action label 和执行投影：让模型预测 `delta_in_ref_frame`，再由确定性几何变换转成 robot/world action。

假设 action 中每只手包含 position delta 和 rotation delta。训练时：

$$
gt_action_world[t:t+K]
当前 ref_frame pose: T_world_ref[t]
当前 EE pose: T_world_ee[t]
将未来目标/动作转换成 ref frame 下的相对表示：
  T_ref_ee = inverse(T_world_ref) @ T_world_ee
  delta_ref = transform_delta_world_to_ref(delta_world, T_world_ref)
模型监督 delta_ref
$$

推理时：

$$
raw_delta_ref = policy(obs, condition)
masked_delta_ref = action_mask * raw_delta_ref + (1 - action_mask) * hold_delta_ref
delta_world = transform_delta_ref_to_world(masked_delta_ref, T_world_ref)
final_env_action = pack(delta_world, gripper)
$$

如果当前 action 是绝对 target pose 而不是 delta，也可以用：

$$
模型输出 target_pose_in_ref
执行前 target_pose_world = T_world_ref @ target_pose_in_ref
$$

对 drawer phase，ref_frame 可以是 drawer_handle 或 drawer。拉抽屉时，action mask 可以只开放 ref frame 下的 pull axis：

```python
mask_left_pos_ref = [1, 0, 0]  # 假设 x 是拉出方向
mask_left_rot_ref = [0, 0, 0, 0, 0, 0]
```

这样模型即使 raw prediction 在 y/z/rot 上有噪声，也不会执行。相比 world frame mask，ref frame mask 更有意义，因为 drawer handle 的方向可能随 layout 变化；在 ref frame 下 “沿抽屉拉出方向” 永远是同一个轴。

### 5.4 Chunk 级别的约束：不要只约束第一步

ACT 输出 action chunk `[B, K, A]`。约束层必须作用于整个 chunk，而不是只作用于第一步。否则 temporal aggregation 或 chunk 执行后半段仍可能出现漂移。建议 condition 也生成 chunk 级别 mask 和 hold：

```python
action_mask_chunk: [B, K, A]
hold_action_chunk: [B, K, A]
ref_frame_pose_chunk or current_ref_frame_pose: [B, K, ...]
```

最小版本可以假设 chunk 内 ref_frame 不变，用当前 `T_world_ref[t]` 转换所有 K 步；更准确的版本可以根据环境状态预测未来 ref_frame，但这复杂且不必要。对于 drawer 拉动，ref_frame 会随 drawer joint 移动，严格来说 future handle frame 会变化；但 ACT chunk 短时 horizon 下，用当前 frame 作为局部 delta frame 通常足够。

如果 chunk 跨 phase，mask 会突然变化。建议训练采样时避免跨 phase chunk，rollout 时在 phase 切换后清空 temporal aggregation buffer，重新请求 action chunk。否则旧 phase 的 chunk 后半段可能继续执行，导致 phase 1 刚开始还在执行 phase 0 grasp 的动作。

### 5.5 Loss 设计：active、hold、constraint violation 分开

输出层有 hard projection 后，训练 loss 应该同时监督 raw action 和 projected action，但权重要谨慎。

推荐：

```python
raw_pred = policy(...)
proj_pred = mask * raw_pred + (1 - mask) * hold_action

loss_active = L1(mask * raw_pred, mask * gt_action)
loss_hold = L1((1 - mask) * raw_pred, (1 - mask) * hold_action)
loss_projected = L1(proj_pred, gt_action_projected_or_gt_action)
loss = loss_active + lambda_hold * loss_hold + lambda_projected * loss_projected + kl
```

解释：

- `loss_active` 让模型在允许维度学示教动作。
- `loss_hold` 让模型自己也倾向于输出 hold，而不是完全依赖 projection。这样如果 mask 稍有放宽，模型也不会乱动。
- `loss_projected` 保证实际执行动作接近训练目标。

如果 gt_action 在 inactive 维度本身有示教者微小移动，是否应该学？这取决于任务。对当前问题，我们的目标是消除 inactive arm 乱晃，因此建议 inactive 维度的监督目标用 `hold_action`，而不是原始 gt。否则模型会学习示教中无关的手部微动。

### 5.6 Hard projection 在 rollout 代码中怎么接

当前 `rollout_policy.py` 或 policy interface 通常拿到 `a_hat` 后直接转为 env action。需要插入：

```python
raw_action_chunk = policy(qpos, image, condition)
exec_action_chunk = constraint_projector(raw_action_chunk, condition, env_state)
```

`constraint_projector` 应该是独立模块，而不是散落在 rollout 脚本里。建议位置：

```text
source/miGenRL/constraints/action_projector.py
```

接口：

```python
class ActionConstraintProjector:
    def project(self, raw_action_chunk, condition, obs_or_env_state):
        ...
        return projected_action_chunk
```

它需要知道 action layout。例如当前 action_dim=62 时，可能是 left pos 3、left rot6d 6、right pos 3、right rot6d 6、left gripper 22、right gripper 22。不要在多个文件硬编码切片，应该复用或扩展 `miGenRL.policies.action_layout`。

投影逻辑必须和训练一致。训练 dataset collate 时也应生成同样的 `action_mask` 和 `hold_action`，policy loss 中调用同一套 projector 或同等函数，避免 train/rollout mismatch。

### 5.7 约束不是越硬越好：如何避免过度限制

用户担心“phase 限制太小”，这是对的。Hard constraint 也不能简单写成 phase 0 左手全部锁死、phase 1 右手全部锁死。更合理的是 **condition-specific constraint**：

- phase 0 左手如果确实不参与，应 hold 当前 pose；但如果需要提前避障或维持平衡，可以开放小范围低权重 delta。
- phase 1 右手拿着 can，不应 world 固定；应该保持 can/gripper 相对稳定，允许随身体或物体微调。
- phase 1 左手拉抽屉，不是所有左手维度 active；主要开放 drawer pull axis，其他维度做软限制。
- phase 2 放置时，可能右手 active，左手可 hold drawer open；如果抽屉会回弹，左手 role 是 support_hold_drawer，而不是 inactive。

因此 mask 可以是连续值：

$$
0.0 = hard hold
0.2 = allow small correction
1.0 = fully active
$$

执行投影可以写成：

```python
final = hold + mask * clamp(raw - hold, max_delta_per_dim)
```

这比二值 mask 更柔和，也更符合“条件约束”而非“死板 phase 规则”。

### 5.8 和前四种方法的关系

Hard projection 不应该单独使用。单独 hard mask 会把 policy 变成规则系统，可能掩盖模型没有学会正确动作的问题。最佳组合是：

```text
condition tokens/cross-attention/AdaLN/expert 让模型知道条件
frame-aware action label 降低学习难度
hard projection 保证执行满足关键约束
mask-aware loss 让训练和执行一致
```

如果只能选一个最能解决左手漂移的改动，那是 hard projection；如果要提高整体成功率，还必须配合 frame-aware action 和 condition injection。因为左手不漂只是必要条件，不代表右手能抓准、左手能拉开、右手能放入。

### 5.9 优点、局限和推荐落地顺序

优点：

- 对 inactive arm drift 最直接、最有效。
- 能把 ref frame 变换从神经网络中拿出来，减少学习难度。
- 训练和执行都可解释，失败时能区分 policy 错还是 constraint 错。
- 类似大模型 constrained decoding：重要规则由系统保证，而不是靠 prompt 祈祷。

局限：

- 需要可靠的 condition 和 frame pose。如果 frame 计算错，projection 会系统性错误。
- 约束过硬会限制策略修正能力，可能在接触不确定时失败。
- 需要仔细维护 action layout、mask、hold target、frame transform 的一致性。

推荐落地：与方法一同步做最小 hard projection。第一版可以只做 inactive arm hold 和 gripper hold，不立即做完整 frame-aware delta。第二版加入 ref frame 下的 action label 和 pull-axis mask。第三版再支持连续 mask 和 object-relative hold。

---

## 6. 推荐总体方案：Structured-Condition ACT 的分阶段实现路线

基于上面五种方法，我建议不要一次把所有东西都实现完，而是按“数据结构 → token → hard projection → decoder condition → AdaLN/expert”的顺序推进。因为当前实验已经证明 ACT/InterACT 本身不是关键瓶颈，关键是条件表达和输出约束。推荐路线如下：

### 阶段 A：统一 condition 数据结构

先定义每个 timestep 的结构化条件：

```yaml
phase_id
ref_frame_id
left_role_id
right_role_id
target_object_id
action_mask
hold_action
goal_pose_in_ref
left_ee_pose_in_ref
right_ee_pose_in_ref
target_pose_in_ref
```

从 qpos 中移除 phase scalar，避免重复和语义混淆。dataset、training、rollout 都使用同一套 condition builder。

### 阶段 B：Condition tokens + mask-aware loss

实现方法一：condition tokens 拼入 ACT encoder memory。loss 增加 active/inactive 指标和 hold loss，但 rollout 先可以不 hard mask或只开 debug 开关。

### 阶段 C：Hard projection 初版

实现方法五最小版：

```python
projected_action = mask * raw_action + (1 - mask) * hold_action
```

先解决 phase 0 左手漂移。跑 train-layout debug，只看 6 条即可，不做正式 rollout。

### 阶段 D：Frame-aware action

将 action label 改成 ref frame 下 delta/target pose。phase 0 can frame，phase 1 drawer_handle frame，phase 2 drawer frame。执行前转换回 env action。注意旋转表示保持项目规范；如用 quat，顺序必须是 `[x, y, z, w]`。

### 阶段 E：Condition cross-attention / AdaLN / expert

如果阶段 B-D 后仍然失败，再逐个增强：

1. decoder condition cross-attention；
2. decoder AdaLN；
3. expert query + expert residual；
4. role-specific head。

不要同时打开所有增强，否则无法定位哪个模块有效。

---

## 7. 参考资料

- ACT 原论文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Action Chunking Transformer 用于双手细粒度操作：https://arxiv.org/abs/2304.13705
- NL-ACT：Integrating Natural Language Instructions into the Action Chunking Transformer，将 instruction embedding 集成到 ACT Transformer encoder input：https://github.com/krohling/nl-act
- Octo：Open-Source Generalist Robot Policy，使用 task tokens / observation tokens 的通用机器人策略：https://arxiv.org/abs/2405.12213
- VIMA：General Robot Manipulation with Multimodal Prompts，prompt encoder + prompt-conditioned action decoder：https://vimalabs.github.io/
- RT-1：Robotics Transformer for Real-World Control at Scale，语言指令条件化机器人 Transformer：https://robotics-transformer1.github.io/ ，论文页：https://arxiv.org/abs/2212.06817
- BC-Z：Zero-Shot Task Generalization with Robotic Imitation Learning，机器人 imitation system 可由自然语言或人类视频等任务信息条件化：https://proceedings.mlr.press/v164/jang22a.html
- MAE-ACT：Mixture of Action Expert Embeddings: Multi-Task ACT，将 action expert embeddings 集成到 ACT 以提升多任务双手操作：https://aair-lab.github.io/genplan25/papers/43.pdf
