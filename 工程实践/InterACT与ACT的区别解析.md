---
title: InterACT 与 ACT 的区别解析
order: 3
tags: [模仿学习, ACT, 双臂]
category: 工程实践
star: 3
---

# InterACT 与普通 ACT 的区别：从直觉到源码的深度解析

本文解释当前仓库中 `InterACT` 和普通 `ACT` 的区别。这里的“当前仓库”指 embodied-arena 里的 MiGenRL 实现，而不是泛泛讨论论文或其他项目中的所有 ACT 变体。结论先放在前面：

普通 ACT 是主模型。它把当前观测编码成一组未来动作 chunk，也就是一次预测接下来若干步该怎么动。当前仓库里的 `InterACT` 不是另起炉灶的全新策略，也不是新的 rollout 后端；它复用同一个 `ACTPolicy` 和同一个 `DETRVAE` 主体，只是在普通 ACT 预测出的动作上额外叠加了一条“interaction residual”分支。这个分支用一个很小的 Transformer Encoder 处理 `state token`、`left token`、`right token` 三个 token，然后输出一个 residual action，最后执行：

$$
InterACT 动作 = 普通 ACT 主干动作 + interaction residual
$$

所以，如果从最直观的角度讲：普通 ACT 更像“看当前画面和机器人状态，直接预测下一段动作”；InterACT 则是在这个预测之外，再给模型一个专门表达“左右手和当前状态怎么协调”的修正通道。它的目的不是替代 ACT，而是补强 ACT 在双手交互、左右手配合、阶段切换和混合数据训练中的表达能力。

## 目录

- [先用一个生活例子理解 ACT](#先用一个生活例子理解-act)
- [普通 ACT 在仓库里到底做什么](#普通-act-在仓库里到底做什么)
- [InterACT 在仓库里到底加了什么](#interact-在仓库里到底加了什么)
- [源码路径：两者从哪里分叉](#源码路径两者从哪里分叉)
- [训练配置差异：算法差异和实验差异要分开看](#训练配置差异算法差异和实验差异要分开看)
- [推理和 rollout：InterACT 仍走 ACT 后端](#推理和-rolloutinteract-仍走-act-后端)
- [checkpoint 兼容：为什么可以从普通 ACT 继续训 InterACT](#checkpoint-兼容为什么可以从普通-act-继续训-interact)
- [从模型能力看：InterACT 解决了什么，没解决什么](#从模型能力看interact-解决了什么没解决什么)
- [初学者最容易误解的地方](#初学者最容易误解的地方)
- [如何判断一个实验应该用 ACT 还是 InterACT](#如何判断一个实验应该用-act-还是-interact)
- [总结](#总结)

## 先用一个生活例子理解 ACT

先不要急着看代码。我们先用一个简单例子理解“动作 chunk”和“交互修正”。

假设你要教一个人打开笔记本电脑。你给他看一张当前画面：桌上有笔记本，左手和右手在某些位置。你还告诉他双手关节状态。然后你让他说出接下来 50 个时间步的动作：左手移动到哪、右手移动到哪、手指怎么收、手腕怎么转。

普通 ACT 做的事情就很像这样。它不是一步一步只预测“下一帧”，而是一次预测一个动作序列。例如 `chunk_size: 50` 时，模型输出的是：

```text
第 1 步动作
第 2 步动作
...
第 50 步动作
```

这对机器人控制很有用。因为机器人任务通常不是独立的一步，而是一小段连续运动。比如抓取杯子时，接近、闭合手指、抬起，本来就是连续相关的。如果模型每一帧都重新决定一次，动作可能抖动；如果一次生成一段，动作会更有计划性。

但是双手任务会带来一个额外问题：左右手不是两个互不相干的机械臂。打开笔记本时，一只手可能压住底座，另一只手抬屏幕；开抽屉时，一只手可能稳定身体或避让，另一只手拉把手；搬运物体时，两只手可能共同抓住同一个物体。普通 ACT 的 Transformer 主干当然也可以从数据中学习这些关系，但它没有一个很明确的“小模块”专门表示“左手、右手、状态之间的交互关系”。

InterACT 的想法可以理解为：在普通 ACT 已经给出一个基础动作后，再加一个专门思考“左右手交互”的小脑。这个小脑不直接从零生成全部动作，而是输出一个修正量。修正量一开始被初始化得接近 0，这意味着刚开始训练时它和普通 ACT 几乎一样；随着训练进行，它逐渐学会在需要左右手协调时修正动作。

这个设计有两个直观好处：

1. 它保留了普通 ACT 已学到的能力。尤其当 InterACT 从普通 ACT checkpoint 恢复训练时，旧模型的主干能力可以继续用。
2. 它把新增能力集中在 residual 分支里。模型不必推翻原有动作预测，只需要学习“哪里需要额外修一下”。

## 普通 ACT 在仓库里到底做什么

当前仓库里的普通 ACT 主要由 `source/miGenRL/policy.py` 中的 `ACTPolicy` 包装，底层模型由 `source/miGenRL/detr/models/detr_vae.py` 中的 `DETRVAE` 实现。它的核心输入输出可以这样理解：

```text
输入：
  qpos   当前机器人/任务状态向量
  image  多相机图像或图像特征
  actions 训练时的专家动作序列
  is_pad 训练时哪些动作位置是 padding

输出：
  a_hat  模型预测的动作 chunk，形状近似为 [batch, chunk_size, action_dim]
```

普通 ACT 的名字里有两个关键点：

1. `A`ction：它学习动作。
2. `C`hunking：它一次预测一段动作，不只是一个动作。

仓库中的普通 ACT 还带有 CVAE 风格的潜变量。训练时模型看到专家动作序列，会用 encoder 把动作序列、当前状态一起编码成一个潜变量分布，再采样 latent；推理时没有专家动作，模型使用零 latent 作为 prior。这样做的目的是让模型在训练时能建模同一个状态下可能存在的多种合理动作模式，同时在推理时给出稳定动作。

从 `DETRVAE.forward()` 的流程看，普通 ACT 大致分为四段。

第一段是训练时编码动作序列。训练时 `actions is not None`，模型把专家动作投影到 hidden dim，把当前 qpos 也投影到 hidden dim，再加上一个 CLS token，然后送进 encoder。encoder 的 CLS 输出被投影成 `mu` 和 `logvar`，再通过 reparameterization 得到 latent。这个 latent 表示“这段专家动作的风格或模式”。

第二段是编码图像和机器人状态。如果配置了相机，模型会用 backbone 提取每个相机的特征，把相机特征拼起来，再把 qpos 投影成 proprioception token。当前仓库中多相机特征会被拼到宽度维度上，并加上 camera embedding。相机名等来自配置，比如 `head`、`left_wrist`、`right_wrist`。

第三段是 Transformer decoder 生成 query 表示。`num_queries` 就是 action chunk 的长度。每个 query 对应未来动作序列中的一个时间位置。假设 `chunk_size: 50`，那模型有 50 个 query，最后会输出 50 个动作。

第四段是 action head。普通情况下，`action_head` 是一个从 hidden dim 到 action dim 的线性层：

```text
hidden state for query t -> action vector at future step t
```

如果打开 `split_action_heads` 且 action dim 满足特定布局，模型也支持拆分动作头，例如分别预测左手位置、左手 rot6d、右手位置、右手 rot6d、左右手夹爪等。但在 InterACT 和普通 ACT 的本质区别上，最关键的不是 split head，而是有没有 `interact_encoder` 和 residual branch。

普通 ACT 的训练损失主要包括：

$$
loss = L1(action, predicted_action) + kl_weight * KL(latent_distribution)
$$

如果配置了 `action_delta_loss_weight`，还会额外加入相邻动作差分的 L1 loss，用来鼓励预测动作的变化趋势更接近专家数据。动作 loss 还可以按 action group 加权，例如左右手位置、旋转、夹爪分别不同权重。

对初学者来说，可以把普通 ACT 理解成一个“看图和状态，输出未来 50 帧控制命令”的模型。它的难点在于：这 50 帧不是简单复制，而是要从视觉、状态、任务阶段里推断出合理的连续动作。

## InterACT 在仓库里到底加了什么

InterACT 的关键新增逻辑在 `DETRVAE.__init__()` 和 `DETRVAE.forward()`。

当配置里设置：

```yaml
policy:
  type: InterACT
  backend: act
  interact_segment_layers: 1
```

训练配置构建时会把 `interact_segment_layers` 传进模型。模型看到这个值大于 0，就创建几个额外模块：

```text
interact_encoder
interact_left_query
interact_right_query
interact_state_proj
interact_residual_head
```

这些模块构成 interaction residual 分支。

这个分支的输入不是原始图片，也不是完整动作序列。它在普通 ACT 主干已经得到 query hidden states `hs` 后工作。具体流程是：

1. 把当前 qpos 通过 `interact_state_proj` 投影成一个 `state_token`。
2. 创建一个可学习的 `left_token`。
3. 创建一个可学习的 `right_token`。
4. 把 `[state_token, left_token, right_token]` 三个 token 放进 `interact_encoder`。
5. 得到更新后的三个 token。
6. 对每个未来 query，把普通 ACT 的 query hidden state `hs` 和 `state_token` 相加，再拼上 `left_token`、`right_token`。
7. 用 `interact_residual_head` 输出一个 action_dim 维的 residual。
8. 把 residual 加到普通 ACT 的动作预测 `a_hat` 上。

用公式写就是：

$$
base_action = ACTHead(hs)

state_token = Linear(qpos)
left_token = LearnableEmbedding("left")
right_token = LearnableEmbedding("right")

state_token, left_token, right_token =
    InteractEncoder([state_token, left_token, right_token])

residual = ResidualHead([hs + state_token, left_token, right_token])

final_action = base_action + residual
$$

这就是当前仓库中 InterACT 和普通 ACT 的核心差别。

注意这里的 `left_token` 和 `right_token` 是可学习参数，不是直接从左手状态切片得到的 token。`state_token` 来自完整 qpos。也就是说，当前实现并没有显式地把 qpos 的左手维度切出来喂给 left token、把右手维度切出来喂给 right token。它是用两个可学习 token 加一个状态 token，让小 Transformer 自己学习“左/右/状态”的交互表示。

还有一个重要细节：`interact_residual_head` 的最后一层被初始化为 0。源码里有 `_zero_last_linear(self.interact_residual_head)`。这意味着模型刚创建出来时，interaction residual 输出接近 0，因此：

$$
final_action ≈ base_action
$$

这对于从普通 ACT checkpoint 继续训练特别重要。因为普通 ACT checkpoint 没有 `model.interact_*` 参数，如果直接换成 InterACT，新增参数会随机初始化。若 residual head 一开始随机输出很大动作，模型行为会被破坏。现在 residual 最后一层置零，新增分支初始影响很小，主干能力可以保留。

这也是为什么 InterACT 当前实现很像“可学习的残差适配器”：它先站在普通 ACT 的肩膀上，再学习额外修正。

## 源码路径：两者从哪里分叉

最容易把关系看清楚的地方是策略工厂：

```python
def make_policy(policy_type: str, policy_config: dict):
    from policy import ACTPolicy, CNNMLPPolicy, MLPChunkPolicy

    policy_type = policy_type.lower()
    if policy_type in {"act", "interact"}:
        return ACTPolicy(policy_config)
```

也就是说，`act` 和 `interact` 都返回 `ACTPolicy`。它们不是两个完全不同的 Python class。真正的区别藏在 `policy_config` 里：InterACT 会多带一个 `interact_segment_layers`。

训练配置构建位于 `source/miGenRL/miGenRL/training/bc_trainer.py`。关键逻辑是：

```python
policy_type = str(policy_cfg.get("type", "ACT"))
policy_backend = str(policy_cfg.get("backend") or policy_type).lower()
is_interact = policy_type.lower() == "interact"
if is_interact:
    if policy_backend != "act":
        raise ValueError(...)
    policy_type = "interact"
```

这说明当前 `InterACT` 只支持 `backend: act`。如果有人配置 `type: InterACT` 但写了别的 backend，会直接报错。

随后配置构建继续走普通 ACT 的 common config，包括：

```text
lr
lr_backbone
backbone
camera_names
state_dim
action_dim
num_queries
hidden_dim
dim_feedforward
enc_layers
dec_layers
nheads
kl_weight
action_loss_weights
action_delta_loss_weight
split_action_heads
action_head_hidden_dim
```

只有当 `is_interact` 为真时，才额外加入：

```text
interact_segment_layers
segment_layout
```

再往下，`ACTPolicy` 调用 `build_ACT_model_and_optimizer()`，后者调用 `build_ACT_model()`，最后进到 `DETRVAE(...)`。`DETRVAE` 接收 `interact_segment_layers`，决定是否创建 interaction 分支。

所以完整分叉路径是：

$$
YAML policy.type
  -> bc_trainer._build_policy_config()
    -> policy_type = "act" or "interact"
      -> make_policy()
        -> 两者都返回 ACTPolicy
          -> build_ACT_model_and_optimizer()
            -> DETRVAE(... interact_segment_layers=0 或 >0 ...)
$$

普通 ACT：

$$
interact_segment_layers = 0
interact_encoder = None
动作 = action_head(hs)
$$

InterACT：

$$
interact_segment_layers > 0
interact_encoder 存在
动作 = action_head(hs) + interact_residual_head(...)
$$

## 训练配置差异：算法差异和实验差异要分开看

当前仓库里有一个代表性 InterACT 配置：

```yaml
config/migenrl/open_laptop_bimanual_interact_activity_depth3_main_plus_sim_spartn_resume.yaml
```

它继承：

```yaml
base:
  - open_laptop_bimanual_activity_depth3.yaml
```

而 `open_laptop_bimanual_activity_depth3.yaml` 又继承双手 activity depth3 训练配置。也就是说，InterACT 的实验配置不是从空白开始的，它先拿普通 ACT 的 open laptop 双手任务配置作为基底。

这个 InterACT 配置主要改了几件事。

第一，策略类型改成 InterACT：

```yaml
policy:
  type: InterACT
  backend: act
  interact_segment_layers: 1
```

这是算法结构差异。它会让模型多出 interaction residual 分支。

第二，加入了 `segment_layout`：

```yaml
segment_layout:
  mode: from_data_keys
  state_tokens:
    left:
      - left
    right:
      - right
  action_tokens:
    left:
      - left
    right:
      - right
```

这看起来像是在声明左右手相关的数据 key 如何对应到左右 token。但当前代码中，`segment_layout` 被传入 `DETRVAE` 并保存到 `self.segment_layout`，实际 forward 里没有用它去切 qpos 或 action，也没有按 layout 生成不同 token。因此，当前实现中真正生效的是 `interact_segment_layers`，而 `segment_layout` 更像预留的语义配置或实验元信息。写文档时必须说清楚这一点，否则读者会以为模型已经按配置显式拆分左右手维度。

第三，数据源发生变化：

```yaml
data:
  names:
    - /home/wahaha/data/Ego_2_raw/open_laptop
    - /home/wahaha/data/Ego_2_raw/open_laptop_sim_spartn_rot6d_geometry_tree/standard_episodes_full
  train_dataset_weights:
    - 0.50
    - 0.50
```

这不是 InterACT 架构本身必需的差异，而是当前实验设置的差异。这里把真实或主数据 `open_laptop` 和 Sim-SPARTN 标准 episode 数据混合训练，并按 0.5/0.5 加权。它可能对实验效果很关键，但不能简单说“InterACT 就等于混合数据训练”。更准确的说法是：当前这个 InterACT open laptop 实验使用了混合数据训练，而普通 open laptop ACT 配置只列了一个数据源。

第四，从普通 ACT checkpoint 恢复：

```yaml
train:
  resume_from: runs/migenrl/checkpoints/open_laptop_bimanual_activity_depth3_20260617_105839/training_state_last.pt
```

这同样是当前实验策略，不是 InterACT 架构必须要求。但它和 InterACT 的 residual 设计非常契合。因为新增的 `model.interact_*` 参数可以缺失，checkpoint 加载兼容逻辑允许这些参数 missing；旧的 ACT 主干参数则从 checkpoint 恢复。于是训练可以从一个已有普通 ACT 能力出发，让新增 interaction 分支慢慢学修正。

把这些差异分成两类比较清楚：

$$
架构差异：
  - policy.type = InterACT
  - backend 必须是 act
  - interact_segment_layers > 0
  - DETRVAE 中启用 interact_encoder 和 interact_residual_head

当前实验差异：
  - 混合 open_laptop 与 sim_spartn 数据
  - train_dataset_weights = [0.50, 0.50]
  - 从普通 ACT training_state_last.pt 恢复
  - rollout 输出目录不同
$$

这两类不要混在一起。否则以后换一个 InterACT 配置，没有混合数据或没有 resume，读者会误以为它不是 InterACT。

## 推理和 rollout：InterACT 仍走 ACT 后端

很多初学者会问：训练时 InterACT 多了模块，那么 rollout 时是不是也有一套新的执行逻辑？当前仓库答案是：没有独立的 InterACT rollout 后端。它仍走 ACT rollout。

rollout 配置里通常有：

```yaml
rollout:
  backend: act
  action_execution: chunk
  chunk_size: 50
  temporal_aggregation: true
```

rollout 代码会加载 policy checkpoint，取出 `policy_config`，构建 policy。这里同样通过 `policy.type` 判断是否要构建 InterACT。构建完成后，执行时只调用：

```python
pred = policy(norm_qpos, policy_images)
```

如果 policy 是普通 ACT，`pred` 就是普通 ACT 输出。如果 policy 是 InterACT，`policy(...)` 内部会走同一个 `ACTPolicy.forward()`，再进入 `DETRVAE.forward()`，在那里自动加上 interaction residual。rollout 外层并不需要知道里面有没有 residual。

动作执行逻辑也相同。模型预测的是一个 chunk，rollout 可以按 chunk 执行，也可以做 temporal aggregation。当前配置中常见的是：

$$
action_execution = chunk
chunk_size = 50
temporal_aggregation = true
$$

temporal aggregation 的直觉是：每一步都可能重新预测一段未来动作，那么同一个未来时间点可能被多个不同 chunk 覆盖。代码会对这些重叠预测做指数加权平均，让动作更平滑。

例如在第 0 步模型预测第 0 到第 49 步动作；第 1 步又预测第 1 到第 50 步动作。第 10 步的动作可能来自多个预测 chunk 的重叠结果。temporal aggregation 会把这些预测加权平均，而不是只信最后一次或第一次。

这和 InterACT 没有直接绑定。普通 ACT 可以用 temporal aggregation，InterACT 也可以用。区别仍然在 policy 内部：InterACT 的每个 predicted action 已经包含 residual 修正。

rollout 里还有一个 action context 变化检测逻辑。对于目标物体 frame 或 active frame 表示，如果任务阶段、目标对象或支撑对象变化，代码会清掉已规划动作和 temporal buffer，必要时执行 hold 动作。这个机制同样不是 InterACT 专属，而是 ACT rollout 的通用机制。它对 chunk 模型很重要，因为旧 chunk 可能是在旧 action context 下生成的，一旦上下文变了，继续执行旧 chunk 就可能错。

所以推理侧的准确描述是：

```text
普通 ACT 和 InterACT 使用同一个 ACT rollout 管线。
InterACT 的差异发生在 policy forward 内部。
rollout 外层仍按 chunk、temporal aggregation、action context reset 等通用 ACT 规则执行。
```

## checkpoint 兼容：为什么可以从普通 ACT 继续训 InterACT

当前仓库有专门的兼容加载逻辑。核心规则是：加载 checkpoint 时允许缺失 `model.interact_*` 开头的参数。

普通 ACT checkpoint 里没有这些参数：

```text
model.interact_encoder.*
model.interact_left_query.*
model.interact_right_query.*
model.interact_state_proj.*
model.interact_residual_head.*
```

如果严格加载，PyTorch 会报 missing keys。但仓库里的兼容函数会过滤这些 missing key，只要缺失的是 `model.interact_` 开头，就不算错误。这让下面的训练路径成为可能：

```text
先训练普通 ACT
  -> 保存普通 ACT checkpoint
  -> 切到 InterACT 配置
  -> 加载普通 ACT checkpoint
  -> 普通 ACT 主干参数恢复
  -> InterACT 新增参数初始化
  -> 继续训练
```

这条路径很合理，因为 InterACT 是在普通 ACT 上加 residual。它不是把模型结构完全换掉，而是在原有模型上加一小组参数。

更细一点看，兼容性不仅依赖 checkpoint 加载，还依赖 residual 初始化。假设允许 missing keys，但 residual head 随机初始化且直接加到动作上，那么刚开始 InterACT 输出可能会偏离普通 ACT 很多，恢复训练会不稳定。当前实现把 residual head 最后一层置零，这样新增分支初始输出接近 0，模型一开始基本保留普通 ACT 行为。这是一个很关键的工程设计。

可以用下面的类比理解：

普通 ACT checkpoint 是一个已经会开笔记本的学生。InterACT 不是让学生换一个大脑，而是给他加了一个新的“左右手协调笔记本”。刚开始这个笔记本是空白的，所以学生仍按原来方式做；训练过程中，笔记本逐渐写入“什么时候左手压住、什么时候右手抬起、什么时候两手都别乱动”等修正经验。

## 从模型能力看：InterACT 解决了什么，没解决什么

InterACT 的优势主要体现在需要显式交互结构的任务中，特别是双手任务或多阶段任务。

### 1. 对双手协调更友好

普通 ACT 的 Transformer 主干当然也能学习左右手关系。但它的 action head 是直接从每个 query hidden state 映射到完整 action vector。左右手协调关系需要在大主干里被隐式学出来。

InterACT 给模型增加了 `left_token` 和 `right_token`。虽然当前实现中的左右 token 是可学习 embedding，不是直接切分左右手状态，但它仍提供了两个稳定的语义槽位，让 residual 分支可以围绕“左”和“右”形成额外参数化空间。对于双手任务，这种归纳偏置可能帮助模型更容易学习左右手之间的配合。

### 2. 对已有 ACT 能力的破坏较小

因为 residual 初始化为 0，InterACT 可以从普通 ACT 平滑过渡。相比直接换一个大模型，这更适合做增量实验。如果普通 ACT 已经有不错的成功率，而失败主要发生在交互、阶段切换或某些数据分布上，InterACT residual 是一个低风险的改进方向。

### 3. 对混合数据训练更自然

当前 InterACT open laptop 配置把主数据和 Sim-SPARTN 数据混合。混合数据常见问题是：不同数据源的状态分布、动作风格、初始化分布可能不完全一致。一个 residual 分支可以承担部分“适配差异”的角色，让主干保持原能力，同时让新增分支学习新数据里的修正模式。

不过必须强调：这是一个合理解释，不等于代码里显式写了“InterACT 专门处理 Sim-SPARTN 数据”。代码只知道它在训练一个带 residual 的 ACT 模型，数据混合由 dataloader 和 config 处理。

### 4. 对阶段切换有潜在帮助，但不是完整解决方案

当前数据集里有 action context 相关逻辑，例如 active frame kind、task phase、目标对象等。训练集会在 action context 变化时截断 action chunk，keyframe 采样也会关注 context change 附近。rollout 时上下文变化会清掉旧计划。InterACT residual 可以从 qpos 和 learned tokens 中学习一些阶段相关修正，但它没有直接把 `action_context` token 输入模型。阶段切换的主要工程保障仍来自数据处理和 rollout 管线。

所以不要把 InterACT 理解成“自动解决所有阶段切换问题”。它只是给模型额外容量和交互偏置；阶段切换是否稳定，还取决于数据标注、action context 截断、采样策略、rollout reset 策略和动作表示。

### 5. 当前实现没有真正使用 segment_layout 做维度切片

这是非常重要的限制。配置里写了：

```yaml
segment_layout:
  mode: from_data_keys
  state_tokens:
    left: [left]
    right: [right]
  action_tokens:
    left: [left]
    right: [right]
```

但在 `DETRVAE.forward()` 中，interaction 分支只做：

$$
state_token = Linear(qpos)
left_token = learned embedding
right_token = learned embedding
$$

没有看到按 `segment_layout` 切分 qpos，也没有按 `segment_layout` 切分 action head。因此当前 InterACT 不是“严格按左右手 segment 分别建模”的版本。它更准确地说是“带 state/left/right learned token residual adapter 的 ACT”。

这不代表设计没用，只代表当前实现的有效机制要讲准确。未来如果要增强，可以让 `segment_layout` 真正参与模型：例如从 qpos 中切出左手状态生成 left token，从右手状态生成 right token；或者让 residual head 分别预测左手 action residual 和右手 action residual，再按 action layout 拼回完整动作。

## 初学者最容易误解的地方

### 误解一：InterACT 是一个完全不同的 policy class

不是。当前 `make_policy()` 里 `act` 和 `interact` 都返回 `ACTPolicy`。区别不是 class 名，而是 `policy_config` 里有没有 `interact_segment_layers`。最终区别发生在 `DETRVAE` 里。

### 误解二：InterACT 有独立的 rollout 后端

没有。当前 `InterACT` 要求 `backend: act`。rollout 仍走 ACT 管线，仍使用 chunk、temporal aggregation、action context reset、action representation 等通用机制。

### 误解三：segment_layout 已经控制了左右手维度切分

当前没有。`segment_layout` 被保存，但 forward 没使用它切分状态或动作。真正生效的是固定的 state token、learned left token、learned right token 和 residual head。

### 误解四：InterACT 一定比 ACT 好

不一定。InterACT 增加了参数和表达能力，但也增加了过拟合风险和调参空间。如果普通 ACT 数据量不足，或者任务本身不是交互瓶颈，InterACT 未必更好。它更适合普通 ACT 已经有基础能力，但在双手配合、交互动作、数据分布变化上还差一点的情况。

### 误解五：从普通 ACT resume 只是为了省时间

省时间是一方面，更重要的是稳定性。InterACT residual 初始化为 0，使得从普通 ACT checkpoint 继续训练变成一种自然的增量学习方式。模型不必重新学视觉、qpos、动作 chunk 的全部基础能力。

### 误解六：InterACT 会自动知道哪个动作维度属于左手或右手

当前不会显式知道。它可以通过训练数据统计和 action loss 学到某些模式，但没有在 interaction 分支中直接使用 action slice 元信息。动作维度的 group 信息主要用于 loss weight 和 group metrics，不是 InterACT token 的直接输入。

## 如何判断一个实验应该用 ACT 还是 InterACT

如果你是刚上手这个项目，可以按下面的顺序判断。

### 优先用普通 ACT 的情况

普通 ACT 更适合作为 baseline。以下情况建议先用 ACT：

1. 任务是单臂或交互结构不复杂。
2. 你还没有确认数据管线、动作表示、图像输入、rollout 表示是否正确。
3. 你需要一个可解释的基线成功率。
4. 数据量较少，担心额外参数过拟合。
5. 当前失败主要来自感知错误、动作维度不匹配、坐标系错误，而不是左右手协调。

普通 ACT 的好处是变量少。你先把普通 ACT 训通，确认 dataset stats、qpos dim、action dim、camera_names、rollout action representation 都没问题，再考虑 InterACT。

### 适合尝试 InterACT 的情况

InterACT 更适合这些情况：

1. 普通 ACT 已经能完成部分动作，但双手配合不稳定。
2. 失败常发生在“左手该稳住但没稳住”“右手该配合但提前动了”“两只手动作互相干扰”这类问题上。
3. 任务有明显的交互对象，例如笔记本、抽屉、门、容器等。
4. 你希望从已有 ACT checkpoint 上做增量改进。
5. 你在混合主数据和 rollout/synthetic 数据，想给模型一点额外适配能力。
6. 你愿意额外检查 InterACT residual 是否真的在学习有意义的修正。

实践上，一个稳妥路线是：

```text
1. 先训练普通 ACT baseline
2. rollout 看失败模式
3. 如果失败集中在双手交互或阶段切换附近，再启用 InterACT
4. 用普通 ACT checkpoint resume
5. 对比相同 reset 分布下的成功率和失败类型
```

不要只看 training loss。InterACT residual 可能降低 validation loss，但 rollout 未必更稳定。机器人策略最终要看闭环执行。

## 如何调试 InterACT

调试 InterACT 时，建议分层检查。

第一层，确认配置真的启用了 InterACT。看训练输出目录里的 `policy_config.yaml`，里面应该有：

```yaml
policy_type: interact
policy_config:
  interact_segment_layers: 1
```

如果 `policy_type` 还是 `ACT` 或 `act`，那就没有启用 InterACT。

第二层，确认 checkpoint 加载没有误报兼容问题。从普通 ACT resume InterACT 时，缺失 `model.interact_*` 是正常的；如果缺失其他主干参数，或出现 unexpected keys，就要小心配置是否不匹配。

第三层，确认 action dim 和 state dim 没变。InterACT residual head 输出的是完整 `action_dim`，所以它要求数据、stats、checkpoint、rollout 中 action dim 一致。如果混合数据源，一个数据源 action key 或 hand synergy 配置不同，会导致训练前就出错，或者更隐蔽地导致动作语义错位。

第四层，对比 base action 和 residual。当前代码没有直接输出 residual，但可以临时在调试分支里记录 `a_hat` 加 residual 前后的差值，或者注册 hook 观察 `interact_residual_head` 输出。一个健康的 residual 不应该一开始就巨大；训练后它应该在关键交互阶段更明显，而不是全程随机漂移。

第五层，看 rollout 失败视频和 action trace。如果打开 `trace_actions`，可以观察 replan、planned_action_index、动作位置等。InterACT 的问题如果只在推理阶段出现，可能不是模型结构本身，而是 temporal aggregation、action context change、hold action 或 action representation 的问题。

第六层，固定 reset 分布做公平比较。当前 iterative 配置里有 `reset_distribution` 和 `distribution`，例如 xy grid、least_filled 等。如果普通 ACT 和 InterACT 用不同 reset 分布，成功率不可直接比较。

## 更深一层：为什么 residual 分支比直接换大模型更保守

从优化角度看，InterACT residual 是一种保守扩展。普通 ACT 学到的函数可以记为：

```text
f_base(obs) -> action_chunk
```

InterACT 学到的是：

$$
f_interact(obs) = f_base(obs) + g_interact(obs)
$$

其中 `g_interact` 初始接近 0。这样模型空间包含普通 ACT：只要 `g_interact = 0`，InterACT 就退化成普通 ACT。因此从函数空间看，InterACT 至少有能力表示普通 ACT 的行为。

当然，训练时参数会更新，主干也可能被继续改变，所以实际不保证永远不退化。但这种初始化让优化起点接近原模型，比随机初始化一个全新模型稳得多。

为什么不直接把 Transformer hidden dim 加大，或者多加 decoder 层？因为那会改变主干表示，checkpoint 兼容性更差，旧能力更容易被破坏。Residual adapter 的优点是“增量、局部、可兼容”。缺点是表达能力受限，尤其当前 token 设计还比较简单。

可以把它和 LoRA、adapter tuning 的思路类比：不一定改动整个大模型，而是在已有模型旁边加一条轻量可训练路径，用来适配新任务或新分布。InterACT 不是 LoRA，但工程动机有相似之处。

## 更深一层：InterACT 的 token 设计意味着什么

当前 InterACT token 有三个：

```text
state_token: 来自 qpos 的线性投影
left_token: 可学习 embedding
right_token: 可学习 embedding
```

这三个 token 经过一个小 Transformer Encoder。Transformer Encoder 的自注意力会让三者互相读信息。由于 left/right token 本身不来自左/右手状态，它们的意义完全来自训练：模型可以学习把某些 residual 方向和 left token 关联，把另一些 residual 方向和 right token 关联。

这是一种弱语义 token。它比没有 left/right token 强，因为至少给了模型两个稳定槽位；但它比真正从左右手状态构造 token 弱，因为语义不是硬绑定的。

如果以后要增强，这里有几条自然路线：

1. 从 qpos_keys 中按 left/right 切片，分别投影成 left_state_token 和 right_state_token。
2. 从 action_keys 或 action layout 中按 left/right 建立 residual head，只让 left token 影响 left action slice，right token 影响 right action slice。
3. 把 active frame、phase index、target object embedding 作为额外 token 输入 interaction encoder。
4. 让 `segment_layout` 真正驱动 token 构建，而不只是被保存。
5. 对 residual 加正则，避免它全局覆盖 base action。

这些都不是当前代码已经完成的事情，但它们解释了 `segment_layout` 这个配置字段可能想往哪里发展。

## 与 MLPChunk 的区别顺带说明

当前配置里还有 `open_laptop_bimanual_activity_depth3_mlp_chunk.yaml` 以及 debug 对比配置。`MLPChunk` 和 ACT/InterACT 是另一条差异线。

`MLPChunk` 不使用 DETRVAE 的 Transformer chunk decoder，而是把 qpos 和图像特征展平后送进 MLP，直接输出 `num_queries * action_dim`，再 reshape 成 chunk。它更简单，变量少，但表达复杂时序、多相机空间关系和多模态动作的能力通常不如 ACT。

InterACT 不是 MLPChunk。InterACT 仍是 ACT 主干，只是在 ACT 上加 interaction residual。比较关系可以这样画：

```text
MLPChunk:
  qpos + image -> MLP -> action chunk

ACT:
  qpos + image -> DETR/CVAE/Transformer -> action chunk

InterACT:
  qpos + image -> DETR/CVAE/Transformer -> base action chunk
                         +
                  interaction residual branch -> residual action chunk
```

所以如果你在做实验表格，最好把三者分开命名：

```text
MLPChunk baseline
ACT baseline
InterACT residual ACT
```

## 用一张表总结 ACT 和 InterACT

| 维度 | 普通 ACT | InterACT |
| --- | --- | --- |
| policy class | `ACTPolicy` | 也是 `ACTPolicy` |
| rollout backend | `act` | `act` |
| 主体模型 | `DETRVAE` | `DETRVAE` |
| action chunk | 支持 | 支持 |
| CVAE latent | 支持 | 支持 |
| 图像 backbone | 相同配置下相同 | 相同配置下相同 |
| 主要新增模块 | 无 | `interact_encoder`、left/right query、state projection、residual head |
| 输出形式 | `a_hat = action_head(hs)` | `a_hat = action_head(hs) + residual` |
| checkpoint 兼容 | 普通加载 | 允许缺失 `model.interact_*`，可从 ACT resume |
| 当前 `segment_layout` | 不使用 | 保存但 forward 未实际切分使用 |
| 适合场景 | baseline、单臂、普通 chunk imitation | 双手交互、增量适配、混合数据修正 |
| 风险 | 表达交互可能不足 | 更多参数，可能过拟合，语义 token 当前较弱 |

## 一个更准确的命名

如果只看当前实现，“InterACT”这个名字可能让人以为它已经完整实现了按 action/state segment 的交互建模。严格说，当前版本更接近：

```text
ACT + state/left/right token interaction residual adapter
```

这个命名更长，但更准确。它说明三件事：

1. 它还是 ACT。
2. 它多了 state/left/right token。
3. 它输出 residual，而不是替代主动作头。

实际写实验报告时，可以这样描述：

```text
We use an InterACT variant implemented as an ACT backbone with an additional
state-left-right interaction encoder and a zero-initialized residual action head.
The rollout backend and chunk execution remain identical to ACT.
```

中文可以写：

```text
我们使用的 InterACT 是在 ACT 主干上增加 state/left/right 交互 token 和零初始化 residual action head 的变体；rollout 后端与 chunk 执行逻辑仍与普通 ACT 相同。
```

这比简单说“用了 InterACT 替代 ACT”更准确。

## 总结

普通 ACT 和当前 InterACT 的关系不是“两个完全不同模型”的关系，而是“主干和增强版”的关系。普通 ACT 负责从视觉和状态生成未来动作 chunk；InterACT 在同一个 ACT 主干上增加一个小型 interaction residual 分支，用 state token、left token、right token 学习额外修正。

最核心的源码事实有三条：

1. `act` 和 `interact` 都通过 `make_policy()` 构建为 `ACTPolicy`。
2. `InterACT` 要求 `backend: act`，训练配置只额外传入 `interact_segment_layers` 和可选 `segment_layout`。
3. `DETRVAE.forward()` 中普通 ACT 先得到 `a_hat`，InterACT 再计算 residual 并执行 `a_hat = a_hat + residual`。

最核心的工程意义也有三条：

1. InterACT 可以从普通 ACT checkpoint 平滑恢复，因为缺失的 `model.interact_*` 参数被允许，residual head 初始输出接近 0。
2. InterACT 的 rollout 管线和普通 ACT 相同，差异在模型 forward 内部。
3. 当前 `segment_layout` 还没有真正驱动左右手维度切分，所以不要把它理解成已经完成的强 segment-aware architecture。

给初学者的最终建议是：先把普通 ACT 当作必须建立的 baseline；当普通 ACT 的主要失败集中在双手交互、左右手协调、对象交互阶段或混合数据适配时，再尝试 InterACT。尝试时最好从普通 ACT checkpoint resume，并在相同 reset 分布下比较 rollout 成功率和失败类型。这样才能判断 InterACT residual 分支到底是在解决问题，还是只是增加了模型复杂度。

