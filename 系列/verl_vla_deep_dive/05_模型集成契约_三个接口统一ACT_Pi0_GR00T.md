---
title: "模型集成契约：三个接口统一 ACT / Pi0.5 / GR00T"
series:
  id: verl_vla_deep_dive
  chapter: 5
order: 5
---

# 第 05 章 模型集成契约：三个接口统一 ACT / Pi0.5 / GR00T

> 前情提要：第 04 章看到 `VLAFSDPEngine._build_module()` 调用 `build_vla_model` 来构造真正的策略网络。这一章打开模型集成层，看 verl-vla 怎么用三个契约类，把架构完全不同的 ACT、Pi0.5、GR00T N1.6 统一接进同一套训练/推理接口，同时不破坏它们各自的原生 checkpoint 格式。

## 知识链接

- 上一章：[Worker 体系：从 EnvWorker 到 FSDP 训练引擎](./04_Worker体系_从EnvWorker到FSDP训练引擎)
- 下一章：[Flow-SDE 与 DSRL：给流匹配策略装上 SAC](./06_FlowSDE与DSRL_给流匹配策略装上SAC)
- [系列目录](./index)
- [Q 函数与 Value 函数](/前置知识/000o_前置知识_Q函数与Value函数) — 理解 critic 契约的基础
- [SAC (Soft Actor-Critic)](/前置知识/000k_前置知识_SAC_Soft_Actor_Critic) — SAC 契约要实现的核心接口
- [LoRA 低秩适配基础](/前置知识/000x_前置知识_LoRA低秩适配基础) — `apply_lora` 用到的技术

---

## 1. 问题：三个模型，三种截然不同的原生实现

ACT 来自 [LeRobot](https://github.com/huggingface/lerobot)，是一个 CVAE + Transformer decoder 的确定性动作块预测模型；Pi0.5 是基于 PaliGemma 视觉语言模型加一个流匹配（Flow Matching）动作专家的架构；GR00T N1.6 是 NVIDIA 的双流 DiT 架构，同样用流匹配生成动作。三者的模型定义、checkpoint 格式、推理接口完全不兼容。

如果 verl-vla 想让 SAC trainer 能不加改动地训练这三种模型，就需要一层**统一接口**，把"这个模型支持哪些训练能力"和"这个模型具体怎么实现"分离开。

## 2. 第一原则：保留原生策略，不重新实现模型

verl-vla 的模型集成有个明确原则（在 `model-integration.md` 文档里写得很清楚）：**上游原生策略保持不变，作为一个普通 `nn.Module` 被原样加载**。集成层只做"适配"，不做"重写"。

```python
class TrainableVLAModelBase(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy    # 原生策略：lerobot ACTPolicy / GR00T Gr00tN1d6 / PI0Policy
```

好处直接体现在两处：
1. **checkpoint 兼容性**：`export_policy()` 调用原生的 `save_pretrained`，导出的 checkpoint 能直接被 lerobot/gr00t/pi0 官方代码加载，不依赖 verl-vla。
2. **升级成本低**：上游模型代码有更新时，只需要确认适配层接口没变，不需要把新特性重新移植进 verl-vla 自己的模型实现里。

## 3. 三个契约类：能力的显式声明

`models/base.py` 定义了三个基类，模型按需选择继承：

```python
class TrainableVLAModelBase(nn.Module):
    """公共基类，只持有一个原生 policy，提供 LoRA 注入和原生导出能力"""

class SupportSFTTraining:
    """声明"这个模型支持监督微调"，唯一入口是 sft_loss"""
    def sft_loss(self, obs, tokenizer, actions, valids, action_mask, target_values) -> Tensor: ...

class SupportSACTraining:
    """声明"这个模型支持 SAC 训练"，把生成式策略改造成 actor-critic 的全部接口"""
    def sac_sample_actions(self, obs, tokenizer=None, eval=False) -> ModelOutput: ...
    def sac_get_critic_value(self, obs, actions, tokenizer=None) -> Tensor: ...
    def sac_forward_critic(self, a, state_features, *, use_target_network=False, method="cat", requires_grad=False) -> Tensor: ...
    def sac_forward_actor(self, state_features, task_ids=None, ...): ...
    def sac_forward_state_features(self, obs, tokenizer) -> Any: ...
    def sac_update_target_network(self, tau: float): ...
```

一个只做 SFT 的策略，只需要继承 `TrainableVLAModelBase + SupportSFTTraining`，实现一个 `sft_loss` 方法就够了。ACT/Pi0.5/GR00T 三者都同时继承了 `SupportSFTTraining` 和 `SupportSACTraining`，说明它们既能做行为克隆式的监督训练，也能接入 SAC 做强化学习。

**这两个契约类不是抽象基类（ABC）**——代码注释解释了原因：FSDP 会在运行时对模型类做重写（wrap 出一个新的动态类型），ABCMeta 的元类机制会破坏这种运行时类替换。所以改用"约定优于强制"，靠 `register_fsdp_forward_method` 把契约方法注册成 FSDP 感知的 forward 方法：

```python
def sft_init(self):
    register_fsdp_forward_method(self, "sft_loss")
```

这样 FSDP wrap 之后调用 `model.sft_loss(...)` 仍然能正确触发分片权重的 all-gather，而不需要 `sft_loss` 本身继承任何特殊基类。

## 4. `SupportSACTraining` 的关键设计：backbone 只算一次

看 SAC 契约里的方法签名会发现一个设计意图：`sac_forward_state_features(obs, tokenizer)` 是独立于 actor 和 critic 的一步，返回的 `state_features` 会同时喂给 `sac_forward_actor` 和 `sac_forward_critic`：

```python
state_features = model.sac_forward_state_features(obs, tokenizer)   # VLM backbone 前向,只算一次
actions, log_probs, _ = model.sac_forward_actor(state_features, task_ids=task_ids)
q_values = model.sac_forward_critic({"action": actions}, state_features, task_ids=task_ids)
```

**为什么要这样设计**：VLA 模型的视觉-语言编码（VLM backbone）是整个前向里计算量最大的部分。Actor 需要这份编码来决定动作，Critic 需要同一份编码来评估动作好坏——如果分别调用 actor 和 critic 各自跑一遍 backbone，计算量直接翻倍。让两者共享同一份 `state_features`，是"图像/语言只编码一次，动作和价值都基于这份共享表示计算"的标准做法，在三个模型的实现里都能看到。

`sac_forward_critic` 的签名里 `method="cat"|"min"` 和 `use_target_network`/`requires_grad` 三个参数值得单独说明：

- `method="min"`：多个 critic head 的输出取最小值——这是 SAC 标准的双 Q（或多 Q）取小技巧，用于抑制价值高估。
- `method="cat"`：把多个 head 的输出拼接返回（不取 min），用于需要单独访问每个 head 的场景（比如计算 critic loss 时每个 head 各自算 TD error）。
- `use_target_network`：切换到 Polyak 平均得到的 target 网络（用于计算 Bellman target，避免 bootstrap 用同一套正在更新的网络导致不稳定）。
- `requires_grad`：控制梯度是否流回 critic 参数——actor 更新时只需要 critic 输出的数值用于计算 actor loss，不希望这次前向顺带更新 critic 的参数，所以传 `requires_grad=False`（冻结）。

## 5. `builder.py`：显式分派而非 AutoClass 注册

`build_vla_model(model_config, torch_dtype)` 按 `model_config.native_architecture` 字符串做显式 if/elif 分派（和第 04 章 `EnvWorker.init_worker` 的风格一致）：

```python
def build_vla_model(model_config, *, torch_dtype):
    if model_config.native_architecture == "pi0":
        from verl_vla.models.pi0_torch import PI0TrainableModel
        ...
    elif model_config.native_architecture == "act":
        from verl_vla.models.act_torch import ACTTrainableModel
        if overrides:
            raise ValueError("ACT does not support override_config")   # checkpoint 与结构强耦合
        ...
    elif model_config.native_architecture == "gr00t_n1d6":
        ...
```

`native_architecture` 的值在 `VLAModelConfig.__post_init__` 里通过读取 checkpoint 目录下的 `config.json`（检查 `_class_name`/`model_type`/`architectures` 字段）自动推断，识别不出来就直接报错——**拒绝隐式猜测**。

**为什么不用 Transformers 的 `AutoModel` 自动注册机制**：AutoClass 依赖一个全局注册表和懒加载副作用，多个模型库同时 `import` 时容易出现命名冲突、版本不兼容的隐式行为，且新模型接入时不清楚具体走了哪条加载路径。显式 if/elif + 惟独导入（每个分支只在被选中时才 `import` 对应包），换来的是"新增一个模型，去这一个函数加一个分支就行"的可预测性，代价是每加一个模型需要手写一行分派代码——这是 verl-vla 愿意付出的工程权衡。

ACT 分支的 `if overrides: raise ValueError(...)` 值得注意：ACT 的配置和权重强耦合（lerobot 的 config 决定了网络结构，改配置等于换模型），所以不允许通过 `override_config` 在构造时二次修改结构，这是"保护原生 checkpoint 完整性"这条原则的具体体现。

## 6. 三个模型的共同模式对照表

| 维度 | ACT | Pi0.5 | GR00T N1.6 |
|---|---|---|---|
| 基类继承 | `TrainableVLAModelBase + SupportSACTraining + SupportSFTTraining` | 相同 | 相同 |
| 原生随机性 | 无（确定性 CVAE decoder） | 无（确定性 Flow ODE） | 无（确定性 Flow ODE） |
| Critic backend 机制 | `CRITIC_BACKENDS` 注册表 + 抽象基类 | 相同模式 | 相同模式（选项更多） |
| Critic 池化方式 | mean pool | mean pool / cross-attn / 多路 cross-attn | mean pool / cross-attn |
| Target 网络更新 | Polyak（`lerp_`） | Polyak（等价实现） | Polyak（`lerp_`） |
| DSRL 支持 | 无 | 有 | 有 |
| state_features 复用 | actor/critic 共享 encoder 输出 | actor/critic 共享 embed_prefix 输出 | actor/critic 共享 backbone 输出 |

共同点很鲜明：**backbone 编码只算一次**（对应第 4 节），**critic 是外挂的独立参数**（`sac_get_critic_parameters()` 和 `sac_get_named_actor_parameters()` 严格分开返回不同的参数集合，训练时可以用不同的优化器分别更新），**target 网络全部走 Polyak 平均**且和 online 网络结构完全对称（deepcopy 或同构造后 `load_state_dict`）。

## 7. Critic Backend 注册表：三个模型独立定义、模式一致

三个模型各自在自己的目录下定义 `CriticBackend` 抽象基类（`init`/`forward`/`get_critic_parameters`/`update_target_network` 四个方法），用字典按名字选择具体池化实现：

```python
CRITIC_BACKENDS = {
    "mean_pool": MeanPoolCriticBackend,
    "cross_attn": CrossAttentionCriticBackend,
}
```

以 Pi0.5 的 `CrossAttentionCriticGroup` 为例，展示"怎么把一段变长的 prefix token 序列（图像+语言编码）压成一个固定维向量喂给 MLP"：

```python
class CrossAttentionCriticGroup(nn.Module):
    def __init__(self, head_num, attn_heads, input_dim, hidden_dims, prefix_embed_dim):
        self.critic_state_token = nn.Parameter(torch.zeros(1, 1, prefix_embed_dim))   # 可学习的query token
        self.critic_prefix_cross_attn = nn.MultiheadAttention(embed_dim=prefix_embed_dim, num_heads=attn_heads, batch_first=True)
        self.critic_heads = nn.ModuleList([MLP(input_dim, hidden_dims, 1) for _ in range(head_num)])

    def _cross_attention_pool_prefix(self, prefix_embs, prefix_pad_masks):
        query = self.critic_state_token.expand(batch_size, -1, -1)
        pooled, _ = self.critic_prefix_cross_attn(query=query, key=prefix_embs, value=prefix_embs,
                                                     key_padding_mask=~prefix_pad_masks.bool())
        return pooled.squeeze(1)   # (B, D) 固定维
```

**这段代码在做什么**：不用简单的均值池化（会平等对待每个 token，忽略了"哪些 token 对判断价值更重要"），而是引入一个可学习的查询向量，用标准多头注意力去"询问"整段 prefix 序列里哪些位置的信息对价值判断更重要，加权汇总成一个固定维的表示。这本质是把 Transformer 里 `[CLS]` token 的做法搬到了 critic 池化上。拿到 `pooled_prefix_embs` 之后跟 `state`、`action` 拼接送入多头 MLP 集成，`method="min"` 时取最小值，是 SAC 双 Q 技巧向多头的自然推广。

GR00T 的实现（`Gr00tCriticGroup`）在这个基础上多了一处工程细节——用 `nn.Embedding` 而不是裸 `nn.Parameter` 存查询 token：

```python
self.critic_state_token = nn.Embedding(1, d)
```

原因是 HuggingFace 的 `from_pretrained` 加载流程里的 `_fast_init` 优化只会初始化 `nn.Embedding`/`nn.Linear` 类型的参数，裸 `nn.Parameter` 在快速初始化路径下可能不会被正确初始化——这是个容易踩坑但很少被文档提及的细节。

GR00T 的 critic 还支持 **privileged observation**（特权观测）——一种只有 critic 能看到、actor 看不到的额外状态信息（比如仿真器里的精确物体位姿，真实部署时不可得）。这是 asymmetric actor-critic 的标准技巧：critic 在训练阶段可以利用额外信息更准确地评估价值，帮助 actor 学得更快，而 actor 本身的输入始终只有真实可获得的观测，保证训练和部署时行为一致。

## 8. ACT 的 SAC 化：最朴素的加噪探索

ACT 本身是确定性模型，`sac_forward_actor` 手动重放 ACT decoder 的推理路径，直接对输出动作加高斯噪声做探索：

```python
decoder_out = self.policy.model.decoder(decoder_in, encoder_out, ...)
actions = self.policy.model.action_head(decoder_out)
if noise_scale > 0:
    actions = actions + torch.randn_like(actions) * noise_scale
```

注意这里**没有计算 log_prob**（返回 `None`），本质上更接近 TD3 的"确定性策略+噪声探索"而不是严格意义上带最大熵项的 SAC。这是三个模型里 SAC 化最朴素的一种，Pi0.5 和 GR00T 用的 Flow-SDE 方法（下一章详细讲）要复杂得多但理论上更严谨——因为流匹配策略的采样天然是多步 ODE 积分，没法简单套用"加噪声"这么直接的手段。

## 小结

| 概念 | 要点 |
|---|---|
| 保留原生策略 | `policy` 字段持有原生 nn.Module，export_policy 走原生 save_pretrained，checkpoint 完全兼容上游 |
| 三个契约 | TrainableVLAModelBase（公共骨架）+ SupportSFTTraining（sft_loss）+ SupportSACTraining（actor-critic 全套接口） |
| 非 ABC 设计 | 避免 ABCMeta 元类与 FSDP 运行时类重写冲突，靠 register_fsdp_forward_method 保证 FSDP 兼容 |
| backbone 只算一次 | sac_forward_state_features 独立出来，actor 和 critic 共享同一份编码结果 |
| 显式分派 builder | 按 native_architecture 字符串 if/elif，拒绝 AutoClass 隐式注册，新模型接入路径可预测 |
| Critic backend 注册表 | 三模型独立实现同一套 mean_pool/cross_attn 模式，cross-attn 用可学习 query token 做加权池化 |
| ACT 的加噪探索 | 直接对确定性输出加高斯噪声，无 log_prob，接近 TD3 而非严格 SAC |

## 下章预告

[第 06 章](./06_FlowSDE与DSRL_给流匹配策略装上SAC) 讲清楚 Pi0.5 和 GR00T 这类流匹配策略是怎么获得 log_prob、进而套用标准 SAC 最大熵目标的（Flow-SDE 方法），以及另一条完全不同的路线 DSRL——冻结整个 VLA，只在噪声空间训练一个极小的 SAC actor。
