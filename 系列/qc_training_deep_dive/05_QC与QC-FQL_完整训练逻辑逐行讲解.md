---
title: "QC 与 QC-FQL：完整训练逻辑逐行讲解"
series:
  id: qc_training_deep_dive
  chapter: 5
order: 5
---

# QC 与 QC-FQL：完整训练逻辑逐行讲解

> 第 3 章已经讲清 QC 的基础流程与优化目标，第 4 章又单独说明 FQL 替换了哪个决策环节。本章把两者重新放回同一个 `ACFQLAgent`，逐行查看动作采样、各项 loss、梯度和参数更新。

## 一、QC 和 QC-FQL 是同一个类，靠一个开关切换

打开 `agents/acfql.py`，只有一个类 `ACFQLAgent`。区分 QC 和 QC-FQL 的是配置项 `actor_type`：

```python
actor_type="distill-ddpg",    # 默认值，对应 QC-FQL
actor_num_samples=32,         # actor_type="best-of-n" 时才用得到，对应 QC
```

`actor_type="best-of-n"` 时走 [Q-Chunking 精读 4.2 节](/论文综述/071_QChunking_RL与动作分块#4.2-qc:隐式-kl-约束-+-best-of-n-采样) 讲的 QC 方案：不训练独立的高价值策略网络，靠“采样 $N$ 个候选 + Critic 挑最优”构成隐式策略。`actor_type="distill-ddpg"` 时走 [4.3 节](/论文综述/071_QChunking_RL与动作分块#4.3-qc-fql:显式-wasserstein-约束-+-单步蒸馏策略) 讲的 QC-FQL 方案：额外训练单步蒸馏网络 `actor_onestep_flow`。

这两个值互斥，并且选定后从离线阶段到在线阶段始终不变。下面把两个分支放在一起，是为了对照它们复用的代码，不代表一次训练会同时运行两种方法。

## 二、一次 `update` 的代码外壳：loss 怎样变成参数更新

```python
@jax.jit
def total_loss(self, batch, grad_params, rng=None):
    critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
    actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
    loss = critic_loss + actor_loss
    return loss, info
```

```python
@staticmethod
def _update(agent, batch):
    def loss_fn(grad_params):
        return agent.total_loss(batch, grad_params, rng=rng)
    new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
    agent.target_update(new_network, 'critic')
    return agent.replace(network=new_network, rng=new_rng), info
```

逐行对齐它们的职责：

1. `critic_loss(...)` 构造 TD 目标并计算 Critic 的均方误差。
2. `actor_loss(...)` 计算 BC-flow loss；QC-FQL 还会加入蒸馏 loss 和 Q loss。
3. 两组 loss 先相加成一个标量。
4. `apply_loss_fn` 内部只调用一次 `jax.grad`，但每项 loss 中的参数传递已经决定梯度流向哪个网络。
5. `apply_gradients` 一次性更新 `critic`、`actor_bc_flow`，以及 QC-FQL 使用的 `actor_onestep_flow`。
6. `target_update` 不走梯度，单独做软更新：

$$
\bar\theta\leftarrow\tau\theta+(1-\tau)\bar\theta,
\qquad \tau=0.005.
$$

下面三节分别展开这里调用的 `sample_actions`、`critic_loss` 和 `actor_loss`。

## 三、动作是怎么被生成的：`sample_actions` 完整讲解

不管是训练时给 Critic 提供 bootstrap 动作，还是在线阶段真正和环境交互，最终都要调用 `sample_actions` 拿到一个动作块。这是 Agent 里“策略”概念真正落地的地方。

### 3.1 `distill-ddpg` 分支（QC-FQL）：一次前向搞定

```python
if self.config["actor_type"] == "distill-ddpg":
    noises = jax.random.normal(rng, (*batch_shape, action_dim))
    actions = self.network.select('actor_onestep_flow')(observations, noises)
    actions = jnp.clip(actions, -1, 1)
```

随机采一个和动作块维度相同的高斯噪声向量，直接喂给 `actor_onestep_flow`。这个网络不接收时间参数，因此被当作单步映射：一次前向输出动作块，再裁剪到环境统一使用的 $[-1,1]$ 范围。

这里没有多步积分，也没有候选评分。这正是 QC-FQL 决策较快的直接原因。

### 3.2 `best-of-n` 分支（QC）：采样 + 挑选

```python
elif self.config["actor_type"] == "best-of-n":
    noises = jax.random.normal(rng, (*batch_shape, actor_num_samples, action_dim))
    observations = jnp.repeat(observations[..., None, :], actor_num_samples, axis=-2)
    actions = self.compute_flow_actions(observations, noises)
    actions = jnp.clip(actions, -1, 1)
    q = self.network.select("critic")(observations, actions).mean(axis=0)  # 或 .min(axis=0)
    indices = jnp.argmax(q, axis=-1)
    actions = jnp.reshape(actions, (-1, actor_num_samples, action_dim))[
        jnp.arange(bsize), indices, :
    ]
```

逐步看这段代码：

1. `jax.random.normal` 一次采出 `actor_num_samples`（论文中的 $N$）份独立噪声。
2. `jnp.repeat` 把同一个观测复制 $N$ 份，让每份噪声都和同一状态配对。
3. `compute_flow_actions` 把 $N$ 份噪声分别变成 $N$ 个候选动作块。
4. `critic` 给每个候选打分，`.mean(axis=0)` 或 `.min(axis=0)` 聚合 ensemble 的多个判断。
5. `jnp.argmax` 找到分数最高候选的下标。
6. 最后一行按下标取出这个候选，作为本次真正返回的动作块。

### 3.3 `compute_flow_actions`：Flow Matching 的 Euler 积分

```python
def compute_flow_actions(self, observations, noises):
    if self.config['encoder'] is not None:
        observations = self.network.select('actor_bc_flow_encoder')(observations)
    actions = noises
    for i in range(self.config['flow_steps']):        # 默认 10 步
        t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
        vels = self.network.select('actor_bc_flow')(
            observations, actions, t, is_encoded=True
        )
        actions = actions + vels / self.config['flow_steps']
    actions = jnp.clip(actions, -1, 1)
    return actions
```

这是标准 Euler 数值积分：从随机噪声出发，把时间 $0$ 到 $1$ 分成 `flow_steps` 个小步。每一步都让 `actor_bc_flow` 根据当前状态、当前位置和时间预测速度，再按

$$
\text{actions}\leftarrow\text{actions}+\frac{\text{vels}}{\text{flow_steps}}
$$

向前移动。

默认 10 步走完后，噪声变成一个像数据分布样本的动作块。QC 一次决策要并行处理 $N=32$ 个候选：代码执行 10 次 batch 扩大了 32 倍的 flow 前向，然后再用 Critic 给 32 个候选评分。它不是 320 次串行调用，但总计算量会随 $N$ 近似线性增加。QC-FQL 的学生只需一次普通前向。

## 四、Critic Loss：怎么训练 $Q_\theta$

```python
def critic_loss(self, batch, grad_params, rng):
    if self.config["action_chunking"]:
        batch_actions = jnp.reshape(
            batch["actions"], (batch["actions"].shape[0], -1)
        )
    else:
        batch_actions = batch["actions"][..., 0, :]

    next_actions = self.sample_actions(
        batch['next_observations'][..., -1, :], rng=sample_rng
    )
    next_qs = self.network.select('target_critic')(
        batch['next_observations'][..., -1, :], actions=next_actions
    )
    next_q = (
        next_qs.min(axis=0)
        if self.config['q_agg'] == 'min'
        else next_qs.mean(axis=0)
    )

    target_q = batch['rewards'][..., -1] + \
        (self.config['discount'] ** self.config["horizon_length"]) * \
        batch['masks'][..., -1] * next_q

    q = self.network.select('critic')(
        batch['observations'], actions=batch_actions, params=grad_params
    )
    critic_loss = (
        jnp.square(q - target_q) * batch['valid'][..., -1]
    ).mean()
    return critic_loss, {...}
```

这就是 [Q-Chunking 精读 4.2.4 节](/论文综述/071_QChunking_RL与动作分块#4.2.4-qc-的完整-td-loss) 中 TD loss 的实现。按数据流拆开：

- `batch_actions`：把数据中的真实动作从 `(batch, horizon_length, action_dim)` 拉平为 `(batch, horizon_length * action_dim)`。若关闭动作分块，则只取第一个单步动作。
- `next_actions`：在块结束后的状态上调用第三节的统一采样接口。QC 和 QC-FQL 的外部调用相同，内部生成方式不同。
- `next_qs`：让不直接参与本次梯度更新的 `target_critic` 给下一动作块打分。
- `next_q`：对 ensemble 输出取最小值或平均值，变成一个 bootstrap 价值。
- `target_q`：把数据预处理得到的 $h$ 步累计奖励与折扣后的未来价值相加：

$$
y=G_t^{(h)}+\gamma^h m_t Q_{\bar\theta}(s_{t+h},\mathbf a').
$$

- `q`：当前 Critic 给数据中的真实动作块评分。只有这里显式传入 `params=grad_params`，因此 `critic_loss` 只更新当前 Critic；前面的采样和目标网络只负责提供目标值。
- `valid`：屏蔽跨过轨迹边界的无效序列，避免把两条 episode 错拼成一块。

## 五、Actor Loss：怎么训练 $f_\xi$，以及 QC-FQL 多出的 $\mu_\psi$

```python
def actor_loss(self, batch, grad_params, rng):
    batch_actions = jnp.reshape(batch["actions"], (batch_size, -1))
    x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
    x_1 = batch_actions
    t = jax.random.uniform(t_rng, (batch_size, 1))
    x_t = (1 - t) * x_0 + t * x_1
    vel = x_1 - x_0

    pred = self.network.select('actor_bc_flow')(
        batch['observations'], x_t, t, params=grad_params
    )
    bc_flow_loss = jnp.mean(jnp.square(pred - vel))

    if self.config["actor_type"] == "distill-ddpg":
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(
            batch['observations'], noises=noises
        )
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(
            batch['observations'], actions=actor_actions
        )
        q_loss = -jnp.mean(qs, axis=0).mean()
    else:
        distill_loss = jnp.zeros(())
        q_loss = jnp.zeros(())

    actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
    return actor_loss, {...}
```

这段代码对应 [Q-Chunking 精读 4.3.3 节](/论文综述/071_QChunking_RL与动作分块#4.3.3-qc-fql-的完整训练目标) 的三项损失。

### 5.1 第一项：BC-flow loss 训练 `actor_bc_flow`

```python
x_0 = jax.random.normal(...)      # 随机噪声起点
x_1 = batch_actions               # 数据中的真实动作块
t = jax.random.uniform(...)       # 随机插值时刻
x_t = (1 - t) * x_0 + t * x_1
vel = x_1 - x_0                   # 直线路径的真实速度
pred = self.network.select('actor_bc_flow')(
    observations, x_t, t, params=grad_params
)
bc_flow_loss = jnp.mean(jnp.square(pred - vel))
```

代码先构造一条从噪声 $x_0$ 到真实动作块 $x_1$ 的直线路径，再随机取中间点 $x_t$。直线路径的正确速度恒为 $x_1-x_0$，所以网络可以直接用监督学习预测它。

`params=grad_params` 表明这项 loss 更新 `actor_bc_flow`。训练后，它才能通过第三节的 Euler 积分把噪声搬运成数据分布附近的动作块。

### 5.2 第二项：QC-FQL 的蒸馏 loss

```python
target_flow_actions = self.compute_flow_actions(observations, noises=noises)
actor_actions = self.network.select('actor_onestep_flow')(
    observations, noises, params=grad_params
)
distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
```

教师 `actor_bc_flow` 和学生 `actor_onestep_flow` 接收同一个状态与同一份噪声。教师走完整的 10 步 Euler 积分，学生只做一次前向。MSE 让学生逼近教师终点，这就是 [FQL 前置知识](/前置知识/001p_前置知识_FQL_Flow_Q_Learning) 的教师-学生蒸馏在动作块上的实现。

教师路径没有传入 `grad_params`，因此这个 loss 不会反过来修改教师；学生路径传入了 `grad_params`，所以只更新学生。

### 5.3 第三项：QC-FQL 的 Q loss

```python
qs = self.network.select('critic')(observations, actions=actor_actions)
q_loss = -jnp.mean(qs, axis=0).mean()
```

负号把“最小化 loss”变成“最大化 Critic 分数”。Critic 在这里没有传 `grad_params`，只提供动作优化方向；但 `actor_actions` 是学生网络算出的可导结果，所以梯度会穿过动作回到 `actor_onestep_flow`。

这项负责“追求高价值”，上一项蒸馏 loss 负责“不要离数据分布太远”。

### 5.4 三项怎样合并

```python
actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
```

- QC 分支把 `distill_loss` 和 `q_loss` 都设为零，所以只训练 BC flow。
- QC-FQL 分支三项都存在；`alpha`（默认 100.0）控制学生贴近教师的强度。
- `alpha` 越大，学生越保守；越小，Q loss 的相对影响越强。

## 六、把梯度路径重新对齐到 `total_loss`

读完三段内部代码后，再看最外层的

```python
loss = critic_loss + actor_loss
```

就能明确它并不是让所有 loss 更新所有网络：

| loss 项 | 提供梯度给谁 | 只被读取、不在此项更新的网络 |
|---|---|---|
| `critic_loss` | `critic` | `target_critic`、动作生成网络 |
| `bc_flow_loss` | `actor_bc_flow` | `critic` |
| `distill_loss` | `actor_onestep_flow` | 教师 `actor_bc_flow` |
| `q_loss` | `actor_onestep_flow` | 评分用的 `critic` |

因此，一次 `jax.grad` 可以安全地得到多个网络各自的梯度；随后统一应用梯度，再单独软更新 `target_critic`。这就完成了一个 batch 从进入 `ACFQLAgent.update` 到参数真正变化的完整链路。

## 七、下一章要解决的问题

本章讲完了 QC 和 QC-FQL 共用的 `ACFQLAgent` 内部训练逻辑。下一章转向另一条独立路线：[RLPD-AC 与 QC-RLPD](./06_RLPD-AC与QC-RLPD_另一条技术路线)。它不用 flow matching，而是用 SAC 风格的高斯策略，离线和在线数据的组织方式也不同。
