---
title: "推理去噪过程：Euler 积分从噪声到动作"
series:
  id: groot_n1d7_deep_dive
  chapter: 23
order: 23
---

# 推理去噪过程：Euler 积分从噪声到动作

> 推理时模型是怎么从一堆随机数变成精确的机器人动作轨迹的？本章完整走读4步Euler积分的每一个细节。

## 相关阅读

- [多具身体混合训练](./22_多具身体混合训练)（上一章）
- [RTC实时控制](./24_RTC实时控制_动作块重叠)（下一章）
- [Flow Matching 数学基础](./09_Flow_Matching数学基础)

---

## 前情提要

第9-10章我们讲过Flow Matching的训练目标（预测速度场）和噪声调度。
第19章我们看过训练时前向传播的完整流程。本章聚焦**推理**——
当你部署好的模型面对一个新的观测时，具体怎么一步步生成最终动作。

---

## 1. 训练和推理的关键区别

训练时我们**知道**真实动作 $a$，可以构造 $x_t = (1-t)\epsilon + ta$ 作为
模型输入，然后计算loss。推理时我们**不知道**真实动作——这正是我们要模型
预测的东西！所以推理必须换一种方式：从纯噪声开始，让模型迭代地"猜测并修正"，
直到收敛到一个具体的动作值。

---

## 2. 推理流程总览

回顾第9章的Euler积分公式 $x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)$。
推理时我们把这个公式重复应用 `num_inference_timesteps=4` 次，每次都用当前的
$x_t$ 去调用一次完整的模型前向传播（骨干已经算好，只需要重跑DiT），
得到预测速度后更新 $x$。

先建立一个直觉：想象你要从一堆随机涂鸦中"雕刻"出一个具体的动作轨迹——
第一次修正可能改变很大（因为起点太随机），后面几次修正逐渐精细化，
最后定型。4次修正听起来不多，但因为Flow Matching学的是接近直线的路径，
经验证明这个步数足够。

---

## 3. 完整推理代码逐步走读

### 3.1 初始化：从纯噪声开始

推理的第一步是采样一个完全随机的初始"动作猜测"——本质上和训练时的噪声 $\epsilon$
是同分布的：

```python
batch_size = vl_embeds.shape[0]
actions = torch.randn(
    size=(batch_size, self.config.action_horizon, self.action_dim),  # [B, 40, 132]
    dtype=vl_embeds.dtype,
    device=device,
)
dt = 1.0 / self.num_inference_timesteps  # dt = 1/4 = 0.25
```

此刻的 `actions` 张量完全是随机高斯噪声，和真实动作没有任何关系——
接下来4次迭代要把它逐步"雕刻"成有意义的动作轨迹。

### 3.2 循环体：每一步做什么

```python
for t in range(self.num_inference_timesteps):  # t = 0, 1, 2, 3
    t_cont = t / float(self.num_inference_timesteps)      # 0.0, 0.25, 0.5, 0.75
    t_discretized = int(t_cont * self.num_timestep_buckets) # 0, 250, 500, 750
    
    timesteps_tensor = torch.full((batch_size,), fill_value=t_discretized, device=device)
    
    # 编码当前的actions估计 (第15章讲过的ActionEncoder)
    action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
    
    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs
    
    # 拼接state和action (第19章讲过的做法)
    sa_embs = torch.cat((state_features, action_features), dim=1)
    
    # DiT前向传播 (第11-14章的AlternateVLDiT)
    model_output = self.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embeds,
        timestep=timesteps_tensor,
        image_mask=backbone_output.image_mask,
        backbone_attention_mask=backbone_output.backbone_attention_mask,
    )
    
    # 解码得到预测速度 (第17章的ActionDecoder)
    pred = self.action_decoder(model_output, embodiment_id)
    pred_velocity = pred[:, -self.action_horizon:]  # [B, 40, 132]
    
    # Euler积分更新
    actions = actions + dt * pred_velocity
```

注意每一次循环，都完整地重新执行了"编码→DiT→解码"这一整套流程——
唯一变化的输入是 `actions`（越来越接近真实值）和 `timesteps_tensor`
（从0递增到750）。骨干网络的输出`vl_embeds`则是提前算好、在4次循环中
**复用**的（因为图像和语言在整个推理过程中并不会变化）。

---

## 4. 具体数值追踪(简化为1维动作,方便理解)

延续第9章的例子——假设真实答案是 $a=2.0$，理想情况下模型能学到完美的速度场
$v = a - \epsilon$。假设初始噪声采样到 $x_0 = -1.0$：

```
初始: actions = -1.0

t=0 (t_discretized=0):
  模型看到 actions=-1.0, t=0 (刚开始,几乎全是噪声)
  预测速度 v ≈ 3.0  (模型学到的理想速度 = 2.0-(-1.0) = 3.0)
  更新: actions = -1.0 + 0.25*3.0 = -0.25

t=1 (t_discretized=250):
  模型看到 actions=-0.25, t=250 (已经走了1/4)
  预测速度 v ≈ 3.0  (直线路径下速度理论上是常数)
  更新: actions = -0.25 + 0.25*3.0 = 0.5

t=2 (t_discretized=500):
  模型看到 actions=0.5, t=500 (已经走了一半)
  预测速度 v ≈ 3.0
  更新: actions = 0.5 + 0.25*3.0 = 1.25

t=3 (t_discretized=750):
  模型看到 actions=1.25, t=750 (接近完成)
  预测速度 v ≈ 3.0
  更新: actions = 1.25 + 0.25*3.0 = 2.0  ✓ 精确到达目标!
```

四次迭代后，`actions`从随机噪声`-1.0`精确收敛到目标值`2.0`。
注意实际模型的预测不会是完美的常数（存在预测误差），但由于Flow Matching
学的路径接近直线，误差通常很小。

---

## 5. 为什么每一步都要重新算完整的DiT前向传播?

一个自然的问题：既然骨干输出`vl_embeds`在4步中不变，能不能把DiT的某些计算
也缓存起来，减少重复计算？

答案是：**不能**，因为每一步的输入`actions`和`timestep`都在变化,DiT的
所有32层（包括cross-attention和self-attention）的中间激活值都会因此而改变。
唯一可以复用的是骨干网络的输出（因为骨干的输入图像和文本在整个推理过程中
是完全固定的），DiT部分则必须每步都重新完整计算一次。

这也是为什么第1章提到"骨干截断到16层"对推理速度很重要——虽然骨干只需要
跑一次，DiT需要跑4次，但如果骨干本身很慢，仍然会拖累总体的首次推理延迟。

---

## 6. vel_strength:一个为RTC预留的接口

细心观察代码会发现Euler更新那一行实际上是：

```python
actions = actions + dt * pred_velocity * vel_strength
```

多了一个 `vel_strength` 因子。在标准（非RTC）推理模式下，`vel_strength`
被初始化为全1的张量（`torch.ones_like(actions)`），所以这个因子在标准模式下
不起任何作用——等价于我们前面例子里的公式。

这个字段是专门为下一章要讲的RTC（实时控制）模式预留的——RTC会修改
`vel_strength`中特定位置的值（设为0或渐变值），实现"冻结部分动作不再更新"
的效果。本章先记住这个接口的存在，下一章会看到它的完整用法。

---

## 7. get_action与get_action_with_features的关系

代码中有两个相关方法。`get_action` 是标准调用入口，内部先提取特征
再调用 `get_action_with_features` 完成实际的去噪循环：

```python
@torch.no_grad()
def get_action(self, backbone_output, action_input, options=None):
    features = self._encode_features(backbone_output, action_input)  # 编码state
    return self.get_action_with_features(
        backbone_features=features.backbone_features,
        state_features=features.state_features,
        embodiment_id=action_input.embodiment_id,
        backbone_output=backbone_output,
        action_input=action_input,
        options=options,
    )
```

这种拆分的价值在于：如果你在某些场景下想要**复用**已经计算好的
`state_features`（比如连续控制时状态变化不大），可以直接调用
`get_action_with_features` 跳过重复的状态编码步骤——这是一个性能优化的接口设计。

---

## 8. 总结

推理阶段的Euler积分流程：

1. **初始化**：从标准高斯分布采样纯噪声作为初始动作估计
2. **4步循环**：每步用当前动作估计+当前时间步，调用完整DiT前向传播预测速度
3. **Euler更新**：`actions += dt * velocity`，逐步逼近最终动作
4. **骨干只跑一次**：`vl_embeds`在4步循环中被复用，只有DiT+编解码器需要重复计算
5. **`vel_strength`接口**：标准模式下不起作用，是为RTC模式预留的扩展点

---

## 下一章预告

下一章我们深入RTC（Real-Time Control）模式——理解`vel_strength`如何被用来
实现"部分动作冻结、部分动作渐变去噪"，从而让连续控制中的动作块过渡更加平滑。
