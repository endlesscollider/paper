---
title: "多世界并行与 RL 训练：Worlds 机制与 GPU 线程划分"
series:
  id: newton_engine_deep_dive
  chapter: 7
order: 7
tags: [物理仿真, Newton, 强化学习, 并行仿真, GPU并行, Worlds]
category: 系列
---

# 多世界并行与 RL 训练：Worlds 机制与 GPU 线程划分

> 系列第 3 章简单提过 World 概念——这一章把它完整展开：一次性构建几千个并行环境具体怎么操作，GPU 线程网格怎么同时兼顾"环境编号"和"环境内部实体编号"两个维度，以及这套机制天生要面对的"异构环境"限制怎么处理。

**知识链接**：
- [铰接体运动学：广义坐标与最大坐标](./03_铰接体运动学_广义坐标与最大坐标) — 本章第 5 节复习并扩展第 3 章第五节提到的 World 基础概念
- [五件套核心数据模型](./02_五件套核心数据模型_ModelBuilder_Model_State_Control_Contacts) — World 分组信息存在 `Model` 的扁平数组里，是本章讨论的数据基础

---

## 贯穿全文的例子

> **场景**：训练一个机械臂抓取策略,需要用强化学习（PPO 或类似算法）跑几千万步环境交互才能收敛。如果每次只能跑 1 个环境（现实中的机械臂显然只能有一个），几千万步交互需要的真实时间是不可接受的。这一章要讲清楚：怎么让同一个机械臂任务同时在 4096 个"平行宇宙"里跑，每个宇宙互不干扰，但共享同一次 GPU kernel 调用的计算资源。

---

## 一、为什么并行环境数量对强化学习训练速度至关重要

### 1.1 采样效率和"能同时跑多少个环境"直接挂钩

强化学习（尤其是 on-policy 算法如 PPO）的训练效率，很大程度上取决于**单位时间内能收集多少条环境交互轨迹**。如果只有 1 个环境串行跑，即使物理仿真本身很快（比如每步只需要 1 毫秒），几千万步的训练也需要几个小时到几天。而如果能同时跑 4096 个完全独立的环境副本，理论上单位时间收集的数据量能提升几千倍——这正是这几年 GPU 大规模并行仿真（Isaac Gym 最早把这个思路推广开）成为机器人 RL 训练主流做法的核心原因。

### 1.2 关键洞察：GPU 天生擅长"同样的操作，作用在大量数据上"

回顾第 1 章讲的 SIMT 并行模型——GPU 一次可以启动几万个线程，每个线程执行相同的代码逻辑,但处理各自不同的数据。这个模型和"4096 个结构完全相同、只是初始状态和随机种子不同的机械臂环境"这个需求高度契合：每个环境的物理计算逻辑（比如"这个刚体受到多大的重力"）完全一样,只是作用在不同环境的不同刚体上。

---

## 二、World：Newton 对"多个独立环境"的抽象

### 2.1 复习：什么是 World

第 3 章第五节介绍过，World 是 Newton 用来在同一个 `Model` 对象内标记"哪些实体属于同一个逻辑上独立的环境"的机制——每个刚体、关节、形状都带有一个整数 World 索引（`model.body_world`、`model.joint_world`、`model.shape_world` 等数组），碰撞检测和约束求解都会利用这个标记，保证不同 World 之间互不干扰。

### 2.2 三种构建多 World 场景的方式

Newton 提供了从"完全手动"到"一键复制"三个层次的多 World 构建方式：

**方式一：手动作用域**（`begin_world()` / `end_world()`）——最底层、最灵活的方式，适合每个 World 内容都不一样的异构场景：

```python
builder = newton.ModelBuilder()
builder.add_ground_plane()   # 全局实体，World索引-1

# World 0：两个自由飘浮的球
builder.begin_world()
b0 = builder.add_body(mass=1.0)
b1 = builder.add_body(mass=1.0)
builder.add_shape_sphere(b0, radius=0.1)
builder.add_shape_sphere(b1, radius=0.1)
builder.end_world()

# World 1：一个固定基座的转动铰接体
builder.begin_world()
link0 = builder.add_link(mass=1.0)
j0 = builder.add_joint_fixed(parent=-1, child=link0)
link1 = builder.add_link(mass=2.0)
j1 = builder.add_joint_revolute(parent=link0, child=link1)
builder.add_articulation(joints=[j0, j1])
builder.end_world()

model = builder.finalize()
```

**方式二：`add_world()`**——把一个提前搭好的子 `ModelBuilder`（比如本章例子的完整机械臂模板）作为一整个新 World 加进主场景,内部等价于自动帮你调用 `begin_world()` / `add_builder()` / `end_world()`。适合"World 数量不多、但每个都要单独构建一次"的场景。

**方式三：`replicate()`**——最常用于 RL 训练的方式，直接把一个模板 `ModelBuilder` 复制成 $N$ 份，各自作为独立 World：

```python
arm_template = newton.ModelBuilder()
# ...(添加机械臂的连杆、关节、形状,像第2、3章的例子一样)

scene = newton.ModelBuilder()
scene.add_ground_plane()   # 全局共享
scene.replicate(arm_template, world_count=4096, spacing=(2.0, 0.0, 0.0))

model = scene.finalize()
print(model.world_count)    # 4096
print(model.body_count)     # 4096 × (每个机械臂的连杆数)
```

### 2.3 spacing 参数的一个重要提示

`replicate()` 的 `spacing` 参数用来给每个 World 的物理位置做偏移（避免几千个机械臂全部叠在坐标原点、互相视觉重叠不好调试）。但官方文档特别提示：**用物理 spacing 分开多个 World，会把物体移动到离原点很远的地方，可能带来数值稳定性问题**（浮点数精度在离原点较远处会下降）。更推荐的做法是把 `spacing` 设为 `(0, 0, 0)`（所有 World 在物理上叠在同一个坐标原点），只在**可视化层面**用 `viewer.set_world_offsets()` 做视觉上的偏移分离——物理计算始终在数值最稳定的原点附近进行，只是渲染时"假装"它们分散开了。

---

## 三、World 分组信息在 Model 里的具体数据布局

### 3.1 每种实体类型的 world 数组

第 2 章讲过 `Model` 是打包成扁平数组的静态蓝图——World 分组信息也遵循这个模式，每种实体类型都有对应的 `*_world` 数组和 `*_world_start` 起始索引数组：

| 实体类型 | 归属数组 | 起始索引数组 |
|---------|---------|-------------|
| 粒子 | `model.particle_world` | `model.particle_world_start` |
| 刚体 | `model.body_world` | `model.body_world_start` |
| 形状 | `model.shape_world` | `model.shape_world_start` |
| 关节 | `model.joint_world` | `model.joint_world_start` |
| 铰接体 | `model.articulation_world` | `model.articulation_world_start` |

### 3.2 `*_world_start` 数组的特殊格式：为什么长度是 world_count+2

`*_world_start` 数组的长度设计成 `world_count + 2`，而不是直观的 `world_count + 1`（常见的"前缀和"数组通常这么设计,用来快速算出每一段的区间长度）——多出来的这一个位置，是专门为了容纳**全局实体**（World 索引 -1）可能同时出现在数组**前段和后段**这个特殊情况。回顾第 2 节例子里"地面在最前面添加、其他全局实体可能在所有本地 World 之后添加"这种模式——`shape_world` 数组的实际布局可能是：

```
形状索引:        0   1   2   3   4   5
shape_world:    -1   0   0   1   1  -1
shape_world_start: [1, 3, 5, 6]
```

这里 `shape_world_start` 的 4 个数分别代表：World 0 的形状从索引 1 开始、World 1 的形状从索引 3 开始、后段全局实体从索引 5 开始、总形状数是 6。**全局实体因此分成前段（索引 0）和后段（索引 5）两段，不能用简单的一段连续切片同时选出所有全局实体**——如果需要选出所有全局形状,正确做法是直接判断 `model.shape_world.numpy() == -1`，而不是依赖 `shape_world_start` 做切片。这是一个容易被忽略但会直接导致 bug 的细节。

---

## 四、GPU 线程网格怎么同时兼顾"哪个 World"和"World 内哪个实体"

### 4.1 二维线程网格划分

第 1 章讲过 Warp kernel 用 `wp.tid()` 拿到当前线程的编号,负责处理数组里对应的一个元素。对于多 World 场景，一个常见的优化模式是用**二维**线程网格,同时索引"这是哪个 World"和"这是 World 内第几个实体"：

```python
@wp.kernel
def world_body_2d_kernel(
    body_world_start: wp.array(dtype=wp.int32),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    world_id, body_local_id = wp.tid()   # 二维线程索引
    world_start = body_world_start[world_id]
    num_bodies_in_world = body_world_start[world_id + 1] - world_start
    if body_local_id < num_bodies_in_world:
        global_body_id = world_start + body_local_id
        twist = body_qd[global_body_id]
        # ... 对这个刚体的速度做某种计算 ...

# 启动kernel时指定二维网格:(World数量, 每个World最多的刚体数)
wp.launch(
    world_body_2d_kernel,
    dim=(model.world_count, max_bodies_per_world),
    inputs=[model.body_world_start, state.body_qd],
)
```

**代码里值得注意的细节**：`wp.tid()` 在这个 kernel 里返回一个二元组 `(world_id, body_local_id)`，对应启动时指定的二维 `dim` 参数——`world_id` 告诉这个线程"我负责哪个 World"，`body_local_id` 告诉它"我负责这个 World 内的第几个刚体（相对索引）"。真正要访问的全局数组索引，需要用 `world_start + body_local_id` 换算——这正是第 3 节讲的 `*_world_start` 起始索引数组发挥作用的地方。

### 4.2 为什么要 if 判断"是否在范围内"

上面代码里的 `if body_local_id < num_bodies_in_world` 判断,是应对**异构多 World**（不同 World 刚体数量不同）场景的必要防护——二维线程网格的第二维尺寸必须取"所有 World 里刚体数最多的那个"（`max_bodies_per_world`），但刚体数较少的 World，超出自己实际刚体数的那些线程本该"无事可做"，需要用这个条件判断让它们提前退出，避免访问越界的数组索引。

### 4.3 同构场景下的简化：这就是"批处理"

如果所有 World 结构完全一致（本章场景所在的典型 RL 训练情形——4096 个完全相同的机械臂），`max_bodies_per_world` 就是一个固定常数（不随 World 变化），这时候第二维线程网格的意义，本质上退化成了机器学习里熟悉的**批处理**（batching）概念——"World 维度"就是"batch 维度"。这也是为什么用 Newton（或类似的 GPU 并行物理引擎）做 RL 训练时,很自然地能把"环境数量"映射成策略网络训练里的"batch size"，两者在计算范式上是高度一致的。

---

## 五、碰撞检测中的 World 隔离

### 5.1 复习第 4 章的过滤规则

第 4 章第六节讲过 Newton 的三层碰撞过滤机制,World 索引过滤是最外层、最基础的一层——不同 World（都是非负索引）之间的形状**默认永不互相碰撞检测**,这个规则甚至发生在 Broad phase 阶段之前，直接从根源上避免了"World 3 的机械臂和 World 7 的箱子发生碰撞"这种逻辑上不应该出现的情况。全局实体（World 索引 -1，比如地面）例外——它们和所有 World 都参与碰撞检测，这正好符合直觉："地面"应该对每一个平行宇宙里的机械臂都起作用。

### 5.2 这个隔离机制对性能的意义

值得强调的是，这个隔离不只是"逻辑正确性"层面的考量——它同时是一个重要的**性能优化**：如果没有 World 索引这层过滤,4096 个 World、每个 World 10 个形状,总共 40960 个形状,理论上 Broad phase 需要考虑的候选对数量级是形状总数的平方；而利用 World 分组，碰撞检测可以直接把候选范围限制在"同一个 World 内的形状对"，把原本 $O((4096\times10)^2)$ 量级的问题，拆解成 4096 个独立的 $O(10^2)$ 量级的小问题——总计算量从平方级别的"跨 World 组合数"降低到线性级别的"World 数 × 单 World 内的平方复杂度"，这个差异在大规模并行场景下是决定性的。

---

## 六、World 级别的独立配置：不只是"隔离"，还能"各自不同"

### 6.1 每个 World 独立的重力设置

World 机制不仅提供隔离,还允许每个 World 拥有**独立的物理参数**——比如重力。这在需要模拟"同一个机器人,在不同重力环境下的行为差异"这类课程学习（curriculum learning）或域随机化（domain randomization）场景下很有用：

```python
model.set_gravity((0.0, 0.0, -9.81), world=0)   # World 0: 地球重力
model.set_gravity((0.0, 0.0, -1.62), world=1)   # World 1: 月球重力
model.set_gravity((0.0, 0.0, -3.71), world=-1)  # 全局实体的重力
```

### 6.2 域随机化的自然落脚点

这个"每个 World 独立参数"的机制,天然是**域随机化**（domain randomization，训练时给不同并行环境注入不同的物理参数扰动，提高策略对真实世界参数不确定性的鲁棒性）这一常见 RL 训练技巧的底层支撑——不需要额外的框架层，直接利用 World 分组的数据结构,就能让 4096 个并行环境各自有稍微不同的重力、摩擦系数、质量等参数,构成一批自带多样性的训练数据。

---

## 七、按 World 做局部操作：Reset Mask 机制

### 7.1 为什么需要"只 reset 部分 World"

RL 训练里一个常见的操作模式是：当某个环境的 episode 结束（比如机械臂任务成功完成或者失败超时），需要把**这一个**环境重置到初始状态，而其他还在进行中的环境不受影响继续跑——这在向量化环境（vectorized environment）的训练框架里是标准做法，因为不同环境的 episode 长度通常不同步，如果每次都要求所有环境同时 reset，会造成大量算力浪费（等最慢的环境也结束）。

### 7.2 world_mask 参数

Newton 的求解器 `reset()` 方法支持一个可选的 `world_mask` 参数——一个长度为 `world_count + 1` 的布尔数组（前 `world_count` 项对应各本地 World，最后一项对应全局实体），只有被标记为 `True` 的 World 会被重置，其他 World 的状态保持不变。这个设计让"部分环境 reset、部分环境继续跑"这个 RL 训练里的常见需求，能够用一次批量 GPU 调用完成，而不需要在 CPU 侧写循环逐个判断"这个环境需不需要 reset"。

---

## 八、总结

### 一句话

> World 机制让 Newton 能在同一个 `Model`、同一次 GPU kernel 调用里，同时处理成千上万个互相隔离但结构相同（或部分相同）的仿真环境，这是 GPU 大规模并行强化学习训练能够落地的数据结构基础——`replicate()` 提供一键复制的便利入口，World 索引过滤保证不同环境互不干扰，二维线程网格划分让"环境数"自然对应机器学习里的"batch size"概念。

### 核心要点

1. 大规模并行环境是现代 GPU-based RL 训练采样效率的关键，Newton 用 World 机制在单个 `Model` 内实现
2. `begin_world()`/`end_world()`、`add_world()`、`replicate()` 三种方式覆盖从异构到同构场景的不同需求
3. `spacing` 参数应优先设为 0，用 viewer 层面的偏移实现可视化分离，避免物理层面离原点太远带来的数值问题
4. `*_world_start` 数组长度是 `world_count+2`，因为全局实体（-1）可能分布在数组的前段和后段两处
5. 二维 GPU 线程网格（World 维度 + World 内实体维度）是同构场景下"批处理"概念在物理仿真里的直接体现
6. World 索引隔离不仅保证逻辑正确性，还把碰撞检测的候选对数量从平方级降到"World数×单环境平方"级别
7. World 级别可以配置独立物理参数（重力等），天然支撑域随机化训练技巧
8. `world_mask` 支持部分 World 独立 reset，配合向量化 RL 训练框架里"环境异步结束"的常见需求

### 知识链

```mermaid
flowchart LR
    Ch3["第3章 铰接体运动学"] --> This["第7章 多世界并行"]
    Ch4["第4章 碰撞检测流水线"] --> This
    This --> Ch8["第8章 可微仿真"]
```

---

## 延伸阅读

- [Newton 官方文档 Worlds](https://newton-physics.github.io/newton/latest/concepts/worlds.html)
- 上一章：[MuJoCo Warp 深度集成](./06_MuJoCo_Warp深度集成_状态同步与参数映射)
- 下一章：[可微仿真：梯度怎么从仿真结果传回控制输入](./08_可微仿真_梯度怎么传回控制输入)
