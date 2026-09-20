---
title: "策略一二：域随机化与 Co-training 的工程实现"
series:
  id: nvidia_real2sim2real_deep_dive
  chapter: 6
order: 6
---

# 第六章：策略一二——域随机化与 Co-training 的工程实现

> **前情提要**：第五章讲清楚了官方对 Sim-to-Real Gap 的四类分类。本章开始逐一拆解官方教程给出的四种策略,先讲最基础的两种：Strategy 1（Domain Randomization）和 Strategy 2（Co-training）。本章代码全部来自官方教程 [Sim-to-Real Strategy 1](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html) 和 [Strategy 2](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html)。

**知识链接**：
- [第五章：Sim-to-Real Gap 全景](./05_Sim2Real_Gap全景_四类差距与四种策略)
- [Sim-to-Real 迁移综述](/论文综述/S04_Sim_to_Real迁移综述#2.-域随机化-domain-randomization) — 域随机化的通用方法论背景

---

## 一、Strategy 1：Domain Randomization

### 1.1 核心思路（复习一句话）

域随机化的思路是：**不追求让仿真完美匹配真实，而是在训练时随机化仿真参数，让策略对这个范围内的任何取值（包括真实世界的取值）都保持鲁棒**。官方教程给的类比很直观——学接球时,如果每次都用同一个姿势接同一个位置的球，遇到不同来球位置就会手忙脚乱；如果练习时故意让球从各个不同位置飞来，才能真正学会一个通用的"接球策略"。

### 1.2 官方案例的具体随机化对象

官方教程的案例任务是：SO-101 机械臂把离心管（vial）放进架子（rack）。在 Isaac Lab 里，通过 `EventTerm`（重置事件处理器）机制注入三类随机化：

**光照随机化**（`randomize_sky_light`）：

```python
def randomize_sky_light(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    temperature_range: tuple[float, float],
    textures_root: str,
    asset_cfg: SceneEntityCfg = None,
):
    # 采样随机曝光度和色温
    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()
    temperature = math_utils.sample_uniform(*temperature_range, (1,), device="cpu").item()

    # 从可选HDRI贴图库中随机选一张
    textures = glob.glob(os.path.join(textures_root, "*.exr"))
    texture = textures[torch.randint(0, len(textures), (1,)).item()]

    # 应用到场景的穹顶光源
    prim.GetAttribute("inputs:exposure").Set(exposure)
    prim.GetAttribute("inputs:colorTemperature").Set(temperature)
    prim.GetAttribute("inputs:texture:file").Set(Sdf.AssetPath(texture))
```

**相机位姿随机化**（`randomize_camera_pose`）：给外部相机加上小幅的位置和旋转偏移,模拟真实相机安装位置的微小误差。

**物体位姿随机化**（`reset_vials_rack`）：随机化离心管和架子的位置/朝向,并以一定概率（示例中 `rack_placement_prob=0.33`）预先把某个离心管放进架子的某个槽位,增加场景初始状态的多样性。

### 1.3 怎么把这些随机化"接"到训练循环里

```python
@configclass
class TaskEventCfg(EventCfg):

    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range": (-4.0, 3.0),
            "temperature_range": (2500.0, 9500.0),
            "textures_root": f"{assets_path}/hdri",
            "asset_cfg": SceneEntityCfg("sky_light"),
        },
    )

    reset_camera_external_pose = EventTerm(
        func=randomize_camera_pose,
        mode="reset",
        params={
            "prim_path_pattern": "{ENV_REGEX_NS}/LightStudio/LightBox/camera_mount",
            "pos_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
            "rot_range": {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)},
        },
    )
```

每次环境 reset（回合重置）时，Isaac Lab 会调用所有注册了 `mode="reset"` 的 `EventTerm`,执行一次新的随机化。想要调整随机化的强度，只需要改 `exposure_range`、`pos_range` 这类参数字典里的范围；想要临时关闭某一类随机化,直接注释掉对应的 `EventTerm` 即可。

### 1.4 仿真遥操作 vs 真实遥操作的对比

官方文档给出了一张对比表,总结了两种数据采集方式的权衡：

| 维度 | 仿真遥操作 | 真实遥操作 |
|------|-----------|-----------|
| 域随机化 | 自动 | 手动，且受限于物理环境实际能改变的范围 |
| 数据采集速度 | 重置更快，可并行多环境 | 仅能实时进行 |
| 硬件磨损 | 无 | 会累积 |
| 视觉多样性 | 程序化生成 | 需要人工手动变化场景 |
| 物理精度 | 近似 | 真实（ground truth） |

**什么时候用仿真、什么时候用真实**（官方给出的判断依据）：
- **用仿真**：搭建初版数据集（配合域随机化）、硬件资源有限或被占用、想快速安全地探索任务/策略变体、真实环境还没准备好
- **用真实**：采集高质量的 ground truth 数据、验证仿真训练出的策略、捕捉真实世界特有的细节（摩擦、光照的真实表现）

---

## 二、Strategy 2：Co-training

### 2.1 核心思路：优势互补

Co-training（协同训练）的做法是：**把少量真实遥操作数据和大量仿真数据混合在一起训练同一个策略**。官方教程给出的数据规模对比很直观——用**5 条**真实演示,配合**70-100 条**仿真演示。

这个思路解决的问题很直接——单看两个数据源各自的短板：

| 数据源 | 优点 | 短板 |
|--------|------|------|
| 仿真数据 | 数量充足，一致性好 | 和真实世界的分布不完全匹配（近似） |
| 真实遥操作数据 | 完全匹配真实分布 | 数量受限（采集成本高） |

**协同训练同时利用两者**——用仿真数据的数量优势打好基础，用少量真实数据把整体分布"拉"回真实世界。

### 2.2 采集真实数据的命令（供参考）

```bash
export HF_USER=your-hf-username

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=$ROBOT_PORT \
  --robot.id=$ROBOT_ID \
  --robot.cameras='{
    "wrist": {"type": "opencv", "index_or_path": '"$CAMERA_GRIPPER"', "width": 640, "height": 480, "fps": 30},
    "front": {"type": "opencv", "index_or_path": '"$CAMERA_EXTERNAL"', "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=$TELEOP_PORT \
  --teleop.id=$TELEOP_ID \
  --dataset.repo_id=${HF_USER}/so101-teleop-vials-to-rack-real \
  --dataset.num_episodes=5 \
  --dataset.single_task="Pick up the vial and place it in the yellow rack"
```

采集完成后,把这份真实数据集和仿真数据集合并，一起训练 GR00T 策略。

### 2.3 部署与验证流程

官方教程给出的部署方式是标准的"服务器+客户端"模式——在一个终端启动 GR00T 策略服务器,在另一个终端跑真实机器人的评测客户端：

```bash
# 终端1：启动策略服务器
export MODEL=aravindhs-NV/grootn16-finetune_sreetz-so101_teleop_vials_rack_left_sim_and_real/checkpoint-10000
python Isaac-GR00T/gr00t/eval/run_gr00t_server.py --model-path /workspace/models/$MODEL

# 终端2：运行真实机器人评测
python Isaac-GR00T/gr00t/eval/real_robot/SO100/so101_eval.py \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --lang_instruction="Pick up the vial and place it in the yellow rack"
```

---

## 三、两种策略的组合关系

Strategy 1 和 Strategy 2 不是竞争关系——官方教程的实际操作顺序是：**先用 Strategy 1（域随机化）采集大量仿真数据打基础，再用 Strategy 2（Co-training）混入少量真实数据做校准**。这也是官方给出的"组合策略往往优于单一策略"这条结论最直接的体现——第一到六章讲到的所有工具,最终都是要组合在同一套训练流程里使用的构件,不是互相排斥的备选方案。

---

## 四、本章小结

| 策略 | 核心机制 | 关键代码/命令 | 需要的真实数据 |
|------|---------|--------------|---------------|
| Strategy 1: Domain Randomization | Isaac Lab `EventTerm` 在每次 reset 时注入随机化 | `randomize_sky_light`、`randomize_camera_pose`、`reset_vials_rack` | 不需要 |
| Strategy 2: Co-training | 少量真实数据 + 大量仿真数据混合训练 | `lerobot-record` 采集 + 数据集合并 | 约 5 条演示（教程案例） |

## 下章预告

第七章讲 Strategy 3——用 Cosmos 生成式模型对已有数据做视觉增强，这条路线比 Domain Randomization 更进一步，能生成"真正照片级真实"而不是"仿真渲染+随机贴图"的画面。

---

## 延伸阅读

- [Sim-to-Real Strategy 1: Domain Randomization（官方文档，本章前半部分来源）](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
- [Sim-to-Real Strategy 2: Co-Training With Real Data（官方文档，本章后半部分来源）](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html)
- Sim-and-Real Co-Training paper, RSS 2025 — [co-training.github.io](https://co-training.github.io/)
