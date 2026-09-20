import math

# 演示 alpha blending：三个不同深度的高斯覆盖同一个像素(屏幕中心)，
# 逐帧展示从远到近依次"叠色"的过程，并在 metrics 里打印每一层的
# alpha_i、累积透过率 T_i 以及它对最终颜色的贡献。对应第9章第三、四节。

# 三层高斯，按深度从远到近排列(layers[0]最远，layers[-1]最近)
layers = [
    {"depth": 3.0, "color": (0.90, 0.25, 0.25), "alpha0": 0.55, "G": 0.9, "label": "远层(红)"},
    {"depth": 2.0, "color": (0.20, 0.55, 0.95), "alpha0": 0.60, "G": 0.85, "label": "中层(蓝)"},
    {"depth": 1.0, "color": (0.25, 0.75, 0.35), "alpha0": 0.70, "G": 0.95, "label": "近层(绿)"},
]

for L in layers:
    L["alpha"] = L["alpha0"] * L["G"]  # 有效不透明度 = alpha0 * G，对应第一节公式


def blend(layers_so_far):
    # 严格按第三、四节公式：T从1开始，逐层累积
    T = 1.0
    C = [0.0, 0.0, 0.0]
    per_layer = []
    for L in layers_so_far:
        a = L["alpha"]
        contrib = [c * a * T for c in L["color"]]
        C = [C[k] + contrib[k] for k in range(3)]
        per_layer.append({"label": L["label"], "alpha": a, "T_before": T, "contrib": contrib})
        T = T * (1 - a)  # 递推更新透过率，对应第四节公式
    return C, T, per_layer


def to_hex(color):
    r, g, b = [max(0, min(255, int(c * 255))) for c in color]
    return f"#{r:02x}{g:02x}{b:02x}"


frames = []
# 第一部分：依次揭示 1层、2层、3层的合成结果(每个阶段停留若干帧)
stages = [layers[:1], layers[:2], layers[:3]]
hold_frames = 18

for stage_idx, stage_layers in enumerate(stages):
    C, T_final, per_layer = blend(stage_layers)
    for _ in range(hold_frames):
        objects = []
        # 画出所有层的椭圆(按深度从远到近的堆叠顺序，配合组件自身的深度排序)
        for idx, L in enumerate(layers):
            included = L in stage_layers
            objects.append({
                "type": "splat",
                "x": 0.0, "y": 0.0,
                "radiusA": 1.2 - idx * 0.25, "radiusB": 1.2 - idx * 0.25,
                "angle": 0.0,
                "depth": L["depth"],
                "color": to_hex(L["color"]),
                "alpha": L["alpha"] if included else 0.05,
                "label": L["label"] if included else f"{L['label']}(未加入)",
            })

        metrics = {"当前叠加层数": len(stage_layers), "最终颜色": to_hex(C),
                    "剩余透过率T": round(T_final, 4)}
        for pl in per_layer:
            metrics[f"{pl['label']} α"] = round(pl["alpha"], 3)
            metrics[f"{pl['label']} T(叠加前)"] = round(pl["T_before"], 3)

        frames.append({"objects": objects, "metrics": metrics,
                        "pixel": {"x": 0.0, "y": 0.0, "color": to_hex(C), "layers": []}})

simulation = {
    "bounds": [-2.0, -2.0, 2.0, 2.0],
    "fps": 12,
    "loop": True,
    "frames": frames,
}

print("三层从远到近: 红(depth=3) -> 蓝(depth=2) -> 绿(depth=1)")
for L in layers:
    print(f"{L['label']}: alpha0={L['alpha0']}, G={L['G']}, 有效alpha={round(L['alpha'],3)}")

C_all, T_all, per_layer_all = blend(layers)
print(f"\n三层全部叠加后的最终颜色(0~1): {[round(c,3) for c in C_all]}")
print(f"最终剩余透过率 T = {round(T_all,4)}  (值越小说明像素已经\"看饱和\"了)")
for pl in per_layer_all:
    print(f"  {pl['label']}: alpha={round(pl['alpha'],3)}, 叠加前T={round(pl['T_before'],3)}, "
          f"贡献={[round(c,3) for c in pl['contrib']]}")
