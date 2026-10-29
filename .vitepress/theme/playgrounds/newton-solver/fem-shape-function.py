# 单个三角形：拖动内部测试点，观察重心坐标（=形函数值）如何变化。
# 然后把顶点 v2 向下压缩 40%，看弹性回复力的方向和大小。
import math

# ============ 参考构型：一个直角三角形 ============
tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

# 形函数梯度（只依赖参考构型）
x0, x1, x2 = tri
area2 = (x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])
area = abs(area2) / 2
gN = [
    [(x1[1]-x2[1])/area2, (x2[0]-x1[0])/area2],
    [(x2[1]-x0[1])/area2, (x0[0]-x2[0])/area2],
    [(x0[1]-x1[1])/area2, (x1[0]-x0[0])/area2],
]

# ============ 演示 1：重心坐标 ============
# 在三角形内部取几个采样点，展示形函数值
sample_points = [[0.25, 0.25], [0.5, 0.1], [0.1, 0.6], [0.33, 0.33]]

frames_bary = []
for px, py in sample_points:
    # 用面积比算重心坐标（= 形函数值）
    def tri_area(ax, ay, bx, by, cx, cy):
        return 0.5 * abs((bx-ax)*(cy-ay) - (cx-ax)*(by-ay))

    total = tri_area(tri[0][0], tri[0][1], tri[1][0], tri[1][1], tri[2][0], tri[2][1])
    N0 = tri_area(px, py, tri[1][0], tri[1][1], tri[2][0], tri[2][1]) / total
    N1 = tri_area(tri[0][0], tri[0][1], px, py, tri[2][0], tri[2][1]) / total
    N2 = tri_area(tri[0][0], tri[0][1], tri[1][0], tri[1][1], px, py) / total

    objects = [
        # 三个子三角形用不同颜色
        {"type": "polygon", "points": [[px, py], tri[1][:], tri[2][:]], "fill": "#FFCDD2", "stroke": "none"},
        {"type": "polygon", "points": [tri[0][:], [px, py], tri[2][:]], "fill": "#BBDEFB", "stroke": "none"},
        {"type": "polygon", "points": [tri[0][:], tri[1][:], [px, py]], "fill": "#C8E6C9", "stroke": "none"},
        # 大三角形边框
        {"type": "polygon", "points": [tri[0][:], tri[1][:], tri[2][:]], "fill": "none", "stroke": "#1565C0", "width": 2},
        # 三个顶点
        {"type": "circle", "x": tri[0][0], "y": tri[0][1], "radius": 0.035, "fill": "#D32F2F", "label": "v0"},
        {"type": "circle", "x": tri[1][0], "y": tri[1][1], "radius": 0.035, "fill": "#1976D2", "label": "v1"},
        {"type": "circle", "x": tri[2][0], "y": tri[2][1], "radius": 0.035, "fill": "#388E3C", "label": "v2"},
        # 采样点
        {"type": "circle", "x": px, "y": py, "radius": 0.03, "fill": "#FF6F00", "label": "x"},
    ]
    frames_bary.append({
        "objects": objects,
        "metrics": {"N0": f"{N0:.3f}", "N1": f"{N1:.3f}", "N2": f"{N2:.3f}",
                    "sum": f"{N0+N1+N2:.3f}"},
    })

# ============ 演示 2：压缩变形 + 弹性回复力 ============
# 把 v2 从 (0,1) 压到 (0,0.6)，看形变梯度和回复力
cur = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.6]]
mu, lam = 0.1, 0.2

# 算 F
F00 = cur[0][0]*gN[0][0] + cur[1][0]*gN[1][0] + cur[2][0]*gN[2][0]
F01 = cur[0][0]*gN[0][1] + cur[1][0]*gN[1][1] + cur[2][0]*gN[2][1]
F10 = cur[0][1]*gN[0][0] + cur[1][1]*gN[1][0] + cur[2][1]*gN[2][0]
F11 = cur[0][1]*gN[0][1] + cur[1][1]*gN[1][1] + cur[2][1]*gN[2][1]
J = F00*F11 - F01*F10

# 算 PK1 应力
inv_det = 1.0 / (F00*F11 - F01*F10)
Fit00, Fit01 = F11*inv_det, -F10*inv_det
Fit10, Fit11 = -F01*inv_det, F00*inv_det
c = lam * (J - 1) * J
P00 = mu*(F00-Fit00) + c*Fit00
P01 = mu*(F01-Fit01) + c*Fit01
P10 = mu*(F10-Fit10) + c*Fit10
P11 = mu*(F11-Fit11) + c*Fit11

# 算节点力
node_forces = []
for i in range(3):
    fx = -area * (P00*gN[i][0] + P01*gN[i][1])
    fy = -area * (P10*gN[i][0] + P11*gN[i][1])
    node_forces.append([fx, fy])

# 力箭头的缩放（让箭头可见）
scale = 3.0

frames_force = []
for t in range(60):
    # 从参考构型渐变到压缩构型（动画效果）
    frac = min(t / 30.0, 1.0)
    interp = [
        [tri[i][0]*(1-frac) + cur[i][0]*frac,
         tri[i][1]*(1-frac) + cur[i][1]*frac]
        for i in range(3)
    ]
    objects = [
        # 参考构型（灰色虚线）
        {"type": "polygon", "points": [tri[0][:], tri[1][:], tri[2][:]], "fill": "none", "stroke": "#bdbdbd", "width": 1},
        # 当前构型
        {"type": "polygon", "points": [interp[0][:], interp[1][:], interp[2][:]], "fill": "#BBDEFB", "stroke": "#1565C0", "width": 2},
        {"type": "circle", "x": interp[0][0], "y": interp[0][1], "radius": 0.035, "fill": "#D32F2F", "label": "v0"},
        {"type": "circle", "x": interp[1][0], "y": interp[1][1], "radius": 0.035, "fill": "#1976D2", "label": "v1"},
        {"type": "circle", "x": interp[2][0], "y": interp[2][1], "radius": 0.035, "fill": "#388E3C", "label": "v2"},
    ]
    # 压缩完成后显示力箭头
    if frac >= 1.0:
        colors = ["#D32F2F", "#1976D2", "#388E3C"]
        for i in range(3):
            mag = math.hypot(node_forces[i][0], node_forces[i][1])
            if mag > 1e-6:
                objects.append({
                    "type": "segment",
                    "from": cur[i][:],
                    "to": [cur[i][0] + scale*node_forces[i][0],
                           cur[i][1] + scale*node_forces[i][1]],
                    "stroke": "#D32F2F", "width": 3,
                })
    frames_force.append({
        "objects": objects,
        "metrics": {"J": f"{J:.3f}", "area ratio": f"{J:.0%}",
                    "F": f"[[{F00:.2f},{F01:.2f}],[{F10:.2f},{F11:.2f}]]"},
    })

# 合并：先播放重心坐标（每帧停 1 秒 = 30 帧），再播放压缩动画
all_frames = []
for f in frames_bary:
    for _ in range(30):
        all_frames.append(f)
all_frames += frames_force

f_sum = [node_forces[0][i]+node_forces[1][i]+node_forces[2][i] for i in range(2)]
print(f"F = [[{F00:.3f}, {F01:.3f}], [{F10:.3f}, {F11:.3f}]]")
print(f"J = {J:.3f}")
print(f"force sum = [{f_sum[0]:.6f}, {f_sum[1]:.6f}]  (should be ~0)")

simulation = {"bounds": [-0.2, -0.4, 1.2, 1.15], "fps": 30, "loop": True, "frames": all_frames}
