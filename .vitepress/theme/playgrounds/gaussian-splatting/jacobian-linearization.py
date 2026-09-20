# 用透视除法 y = 1/x 演示"局部线性化"：在参考点 x0 处画一条切线，
# 对比切线预测的 y 和真实的 y，观察 x0 越靠近原点（曲率越大）误差越明显。
# 你可以修改 x0 的值重新运行，对照第 5 章第三节的数值算例。

x0 = 1.0          # 参考点：对应"深度"，可以改成 0.3 试试曲率更大的区域
dx_max = 0.6      # 演示的位移范围，固定不变

def f(x):
    return 1.0 / x

def f_prime(x):
    # y = 1/x 的导数是 -1/x^2，这就是一维情形下的"雅可比矩阵"（1x1）
    return -1.0 / (x ** 2)

y0 = f(x0)
slope = f_prime(x0)

frames = []
steps = 60
for i in range(steps + 1):
    # dx 从 -dx_max 线性扫到 +dx_max，模拟"输入沿一个方向连续移动"
    dx = -dx_max + 2 * dx_max * i / steps
    x = x0 + dx
    if x <= 0.05:
        continue

    y_true = f(x)
    y_approx = y0 + slope * dx  # 局部线性近似：y0 + 斜率 * 位移

    objects = [
        # 用密集的小圆点近似画出真实曲线 y=1/x 在展示区间内的形状
    ]
    # 画真实曲线（用一串小段落连接近似展示曲线形状）
    n_curve = 40
    curve_points = []
    for j in range(n_curve + 1):
        xx = 0.15 + (2.2 - 0.15) * j / n_curve
        curve_points.append([xx, f(xx)])
    for j in range(len(curve_points) - 1):
        objects.append({
            "type": "segment", "from": curve_points[j], "to": curve_points[j + 1],
            "stroke": "#2196f3", "width": 2,
        })

    # 画切线（局部线性近似），只在参考点附近一段区间画出
    tangent_a = [x0 - dx_max, y0 + slope * (-dx_max)]
    tangent_b = [x0 + dx_max, y0 + slope * (dx_max)]
    objects.append({"type": "segment", "from": tangent_a, "to": tangent_b,
                     "stroke": "#ff9800", "width": 2, "label": "切线(线性近似)"})

    # 参考点
    objects.append({"type": "point", "x": x0, "y": y0, "fill": "#111827", "radius": 0.03, "label": "x0"})
    # 真实点和近似点
    objects.append({"type": "point", "x": x, "y": y_true, "fill": "#2196f3", "radius": 0.03, "label": "真实值"})
    objects.append({"type": "point", "x": x, "y": y_approx, "fill": "#ff9800", "radius": 0.03, "label": "近似值"})

    error = abs(y_true - y_approx)
    frames.append({
        "objects": objects,
        "metrics": {
            "x0": round(x0, 3),
            "dx": round(dx, 3),
            "真实y": round(y_true, 4),
            "近似y": round(y_approx, 4),
            "误差": round(error, 4),
        },
    })

simulation = {
    "bounds": [0.0, 0.0, 2.4, 1.0 / 0.15 * 0.15 + 2.0],
    "fps": 20,
    "loop": True,
    "frames": frames,
}
# 由于 y=1/x 在 x 很小时数值很大，收紧一个更适合观察的显示范围。
simulation["bounds"] = [0.0, -0.5, 2.4, 7.0]

print(f"参考点 x0={x0}, 局部斜率={round(slope,4)}")
print("拖动上面的运行按钮观察 dx 扫过 [-0.6, 0.6] 时，真实值(蓝)和线性近似(橙)之间的误差如何变化。")
