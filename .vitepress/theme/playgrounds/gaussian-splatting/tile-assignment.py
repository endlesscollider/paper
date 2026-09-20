import math

# 演示 tile 分配：屏幕被切成若干 tile 网格，一个椭圆(用轴对齐包围盒近似)
# 移动/变大时，哪些 tile 会被标记为"需要处理这个高斯"。
# 对应第 8 章第二节的包围盒近似和 duplicate 步骤。

screen_w, screen_h = 8.0, 6.0   # 用"世界单位"模拟屏幕范围
tile_size = 1.0                  # 每个 tile 的边长
n_tiles_x = int(screen_w / tile_size)
n_tiles_y = int(screen_h / tile_size)

frames = []
n_frames = 60
for i in range(n_frames):
    t = i / n_frames
    # 椭圆中心沿一条对角线来回移动，半径周期性变化
    center_x = screen_w / 2 + 2.5 * math.sin(2 * math.pi * t)
    center_y = screen_h / 2 + 1.8 * math.cos(2 * math.pi * t * 1.3)
    radius_a = 0.6 + 0.5 * (0.5 + 0.5 * math.sin(2 * math.pi * t * 0.7))
    radius_b = radius_a * 0.6
    angle = t * math.pi

    # 包围盒：简化地取 max(radius_a, radius_b) 的 1.0 倍作为半宽半高
    # (完整实现应考虑旋转后的真实投影范围，这里用保守的圆形包围近似，
    #  对应第8章"包围盒近似，允许假阳性"的取舍)
    half_extent = max(radius_a, radius_b) * 1.15
    bbox_min_x, bbox_max_x = center_x - half_extent, center_x + half_extent
    bbox_min_y, bbox_max_y = center_y - half_extent, center_y + half_extent

    objects = []
    covered_tiles = 0
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_min_x, tile_max_x = tx * tile_size, (tx + 1) * tile_size
            tile_min_y, tile_max_y = ty * tile_size, (ty + 1) * tile_size
            # 轴对齐矩形相交测试：两个矩形在x和y方向都有重叠才算相交
            overlap = not (bbox_max_x < tile_min_x or bbox_min_x > tile_max_x or
                            bbox_max_y < tile_min_y or bbox_min_y > tile_max_y)
            if overlap:
                covered_tiles += 1
                objects.append({
                    "type": "segment",
                    "from": [tile_min_x, tile_min_y], "to": [tile_max_x, tile_min_y],
                    "stroke": "#ff9800", "width": 2,
                })
            # 画出所有 tile 的网格线（浅色），帮助看清整体划分
            objects.append({"type": "segment", "from": [tile_min_x, tile_min_y],
                             "to": [tile_max_x, tile_min_y], "stroke": "#cbd5e1", "width": 1})
            objects.append({"type": "segment", "from": [tile_min_x, tile_min_y],
                             "to": [tile_min_x, tile_max_y], "stroke": "#cbd5e1", "width": 1})

    # 画椭圆本身（真正的高斯投影形状，包围盒只是它的近似）
    objects.append({
        "type": "splat", "x": center_x, "y": center_y,
        "radiusA": radius_a, "radiusB": radius_b, "angle": angle,
        "depth": 1.0, "color": "#2196f3", "alpha": 0.55, "label": "高斯投影椭圆",
    })
    # 画包围盒边界（四条线）
    box_corners = [
        [bbox_min_x, bbox_min_y], [bbox_max_x, bbox_min_y],
        [bbox_max_x, bbox_max_y], [bbox_min_x, bbox_max_y],
    ]
    for k in range(4):
        objects.append({"type": "segment", "from": box_corners[k], "to": box_corners[(k+1) % 4],
                         "stroke": "#111827", "width": 2})

    frames.append({
        "objects": objects,
        "metrics": {
            "覆盖tile数": covered_tiles,
            "总tile数": n_tiles_x * n_tiles_y,
            "包围盒半宽": round(half_extent, 2),
        },
    })

simulation = {
    "bounds": [0, 0, screen_w, screen_h],
    "fps": 20,
    "loop": True,
    "frames": frames,
}

print(f"屏幕划分为 {n_tiles_x} x {n_tiles_y} = {n_tiles_x*n_tiles_y} 个 tile。")
print("黑色矩形是椭圆的轴对齐包围盒；橙色描边的 tile 就是被这个高斯 duplicate 到的 tile。")
