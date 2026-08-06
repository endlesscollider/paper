#!/usr/bin/env python3
"""
通用函数图形绘制工具 —— 供 agent 在写文章时生成数学函数的可视化图片。

⚠️ figsize 选择指南（重要！不要一律用大图）：
┌────────────────────────────────┬─────────────────┬──────────────────────────┐
│ 图的内容                        │ 推荐 figsize     │ 文章中的 CSS class        │
├────────────────────────────────┼─────────────────┼──────────────────────────┤
│ 单曲线/单概念（如一个分布 PDF）    │ [6, 4]          │ 默认（不加 class）         │
│ 单面板内容较多（工作空间等）       │ [6, 5.5]~[7, 6] │ 默认                      │
│ 双子图并排对比                   │ [11, 5]~[13,5.5]│ class="img-wide"         │
│ 多子图（2×1 或 2×2）            │ [10, 8]~[12,10] │ class="img-wide"         │
│ 全景架构图/时间线                │ [14, 6]+        │ class="img-full"         │
└────────────────────────────────┴─────────────────┴──────────────────────────┘
原则：图片物理尺寸应与信息量成正比。单曲线图不需要 [9, 5.5]！

文章中引用方式：
- 默认（简单图）:  ![描述](/filename.png)
- 宽图（对比图）:  <img src="/filename.png" alt="描述" class="img-wide">
- 全宽（架构图）:  <img src="/filename.png" alt="描述" class="img-full">

用法：
    uv run --with matplotlib --with numpy --with scipy python scripts/plot_function.py <config.json>

config.json 格式（所有字段见下方示例）：

{
  "output": "public/my_plot.png",
  "title": "My Function Plot",
  "xlabel": "x",
  "ylabel": "f(x)",
  "xlim": [0, 1],
  "ylim": [0, 5],
  "figsize": [6, 4],
  "dpi": 150,
  "grid": true,
  "legend_loc": "upper right",
  "reference_lines": [
    {"y": 1, "label": "y=1 baseline", "color": "gray", "style": "--"}
  ],
  "curves": [
    {
      "expr": "np.sin(x)",
      "x_range": [0, 6.28, 500],
      "label": "sin(x)",
      "color": "#2196F3",
      "linewidth": 2.0,
      "linestyle": "-",
      "fill_alpha": 0.0
    },
    {
      "expr": "scipy.stats.beta.pdf(x, 2, 5)",
      "x_range": [0.001, 0.999, 500],
      "label": "Beta(2,5)",
      "color": "#E91E63",
      "linewidth": 2.2,
      "linestyle": "-",
      "fill_alpha": 0.1
    }
  ],
  "annotations": [
    {"x": 0.5, "y": 2.0, "text": "peak", "fontsize": 10, "color": "#333"}
  ],
  "points": [
    {"x": 0.5, "y": 2.0, "color": "#E91E63", "size": 8, "label_text": "(0.5, 2.0)"}
  ],
  "vertical_lines": [
    {"x": 0.6, "color": "#2196F3", "style": "--", "label": "mean=0.6"}
  ]
}

支持的数学表达式环境：
- numpy 以 np 暴露
- scipy.stats 以 scipy.stats 暴露
- math 标准库以 math 暴露
- x 是自变量（numpy array）

示例调用：
    uv run --with matplotlib --with numpy --with scipy python scripts/plot_function.py /tmp/plot_config.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# 让 scipy.stats 可用于 eval
try:
    import scipy.stats
except ImportError:
    scipy = None

import math


def plot_from_config(config: dict) -> str:
    """根据配置字典生成图片，返回输出路径。"""
    
    figsize = config.get("figsize", [6, 4])
    dpi = config.get("dpi", 150)
    
    # ⚠️ 提醒 agent：单面板图不需要很大的 figsize
    num_curves = len(config.get("curves", []))
    if figsize[0] >= 9 and num_curves <= 3:
        print(f"⚠️  提示：当前 figsize={figsize}，但只有 {num_curves} 条曲线。"
              f"考虑用更小的 figsize（如 [6, 4]）以避免图片在文章中过大。")
    
    fig, ax = plt.subplots(figsize=tuple(figsize))
    
    # 画参考线
    for ref in config.get("reference_lines", []):
        if "y" in ref:
            ax.axhline(y=ref["y"], color=ref.get("color", "gray"),
                      linestyle=ref.get("style", "--"), alpha=ref.get("alpha", 0.5),
                      linewidth=ref.get("linewidth", 1))
            if ref.get("label"):
                ax.text(0.02, ref["y"] + 0.05, ref["label"], fontsize=9,
                       color=ref.get("color", "gray"), alpha=0.7,
                       transform=ax.get_yaxis_transform())
        if "x" in ref:
            ax.axvline(x=ref["x"], color=ref.get("color", "gray"),
                      linestyle=ref.get("style", "--"), alpha=ref.get("alpha", 0.5),
                      linewidth=ref.get("linewidth", 1))
    
    # 画曲线
    for curve in config.get("curves", []):
        x_range = curve.get("x_range", [0, 1, 500])
        x = np.linspace(x_range[0], x_range[1], int(x_range[2]) if len(x_range) > 2 else 500)
        
        # 安全 eval 表达式
        expr = curve["expr"]
        eval_ns = {"np": np, "numpy": np, "math": math, "x": x}
        if scipy:
            eval_ns["scipy"] = __import__("scipy")
            eval_ns["stats"] = scipy.stats
        y = eval(expr, eval_ns)
        
        plot_kwargs = {
            "linewidth": curve.get("linewidth", 2.0),
            "color": curve.get("color"),
            "linestyle": curve.get("linestyle", "-"),
            "label": curve.get("label"),
            "alpha": curve.get("alpha", 1.0),
        }
        # 移除 None 值
        plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
        
        ax.plot(x, y, **plot_kwargs)
        
        # 可选填充
        fill_alpha = curve.get("fill_alpha", 0.0)
        if fill_alpha > 0:
            ax.fill_between(x, y, alpha=fill_alpha, color=curve.get("color"))
    
    # 画标注点
    for pt in config.get("points", []):
        ax.plot(pt["x"], pt["y"], 'o', markersize=pt.get("size", 8),
               color=pt.get("color", "#E91E63"), zorder=5)
        if pt.get("label_text"):
            ax.annotate(pt["label_text"], xy=(pt["x"], pt["y"]),
                       xytext=(pt["x"] + pt.get("text_offset_x", 0.02),
                               pt["y"] + pt.get("text_offset_y", 0.1)),
                       fontsize=pt.get("fontsize", 10), color=pt.get("text_color", "#333"),
                       fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='#999', lw=1.2))
    
    # 画竖直线
    for vl in config.get("vertical_lines", []):
        ax.axvline(x=vl["x"], color=vl.get("color", "#2196F3"),
                  linestyle=vl.get("style", "--"), alpha=vl.get("alpha", 0.7),
                  linewidth=vl.get("linewidth", 1.5), label=vl.get("label"))
    
    # 画文字注释
    for ann in config.get("annotations", []):
        ax.text(ann["x"], ann["y"], ann["text"],
               fontsize=ann.get("fontsize", 10), color=ann.get("color", "#333"),
               ha=ann.get("ha", "left"), va=ann.get("va", "bottom"),
               fontweight=ann.get("fontweight", "normal"),
               bbox=ann.get("bbox"))
    
    # 设置坐标轴
    if "xlim" in config:
        ax.set_xlim(config["xlim"])
    if "ylim" in config:
        ax.set_ylim(config["ylim"])
    
    ax.set_xlabel(config.get("xlabel", "x"), fontsize=13)
    ax.set_ylabel(config.get("ylabel", "y"), fontsize=13)
    
    if config.get("title"):
        ax.set_title(config["title"], fontsize=14, fontweight='bold')
    
    if config.get("grid", True):
        ax.grid(True, alpha=0.3)
    
    # 图例
    legend_loc = config.get("legend_loc", "best")
    if legend_loc and any(c.get("label") for c in config.get("curves", [])):
        ax.legend(fontsize=config.get("legend_fontsize", 10), loc=legend_loc, framealpha=0.9)
    
    plt.tight_layout()
    
    output = config.get("output", "public/plot.png")
    plt.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    return output


def main():
    if len(sys.argv) < 2:
        # 如果没有提供配置文件，从 stdin 读取
        config_text = sys.stdin.read()
    else:
        config_path = sys.argv[1]
        with open(config_path, 'r') as f:
            config_text = f.read()
    
    config = json.loads(config_text)
    
    # 支持多子图模式
    if "subplots" in config:
        # 多子图：每个 subplot 是一个独立的 config
        nrows = config.get("nrows", len(config["subplots"]))
        ncols = config.get("ncols", 1)
        figsize = config.get("figsize", [9, 5 * nrows])
        dpi = config.get("dpi", 150)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=tuple(figsize))
        if nrows * ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = list(axes)
        else:
            axes = axes.flatten().tolist()
        
        for i, subplot_config in enumerate(config["subplots"]):
            ax = axes[i]
            plt.sca(ax)
            
            # 画参考线
            for ref in subplot_config.get("reference_lines", []):
                if "y" in ref:
                    ax.axhline(y=ref["y"], color=ref.get("color", "gray"),
                              linestyle=ref.get("style", "--"), alpha=ref.get("alpha", 0.5),
                              linewidth=ref.get("linewidth", 1))
                if "x" in ref:
                    ax.axvline(x=ref["x"], color=ref.get("color", "gray"),
                              linestyle=ref.get("style", "--"), alpha=ref.get("alpha", 0.5),
                              linewidth=ref.get("linewidth", 1))
            
            # 画曲线
            for curve in subplot_config.get("curves", []):
                x_range = curve.get("x_range", [0, 1, 500])
                x = np.linspace(x_range[0], x_range[1], int(x_range[2]) if len(x_range) > 2 else 500)
                expr = curve["expr"]
                eval_ns = {"np": np, "numpy": np, "math": math, "x": x}
                if scipy:
                    eval_ns["scipy"] = __import__("scipy")
                    eval_ns["stats"] = scipy.stats
                y = eval(expr, eval_ns)
                
                plot_kwargs = {
                    "linewidth": curve.get("linewidth", 2.0),
                    "color": curve.get("color"),
                    "linestyle": curve.get("linestyle", "-"),
                    "label": curve.get("label"),
                    "alpha": curve.get("alpha", 1.0),
                }
                plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
                ax.plot(x, y, **plot_kwargs)
                
                fill_alpha = curve.get("fill_alpha", 0.0)
                if fill_alpha > 0:
                    ax.fill_between(x, y, alpha=fill_alpha, color=curve.get("color"))
            
            # 画标注点
            for pt in subplot_config.get("points", []):
                ax.plot(pt["x"], pt["y"], 'o', markersize=pt.get("size", 8),
                       color=pt.get("color", "#E91E63"), zorder=5)
                if pt.get("label_text"):
                    ax.annotate(pt["label_text"], xy=(pt["x"], pt["y"]),
                               xytext=(pt["x"] + pt.get("text_offset_x", 0.02),
                                       pt["y"] + pt.get("text_offset_y", 0.1)),
                               fontsize=pt.get("fontsize", 10), color=pt.get("text_color", "#333"),
                               arrowprops=dict(arrowstyle='->', color='#999', lw=1.2))
            
            # 画竖直线
            for vl in subplot_config.get("vertical_lines", []):
                ax.axvline(x=vl["x"], color=vl.get("color", "#2196F3"),
                          linestyle=vl.get("style", "--"), alpha=vl.get("alpha", 0.7),
                          linewidth=vl.get("linewidth", 1.5), label=vl.get("label"))
            
            # 设置
            if "xlim" in subplot_config:
                ax.set_xlim(subplot_config["xlim"])
            if "ylim" in subplot_config:
                ax.set_ylim(subplot_config["ylim"])
            ax.set_xlabel(subplot_config.get("xlabel", "x"), fontsize=12)
            ax.set_ylabel(subplot_config.get("ylabel", "y"), fontsize=12)
            if subplot_config.get("title"):
                ax.set_title(subplot_config["title"], fontsize=13, fontweight='bold')
            if subplot_config.get("grid", True):
                ax.grid(True, alpha=0.3)
            if any(c.get("label") for c in subplot_config.get("curves", [])):
                ax.legend(fontsize=9, loc=subplot_config.get("legend_loc", "best"), framealpha=0.9)
        
        plt.tight_layout()
        output = config.get("output", "public/plot.png")
        plt.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✅ {output}")
    else:
        # 单图模式
        output = plot_from_config(config)
        print(f"✅ {output}")


if __name__ == "__main__":
    main()
