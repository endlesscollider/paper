# 可视化规范：多画图！

## 核心原则

**数学概念必须配图！** 文字和公式再清楚，都不如一张曲线图来得直观。尤其是概率分布、损失函数曲线、激活函数形状、梯度变化趋势——这些天生就是"图形胜过千言"的内容。

### ⚠️ 强制画图的场景

以下场景中**必须**生成可视化图片，不能只用文字/公式/表格：

1. **概率分布**：任何分布（高斯、Beta、Gamma、均匀、Dirichlet 等）都必须画出 PDF/CDF 曲线图，展示不同参数下的形状变化
2. **激活函数**：ReLU、GELU、Swish、Sigmoid、Tanh 等——必须画出函数曲线
3. **损失函数形状**：MSE vs Huber vs L1，clip 函数的形状，PPO 的 clip 区域——必须画图
4. **学习率调度**：Cosine、Linear warmup、Step decay——必须画出 lr 随 step 变化的曲线
5. **噪声调度**：扩散模型的 $\alpha_t$, $\sigma_t$ 随时间变化——必须画图
6. **数值对比**：当表格中有多组数值结果需要对比趋势时，应同时提供折线图/柱状图
7. **函数形状对比**：任何"A 和 B 的区别"如果可以通过画图展示，就必须画图

### 建议画图但不强制的场景

- 训练曲线示意图（loss 下降过程）
- 网络结构示意图（用 Mermaid 即可）
- 采样过程的示意（扩散的去噪过程）

---

## 画图工具使用方法

项目提供了通用画图脚本 `scripts/plot_function.py`，支持通过 JSON 配置生成任意函数图形。

### 调用方式

```bash
uv run --with matplotlib --with numpy --with scipy python scripts/plot_function.py <config.json路径>
```

### 最小配置示例

```json
{
  "output": "public/my_plot_name.png",
  "title": "Plot Title (English)",
  "xlabel": "x",
  "ylabel": "f(x)",
  "xlim": [0, 1],
  "ylim": [0, 5],
  "curves": [
    {
      "expr": "scipy.stats.beta.pdf(x, 2, 5)",
      "x_range": [0.001, 0.999, 500],
      "label": "Beta(2, 5)",
      "color": "#9C27B0"
    }
  ]
}
```

### 完整功能清单

| 字段 | 说明 |
|------|------|
| `output` | 输出路径，**必须**放在 `public/` 目录下 |
| `title` | 图标题（用英文，避免中文字体问题） |
| `xlabel` / `ylabel` | 坐标轴标签 |
| `xlim` / `ylim` | 坐标轴范围 `[min, max]` |
| `figsize` | 图片尺寸 `[width, height]`，默认 `[6, 4]`（见下方尺寸规范） |
| `dpi` | 分辨率，默认 150 |
| `grid` | 是否显示网格，默认 true |
| `legend_loc` | 图例位置，如 `"upper left"`、`"best"` |
| `curves` | 曲线数组（见下） |
| `reference_lines` | 参考线（水平 `y` 或竖直 `x`） |
| `points` | 标注点（带箭头文字） |
| `vertical_lines` | 竖直标注线 |
| `annotations` | 自由文字标注 |

#### curves 字段

```json
{
  "expr": "np.sin(x)",          // Python 表达式，x 是 numpy array
  "x_range": [0, 6.28, 500],   // [start, end, num_points]
  "label": "sin(x)",           // 图例文字
  "color": "#2196F3",          // 颜色
  "linewidth": 2.0,            // 线宽
  "linestyle": "-",            // "-", "--", ":", "-."
  "fill_alpha": 0.1            // 曲线下方填充透明度，0=不填充
}
```

#### 表达式中可用的变量/模块

- `x` — 自变量（numpy array）
- `np` / `numpy` — NumPy
- `scipy.stats` — SciPy 统计分布（如 `scipy.stats.beta.pdf(x, a, b)`）
- `math` — Python math 标准库

#### 多子图模式

```json
{
  "output": "public/multi.png",
  "nrows": 2,
  "ncols": 1,
  "figsize": [9, 10],
  "subplots": [
    { "title": "Plot 1", "curves": [...] },
    { "title": "Plot 2", "curves": [...] }
  ]
}
```

---

## 图片在文章中的引用

图片放在 `public/` 目录后，在 Markdown 中用绝对路径引用：

```markdown
![图片描述文字（中文）](/filename.png)
```

**图片命名规范**：
- 使用英文 + 下划线
- 格式：`<概念名>_<描述>.png`
- 例：`beta_distribution_pdf.png`、`ppo_clip_function.png`、`cosine_lr_schedule.png`

**图片描述文字规范**：
- 用中文写，简洁说明图的内容
- 例：`![Beta 分布 PDF 曲线——不同参数下的形状变化](/beta_distribution_pdf.png)`

**图片后必须跟说明**：
- 图片下方用 `>` 引用格式写 1-2 句话的图解说明
- 告诉读者应该重点关注图中的什么特征

---

## 画图时的美学规范

1. **颜色**：使用 Material Design 色板，不要用纯红纯蓝等高饱和色。推荐色号：
   - 蓝：`#2196F3`  绿：`#4CAF50`  紫：`#9C27B0`  橙：`#FF9800`
   - 红：`#F44336`  粉：`#E91E63`  青：`#009688`  灰：`#607D8B`
2. **线宽**：主曲线 2.0-2.5，参考线 1.0-1.5
3. **标题和标签用英文**：避免中文字体渲染问题（服务器上可能没有中文字体）
4. **图例不能挡住关键曲线**：选择合适的 `legend_loc`
5. **参考线**：画均匀分布/零线等基准，帮助读者判断"偏离基准多少"
6. **标注关键点**：如果文中算了具体数值（如 $f(0.75) = 1.30$），在图上标注出来
7. **figsize 根据内容复杂度选择**（重要！不要一律用大图）：

   | 图的内容 | 推荐 figsize | 文章中的 class |
   |----------|-------------|----------------|
   | 单曲线/单概念（如一个分布 PDF） | `[6, 4]` | 默认（不加 class） |
   | 单面板但内容多（如工作空间示意图） | `[6, 5.5]` ~ `[7, 6]` | 默认 |
   | 双子图并排对比 | `[11, 5]` ~ `[13, 5.5]` | `class="img-wide"` |
   | 多子图（2×1 或 2×2） | `[10, 8]` ~ `[12, 10]` | `class="img-wide"` |
   | 全景架构图/时间线 | `[14, 6]`+ | `class="img-full"` |

   **原则：图片的物理尺寸应该和它承载的信息量成正比。** 一张只有一条曲线的图，不需要占满整个屏幕宽度。

---

## 图片在文章中的尺寸控制

CSS 已配置三级图片宽度：

- **默认**：`max-width: 560px` — 适合单面板、简单内容的图
- **`class="img-wide"`**：`max-width: 780px` — 适合双子图、内容丰富的对比图
- **`class="img-full"`**：`max-width: 100%` — 适合全宽架构图

**使用方式**：
```markdown
<!-- 默认宽度（简单图）-->
![描述](/filename.png)

<!-- 宽图（双子图、对比图）-->
<img src="/filename.png" alt="描述" class="img-wide">

<!-- 全宽图（架构图、时间线）-->
<img src="/filename.png" alt="描述" class="img-full">
```

**判断标准**：如果图里只有 1-2 条曲线或一个简单示意图，用默认宽度。如果图是多面板对比或包含大量标注信息，用 `img-wide`。只有确实需要占满页面宽度的图（如多列架构图、长时间线）才用 `img-full`。

---

## 自查清单

写完一篇包含数学概念的文章后，问自己：

- [ ] 文中出现的概率分布有对应的曲线图吗？
- [ ] 文中出现的函数（激活函数、损失函数等）有形状图吗？
- [ ] 图中标注了文章里算过的关键数值点吗？
- [ ] 有没有"光看图就能理解 80% 内容"的可能——如果有，说明图做对了
- [ ] 图片存放在 `public/` 目录下，路径引用正确吗？

**记住**：一张好图胜过十段文字。读者扫一眼图就能建立直觉，然后再看文字去确认细节。图是"第一印象"，文字是"深入理解"。两者缺一不可，但如果只能选一个，选图。
