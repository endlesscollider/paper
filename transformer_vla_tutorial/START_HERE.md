# 从这里开始看

如果你不知道怎么看这个教程，按下面顺序来。

## 1. 先打开入口

```bash
cd /home/wahaha/paper/transformer_vla_tutorial
```

然后看：

```text
README.md
```

## 2. 推荐阅读顺序

新手顺序：

```text
chapters/00_big_picture.md
chapters/01_ml_minimum.md
chapters/02_attention_by_hand.md
chapters/03_transformer_architecture.md
chapters/04_from_language_to_control.md
chapters/05_act_action_chunking.md
chapters/06_vla_transformers.md
```

如果你只想快速理解 ACT / VLA：

```text
chapters/00_big_picture.md
chapters/02_attention_by_hand.md
chapters/05_act_action_chunking.md
chapters/06_vla_transformers.md
```

## 3. 怎么看图

这些文件是 Markdown。里面的图是 Mermaid 图。

推荐用：

- VS Code：右键 Markdown 文件，选择 `Open Preview`
- Typora
- Obsidian
- GitHub / GitLab Markdown 预览

如果只在终端里用 `cat` 或 `less`，图会显示成代码块，不会渲染成图。

## 4. 跑 attention 小实验

不用安装任何依赖：

```bash
python3 code/attention_no_dependencies.py
```

如果你安装了 NumPy，也可以运行：

```bash
python3 code/numpy_attention.py
```

## 5. 每章怎么学

每章按这个流程：

1. 先看图。
2. 再读文字。
3. 做章末思考练习。
4. 去 `exercises/answers_xx.md` 对答案。
5. 如果看不懂，就回到上一章。
