#!/usr/bin/env python3
"""
每日快报自动生成脚本
===================
数据源：
  1. arXiv RSS (cs.RO, cs.LG, cs.AI, cs.MA, q-fin)
  2. Hugging Face Daily Papers
  3. GitHub Trending (machine-learning, reinforcement-learning, robotics)

生成：一篇 Markdown 快报，放到 每日快报/ 目录。

环境变量：
  OPENAI_API_KEY    — OpenAI 兼容 API key（必需）
  OPENAI_BASE_URL   — API base URL（可选，默认 https://api.openai.com/v1）
  OPENAI_MODEL      — 模型名（可选，默认 gpt-4o-mini）
  REPORT_DATE       — 指定日期 YYYY-MM-DD（可选，默认今天）
"""

import os
import sys
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

ARXIV_CATEGORIES = [
    "cs.RO",   # Robotics
    "cs.LG",   # Machine Learning
    "cs.AI",   # Artificial Intelligence
    "cs.MA",   # Multi-Agent Systems
    "q-fin",   # Quantitative Finance (all sub-categories)
    "stat.ML", # Statistics - Machine Learning
]

GITHUB_TOPICS = [
    "reinforcement-learning",
    "robotics",
    "deep-learning",
    "robot-simulation",
    "quantitative-finance",
]

# 关键词过滤：只保留跟我们领域高度相关的论文
KEYWORDS = [
    # 机器人
    "robot", "manipulation", "grasp", "locomotion", "embodied",
    "sim-to-real", "sim2real", "imitation learning", "teleoperation",
    "action chunking", "VLA", "visuomotor", "dexterous",
    # 深度学习
    "transformer", "diffusion", "foundation model", "vision-language",
    "large language model", "LLM", "multimodal", "representation learning",
    # 强化学习
    "reinforcement learning", "policy gradient", "PPO", "SAC", "RLHF",
    "reward model", "offline RL", "model-based RL", "multi-agent",
    # 仿真
    "simulation", "Isaac", "MuJoCo", "PyBullet", "SAPIEN",
    "digital twin", "physics engine",
    # 金融量化
    "quantitative", "trading", "portfolio", "financial",
    "market making", "option pricing", "risk",
]

MAX_PAPERS_PER_CATEGORY = 30  # 从每个 arXiv 类别最多取多少篇
MAX_FINAL_ITEMS = 8  # 每个板块最终展示的条目数


# ──────────────────────────────────────────────
# 数据拉取
# ──────────────────────────────────────────────

def fetch_arxiv_papers() -> list[dict]:
    """从 arXiv RSS 获取最新论文"""
    papers = []
    for cat in ARXIV_CATEGORIES:
        url = f"http://export.arxiv.org/rss/{cat}"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:MAX_PAPERS_PER_CATEGORY]:
                # 清理标题（arXiv RSS 标题可能带有 HTML tag）
                title = re.sub(r'<[^>]+>', '', entry.get("title", "")).strip()
                # 去除 arXiv 格式的 "(arXiv:xxxx.xxxxx vN [cs.RO])" 后缀
                title = re.sub(r'\s*\(arXiv:[^)]+\)\s*$', '', title)
                summary = entry.get("summary", "")[:500]
                link = entry.get("link", "")
                papers.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "source": f"arXiv:{cat}",
                })
        except Exception as e:
            print(f"  [WARN] Failed to fetch arXiv {cat}: {e}")
    return papers


def fetch_hf_daily_papers() -> list[dict]:
    """从 Hugging Face Daily Papers 页面抓取"""
    papers = []
    url = "https://huggingface.co/papers"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "DailyReport/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # HF papers 页面结构：每篇论文在 article 标签或特定 class 下
        # 使用通用匹配策略
        for article in soup.select("article, [class*='paper']")[:20]:
            title_el = article.select_one("h3, h2, [class*='title']")
            link_el = article.select_one("a[href*='/papers/']")
            if title_el:
                title = title_el.get_text(strip=True)
                link = ""
                if link_el:
                    href = link_el.get("href", "")
                    link = f"https://huggingface.co{href}" if href.startswith("/") else href
                papers.append({
                    "title": title,
                    "summary": "",
                    "link": link,
                    "source": "HuggingFace Daily",
                })
    except Exception as e:
        print(f"  [WARN] Failed to fetch HF papers: {e}")
    return papers


def fetch_github_trending() -> list[dict]:
    """从 GitHub Trending 抓取相关项目"""
    projects = []
    for topic in GITHUB_TOPICS:
        url = f"https://github.com/trending?since=daily&spoken_language_code=&language=&topic={topic}"
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "DailyReport/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for row in soup.select("[class*='Box-row'], article.Box-row")[:5]:
                name_el = row.select_one("h2 a, h1 a")
                desc_el = row.select_one("p")
                if name_el:
                    name = name_el.get_text(strip=True).replace("\n", "").replace(" ", "")
                    href = name_el.get("href", "")
                    link = f"https://github.com{href}" if href.startswith("/") else href
                    desc = desc_el.get_text(strip=True) if desc_el else ""
                    projects.append({
                        "title": name,
                        "summary": desc,
                        "link": link,
                        "source": f"GitHub Trending ({topic})",
                    })
        except Exception as e:
            print(f"  [WARN] Failed to fetch GitHub trending for {topic}: {e}")
    return projects


def keyword_filter(items: list[dict]) -> list[dict]:
    """按关键词过滤，保留领域相关的条目"""
    filtered = []
    for item in items:
        text = (item["title"] + " " + item.get("summary", "")).lower()
        if any(kw.lower() in text for kw in KEYWORDS):
            filtered.append(item)
    # 如果过滤后太少，放宽标准返回全部
    return filtered if len(filtered) >= 5 else items


# ──────────────────────────────────────────────
# LLM 生成
# ──────────────────────────────────────────────

def generate_report_with_llm(
    arxiv_papers: list[dict],
    hf_papers: list[dict],
    github_projects: list[dict],
    report_date: str,
) -> str:
    """调用 LLM 生成结构化快报"""
    
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # 准备输入数据摘要
    data_summary = f"""
## 今日原始数据（{report_date}）

### arXiv 论文（共 {len(arxiv_papers)} 篇）
{json.dumps(arxiv_papers[:50], ensure_ascii=False, indent=None)[:8000]}

### Hugging Face Daily Papers（共 {len(hf_papers)} 篇）
{json.dumps(hf_papers[:20], ensure_ascii=False, indent=None)[:3000]}

### GitHub Trending 项目（共 {len(github_projects)} 个）
{json.dumps(github_projects[:30], ensure_ascii=False, indent=None)[:4000]}
"""

    system_prompt = """你是一个 AI/机器人领域的资深技术编辑，负责生成高质量中文每日快报。

## 分类板块

从原始数据中筛选最重要的条目，按以下板块分类：
- 🤖 机器人 & 具身智能
- 🧠 深度学习 & 大模型
- 🎮 强化学习
- 🏭 机器人仿真
- 💰 金融量化
- 💻 工程项目 / GitHub（如有重要开源项目）

每个板块选 3-8 条最值得关注的。如果某板块没有相关内容可省略。

## ⚠️ 每条论文/项目的格式要求（严格执行，不可简化）

每条必须包含以下全部字段，缺一不可：

```
### N. [论文标题](链接) ★★★☆☆

**机构**：第一作者所在机构全称、通讯作者背景（教授职称、实验室名、代表性方向），如有知名实验室或公司标注出来。

**背景**：2-3 句话说明这篇工作解决什么问题、为什么这个问题重要、现有方法有什么不足。

**方法**：3-5 句话说明核心技术方案，具体到方法名称、关键模块设计、与已有方法的区别。

**效果**：量化结果——在什么数据集/任务上、比什么基线好多少（给具体数字：百分比提升、绝对指标等）。如论文有被顶会接收请标注（如 ICRA 2026、IROS 2026 等）。

**点评**：1-2 句话给出你的专业判断——这篇工作的亮点、局限性、对领域的意义。
```

### 星级评价标准（★ 1-5 星）

- ★★★★★：顶级团队 + 突破性方法 + 强量化验证 + 顶会接收
- ★★★★☆：知名团队或顶会接收 + 方法有明确创新 + 效果显著
- ★★★☆☆：方法有亮点但改进渐进，或团队/venue 一般但思路新颖
- ★★☆☆☆：工作扎实但缺乏突破，或验证不充分
- ★☆☆☆☆：初步探索、workshop 论文、或存在明显不足

评价要综合考虑：机构声誉、发表 venue（已接收 > 预印本）、方法创新程度、实验验证充分度（真机 > 仿真）、量化效果的绝对水平。

## 🔥 今日亮点

最后加一个「## 🔥 今日亮点」板块，挑 2-3 条当天最重磅的工作展开讲解：
- 每条 5-8 句话
- 要说清楚：为什么这篇是今天最值得关注的、它解决了什么核心痛点、对领域的推动是什么

## 其他要求

- 用中文写作，技术术语保留英文
- 不要编造不存在的链接，只使用提供数据中的链接
- 如果数据中没有 abstract/summary，根据标题推测方向，标注"（摘要待补充）"
- 每条的机构信息如果从数据中无法确定，写"机构信息待核实"，不要编造

输出纯 Markdown 正文（不含 frontmatter，不含一级标题）。"""

    user_prompt = f"请根据以下今日数据生成 {report_date} 的每日快报：\n{data_summary}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=12000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] LLM API call failed: {e}")
        return generate_fallback_report(arxiv_papers, hf_papers, github_projects)


def generate_fallback_report(
    arxiv_papers: list[dict],
    hf_papers: list[dict],
    github_projects: list[dict],
) -> str:
    """LLM 调用失败时的降级方案：直接列出条目"""
    lines = []
    
    if arxiv_papers:
        lines.append("## 📄 今日 arXiv 论文\n")
        for p in arxiv_papers[:15]:
            title = p["title"]
            link = p["link"]
            source = p["source"]
            lines.append(f"- **[{title}]({link})** `{source}`")
        lines.append("")

    if hf_papers:
        lines.append("## 🤗 Hugging Face Daily Papers\n")
        for p in hf_papers[:10]:
            title = p["title"]
            link = p.get("link", "")
            if link:
                lines.append(f"- **[{title}]({link})**")
            else:
                lines.append(f"- **{title}**")
        lines.append("")

    if github_projects:
        lines.append("## 🔥 GitHub Trending\n")
        for p in github_projects[:10]:
            title = p["title"]
            link = p["link"]
            desc = p.get("summary", "")
            lines.append(f"- **[{title}]({link})** — {desc}")
        lines.append("")

    if not lines:
        lines.append("> 今日数据拉取异常，请稍后查看。")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def get_weekday_cn(date: datetime) -> str:
    """获取中文星期"""
    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return days[date.weekday()]


def main():
    # 确定日期
    date_str = os.environ.get("REPORT_DATE", "")
    if date_str:
        report_date = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        # 用北京时间
        report_date = datetime.now(timezone(timedelta(hours=8)))
    
    date_formatted = report_date.strftime("%Y-%m-%d")
    weekday = get_weekday_cn(report_date)
    
    print(f"📰 生成每日快报: {date_formatted} ({weekday})")
    
    # 1. 拉取数据
    print("  [1/4] 拉取 arXiv RSS...")
    arxiv_papers = fetch_arxiv_papers()
    print(f"        获取 {len(arxiv_papers)} 篇论文")
    
    print("  [2/4] 拉取 Hugging Face Daily Papers...")
    hf_papers = fetch_hf_daily_papers()
    print(f"        获取 {len(hf_papers)} 篇论文")
    
    print("  [3/4] 拉取 GitHub Trending...")
    github_projects = fetch_github_trending()
    print(f"        获取 {len(github_projects)} 个项目")
    
    # 2. 关键词过滤
    arxiv_papers = keyword_filter(arxiv_papers)
    print(f"        关键词过滤后: arXiv {len(arxiv_papers)} 篇")
    
    # 3. 生成报告
    print("  [4/4] 调用 LLM 生成快报...")
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    if has_api_key and not os.environ.get("OPENAI_API_KEY", "").startswith("你"):
        report_body = generate_report_with_llm(
            arxiv_papers, hf_papers, github_projects, date_formatted
        )
    else:
        if not has_api_key:
            print("        [INFO] 未设置 OPENAI_API_KEY，使用降级模式")
        else:
            print("        [INFO] OPENAI_API_KEY 未配置有效值，使用降级模式")
        report_body = generate_fallback_report(arxiv_papers, hf_papers, github_projects)
    
    # 4. 组装 Markdown
    order_num = int(report_date.strftime("%Y%m%d"))
    
    markdown = f"""---
title: "每日快报 {date_formatted}"
order: {order_num}
tags: [每日快报, 机器人, 强化学习, 深度学习, 金融量化]
category: 每日快报
date: {date_formatted}
---

# 🗞️ 每日快报 | {date_formatted}（{weekday}）

> 自动搜罗 **机器人 · 深度学习 · 强化学习 · 机器人仿真 · 金融量化** 领域的今日重要动态。

{report_body}

---

*本快报由 GitHub Actions 自动生成。数据来源：arXiv、Hugging Face Daily Papers、GitHub Trending。*
"""

    # 5. 写入文件
    output_dir = Path(__file__).parent.parent / "每日快报"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{date_formatted}.md"
    output_file.write_text(markdown, encoding="utf-8")
    
    print(f"\n✅ 快报已生成: {output_file}")
    print(f"   大小: {len(markdown)} 字符")


if __name__ == "__main__":
    main()
