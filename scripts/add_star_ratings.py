#!/usr/bin/env python3
"""
为所有论文文章 frontmatter 中添加 star 字段（1-5星）。
评分依据：
- 5星：顶会最佳论文、影响力极大（如 Attention Is All You Need 级别）
- 4星：顶会论文（NeurIPS, ICML, ICLR, CoRL, RSS, ICRA Best Paper）+ 顶级机构（Google, DeepMind, OpenAI, Meta FAIR, Stanford, MIT, CMU, Berkeley, Princeton）
- 3星：普通顶会论文、优质 arXiv 预印本、知名团队
- 2星：一般 arXiv 预印本、工作坊论文
- 1星：尚未发表/早期版本

综述类文章统一 4 星，工程实践/项目类统一 3 星。
精读论文根据发表会议和机构打分。
"""

import os
import re

# 顶级会议关键词
TOP_VENUES = ['NeurIPS', 'ICML', 'ICLR', 'CoRL', 'RSS', 'CVPR', 'ICCV', 'ECCV', 'AAAI', 'IJCAI',
              'Nature', 'Science', 'T-RO', 'RA-L', 'ICRA']

# 顶级机构关键词
TOP_INSTITUTIONS = ['Google', 'DeepMind', 'OpenAI', 'Meta', 'FAIR', 'Stanford', 'MIT', 'CMU',
                    'Berkeley', 'Princeton', 'Microsoft', 'NVIDIA', 'Toyota Research',
                    'Physical Intelligence', 'Pieter Abbeel', 'Sergey Levine', 'Chelsea Finn']

def rate_article(filepath):
    """读取文章内容，返回星级评分"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析 frontmatter
    fm_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not fm_match:
        return 3

    fm = fm_match.group(1)

    # 如果已有 star 字段，跳过
    if 'star:' in fm:
        return None

    # 综述统一 4 星
    if 'category: 综述' in fm:
        return 4

    # 工程类统一 3 星
    if 'category: 工程实践' in fm or 'category: 工程项目' in fm:
        return 3

    # 精读论文：根据内容评分
    score = 3  # 基础分

    # 检查顶级会议
    for venue in TOP_VENUES:
        if venue.lower() in content.lower():
            score = max(score, 4)
            break

    # 检查顶级机构
    institution_count = 0
    for inst in TOP_INSTITUTIONS:
        if inst.lower() in content.lower():
            institution_count += 1
    if institution_count >= 2:
        score = max(score, 4)

    # 特别知名的论文加到5星
    famous_papers = ['Pi0', 'RT-2', 'RT2', 'OpenVLA', 'Octo', 'GR00T', 'OpenX', 'DROID']
    for paper in famous_papers:
        if paper.lower() in os.path.basename(filepath).lower():
            score = max(score, 5)
            break

    return score


def add_star_to_file(filepath, star):
    """在 frontmatter 中添加 star 字段"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 已有 star 字段则跳过
    if re.search(r'^star:', content, re.MULTILINE):
        return False

    # 在 --- 结束前插入 star 字段
    content = re.sub(
        r'^(---\n.*?)(^---)',
        lambda m: m.group(1) + f'star: {star}\n' + m.group(2),
        content,
        count=1,
        flags=re.DOTALL | re.MULTILINE
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return True


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scan_dirs = ['论文综述', '工程实践', '工程项目']

    total = 0
    updated = 0

    for d in scan_dirs:
        full_dir = os.path.join(base_dir, d)
        if not os.path.isdir(full_dir):
            continue
        for f in sorted(os.listdir(full_dir)):
            if not f.endswith('.md') or f == 'index.md':
                continue
            filepath = os.path.join(full_dir, f)
            star = rate_article(filepath)
            if star is None:
                print(f"  SKIP (already has star): {f}")
                continue
            total += 1
            if add_star_to_file(filepath, star):
                updated += 1
                print(f"  ★{star} -> {f}")

    print(f"\nDone: {updated}/{total} files updated")


if __name__ == '__main__':
    main()
