#!/usr/bin/env python3
"""
根据文章中实际记录的发表会议、作者机构、论文知名度来更新星级评分。
解析每篇文章正文中的 > **发表**: 和 > **机构**: 字段。

评分标准：
  5星 - 领域里程碑：开创性工作/被大量后续引用/改变了研究方向
        (如 Attention Is All You Need, RT-2, Pi0, OpenVLA, DROID, OpenX)
  4星 - 顶会+顶级机构：NeurIPS/ICML/ICLR/CoRL/RSS/CVPR oral + 知名团队
  3星 - 优质工作：顶会/知名团队 arXiv 预印本，有实质贡献
  2星 - 一般工作：普通 arXiv 预印本，中等影响力
  1星 - 早期/弱工作

综述文章：
  S系列综述统一 4 星（我们自己写的知识梳理，质量高）

工程类：
  根据工具本身的影响力评分（LeRobot 4星，一般实践 3星）
"""

import os
import re

# === 评分知识库 ===

# 里程碑级别论文（5星）- 按文件名匹配
LANDMARK_PAPERS = {
    'Pi0': 5,
    'RT2': 5,
    'RT-2': 5,
    'OpenVLA': 5,
    'OpenX': 5,
    'DROID': 5,
    'GR00T': 5,
    'Octo': 4,  # 重要但不算里程碑
}

# 知名开源项目/工具（影响力加分）
NOTABLE_PROJECTS = {
    'LeRobot': 4,
    'HPT': 4,
    'CrossFormer': 4,
}

# 顶级会议（确认发表 = 高质量保证）
TIER1_VENUES = ['NeurIPS', 'ICML', 'ICLR', 'CoRL', 'RSS', 'CVPR', 'ICCV', 'Nature', 'Science']
TIER2_VENUES = ['ICRA', 'IROS', 'RA-L', 'T-RO', 'AAAI', 'ECCV', 'IJCAI', 'RAL']

# 顶级机构（多个 = 实力强）
TOP_INSTITUTIONS = [
    'Google', 'DeepMind', 'Google DeepMind',
    'OpenAI',
    'Physical Intelligence',
    'Meta', 'FAIR',
    'Stanford', 'MIT', 'CMU', 'Berkeley', 'Princeton',
    'Microsoft Research', 'NVIDIA',
    'Toyota Research',
    'Tsinghua', 'Peking University',
]

# 知名研究者（出现则加分）
NOTABLE_AUTHORS = [
    'Sergey Levine', 'Pieter Abbeel', 'Chelsea Finn',
    'Kaiming He', 'Yann LeCun', 'Fei-Fei Li',
    'Danny Driess', 'Kevin Black',  # Pi0 team
]


def extract_metadata(content):
    """从正文中提取发表信息和机构"""
    venue = ''
    institution = ''
    authors = ''

    # 匹配 > **发表**: xxx
    m = re.search(r'>\s*\*\*发表\*\*\s*[:：]\s*(.+)', content)
    if m:
        venue = m.group(1).strip()

    # 匹配 > **机构**: xxx
    m = re.search(r'>\s*\*\*机构\*\*\s*[:：]\s*(.+)', content)
    if m:
        institution = m.group(1).strip()

    # 匹配 > **作者**: xxx
    m = re.search(r'>\s*\*\*作者\*\*\s*[:：]\s*(.+)', content)
    if m:
        authors = m.group(1).strip()

    return venue, institution, authors


def compute_star(filename, content, category):
    """计算星级"""

    # 综述统一 4 星
    if category == '综述':
        return 4

    # 工程类默认 3 星，特殊项目除外
    if category in ('工程实践', '工程项目'):
        for name, star in NOTABLE_PROJECTS.items():
            if name.lower() in filename.lower() or name.lower() in content.lower()[:500]:
                return star
        return 3

    # === 精读论文评分 ===
    venue, institution, authors = extract_metadata(content)
    basename = os.path.splitext(os.path.basename(filename))[0]

    # 1. 检查是否里程碑论文
    for paper, star in LANDMARK_PAPERS.items():
        if paper.lower() in basename.lower():
            return star

    score = 3  # 基础分

    # 2. 发表会议加分
    venue_upper = venue.upper() if venue else ''
    for v in TIER1_VENUES:
        if v.upper() in venue_upper:
            score = max(score, 4)
            break

    if score < 4:
        for v in TIER2_VENUES:
            if v.upper() in venue_upper:
                score = max(score, 3)  # 确认 3 星
                break

    # 3. 机构加分
    inst_hits = 0
    for inst in TOP_INSTITUTIONS:
        if inst.lower() in (institution or '').lower():
            inst_hits += 1
    if inst_hits >= 2:
        score = max(score, 4)
    elif inst_hits >= 1 and score >= 3:
        # 单个顶级机构 + 顶会 = 保持4星；单个顶级机构无顶会 = 保持3星
        pass

    # 4. 知名作者加分
    for author in NOTABLE_AUTHORS:
        if author.lower() in (authors or '').lower():
            score = max(score, 4)
            break

    # 5. 仅 arXiv（未正式发表）且机构不突出，降到 3
    #    但如果发表字段同时包含顶会名称，说明已被接收，不降级
    #    如果有知名作者加分，也不降级
    has_top_venue = any(v.upper() in venue_upper for v in TIER1_VENUES + TIER2_VENUES)
    has_notable_author = any(a.lower() in (authors or '').lower() for a in NOTABLE_AUTHORS)

    if venue and 'arxiv' in venue.lower() and 'oral' not in venue.lower():
        if not has_top_venue and not has_notable_author:
            if inst_hits < 2 and score == 4:
                # 纯 arXiv 预印本，没有顶会接收、没有知名作者、机构不够突出
                elite = ['physical intelligence', 'google deepmind', 'openai', 'deepmind',
                         'berkeley', 'stanford', 'mit', 'cmu', 'princeton', 'nvidia', 'meta', 'fair']
                if not any(e in (institution or '').lower() for e in elite):
                    score = 3

    return score


def update_star_in_file(filepath, new_star):
    """更新文件中的 star 字段"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换已有的 star 字段
    if re.search(r'^star:\s*\d+', content, re.MULTILINE):
        new_content = re.sub(r'^star:\s*\d+', f'star: {new_star}', content, count=1, flags=re.MULTILINE)
    else:
        # 没有则在 --- 前插入
        new_content = re.sub(
            r'^(---\n.*?)(^---)',
            lambda m: m.group(1) + f'star: {new_star}\n' + m.group(2),
            content, count=1, flags=re.DOTALL | re.MULTILINE
        )

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


def get_category_from_frontmatter(content):
    m = re.search(r'^category:\s*(.+)', content, re.MULTILINE)
    return m.group(1).strip() if m else ''


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scan_dirs = ['论文综述', '工程实践', '工程项目']

    results = {'5': [], '4': [], '3': [], '2': [], '1': []}
    updated = 0

    for d in scan_dirs:
        full_dir = os.path.join(base_dir, d)
        if not os.path.isdir(full_dir):
            continue
        for f in sorted(os.listdir(full_dir)):
            if not f.endswith('.md') or f == 'index.md':
                continue
            filepath = os.path.join(full_dir, f)
            with open(filepath, 'r', encoding='utf-8') as fh:
                content = fh.read()

            category = get_category_from_frontmatter(content)
            new_star = compute_star(f, content, category)
            results[str(new_star)].append(f)

            if update_star_in_file(filepath, new_star):
                updated += 1

    print("=" * 60)
    print("星级分布:")
    print("=" * 60)
    for star in ['5', '4', '3', '2', '1']:
        items = results[star]
        if items:
            print(f"\n{'★' * int(star)}{'☆' * (5 - int(star))} ({len(items)} 篇)")
            for item in items:
                print(f"  {item}")

    print(f"\n{'=' * 60}")
    print(f"共更新 {updated} 个文件")


if __name__ == '__main__':
    main()
