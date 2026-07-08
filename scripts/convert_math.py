#!/usr/bin/env python3
"""
Convert ```text blocks that contain math formulas to proper LaTeX $$ blocks.
Keeps non-math text blocks (tables, diagrams, pseudocode, etc.) unchanged.
"""

import re
import os
import glob

def is_math_block(content: str) -> bool:
    """
    Heuristic: a text block is "math" if it contains math-like patterns
    and does NOT look like a table, diagram, or long prose.
    """
    lines = content.strip().split('\n')
    
    # If it's a table (contains ─ or | aligned), keep as code
    if any('───' in l or '─────' in l for l in lines):
        return False
    # If it looks like a structured table with | separators in most lines
    pipe_lines = sum(1 for l in lines if l.strip().startswith('|') or '  |  ' in l)
    if pipe_lines > len(lines) * 0.5 and len(lines) > 3:
        return False
    
    # If it looks like a flowchart/diagram
    if any('→' in l and ('↓' in l or '↑' in l) for l in lines):
        return False
    if any(l.strip().startswith('┌') or l.strip().startswith('└') or l.strip().startswith('│') for l in lines):
        return False
    
    # If it's a step-by-step process with lots of prose (more than 5 lines of plain text)
    # Check for math indicators
    math_indicators = [
        r'[=≈≥≤∝∈∏∑∫]',  # math symbols
        r'\\[a-zA-Z]',     # latex commands
        r'\^{',            # superscripts
        r'_{',             # subscripts  
        r'‖',             # norm
        r'π_θ|π_',        # policy notation
        r'∇_θ|∇_',        # gradient notation
        r'log\s*π|log\s*p|log\s*N',  # log probability
        r'N\(.*[,;].*\)',  # Normal distribution
        r'E_\{|E\[',      # Expectation
        r'exp\(',          # exponential
        r'σ[²_]|μ[_θ]',   # sigma/mu notation
        r'‖.*‖',          # norm notation
        r'×',              # multiplication
        r'√',              # square root
        r'Σᵢ|∏ᵢ|Σ_',     # summation/product
        r'←.*←|→.*→',     # multiple arrows (assignment chains)
    ]
    
    math_score = 0
    for line in lines:
        for pattern in math_indicators:
            if re.search(pattern, line):
                math_score += 1
                break
    
    # If more than 60% of lines have math, treat as math
    if len(lines) > 0 and math_score / len(lines) > 0.5:
        return True
    
    # Short blocks (1-3 lines) with any math indicator -> math
    if len(lines) <= 3 and math_score > 0:
        return True
    
    # If block has "假设", "输入:", "步骤", numbered lists with Chinese -> not math
    chinese_prose_indicators = ['假设', '输入:', '输出:', '步骤', '比如', '例子', 
                                '场景:', '问题:', '观察:', '结论:', '原因:',
                                '优势:', '局限:', '为什么', '类比:',
                                '直觉', '解释:', '注意:', '现象:',
                                '训练时:', '测试时:', '实验', '结果:',
                                '第一步', '第二步', '第三步',
                                '问题 1:', '问题 2:', '问题 3:']
    prose_lines = sum(1 for l in lines if any(ind in l for ind in chinese_prose_indicators))
    if prose_lines > len(lines) * 0.3 and len(lines) > 3:
        return False
    
    # Blocks that look like "explanation with arrows" (← comments)
    arrow_comment_lines = sum(1 for l in lines if '←' in l and len(l) > 30)
    if arrow_comment_lines > len(lines) * 0.4 and len(lines) > 4:
        return False
    
    return math_score >= 2


def convert_math_block(content: str) -> str:
    """Convert a math text block content to LaTeX."""
    lines = content.strip().split('\n')
    
    # For single-line or very short pure formulas, use $$ display math
    # For multi-line with explanations mixed in, we'll use a mix
    
    result_lines = []
    for line in lines:
        # Convert common notation to LaTeX
        converted = line
        # Don't convert lines that are purely comments/explanations
        result_lines.append(converted)
    
    return '\n'.join(result_lines)


def process_file(filepath: str) -> bool:
    """Process a single markdown file. Returns True if changes were made."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all ```text ... ``` blocks
    pattern = r'```text\n(.*?)```'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        return False
    
    # Process from end to start to preserve positions
    new_content = content
    changed = False
    for match in reversed(matches):
        block_content = match.group(1)
        if is_math_block(block_content):
            # Replace with $$ math block
            # Clean up the content for display as math
            replacement = f'$$\n{block_content.rstrip()}\n$$'
            new_content = new_content[:match.start()] + replacement + new_content[match.end():]
            changed = True
    
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    return changed


def main():
    base_dir = '/home/wahaha/paper'
    md_files = glob.glob(os.path.join(base_dir, '**/*.md'), recursive=True)
    # Exclude node_modules, .vitepress, scripts
    md_files = [f for f in md_files if 'node_modules' not in f 
                and '.vitepress' not in f 
                and 'scripts' not in f]
    
    total_changed = 0
    for filepath in sorted(md_files):
        if process_file(filepath):
            rel = os.path.relpath(filepath, base_dir)
            print(f'  ✓ {rel}')
            total_changed += 1
        else:
            rel = os.path.relpath(filepath, base_dir)
            # print(f'  - {rel} (no changes)')
    
    print(f'\nDone: {total_changed} files modified out of {len(md_files)} total.')


if __name__ == '__main__':
    main()
