#!/usr/bin/env node
/**
 * 公式折叠讲解检查工具（check_formula_fold）
 *
 * 背景：写作规范要求每个独立公式（$$...$$）后面必须紧跟"逐符号拆解 + 数值代入"
 * 的详细讲解，但这部分内容篇幅长，约定用 VitePress 内置的 `::: details` 容器
 * 折叠起来，默认收起、按需展开。折叠区域外面必须留一句"这个公式在做什么"的
 * 可见摘要，保证读者不展开也能看懂公式大意。
 *
 * 本脚本扫描 Markdown 文章，检查每个独立公式是否满足：
 *   1. 公式后面（在下一个公式/标题之前）有可见的一句话摘要
 *      （形如 **这个公式在做什么** / **一句话** / > 引用块）
 *   2. 公式后面有一个 `::: details` 折叠块承载逐符号拆解和数值代入
 *
 * 用法：
 *   node scripts/check_formula_fold.mjs <文件或目录路径...>
 *   node scripts/check_formula_fold.mjs                      # 默认扫描全部内容目录
 *   node scripts/check_formula_fold.mjs --fix-hint            # 额外打印修复建议模板
 *
 * 退出码：发现问题时返回 1，方便接入 hook / CI。
 */

import fs from 'node:fs'
import path from 'node:path'

const CONTENT_DIRS = ['前置知识', '论文综述', '工程实践', '工程项目', '系列']
const REPO_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')

const VISIBLE_SUMMARY_MARKERS = [
  /\*\*这个公式在做什么\*\*/,
  /\*\*一句话\*\*/,
  /^>\s*\*\*一句话/m,
  /\*\*为什么需要这个公式\*\*/,
]

function collectMarkdownFiles(inputPaths) {
  const files = []
  const targets = inputPaths.length > 0
    ? inputPaths.map(p => path.resolve(p))
    : CONTENT_DIRS.map(d => path.join(REPO_ROOT, d))

  function walk(p) {
    if (!fs.existsSync(p)) return
    const stat = fs.statSync(p)
    if (stat.isDirectory()) {
      for (const entry of fs.readdirSync(p)) {
        walk(path.join(p, entry))
      }
    } else if (stat.isFile() && p.endsWith('.md')) {
      files.push(p)
    }
  }
  for (const t of targets) walk(t)
  return files
}

/**
 * 扫描单个文件内容，返回问题列表
 * 每个问题: { line, formulaPreview, missingSummary, missingFold }
 */
function checkFile(content) {
  const lines = content.split('\n')
  const issues = []

  let i = 0
  let insideDetailsFold = false

  while (i < lines.length) {
    const line = lines[i]

    // 追踪是否身处 ::: details ... ::: 折叠块内部
    // （折叠块内部出现的 $$ 公式，比如数值代入公式，不需要再单独套一层折叠）
    if (/^:::\s*details/.test(line.trim())) {
      insideDetailsFold = true
      i++
      continue
    }
    if (insideDetailsFold && line.trim() === ':::') {
      insideDetailsFold = false
      i++
      continue
    }

    // 命中一个独立公式的起始 "$$"
    if (line.trim() === '$$' && !insideDetailsFold) {
      const startLine = i
      let j = i + 1
      // 找到公式结束的 "$$"
      while (j < lines.length && lines[j].trim() !== '$$') j++
      const formulaContent = lines.slice(startLine + 1, j).join(' ').trim().slice(0, 60)
      const closeLine = j

      // 在公式结束之后，扫描到下一个 "$$" 起始或下一个标题（## / ###）为止，
      // 检查这段窗口内是否有可见摘要标记 + details 折叠块
      let k = closeLine + 1
      let hasSummary = false
      let hasFold = false
      while (k < lines.length) {
        const l = lines[k]
        if (l.trim() === '$$') break // 下一个公式开始，停止窗口
        if (/^#{2,4}\s+/.test(l)) break // 下一个标题，停止窗口
        if (VISIBLE_SUMMARY_MARKERS.some(re => re.test(l))) hasSummary = true
        if (/^:::\s*details/.test(l.trim())) hasFold = true
        k++
      }

      if (!hasSummary || !hasFold) {
        issues.push({
          line: startLine + 1, // 1-indexed
          formulaPreview: formulaContent || '(空公式)',
          missingSummary: !hasSummary,
          missingFold: !hasFold,
        })
      }

      i = closeLine + 1
      continue
    }

    i++
  }

  return issues
}

function main() {
  const args = process.argv.slice(2).filter(a => a !== '--fix-hint')
  const showFixHint = process.argv.includes('--fix-hint')
  const files = collectMarkdownFiles(args)

  let totalIssues = 0
  for (const file of files) {
    const content = fs.readFileSync(file, 'utf-8')
    const issues = checkFile(content)
    if (issues.length === 0) continue

    const relPath = path.relative(REPO_ROOT, file)
    for (const issue of issues) {
      totalIssues++
      const problems = []
      if (issue.missingSummary) problems.push('缺可见的一句话摘要（**这个公式在做什么**/**一句话**）')
      if (issue.missingFold) problems.push('缺 ::: details 折叠块（逐符号拆解+数值代入）')
      console.log(`${relPath}:${issue.line}  公式「${issue.formulaPreview}...」  → ${problems.join('；')}`)
    }
  }

  if (totalIssues === 0) {
    console.log(`✅ 检查了 ${files.length} 个文件，所有公式都有可见摘要 + 折叠详解。`)
    process.exit(0)
  }

  console.log(`\n共发现 ${totalIssues} 处公式缺少完整讲解（可见摘要 + 折叠详解），涉及 ${files.length} 个被扫描文件。`)

  if (showFixHint) {
    console.log(`
修复模板：

$$
你的公式
$$

**这个公式在做什么**：1-2 句话，说明这个公式的计算目标是什么（这部分永远可见，不折叠）。

::: details 📐 逐符号拆解 + 数值代入（点击展开）
**逐符号拆解**：

| 符号 | 含义 | 具体是什么 |
|------|------|-----------|
| ... | ... | ... |

**数值代入**：完整算一遍，展示所有中间变量。

**为什么是这个形式**：设计动机（1-3 句话）。
:::
`)
  }

  process.exit(1)
}

main()
