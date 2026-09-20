#!/usr/bin/env node
/**
 * 写作前置调研工具（prep_writing）
 *
 * 背景：按 writing-rules.md 的要求，写任何新文章之前必须先确认：
 *   1. 这个主题/相关概念在项目里有没有写过（避免重复挖坑、方便加链接）
 *   2. 新文章该用哪个文件名前缀、哪个 order 数字（避免和现有文章冲突）
 *   3. 目标类型（前置知识/论文精读/综述/工程项目/系列）的命名和目录规范是什么
 *
 * 这些原本需要多次 grep_search + list_directory + read_file 才能拼出来的信息，
 * 这个脚本一次调用全部给出。
 *
 * 用法：
 *   node scripts/prep_writing.mjs <关键词1> [关键词2] [关键词3] ...
 *   node scripts/prep_writing.mjs                          # 不给关键词，只打印编号/命名规范
 *
 * 示例：
 *   node scripts/prep_writing.mjs real2sim GR00T-Mimic MimicGen DreamGen SimFoundry
 */

import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

const REPO_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')

const CONTENT_DIRS = [
  { dir: '前置知识', label: '前置知识' },
  { dir: '论文综述', label: '论文综述' },
  { dir: '工程项目', label: '工程项目' },
  { dir: '工程实践', label: '工程实践' },
  { dir: '硬件基础', label: '硬件基础' },
]

function walkMarkdownFiles(rootRelDir) {
  const full = path.join(REPO_ROOT, rootRelDir)
  const out = []
  if (!fs.existsSync(full)) return out
  function walk(p) {
    const stat = fs.statSync(p)
    if (stat.isDirectory()) {
      for (const entry of fs.readdirSync(p)) {
        if (entry === 'node_modules' || entry.startsWith('.')) continue
        walk(path.join(p, entry))
      }
    } else if (stat.isFile() && p.endsWith('.md')) {
      out.push(p)
    }
  }
  walk(full)
  return out
}

// ---------- 1. 关键词覆盖检查 ----------

// frontmatter 解析容错：单个文件 YAML 格式有问题（比如标题里嵌套了引号）不该
// 让整个脚本崩溃，跳过该文件的 frontmatter 字段、仍然保留关键词命中信息。
function safeMatter(content) {
  try {
    return matter(content)
  } catch {
    return { data: {} }
  }
}

function searchKeyword(keyword) {
  const results = []
  const kwLower = keyword.toLowerCase()

  for (const { dir, label } of CONTENT_DIRS) {
    for (const file of walkMarkdownFiles(dir)) {
      const content = fs.readFileSync(file, 'utf-8')
      const contentLower = content.toLowerCase()
      const occurrences = contentLower.split(kwLower).length - 1
      if (occurrences === 0) continue

      const { data } = safeMatter(content)
      const relPath = path.relative(REPO_ROOT, file)
      results.push({
        file: relPath,
        label,
        title: data.title || path.basename(file, '.md'),
        order: data.order ?? null,
        category: data.category ?? null,
        occurrences,
      })
    }
  }

  // 系列文章（章节文件也扫，因为概念可能藏在具体章节里）
  const seriesRoot = path.join(REPO_ROOT, '系列')
  if (fs.existsSync(seriesRoot)) {
    for (const sub of fs.readdirSync(seriesRoot)) {
      const subDir = path.join(seriesRoot, sub)
      if (!fs.statSync(subDir).isDirectory()) continue
      for (const file of fs.readdirSync(subDir)) {
        if (!file.endsWith('.md')) continue
        const full = path.join(subDir, file)
        const content = fs.readFileSync(full, 'utf-8')
        const contentLower = content.toLowerCase()
        const occurrences = contentLower.split(kwLower).length - 1
        if (occurrences === 0) continue
        const { data } = safeMatter(content)
        results.push({
          file: path.relative(REPO_ROOT, full),
          label: '系列',
          title: data.title || `${sub}/${file}`,
          order: data.order ?? null,
          category: data.category ?? '系列',
          occurrences,
        })
      }
    }
  }

  results.sort((a, b) => b.occurrences - a.occurrences)
  return results
}

// ---------- 2. 编号 / 命名规范扫描 ----------

function scanPrereqNumbering() {
  const files = fs.readdirSync(path.join(REPO_ROOT, '前置知识')).filter(f => f.endsWith('.md') && f !== 'index.md')
  let maxOrder = 0
  const prefixGroups = {} // "000" -> ["a","b",...]
  for (const f of files) {
    const m = f.match(/^(\d{3})([a-z]\d?)_/)
    if (m) {
      const [, num, letter] = m
      prefixGroups[num] = prefixGroups[num] || []
      prefixGroups[num].push(letter)
    }
    const content = fs.readFileSync(path.join(REPO_ROOT, '前置知识', f), 'utf-8')
    const { data } = safeMatter(content)
    if (typeof data.order === 'number' && data.order > maxOrder && data.order < 200) maxOrder = data.order
  }
  const maxPrefixNum = Object.keys(prefixGroups).sort().pop()
  const lettersInMaxPrefix = (prefixGroups[maxPrefixNum] || []).sort()
  const lastLetter = lettersInMaxPrefix[lettersInMaxPrefix.length - 1]
  return { maxOrder, maxPrefixNum, lastLetter, totalFiles: files.length }
}

function scanPaperNumbering() {
  const files = fs.readdirSync(path.join(REPO_ROOT, '论文综述')).filter(f => f.endsWith('.md') && f !== 'index.md')
  let maxReadingNum = 0   // 三位数字精读 001_xxx.md
  let maxSurveyNum = 0    // S01_xxx.md 综述
  let maxOrder = 0
  for (const f of files) {
    const mNum = f.match(/^(\d{3})_/)
    if (mNum) maxReadingNum = Math.max(maxReadingNum, parseInt(mNum[1], 10))
    const mS = f.match(/^S(\d{2})_/)
    if (mS) maxSurveyNum = Math.max(maxSurveyNum, parseInt(mS[1], 10))
    const content = fs.readFileSync(path.join(REPO_ROOT, '论文综述', f), 'utf-8')
    const { data } = safeMatter(content)
    if (typeof data.order === 'number' && data.order > maxOrder) maxOrder = data.order
  }
  return { maxReadingNum, maxSurveyNum, maxOrder, totalFiles: files.length }
}

function scanEngProjectNumbering() {
  const dir = path.join(REPO_ROOT, '工程项目')
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.md') && f !== 'index.md')
  let maxOrder = 0
  const titles = []
  for (const f of files) {
    const content = fs.readFileSync(path.join(dir, f), 'utf-8')
    const { data } = safeMatter(content)
    if (typeof data.order === 'number' && data.order > maxOrder) maxOrder = data.order
    titles.push({ file: f, title: data.title, order: data.order })
  }
  return { maxOrder, totalFiles: files.length, titles }
}

function scanSeriesNumbering() {
  const seriesRoot = path.join(REPO_ROOT, '系列')
  const entries = []
  for (const sub of fs.readdirSync(seriesRoot)) {
    const idxFile = path.join(seriesRoot, sub, 'index.md')
    if (!fs.existsSync(idxFile)) continue
    const content = fs.readFileSync(idxFile, 'utf-8')
    const { data } = safeMatter(content)
    entries.push({
      dir: sub,
      order: data.order ?? null,
      totalChapters: data.series?.totalChapters ?? null,
      title: data.title ?? sub,
    })
  }
  entries.sort((a, b) => (a.order ?? 9999) - (b.order ?? 9999))
  return entries
}

// ---------- 输出 ----------

function printKeywordReport(keywords) {
  console.log('='.repeat(70))
  console.log('【关键词覆盖检查】—— 判断这些概念项目里是否已经写过')
  console.log('='.repeat(70))
  for (const kw of keywords) {
    const results = searchKeyword(kw)
    console.log(`\n关键词「${kw}」：`)
    if (results.length === 0) {
      console.log('  （未找到任何提及，项目里还没有覆盖这个概念/主题）')
      continue
    }
    for (const r of results.slice(0, 12)) {
      console.log(`  [${r.label}] ${r.title}  (order=${r.order}, 命中${r.occurrences}次)  → ${r.file}`)
    }
    if (results.length > 12) console.log(`  ...还有 ${results.length - 12} 个文件命中，未全部列出`)
  }
}

function printNumberingReport() {
  console.log('\n' + '='.repeat(70))
  console.log('【编号与命名规范】—— 新文章该用什么文件名 / order')
  console.log('='.repeat(70))

  const prereq = scanPrereqNumbering()
  console.log(`\n前置知识（共 ${prereq.totalFiles} 篇）：`)
  console.log(`  文件名格式：<三位数字><字母>_前置知识_<主题名>.md，如 005m_前置知识_XXX.md`)
  console.log(`  当前最大数字前缀分组：${prereq.maxPrefixNum}，该分组最后一个字母：${prereq.lastLetter}`)
  console.log(`  → 新文章建议：沿用 ${prereq.maxPrefixNum} 分组下一个字母，或开新分组 ${String(Number(prereq.maxPrefixNum) + 1).padStart(3, '0')}a`)
  console.log(`  当前 order 最大值（<200 范围内）：${prereq.maxOrder} → 新文章建议 order: ${prereq.maxOrder + 1}`)
  console.log(`  category 取值示例：前置知识 / 强化学习 / 深度学习基础 / 机器人基础 / 数学基础`)

  const paper = scanPaperNumbering()
  console.log(`\n论文综述（共 ${paper.totalFiles} 篇）：`)
  console.log(`  论文精读文件名：<三位数字>_<英文简称>_<中文主题>.md，如 114_XXX_YYY.md`)
  console.log(`  当前最大精读编号：${String(paper.maxReadingNum).padStart(3, '0')} → 新精读建议编号：${String(paper.maxReadingNum + 1).padStart(3, '0')}`)
  console.log(`  综述文件名：S<两位数字>_<主题>.md`)
  console.log(`  当前最大综述编号：S${String(paper.maxSurveyNum).padStart(2, '0')} → 新综述建议编号：S${String(paper.maxSurveyNum + 1).padStart(2, '0')}`)
  console.log(`  当前 order 最大值：${paper.maxOrder} → 新精读文章 order 建议 > ${paper.maxOrder}（随意取不冲突的较大值即可，如 ${paper.maxOrder + 5}）`)
  console.log(`  精读 category: 精读；综述 category: 综述`)

  const eng = scanEngProjectNumbering()
  console.log(`\n工程项目（共 ${eng.totalFiles} 篇）：`)
  console.log(`  文件名格式：<英文项目名>_<中文简介>.md（无数字前缀）`)
  console.log(`  当前 order 最大值：${eng.maxOrder} → 新文章建议 order: ${eng.maxOrder + 1}`)

  const series = scanSeriesNumbering()
  console.log(`\n系列文章（共 ${series.length} 个系列）：`)
  console.log(`  目录结构：系列/<series_id>/index.md + 系列/<series_id>/01_xxx.md ...`)
  console.log(`  已使用的 order 区段：`)
  for (const s of series) {
    console.log(`    order=${s.order}  ${s.dir}  (${s.totalChapters} 章)  ${s.title}`)
  }
  console.log(`  → 新系列建议：避开以上 order，选一个未使用的整百或相邻空位（如 ${suggestFreeOrder(series.map(s => s.order))}）`)
}

function suggestFreeOrder(usedOrders) {
  const used = new Set(usedOrders.filter(Boolean))
  const hundreds = [100, 200, 300, 400, 500, 600, 700, 800, 900]
  for (const h of hundreds) {
    if (!used.has(h)) return h
  }
  // 300 区间常见，找一个空的三位数
  for (let n = 300; n < 400; n += 1) {
    if (!used.has(n)) return n
  }
  return Math.max(...usedOrders) + 10
}

function printStaticCheatsheet() {
  console.log('\n' + '='.repeat(70))
  console.log('【固定规范提示】（详见 .kiro/steering/writing-rules.md 第 8、10 节）')
  console.log('='.repeat(70))
  console.log(`
- 判断文章类型：单概念讲透(3000+字) → 前置知识；单论文精读(5000+字) → 论文综述(数字前缀)；
  跨论文分类对比(8000+字) → 论文综述(S前缀)；开源项目/工具链解析 → 工程项目；
  代码级工程实现细节 → 工程实践；总字数2万+且需要分步 → 系列文章。
- 写前必须做（本脚本已自动完成第1步，2、3步仍需人工过一遍）：
  1. 用本脚本查关键词覆盖，找到可复用/需链接的已有文章
  2. 前置知识遇到的新概念，检查是否需要新建前置知识文章（不能空白留白）
  3. 综述必须按分类/演进组织，不能逐篇罗列（见 writing-rules.md 3.5 节）
- 系列 index.md frontmatter 必须包含 series.id / totalChapters / dir，新增章节要同步更新 totalChapters。
- 写完包含公式的文章后运行：node scripts/check_formula_fold.mjs <文件路径> 自查折叠格式。
- 涉及分布/激活函数/loss形状/调度曲线等，必须调用 scripts/plot_function.py 画图（见 visualization-rules.md）。
`)
}

function main() {
  const keywords = process.argv.slice(2)
  if (keywords.length > 0) {
    printKeywordReport(keywords)
  } else {
    console.log('（未提供关键词，跳过覆盖检查。用法：node scripts/prep_writing.mjs 关键词1 关键词2 ...）')
  }
  printNumberingReport()
  printStaticCheatsheet()
}

main()
