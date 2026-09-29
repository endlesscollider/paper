#!/usr/bin/env node
/**
 * 检测「表格单元格内的行内数学含裸竖线」导致的渲染塌陷。
 *
 * 背景：Markdown 表格用 | 分隔单元格。若某个单元格里的行内数学 $...$ 内部
 * 含未转义的裸 |，渲染时该单元格会被切碎，MathJax 收到一段畸形公式，
 * 结果不是抛错，而是把公式所在单元格的内容整体丢弃（静默丢内容）。
 *
 * 判据不用模式匹配，而是「实际渲染 + 对比」：
 *   1. 渲染整篇文章，数出所有 <td> 里非空的单元格；
 *   2. 逐个把表格里的裸 | 替换成 \lvert / \rvert（不动 \|），再渲染；
 *   3. 若替换后 <td> 文本总量变多，说明原版丢了内容 → 该处是真隐患。
 *
 * 用法：
 *   node scripts/check_math_pipe_in_table.mjs                 # 扫全站
 *   node scripts/check_math_pipe_in_table.mjs <文件...>        # 只查指定文件
 */
import fs from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve(import.meta.dirname, '..')
const DIRS = ['前置知识', '论文综述', '工程实践', '工程项目', '系列']

function walk(dir, out = []) {
  if (!fs.existsSync(dir)) return out
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p, out)
    else if (e.name.endsWith('.md')) out.push(p)
  }
  return out
}

/** 表格行里的裸竖线：只处理行内数学 $...$ 内部；\| 视为安全。 */
function findBarePipes(line) {
  const spans = []
  const re = /\$([^$\n]+)\$/g
  let m
  while ((m = re.exec(line))) {
    for (let i = 0; i < m[1].length; i++) {
      if (m[1][i] !== '|') continue
      if (i > 0 && m[1][i - 1] === '\\') continue // \| 安全
      spans.push([m.index + 1 + i, m.index + 1 + i + 1])
    }
  }
  return spans
}

/** 核心修复：把数学里的裸 | 换成 \lvert / \rvert（按出现顺序成对分配）。 */
function neutralize(src) {
  const lines = src.split('\n')
  let count = 0
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i]
    if (!l.trim().startsWith('|')) continue
    const spans = findBarePipes(l)
    if (!spans.length) continue
    // 从后往前替换，避免偏移错乱
    let out = l
    for (let k = spans.length - 1; k >= 0; k--) {
      const [a, b] = spans[k]
      const isOpen = k % 2 === 0
      out = out.slice(0, a) + (isOpen ? '\\lvert' : '\\rvert') + out.slice(b)
      count++
    }
    lines[i] = out
  }
  return { text: lines.join('\n'), count }
}

function tdText(html) {
  const cells = [...html.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map((m) =>
    m[1].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim()
  )
  return { count: cells.length, len: cells.reduce((a, c) => a + c.length, 0) }
}

async function main() {
  const args = process.argv.slice(2)
  const files = args.length
    ? args.map((a) => path.resolve(a))
    : DIRS.flatMap((d) => walk(path.join(ROOT, d)))

  let md, mdPlain
  try {
    const MarkdownIt = (await import('markdown-it')).default
    const mj = await import('markdown-it-mathjax3')
    const mathjax3 = mj.default ?? mj
    md = new MarkdownIt({ html: true }).use(mathjax3)
    mdPlain = new MarkdownIt({ html: true })
  } catch (e) {
    console.error('需要 markdown-it 与 markdown-it-mathjax3，请在项目根目录运行。')
    console.error(String(e.message || e))
    process.exit(2)
  }

  const hits = []
  for (const f of files) {
    const src = fs.readFileSync(f, 'utf8')
    if (!/(?<!\\)\|\S*\$/.test(src) && !/\\mathcal\{[^}]*\}\(i\)/.test(src)) {
      // 快速预筛：表格行里出现 $...|...$ 才深查
    }
    const rows = src.split('\n').filter(
      (l) => l.trim().startsWith('|') && /\$[^$\n]*\|[^$\n]*\$/.test(l)
    )
    if (!rows.length) continue

    const before = tdText(md.render(src))
    const { text: fixed, count } = neutralize(src)
    if (!count) continue
    const after = tdText(md.render(fixed))

    if (after.len > before.len) {
      hits.push({ file: path.relative(ROOT, f), count, before: before.len, after: after.len })
    }
  }

  console.log(`扫描 ${files.length} 个文件`)
  if (!hits.length) {
    console.log('✅ 未发现「表格内裸竖线导致静默丢内容」的隐患')
    return
  }
  console.log(`\n❌ 发现 ${hits.length} 个文件存在隐患（替换后单元格文本量增加 = 原版丢了内容）：\n`)
  for (const h of hits) {
    console.log(`  ${h.file}`)
    console.log(`      裸竖线 ${h.count} 处；单元格文本 ${h.before} → ${h.after} 字符`)
  }
  console.log('\n修复方法：把数学里的裸 | 改为 \\lvert / \\rvert（见 .kiro/steering/writing-rules.md）')
}

main()
