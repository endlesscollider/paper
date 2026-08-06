import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

export interface DailyHighlight {
  title: string
  url?: string
}

export interface DailySectionCount {
  name: string
  emoji: string
  count: number
}

export interface DailyArticleMeta {
  title: string
  link: string
  order: number
  date: string
  weekday: string
  sections: DailySectionCount[]
  highlights: DailyHighlight[]
  totalItems: number
  isEmpty: boolean
}

const WEEKDAYS: Record<string, string> = {
  '0': '周日', '1': '周一', '2': '周二', '3': '周三',
  '4': '周四', '5': '周五', '6': '周六',
}

function getWeekday(dateStr: string): string {
  const d = new Date(dateStr + 'T00:00:00+08:00')
  return WEEKDAYS[String(d.getDay())] || ''
}

/**
 * 从正文中提取各板块条目数
 */
function extractSections(content: string): DailySectionCount[] {
  const sections: DailySectionCount[] = []

  const sectionDefs = [
    { pattern: /##\s*🤖\s*机器人/, name: '机器人', emoji: '🤖' },
    { pattern: /##\s*🧠\s*深度学习/, name: '深度学习', emoji: '🧠' },
    { pattern: /##\s*🎮\s*强化学习/, name: '强化学习', emoji: '🎮' },
    { pattern: /##\s*🏭\s*机器人仿真/, name: '仿真', emoji: '🏭' },
    { pattern: /##\s*💰\s*金融量化/, name: '量化', emoji: '💰' },
    { pattern: /##\s*💻\s*工程项目/, name: '工程项目', emoji: '💻' },
  ]

  const lines = content.split('\n')

  for (const def of sectionDefs) {
    // Find section start
    const startIdx = lines.findIndex(l => def.pattern.test(l))
    if (startIdx === -1) continue

    // Count items (bold links or ### headings with links) until next ## heading
    let count = 0
    for (let i = startIdx + 1; i < lines.length; i++) {
      const line = lines[i]
      if (/^##\s/.test(line) && !(/^###/.test(line))) break
      // Count: "### N. [Title]" or "**[Title]" or "- **[Title]"
      if (/^###\s*\d+\.\s*\[/.test(line) || /^\*\*\[/.test(line) || /^-\s*\*\*\[/.test(line) || /^###\s*\[/.test(line)) {
        count++
      }
    }
    if (count > 0) {
      sections.push({ name: def.name, emoji: def.emoji, count })
    }
  }

  return sections
}

/**
 * 从 "🔥 今日亮点" 部分提取亮点标题
 */
function extractHighlights(content: string): DailyHighlight[] {
  const highlights: DailyHighlight[] = []
  const lines = content.split('\n')

  // Find "今日亮点" section
  const startIdx = lines.findIndex(l => /🔥\s*今日亮点/.test(l))
  if (startIdx === -1) return highlights

  for (let i = startIdx + 1; i < lines.length; i++) {
    const line = lines[i]
    // Stop at next ## section or ---
    if (/^##\s/.test(line) || /^---/.test(line)) break

    // Match patterns:
    // 1. "**N. [Title](url)..." or "**[Title](url)..."
    // 2. "N. **[Title](url)..."
    const linkMatch = line.match(/\*\*\[([^\]]+)\]\(([^)]+)\)/)
    if (linkMatch) {
      // Clean up the title - remove trailing ** and other markup
      let title = linkMatch[1]
      // Try to get the dash description after the link
      const descMatch = line.match(/\]\([^)]+\)\*?\*?\s*[—–-]\s*(.+)/)
      if (descMatch) {
        // Get first sentence of description (truncate if too long)
        let desc = descMatch[1].replace(/\*\*/g, '').trim()
        if (desc.length > 60) {
          desc = desc.substring(0, 57) + '…'
        }
        title = title + ' — ' + desc
      }
      highlights.push({ title, url: linkMatch[2] })
      continue
    }

    // Match: "**1. Title: description**" (without link)
    const boldMatch = line.match(/^\*\*\d+\.\s*(.+?)\*\*/)
    if (boldMatch) {
      let title = boldMatch[1].replace(/[：:]\s*$/, '').trim()
      if (title.length > 80) {
        title = title.substring(0, 77) + '…'
      }
      highlights.push({ title })
    }
  }

  return highlights
}

function checkEmpty(content: string): boolean {
  return /今日.*(?:暂无|为空|无条目|无法生成)/.test(content) ||
    /原始数据为空/.test(content)
}

declare const data: DailyArticleMeta[]
export { data }

export default {
  load(): DailyArticleMeta[] {
    const dailyDir = path.resolve(__dirname, '../../每日快报')
    if (!fs.existsSync(dailyDir)) return []

    const files = fs.readdirSync(dailyDir)
      .filter(f => f.endsWith('.md') && f !== 'index.md')
      .sort()
      .reverse()

    return files.map(file => {
      const raw = fs.readFileSync(path.join(dailyDir, file), 'utf-8')
      const { data: fm, content } = matter(raw)
      const slug = file.replace(/\.md$/, '')
      const dateStr = fm.date || slug
      const isEmpty = checkEmpty(content)
      const sections = isEmpty ? [] : extractSections(content)
      const highlights = isEmpty ? [] : extractHighlights(content)
      const totalItems = sections.reduce((sum, s) => sum + s.count, 0)

      return {
        title: fm.title || `每日快报 ${slug}`,
        link: `/每日快报/${slug}`,
        order: fm.order ?? (parseInt(slug.replace(/-/g, ''), 10) || 0),
        date: dateStr,
        weekday: getWeekday(dateStr),
        sections,
        highlights,
        totalItems,
        isEmpty,
      } as DailyArticleMeta
    })
  }
}
