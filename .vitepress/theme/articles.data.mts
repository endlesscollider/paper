import fs from 'node:fs'
import path from 'node:path'
import { execSync } from 'node:child_process'
import matter from 'gray-matter'

export interface SeriesMeta {
  id: string
  totalChapters: number
  dir: string
}

export interface ArticleMeta {
  title: string
  link: string
  order: number
  tags: string[]
  category: string
  star: number
  date: string
  lastUpdated: number  // unix timestamp for sorting "最新收录"
  series?: SeriesMeta
}

// 扫描全部内容目录，让 /tags 全局搜索和各目录首页都能拿到完整数据
const SCAN_DIRS = [
  { dir: '论文综述', base: '/论文综述' },
  { dir: '工程实践', base: '/工程实践' },
  { dir: '工程项目', base: '/工程项目' },
  { dir: '前置知识', base: '/前置知识' },
  { dir: '硬件基础', base: '/硬件基础' },
  { dir: '数据集', base: '/数据集' },
  { dir: '系列', base: '/系列', indexOnly: true }, // 系列目录只取各系列的 index.md
]

const _repoRoot = path.resolve(__dirname, '../../')
let _gitTimestampCache: Map<string, number> | null = null

/**
 * 一次性批量读取所有文件的 git 最后修改时间，缓存到内存。
 * 原先每个文件单独调 git log，470 个文件就是 470 次子进程，
 * 导致 dev server 启动耗时数分钟。现在只 spawn 一次 git。
 */
function _buildGitCache(): Map<string, number> {
  if (_gitTimestampCache) return _gitTimestampCache
  const cache = new Map<string, number>()
  try {
    const out = execSync(
      'git -c core.quotePath=false log --no-merges --pretty=format:"COMMIT %ct" --name-only',
      { cwd: _repoRoot, encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 }
    )
    let ts = 0
    for (const line of out.split('\n')) {
      const t = line.trim()
      if (!t) continue
      const m = t.match(/^COMMIT (\d+)$/)
      if (m) { ts = parseInt(m[1], 10) * 1000; continue }
      // git 输出的是仓库相对路径；第一次出现 = 最近一次 commit
      if (!cache.has(t)) cache.set(t, ts)
    }
  } catch {}
  _gitTimestampCache = cache
  return cache
}

/** 获取文件的最近更新时间戳（优先 git 批量缓存，fallback 到 fs mtime） */
function getLastUpdated(filePath: string): number {
  const rel = path.relative(_repoRoot, filePath)
  const ts = _buildGitCache().get(rel)
  if (ts) return ts
  try { return fs.statSync(filePath).mtimeMs } catch {}
  return 0
}

function scanDir(dir: string, base: string, indexOnly = false): ArticleMeta[] {
  const fullDir = path.resolve(__dirname, '../../', dir)
  if (!fs.existsSync(fullDir)) return []

  // 系列目录：只扫描各子文件夹中的 index.md
  if (indexOnly) {
    const subdirs = fs.readdirSync(fullDir).filter(f => {
      const full = path.join(fullDir, f)
      return fs.statSync(full).isDirectory()
    })

    return subdirs.map(sub => {
      const indexFile = path.join(fullDir, sub, 'index.md')
      if (!fs.existsSync(indexFile)) return null
      const content = fs.readFileSync(indexFile, 'utf-8')
      const { data } = matter(content)
      if (data.hidden) return null

      const seriesMeta: SeriesMeta | undefined = data.series
        ? { id: data.series.id, totalChapters: data.series.totalChapters, dir: `${base}/${sub}` }
        : undefined

      return {
        title: data.title || sub.replace(/_/g, ' '),
        link: `${base}/${sub}/`,
        order: data.order ?? 999,
        tags: data.tags ?? [],
        category: data.category ?? '系列',
        star: data.star ?? 3,
        date: data.date ?? '',
        lastUpdated: getLastUpdated(indexFile),
        series: seriesMeta,
      } as ArticleMeta
    }).filter(Boolean) as ArticleMeta[]
  }

  const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md')

  return files.map(file => {
    const content = fs.readFileSync(path.join(fullDir, file), 'utf-8')
    const { data } = matter(content)
    // 隐藏的文章不出现在卡片列表中（系列章节用）
    if (data.hidden) return null
    const slug = file.replace(/\.md$/, '')

    const seriesMeta: SeriesMeta | undefined = data.series
      ? { id: data.series.id, totalChapters: data.series.totalChapters, dir: data.series.dir }
      : undefined

    return {
      title: data.title || slug.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' '),
      link: `${base}/${slug}`,
      order: data.order ?? 999,
      tags: data.tags ?? [],
      category: data.category ?? inferCategory(dir),
      star: data.star ?? 3,
      date: data.date ?? '',
      lastUpdated: getLastUpdated(path.join(fullDir, file)),
      series: seriesMeta,
    } as ArticleMeta
  }).filter(Boolean) as ArticleMeta[]
}

function inferCategory(dir: string): string {
  if (dir.includes('工程实践')) return '工程实践'
  if (dir.includes('工程项目')) return '工程项目'
  if (dir.includes('前置知识')) return '前置知识'
  if (dir.includes('硬件基础')) return '硬件基础'
  return '综述'
}

declare const data: ArticleMeta[]
export { data }

export default {
  load(): ArticleMeta[] {
    const all: ArticleMeta[] = []
    for (const { dir, base, indexOnly } of SCAN_DIRS as Array<{ dir: string; base: string; indexOnly?: boolean }>) {
      all.push(...scanDir(dir, base, indexOnly))
    }
    return all.sort((a, b) => a.order - b.order || a.link.localeCompare(b.link))
  }
}
