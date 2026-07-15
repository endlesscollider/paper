import fs from 'node:fs'
import path from 'node:path'
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
  series?: SeriesMeta
}

// 只扫描论文和工程相关目录，前置知识和教程不混入
const SCAN_DIRS = [
  { dir: '论文综述', base: '/论文综述' },
  { dir: '工程实践', base: '/工程实践' },
  { dir: '工程项目', base: '/工程项目' },
  { dir: '系列', base: '/系列', indexOnly: true }, // 系列目录只取各系列的 index.md
]

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
      series: seriesMeta,
    } as ArticleMeta
  }).filter(Boolean) as ArticleMeta[]
}

function inferCategory(dir: string): string {
  if (dir.includes('工程实践')) return '工程实践'
  if (dir.includes('工程项目')) return '工程项目'
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
