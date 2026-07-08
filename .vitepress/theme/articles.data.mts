import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

export interface ArticleMeta {
  title: string
  link: string
  order: number
  tags: string[]
  category: string
  star: number
  date: string
}

// 只扫描论文和工程相关目录，前置知识和教程不混入
const SCAN_DIRS = [
  { dir: '论文综述', base: '/论文综述' },
  { dir: '工程实践', base: '/工程实践' },
  { dir: '工程项目', base: '/工程项目' },
]

function scanDir(dir: string, base: string): ArticleMeta[] {
  const fullDir = path.resolve(__dirname, '../../', dir)
  if (!fs.existsSync(fullDir)) return []

  const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md')

  return files.map(file => {
    const content = fs.readFileSync(path.join(fullDir, file), 'utf-8')
    const { data } = matter(content)
    const slug = file.replace(/\.md$/, '')

    return {
      title: data.title || slug.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' '),
      link: `${base}/${slug}`,
      order: data.order ?? 999,
      tags: data.tags ?? [],
      category: data.category ?? inferCategory(dir),
      star: data.star ?? 3,
      date: data.date ?? '',
    }
  }).sort((a, b) => a.order - b.order || a.link.localeCompare(b.link))
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
    for (const { dir, base } of SCAN_DIRS) {
      all.push(...scanDir(dir, base))
    }
    return all
  }
}
