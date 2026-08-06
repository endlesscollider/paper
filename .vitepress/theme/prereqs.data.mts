import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

export interface PrereqMeta {
  title: string
  link: string
  order: number
  tags: string[]
  category: string
}

function scanPrereqs(): PrereqMeta[] {
  const fullDir = path.resolve(__dirname, '../../前置知识')
  if (!fs.existsSync(fullDir)) return []

  const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md')

  return files.map(file => {
    const content = fs.readFileSync(path.join(fullDir, file), 'utf-8')
    const { data } = matter(content)
    if (data.hidden) return null
    const slug = file.replace(/\.md$/, '')
    return {
      title: data.title || slug.replace(/^\d+[a-z]?_前置知识_/, '').replace(/_/g, ' '),
      link: `/前置知识/${slug}`,
      order: data.order ?? 999,
      tags: data.tags ?? [],
      category: data.category ?? '前置知识',
    } as PrereqMeta
  }).filter(Boolean) as PrereqMeta[]
}

declare const data: PrereqMeta[]
export { data }

export default {
  load(): PrereqMeta[] {
    return scanPrereqs().sort((a, b) => a.order - b.order || a.link.localeCompare(b.link))
  }
}
