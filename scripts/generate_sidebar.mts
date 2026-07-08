/**
 * 自动扫描文章目录，读取 frontmatter，生成 sidebar 配置。
 * 
 * 用法：在 config.mts 中 import 使用。
 * 
 * 文章 frontmatter 格式：
 * ---
 * title: 文章标题
 * order: 1
 * tags: [强化学习, 机器人]
 * category: 综述 | 精读 | 前置知识 | 工程实践
 * ---
 */

import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

export interface ArticleMeta {
  title: string
  file: string
  link: string
  order: number
  tags: string[]
  category: string
}

/**
 * 扫描指定目录下所有 .md 文件，提取 frontmatter 信息
 */
export function scanArticles(dir: string, baseLink: string): ArticleMeta[] {
  const fullDir = path.resolve(__dirname, '..', dir)
  if (!fs.existsSync(fullDir)) return []

  const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md')
  
  return files.map(file => {
    const content = fs.readFileSync(path.join(fullDir, file), 'utf-8')
    const { data } = matter(content)
    const slug = file.replace(/\.md$/, '')
    
    return {
      title: data.title || slug.replace(/^\d+_/, '').replace(/_/g, ' '),
      file,
      link: `${baseLink}/${slug}`,
      order: data.order ?? 999,
      tags: data.tags ?? inferTags(slug, data.category),
      category: data.category ?? inferCategory(dir, slug),
    }
  }).sort((a, b) => a.order - b.order || a.file.localeCompare(b.file))
}

/**
 * 根据文件名和目录推断 tags
 */
function inferTags(slug: string, category?: string): string[] {
  const tags: string[] = []
  const lower = slug.toLowerCase()
  
  if (lower.includes('rl') || lower.includes('强化学习') || lower.includes('ppo') || lower.includes('sac') || lower.includes('dqn')) {
    tags.push('强化学习')
  }
  if (lower.includes('diffus') || lower.includes('扩散') || lower.includes('ddpm') || lower.includes('flow')) {
    tags.push('扩散模型')
  }
  if (lower.includes('robot') || lower.includes('机器人') || lower.includes('bimanual') || lower.includes('双臂') || lower.includes('locomotion')) {
    tags.push('机器人')
  }
  if (lower.includes('模仿') || lower.includes('imitation') || lower.includes('克隆') || lower.includes('act') || lower.includes('vla')) {
    tags.push('模仿学习')
  }
  if (lower.includes('sim') || lower.includes('real') || lower.includes('迁移')) {
    tags.push('Sim-to-Real')
  }
  if (lower.includes('transformer') || lower.includes('attention')) {
    tags.push('Transformer')
  }
  if (lower.includes('工程') || lower.includes('实现') || lower.includes('implementation')) {
    tags.push('工程实践')
  }
  
  if (tags.length === 0) tags.push('深度学习')
  return tags
}

/**
 * 根据目录和文件名推断 category
 */
function inferCategory(dir: string, slug: string): string {
  if (dir.includes('前置知识')) return '前置知识'
  if (dir.includes('工程实践')) return '工程实践'
  if (slug.startsWith('S')) return '综述'
  return '精读'
}

/**
 * 按 category 分组生成 sidebar items
 */
export function generateSidebar(dir: string, baseLink: string) {
  const articles = scanArticles(dir, baseLink)
  
  // 按 category 分组
  const groups = new Map<string, ArticleMeta[]>()
  for (const article of articles) {
    const cat = article.category
    if (!groups.has(cat)) groups.set(cat, [])
    groups.get(cat)!.push(article)
  }
  
  // 转成 sidebar 格式
  return Array.from(groups.entries()).map(([category, items]) => ({
    text: category,
    collapsed: false,
    items: items.map(a => ({ text: a.title, link: a.link }))
  }))
}

/**
 * 获取所有文章（跨目录），用于标签页
 */
export function getAllArticles(): ArticleMeta[] {
  const dirs = [
    { dir: '前置知识', base: '/前置知识' },
    { dir: '论文综述', base: '/论文综述' },
    { dir: '工程实践', base: '/工程实践' },
    { dir: '工程项目', base: '/工程项目' },
  ]
  
  const all: ArticleMeta[] = []
  for (const { dir, base } of dirs) {
    all.push(...scanArticles(dir, base))
  }
  return all
}

/**
 * 按 tag 分组所有文章
 */
export function getArticlesByTag(): Record<string, ArticleMeta[]> {
  const all = getAllArticles()
  const byTag: Record<string, ArticleMeta[]> = {}
  
  for (const article of all) {
    for (const tag of article.tags) {
      if (!byTag[tag]) byTag[tag] = []
      byTag[tag].push(article)
    }
  }
  return byTag
}
