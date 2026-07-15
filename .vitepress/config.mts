import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

// --- 自动 sidebar 生成 ---

interface SidebarItem {
  text: string
  link: string
}

interface SidebarGroup {
  text: string
  collapsed?: boolean
  items: SidebarItem[]
}

/**
 * 从 markdown 正文中解析"知识链接"区域的链接
 * 格式: - [标题](链接) — 描述
 */
function parseReferencesFromContent(content: string): SidebarItem[] {
  const refs: SidebarItem[] = []
  // 匹配 **知识链接** 区域
  const sectionMatch = content.match(/\*\*知识链接\*\*[：:]?\s*\n([\s\S]*?)(?:\n---|\n##|\n\*\*[^知]|$)/)
  if (!sectionMatch) return refs

  const section = sectionMatch[1]
  // 匹配 markdown 链接: - [title](link)
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g
  let match: RegExpExecArray | null
  while ((match = linkRegex.exec(section)) !== null) {
    refs.push({ text: match[1], link: match[2] })
  }
  return refs
}

/**
 * 为指定目录生成 per-article sidebar：每篇文章的 sidebar 只显示它自己的知识链接
 * 同时返回目录首页的分组式 sidebar
 */
function scanArticleSidebar(dir: string, base: string): Record<string, SidebarGroup[]> {
  const fullDir = path.resolve(__dirname, '..', dir)
  if (!fs.existsSync(fullDir)) return {}

  const result: Record<string, SidebarGroup[]> = {}
  const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md')

  const articlesForIndex: Array<{
    title: string
    link: string
    order: number
    category: string
  }> = []

  for (const file of files) {
    const filePath = path.join(fullDir, file)
    const raw = fs.readFileSync(filePath, 'utf-8')
    const { data, content } = matter(raw)
    const slug = file.replace(/\.md$/, '')
    const link = `${base}/${slug}`
    const title = data.title || slug.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' ')

    articlesForIndex.push({
      title,
      link,
      order: data.order ?? 999,
      category: data.category ?? '未分类',
    })

    // 为每篇文章生成独立的 sidebar（显示知识链接）
    const refs = parseReferencesFromContent(raw)
    const groups: SidebarGroup[] = [
      {
        text: '📄 当前文章',
        collapsed: false,
        items: [{ text: title, link }]
      }
    ]

    if (refs.length > 0) {
      groups.push({
        text: '🔗 知识链接',
        collapsed: false,
        items: refs
      })
    }

    // 返回目录首页入口
    groups.push({
      text: '📚 返回目录',
      collapsed: false,
      items: [{ text: `← ${dir}`, link: `${base}/` }]
    })

    result[link] = groups
  }

  // 目录首页（index.md）的 sidebar：按 category 分组显示所有文章列表
  articlesForIndex.sort((a, b) => a.order - b.order || a.link.localeCompare(b.link))
  const groups = new Map<string, typeof articlesForIndex>()
  for (const article of articlesForIndex) {
    const cat = article.category
    if (!groups.has(cat)) groups.set(cat, [])
    groups.get(cat)!.push(article)
  }

  result[`${base}/`] = Array.from(groups.entries()).map(([category, items]) => ({
    text: category,
    collapsed: false,
    items: items.map(a => ({ text: a.title, link: a.link }))
  }))

  return result
}

/**
 * 旧的 scanSidebar 保留给前置知识等仍需要完整目录的页面
 */
function scanSidebar(dir: string, base: string): SidebarGroup[] {
  const fullDir = path.resolve(__dirname, '..', dir)
  if (!fs.existsSync(fullDir)) return []

  const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md')

  const articles = files.map(file => {
    const content = fs.readFileSync(path.join(fullDir, file), 'utf-8')
    const { data } = matter(content)
    const slug = file.replace(/\.md$/, '')
    return {
      title: data.title || slug.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' '),
      link: `${base}/${slug}`,
      order: data.order ?? 999,
      category: data.category ?? '未分类',
    }
  }).sort((a, b) => a.order - b.order || a.link.localeCompare(b.link))

  // 按 category 分组
  const groups = new Map<string, typeof articles>()
  for (const article of articles) {
    const cat = article.category
    if (!groups.has(cat)) groups.set(cat, [])
    groups.get(cat)!.push(article)
  }

  return Array.from(groups.entries()).map(([category, items]) => ({
    text: category,
    collapsed: false,
    items: items.map(a => ({ text: a.title, link: a.link }))
  }))
}

/**
 * 扫描系列文章子目录，为每个系列生成 sidebar
 */
function scanSeriesSidebar(): Record<string, SidebarGroup[]> {
  const seriesRoot = path.resolve(__dirname, '..', '系列')
  if (!fs.existsSync(seriesRoot)) return {}

  const result: Record<string, SidebarGroup[]> = {}
  const subdirs = fs.readdirSync(seriesRoot).filter(f => {
    return fs.statSync(path.join(seriesRoot, f)).isDirectory()
  })

  for (const sub of subdirs) {
    const subDir = path.join(seriesRoot, sub)
    const files = fs.readdirSync(subDir).filter(f => f.endsWith('.md'))

    // 读取 index.md 获取系列标题
    const indexFile = path.join(subDir, 'index.md')
    let seriesTitle = sub.replace(/_/g, ' ')
    if (fs.existsSync(indexFile)) {
      const { data } = matter(fs.readFileSync(indexFile, 'utf-8'))
      seriesTitle = data.title || seriesTitle
    }

    // 章节文件（排除 index.md）
    const chapters = files
      .filter(f => f !== 'index.md')
      .map(file => {
        const content = fs.readFileSync(path.join(subDir, file), 'utf-8')
        const { data } = matter(content)
        const slug = file.replace(/\.md$/, '')
        return {
          title: data.title || slug.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' '),
          link: `/系列/${sub}/${slug}`,
          order: data.order ?? (data.series?.chapter ?? 999),
        }
      })
      .sort((a, b) => a.order - b.order)

    result[`/系列/${sub}/`] = [
      {
        text: seriesTitle,
        collapsed: false,
        items: [
          { text: '系列概览', link: `/系列/${sub}/` },
          ...chapters.map(c => ({ text: c.title, link: c.link })),
        ]
      }
    ]
  }

  return result
}

// --- 配置 ---

export default withMermaid(defineConfig({
  title: '机器人学习笔记',
  description: '从 Transformer 到 ACT/VLA，从行为克隆到 RL 微调',
  lang: 'zh-CN',
  base: process.env.GITHUB_ACTIONS ? '/paper/' : '/',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      {
        text: '论文阅读',
        items: [
          { text: '论文综述', link: '/论文综述/' },
          { text: '论文精读', link: '/论文综述/#论文精读' },
        ]
      },
      { text: '前置知识', link: '/前置知识/' },
      {
        text: '工程笔记',
        items: [
          { text: '工程实践', link: '/工程实践/' },
          { text: '工程项目', link: '/工程项目/' },
        ]
      },
      { text: '系列文章', link: '/系列/' },
      { text: '按标签浏览', link: '/tags' },
    ],

    sidebar: {
      '/前置知识/': scanSidebar('前置知识', '/前置知识'),
      // 论文综述、工程实践、工程项目：每篇文章 sidebar 只显示知识链接
      ...scanArticleSidebar('论文综述', '/论文综述'),
      ...scanArticleSidebar('工程实践', '/工程实践'),
      ...scanArticleSidebar('工程项目', '/工程项目'),
      // 系列文章：sidebar 显示系列章节目录
      ...scanSeriesSidebar(),

      '/transformer_vla_tutorial/': [
        {
          text: 'Transformer & VLA 教程',
          collapsed: false,
          items: [
            { text: '教程简介', link: '/transformer_vla_tutorial/' },
            { text: '学习大纲', link: '/transformer_vla_tutorial/SYLLABUS' },
            { text: '学习方法', link: '/transformer_vla_tutorial/HOW_TO_STUDY' },
          ]
        },
        {
          text: '正文章节',
          collapsed: false,
          items: [
            { text: '00 全局地图', link: '/transformer_vla_tutorial/chapters/00_big_picture' },
            { text: '01 机器学习最小基础', link: '/transformer_vla_tutorial/chapters/01_ml_minimum' },
            { text: '02 Attention 手算', link: '/transformer_vla_tutorial/chapters/02_attention_by_hand' },
            { text: '03 Transformer 架构', link: '/transformer_vla_tutorial/chapters/03_transformer_architecture' },
            { text: '04 从语言到控制', link: '/transformer_vla_tutorial/chapters/04_from_language_to_control' },
            { text: '05 ACT: Action Chunking', link: '/transformer_vla_tutorial/chapters/05_act_action_chunking' },
            { text: '06 VLA Transformers', link: '/transformer_vla_tutorial/chapters/06_vla_transformers' },
          ]
        },
        {
          text: '练习答案',
          collapsed: true,
          items: [
            { text: '第 0 章答案', link: '/transformer_vla_tutorial/exercises/answers_00' },
            { text: '第 1 章答案', link: '/transformer_vla_tutorial/exercises/answers_01' },
            { text: '第 2 章答案', link: '/transformer_vla_tutorial/exercises/answers_02' },
            { text: '第 3 章答案', link: '/transformer_vla_tutorial/exercises/answers_03' },
            { text: '第 4 章答案', link: '/transformer_vla_tutorial/exercises/answers_04' },
            { text: '第 5 章答案', link: '/transformer_vla_tutorial/exercises/answers_05' },
            { text: '第 6 章答案', link: '/transformer_vla_tutorial/exercises/answers_06' },
          ]
        },
      ],
    },

    socialLinks: [
      // { icon: 'github', link: 'https://github.com/your-username/your-repo' }
    ],

    outline: {
      level: [2, 3],
      label: '目录'
    },

    search: {
      provider: 'local'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新'
    }
  },

  // topics/index.md 中有些旧链接目标文件尚未迁移，暂时忽略
  ignoreDeadLinks: true,

  markdown: {
    lineNumbers: true,
    math: true
  },

  mermaid: {
    // mermaid options
  }
}))
