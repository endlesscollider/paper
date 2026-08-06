import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import { refSectionsPlugin, generateRefSections } from './refSectionsPlugin.mts'

// --- 自动 sidebar 生成 ---

interface SidebarItem {
  text: string
  link?: string
  collapsed?: boolean
  items?: SidebarItem[]
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
  // 每日快报按时间倒序（最新在前），其他按 order 升序
  if (dir === '每日快报') {
    articlesForIndex.sort((a, b) => b.order - a.order || b.link.localeCompare(a.link))
  } else {
    articlesForIndex.sort((a, b) => a.order - b.order || a.link.localeCompare(b.link))
  }
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
    // 超过 20 项的分组默认收起，避免 sidebar 过长显示不全
    collapsed: items.length > 20 ? true : false,
    items: items.map(a => ({ text: a.title, link: a.link }))
  }))
}

/**
 * 从系列 index.md 的"## 章节目录"区域解析"第N部分：标题"分组信息。
 * 兼容两种写法：
 *   1. 表格内的加粗分组行： | **第一部分：全局认知** | | |
 *   2. 三级标题分组：       ### 第一部分：设计哲学与全局架构
 * 返回：每个分组标题的出现顺序（partOrder），以及章节号 -> 分组标题的映射
 */
function parseSeriesPartsFromIndex(content: string): {
  partOrder: string[]
  chapterPart: Map<number, string>
} {
  const partOrder: string[] = []
  const chapterPart = new Map<number, string>()
  const lines = content.split('\n')

  let inSection = false
  let currentPart: string | null = null

  for (const line of lines) {
    if (/^##\s+章节目录/.test(line)) {
      inSection = true
      continue
    }
    if (!inSection) continue
    // 遇到下一个二级标题，说明"章节目录"区域结束
    if (/^##\s+/.test(line) && !/^##\s+章节目录/.test(line)) break

    // 三级/四级标题式分组： ### 第一部分：xxx
    let m = line.match(/^#{3,4}\s*(第[一二三四五六七八九十百]+部分[:：].*)$/)
    if (m) {
      currentPart = m[1].trim()
      if (!partOrder.includes(currentPart)) partOrder.push(currentPart)
      continue
    }

    // 表格加粗行式分组： | **第一部分：xxx** | | |
    m = line.match(/^\|\s*\*\*(第[一二三四五六七八九十百]+部分[:：][^*]*)\*\*\s*\|/)
    if (m) {
      currentPart = m[1].trim()
      if (!partOrder.includes(currentPart)) partOrder.push(currentPart)
      continue
    }

    // 章节行： | 01 | [标题](链接) | ... |
    m = line.match(/^\|\s*(\d+)\s*\|\s*\[([^\]]+)\]\(([^)]+)\)/)
    if (m && currentPart) {
      chapterPart.set(parseInt(m[1], 10), currentPart)
    }
  }

  return { partOrder, chapterPart }
}

/**
 * 扫描系列文章子目录，为每个系列生成 sidebar。
 * 如果该系列的 index.md 中定义了"第N部分"分组，则 sidebar 按分组显示
 * 二级层级（系列 → 部分 → 章节），并给每章加上两位数序号前缀，
 * 让读者一眼看出章节顺序和所属阶段。
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

    // 读取 index.md 获取系列标题 + 分组信息
    const indexFile = path.join(subDir, 'index.md')
    let seriesTitle = sub.replace(/_/g, ' ')
    let partOrder: string[] = []
    let chapterPart = new Map<number, string>()
    if (fs.existsSync(indexFile)) {
      const raw = fs.readFileSync(indexFile, 'utf-8')
      const { data, content } = matter(raw)
      seriesTitle = data.title || seriesTitle
      const parsed = parseSeriesPartsFromIndex(content)
      partOrder = parsed.partOrder
      chapterPart = parsed.chapterPart
    }

    // 章节文件（排除 index.md），带上两位数序号前缀方便识别顺序
    const chapters = files
      .filter(f => f !== 'index.md')
      .map(file => {
        const content = fs.readFileSync(path.join(subDir, file), 'utf-8')
        const { data } = matter(content)
        const slug = file.replace(/\.md$/, '')
        const chapterNum: number = data.order ?? (data.series?.chapter ?? 999)
        const rawTitle = data.title || slug.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' ')
        const numLabel = chapterNum < 1000 ? String(chapterNum).padStart(2, '0') : ''
        return {
          title: numLabel ? `${numLabel}. ${rawTitle}` : rawTitle,
          link: `/系列/${sub}/${slug}`,
          order: chapterNum,
          part: chapterPart.get(chapterNum) ?? null,
        }
      })
      .sort((a, b) => a.order - b.order)

    let items: SidebarItem[]
    if (partOrder.length > 0) {
      // 按"第N部分"分组，未被任何分组覆盖的章节归到"其他章节"
      const grouped = new Map<string, typeof chapters>()
      const ungrouped: typeof chapters = []
      for (const c of chapters) {
        if (c.part) {
          if (!grouped.has(c.part)) grouped.set(c.part, [])
          grouped.get(c.part)!.push(c)
        } else {
          ungrouped.push(c)
        }
      }
      items = [
        { text: '系列概览', link: `/系列/${sub}/` },
        ...partOrder
          .filter(p => grouped.has(p))
          .map(p => ({
            text: p,
            collapsed: false,
            items: grouped.get(p)!.map(c => ({ text: c.title, link: c.link })),
          })),
        ...(ungrouped.length > 0
          ? [{
              text: '其他章节',
              collapsed: false,
              items: ungrouped.map(c => ({ text: c.title, link: c.link })),
            }]
          : []),
      ]
    } else {
      // 没有分组信息：保持扁平列表，但仍然带序号前缀
      items = [
        { text: '系列概览', link: `/系列/${sub}/` },
        ...chapters.map(c => ({ text: c.title, link: c.link })),
      ]
    }

    result[`/系列/${sub}/`] = [
      {
        text: seriesTitle,
        collapsed: false,
        items,
      }
    ]
  }

  return result
}

/**
 * 为系列总索引页（/系列/index.md）生成 sidebar，列出所有系列
 */
function scanSeriesIndexSidebar(): SidebarGroup[] {
  const seriesRoot = path.resolve(__dirname, '..', '系列')
  if (!fs.existsSync(seriesRoot)) return []

  const subdirs = fs.readdirSync(seriesRoot).filter(f => {
    return fs.statSync(path.join(seriesRoot, f)).isDirectory()
  })

  const items: SidebarItem[] = subdirs.map(sub => {
    const indexFile = path.join(seriesRoot, sub, 'index.md')
    let title = sub.replace(/_/g, ' ')
    if (fs.existsSync(indexFile)) {
      const { data } = matter(fs.readFileSync(indexFile, 'utf-8'))
      title = data.title || title
    }
    return { text: title, link: `/系列/${sub}/` }
  })

  return [
    {
      text: '📖 全部系列',
      collapsed: false,
      items,
    }
  ]
}

/**
 * 转义 LaTeX 源码，使其可以安全地放进 HTML 属性（data-tex="..."）
 */
function escapeTexForAttr(tex: string): string {
  return tex
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
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
      {
        text: '前置知识',
        items: [
          { text: '前置知识目录', link: '/前置知识/' },
          { text: '按标签搜索', link: '/prereq-search' },
        ]
      },
      {
        text: '工程笔记',
        items: [
          { text: '工程实践', link: '/工程实践/' },
          { text: '工程项目', link: '/工程项目/' },
        ]
      },
      { text: '系列文章', link: '/系列/' },
      { text: '硬件基础', link: '/硬件基础/' },
      { text: '每日快报', link: '/每日快报/' },
      { text: '按标签浏览', link: '/tags' },
    ],

    sidebar: {
      '/前置知识/': scanSidebar('前置知识', '/前置知识'),
      '/硬件基础/': scanSidebar('硬件基础', '/硬件基础'),
      // 论文综述、工程实践、工程项目：每篇文章 sidebar 只显示知识链接
      ...scanArticleSidebar('论文综述', '/论文综述'),
      ...scanArticleSidebar('工程实践', '/工程实践'),
      ...scanArticleSidebar('工程项目', '/工程项目'),
      ...scanArticleSidebar('每日快报', '/每日快报'),
      // 系列文章：sidebar 显示系列章节目录
      // 注意：系列子目录的 key（如 /系列/xxx/）必须在 '/系列/' 之前展开，
      // 否则 VitePress 的 startsWith 最长前缀匹配会优先命中子目录 key
      ...scanSeriesSidebar(),
      // 系列总索引页的 sidebar
      '/系列/': scanSeriesIndexSidebar(),

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
    math: true,
    // 自定义 anchor slugify：去掉中文标点，使锚点链接可预测、易书写
    anchor: {
      slugify(str: string) {
        return str
          .trim()
          .toLowerCase()
          // 中文标点和引号替换为空格（后续统一转 -）
          .replace(/[、，。：；！？…""''（）【】《》"']/g, ' ')
          // 全角破折号 ——（\u2014）替换为空格
          .replace(/\u2014+/g, ' ')
          // 百分号替换为空（避免 decodeURI 误认为 percent-encoding）
          .replace(/%/g, '')
          // 空格替换为 -
          .replace(/\s+/g, '-')
          // 连续 - 合并
          .replace(/-{2,}/g, '-')
          // 去掉首尾 -
          .replace(/^-+|-+$/g, '')
      }
    },
    // 给每个公式的 <mjx-container> 注入 data-tex 属性，保存原始 LaTeX 源码。
    // MathJax 把公式渲染成纯 SVG 路径，渲染结果里根本没有可选中的文本，
    // 所以"点击复制公式"功能必须依赖这个属性才能拿到原始 LaTeX。
    config: (md) => {
      const wrap = (renderRule: any) => {
        return (tokens: any, idx: number, options2: any, env: any, self: any) => {
          const html = renderRule(tokens, idx, options2, env, self)
          const tex = escapeTexForAttr(tokens[idx].content ?? '')
          return html.replace(/^<mjx-container /, `<mjx-container data-tex="${tex}" `)
        }
      }
      if (md.renderer.rules.math_inline) {
        md.renderer.rules.math_inline = wrap(md.renderer.rules.math_inline)
      }
      if (md.renderer.rules.math_block) {
        md.renderer.rules.math_block = wrap(md.renderer.rules.math_block)
      }
    }
  },

  mermaid: {
    // mermaid options
  },

  vite: {
    plugins: [
      // “分栏引用”功能 dev 模式下的即时渲染中间件，
      // 详见 .vitepress/refSectionsPlugin.mts 顶部注释。
      refSectionsPlugin({
        srcDir: path.resolve(__dirname, '..'),
        base: process.env.GITHUB_ACTIONS ? '/paper/' : '/',
        markdown: { math: true }
      })
    ]
  },

  // build 模式下生成"分栏引用"功能所需的静态 JSON 索引。
  // 必须用 VitePress 顶层的 buildEnd(siteConfig) 钩子而不是 vite 插件的
  // buildEnd 钩子，原因见 refSectionsPlugin.mts 顶部注释。
  async buildEnd(siteConfig) {
    await generateRefSections({
      srcDir: path.resolve(__dirname, '..'),
      base: process.env.GITHUB_ACTIONS ? '/paper/' : '/',
      markdown: { math: true },
      outDir: siteConfig.outDir
    })
  }
}))
