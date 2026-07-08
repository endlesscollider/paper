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
      { text: '按标签浏览', link: '/tags' },
    ],

    sidebar: {
      '/前置知识/': scanSidebar('前置知识', '/前置知识'),
      '/论文综述/': scanSidebar('论文综述', '/论文综述'),
      '/工程实践/': scanSidebar('工程实践', '/工程实践'),
      '/工程项目/': scanSidebar('工程项目', '/工程项目'),

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
