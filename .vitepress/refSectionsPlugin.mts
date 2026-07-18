// Vite 插件：生成"分节引用"功能所需要的静态 JSON 索引。
//
// 背景：写文章时可以用标准 Markdown 链接精确引用另一篇文章的某个标题
// （比如 `[回顾 Q-Chunking 精读 第 4.2 节](/论文综述/071_xxx#_4-2-xxx)`）。
// 读者点击这类链接时，不整页跳转，而是在右侧弹出一个分栏面板，展示目标
// 文章那一节的渲染结果。这个面板需要的数据来源就是本插件产出的 JSON：
//   /__ref-sections__/<文章key>.json
//   -> { articleUrl, articleTitle, sections: [{ id, level, title, breadcrumb, html }] }
//
// 实现方式：
//   - dev 模式：注册一个 Vite dev server 中间件，请求到来时才现算现渲染
//     （复用 VitePress 自己的 markdown-it 渲染器，保证公式、mermaid 占位、
//     标题 id slug 规则都和正式页面完全一致），不做磁盘缓存，方便改文章
//     立刻生效。
//   - build 模式：调用 generateRefSections()，把 srcDir 下所有 .md 文件
//     都渲染一遍并写入 outDir/__ref-sections__/*.json，作为纯静态资源随
//     dist 一起发布，前端用 fetch() 读取即可。
//
//     注意：这个函数**不能**放进 Vite/Rollup 插件的 buildEnd 钩子里调用。
//     那个钩子的签名是 (error?: Error) => void，并不会传入 VitePress 的
//     siteConfig（也就拿不到真正的 outDir），而且 VitePress 对 client
//     包构建时会用 emptyOutDir 清空 outDir，写入时机稍早就会被清掉。
//     正确的挂载点是 VitePress 自己的顶层配置项 buildEnd(siteConfig)——
//     它在 client + ssr 两个 bundle 都写盘完毕之后才执行一次，此时
//     siteConfig.outDir 才是最终稳定的目录。调用方式见 config.mts。
//
// 两端算文件名的逻辑必须一致，所以都调用 shared/refLink.mts 里的
// articleKeyFromRelativePath + keyToSectionsFilename。

import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import { createMarkdownRenderer } from 'vitepress'
import { extractSections } from './refSections.mjs'
import { REF_SECTIONS_DIR, articleKeyFromRelativePath, keyToSectionsFilename } from './shared/refLink.mts'

const IGNORED_TOP_LEVEL_DIRS = new Set([
  'node_modules',
  '.vitepress',
  '.git',
  '.github',
  '.kiro',
  '.agents',
  'public',
  'scripts'
])

/** 扫描 srcDir 下所有会被当作正文页面的 .md 文件（排除 node_modules / dist 等） */
async function globMarkdownFiles(srcDir: string): Promise<string[]> {
  const results: string[] = []
  const walk = (dir: string, relDir: string) => {
    let entries: fs.Dirent[]
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      const relPath = relDir ? `${relDir}/${entry.name}` : entry.name
      if (entry.isDirectory()) {
        if (relDir === '' && IGNORED_TOP_LEVEL_DIRS.has(entry.name)) continue
        if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue
        walk(path.join(dir, entry.name), relPath)
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        results.push(relPath)
      }
    }
  }
  walk(srcDir, '')
  return results
}

async function renderOneFile(
  md: Awaited<ReturnType<typeof createMarkdownRenderer>>,
  srcDir: string,
  relFile: string
) {
  const absFile = path.join(srcDir, relFile)
  const raw = fs.readFileSync(absFile, 'utf-8')
  const { content, data } = matter(raw)
  const relativePath = relFile.split(path.sep).join('/')
  const env: any = { path: absFile, relativePath, frontmatter: data, cleanUrls: false }
  const html = md.render(content, env)
  const sections = extractSections(html)
  const key = articleKeyFromRelativePath(relativePath)
  const articleUrl = '/' + (key || '')
  return {
    key,
    payload: {
      articleUrl,
      articleTitle: data.title || sections.find((s) => s.level === 1)?.title || '',
      sections
    }
  }
}

export interface RefSectionsPluginOptions {
  srcDir: string
  base: string
  markdown?: any
}

/**
 * 在 build 完成后调用一次，把 srcDir 下所有 .md 渲染成分节 JSON 写入
 * outDir/__ref-sections__/。供 config.mts 顶层的 buildEnd(siteConfig)
 * 钩子调用，此时 siteConfig.outDir 已经是最终稳定的输出目录。
 */
export async function generateRefSections(options: RefSectionsPluginOptions & { outDir: string }) {
  const { srcDir, base, markdown, outDir } = options
  const md = await createMarkdownRenderer(srcDir, markdown ?? {}, base)
  const files = await globMarkdownFiles(srcDir)

  const targetDir = path.join(outDir, REF_SECTIONS_DIR)
  fs.mkdirSync(targetDir, { recursive: true })

  for (const relFile of files) {
    const { key, payload } = await renderOneFile(md, srcDir, relFile)
    const outFile = path.join(targetDir, keyToSectionsFilename(key))
    fs.writeFileSync(outFile, JSON.stringify(payload), 'utf-8')
  }

  console.log(`[ref-sections] 已生成 ${files.length} 篇文章的分节索引 -> ${targetDir}`)
}

export function refSectionsPlugin(options: RefSectionsPluginOptions) {
  const { srcDir, base, markdown } = options
  let mdRendererPromise: ReturnType<typeof createMarkdownRenderer> | null = null
  const getRenderer = () => {
    if (!mdRendererPromise) {
      mdRendererPromise = createMarkdownRenderer(srcDir, markdown ?? {}, base)
    }
    return mdRendererPromise
  }

  return {
    name: 'vitepress-ref-sections',

    configureServer(server: any) {
      server.middlewares.use(async (req: any, res: any, next: any) => {
        const url: string = req.url || ''
        const prefix = `/${REF_SECTIONS_DIR}/`
        if (!url.startsWith(prefix)) return next()

        const filename = decodeURIComponent(url.slice(prefix.length).split('?')[0])
        if (!filename.endsWith('.json')) return next()
        const wantedName = filename

        try {
          const files = await globMarkdownFiles(srcDir)
          for (const relFile of files) {
            const key = articleKeyFromRelativePath(relFile.split(path.sep).join('/'))
            if (keyToSectionsFilename(key) === wantedName) {
              const md = await getRenderer()
              const { payload } = await renderOneFile(md, srcDir, relFile)
              res.setHeader('Content-Type', 'application/json; charset=utf-8')
              res.end(JSON.stringify(payload))
              return
            }
          }
          res.statusCode = 404
          res.end(JSON.stringify({ error: 'not found' }))
        } catch (err: any) {
          res.statusCode = 500
          res.end(JSON.stringify({ error: String(err?.message || err) }))
        }
      })
    }
  }
}
