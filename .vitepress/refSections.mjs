// 构建期共享模块：把一篇文章渲染后的正文 HTML 切分成"按标题分节"的片段索引。
//
// 用途：写文章时用标准 Markdown 链接引用另一篇文章的某个标题锚点
// （如 `[回顾 XXX 精读 第 4.2 节](/论文综述/071_xxx#_4-2-xxx)`），
// 前端点击这类链接时不整页跳转，而是 fetch 这篇文章的分节 JSON，按锚点
// 找到对应 section，渲染进右侧分栏面板。

import { load as cheerioLoad } from 'cheerio'

const HEADING_LEVEL = { H1: 1, H2: 2, H3: 3, H4: 4, H5: 5, H6: 6 }

/**
 * Mermaid 代码块被 vitepress-plugin-mermaid 转换成 <Suspense><Mermaid .../></Suspense>
 * 这种"伪 HTML"，必须经过 Vue SFC 编译才会变成真正的图。分节面板是直接把 HTML
 * 字符串当 innerHTML 注入，不会经过 Vue 编译，所以这里替换成一段提示文字，
 * 引导用户点击"跳转阅读全文"查看完整图表。
 */
function stripUnrenderableWidgets(html) {
  return html.replace(
    /<Suspense>[\s\S]*?<\/Suspense>/g,
    '<div class="ref-panel-notice">📊 此处包含 Mermaid 图表，预览暂不支持渲染，请点击"跳转阅读全文"查看</div>'
  )
}

/**
 * 从渲染后的正文 HTML（不含 frontmatter）中提取所有带 id 的标题及其对应的
 * "小节内容"——标题本身 + 一直到下一个同级或更高级标题之前的所有内容
 * （因此天然包含所有更深层的子标题，比如引用 "4.2 节" 会连带 "4.2.0"、"4.2.1" 等子节）。
 */
export function extractSections(bodyHtml) {
  const $ = cheerioLoad(`<div id="__root">${bodyHtml}</div>`, { decodeEntities: false })
  const root = $('#__root')
  const children = root.contents().toArray()

  const headingIdx = []
  for (let i = 0; i < children.length; i++) {
    const node = children[i]
    if (node.type === 'tag' && HEADING_LEVEL[node.name.toUpperCase()]) {
      headingIdx.push(i)
    }
  }

  const sections = []
  const stack = [] // 标题层级栈，用于生成 breadcrumb（面包屑，告诉读者这一节在文章里的上下文位置）

  for (let k = 0; k < headingIdx.length; k++) {
    const startIdx = headingIdx[k]
    const node = children[startIdx]
    const level = HEADING_LEVEL[node.name.toUpperCase()]
    const id = $(node).attr('id')
    if (!id) continue

    // 标题纯文本（去掉 markdown-it-anchor 注入的 "#" permalink 图标）
    const title = $(node).clone().find('a.header-anchor').remove().end().text().trim()

    // 找到下一个"同级或更高级"标题的位置作为本节结束边界
    let endIdx = children.length
    for (let j = k + 1; j < headingIdx.length; j++) {
      const otherNode = children[headingIdx[j]]
      const otherLevel = HEADING_LEVEL[otherNode.name.toUpperCase()]
      if (otherLevel <= level) {
        endIdx = headingIdx[j]
        break
      }
    }

    const sectionHtml = children
      .slice(startIdx, endIdx)
      .map((n) => $.html(n))
      .join('')

    while (stack.length && stack[stack.length - 1].level >= level) {
      stack.pop()
    }
    const breadcrumb = stack.map((s) => s.title)
    stack.push({ level, title })

    sections.push({
      id,
      level,
      title,
      breadcrumb,
      html: stripUnrenderableWidgets(sectionHtml)
    })
  }

  return sections
}
