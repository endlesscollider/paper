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
 * 从渲染后的正文 HTML（不含 frontmatter）中提取所有带 id 的标题，切成
 * "原子小节"——每个标题只包含它自己到**下一个任意级别标题**之前的内容，
 * 不包含任何子标题的内容（子标题是数组里紧随其后的独立条目）。
 *
 * 为什么不像最初版本那样让父节点直接内嵌所有子节点的完整 HTML：
 * 如果 "4.2 节" 的 html 字段里塞进了 "4.2.0"、"4.2.1"……所有子节的完整内容，
 * 而这些子节自己又在数组里各存一份，同一段内容（尤其是公式渲染出的一大段
 * SVG 路径）就会被物理复制 N 次（N = 嵌套深度）。一篇公式密集的文章实测能
 * 把索引文件从几百 KB 撑到 10MB+，全站构建产物直接翻倍。
 *
 * 现在的方案：数组本身已经是按文档顺序排列的扁平列表，每个 section 只存
 * "原子内容"。读者想看"4.2 节"（连带它的子节）时，由前端按 level 做一次
 * 区间扫描——从 "4.2" 出发，往后累加所有 level 严格大于它的条目，直到遇到
 * 下一个 level ≤ 它的条目为止，再拼接这些条目的 html——用一次 O(n) 的字符
 * 串拼接换掉构建期的 O(n·depth) 物理复制，且不需要多存任何额外索引字段
 * （level 本身就足够重建区间）。具体拼接逻辑见
 * .vitepress/theme/composables/useRefPanel.ts 的 collectSectionRange()。
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

    // 原子边界：下一个"任意级别"的标题（不再要求同级或更高级），
    // 保证这里只截取属于本标题自己、还没被任何子标题占用的内容
    const endIdx = k + 1 < headingIdx.length ? headingIdx[k + 1] : children.length

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
