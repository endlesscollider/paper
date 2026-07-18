// 纯函数模块，不依赖 Node API，因此可以同时被两端导入并保持行为完全一致：
//   1. 构建期 / dev 中间件（.vitepress/refSectionsPlugin.mts）：
//      扫描 srcDir 下的 .md 文件，为每篇文章计算出 "分节 JSON 应该叫什么文件名"。
//   2. 浏览器端组件（.vitepress/theme/composables/useRefPanel.ts）：
//      拿到用户点击的 <a href="/论文综述/xxx.html#锚点"> 之后，反推出要 fetch
//      哪一份分节 JSON。
//
// 两端只要用同一个 normalizeArticleKey + urlToSectionsFilename，就不需要维护
// 任何"反查表"——两边各自独立算出同一个文件名字符串即可对上。

/** 存放所有"分节 JSON"的目录名，相对于站点 base */
export const REF_SECTIONS_DIR = '__ref-sections__'

/**
 * 把"去掉扩展名后的路径"归一化成文章的唯一 key。
 * 例如：
 *   "论文综述/071_QChunking_RL与动作分块" -> "论文综述/071_QChunking_RL与动作分块"
 *   "系列/rlinf_deep_dive/index"          -> "系列/rlinf_deep_dive/"
 *   "index"                                -> ""
 */
function normalizeArticleKey(pathNoExt: string): string {
  let p = pathNoExt.replace(/^\/+/, '')
  if (p === 'index') return ''
  if (p.endsWith('/index')) return p.slice(0, -'index'.length)
  return p
}

/** 从 VitePress 的 relativePath（如 "论文综述/071_xxx.md"）算出文章 key */
export function articleKeyFromRelativePath(relativePath: string): string {
  return normalizeArticleKey(relativePath.replace(/\.md$/, ''))
}

/** 从渲染出的 <a href> 路径部分（不含 base，如 "/论文综述/072_x.html"）算出文章 key */
export function articleKeyFromHref(hrefPathname: string): string {
  return normalizeArticleKey(hrefPathname.replace(/^\/+/, '').replace(/\.html$/, ''))
}

/** 文章 key -> 分节 JSON 文件名（不含目录前缀） */
export function keyToSectionsFilename(key: string): string {
  const safe = key.replace(/\/+$/, '').replace(/\//g, '_') || 'index'
  return `${safe}.json`
}
