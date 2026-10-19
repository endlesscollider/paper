// 独立进程入口：`vitepress build` 结束后单独跑一遍分节索引生成。
//
// 为什么拆成独立进程：全站 860+ 篇文章需要各渲染两遍（一遍出页面，
// 一遍出分节 JSON）。两遍放在同一个 Node 进程里时，第一遍的页面渲染
// 结果仍留在堆上，第二遍的 cheerio DOM 再叠加，实测峰值超过 13GB，
// 放在 GitHub Actions runner（约 7GB 内存）上必然被 OOM-Kill。
// 拆成两个进程后，主渲染进程退出即整体归还内存，本进程从零开始，
// 峰值只取决于单遍渲染，可以用独立且更小的 --max-old-space-size 控制。
//
// 调用方式见 package.json 的 docs:build。

import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { generateRefSections } from './refSectionsPlugin.mts'

const here = path.dirname(fileURLToPath(import.meta.url))
const srcDir = path.resolve(here, '..')

// 与 vitepress 的 outDir 默认值一致；CI 上 base 为 /paper/
const outDir = path.join(here, 'dist')
const base = process.env.GITHUB_ACTIONS ? '/paper/' : '/'

await generateRefSections({
  srcDir,
  base,
  markdown: { math: true },
  outDir
})
