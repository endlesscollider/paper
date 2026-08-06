// 检查项目中所有锚点链接是否匹配目标文件中的实际标题（使用新 slugify）
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// 新的自定义 slugify（和 config.mts 保持一致）
function slugify(text) {
  return text
    .trim()
    .toLowerCase()
    .replace(/[、，。：；！？…\u201c\u201d\u2018\u2019（）【】《》"']/g, ' ')
    .replace(/\u2014+/g, ' ')
    .replace(/\s+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
}

function findMdFiles(dir) {
  const results = []
  if (!fs.existsSync(dir)) return results
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  for (const entry of entries) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules' && !entry.name.startsWith('40cd')) {
      results.push(...findMdFiles(full))
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      results.push(full)
    }
  }
  return results
}

function extractHeadings(content) {
  const headingRegex = /^#{1,6}\s+(.+)$/gm
  const headings = []
  let match
  while ((match = headingRegex.exec(content)) !== null) {
    // Remove markdown formatting from heading text
    const text = match[1]
      .replace(/\*\*/g, '')
      .replace(/`/g, '')
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/\$[^$]*\$/g, '') // remove inline math
      .trim()
    headings.push(slugify(text))
  }
  return headings
}

function extractLinks(content, filePath) {
  const linkRegex = /\[([^\]]*)\]\(([^)]+)\)/g
  const links = []
  let match
  while ((match = linkRegex.exec(content)) !== null) {
    const href = match[2]
    if (href.includes('#') && !href.startsWith('http')) {
      const hashIdx = href.indexOf('#')
      const pathPart = href.substring(0, hashIdx)
      const anchor = href.substring(hashIdx + 1)
      if (anchor) {
        links.push({ text: match[1], path: pathPart, anchor, source: filePath })
      }
    }
  }
  return links
}

function resolveLink(linkPath, sourceFile) {
  if (!linkPath) return sourceFile
  if (linkPath.startsWith('/')) {
    return path.join(__dirname, linkPath + '.md')
  } else {
    return path.join(path.dirname(sourceFile), linkPath + '.md')
  }
}

const allFiles = findMdFiles(__dirname)
const brokenLinks = []

for (const file of allFiles) {
  const content = fs.readFileSync(file, 'utf-8')
  const links = extractLinks(content, file)

  for (const link of links) {
    const targetFile = resolveLink(link.path, file)
    if (!fs.existsSync(targetFile)) continue
    const targetContent = fs.readFileSync(targetFile, 'utf-8')
    const headings = extractHeadings(targetContent)

    const normalizedLinkAnchor = link.anchor.toLowerCase()
    const found = headings.some(h => h === normalizedLinkAnchor)

    if (!found) {
      const relSource = path.relative(__dirname, file)
      brokenLinks.push({
        source: relSource,
        target: link.path || '(same file)',
        anchor: link.anchor,
        text: link.text,
        // Find closest match
        closest: headings
          .filter(h => {
            // fuzzy: at least half the words match
            const words = normalizedLinkAnchor.split('-').filter(w => w.length > 1)
            const matched = words.filter(w => h.includes(w))
            return matched.length >= words.length * 0.5
          })
          .slice(0, 3)
      })
    }
  }
}

console.log(`\n=== 使用新 slugify 检查 ===`)
console.log(`共检查 ${allFiles.length} 个文件`)
console.log(`发现 ${brokenLinks.length} 个可能失效的锚点链接:\n`)

for (const bl of brokenLinks) {
  console.log(`❌ ${bl.source}`)
  console.log(`   → ${bl.target}#${bl.anchor}`)
  if (bl.closest.length > 0) {
    console.log(`   可能应为: ${bl.closest[0]}`)
  }
  console.log('')
}
