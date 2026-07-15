import { onMounted, onUnmounted, ref } from 'vue'
import { useRoute, withBase } from 'vitepress'

const STORAGE_KEY = 'reading-progress'

export interface ProgressData {
  [path: string]: {
    progress: number    // 0-100
    lastVisited: number // timestamp
  }
}

/**
 * 将路径规范化：去掉 base 前缀、.html 后缀、尾部斜杠
 * 确保存储和查询使用相同的 key 格式
 */
function normalizePath(rawPath: string): string {
  let p = rawPath
  // 去掉 base 前缀（如 /paper/）
  // withBase('/') 会返回 base 路径，如 '/paper/' 或 '/'
  try {
    const base = withBase('/').replace(/\/$/, '')
    if (base && base !== '' && p.startsWith(base)) {
      p = p.slice(base.length)
    }
  } catch {
    // SSR 环境下 withBase 可能不可用
  }
  // 去掉 .html 后缀
  p = p.replace(/\.html$/, '')
  // 去掉尾部斜杠（但保留根路径 /）
  if (p.length > 1) {
    p = p.replace(/\/$/, '')
  }
  // 确保以 / 开头
  if (!p.startsWith('/')) {
    p = '/' + p
  }
  // 解码 URL 编码的中文
  try {
    p = decodeURIComponent(p)
  } catch {}
  return p
}

function getStoredProgress(): ProgressData {
  if (typeof window === 'undefined') return {}
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function saveProgress(data: ProgressData) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
  } catch { /* quota exceeded, ignore */ }
}

/**
 * 获取指定路径的阅读进度 (0-100)
 */
export function getProgressForPath(path: string): number {
  const data = getStoredProgress()
  const normalized = normalizePath(path)
  // 先查规范化的 key
  if (data[normalized]?.progress) return data[normalized].progress
  // 兼容旧数据：遍历所有 key 做规范化比较
  for (const [key, val] of Object.entries(data)) {
    if (val?.progress && normalizePath(key) === normalized) {
      return val.progress
    }
  }
  return 0
}

/**
 * 获取所有文章的阅读进度（返回原始数据 + 规范化索引）
 */
export function getAllProgress(): ProgressData {
  const raw = getStoredProgress()
  // 构建一个同时包含原始 key 和规范化 key 的映射
  const merged: ProgressData = { ...raw }
  for (const [key, val] of Object.entries(raw)) {
    if (val) {
      const normalized = normalizePath(key)
      if (!merged[normalized] || (val.progress > (merged[normalized]?.progress ?? 0))) {
        merged[normalized] = val
      }
    }
  }
  return merged
}

/**
 * 在文章页面中使用：自动追踪滚动进度
 */
export function useReadProgress() {
  const progress = ref(0)
  const route = useRoute()
  let ticking = false

  function calcProgress(): number {
    const docEl = document.documentElement
    const scrollTop = window.scrollY || docEl.scrollTop
    const scrollHeight = docEl.scrollHeight - docEl.clientHeight
    if (scrollHeight <= 0) return 100
    return Math.min(100, Math.round((scrollTop / scrollHeight) * 100))
  }

  function onScroll() {
    if (ticking) return
    ticking = true
    requestAnimationFrame(() => {
      const current = calcProgress()
      progress.value = current

      // 使用规范化路径作为存储 key
      const normalizedPath = normalizePath(route.path)
      const data = getStoredProgress()
      const existing = data[normalizedPath]?.progress ?? 0

      if (current > existing) {
        data[normalizedPath] = {
          progress: current,
          lastVisited: Date.now()
        }
        saveProgress(data)
      } else if (!data[normalizedPath]) {
        data[normalizedPath] = {
          progress: current,
          lastVisited: Date.now()
        }
        saveProgress(data)
      }

      ticking = false
    })
  }

  onMounted(() => {
    const normalizedPath = normalizePath(route.path)
    const data = getStoredProgress()
    progress.value = data[normalizedPath]?.progress ?? 0

    // 记录访问
    if (!data[normalizedPath]) {
      data[normalizedPath] = { progress: 0, lastVisited: Date.now() }
      saveProgress(data)
    }

    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()
  })

  onUnmounted(() => {
    window.removeEventListener('scroll', onScroll)
  })

  return { progress }
}
