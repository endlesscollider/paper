import { onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vitepress'

const STORAGE_KEY = 'reading-progress'

export interface ProgressData {
  [path: string]: {
    progress: number    // 0-100
    lastVisited: number // timestamp
  }
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
  return data[path]?.progress ?? 0
}

/**
 * 获取所有文章的阅读进度
 */
export function getAllProgress(): ProgressData {
  return getStoredProgress()
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

      // 只保存最大进度（不因为回滚而降低）
      const data = getStoredProgress()
      const path = route.path
      const existing = data[path]?.progress ?? 0
      if (current > existing) {
        data[path] = {
          progress: current,
          lastVisited: Date.now()
        }
        saveProgress(data)
      } else if (!data[path]) {
        data[path] = {
          progress: current,
          lastVisited: Date.now()
        }
        saveProgress(data)
      }

      ticking = false
    })
  }

  onMounted(() => {
    // 初始化：读取已有进度
    const data = getStoredProgress()
    progress.value = data[route.path]?.progress ?? 0

    // 记录访问
    if (!data[route.path]) {
      data[route.path] = { progress: 0, lastVisited: Date.now() }
      saveProgress(data)
    }

    window.addEventListener('scroll', onScroll, { passive: true })
    // 初始计算一次
    onScroll()
  })

  onUnmounted(() => {
    window.removeEventListener('scroll', onScroll)
  })

  return { progress }
}
