import { ref, onMounted } from 'vue'

const STORAGE_KEY = 'recently-viewed-articles'
const MAX_ITEMS = 8

export interface RecentItem {
  title: string
  link: string
  timestamp: number
}

export function useRecentlyViewed() {
  const items = ref<RecentItem[]>([])

  function load() {
    if (typeof window === 'undefined') return
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      items.value = raw ? JSON.parse(raw) : []
    } catch {
      items.value = []
    }
  }

  function record(title: string, link: string) {
    if (typeof window === 'undefined') return
    // 移除已有的同链接记录
    const list = items.value.filter(i => i.link !== link)
    // 添加到最前面
    list.unshift({ title, link, timestamp: Date.now() })
    // 限制数量
    items.value = list.slice(0, MAX_ITEMS)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items.value))
    } catch { /* ignore */ }
  }

  onMounted(load)

  return { items, record, load }
}
