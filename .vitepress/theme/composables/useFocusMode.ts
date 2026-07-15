import { ref } from 'vue'

const STORAGE_KEY = 'vp-focus-mode'

// 单例状态，保证多个组件实例共享同一个开关
const isFocusMode = ref(false)
let initialized = false

function applyClass(value: boolean) {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle('focus-mode', value)
}

function initFocusMode() {
  if (initialized || typeof window === 'undefined') return
  initialized = true
  const saved = window.localStorage.getItem(STORAGE_KEY)
  isFocusMode.value = saved === '1'
  applyClass(isFocusMode.value)
}

function toggleFocusMode() {
  isFocusMode.value = !isFocusMode.value
  applyClass(isFocusMode.value)
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEY, isFocusMode.value ? '1' : '0')
  }
}

export function useFocusMode() {
  initFocusMode()
  return { isFocusMode, toggleFocusMode }
}
