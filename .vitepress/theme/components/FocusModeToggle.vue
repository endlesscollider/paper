<script setup>
import { useFocusMode } from '../composables/useFocusMode'

const { isFocusMode, toggleFocusMode } = useFocusMode()
</script>

<template>
  <button
    class="focus-mode-toggle"
    :class="{ active: isFocusMode }"
    :title="isFocusMode ? '退出全屏阅读' : '全屏阅读（隐藏侧边栏）'"
    @click="toggleFocusMode"
  >
    <svg v-if="!isFocusMode" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
    </svg>
    <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
    </svg>
    <span class="focus-mode-label">{{ isFocusMode ? '退出全屏' : '全屏阅读' }}</span>
  </button>
</template>

<style scoped>
.focus-mode-toggle {
  position: fixed;
  right: 16px;
  bottom: 16px;
  z-index: 90;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  border-radius: 20px;
  border: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text-2);
  font-size: 12px;
  cursor: pointer;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
}

.focus-mode-toggle:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

.focus-mode-toggle.active {
  background: var(--vp-c-brand-1);
  color: white;
  border-color: var(--vp-c-brand-1);
}

.focus-mode-label {
  white-space: nowrap;
}

@media (max-width: 640px) {
  .focus-mode-label {
    display: none;
  }
  .focus-mode-toggle {
    padding: 10px;
    right: 12px;
    bottom: 12px;
  }
}
</style>
