<script setup>
import { ref, onMounted } from 'vue'
import { withBase } from 'vitepress'
import { getAllProgress } from '../composables/useReadProgress'

const props = defineProps({
  link: { type: String, required: true }
})

const progress = ref(0)
const label = ref('')
const badgeClass = ref('')

onMounted(() => {
  const allProgress = getAllProgress()

  // 生成所有可能的 key 格式
  const basedLink = withBase(props.link)
  const candidates = [
    props.link,
    props.link + '.html',
    props.link + '/',
    props.link.replace(/\/$/, ''),
    basedLink,
    basedLink + '.html',
    basedLink + '/',
    basedLink.replace(/\/$/, ''),
  ]

  // 处理中文路径编码
  const allCandidates = []
  for (const c of candidates) {
    allCandidates.push(c)
    try {
      const decoded = decodeURIComponent(c)
      if (decoded !== c) allCandidates.push(decoded)
    } catch {}
    try {
      const encoded = encodeURI(c)
      if (encoded !== c) allCandidates.push(encoded)
    } catch {}
  }

  let found = 0
  for (const candidate of allCandidates) {
    if (allProgress[candidate]?.progress) {
      found = Math.max(found, allProgress[candidate].progress)
    }
  }

  // 模糊匹配 fallback
  if (found === 0) {
    const slug = props.link.split('/').pop()
    if (slug) {
      for (const [key, val] of Object.entries(allProgress)) {
        if (val?.progress && (key.includes(slug) || key.includes(encodeURIComponent(slug)))) {
          found = Math.max(found, val.progress)
        }
      }
    }
  }

  progress.value = found

  if (found === 0) {
    label.value = '未浏览'
    badgeClass.value = 'badge-unread'
  } else if (found >= 90) {
    label.value = '已读完'
    badgeClass.value = 'badge-done'
  } else {
    label.value = `${found}%`
    badgeClass.value = 'badge-reading'
  }
})
</script>

<template>
  <span class="read-badge" :class="badgeClass">{{ label }}</span>
</template>

<style scoped>
.read-badge {
  display: inline-block;
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 8px;
  margin-left: 8px;
  font-weight: 500;
  vertical-align: middle;
}
.badge-unread {
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-3);
}
.badge-reading {
  background: #fff3cd;
  color: #856404;
}
.badge-done {
  background: #d4edda;
  color: #155724;
}

.dark .badge-reading {
  background: #4a3f00;
  color: #ffc107;
}
.dark .badge-done {
  background: #0d3320;
  color: #28a745;
}
</style>
