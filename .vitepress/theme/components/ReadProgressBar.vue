<script setup>
import { useReadProgress } from '../composables/useReadProgress'

const { progress } = useReadProgress()
</script>

<template>
  <div class="progress-bar-container" v-if="progress > 0 && progress < 100">
    <div class="progress-bar-fill" :style="{ width: progress + '%' }"></div>
  </div>
</template>

<style scoped>
.progress-bar-container {
  position: fixed;
  top: var(--vp-nav-height, 64px);
  left: 0;
  right: 0;
  height: 3px;
  background: transparent;
  z-index: 100;
}
.progress-bar-fill {
  height: 100%;
  background: var(--vp-c-brand-1);
  transition: width 0.1s ease-out;
}
</style>
