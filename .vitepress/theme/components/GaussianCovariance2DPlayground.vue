<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

// 二维协方差交互演示：只有三个自由度（两个方向的宽度 + 一个旋转角），
// 目的是让读者在最少的参数下看清「宽度是什么」和「旋转如何在非对角项 σxy 上留下痕迹」。
// 用一个纯 2D canvas 画出高斯浓度热力图 + 等值椭圆 + 两条主轴，不依赖 three.js。

type Preset = { label: string; sx: number; sy: number; theta: number }

const canvasRef = ref<HTMLCanvasElement | null>(null)

// sx / sy 是沿椭圆两条局部主轴的标准差（宽度），theta 是把这个正椭圆整体转过的角度（度）。
const sx = ref(1.6)
const sy = ref(0.6)
const theta = ref(30)

const presets: Preset[] = [
  { label: '圆形 (σx=σy)', sx: 1.0, sy: 1.0, theta: 0 },
  { label: '横平竖直椭圆', sx: 1.7, sy: 0.55, theta: 0 },
  { label: '斜 30° 椭圆', sx: 1.6, sy: 0.6, theta: 30 },
  { label: '斜 -60° 细长条', sx: 1.9, sy: 0.35, theta: -60 },
]

function setPreset(p: Preset) {
  sx.value = p.sx
  sy.value = p.sy
  theta.value = p.theta
}

// 由两条主轴宽度 + 旋转角合成 2×2 协方差矩阵：Σ = R · diag(σx², σy²) · Rᵀ
// 展开后可以直接看到旋转角 θ 怎么把能量「渗」进非对角位置 σxy。
const covariance = computed(() => {
  const rad = (theta.value * Math.PI) / 180
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  const lx = sx.value ** 2
  const ly = sy.value ** 2
  // R = [[c, -s], [s, c]]，Σ = R diag(lx, ly) Rᵀ
  const a = c * c * lx + s * s * ly // Σ11 = σx²
  const b = c * s * lx - c * s * ly // Σ12 = Σ21 = σxy
  const d = s * s * lx + c * c * ly // Σ22 = σy²
  return { a, b, d }
})

// 判断非对角项在当前参数下是否「几乎为 0」，用来给读者高亮提示。
const offDiagNearZero = computed(() => Math.abs(covariance.value.b) < 1e-3)

let animationId = 0
let devicePixelRatioCached = 1

function draw() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const cssW = canvas.clientWidth || 460
  const cssH = canvas.clientHeight || 460
  const dpr = devicePixelRatioCached
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr)
    canvas.height = Math.round(cssH * dpr)
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)

  // 世界坐标范围 [-range, range]，映射到画布像素。
  const range = 3.2
  const toPx = (wx: number, wy: number): [number, number] => [
    ((wx + range) / (2 * range)) * cssW,
    ((range - wy) / (2 * range)) * cssH, // y 轴翻转，让 +y 朝上
  ]

  const { a, b, d } = covariance.value
  // 协方差矩阵的逆，用于计算马氏距离（高斯指数里的二次型）。
  const det = a * d - b * b
  const inv00 = d / det
  const inv01 = -b / det
  const inv11 = a / det

  // 1) 高斯浓度热力图：逐像素算 G(x)=exp(-0.5·xᵀΣ⁻¹x)，蓝色越深越浓。
  const step = 4 // 每 4px 画一个小方块，兼顾清晰度与性能
  for (let py = 0; py < cssH; py += step) {
    for (let px = 0; px < cssW; px += step) {
      const wx = (px / cssW) * 2 * range - range
      const wy = range - (py / cssH) * 2 * range
      const q = inv00 * wx * wx + 2 * inv01 * wx * wy + inv11 * wy * wy
      const g = Math.exp(-0.5 * q)
      if (g < 0.02) continue
      const alpha = Math.min(g, 1)
      ctx.fillStyle = `rgba(33, 150, 243, ${alpha * 0.85})`
      ctx.fillRect(px, py, step, step)
    }
  }

  // 2) 坐标轴（世界 x / y 轴），帮助读者判断椭圆相对坐标轴歪了多少。
  ctx.strokeStyle = 'rgba(96, 125, 139, 0.5)'
  ctx.lineWidth = 1
  ctx.beginPath()
  const [ax0] = toPx(-range, 0)
  const [, ay0] = toPx(0, 0)
  ctx.moveTo(0, ay0)
  ctx.lineTo(cssW, ay0)
  ctx.moveTo(ax0 + ((range) / (2 * range)) * 0, 0)
  const [cx] = toPx(0, 0)
  ctx.moveTo(cx, 0)
  ctx.lineTo(cx, cssH)
  ctx.stroke()

  // 3) 1σ 等值椭圆轮廓：参数方程用主轴宽度 + 旋转角直接画，避免特征分解。
  const rad = (theta.value * Math.PI) / 180
  const cos = Math.cos(rad)
  const sin = Math.sin(rad)
  ctx.strokeStyle = '#1565c0'
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i <= 100; i++) {
    const t = (i / 100) * Math.PI * 2
    // 正椭圆上的点 (σx cos t, σy sin t)，再旋转 θ
    const ex = sx.value * Math.cos(t)
    const ey = sy.value * Math.sin(t)
    const wx = cos * ex - sin * ey
    const wy = sin * ex + cos * ey
    const [ppx, ppy] = toPx(wx, wy)
    if (i === 0) ctx.moveTo(ppx, ppy)
    else ctx.lineTo(ppx, ppy)
  }
  ctx.closePath()
  ctx.stroke()

  // 4) 两条主轴：红=长轴(σx方向)，绿=短轴(σy方向)，直观显示椭圆朝向。
  const drawAxis = (len: number, dirX: number, dirY: number, color: string) => {
    const wx = cos * (len * dirX) - sin * (len * dirY)
    const wy = sin * (len * dirX) + cos * (len * dirY)
    const [x0, y0] = toPx(0, 0)
    const [x1, y1] = toPx(wx, wy)
    ctx.strokeStyle = color
    ctx.lineWidth = 2.5
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.lineTo(x1, y1)
    ctx.stroke()
  }
  drawAxis(sx.value, 1, 0, '#f44336') // 长轴
  drawAxis(sy.value, 0, 1, '#4caf50') // 短轴

  // 5) 中心点
  const [cx0, cy0] = toPx(0, 0)
  ctx.fillStyle = '#212121'
  ctx.beginPath()
  ctx.arc(cx0, cy0, 3, 0, Math.PI * 2)
  ctx.fill()
}

function scheduleDraw() {
  cancelAnimationFrame(animationId)
  animationId = requestAnimationFrame(draw)
}

watch([sx, sy, theta], scheduleDraw)

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  devicePixelRatioCached = Math.min(window.devicePixelRatio || 1, 2)
  draw()
  if (canvasRef.value) {
    resizeObserver = new ResizeObserver(() => scheduleDraw())
    resizeObserver.observe(canvasRef.value)
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(animationId)
  resizeObserver?.disconnect()
})
</script>

<template>
  <section class="gaussian2d-playground" aria-label="二维高斯协方差交互演示">
    <div class="pg-head">
      <div>
        <h3>二维版：只有一个旋转角，看清 σxy 从哪来</h3>
        <p>拖动两条主轴宽度和唯一的旋转角，观察椭圆怎么歪、右侧 2×2 矩阵的非对角项 σxy 怎么从 0 变成非零。</p>
      </div>
      <div class="preset-row" role="group" aria-label="预设形状">
        <button v-for="p in presets" :key="p.label" type="button" @click="setPreset(p)">{{ p.label }}</button>
      </div>
    </div>
    <div class="pg-grid">
      <div class="canvas-wrap">
        <canvas ref="canvasRef" class="gaussian2d-canvas" aria-label="二维高斯浓度热力图与等值椭圆"></canvas>
        <div class="legend">
          <span><i class="dot red"></i>长轴 (σx 方向)</span>
          <span><i class="dot green"></i>短轴 (σy 方向)</span>
          <span><i class="dot blue"></i>1σ 等值椭圆</span>
        </div>
      </div>
      <div class="controls-panel">
        <div class="control-group">
          <h4>两条主轴的宽度</h4>
          <label>σx <input v-model.number="sx" type="range" min="0.2" max="2.4" step="0.01" /><output>{{ sx.toFixed(2) }}</output></label>
          <label>σy <input v-model.number="sy" type="range" min="0.2" max="2.4" step="0.01" /><output>{{ sy.toFixed(2) }}</output></label>
        </div>
        <div class="control-group">
          <h4>旋转角 θ（唯一的朝向参数）</h4>
          <label>θ <input v-model.number="theta" type="range" min="-90" max="90" step="1" /><output>{{ theta }}°</output></label>
        </div>
        <div class="matrix-block">
          <h4>当前协方差矩阵 Σ</h4>
          <div class="matrix2">
            <span class="cell">{{ covariance.a.toFixed(2) }}</span>
            <span class="cell" :class="{ hot: !offDiagNearZero }">{{ covariance.b.toFixed(2) }}</span>
            <span class="cell" :class="{ hot: !offDiagNearZero }">{{ covariance.b.toFixed(2) }}</span>
            <span class="cell">{{ covariance.d.toFixed(2) }}</span>
          </div>
          <p class="matrix-note">
            <template v-if="offDiagNearZero">
              θ = 0（或 ±90°），椭圆主轴和坐标轴对齐 → 非对角项 <b>σxy ≈ 0</b>，两个方向互不联动。
            </template>
            <template v-else>
              θ ≠ 0，椭圆歪了 → 非对角项 <b>σxy = {{ covariance.b.toFixed(2) }}</b> 变成非零。这个数就是「旋转」在坐标系里留下的痕迹。
            </template>
          </p>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.gaussian2d-playground{margin:1.5rem 0 2rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg-soft);padding:1.1rem}
.pg-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-bottom:1rem}
.pg-head h3{margin:0 0 .35rem;font-size:1.08rem}
.pg-head p{margin:0;color:var(--vp-c-text-2);font-size:.9rem}
.preset-row{display:flex;flex-wrap:wrap;gap:.45rem;justify-content:flex-end}
button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.38rem .62rem;border-radius:5px;cursor:pointer;white-space:nowrap;font-size:.82rem}
button:hover{background:var(--vp-c-brand-soft)}
.pg-grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(240px,.9fr);gap:1rem}
.canvas-wrap{display:flex;flex-direction:column;gap:.5rem}
.gaussian2d-canvas{width:100%;aspect-ratio:1/1;max-width:460px;border:1px solid var(--vp-c-divider);background:#f7f9fc;display:block}
.legend{display:flex;flex-wrap:wrap;gap:.8rem;font-size:.78rem;color:var(--vp-c-text-2)}
.legend .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.28rem;vertical-align:middle}
.dot.red{background:#f44336}.dot.green{background:#4caf50}.dot.blue{background:#1565c0}
.controls-panel{display:grid;gap:.8rem;align-content:start}
.control-group,.matrix-block{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);padding:.75rem .8rem}
.control-group h4,.matrix-block h4{margin:0 0 .6rem;font-size:.92rem}
label{display:grid;grid-template-columns:34px minmax(0,1fr) 48px;align-items:center;gap:.45rem;font-size:.84rem;margin:.48rem 0}
input[type=range]{width:100%;accent-color:var(--vp-c-brand-1)}
output{text-align:right;font-variant-numeric:tabular-nums;color:var(--vp-c-text-2)}
.matrix2{display:grid;grid-template-columns:1fr 1fr;gap:.3rem;max-width:180px;margin:0 auto}
.matrix2 .cell{text-align:center;padding:.5rem 0;background:var(--vp-c-bg-soft);font:.9rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--vp-c-divider);border-radius:4px}
.matrix2 .cell.hot{background:rgba(255,152,0,.18);border-color:#ff9800;color:var(--vp-c-text-1);font-weight:700}
.matrix-note{margin:.6rem 0 0;color:var(--vp-c-text-2);font-size:.78rem;line-height:1.5}
.matrix-note b{color:var(--vp-c-brand-1)}
@media(max-width:720px){.pg-head{display:block}.preset-row{justify-content:flex-start;margin-top:.75rem}.pg-grid{grid-template-columns:1fr}}
</style>
