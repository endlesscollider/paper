<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

type Preset = { label: string; values: [number, number, number, number, number, number] }
const canvasHost = ref<HTMLDivElement | null>(null)
const sx = ref(1.45), sy = ref(0.68), sz = ref(0.28)
const rx = ref(-18), ry = ref(0), rz = ref(28)
const presets: Preset[] = [
  { label: '球形', values: [0.9, 0.9, 0.9, 0, 0, 0] },
  { label: '表面贴片', values: [1.5, 0.8, 0.16, -18, 0, 28] },
  { label: '细长条', values: [1.8, 0.3, 0.3, 0, 0, 42] },
]

let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let renderer: THREE.WebGLRenderer | null = null
let controls: OrbitControls | null = null
let ellipsoid: THREE.Mesh | null = null
let axes: THREE.Group | null = null
let animationId = 0
let resizeObserver: ResizeObserver | null = null

const covariance = computed(() => {
  const euler = new THREE.Euler(THREE.MathUtils.degToRad(rx.value), THREE.MathUtils.degToRad(ry.value), THREE.MathUtils.degToRad(rz.value), 'XYZ')
  const rotation = new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromEuler(euler))
  const diagonal = new THREE.Matrix3().set(sx.value ** 2, 0, 0, 0, sy.value ** 2, 0, 0, 0, sz.value ** 2)
  const result = rotation.clone().multiply(diagonal).multiply(rotation.clone().transpose())
  return [[result.elements[0], result.elements[3], result.elements[6]], [result.elements[1], result.elements[4], result.elements[7]], [result.elements[2], result.elements[5], result.elements[8]]]
})

function setPreset(preset: Preset) {
  ;[sx.value, sy.value, sz.value, rx.value, ry.value, rz.value] = preset.values
  updateShape()
}

function updateShape() {
  if (!ellipsoid || !axes) return
  ellipsoid.scale.set(sx.value, sy.value, sz.value)
  const rotation = [THREE.MathUtils.degToRad(rx.value), THREE.MathUtils.degToRad(ry.value), THREE.MathUtils.degToRad(rz.value)]
  ellipsoid.rotation.set(...rotation)
  axes.rotation.set(...rotation)
  axes.clear()
  const lengths = [sx.value, sy.value, sz.value]
  const colors = [0xf44336, 0x4caf50, 0xff9800]
  const directions = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)]
  lengths.forEach((length, index) => {
    const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-length, 0, 0), new THREE.Vector3(length, 0, 0)])
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: colors[index] }))
    line.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), directions[index])
    axes?.add(line)
  })
}

function initScene() {
  if (!canvasHost.value) return
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0xf7f9fc)
  camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100)
  // 留出最大尺度（2.2）和旋转后的椭球边界，避免拖到极值时被裁掉。
  camera.position.set(5.2, 3.8, 6.4)
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(640, 420)
  renderer.outputColorSpace = THREE.SRGBColorSpace
  canvasHost.value.appendChild(renderer.domElement)
  scene.add(new THREE.HemisphereLight(0xffffff, 0x607d8b, 2.4))
  const key = new THREE.DirectionalLight(0xffffff, 2.2)
  key.position.set(3, 4, 5); scene.add(key)
  scene.add(new THREE.AxesHelper(2.1))
  scene.add(new THREE.GridHelper(4.5, 9, 0xc8d0d8, 0xe1e6eb))
  ellipsoid = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), new THREE.MeshPhysicalMaterial({ color: 0x2196f3, transparent: true, opacity: 0.7, roughness: 0.32 }))
  scene.add(ellipsoid)
  axes = new THREE.Group(); scene.add(axes); updateShape()
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true; controls.target.set(0, 0, 0)
  resizeObserver = new ResizeObserver(() => {
    if (!canvasHost.value || !camera || !renderer) return
    const { width, height } = canvasHost.value.getBoundingClientRect()
    camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix(); renderer.setSize(width, height, false)
  })
  resizeObserver.observe(canvasHost.value)
  const tick = () => { if (!scene || !camera || !renderer) return; controls?.update(); renderer.render(scene, camera); animationId = requestAnimationFrame(tick) }
  tick()
}

onMounted(initScene)
onBeforeUnmount(() => { cancelAnimationFrame(animationId); resizeObserver?.disconnect(); controls?.dispose(); renderer?.dispose(); ellipsoid?.geometry.dispose(); (ellipsoid?.material as THREE.Material | undefined)?.dispose() })
</script>

<template>
  <section class="gaussian-playground" aria-label="3D 高斯协方差交互演示">
    <div class="playground-head">
      <div><h3>拖动参数，看协方差如何改变高斯椭球</h3><p>拖动左侧视角；右侧调节尺度和旋转角。矩阵会同步更新。</p></div>
      <div class="preset-row" role="group" aria-label="预设形状"><button v-for="preset in presets" :key="preset.label" type="button" @click="setPreset(preset)">{{ preset.label }}</button></div>
    </div>
    <div class="playground-grid">
      <div ref="canvasHost" class="gaussian-canvas" aria-label="可旋转的三维高斯椭球"></div>
      <div class="controls-panel">
        <div class="control-group"><h4>三个方向的尺度</h4>
          <label>sx <input v-model.number="sx" type="range" min="0.1" max="2.2" step="0.01" @input="updateShape" /><output>{{ sx.toFixed(2) }}</output></label>
          <label>sy <input v-model.number="sy" type="range" min="0.1" max="2.2" step="0.01" @input="updateShape" /><output>{{ sy.toFixed(2) }}</output></label>
          <label>sz <input v-model.number="sz" type="range" min="0.1" max="2.2" step="0.01" @input="updateShape" /><output>{{ sz.toFixed(2) }}</output></label>
        </div>
        <div class="control-group"><h4>整体旋转角度</h4>
          <label>绕 X <input v-model.number="rx" type="range" min="-90" max="90" step="1" @input="updateShape" /><output>{{ rx }}°</output></label>
          <label>绕 Y <input v-model.number="ry" type="range" min="-90" max="90" step="1" @input="updateShape" /><output>{{ ry }}°</output></label>
          <label>绕 Z <input v-model.number="rz" type="range" min="-180" max="180" step="1" @input="updateShape" /><output>{{ rz }}°</output></label>
        </div>
        <div class="matrix-block"><h4>当前协方差矩阵 Σ</h4><div class="matrix"><span v-for="(row, rowIndex) in covariance" :key="rowIndex">{{ row.map(value => value.toFixed(2)).join('   ') }}</span></div><p class="matrix-note">对角线主要反映宽度；非对角线让主轴倾斜。3DGS 优化尺度和旋转，再合成矩阵。</p></div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.gaussian-playground{margin:1.5rem 0 2rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg-soft);padding:1.1rem}.playground-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-bottom:1rem}.playground-head h3{margin:0 0 .35rem;font-size:1.08rem}.playground-head p{margin:0;color:var(--vp-c-text-2);font-size:.9rem}.preset-row{display:flex;flex-wrap:wrap;gap:.45rem;justify-content:flex-end}button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.38rem .62rem;border-radius:5px;cursor:pointer;white-space:nowrap}button:hover{background:var(--vp-c-brand-soft)}.playground-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(250px,.85fr);gap:1rem}.gaussian-canvas{min-height:360px;height:min(52vw,420px);overflow:hidden;border:1px solid var(--vp-c-divider);background:#f7f9fc}.gaussian-canvas :deep(canvas){display:block;width:100%;height:100%}.controls-panel{display:grid;gap:.8rem;align-content:start}.control-group,.matrix-block{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);padding:.75rem .8rem}.control-group h4,.matrix-block h4{margin:0 0 .6rem;font-size:.92rem}label{display:grid;grid-template-columns:44px minmax(0,1fr) 44px;align-items:center;gap:.45rem;font-size:.84rem;margin:.48rem 0}input[type=range]{width:100%;accent-color:var(--vp-c-brand-1)}output{text-align:right;font-variant-numeric:tabular-nums;color:var(--vp-c-text-2)}.matrix{display:grid;gap:.22rem;padding:.6rem .5rem;background:var(--vp-c-bg-soft);font:.82rem/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;overflow-x:auto}.matrix-note{margin:.55rem 0 0;color:var(--vp-c-text-2);font-size:.78rem;line-height:1.45}@media(max-width:720px){.playground-head{display:block}.preset-row{justify-content:flex-start;margin-top:.75rem}.playground-grid{grid-template-columns:1fr}.gaussian-canvas{height:320px;min-height:280px}}
</style>
