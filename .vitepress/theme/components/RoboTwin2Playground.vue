<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'

// RoboTwin 2.0 互动演示：
//  标签 1 数据工厂闭环：模拟"写代码 → 仿真跑 → VLM 看 → 修代码"的迭代，成功率逐步上升
//  标签 2 域随机化：开关五个维度，场景随之变化
//  标签 3 体态感知抓取：切换 Franka / Piper，看抓取方式与成功率差异

type Tab = 'loop' | 'dr' | 'grasp'
const tab = ref<Tab>('loop')

// ---------------- 标签 1：数据工厂闭环 ----------------
interface Stage { emoji: string; name: string; detail: string }
const loopStages: Stage[] = [
  { emoji: '🧾', name: '代码生成智能体', detail: '给定任务名 + 自然语言描述 + API 列表 + 示例调用 + 分层约束，合成一段逐步的任务代码。' },
  { emoji: '🎬', name: '仿真执行 ×10', detail: '程序在仿真里跑 10 次以平均随机性，产出结构化执行日志（成功/失败 + 失败原因）。' },
  { emoji: '👁️', name: 'VLM 观察者', detail: '看全部 10 次录像，逐帧判定每步成败、定位失败点、诊断根因（逻辑 / API 用法 / 系统问题）。' },
  { emoji: '🔧', name: '代码修复', detail: '合并"定量日志 + 定性诊断"，改写易错指令，进入下一轮；连续 5 轮未达标则停止。' },
]

// 每一轮迭代后达到的 ASR（取自论文 RoboTwin 2.0 的收敛趋势）
const hist = [62.1, 66.7, 69.0, 71.3]
const iter = ref(0)
const activeStage = ref(-1)
const running = ref(false)
let timers: number[] = []
const asr = computed(() => (iter.value === 0 ? 62.1 : hist[Math.min(iter.value - 1, hist.length - 1)]))
const converged = computed(() => iter.value >= hist.length)

function clearTimers() {
  timers.forEach((t) => clearTimeout(t))
  timers = []
}
function runLoop() {
  if (running.value) return
  clearTimers()
  running.value = true
  activeStage.value = 0
  for (let s = 0; s < loopStages.length; s++) {
    timers.push(window.setTimeout(() => { activeStage.value = s }, s * 620))
  }
  timers.push(
    window.setTimeout(() => {
      activeStage.value = -1
      if (iter.value < hist.length) iter.value += 1
      running.value = false
    }, loopStages.length * 620),
  )
}
function resetLoop() {
  clearTimers()
  running.value = false
  activeStage.value = -1
  iter.value = 0
}
onBeforeUnmount(clearTimers)

// ---------------- 标签 2：五轴域随机化 ----------------
const dr = ref({ clutter: true, background: true, lighting: true, height: false, language: true })
const bgPool = ['#dfe7ef', '#e9dfe7', '#dfe9db', '#e7e0cf']
const bgIndex = ref(0)
const langPool = [
  'Use left arm to place can to the left of the pot',
  'Place the white sauce can to the left of the gray kitchenpot',
  'Put the can on the left side of the pot for cooking',
]
const langIndex = ref(0)
function toggle(key: keyof typeof dr.value) {
  dr.value[key] = !dr.value[key]
  if (key === 'background' && dr.value.background) bgIndex.value = (bgIndex.value + 1) % bgPool.length
  if (key === 'language' && dr.value.language) langIndex.value = (langIndex.value + 1) % langPool.length
}

// ---------------- 标签 3：体态感知抓取 ----------------
const robot = ref<'franka' | 'piper'>('franka')
const graspInfo = computed(() =>
  robot.value === 'franka'
    ? { dof: 7, style: '俯视精确抓取', r1: 67.3, r2: 67.2, note: '7-DoF，可达空间大，本来就灵活，加了候选抓取收益有限。' }
    : { dof: 6, style: '侧向抓取', r1: 2.4, r2: 25.1, note: '6-DoF，可达空间受限，靠侧向抓取；体态感知让它从"几乎不可用"变成"可用"。' },
)
</script>

<template>
  <section class="rt-pg" aria-label="RoboTwin 2.0 互动演示">
    <div class="pg-head">
      <h3>🏭 RoboTwin 2.0 长什么样：一座会自我纠错的数据工厂</h3>
      <p>它把"造数据"拆成三件事：让 AI 自动写任务代码并自我纠错、沿五个维度随机化场景、让抓取适配不同机械臂。逐个标签页点起来。</p>
      <div class="tabs">
        <button :class="{ on: tab === 'loop' }" @click="tab = 'loop'">① 数据工厂闭环</button>
        <button :class="{ on: tab === 'dr' }" @click="tab = 'dr'">② 五轴域随机化</button>
        <button :class="{ on: tab === 'grasp' }" @click="tab = 'grasp'">③ 体态感知抓取</button>
      </div>
    </div>

    <!-- 标签 1 -->
    <div v-if="tab === 'loop'" class="loop">
      <div class="loop-row">
        <div
          v-for="(s, i) in loopStages"
          :key="i"
          class="stage"
          :class="{ on: i === activeStage, done: i < activeStage }"
        >
          <div class="emoji">{{ s.emoji }}</div>
          <div class="name">{{ s.name }}</div>
          <div v-if="i < loopStages.length - 1" class="arrow">→</div>
        </div>
      </div>
      <div class="backedge">↺ 修复后的代码重新进入"代码生成"，循环迭代</div>

      <div class="loop-detail">
        <template v-if="activeStage >= 0">
          <b>{{ loopStages[activeStage].emoji }} {{ loopStages[activeStage].name }}：</b>{{ loopStages[activeStage].detail }}
        </template>
        <template v-else>
          点下面的按钮，模拟一次"生成 → 执行 → 观察 → 修复"的迭代。
        </template>
      </div>

      <div class="meter">
        <div class="meter-head">
          <span>任务平均成功率 ASR</span>
          <b>{{ asr.toFixed(1) }}%</b>
        </div>
        <div class="bar"><i :style="{ width: asr + '%' }"></i></div>
        <div class="meter-foot">
          迭代轮数：{{ iter }} / {{ hist.length }}
          <span v-if="converged" class="ok">✓ 达到成功率，输出专家级任务代码</span>
        </div>
      </div>

      <div class="btn-row">
        <button :disabled="running || converged" @click="runLoop">▶ 运行一轮迭代</button>
        <button @click="resetLoop">重置</button>
      </div>
      <p class="takeaway">
        关键在"双路反馈"：结构化日志告诉你"第几步失败"，VLM 诊断告诉你"为什么失败"。两者合并才能对症修改。论文报告，加多模态反馈后 ASR 从 47.4% 提升到 71.3%。
      </p>
    </div>

    <!-- 标签 2 -->
    <div v-else-if="tab === 'dr'" class="dr">
      <div class="dr-grid">
        <div class="scene">
          <svg viewBox="0 0 320 200" role="img" aria-label="随域随机化变化的场景示意">
            <!-- 背景 -->
            <rect x="0" y="0" width="320" height="200" :fill="dr.background ? bgPool[bgIndex] : '#eef2f6'" />
            <!-- 光照叠加 -->
            <rect v-if="dr.lighting" x="0" y="0" width="320" height="200" fill="#FFB74D" opacity="0.16" />
            <circle v-if="dr.lighting" cx="272" cy="30" r="14" fill="#FFC107" />
            <!-- 机械臂抽象 -->
            <rect x="24" y="60" width="10" height="70" rx="4" fill="#90A4AE" />
            <rect x="34" y="56" width="52" height="10" rx="5" fill="#90A4AE" />
            <rect x="286" y="60" width="10" height="70" rx="4" fill="#90A4AE" />
            <rect x="234" y="56" width="52" height="10" rx="5" fill="#90A4AE" />
            <!-- 桌面（高度可变） -->
            <rect x="16" :y="dr.height ? 150 : 128" width="288" height="12" rx="3" fill="#8D6E63" />
            <!-- 任务物体（罐头 + 锅） -->
            <rect x="138" :y="(dr.height ? 150 : 128) - 26" width="18" height="26" rx="3" fill="#E91E63" />
            <ellipse cx="182" :cy="(dr.height ? 150 : 128) - 8" rx="20" ry="8" fill="#546E7A" />
            <!-- 杂物干扰物 -->
            <template v-if="dr.clutter">
              <circle cx="70" :cy="(dr.height ? 150 : 128) - 8" r="8" fill="#4CAF50" />
              <circle cx="98" :cy="(dr.height ? 150 : 128) - 6" r="6" fill="#9C27B0" />
              <rect x="228" :y="(dr.height ? 150 : 128) - 16" width="12" height="16" rx="2" fill="#FF9800" />
              <circle cx="256" :cy="(dr.height ? 150 : 128) - 7" r="5" fill="#2196F3" />
            </template>
          </svg>
          <div class="lang" v-if="dr.language">🗣️ {{ langPool[langIndex] }}</div>
          <div class="lang muted" v-else>🗣️ （固定单一句式）</div>
        </div>
        <div class="switches">
          <label class="sw"><input type="checkbox" :checked="dr.clutter" @change="toggle('clutter')"><span>场景杂物 clutter</span></label>
          <label class="sw"><input type="checkbox" :checked="dr.background" @change="toggle('background')"><span>背景纹理 background</span></label>
          <label class="sw"><input type="checkbox" :checked="dr.lighting" @change="toggle('lighting')"><span>光照条件 lighting</span></label>
          <label class="sw"><input type="checkbox" :checked="dr.height" @change="toggle('height')"><span>台面高度 table height</span></label>
          <label class="sw"><input type="checkbox" :checked="dr.language" @change="toggle('language')"><span>语言指令 language</span></label>
          <p class="sw-note">勾选/取消每个维度，右边场景实时变化。五轴同时打开，就是在构造"又乱又暗、背景陌生、说法各异"的困难训练分布。</p>
        </div>
      </div>
      <p class="takeaway">
        域随机化不是"随便加噪声"，而是<b>结构化</b>地覆盖真实世界最相关的五种差异。论文显示：干净仿真数据几乎无助于鲁棒性，而域随机化预训练带来约 30% 的相对提升。
      </p>
    </div>

    <!-- 标签 3 -->
    <div v-else class="grasp">
      <div class="seg">
        <button :class="{ on: robot === 'franka' }" @click="robot = 'franka'">Franka（7-DoF）</button>
        <button :class="{ on: robot === 'piper' }" @click="robot = 'piper'">Piper（6-DoF）</button>
      </div>

      <div class="grasp-grid">
        <svg viewBox="0 0 220 160" class="grasp-svg" role="img" aria-label="不同机械臂的抓取方式">
          <rect x="20" y="120" width="180" height="10" rx="3" fill="#8D6E63" />
          <rect x="98" y="86" width="24" height="34" rx="4" fill="#E91E63" />
          <text x="132" y="108" font-size="9" fill="var(--vp-c-text-3)">can</text>
          <template v-if="robot === 'franka'">
            <!-- 俯视抓取：两个手指从上往下 -->
            <line x1="98" y1="30" x2="98" y2="80" stroke="#2196F3" stroke-width="4" marker-end="url(#ah)" />
            <line x1="122" y1="30" x2="122" y2="80" stroke="#2196F3" stroke-width="4" />
            <text x="60" y="24" font-size="10" fill="#2196F3">top-down</text>
          </template>
          <template v-else>
            <!-- 侧向抓取：从侧面伸向物体 -->
            <line x1="40" y1="103" x2="90" y2="103" stroke="#4CAF50" stroke-width="4" />
            <line x1="40" y1="112" x2="90" y2="112" stroke="#4CAF50" stroke-width="4" />
            <text x="24" y="92" font-size="10" fill="#4CAF50">side</text>
          </template>
          <defs>
            <marker id="ah" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill="#2196F3" />
            </marker>
          </defs>
        </svg>

        <div class="grasp-info">
          <div class="gi-row"><span>自由度</span><b>{{ graspInfo.dof }}</b></div>
          <div class="gi-row"><span>偏好抓取</span><b>{{ graspInfo.style }}</b></div>
          <div class="gi-row"><span>RoboTwin 1.0</span><b>{{ graspInfo.r1 }}%</b></div>
          <div class="gi-row"><span>RoboTwin 2.0</span><b class="hl">{{ graspInfo.r2 }}%</b></div>
          <p class="gi-note">{{ graspInfo.note }}</p>
        </div>
      </div>
      <p class="takeaway">
        同一句"抓起罐头"，Franka 俯视就行、Piper 必须侧向。体态感知抓取先生成覆盖多方向的候选抓取，再让每种机械臂各取所需——所以低自由度平台的提升远大于高自由度平台。
      </p>
    </div>
  </section>
</template>

<style scoped>
.rt-pg{margin:1.5rem 0 2rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg-soft);border-radius:10px;padding:1.1rem 1.2rem}
.pg-head h3{margin:0 0 .4rem;font-size:1.1rem}
.pg-head p{margin:0 0 .8rem;color:var(--vp-c-text-2);font-size:.9rem;line-height:1.6}
.tabs{display:flex;gap:.5rem;flex-wrap:wrap}
.tabs button{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);color:var(--vp-c-text-1);padding:.42rem .8rem;border-radius:8px;cursor:pointer;font-size:.85rem}
.tabs button.on{border-color:var(--vp-c-brand-1);background:var(--vp-c-brand-soft);color:var(--vp-c-brand-1);font-weight:600}

.loop-row{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin-top:1rem}
.stage{position:relative;flex:1 1 130px;min-width:120px;border:2px solid var(--vp-c-divider);background:var(--vp-c-bg);border-radius:12px;padding:.6rem .5rem;text-align:center;transition:.2s}
.stage.on{border-color:var(--vp-c-brand-1);background:var(--vp-c-brand-soft);box-shadow:0 0 0 4px rgba(33,150,243,.18);transform:translateY(-2px)}
.stage.done{border-color:#4CAF50}
.stage .emoji{font-size:1.5rem}
.stage .name{font-size:.78rem;font-weight:600;margin-top:.2rem}
.stage .arrow{position:absolute;right:-12px;top:50%;transform:translateY(-50%);color:var(--vp-c-text-3);font-size:.9rem}
.backedge{margin:.5rem 0 0;font-size:.76rem;color:#9C27B0}
.loop-detail{margin:.7rem 0;font-size:.85rem;line-height:1.6;color:var(--vp-c-text-1);min-height:2.6em;background:var(--vp-c-bg);border:1px solid var(--vp-c-divider);border-radius:8px;padding:.6rem .75rem}
.meter{margin:.6rem 0}
.meter-head{display:flex;justify-content:space-between;font-size:.85rem}
.meter-head b{color:var(--vp-c-brand-1)}
.bar{height:14px;background:var(--vp-c-bg);border:1px solid var(--vp-c-divider);border-radius:999px;overflow:hidden;margin:.35rem 0}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#4CAF50,#2196F3);transition:width .5s}
.meter-foot{font-size:.76rem;color:var(--vp-c-text-3);display:flex;justify-content:space-between}
.meter-foot .ok{color:#2E7D32;font-weight:600}
.btn-row{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.4rem}
.btn-row button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.4rem .8rem;border-radius:8px;cursor:pointer;font-size:.83rem}
.btn-row button:disabled{opacity:.45;cursor:not-allowed}

.dr-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(220px,.8fr);gap:1rem;margin-top:1rem}
.scene svg{width:100%;border:1px solid var(--vp-c-divider);border-radius:8px;background:#fff}
.lang{margin-top:.5rem;font-size:.78rem;background:var(--vp-c-bg);border:1px solid var(--vp-c-divider);border-radius:8px;padding:.4rem .6rem}
.lang.muted{color:var(--vp-c-text-3)}
.switches{display:grid;gap:.5rem;align-content:start}
.sw{display:flex;align-items:center;gap:.5rem;font-size:.84rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);border-radius:8px;padding:.45rem .6rem;cursor:pointer}
.sw input{accent-color:var(--vp-c-brand-1);width:16px;height:16px}
.sw-note{margin:.2rem 0 0;font-size:.75rem;color:var(--vp-c-text-3);line-height:1.55}

.seg{display:flex;gap:.5rem;margin-top:1rem}
.seg button{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);color:var(--vp-c-text-1);padding:.4rem .9rem;border-radius:8px;cursor:pointer;font-size:.85rem}
.seg button.on{border-color:var(--vp-c-brand-1);background:var(--vp-c-brand-soft);color:var(--vp-c-brand-1);font-weight:600}
.grasp-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(200px,.9fr);gap:1rem;margin-top:.8rem;align-items:center}
.grasp-svg{width:100%;max-width:260px;border:1px solid var(--vp-c-divider);border-radius:8px;background:var(--vp-c-bg)}
.grasp-info{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);border-radius:8px;padding:.6rem .75rem}
.gi-row{display:flex;justify-content:space-between;font-size:.84rem;padding:.28rem 0;border-bottom:1px dashed var(--vp-c-divider)}
.gi-row b{color:var(--vp-c-brand-1)}
.gi-row b.hl{color:#2E7D32}
.gi-note{margin:.55rem 0 0;font-size:.76rem;color:var(--vp-c-text-2);line-height:1.55}
.takeaway{margin-top:.9rem;font-size:.82rem;line-height:1.7;color:var(--vp-c-text-2);background:var(--vp-c-bg);border-radius:8px;padding:.7rem .85rem;border:1px solid var(--vp-c-divider)}
@media(max-width:720px){.dr-grid,.grasp-grid{grid-template-columns:1fr}}
</style>
