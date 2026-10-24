import type { FaceLandmarker } from '@mediapipe/tasks-vision'
import { loadMogNet, type MogNet } from './model/mognet'
import { APPEAR_SIZE, appearanceCrop, computeAppearance } from './scoring/appearance'
import { analyzeFace, cropBox, type ShapeModel, type Vec3 } from './scoring/metrics'
import { glowUpPlan, type GlowUpPlan } from './scoring/glowup'
import { rate, type FusionModel, type GeoModel, type Rating, type Sex } from './scoring/scoring'
import { drawFrame, type DrawOptions } from './ui/draw'
import { createLandmarker } from './vision/landmarker'
import { FaceTracker, landmarkBox, scanProgress, type Detection, type Track } from './vision/tracker'

const BASE = import.meta.env.BASE_URL
const CNN_INTERVAL_MS = 120

export interface PersonSnapshot {
  id: number
  frontal: boolean
  /** Scan progress 0–1; the rating appears at 1 and then stays fixed until a rescan. */
  progress: number
  yaw: number
  sex: Sex
  sexAuto: boolean
  rating: Rating | null
  /** Suggested actions with predicted scores (only once the scan has locked). */
  plan: GlowUpPlan | null
}

export interface Models {
  landmarker: FaceLandmarker; mognet: MogNet; geo: GeoModel; fusion: FusionModel; template: Vec3[]; shape: ShapeModel
}

export async function loadModels(onStatus: (s: string) => void): Promise<Models> {
  const json = (f: string) => fetch(`${BASE}models/${f}`).then((r) => r.json())
  onStatus('Loading face landmarker…')
  const landmarker = await createLandmarker()
  onStatus('Loading MogNet…')
  const [mognet, geo, fusion, template, shape] = await Promise.all([
    loadMogNet(), json('geometric.json'), json('fusion.json'), json('face_template.json'), json('shape.json')])
  return { landmarker, mognet, geo, fusion, template, shape }
}

/** Owns the per-frame loop: landmarks -> tracking -> metrics -> throttled CNN -> rating -> draw. */
export class Engine {
  tracker = new FaceTracker()
  ratings = new Map<number, Rating>()
  sexOverride = new Map<number, Sex>()
  fps = 0
  private raf = 0
  private lastTs = -1
  private cnnBusy = false
  private lastCnnRun = 0
  private frameTimes: number[] = []
  private appearCtx = new OffscreenCanvas(APPEAR_SIZE, APPEAR_SIZE).getContext('2d', { willReadFrequently: true })!

  private models: Models
  private canvas: HTMLCanvasElement
  private getSource: () => HTMLVideoElement | HTMLImageElement | null
  opts: DrawOptions

  constructor(models: Models, canvas: HTMLCanvasElement,
    getSource: () => HTMLVideoElement | HTMLImageElement | null, opts: DrawOptions) {
    this.models = models
    this.canvas = canvas
    this.getSource = getSource
    this.opts = opts
  }

  start() {
    const loop = () => { this.tick(); this.raf = requestAnimationFrame(loop) }
    this.raf = requestAnimationFrame(loop)
  }

  stop() { cancelAnimationFrame(this.raf) }

  private tick() {
    const src = this.getSource()
    if (!src) return
    const isVideo = src instanceof HTMLVideoElement
    const w = isVideo ? src.videoWidth : src.naturalWidth
    const h = isVideo ? src.videoHeight : src.naturalHeight
    if (!w || !h || (isVideo && src.readyState < 2)) return
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h }

    let now = performance.now()
    if (now <= this.lastTs) now = this.lastTs + 1
    this.lastTs = now
    const res = this.models.landmarker.detectForVideo(src, now)

    const dets: Detection[] = res.faceLandmarks.map((lm, i) => {
      const { metrics, pose } = analyzeFace(lm, w, h, this.models.template, this.models.shape)
      for (const c of res.faceBlendshapes?.[i]?.categories ?? []) {
        if (c.categoryName !== '_neutral') metrics[`bs_${c.categoryName}`] = c.score
      }
      return { landmarks: lm, box: landmarkBox(lm, w, h), yaw: pose.yaw, pitch: pose.pitch, metrics }
    })
    const visible = this.tracker.update(dets, now)

    if (!this.cnnBusy && now - this.lastCnnRun > CNN_INTERVAL_MS) {
      const t = this.tracker.nextForCnn(now)
      if (t) this.runCnn(t, src, w, h, now)
    }

    for (const t of visible) {
      const r = this.rateTrack(t)
      if (r) this.ratings.set(t.id, r)
      else this.ratings.delete(t.id)
    }

    drawFrame(this.canvas.getContext('2d')!, src, w, h, visible, this.ratings, this.opts)

    this.frameTimes.push(now)
    while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift()
    this.fps = this.frameTimes.length
  }

  private runCnn(t: Track, src: CanvasImageSource, w: number, h: number, now: number) {
    this.cnnBusy = true
    this.lastCnnRun = now
    const box = cropBox(t.landmarks, w, h)
    const appearance = this.sampleAppearance(src, w, h, t, box)
    this.models.mognet.predict(src, box.x, box.y, box.side)
      .then((p) => this.tracker.addCrop(t, appearance, p.score, p.pMale, now))
      .catch((e) => console.error('[MogScan] CNN failed', e))
      .finally(() => { this.cnnBusy = false })
  }

  /** Skin / colouring / hair / leanness pixels, on the same crop as training (appearance_crop). */
  private sampleAppearance(src: CanvasImageSource, w: number, h: number, t: Track,
    box: { x: number; y: number; side: number }) {
    const c = appearanceCrop(t.landmarks, w, h, box)
    const ctx = this.appearCtx
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, APPEAR_SIZE, APPEAR_SIZE)
    ctx.drawImage(src, c.x, c.y, c.side, c.side, 0, 0, APPEAR_SIZE, APPEAR_SIZE)
    const px = ctx.getImageData(0, 0, APPEAR_SIZE, APPEAR_SIZE).data
    return computeAppearance({ data: px, size: APPEAR_SIZE }, c.pts)
  }

  rescan(id: number) {
    this.tracker.rescan(id)
    this.ratings.delete(id)
  }

  sexFor(t: Track): { sex: Sex; auto: boolean } {
    const o = this.sexOverride.get(t.id)
    if (o) return { sex: o, auto: false }
    return { sex: (t.locked?.pMale ?? 0.5) >= 0.5 ? 'M' : 'F', auto: true }
  }

  private rateTrack(t: Track): Rating | null {
    if (!t.locked) return null
    return rate(t.locked.features, t.locked.cnn, this.sexFor(t).sex, this.models.geo, this.models.fusion)
  }

  snapshot(): PersonSnapshot[] {
    const now = this.lastTs
    return this.tracker.tracks.filter((t) => t.lastSeen === now).map((t) => {
      const { sex, auto } = this.sexFor(t)
      const rating = this.ratings.get(t.id) ?? null
      const plan = rating ? glowUpPlan(rating, sex, this.models.geo, this.models.fusion) : null
      return { id: t.id, frontal: t.frontal, progress: scanProgress(t), yaw: t.yaw, sex, sexAuto: auto, rating, plan }
    }).sort((a, b) => a.id - b.id)
  }
}
