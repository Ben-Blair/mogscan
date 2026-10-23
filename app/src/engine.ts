import type { FaceLandmarker } from '@mediapipe/tasks-vision'
import { loadMogNet, type MogNet } from './model/mognet'
import { analyzeFace, cropBox, type Vec3 } from './scoring/metrics'
import { rate, type FusionModel, type GeoModel, type Rating, type Sex } from './scoring/scoring'
import { drawFrame, type DrawOptions } from './ui/draw'
import { createLandmarker } from './vision/landmarker'
import { FaceTracker, landmarkBox, type Detection, type Track } from './vision/tracker'

const BASE = import.meta.env.BASE_URL
const CNN_INTERVAL_MS = 120
const MIN_FRONTAL_FRAMES = 8

export interface PersonSnapshot {
  id: number
  frontal: boolean
  yaw: number
  sex: Sex
  sexAuto: boolean
  rating: Rating | null
}

export interface Models { landmarker: FaceLandmarker; mognet: MogNet; geo: GeoModel; fusion: FusionModel; template: Vec3[] }

export async function loadModels(onStatus: (s: string) => void): Promise<Models> {
  const json = (f: string) => fetch(`${BASE}models/${f}`).then((r) => r.json())
  onStatus('Loading face landmarker…')
  const landmarker = await createLandmarker()
  onStatus('Loading MogNet…')
  const [mognet, geo, fusion, template] = await Promise.all([
    loadMogNet(), json('geometric.json'), json('fusion.json'), json('face_template.json')])
  return { landmarker, mognet, geo, fusion, template }
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

    const dets: Detection[] = res.faceLandmarks.map((lm) => {
      const { metrics, pose } = analyzeFace(lm, w, h, this.models.template)
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
    this.models.mognet.predict(src, box.x, box.y, box.side)
      .then((p) => this.tracker.setCnn(t, p.score, p.pMale, now))
      .catch((e) => console.error('[MogScan] CNN failed', e))
      .finally(() => { this.cnnBusy = false })
  }

  sexFor(t: Track): { sex: Sex; auto: boolean } {
    const o = this.sexOverride.get(t.id)
    if (o) return { sex: o, auto: false }
    return { sex: (t.pMale ?? 0.5) >= 0.5 ? 'M' : 'F', auto: true }
  }

  private rateTrack(t: Track): Rating | null {
    if (!t.metrics || t.frontalFrames < MIN_FRONTAL_FRAMES || t.cnn === null) return null
    return rate(t.metrics, t.cnn, this.sexFor(t).sex, this.models.geo, this.models.fusion)
  }

  snapshot(): PersonSnapshot[] {
    const now = this.lastTs
    return this.tracker.tracks.filter((t) => t.lastSeen === now).map((t) => {
      const { sex, auto } = this.sexFor(t)
      return { id: t.id, frontal: t.frontal, yaw: t.yaw, sex, sexAuto: auto, rating: this.ratings.get(t.id) ?? null }
    }).sort((a, b) => (b.rating?.psl ?? 0) - (a.rating?.psl ?? 0))
  }
}
