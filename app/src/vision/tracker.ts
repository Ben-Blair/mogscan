import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { METRIC_NAMES, type Metrics } from '../scoring/metrics'

export interface Box { x: number; y: number; w: number; h: number }

export interface Track {
  id: number
  box: Box
  landmarks: NormalizedLandmark[]
  yaw: number
  pitch: number
  frontal: boolean
  lastSeen: number
  /** Exponentially smoothed metrics (only updated while frontal). */
  metrics: Metrics | null
  frontalFrames: number
  cnn: number | null
  pMale: number | null
  lastCnn: number
}

const METRIC_ALPHA = 0.15
const CNN_ALPHA = 0.3
const TRACK_TTL_MS = 1000
// Apparent pose from Procrustes alignment, which reads ~1.2x MediaPipe's matrix angles.
export const MAX_YAW = 14
export const MAX_PITCH = 20

function iou(a: Box, b: Box) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y)
  const x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h)
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0)
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9)
}

export function landmarkBox(lm: NormalizedLandmark[], w: number, h: number): Box {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const p of lm) {
    x0 = Math.min(x0, p.x * w); x1 = Math.max(x1, p.x * w)
    y0 = Math.min(y0, p.y * h); y1 = Math.max(y1, p.y * h)
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
}

export interface Detection { landmarks: NormalizedLandmark[]; box: Box; yaw: number; pitch: number; metrics: Metrics }

/** Greedy IoU tracker that keeps per-person identity and smoothed measurements. */
export class FaceTracker {
  tracks: Track[] = []
  private nextId = 1

  update(dets: Detection[], now: number): Track[] {
    const pairs: [number, number, number][] = []
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => pairs.push([iou(t.box, d.box), ti, di])))
    pairs.sort((a, b) => b[0] - a[0])
    const usedT = new Set<number>(), usedD = new Set<number>()
    for (const [score, ti, di] of pairs) {
      if (score < 0.25 || usedT.has(ti) || usedD.has(di)) continue
      usedT.add(ti); usedD.add(di)
      this.apply(this.tracks[ti], dets[di], now)
    }
    dets.forEach((d, di) => {
      if (usedD.has(di)) return
      const t: Track = { id: this.nextId++, box: d.box, landmarks: d.landmarks, yaw: d.yaw, pitch: d.pitch,
        frontal: false, lastSeen: now, metrics: null, frontalFrames: 0, cnn: null, pMale: null, lastCnn: 0 }
      this.apply(t, d, now)
      this.tracks.push(t)
    })
    this.tracks = this.tracks.filter((t) => now - t.lastSeen < TRACK_TTL_MS)
    return this.tracks.filter((t) => t.lastSeen === now)
  }

  private apply(t: Track, d: Detection, now: number) {
    t.box = d.box; t.landmarks = d.landmarks; t.yaw = d.yaw; t.pitch = d.pitch; t.lastSeen = now
    t.frontal = Math.abs(d.yaw) < MAX_YAW && Math.abs(d.pitch) < MAX_PITCH
    if (!t.frontal) return
    t.frontalFrames++
    if (!t.metrics) { t.metrics = { ...d.metrics }; return }
    for (const k of METRIC_NAMES) t.metrics[k] += METRIC_ALPHA * (d.metrics[k] - t.metrics[k])
  }

  setCnn(t: Track, score: number, pMale: number, now: number) {
    t.cnn = t.cnn === null ? score : t.cnn + CNN_ALPHA * (score - t.cnn)
    t.pMale = t.pMale === null ? pMale : t.pMale + CNN_ALPHA * (pMale - t.pMale)
    t.lastCnn = now
  }

  /** The visible, frontal track that has waited longest for a CNN update. */
  nextForCnn(now: number): Track | undefined {
    return this.tracks.filter((t) => t.lastSeen === now && t.frontal).sort((a, b) => a.lastCnn - b.lastCnn)[0]
  }
}
