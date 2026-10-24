import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import type { Features } from '../scoring/metrics'
import { OneEuro } from './oneEuro'

export interface Box { x: number; y: number; w: number; h: number }

export interface Track {
  id: number
  box: Box
  /** Smoothed box for drawing only (One Euro filtered), so the on-screen frame doesn't jitter. */
  displayBox: Box
  boxFilters: OneEuro[]
  landmarks: NormalizedLandmark[]
  yaw: number
  pitch: number
  frontal: boolean
  lastSeen: number
  /** Per-frame landmark features (geometry, shape, expression), collected while frontal during a scan. */
  frames: Features[]
  /** Pixel features + CNN outputs, sampled together about every 120 ms during a scan. */
  crops: { appearance: Features; cnn: number; pMale: number }[]
  /** The scan result: per-feature medians over the scan. Fixed until a rescan, so the score doesn't jitter. */
  locked: { features: Features; cnn: number; pMale: number } | null
  lastCnn: number
}

/** A scan collects this many frontal frames (~2 s) and crop samples before the result locks. */
export const SCAN_FRAMES = 60
export const SCAN_CROPS = 12
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

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

/** Per-key median over a list of feature records. */
function medianFeatures(rows: Features[]): Features {
  const out: Features = {}
  for (const k of Object.keys(rows[0])) out[k] = median(rows.map((r) => r[k]).filter((v) => Number.isFinite(v)))
  return out
}

/** Scan progress 0–1 (1 = locked). */
export function scanProgress(t: Track): number {
  if (t.locked) return 1
  return Math.min(t.frames.length / SCAN_FRAMES, t.crops.length / SCAN_CROPS)
}

export interface Detection { landmarks: NormalizedLandmark[]; box: Box; yaw: number; pitch: number; metrics: Features }

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
      const t: Track = { id: this.nextId++, box: d.box, displayBox: d.box,
        boxFilters: [0, 1, 2, 3].map(() => new OneEuro(0.8, 0.015)), landmarks: d.landmarks, yaw: d.yaw, pitch: d.pitch,
        frontal: false, lastSeen: now, frames: [], crops: [], locked: null, lastCnn: 0 }
      this.apply(t, d, now)
      this.tracks.push(t)
    })
    this.tracks = this.tracks.filter((t) => now - t.lastSeen < TRACK_TTL_MS)
    return this.tracks.filter((t) => t.lastSeen === now)
  }

  private apply(t: Track, d: Detection, now: number) {
    t.box = d.box; t.landmarks = d.landmarks; t.yaw = d.yaw; t.pitch = d.pitch; t.lastSeen = now
    const [fx, fy, fw, fh] = t.boxFilters
    t.displayBox = { x: fx.filter(d.box.x, now), y: fy.filter(d.box.y, now), w: fw.filter(d.box.w, now), h: fh.filter(d.box.h, now) }
    t.frontal = Math.abs(d.yaw) < MAX_YAW && Math.abs(d.pitch) < MAX_PITCH
    if (!t.frontal || t.locked) return
    if (t.frames.length < SCAN_FRAMES) t.frames.push(d.metrics)
    this.tryLock(t)
  }

  addCrop(t: Track, appearance: Features, cnn: number, pMale: number, now: number) {
    t.lastCnn = now
    if (t.locked) return
    if (t.crops.length < SCAN_CROPS) t.crops.push({ appearance, cnn, pMale })
    this.tryLock(t)
  }

  private tryLock(t: Track) {
    if (t.frames.length < SCAN_FRAMES || t.crops.length < SCAN_CROPS) return
    t.locked = {
      features: { ...medianFeatures(t.frames), ...medianFeatures(t.crops.map((c) => c.appearance)) },
      cnn: median(t.crops.map((c) => c.cnn)),
      pMale: median(t.crops.map((c) => c.pMale)),
    }
    t.frames = []
    t.crops = []
  }

  /** Throw away the result and scan this person again. */
  rescan(id: number) {
    const t = this.tracks.find((x) => x.id === id)
    if (t) { t.locked = null; t.frames = []; t.crops = [] }
  }

  /** The visible, frontal, still-scanning track that has waited longest for a crop sample. */
  nextForCnn(now: number): Track | undefined {
    return this.tracks.filter((t) => t.lastSeen === now && t.frontal && !t.locked)
      .sort((a, b) => a.lastCnn - b.lastCnn)[0]
  }
}
