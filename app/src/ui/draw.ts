import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { IDX } from '../scoring/metrics'
import type { Rating } from '../scoring/scoring'
import type { Track } from '../vision/tracker'

export interface DrawOptions { mirror: boolean; showMesh: boolean; showGuides: boolean }

const ACCENT = '#c8ff3d'
const WARN = '#ff5c7a'
const DIM = 'rgba(255,255,255,0.55)'

export function tierColor(psl: number) {
  if (psl >= 6) return '#ffd54a'
  if (psl >= 5.25) return '#c8ff3d'
  if (psl >= 4.5) return '#5ee6ff'
  if (psl >= 3.75) return '#e6e6e6'
  return '#ff8a5c'
}

export function drawFrame(ctx: CanvasRenderingContext2D, source: CanvasImageSource, w: number, h: number,
  tracks: Track[], ratings: Map<number, Rating>, opts: DrawOptions) {
  const X = (x: number) => (opts.mirror ? w - x : x)
  ctx.save()
  if (opts.mirror) { ctx.translate(w, 0); ctx.scale(-1, 1) }
  ctx.drawImage(source, 0, 0, w, h)
  ctx.restore()

  const unit = Math.max(1, w / 1280)
  for (const t of tracks) {
    const lm = t.landmarks
    const P = (i: number) => [X(lm[i].x * w), lm[i].y * h] as const
    const r = ratings.get(t.id)

    if (opts.showMesh) drawMesh(ctx, lm, w, h, X, unit)
    if (opts.showGuides && t.frontal) drawGuides(ctx, P, unit)

    // Corner-bracket box
    const bx = opts.mirror ? w - t.box.x - t.box.w : t.box.x
    const pad = t.box.w * 0.08
    const x0 = bx - pad, y0 = t.box.y - pad * 1.6, bw = t.box.w + 2 * pad, bh = t.box.h + 2.6 * pad
    const color = r ? tierColor(r.psl) : DIM
    ctx.strokeStyle = color
    ctx.lineWidth = 3 * unit
    const c = Math.min(bw, bh) * 0.18
    ctx.beginPath()
    for (const [cx, cy, dx, dy] of [[x0, y0, 1, 1], [x0 + bw, y0, -1, 1], [x0, y0 + bh, 1, -1], [x0 + bw, y0 + bh, -1, -1]]) {
      ctx.moveTo(cx + dx * c, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * c)
    }
    ctx.stroke()

    // Label above the box
    const fs = Math.max(14, bw * 0.11)
    ctx.textBaseline = 'bottom'
    if (!t.frontal) {
      label(ctx, x0, y0 - 6 * unit, `P${t.id} · face the camera`, fs * 0.6, DIM)
    } else if (!r) {
      label(ctx, x0, y0 - 6 * unit, `P${t.id} · scanning…`, fs * 0.6, DIM)
    } else {
      ctx.font = `800 ${fs}px "Space Grotesk", system-ui, sans-serif`
      const psl = r.psl.toFixed(2)
      const tw = ctx.measureText(psl).width
      ctx.fillStyle = 'rgba(8,8,12,0.78)'
      ctx.fillRect(x0, y0 - fs * 1.55, Math.max(bw, tw + fs * 4.2), fs * 1.45)
      ctx.fillStyle = color
      ctx.fillText(psl, x0 + fs * 0.25, y0 - fs * 0.2)
      ctx.font = `700 ${fs * 0.5}px "JetBrains Mono", ui-monospace, monospace`
      ctx.fillStyle = '#fff'
      ctx.fillText(`P${t.id}  ${r.tier.toUpperCase()}`, x0 + tw + fs * 0.55, y0 - fs * 0.72)
      ctx.fillStyle = DIM
      ctx.fillText(`PSL · top ${Math.max(0.1, 100 - r.percentile).toFixed(1)}%`, x0 + tw + fs * 0.55, y0 - fs * 0.2)
    }
  }
}

function label(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, fs: number, color: string) {
  ctx.font = `600 ${fs}px "JetBrains Mono", ui-monospace, monospace`
  const tw = ctx.measureText(text).width
  ctx.fillStyle = 'rgba(8,8,12,0.7)'
  ctx.fillRect(x, y - fs * 1.3, tw + fs, fs * 1.4)
  ctx.fillStyle = color
  ctx.fillText(text, x + fs * 0.5, y - fs * 0.1)
}

function drawMesh(ctx: CanvasRenderingContext2D, lm: NormalizedLandmark[], w: number, h: number,
  X: (x: number) => number, unit: number) {
  ctx.fillStyle = 'rgba(200,255,61,0.45)'
  const s = 1.3 * unit
  for (let i = 0; i < 468; i += 2) ctx.fillRect(X(lm[i].x * w) - s / 2, lm[i].y * h - s / 2, s, s)
}

function drawGuides(ctx: CanvasRenderingContext2D, P: (i: number) => readonly [number, number], unit: number) {
  ctx.lineWidth = 2 * unit
  // Canthal tilt lines (inner -> outer canthus, extended)
  ctx.strokeStyle = ACCENT
  for (const [a, b] of [[IDX.r_eye_inner, IDX.r_eye_outer], [IDX.l_eye_inner, IDX.l_eye_outer]]) {
    const [ax, ay] = P(a), [bx, by] = P(b)
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx + (bx - ax) * 0.4, by + (by - ay) * 0.4); ctx.stroke()
  }
  // Facial thirds ticks: glabella, subnasale, menton
  ctx.strokeStyle = 'rgba(94,230,255,0.8)'
  ctx.setLineDash([6 * unit, 5 * unit])
  const [zrx] = P(IDX.zyg_r), [zlx] = P(IDX.zyg_l)
  const xl = Math.min(zrx, zlx), xr = Math.max(zrx, zlx)
  for (const i of [IDX.glabella, IDX.subnasale, IDX.menton]) {
    const [, y] = P(i)
    ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke()
  }
  ctx.setLineDash([])
  // Jaw and cheekbone widths
  ctx.strokeStyle = 'rgba(255,92,122,0.8)'
  for (const [a, b] of [[IDX.gon_r, IDX.gon_l], [IDX.zyg_r, IDX.zyg_l]]) {
    const [ax, ay] = P(a), [bx, by] = P(b)
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke()
  }
  ctx.strokeStyle = WARN
  ctx.beginPath()
  for (const [i, j] of [[IDX.zyg_r, IDX.gon_r], [IDX.gon_r, IDX.menton], [IDX.menton, IDX.gon_l], [IDX.gon_l, IDX.zyg_l]]) {
    const [ax, ay] = P(i), [bx, by] = P(j)
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by)
  }
  ctx.stroke()
}
