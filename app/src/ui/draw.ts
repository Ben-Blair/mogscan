import type { Rating } from '../scoring/scoring'
import { scanProgress, type Track } from '../vision/tracker'

export interface DrawOptions { mirror: boolean }

const LINE = 'rgba(255,255,255,0.75)'
const TRACK_BG = 'rgba(255,255,255,0.18)'
const PILL = 'rgba(12,12,14,0.62)'
const FONT = '"Inter", system-ui, -apple-system, sans-serif'

export function tierColor(psl: number) {
  if (psl >= 6) return '#e9c46a'
  if (psl >= 5.25) return '#b8d98a'
  if (psl >= 4.5) return '#8ec5d6'
  if (psl >= 3.75) return '#d8d4cc'
  return '#d9a08a'
}

/** Video plus a minimal overlay: thin corner marks, scan progress, and one small label per face. */
export function drawFrame(ctx: CanvasRenderingContext2D, source: CanvasImageSource, w: number, h: number,
  tracks: Track[], ratings: Map<number, Rating>, opts: DrawOptions) {
  ctx.save()
  if (opts.mirror) { ctx.translate(w, 0); ctx.scale(-1, 1) }
  ctx.drawImage(source, 0, 0, w, h)
  ctx.restore()

  const unit = Math.max(1, Math.min(w, h) / 720)
  const many = tracks.length > 1
  for (const t of tracks) {
    const r = ratings.get(t.id)
    const box = t.displayBox
    const bx = opts.mirror ? w - box.x - box.w : box.x
    const pad = box.w * 0.12
    // Keep the frame and its label inside the canvas (photos are often cropped tight to the face).
    const inset = 4 * unit, labelH = 46 * unit
    const x0 = Math.max(inset, bx - pad), y0 = Math.max(inset + labelH, box.y - pad * 1.4)
    const bw = Math.min(w - inset, bx + box.w + pad) - x0
    const bh = Math.min(h - inset - 14 * unit, box.y + box.h + pad) - y0

    corners(ctx, x0, y0, bw, bh, 1.5 * unit, r ? 0.9 : 0.55)

    const prefix = many ? `${t.id}  ·  ` : ''
    if (r) {
      pill(ctx, x0 + bw / 2, y0 - 10 * unit, [
        { text: prefix, color: 'rgba(255,255,255,0.6)', weight: 500 },
        { text: r.psl.toFixed(1), color: tierColor(r.psl), weight: 600 },
        { text: `  ${r.tier}`, color: 'rgba(255,255,255,0.85)', weight: 500 },
      ], 15 * unit)
    } else {
      const msg = t.frontal ? 'Hold still' : 'Face the camera'
      pill(ctx, x0 + bw / 2, y0 - 10 * unit, [{ text: prefix + msg, color: 'rgba(255,255,255,0.85)', weight: 500 }], 13 * unit)
      // Scan progress along the bottom edge.
      const p = scanProgress(t)
      const y = y0 + bh + 8 * unit
      ctx.lineCap = 'round'
      ctx.lineWidth = 2 * unit
      ctx.strokeStyle = TRACK_BG
      ctx.beginPath(); ctx.moveTo(x0 + bw * 0.2, y); ctx.lineTo(x0 + bw * 0.8, y); ctx.stroke()
      if (p > 0) {
        ctx.strokeStyle = LINE
        ctx.beginPath(); ctx.moveTo(x0 + bw * 0.2, y); ctx.lineTo(x0 + bw * (0.2 + 0.6 * p), y); ctx.stroke()
      }
    }
  }
}

function corners(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, lw: number, alpha: number) {
  const c = Math.min(w, h) * 0.1
  ctx.strokeStyle = `rgba(255,255,255,${alpha})`
  ctx.lineWidth = lw
  ctx.shadowColor = 'rgba(0,0,0,0.45)'
  ctx.shadowBlur = 3 * lw
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
    ctx.moveTo(cx + dx * c, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * c)
  }
  ctx.stroke()
  ctx.shadowBlur = 0
}

/** A rounded, translucent label centred at (cx, bottom). */
function pill(ctx: CanvasRenderingContext2D, cx: number, bottom: number,
  parts: { text: string; color: string; weight: number }[], fs: number) {
  const widths = parts.map((p) => { ctx.font = `${p.weight} ${fs}px ${FONT}`; return ctx.measureText(p.text).width })
  const tw = widths.reduce((a, b) => a + b, 0)
  const ph = fs * 1.9, pw = tw + fs * 1.6
  const x = cx - pw / 2, y = bottom - ph
  ctx.fillStyle = PILL
  ctx.beginPath()
  ctx.roundRect(x, y, pw, ph, ph / 2)
  ctx.fill()
  ctx.textBaseline = 'middle'
  let tx = x + fs * 0.8
  parts.forEach((p, i) => {
    ctx.font = `${p.weight} ${fs}px ${FONT}`
    ctx.fillStyle = p.color
    ctx.fillText(p.text, tx, y + ph / 2 + fs * 0.04)
    tx += widths[i]
  })
}
