// Port of compute_appearance / appearance_crop in training/mogscan_metrics.py. Pixel features
// (skin, colouring, hair, leanness) measured on a small square crop around the face.
import type { Features, Vec2 } from './metrics'

export const APPEAR_SIZE = 160
const SKIN_PTS = [50, 280, 151, 199]
const CHEEK_PTS = [50, 280]
const LIP_PTS = [11, 16]
const JAW_PTS = [172, 136, 150, 149, 176, 148, 397, 365, 379, 378, 400, 377]

export const APPEARANCE_GROUPS = {
  skin: ['skin_lum', 'skin_tone_var', 'skin_texture', 'skin_redness', 'skin_yellow', 'under_eye'],
  colouring: ['lip_contrast', 'lip_redness', 'eye_contrast', 'sclera_bright', 'brow_contrast', 'lash_line',
    'iris_lightness', 'iris_blue'],
  hair: ['hair_darkness', 'hair_texture', 'facial_hair'],
  leanness: ['jaw_definition', 'cheek_hollow', 'neck_width'],
} as const
export const APPEARANCE_NAMES: string[] = Object.values(APPEARANCE_GROUPS).flat()

type RGB = [number, number, number]

/** A square RGBA crop (canvas ImageData layout). */
interface Img { data: Uint8ClampedArray | number[]; size: number }

/** Pixels in the (2r+1)^2 box around (cx, cy), clamped to the image, as an RGB grid. */
function patch(img: Img, cx: number, cy: number, r: number): RGB[][] {
  const s = img.size
  const x = Math.floor(cx), y = Math.floor(cy)
  const cl = (v: number) => Math.min(Math.max(v, 0), s - 1)
  const x0 = cl(x - r), x1 = cl(x + r), y0 = cl(y - r), y1 = cl(y + r)
  const rows: RGB[][] = []
  for (let yy = y0; yy <= y1; yy++) {
    const row: RGB[] = []
    for (let xx = x0; xx <= x1; xx++) {
      const o = (yy * s + xx) * 4
      row.push([img.data[o], img.data[o + 1], img.data[o + 2]])
    }
    rows.push(row)
  }
  return rows
}

const lum = (c: RGB) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
const sum3 = (c: RGB) => c[0] + c[1] + c[2]
const chroma = (c: RGB) => (c[0] - c[1]) / (sum3(c) + 1e-6)
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length
const dist = (a: Vec2, b: Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1])
const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]

function meanRGB(rows: RGB[][]): RGB {
  const m: RGB = [0, 0, 0]
  let n = 0
  for (const row of rows) for (const c of row) { m[0] += c[0]; m[1] += c[1]; m[2] += c[2]; n++ }
  return [m[0] / n, m[1] / n, m[2] / n]
}

function meanOf(cs: RGB[]): RGB {
  return [mean(cs.map((c) => c[0])), mean(cs.map((c) => c[1])), mean(cs.map((c) => c[2]))]
}

/** Mean absolute 4-neighbour Laplacian of luminance (fine skin texture / blemishes). */
function texture(rows: RGB[][]): number {
  if (rows.length < 3 || rows[0].length < 3) return 0
  const y = rows.map((r) => r.map(lum))
  let t = 0, n = 0
  for (let i = 1; i < y.length - 1; i++) for (let j = 1; j < y[0].length - 1; j++) {
    t += Math.abs(4 * y[i][j] - y[i - 1][j] - y[i + 1][j] - y[i][j - 1] - y[i][j + 1])
    n++
  }
  return t / n
}

function neckWidth(img: Img, pts: Vec2[], skin: RGB): number {
  const s = img.size
  const faceLen = dist(pts[10], pts[152])
  const y = Math.min(Math.max(Math.floor(pts[152][1] + 0.12 * faceLen), 0), s - 1)
  const xMid = Math.min(Math.max(Math.floor(pts[152][0]), 0), s - 1)
  const ref = Math.hypot(...skin) + 1e-6
  const isSkin = (x: number) => {
    const o = (y * s + x) * 4
    return Math.hypot(img.data[o] - skin[0], img.data[o + 1] - skin[1], img.data[o + 2] - skin[2]) / ref < 0.25
  }
  if (!isSkin(xMid)) return 0
  let x0 = xMid, x1 = xMid
  while (x0 > 0 && isSkin(x0 - 1)) x0--
  while (x1 < s - 1 && isSkin(x1 + 1)) x1++
  return x1 - x0 + 1
}

/** img: square RGBA crop; pts: landmarks in crop pixel coordinates. */
export function computeAppearance(img: Img, pts: Vec2[]): Features {
  const ipd = dist(pts[468], pts[473])
  const rad = (k: number) => Math.max(1, Math.floor(k * ipd + 0.5))
  const rgbAt = (q: Vec2, k: number) => meanRGB(patch(img, q[0], q[1], rad(k)))
  const lumAt = (q: Vec2, k: number) => lum(rgbAt(q, k))

  const skin = SKIN_PTS.map((i) => rgbAt(pts[i], 0.08))
  const skinY = skin.map(lum)
  const sy = mean(skinY) + 1e-6
  const skinStd = Math.sqrt(mean(skinY.map((v) => (v - (sy - 1e-6)) ** 2)))
  const cheeks = CHEEK_PTS.map((i) => rgbAt(pts[i], 0.08))
  const lip = meanOf(LIP_PTS.map((i) => rgbAt(pts[i], 0.03)))
  const iris = meanOf([468, 473].map((i) => rgbAt(pts[i], 0.04)))
  const sclera: RGB[] = []
  for (const [c, corners] of [[468, [33, 133]], [473, [263, 362]]] as [number, number[]][])
    for (const k of corners) sclera.push(rgbAt(lerp(pts[c], pts[k], 0.65), 0.02))
  const brow = meanOf([[105, 52], [334, 282]].map(([a, b]) => rgbAt(lerp(pts[a], pts[b], 0.5), 0.03)))
  const under = meanOf([[145, 50], [374, 280]].map(([a, b]) => rgbAt(lerp(pts[a], pts[b], 0.35), 0.04)))
  const cheekY = mean(cheeks.map(lum)) + 1e-6
  const cheekChroma = mean(cheeks.map(chroma))

  let cx = 0, cy = 0
  for (const p of pts) { cx += p[0]; cy += p[1] }
  cx /= pts.length; cy /= pts.length
  const steps = JAW_PTS.map((i) => {
    const ux = pts[i][0] - cx, uy = pts[i][1] - cy
    const n = (Math.hypot(ux, uy) + 1e-9) / (0.06 * ipd)
    const u: Vec2 = [ux / n, uy / n]
    return Math.abs(lumAt([pts[i][0] - u[0], pts[i][1] - u[1]], 0.02) - lumAt([pts[i][0] + u[0], pts[i][1] + u[1]], 0.02))
  })
  const submalar = [[116, 172], [345, 397]].map(([a, b]) => lerp(pts[a], pts[b], 0.5))
  const hairPt: Vec2 = [pts[10][0] + 0.35 * (pts[10][0] - pts[168][0]), pts[10][1] + 0.35 * (pts[10][1] - pts[168][1])]
  const hairPx = patch(img, hairPt[0], hairPt[1], rad(0.1))
  const beard = mean([lumAt(lerp(pts[0], pts[2], 0.5), 0.03), lumAt(pts[199], 0.06)])

  return {
    skin_lum: sy / 255,
    skin_tone_var: skinStd / sy,
    skin_texture: mean(CHEEK_PTS.map((i) => texture(patch(img, pts[i][0], pts[i][1], rad(0.12))))) / sy,
    skin_redness: cheekChroma,
    skin_yellow: mean(cheeks.map((c) => ((c[0] + c[1]) / 2 - c[2]) / (sum3(c) + 1e-6))),
    under_eye: lum(under) / cheekY,
    lip_contrast: (sy - lum(lip)) / sy,
    lip_redness: chroma(lip) - cheekChroma,
    eye_contrast: (sy - lum(iris)) / sy,
    sclera_bright: mean(sclera.map(lum)) / sy,
    brow_contrast: (skinY[2] - lum(brow)) / (skinY[2] + 1e-6),
    lash_line: mean([159, 386].map((i) => lumAt(pts[i], 0.015))) / sy,
    iris_lightness: lum(iris) / 255,
    iris_blue: (iris[2] - iris[0]) / (sum3(iris) + 1e-6),
    hair_darkness: (sy - lum(meanRGB(hairPx))) / sy,
    hair_texture: texture(hairPx) / sy,
    facial_hair: (cheekY - beard) / cheekY,
    jaw_definition: mean(steps) / sy,
    cheek_hollow: mean(submalar.map((q) => lumAt(q, 0.04))) / (mean([116, 345].map((i) => lumAt(pts[i], 0.04))) + 1e-6),
    neck_width: neckWidth(img, pts, meanOf(cheeks)) / (dist(pts[234], pts[454]) + 1e-6),
  }
}

/** Crop box rounded like training (appearance_crop), and landmarks mapped into crop pixels. */
export function appearanceCrop(lm: { x: number; y: number }[], w: number, h: number,
  box: { x: number; y: number; side: number }) {
  const x = Math.round(box.x), y = Math.round(box.y), side = Math.round(box.side)
  const k = APPEAR_SIZE / side
  return { x, y, side, pts: lm.map((p) => [(p.x * w - x) * k, (p.y * h - y) * k] as Vec2) }
}
