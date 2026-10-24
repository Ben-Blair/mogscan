// Glow-up plan: non-surgical actions mapped to the features they change, with a predicted score.
// The prediction moves each action's features part of the way toward the average of top-rated faces
// and re-runs the feature model. It is an estimate of what the model would say, not a guarantee.
import { finalizeWith, geometric, type FusionModel, type GeoModel, type Rating, type Sex } from './scoring'
import { featureInfo } from './features'
import type { Features } from './metrics'

/** How far toward the top-rated average each action is assumed to move a feature. */
export const PROGRESS = 0.5

interface Action {
  id: string
  title: string
  how: string | ((more: boolean) => string)
  /** Features this action plausibly changes. `bs_*` matches every expression blendshape. */
  features: string[]
  /** Only offer for one sex (e.g. stubble). */
  sex?: Sex
  /** Headline when the data says "more" vs "less" of the main feature is better (optional). */
  titleFor?: (more: boolean) => string
  /** Extra, specific advice keyed by the feature that drives this person's gain the most. */
  tips?: Record<string, string>
}

const ACTIONS: Action[] = [
  { id: 'skincare', title: 'Start a simple skincare routine',
    how: 'Gentle cleanser, moisturiser and SPF 30+ every morning; a retinoid at night. Texture and evenness improve over 8–12 weeks.',
    features: ['skin_texture', 'skin_tone_var'],
    tips: { skin_texture: 'Your skin texture is the main gap: prioritise the night-time retinoid (start 2–3 nights a week) and a gentle exfoliant once a week.',
      skin_tone_var: 'Your uneven tone is the main gap: add a vitamin C serum in the morning and never skip SPF, which stops patches darkening.' } },
  { id: 'redness', title: 'Calm redness',
    how: 'Niacinamide or azelaic acid serum, fragrance-free products. A green-tinted primer works instantly.',
    features: ['skin_redness'] },
  { id: 'undereye', title: 'Brighten your under-eyes',
    how: '7–9 hours of sleep, a cold compress in the morning, and a peach-toned concealer for dark circles.',
    features: ['under_eye'] },
  { id: 'glow', title: 'Warm up your skin tone (without UV)',
    how: 'Gradual self-tanner or a light bronzer. Same glow as a tan, without the skin-cancer and ageing risk of UV.',
    features: ['skin_yellow', 'skin_lum'] },
  { id: 'jaw', title: 'Contour your jawline',
    how: 'Draw a soft line just under the jawbone with a contour pencil or bronzer and blend it downward. It deepens the jaw shadow.',
    features: ['jaw_definition'] },
  { id: 'cheeks', title: 'Contour under your cheekbones',
    how: 'Bronzer from the ear toward the corner of the mouth, just below the cheekbone, blended upward.',
    features: ['cheek_hollow'] },
  { id: 'lean', title: 'Lose facial puffiness',
    how: 'If you have fat to lose, a modest calorie deficit and strength training; less salt and alcohol, more water and sleep. This slims the lower face and reveals the jaw.',
    features: ['lower_face_fullness', 'cheek_hollow', 'jaw_definition'],
    tips: { lower_face_fullness: 'Your lower face reads fuller than top-rated faces, so puffiness, not bone, is the gap: cut salt and alcohol first, since water retention shows within days.',
      jaw_definition: 'Your jaw shadow is soft: losing even a little facial fat sharpens it more than anything else.',
      cheek_hollow: 'Your cheeks read full: leaning out is what creates the hollow under the cheekbone.' } },
  { id: 'brows', title: 'Fill in your brows',
    how: 'Brow pencil or tinted gel in hair-like strokes, or a professional brow tint. Fuller, darker brows frame the eyes.',
    features: ['brow_contrast', 'brow_thickness'],
    tips: { brow_contrast: 'Your brows are lighter than your skin tone needs: go one shade darker than your hair with a tinted gel.',
      brow_thickness: 'Your brows are thin: fill the lower edge with hair-like strokes to make them look thicker and lower.' } },
  { id: 'lashes', title: 'Define your lash line',
    how: 'Tightline the upper lash line with a dark pencil, or use mascara or a lash tint.',
    features: ['lash_line'] },
  { id: 'lips', title: 'Add a little lip colour',
    how: 'A tinted balm one shade deeper than your natural lip colour, and keep lips moisturised.',
    features: ['lip_contrast', 'lip_redness'],
    tips: { lip_contrast: 'Your lips blend into your skin: anything that adds definition to the lip line helps most.',
      lip_redness: 'Your lip colour is pale relative to your cheeks: a rosy or berry tint closes that gap.' } },
  { id: 'facialhair', title: 'Groom your facial hair', sex: 'M',
    titleFor: (more) => (more ? 'Grow even stubble' : 'Go clean-shaven'),
    how: (more) => (more
      ? 'Even 3–5 mm stubble with a clean neckline adds shadow and definition to the lower face.'
      : 'A clean shave (or a very short, even trim) suits your face better on this model.'),
    features: ['facial_hair'] },
  { id: 'hair', title: 'Get a haircut with texture',
    how: 'Ask for volume and texture on top with tidy sides; a matte styling product keeps it defined.',
    features: ['hair_texture', 'hair_darkness'],
    tips: { hair_texture: 'Your hair reads flat in the scan: volume and texture on top is what to ask for.',
      hair_darkness: 'Your hair reads light against your face: a slightly darker, richer shade adds contrast.' } },
  { id: 'eyes', title: 'Brighter, clearer eyes',
    how: 'Sleep, hydration, screen breaks and lubricating eye drops make the whites of the eyes brighter.',
    features: ['sclera_bright'] },
  { id: 'expression', title: 'Relax your expression',
    how: 'Relax your forehead and brows, keep your eyes open, and try a slight, natural smile in photos.',
    features: ['bs_*'] },
]

/** One of this person's measurements behind an action. */
export interface PlanReason { name: string; label: string; you: string; target: string }

export interface PlanItem {
  id: string
  title: string
  how: string
  /** Personal tip for the measurement that drives the gain most (if the action has one). */
  tip: string | null
  /** This person's measurements that the action improves, biggest effect first. */
  reasons: PlanReason[]
  /** Predicted PSL after this action alone, and the gain over the current score. */
  predicted: number
  gain: number
}

export interface GlowUpPlan { items: PlanItem[]; potential: number; current: number }

function expand(names: string[], all: string[]) {
  return names.flatMap((n) => (n === 'bs_*' ? all.filter((a) => a.startsWith('bs_')) : [n]))
}

/** Rank actions by the score gain the model predicts for this face. */
export function glowUpPlan(r: Rating, sex: Sex, geo: GeoModel, fusion: FusionModel): GlowUpPlan {
  const values: Features = {}
  const target: Features = {}
  for (const f of r.features) if (f.measured) { values[f.name] = f.value; target[f.name] = f.target }
  const all = Object.keys(values)
  const base = geometric(values, sex, geo, fusion.scoreStd).score

  const improve = (names: string[], into: Features) => {
    for (const n of names) if (n in values) into[n] = values[n] + PROGRESS * (target[n] - values[n])
  }
  const predict = (changed: Features) => {
    const g = geometric(changed, sex, geo, fusion.scoreStd).score
    return finalizeWith(r, g - base, fusion).psl
  }

  const items: PlanItem[] = []
  const combined: Features = { ...values }
  for (const a of ACTIONS) {
    if (a.sex && a.sex !== sex) continue
    const names = expand(a.features, all).filter((n) => n in values)
    if (!names.length) continue
    const changed = { ...values }
    improve(names, changed)
    const predicted = predict(changed)
    const gain = predicted - r.psl
    if (gain < 0.03) continue
    improve(names, combined)
    // Which of this person's measurements drive the gain: improve each one alone.
    const drivers = names.map((n) => {
      const one = { ...values }
      improve([n], one)
      return { n, g: predict(one) - r.psl }
    }).filter((d) => d.g > 0.005).sort((x, y) => y.g - x.g)
    const top = drivers[0]?.n ?? names[0]
    const more = target[top] > values[top]
    items.push({
      id: a.id, title: a.titleFor ? a.titleFor(more) : a.title,
      how: typeof a.how === 'function' ? a.how(more) : a.how,
      tip: a.tips?.[top] ?? null,
      reasons: drivers.slice(0, 3).map(({ n }) => {
        const info = featureInfo(n)
        return { name: n, label: info.label, you: info.fmt(values[n]), target: info.fmt(target[n]) }
      }),
      predicted, gain,
    })
  }
  items.sort((x, y) => y.gain - x.gain)
  return { items, potential: Math.max(r.psl, predict(combined)), current: r.psl }
}
