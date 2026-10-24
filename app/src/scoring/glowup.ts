// Glow-up plan: non-surgical actions, chosen and predicted from the person's own measurements.
//
// Predictions are *measured*, not guessed: training/simulate_glowup.py applies each action as an edit to 300
// held-out test photos and re-runs the whole app pipeline (landmarks, features, CNN, fusion). An action is
// offered only if it raised the real score for most of those faces. For a given person the prediction is
// the measured CNN effect (adjusted for their own main measurement) plus the feature model's response to
// the measured feature change, combined with the app's fusion weights, so it is on the score they see.
import { geometric, type FusionModel, type GeoModel, type Rating, type Sex } from './scoring'
import { featureInfo } from './features'
import type { Features } from './metrics'

/** Actions are hidden when the predicted gain for this person is smaller than this (PSL). */
export const MIN_GAIN = 0.02

/** Measured effect of one action for one sex (from glowup.json). */
export interface ActionCalibration {
  n: number; meanPsl: number; success: number
  /** CNN effect = cnnA + cnnB * z(main feature). */
  cnnA: number; cnnB: number
  featureDelta: Record<string, number>
}
export interface GlowUpModel {
  minSuccess: number
  actions: Record<string, Partial<Record<Sex, ActionCalibration>>>
  /** Measured combined effect / sum of single effects, per sex. */
  additivity: Record<Sex, number>
}
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
  /** For actions that can't be simulated: which way the action moves each feature (+1 up, -1 down). */
  dirs?: Record<string, 1 | -1>
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
    dirs: { lower_face_fullness: -1, cheek_hollow: -1, jaw_definition: 1 },
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
    dirs: { hair_texture: 1, hair_darkness: 1 },
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
  /** Personal tip for the measurement that matters most for this person (if the action has one). */
  tip: string | null
  /** This person's measurements behind the action, biggest gap first. */
  reasons: PlanReason[]
  /** Predicted PSL after this action alone, and the gain over the current score. Null when untested. */
  predicted: number | null
  gain: number | null
  /** Share of test faces whose real score went up when this action was simulated. */
  success: number | null
}

export interface GlowUpPlan {
  /** Tested actions that raise the score for this person, biggest gain first. */
  items: PlanItem[]
  /** Worth doing but impossible to test on a photo (no prediction). */
  untested: PlanItem[]
  /** Titles of actions that were tested and did *not* raise the score. */
  failed: string[]
  potential: number
  current: number
}

const UNTESTED = new Set(['lean', 'hair', 'expression'])

function expand(names: string[], all: string[]) {
  return names.flatMap((n) => (n === 'bs_*' ? all.filter((a) => a.startsWith('bs_')) : [n]))
}

export function glowUpPlan(r: Rating, sex: Sex, geo: GeoModel, fusion: FusionModel, cal: GlowUpModel): GlowUpPlan {
  const values: Features = {}, target: Features = {}, z: Features = {}, effect: Features = {}
  for (const f of r.features) {
    if (!f.measured) continue
    values[f.name] = f.value; target[f.name] = f.target; z[f.name] = f.z; effect[f.name] = f.psl
  }
  const all = Object.keys(values)
  const base = geometric(values, sex, geo, fusion.scoreStd).score

  /** Measurements behind an action: ones currently costing this person points *and* that the action moves
   * toward the top-rated value (dir +1 = action raises it, -1 = lowers it, 0 = either), worst first. */
  const reasonsFor = (names: string[], dir: (n: string) => number): PlanReason[] => names
    .filter((n) => n in values && effect[n] < -0.005 && (dir(n) === 0 || Math.sign(target[n] - values[n]) === dir(n)))
    .sort((a, b) => effect[a] - effect[b]).slice(0, 3)
    .map((n) => {
      const info = featureInfo(n)
      return { name: n, label: info.label, you: info.fmt(values[n]), target: info.fmt(target[n]) }
    })

  const items: PlanItem[] = [], untested: PlanItem[] = [], failed: string[] = []
  for (const a of ACTIONS) {
    if (a.sex && a.sex !== sex) continue
    const names = expand(a.features, all).filter((n) => n in values)
    const c = UNTESTED.has(a.id) ? undefined : cal.actions[a.id]?.[sex]
    const dir = (n: string) => (c ? Math.sign(c.featureDelta[n] ?? 0) : (a.dirs?.[n] ?? 0))
    const reasons = reasonsFor(names, dir)
    const top = reasons[0]?.name
    const more = top !== undefined && target[top] > values[top]
    const item = {
      id: a.id, title: a.titleFor ? a.titleFor(more) : a.title,
      how: typeof a.how === 'function' ? a.how(more) : a.how,
      // A personal tip only when it is backed by one of this person's measurements.
      tip: top ? a.tips?.[top] ?? null : null, reasons,
    }
    if (UNTESTED.has(a.id)) {
      // No photo edit can simulate these, so no prediction; offer them only when the person's own
      // measurements say they are a weak spot.
      if (reasons.length) untested.push({ ...item, predicted: null, gain: null, success: null })
      continue
    }
    if (!c) continue
    if (c.success < cal.minSuccess || c.meanPsl <= 0) { failed.push(a.title); continue }
    const changed = { ...values }
    for (const [n, d] of Object.entries(c.featureDelta)) if (n in changed) changed[n] += d
    const geoDelta = geometric(changed, sex, geo, fusion.scoreStd).score - base
    const zMain = Math.max(-2.5, Math.min(2.5, z[names[0]] ?? 0))
    const cnnDelta = c.cnnA + c.cnnB * zMain
    const gain = (fusion.wCnn * cnnDelta + fusion.wGeo * geoDelta) / fusion.scoreStd
    if (gain < MIN_GAIN) continue
    items.push({ ...item, predicted: Math.min(8, r.psl + gain), gain, success: c.success })
  }
  items.sort((x, y) => (y.gain ?? 0) - (x.gain ?? 0))
  // Doing several at once was measured to add up to `additivity` x the sum of single effects; never inflate.
  const total = items.reduce((t, it) => t + (it.gain ?? 0), 0) * Math.min(1, cal.additivity[sex] ?? 1)
  return { items, untested, failed, potential: Math.min(8, r.psl + total), current: r.psl }
}
