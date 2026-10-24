import { featureInfo, PILLAR_ORDER, PILLAR_OVERRIDES, PILLARS } from './features'
import type { Features } from './metrics'

export type Sex = 'M' | 'F'

export interface GeoSexModel {
  mean: number[]; std: number[]; lin: number[]; quad: number[]; intercept: number
  contribStd: number[]; topQuintile: number[]; corr: number[]
}
export interface GeoModel { metrics: string[]; groups: Record<string, string[]>; sexes: Record<Sex, GeoSexModel> }
export interface FusionModel {
  wCnn: number; wGeo: number; bias: number
  /** Mean / std of the human ratings (1–5 scale) on the training set. */
  scoreMean: number; scoreStd: number
}

export interface FeatureResult {
  name: string
  group: string
  pillar: string
  value: number
  /** False when the feature could not be measured yet (e.g. pixels not sampled); it then counts as average. */
  measured: boolean
  target: number
  /** Standardised value (SD from the training mean for this sex). */
  z: number
  /** Learned weights: linear and quadratic ("ideal range") terms, in rating points per SD. */
  lin: number
  quad: number
  /** How much this feature moves a typical face, in PSL (SD of its contribution). */
  importance: number
  /** Effect on the predicted score in PSL points, relative to an average face. */
  psl: number
  /** 0–10 rating of this feature. */
  rating: number
  description: string
}

export interface PillarResult { name: string; psl: number; rating: number }

export interface Rating {
  geo: number
  cnn: number | null
  fused: number
  psl: number
  tier: string
  percentile: number
  features: FeatureResult[]
  pillars: PillarResult[]
  /** Share of the final PSL deviation from 4.0 that comes from the CNN vs the feature model. */
  cnnPsl: number | null
  geoPsl: number
}

const pillarOf = (name: string, group: string) => PILLAR_OVERRIDES[name] ?? PILLARS[group] ?? 'Harmony'

/** Feature-model score (1–5 scale) plus per-feature contributions. */
export function geometric(m: Features, sex: Sex, model: GeoModel, scoreStd: number) {
  const s = model.sexes[sex]
  const groupOf: Record<string, string> = {}
  for (const [g, names] of Object.entries(model.groups)) for (const n of names) groupOf[n] = g
  let score = s.intercept
  const features: FeatureResult[] = model.metrics.map((name, i) => {
    const v = m[name]
    const measured = v !== undefined && Number.isFinite(v)
    const z = measured ? Math.max(-4, Math.min(4, (v - s.mean[i]) / s.std[i])) : 0
    // An unmeasured feature contributes its expected value, 0.
    const c = measured ? s.lin[i] * z + s.quad[i] * (z * z - 1) : 0
    score += c
    const info = featureInfo(name)
    return {
      name, group: groupOf[name], pillar: pillarOf(name, groupOf[name]),
      value: measured ? v : s.mean[i], measured, target: s.topQuintile[i], z,
      lin: s.lin[i], quad: s.quad[i], importance: s.contribStd[i] / scoreStd,
      psl: c / scoreStd,
      rating: 5 + 5 * Math.tanh(c / (1.5 * s.contribStd[i] + 1e-9)),
      description: (measured ? v : s.mean[i]) < s.topQuintile[i] ? info.low : info.high,
    }
  })
  return { score, features }
}

/** Roll feature contributions up into Clavicular-style pillars (harmony, dimorphism, …). */
export function pillars(features: FeatureResult[]): PillarResult[] {
  return PILLAR_ORDER.map((name) => {
    const fs = features.filter((f) => f.pillar === name)
    const psl = fs.reduce((t, f) => t + f.psl, 0)
    const spread = Math.sqrt(fs.reduce((t, f) => t + f.importance ** 2, 0)) + 1e-9
    return { name, psl, rating: 5 + 5 * Math.tanh(psl / (1.5 * spread)) }
  }).filter((p) => features.some((f) => f.pillar === p.name))
}

const TIERS: [number, string, string][] = [
  [2, 'Sub3', 'Sub3'],
  [3, 'Sub5', 'Sub5'],
  [3.75, 'LTN', 'LTB'],
  [4.5, 'MTN', 'MTB'],
  [5.25, 'HTN', 'HTB'],
  [6, 'Chadlite', 'Stacylite'],
  [7, 'Chad', 'Stacy'],
  [Infinity, 'Adamlite+', 'Evelite+'],
]

const TIER_NAMES: Record<string, string> = {
  Sub3: 'Well below average', Sub5: 'Below average',
  LTN: 'Low-tier normie', MTN: 'Mid-tier normie', HTN: 'High-tier normie',
  LTB: 'Low-tier Becky', MTB: 'Mid-tier Becky', HTB: 'High-tier Becky',
  Chadlite: 'Near Chad', Stacylite: 'Near Stacy', Chad: 'Top ~2%', Stacy: 'Top ~2%',
  'Adamlite+': 'Model tier', 'Evelite+': 'Model tier',
}

/** Plain-English name for a community tier code. */
export const tierName = (tier: string) => TIER_NAMES[tier] ?? tier

export function tierFor(psl: number, sex: Sex): string {
  for (const [hi, m, f] of TIERS) if (psl < hi) return sex === 'M' ? m : f
  return sex === 'M' ? 'Adamlite+' : 'Evelite+'
}

/** Standard normal CDF (Abramowitz–Stegun 7.1.26). */
export function normCdf(z: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2)
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
    Math.exp(-(z * z) / 2)
  return z >= 0 ? (1 + y) / 2 : (1 - y) / 2
}

/** Fuse model outputs and map to the PSL scale: PSL = 4 + z, where z is the score's
 * standard deviations from the mean human rating. Tier cut-offs follow community usage
 * (Chadlite ≈ top 10%, Chad ≈ top 2%). */
export function rate(m: Features, cnn: number | null, sex: Sex, geo: GeoModel, fusion: FusionModel): Rating {
  const g = geometric(m, sex, geo, fusion.scoreStd)
  const fused = cnn === null ? g.score : fusion.wCnn * cnn + fusion.wGeo * g.score + fusion.bias
  const z = (fused - fusion.scoreMean) / fusion.scoreStd
  const psl = Math.max(1, Math.min(8, 4 + z))
  // Each model's share of the PSL, measured against an average face for the other model.
  const cnnPsl = cnn === null ? null : (fusion.wCnn * (cnn - fusion.scoreMean)) / fusion.scoreStd
  const geoPsl = ((cnn === null ? 1 : fusion.wGeo) * (g.score - fusion.scoreMean)) / fusion.scoreStd
  return { geo: g.score, cnn, fused, psl, tier: tierFor(psl, sex), percentile: normCdf(z) * 100,
    features: g.features, pillars: pillars(g.features), cnnPsl, geoPsl }
}

/** Top strengths and weaknesses (largest positive / negative contributions). */
export function strengthsWeaknesses(features: FeatureResult[], n = 3) {
  // Expression and abstract shape components are not things you can point at, so they are not listed here.
  const sorted = features.filter((f) => f.measured && f.group !== 'expression' && !f.name.startsWith('shape_pc'))
    .sort((a, b) => b.psl - a.psl)
  return {
    strengths: sorted.filter((f) => f.psl > 0.02).slice(0, n),
    weaknesses: sorted.filter((f) => f.psl < -0.02).reverse().slice(0, n),
  }
}
