import { FEATURES } from './features'
import { METRIC_NAMES, type MetricName, type Metrics } from './metrics'

export type Sex = 'M' | 'F'

export interface GeoSexModel {
  mean: number[]; std: number[]; lin: number[]; quad: number[]; intercept: number
  contribStd: number[]; topQuintile: number[]
}
export interface GeoModel { metrics: string[]; sexes: Record<Sex, GeoSexModel> }
export interface FusionModel {
  wCnn: number; wGeo: number; bias: number
  /** Mean / std of the human ratings (1–5 scale) on the training set. */
  scoreMean: number; scoreStd: number
}

export interface FeatureResult {
  name: MetricName
  value: number
  target: number
  /** Effect on the predicted score in PSL points, relative to an average face. */
  psl: number
  /** 0–10 rating of this feature. */
  rating: number
  description: string
}

export interface Rating {
  geo: number
  cnn: number | null
  fused: number
  psl: number
  tier: string
  percentile: number
  features: FeatureResult[]
}

/** Geometric score (1–5 scale) plus per-feature contributions. */
export function geometric(m: Metrics, sex: Sex, model: GeoModel, scoreStd: number) {
  const s = model.sexes[sex]
  let score = s.intercept
  const features: FeatureResult[] = METRIC_NAMES.map((name, i) => {
    const z = Math.max(-4, Math.min(4, (m[name] - s.mean[i]) / s.std[i]))
    const c = s.lin[i] * z + s.quad[i] * (z * z - 1)
    score += c
    const info = FEATURES[name]
    return {
      name, value: m[name], target: s.topQuintile[i],
      psl: c / scoreStd,
      rating: 5 + 5 * Math.tanh(c / (1.5 * s.contribStd[i] + 1e-9)),
      description: m[name] < s.topQuintile[i] ? info.low : info.high,
    }
  })
  return { score, features }
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
export function rate(m: Metrics, cnn: number | null, sex: Sex, geo: GeoModel, fusion: FusionModel): Rating {
  const g = geometric(m, sex, geo, fusion.scoreStd)
  const fused = cnn === null ? g.score : fusion.wCnn * cnn + fusion.wGeo * g.score + fusion.bias
  return finalize(g.score, cnn, fused, g.features, sex, fusion)
}

export function finalize(geoScore: number, cnn: number | null, fused: number, features: FeatureResult[], sex: Sex,
  fusion: FusionModel): Rating {
  const z = (fused - fusion.scoreMean) / fusion.scoreStd
  const psl = Math.max(1, Math.min(8, 4 + z))
  return { geo: geoScore, cnn, fused, psl, tier: tierFor(psl, sex), percentile: normCdf(z) * 100, features }
}

/** Top strengths and weaknesses (largest positive / negative contributions). */
export function strengthsWeaknesses(features: FeatureResult[], n = 3) {
  const sorted = [...features].sort((a, b) => b.psl - a.psl)
  return {
    strengths: sorted.filter((f) => f.psl > 0.02).slice(0, n),
    weaknesses: sorted.filter((f) => f.psl < -0.02).reverse().slice(0, n),
  }
}
