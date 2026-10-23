import type { MetricName } from './metrics'

export interface FeatureInfo {
  label: string
  /** Format a raw metric value for display. */
  fmt: (v: number) => string
  /** Looksmax-community reference, shown next to the data-derived target. */
  community: string
  /** Plain-language description when the value is below / above the top-tier average. */
  low: string
  high: string
}

const ratio = (v: number) => v.toFixed(2)
const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const degs = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}°`

export const FEATURES: Record<MetricName, FeatureInfo> = {
  canthal_tilt: { label: 'Canthal tilt', fmt: degs, community: 'positive, +4° to +8°',
    low: 'neutral/negative tilt', high: 'strong positive tilt' },
  eye_spacing: { label: 'Eye spacing (ESR)', fmt: ratio, community: '0.45–0.47',
    low: 'close-set eyes', high: 'wide-set eyes' },
  fwhr: { label: 'FWHR', fmt: ratio, community: '≥ 1.9 (M), ~1.8 (F)',
    low: 'narrow / long upper face', high: 'wide, compact upper face' },
  midface_ratio: { label: 'Midface ratio', fmt: ratio, community: '~1.0 (compact)',
    low: 'compact midface', high: 'long midface' },
  lower_third: { label: 'Lower third', fmt: pct, community: '~50% of mid+lower',
    low: 'short lower third', high: 'long lower third' },
  chin_philtrum: { label: 'Chin : philtrum', fmt: ratio, community: '2.0–2.25',
    low: 'long philtrum / short chin', high: 'long chin vs philtrum' },
  nose_intercanthal: { label: 'Nose : intercanthal width', fmt: ratio, community: '~1.0',
    low: 'narrow nose', high: 'wide nose' },
  mouth_nose: { label: 'Mouth : nose width', fmt: ratio, community: '1.5–1.62',
    low: 'small mouth vs nose', high: 'wide mouth' },
  jaw_cheek: { label: 'Jaw : cheekbone width', fmt: ratio, community: '0.85–0.95 (M)',
    low: 'tapered jaw, cheekbone-dominant', high: 'wide jaw vs cheekbones' },
  lip_ratio: { label: 'Lower : upper lip', fmt: ratio, community: '~1.6',
    low: 'thin lower lip / full upper', high: 'full lower lip' },
  eye_aspect: { label: 'Eye openness', fmt: ratio, community: 'low = hunter eyes',
    low: 'hooded "hunter" eyes', high: 'round / open "prey" eyes' },
  brow_eye: { label: 'Brow–eye distance', fmt: ratio, community: 'low & compact (M)',
    low: 'low-set brows', high: 'high-set brows' },
  face_lw: { label: 'Face length : width', fmt: ratio, community: '~1.3–1.4 (full face)',
    low: 'short / wide face', high: 'long / narrow face' },
  symmetry: { label: 'Asymmetry', fmt: (v) => `${(v * 100).toFixed(1)}% IPD`, community: 'lower is better',
    low: 'very symmetric', high: 'noticeable asymmetry' },
  jaw_angle: { label: 'Jaw angle (projected)', fmt: (v) => `${v.toFixed(0)}°`, community: 'sharp, ~120–130°',
    low: 'sharp, angular jaw', high: 'soft / rounded jaw' },
}
