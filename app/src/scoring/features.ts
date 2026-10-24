export interface FeatureInfo {
  label: string
  /** Format a raw feature value for display. */
  fmt: (v: number) => string
  /** Looksmax-community reference, shown next to the data-derived target. */
  community: string
  /** Plain-language description when the value is below / above the top-tier average. */
  low: string
  high: string
}

const ratio = (v: number) => v.toFixed(2)
const r3 = (v: number) => v.toFixed(3)
const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const degs = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}°`
const ang = (v: number) => `${v.toFixed(0)}°`
const ipdPct = (v: number) => `${(v * 100).toFixed(1)}% IPD`
const sd = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)} SD`

const f = (label: string, fmt: (v: number) => string, community: string, low: string, high: string): FeatureInfo =>
  ({ label, fmt, community, low, high })

export const FEATURES: Record<string, FeatureInfo> = {
  // Eyes
  canthal_tilt: f('Canthal tilt', degs, 'positive, +4° to +8°', 'neutral/negative tilt', 'strong positive tilt'),
  canthal_asym: f('Canthal tilt asymmetry', degs, 'lower is better', 'eyes tilt evenly', 'eyes tilt unevenly'),
  eye_spacing: f('Eye spacing (ESR)', ratio, '0.45–0.47', 'close-set eyes', 'wide-set eyes'),
  intercanthal_eye: f('Intercanthal : eye width', ratio, '~1.0 ("one eye apart")', 'eyes close together', 'eyes far apart'),
  eye_width: f('Eye width : face width', ratio, 'wide, almond eyes', 'small eyes', 'large, wide eyes'),
  eye_aspect: f('Eye openness', ratio, 'low = hunter eyes', 'hooded "hunter" eyes', 'round / open "prey" eyes'),
  iris_fill: f('Iris : eye width', ratio, 'iris fills the eye', 'small iris / lots of sclera', 'iris fills the eye'),
  scleral_show: f('Lower scleral show', r3, '≤ 0 (none)', 'no white below the iris', 'white visible below the iris'),
  upper_scleral: f('Upper scleral show', r3, '< 0 (lid covers iris top)', 'upper lid covers iris', 'white visible above the iris'),
  // Brows
  brow_eye: f('Brow–eye distance', ratio, 'low & compact (M)', 'low-set brows', 'high-set brows'),
  brow_tilt: f('Brow tilt', degs, 'flat (M), lifted tail (F)', 'drooping brow tails', 'lifted brow tails'),
  brow_thickness: f('Brow thickness', ipdPct, 'thick, dense brows', 'thin brows', 'thick brows'),
  brow_length: f('Brow length : eye width', ratio, 'extends past the eye', 'short brows', 'long brows'),
  brow_gap: f('Brow gap : intercanthal', ratio, '~1.0', 'brows close together', 'brows far apart'),
  // Nose
  nose_intercanthal: f('Nose : intercanthal width', ratio, '~1.0', 'narrow nose', 'wide nose'),
  nose_face: f('Nose : face width', ratio, '~0.25–0.3', 'narrow nose', 'wide nose'),
  nose_length: f('Nose length', pct, '~1/3 of nasion–chin', 'short nose', 'long nose'),
  nose_bridge: f('Bridge : alar width', ratio, 'slim bridge', 'slim nasal bridge', 'wide nasal bridge'),
  nose_tip_width: f('Nose-tip width (bulbousness)', ratio, 'refined, not bulbous', 'refined tip', 'bulbous tip'),
  // Mouth
  mouth_nose: f('Mouth : nose width', ratio, '1.5–1.62', 'small mouth vs nose', 'wide mouth'),
  mouth_ipd: f('Mouth : IPD', ratio, '~1.0', 'narrow mouth', 'wide mouth'),
  lip_ratio: f('Lower : upper lip', ratio, '~1.6', 'thin lower lip / full upper', 'full lower lip'),
  lip_fullness: f('Lip fullness', pct, 'full lips', 'thin lips', 'full lips'),
  cupid_bow: f("Cupid's bow depth", ratio, 'defined bow', 'flat upper lip', "defined cupid's bow"),
  mouth_tilt: f('Mouth-corner tilt', r3, 'neutral to upturned', 'downturned corners', 'upturned corners'),
  philtrum_lower: f('Philtrum : lower third', pct, 'short philtrum', 'short philtrum', 'long philtrum'),
  // Jaw & chin
  chin_philtrum: f('Chin : philtrum', ratio, '2.0–2.25', 'long philtrum / short chin', 'long chin vs philtrum'),
  chin_height: f('Chin height', pct, 'tall chin (M)', 'short chin', 'tall chin'),
  chin_width: f('Chin : jaw width', ratio, 'square chin (M)', 'pointed chin', 'broad, square chin'),
  jaw_cheek: f('Bigonial : bizygomatic', ratio, '0.85–0.95 (M)', 'tapered jaw, cheekbone-dominant', 'wide jaw vs cheekbones'),
  jaw_angle: f('Gonial angle (projected)', ang, 'sharp, ~120–130°', 'sharp, angular jaw', 'soft / rounded jaw'),
  jaw_frame: f('Jaw width : face length', ratio, 'wide frame (M)', 'narrow jaw', 'wide jaw'),
  chin_angle: f('Chin taper angle', ang, 'wide (M), narrow (F)', 'V-shaped taper', 'broad, square lower face'),
  ramus: f('Ramus length', ratio, 'long ramus', 'short ramus', 'long ramus'),
  // Proportions
  fwhr: f('FWHR', ratio, '≥ 1.9 (M), ~1.8 (F)', 'narrow / long upper face', 'wide, compact upper face'),
  midface_ratio: f('Midface ratio', ratio, '~1.0 (compact)', 'compact midface', 'long midface'),
  lower_third: f('Lower third', pct, '~50% of mid+lower', 'short lower third', 'long lower third'),
  upper_third: f('Upper third (visible)', pct, 'balanced thirds', 'short forehead', 'tall forehead'),
  face_lw: f('Face length : width', ratio, '~1.3–1.4 (full face)', 'short / wide face', 'long / narrow face'),
  temple_cheek: f('Temple : cheekbone width', ratio, 'full temples', 'narrow temples', 'full temples'),
  cheekbone_height: f('Cheekbone height', ratio, 'high cheekbones', 'low cheekbones', 'high cheekbones'),
  lower_face_fullness: f('Lower-face fullness (puffiness)', ratio, 'lean, not puffy', 'lean lower face', 'full / puffy lower face'),
  // Depth (pseudo-profile from the 3D mesh)
  nasal_projection: f('Nasal projection', ipdPct, 'moderate', 'flat nose', 'projecting nose'),
  chin_projection: f('Chin projection', ipdPct, 'forward chin', 'recessed chin', 'projecting chin'),
  lip_projection: f('Lips vs E-line', ipdPct, 'slightly behind E-line', 'lips behind E-line', 'lips past E-line'),
  malar_projection: f('Cheekbone projection', ipdPct, 'prominent cheekbones', 'flat cheekbones', 'prominent cheekbones'),
  orbital_depth: f('Brow ridge / deep-set eyes', ipdPct, 'deep-set (M)', 'protruding eyes', 'deep-set eyes, strong brow ridge'),
  forehead_slope: f('Forehead slope', ipdPct, 'moderate', 'upright forehead', 'sloping forehead'),
  facial_convexity: f('Facial convexity', ang, '~165–175°', 'convex profile', 'straight profile'),
  face_depth: f('Face depth', ratio, 'well-projected face', 'flat face', 'deep, projected face'),
  orbital_vector: f('Orbital vector', ipdPct, 'positive (cheek ahead of eye)', 'negative vector (cheek behind eye)', 'positive vector'),
  midface_projection: f('Midface projection', ipdPct, 'forward maxilla', 'recessed midface', 'forward midface'),
  // Symmetry
  symmetry: f('Asymmetry (overall)', ipdPct, 'lower is better', 'very symmetric', 'noticeable asymmetry'),
  sym_eyes: f('Eye asymmetry', ipdPct, 'lower is better', 'symmetric eyes', 'asymmetric eyes'),
  sym_brows: f('Brow asymmetry', ipdPct, 'lower is better', 'symmetric brows', 'asymmetric brows'),
  sym_nose: f('Nose asymmetry', ipdPct, 'lower is better', 'straight nose', 'deviated nose'),
  sym_mouth: f('Mouth asymmetry', ipdPct, 'lower is better', 'symmetric mouth', 'asymmetric mouth'),
  sym_jaw: f('Jaw asymmetry', ipdPct, 'lower is better', 'symmetric jaw', 'asymmetric jaw'),
  // Whole-face shape
  averageness: f('Averageness (distance from mean face)', r3, 'lower = more harmonious', 'close to the average face', 'distinctive / atypical shape'),
  dimorphism: f('Shape dimorphism', r3, 'masculine (M), feminine (F)', 'feminine shape', 'masculine shape'),
  // Skin
  skin_lum: f('Skin brightness', pct, 'even, bright', 'darker / dim lighting', 'bright skin'),
  skin_tone_var: f('Skin-tone unevenness', pct, 'lower is better', 'even skin tone', 'patchy skin tone'),
  skin_texture: f('Skin texture', r3, 'lower = smoother', 'smooth skin', 'textured skin'),
  skin_redness: f('Skin redness', r3, 'healthy flush', 'pale / sallow', 'rosy / red'),
  skin_yellow: f('Skin warmth (undertone)', r3, 'warm, golden', 'cool undertone', 'warm undertone'),
  under_eye: f('Under-eye brightness', ratio, '~1.0 (no dark circles)', 'dark circles', 'bright under-eyes'),
  // Colouring
  lip_contrast: f('Lip contrast', pct, 'high contrast', 'lips blend into skin', 'lips stand out'),
  lip_redness: f('Lip colour', r3, 'red lips', 'pale lips', 'red lips'),
  eye_contrast: f('Eye contrast', pct, 'high contrast', 'low eye contrast', 'striking eyes'),
  sclera_bright: f('Sclera brightness', ratio, 'bright whites', 'dull sclera', 'bright sclera'),
  brow_contrast: f('Brow darkness', pct, 'dark, defined brows', 'faint brows', 'dark brows'),
  lash_line: f('Lash line', ratio, 'lower = darker lashes', 'dense, dark lashes', 'faint lashes'),
  iris_lightness: f('Iris lightness (eye colour)', pct, 'preference varies', 'dark eyes', 'light eyes'),
  iris_blue: f('Iris blueness (eye colour)', r3, 'preference varies', 'brown / hazel', 'blue / grey'),
  // Hair & grooming
  hair_darkness: f('Hair darkness', pct, 'dark, full hair', 'light hair / forehead', 'dark hair'),
  hair_texture: f('Hair texture', r3, 'full, textured hair', 'smooth / sparse', 'textured / full'),
  facial_hair: f('Facial hair / stubble', pct, 'groomed', 'clean-shaven', 'stubble / beard'),
  // Leanness
  jaw_definition: f('Jawline definition', pct, 'sharp, shadowed jaw', 'soft jawline', 'defined jawline'),
  cheek_hollow: f('Cheek hollowness', ratio, 'lower = hollower (lean)', 'hollow cheeks', 'full cheeks'),
  neck_width: f('Neck width', ratio, 'thick neck (M)', 'slim / hidden neck', 'thick neck'),
}

for (let k = 1; k <= 12; k++) {
  FEATURES[`shape_pc${k}`] = f(`Shape component ${k}`, sd, 'learned from data', 'below average', 'above average')
}

/** Look up display info; expression blendshapes get generated labels. */
export function featureInfo(name: string): FeatureInfo {
  if (FEATURES[name]) return FEATURES[name]
  if (name.startsWith('bs_')) {
    const label = name.slice(3).replace(/([A-Z])/g, ' $1').toLowerCase()
    return f(`Expression: ${label}`, pct, 'expression, not structure', 'not showing', 'showing')
  }
  return f(name, r3, '', 'low', 'high')
}

/** The Clavicular-style pillars each feature group rolls up into. */
export const PILLARS: Record<string, string> = {
  eyes: 'Harmony', nose: 'Harmony', mouth: 'Harmony', proportions: 'Harmony', symmetry: 'Harmony', shape: 'Harmony',
  brows: 'Dimorphism', jaw: 'Dimorphism',
  depth: 'Angularity', leanness: 'Angularity',
  skin: 'Softmax', colouring: 'Softmax', hair: 'Softmax',
  expression: 'Expression',
}
export const PILLAR_ORDER = ['Harmony', 'Dimorphism', 'Angularity', 'Softmax', 'Expression']
/** Individual features that belong to a different pillar than their group. */
export const PILLAR_OVERRIDES: Record<string, string> = { dimorphism: 'Dimorphism', lower_face_fullness: 'Angularity' }

export const GROUP_LABELS: Record<string, string> = {
  eyes: 'Eyes', brows: 'Brows', nose: 'Nose', mouth: 'Mouth & lips', jaw: 'Jaw & chin', proportions: 'Proportions',
  depth: 'Profile depth (3D)', symmetry: 'Symmetry', shape: 'Whole-face shape', skin: 'Skin', colouring: 'Colouring & contrast',
  hair: 'Hair & grooming', leanness: 'Leanness', expression: 'Expression',
}
