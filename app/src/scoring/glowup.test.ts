import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { OneEuro } from '../vision/oneEuro'
import { appearanceCrop, computeAppearance } from './appearance'
import { glowUpPlan } from './glowup'
import { analyzeFace, cropBox, type ShapeModel, type Vec3 } from './metrics'
import { rate, type FusionModel, type GeoModel } from './scoring'
import faces from './__fixtures__/parity.json'

const model = (f: string) => JSON.parse(readFileSync(new URL(`../../public/models/${f}`, import.meta.url), 'utf8'))
const geo = model('geometric.json') as GeoModel
const fusion = model('fusion.json') as FusionModel

describe('glow-up plan', () => {
  for (const face of faces.slice(0, 3)) {
    it(face.name, () => {
      const lm = face.landmarks.map(([x, y, z]) => ({ x, y, z }))
      const { metrics } = analyzeFace(lm, face.w, face.h, model('face_template.json') as Vec3[], model('shape.json') as ShapeModel)
      const crop = appearanceCrop(lm, face.w, face.h, cropBox(lm, face.w, face.h))
      const app = computeAppearance({ data: new Uint8ClampedArray(Buffer.from(face.crop, 'base64')), size: face.cropSize }, crop.pts)
      const sex = face.name[1] === 'M' ? 'M' : 'F'
      const r = rate({ ...metrics, ...app }, 3, sex, geo, fusion)
      const plan = glowUpPlan(r, sex, geo, fusion)
      expect(plan.potential).toBeGreaterThanOrEqual(r.psl)
      expect(plan.potential).toBeLessThanOrEqual(8)
      for (const it of plan.items) {
        expect(it.gain).toBeGreaterThan(0)
        expect(it.reasons.length).toBeGreaterThan(0)
        expect(it.predicted).toBeLessThanOrEqual(plan.potential + 1e-9)
        expect(it.how.toLowerCase()).not.toMatch(/surgery|filler|botox|tanning bed/)
      }
      if (sex === 'F') expect(plan.items.find((i) => i.id === 'facialhair')).toBeUndefined()
    })
  }
})

describe('One Euro filter', () => {
  it('removes jitter on a still signal and follows a moving one', () => {
    const f = new OneEuro(0.8, 0.015)
    let maxDev = 0
    for (let i = 0; i < 300; i++) {
      const v = f.filter(100 + (i % 2 ? 3 : -3), i * 33)
      if (i > 60) maxDev = Math.max(maxDev, Math.abs(v - 100))
    }
    expect(maxDev).toBeLessThan(1)
    const g = new OneEuro(0.8, 0.015)
    let v = 0
    for (let i = 0; i < 30; i++) v = g.filter(i * 20, i * 33)  // ~600 px/s
    expect(Math.abs(v - 29 * 20)).toBeLessThan(40)
  })
})
