import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { appearanceCrop, computeAppearance } from './appearance'
import { analyzeFace, cropBox, type ShapeModel, type Vec3 } from './metrics'
import faces from './__fixtures__/parity.json'

const model = (f: string) => JSON.parse(readFileSync(new URL(`../../public/models/${f}`, import.meta.url), 'utf8'))
const template = model('face_template.json') as Vec3[]
const shape = model('shape.json') as ShapeModel

const close = (got: number, expected: number, k: string) =>
  expect(Math.abs(got - expected), k).toBeLessThan(1e-3 * Math.max(1, Math.abs(expected)))

describe('feature parity with Python reference', () => {
  for (const face of faces) {
    it(face.name, () => {
      const lm = face.landmarks.map(([x, y, z]) => ({ x, y, z }))
      const { metrics: m, pose } = analyzeFace(lm, face.w, face.h, template, shape)
      expect(Math.abs(pose.yaw - face.pose.yaw)).toBeLessThan(1e-3)
      expect(Math.abs(pose.pitch - face.pose.pitch)).toBeLessThan(1e-3)
      for (const [k, expected] of Object.entries(face.metrics)) close(m[k], expected, k)

      const crop = appearanceCrop(lm, face.w, face.h, cropBox(lm, face.w, face.h))
      const data = new Uint8ClampedArray(Buffer.from(face.crop, 'base64'))
      const a = computeAppearance({ data, size: face.cropSize }, crop.pts)
      for (const [k, expected] of Object.entries(face.appearance)) close(a[k], expected, k)
    })
  }
})
