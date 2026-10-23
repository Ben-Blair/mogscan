import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { analyzeFace, METRIC_NAMES, type Vec3 } from './metrics'
import faces from './__fixtures__/parity.json'

const template = JSON.parse(readFileSync(new URL('../../public/models/face_template.json', import.meta.url), 'utf8')) as Vec3[]

describe('metrics parity with Python reference', () => {
  for (const face of faces) {
    it(face.name, () => {
      const lm = face.landmarks.map(([x, y, z]) => ({ x, y, z }))
      const { metrics: m, pose } = analyzeFace(lm, face.w, face.h, template)
      expect(Math.abs(pose.yaw - face.pose.yaw)).toBeLessThan(1e-3)
      expect(Math.abs(pose.pitch - face.pose.pitch)).toBeLessThan(1e-3)
      for (const k of METRIC_NAMES) {
        const expected = (face.metrics as Record<string, number>)[k]
        expect(Math.abs(m[k] - expected), k).toBeLessThan(1e-3 * Math.max(1, Math.abs(expected)))
      }
    })
  }
})
