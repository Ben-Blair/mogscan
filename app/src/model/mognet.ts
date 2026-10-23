import * as ort from 'onnxruntime-web'
// The default ort build bundles its JS glue; only the .wasm binary is fetched. Importing it with
// ?url lets Vite serve (dev) and fingerprint (build) it locally, so no CDN is needed.
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'

const BASE = import.meta.env.BASE_URL
const SIZE = 224
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

ort.env.wasm.wasmPaths = { wasm: wasmUrl }

export interface MogNet {
  /** Runs the CNN on a square crop of the frame. Returns the 1–5 beauty score and P(male). */
  predict(source: CanvasImageSource, x: number, y: number, side: number): Promise<{ score: number; pMale: number }>
}

export async function loadMogNet(): Promise<MogNet> {
  const session = await ort.InferenceSession.create(`${BASE}models/mognet.onnx`, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  const canvas = new OffscreenCanvas(SIZE, SIZE)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  const input = new Float32Array(3 * SIZE * SIZE)

  return {
    async predict(source, x, y, side) {
      // Parts of the crop outside the frame stay black, matching PIL's crop() in training.
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, SIZE, SIZE)
      ctx.drawImage(source, x, y, side, side, 0, 0, SIZE, SIZE)
      const px = ctx.getImageData(0, 0, SIZE, SIZE).data
      const plane = SIZE * SIZE
      for (let i = 0; i < plane; i++) {
        input[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0]
        input[plane + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1]
        input[2 * plane + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2]
      }
      const out = await session.run({ image: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) })
      const score = (out.score.data as Float32Array)[0]
      const logit = (out.sex_logit.data as Float32Array)[0]
      return { score, pMale: 1 / (1 + Math.exp(-logit)) }
    },
  }
}
