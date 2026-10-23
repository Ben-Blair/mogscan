// Copies the MediaPipe wasm binaries into public/ so the demo runs fully offline (no CDN
// needed on demo day). Runs automatically after `npm install`. ONNX Runtime's wasm is
// imported through Vite instead (see src/model/mognet.ts).
import { cpSync, mkdirSync } from 'node:fs'

mkdirSync('public/mediapipe', { recursive: true })
cpSync('node_modules/@mediapipe/tasks-vision/wasm', 'public/mediapipe', { recursive: true })
console.log('copied wasm assets to public/')
