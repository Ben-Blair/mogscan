import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const BASE = import.meta.env.BASE_URL

export async function createLandmarker(maxFaces = 6): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}mediapipe`)
  const options = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
    runningMode: 'VIDEO' as const,
    numFaces: maxFaces,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  })
  try {
    return await FaceLandmarker.createFromOptions(fileset, options('GPU'))
  } catch {
    return FaceLandmarker.createFromOptions(fileset, options('CPU'))
  }
}
