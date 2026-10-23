import { useEffect, useRef, useState } from 'react'
import { Engine, loadModels, type Models, type PersonSnapshot } from './engine'
import { PersonCard, type Tone } from './ui/PersonCard'

const FEEDBACK_URL = import.meta.env.VITE_FEEDBACK_URL as string | undefined
const CONSENT_KEY = 'mogscan-consent-v1'
// ?nocam skips getUserMedia (photo-only mode, e.g. for testing without a webcam).
const NO_CAM = new URLSearchParams(location.search).has('nocam')

function readConsent() {
  try { return localStorage.getItem(CONSENT_KEY) === 'yes' } catch { return false }
}

export default function App() {
  const [consented, setConsented] = useState(readConsent)
  const [status, setStatus] = useState('Starting…')
  const [models, setModels] = useState<Models | null>(null)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState<string>('')
  const [photo, setPhoto] = useState<string | null>(null)
  const [people, setPeople] = useState<PersonSnapshot[]>([])
  const [fps, setFps] = useState(0)
  const [tone, setTone] = useState<Tone>('kind')
  const [mirror, setMirror] = useState(true)
  const [showMesh, setShowMesh] = useState(false)
  const [showGuides, setShowGuides] = useState(true)
  const [battle, setBattle] = useState(false)
  const [recording, setRecording] = useState(false)
  const logRef = useRef<object[]>([])
  const recordingRef = useRef(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)

  useEffect(() => {
    if (!consented) return
    loadModels(setStatus).then((m) => { setModels(m); setStatus('') })
      .catch((e) => { console.error(e); setStatus(`Failed to load models: ${e}`) })
  }, [consented])

  // Camera: ask permission once, then list devices so a USB camera can be picked.
  useEffect(() => {
    if (!models || photo || NO_CAM) return
    let stream: MediaStream | null = null
    let cancelled = false
    const video = { width: { ideal: 1280 }, height: { ideal: 720 }, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) }
    navigator.mediaDevices.getUserMedia({ video, audio: false }).then(async (s) => {
      if (cancelled) { s.getTracks().forEach((t) => t.stop()); return }
      stream = s
      const v = videoRef.current!
      v.srcObject = s
      await v.play()
      const all = await navigator.mediaDevices.enumerateDevices()
      setDevices(all.filter((d) => d.kind === 'videoinput'))
      if (!deviceId) setDeviceId(s.getVideoTracks()[0].getSettings().deviceId ?? '')
    }).catch((e) => setStatus(`Camera unavailable (${e.name}). You can still upload a photo.`))
    return () => { cancelled = true; stream?.getTracks().forEach((t) => t.stop()) }
  }, [models, deviceId, photo])

  useEffect(() => {
    if (!models) return
    const engine = new Engine(models, canvasRef.current!,
      () => (photo ? imgRef.current : videoRef.current), { mirror, showMesh, showGuides })
    engineRef.current = engine
    engine.start()
    const id = setInterval(() => {
      const snap = engine.snapshot()
      setPeople(snap)
      setFps(engine.fps)
      if (recordingRef.current) {
        // Study log: numbers only, never pixels.
        logRef.current.push({ t: Math.round(performance.now()), fps: engine.fps, people: snap.map((p) => ({
          id: p.id, frontal: p.frontal, yaw: +p.yaw.toFixed(1), sex: p.sex,
          psl: p.rating ? +p.rating.psl.toFixed(3) : null, tier: p.rating?.tier ?? null,
          cnn: p.rating?.cnn ?? null, geo: p.rating ? +p.rating.geo.toFixed(3) : null,
        })) })
      }
    }, 200)
    return () => { engine.stop(); clearInterval(id) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, photo])

  useEffect(() => {
    if (engineRef.current) engineRef.current.opts = { mirror: mirror && !photo, showMesh, showGuides }
  }, [mirror, showMesh, showGuides, photo])

  const toggleSex = (p: PersonSnapshot) => engineRef.current?.sexOverride.set(p.id, p.sex === 'M' ? 'F' : 'M')

  const snapshot = () => {
    const a = document.createElement('a')
    a.download = `mogscan-${Date.now()}.png`
    a.href = canvasRef.current!.toDataURL('image/png')
    a.click()
  }

  const toggleRecording = () => {
    if (!recording) {
      logRef.current = []
      recordingRef.current = true
      setRecording(true)
      return
    }
    recordingRef.current = false
    setRecording(false)
    const blob = new Blob([JSON.stringify({ app: 'mogscan', version: 1, userAgent: navigator.userAgent,
      frames: logRef.current }, null, 1)], { type: 'application/json' })
    const a = document.createElement('a')
    a.download = `mogscan-session-${Date.now()}.json`
    a.href = URL.createObjectURL(blob)
    a.click()
  }

  const onPhoto = (f: File | undefined) => {
    if (!f) return
    if (photo) URL.revokeObjectURL(photo)
    setPhoto(URL.createObjectURL(f))
  }

  if (!consented) {
    return (
      <div className="gate">
        <div className="gate-card">
          <h1>MogScan</h1>
          <p className="lede">Real-time, multi-person facial aesthetics analysis.</p>
          <ul>
            <li>Everything runs <b>on this device</b>. No images are uploaded or stored.</li>
            <li>Scores are a model's estimate of how one panel of 60 raters judged faces. They are not a
              measure of your worth and they carry the biases of that data.</li>
            <li>For adults (18+) only. Only scan people who have agreed to be scanned.</li>
          </ul>
          <button className="primary" onClick={() => {
            try { localStorage.setItem(CONSENT_KEY, 'yes') } catch { /* private mode */ }
            setConsented(true)
          }}>I'm 18+ and everyone in frame agrees</button>
        </div>
      </div>
    )
  }

  const rated = people.filter((p) => p.rating)
  return (
    <div className="app">
      <header className="top">
        <div className="brand">MOG<span>SCAN</span></div>
        <div className="controls">
          <select value={deviceId} onChange={(e) => { setPhoto(null); setDeviceId(e.target.value) }} aria-label="Camera">
            {devices.length === 0 && <option value="">Camera</option>}
            {devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
          </select>
          <label className="file">Photo<input type="file" accept="image/*" onChange={(e) => onPhoto(e.target.files?.[0])} /></label>
          <button className={tone === 'brutal' ? 'on' : ''} onClick={() => setTone(tone === 'kind' ? 'brutal' : 'kind')}>
            {tone === 'kind' ? 'Kind' : 'Brutal'} mode</button>
          <button className={battle ? 'on' : ''} onClick={() => setBattle(!battle)}>Mog battle</button>
          <button className={showGuides ? 'on' : ''} onClick={() => setShowGuides(!showGuides)}>Guides</button>
          <button className={showMesh ? 'on' : ''} onClick={() => setShowMesh(!showMesh)}>Mesh</button>
          <button className={mirror ? 'on' : ''} onClick={() => setMirror(!mirror)}>Mirror</button>
          <button onClick={snapshot}>Snapshot</button>
          <button className={recording ? 'rec' : ''} onClick={toggleRecording}>{recording ? '■ Stop & save log' : '● Record'}</button>
          {FEEDBACK_URL && <a className="btn" href={FEEDBACK_URL} target="_blank" rel="noreferrer">Feedback</a>}
        </div>
      </header>

      <main className="stage">
        <section className="viewport">
          <canvas ref={canvasRef} />
          <video ref={videoRef} playsInline muted hidden />
          {photo && <img ref={imgRef} src={photo} alt="" hidden />}
          {status && <div className="status">{status}</div>}
          <div className="hud">{fps} fps · {people.length} face{people.length === 1 ? '' : 's'} · on-device</div>
        </section>

        <aside className="panel">
          {battle && rated.length >= 2 && (
            <div className="battle">
              <div className="battle-title">MOG BATTLE</div>
              <ol>{rated.map((p) => <li key={p.id}><b>P{p.id}</b> {p.rating!.tier}<span>{p.rating!.psl.toFixed(2)}</span></li>)}</ol>
              <div className="verdict">P{rated[0].id} mogs {rated.slice(1).map((p) => `P${p.id}`).join(', ')}
                {' '}by {(rated[0].rating!.psl - rated[1].rating!.psl).toFixed(2)} PSL</div>
            </div>
          )}
          {people.length === 0 && !status && <div className="empty">Step into frame. Up to 6 people at once.</div>}
          {people.map((p, i) => (
            <PersonCard key={p.id} p={p} tone={tone} rank={p.rating ? i + 1 : 0} onToggleSex={() => toggleSex(p)} />
          ))}
        </aside>
      </main>
    </div>
  )
}
