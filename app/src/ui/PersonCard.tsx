import { useState } from 'react'
import type { PersonSnapshot } from '../engine'
import { FEATURES } from '../scoring/features'
import { strengthsWeaknesses, type FeatureResult } from '../scoring/scoring'
import { tierColor } from './draw'

export type Tone = 'kind' | 'brutal'

const PREFIX: Record<Tone, { good: string; bad: string }> = {
  kind: { good: 'Strength', bad: 'Room to grow' },
  brutal: { good: 'Halo', bad: 'Failo' },
}

function FeatureRow({ f, tone }: { f: FeatureResult; tone: Tone }) {
  const info = FEATURES[f.name]
  const good = f.psl >= 0
  return (
    <li className={`feat ${good ? 'good' : 'bad'}`}>
      <div className="feat-head">
        <span className="feat-tag">{good ? PREFIX[tone].good : PREFIX[tone].bad}</span>
        <span className="feat-name">{info.label}</span>
        <span className="feat-delta">{good ? '+' : '−'}{Math.abs(f.psl).toFixed(2)} PSL</span>
      </div>
      <div className="feat-body">
        <b>{info.fmt(f.value)}</b> · {f.description}
        <span className="feat-ref"> top-tier avg {info.fmt(f.target)} · community: {info.community}</span>
      </div>
    </li>
  )
}

export function PersonCard({ p, tone, rank, onToggleSex }: {
  p: PersonSnapshot; tone: Tone; rank: number; onToggleSex: () => void
}) {
  const [open, setOpen] = useState(false)
  const r = p.rating
  if (!r) {
    return (
      <article className="card pending">
        <header><span className="pid">P{p.id}</span><span className="muted">
          {p.frontal ? 'Scanning… hold still' : `Turn toward the camera (yaw ${p.yaw.toFixed(0)}°)`}</span></header>
      </article>
    )
  }
  const { strengths, weaknesses } = strengthsWeaknesses(r.features)
  const color = tierColor(r.psl)
  return (
    <article className="card" style={{ ['--tier' as string]: color }}>
      <header>
        <span className="rank">#{rank}</span>
        <span className="pid">P{p.id}</span>
        <button className="sex" onClick={onToggleSex} title="Click to correct the detected sex">
          {p.sex === 'M' ? '♂' : '♀'}{p.sexAuto ? ' auto' : ''}
        </button>
      </header>
      <div className="score-row">
        <div className="psl">{r.psl.toFixed(2)}<small>PSL</small></div>
        <div className="tier-block">
          <div className="tier">{r.tier}</div>
          <div className="muted">top {Math.max(0.1, 100 - r.percentile).toFixed(1)}% · rating {r.fused.toFixed(2)}/5</div>
          <div className="bar"><span style={{ width: `${((r.psl - 1) / 7) * 100}%` }} /></div>
        </div>
      </div>
      <div className="models muted">CNN {r.cnn?.toFixed(2) ?? '—'} · geometry {r.geo.toFixed(2)} · fused {r.fused.toFixed(2)}</div>
      <ul className="feats">
        {strengths.map((f) => <FeatureRow key={f.name} f={f} tone={tone} />)}
        {weaknesses.map((f) => <FeatureRow key={f.name} f={f} tone={tone} />)}
      </ul>
      <button className="link" onClick={() => setOpen(!open)}>{open ? 'Hide' : 'Show'} all 15 measurements</button>
      {open && (
        <table className="all">
          <thead><tr><th>Metric</th><th>You</th><th>Top-tier avg</th><th>Score</th></tr></thead>
          <tbody>
            {[...r.features].sort((a, b) => b.psl - a.psl).map((f) => (
              <tr key={f.name}>
                <td>{FEATURES[f.name].label}</td>
                <td>{FEATURES[f.name].fmt(f.value)}</td>
                <td>{FEATURES[f.name].fmt(f.target)}</td>
                <td><span className="mini" style={{ width: `${f.rating * 10}%` }} />{f.rating.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </article>
  )
}
