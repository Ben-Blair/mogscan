import { useState } from 'react'
import type { PersonSnapshot } from '../engine'
import { featureInfo, GROUP_LABELS } from '../scoring/features'
import { strengthsWeaknesses, tierName, type FeatureResult } from '../scoring/scoring'
import type { GlowUpPlan } from '../scoring/glowup'
import type { Rating } from '../scoring/scoring'
import { tierColor } from './draw'

const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

function FeatureList({ title, items, kind }: { title: string; items: FeatureResult[]; kind: 'good' | 'bad' }) {
  if (!items.length) return null
  return (
    <section className={`traits ${kind}`}>
      <h3>{title}</h3>
      <ul>
        {items.map((f) => (
          <li key={f.name}>
            <span className="trait-name">{featureInfo(f.name).label}</span>
            <span className="trait-desc">{sentence(f.description)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

/** Plain list of every measurement: your value next to the top-rated average. */
function Measurements({ features }: { features: FeatureResult[] }) {
  const groups = [...new Set(features.map((f) => f.group))].filter((g) => g !== 'expression')
  return (
    <div className="measure">
      <div className="measure-row measure-head"><span /><span>You</span><span>Top-rated</span></div>
      {groups.map((g) => (
        <section key={g}>
          <h4>{GROUP_LABELS[g] ?? g}</h4>
          {features.filter((f) => f.group === g && !f.name.startsWith('shape_pc')).map((f) => {
            const info = featureInfo(f.name)
            return (
              <div key={f.name} className="measure-row">
                <span>{info.label}</span>
                <span className={`measure-val ${f.psl >= 0.02 ? 'good' : f.psl <= -0.02 ? 'bad' : ''}`}>{info.fmt(f.value)}</span>
                <span className="measure-ref">{info.fmt(f.target)}</span>
              </div>
            )
          })}
        </section>
      ))}
    </div>
  )
}

/** Compact weight: two decimals, no leading zero (−.05). */
const w2 = (v: number) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2).replace(/^0/, '')}`
const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`

/** Every model input, grouped, with its value, learned weights and effect on this face. */
function AllFeatures({ features }: { features: FeatureResult[] }) {
  const groups = [...new Set(features.map((f) => f.group))]
  return (
    <div className="all-wrap">
      <p className="muted small">
        Weights are learned per sex. <i>w</i> = linear weight, <i>q</i> = ideal-range (quadratic) weight, both in
        rating points per SD. <i>imp</i> = how far this input moves a typical face (PSL). <i>effect</i> = what it
        does to your score.
      </p>
      <table className="all">
        <thead><tr><th>Input</th><th>You</th><th>Top</th><th>z</th><th>w</th><th>q</th><th>imp</th><th>effect</th></tr></thead>
        {groups.map((g) => {
          const fs = features.filter((f) => f.group === g).sort((a, b) => b.importance - a.importance)
          const total = fs.reduce((t, f) => t + f.psl, 0)
          return (
            <tbody key={g}>
              <tr className="group-row"><th colSpan={7}>{GROUP_LABELS[g] ?? g} <span className="muted">({fs.length})</span></th>
                <th className={total >= 0 ? 'pos' : 'neg'}>{signed(total)}</th></tr>
              {fs.map((f) => {
                const info = featureInfo(f.name)
                return (
                  <tr key={f.name} className={f.measured ? '' : 'unmeasured'}>
                    <td>{info.label}</td>
                    <td>{f.measured ? info.fmt(f.value) : '…'}</td>
                    <td>{info.fmt(f.target)}</td>
                    <td>{f.z.toFixed(1)}</td>
                    <td>{w2(f.lin)}</td>
                    <td>{w2(f.quad)}</td>
                    <td>{w2(f.importance)}</td>
                    <td className={f.psl >= 0 ? 'pos' : 'neg'}>{signed(f.psl)}</td>
                  </tr>
                )
              })}
            </tbody>
          )
        })}
      </table>
    </div>
  )
}

function Plan({ plan }: { plan: GlowUpPlan }) {
  if (!plan.items.length) return <p className="hint plan-empty">Nothing in the plan would raise your score noticeably.</p>
  return (
    <div className="plan">
      <ol>
        {plan.items.map((it) => (
          <li key={it.id}>
            <div className="plan-head">
              <span className="plan-title">{it.title}</span>
              <span className="plan-gain">+{it.gain.toFixed(2)}</span>
            </div>
            {it.reasons.length > 0 && (
              <ul className="plan-why">
                {it.reasons.map((rs) => (
                  <li key={rs.name}><span>{rs.label}</span><span>you {rs.you} · top-rated {rs.target}</span></li>
                ))}
              </ul>
            )}
            {it.tip && <p className="plan-tip">{it.tip}</p>}
            <p className="plan-how">{it.how}</p>
            <span className="plan-pred">Predicted {plan.current.toFixed(2)} → {it.predicted.toFixed(2)}</span>
          </li>
        ))}
      </ol>
      <p className="plan-note">
        Estimates from the scoring model, assuming each change gets you halfway to the average of top-rated faces.
        Rescan after a change to see your real result. Nothing here involves surgery or UV tanning.
      </p>
    </div>
  )
}

/** Developer view: model split and every input's learned weights. */
function Technical({ r }: { r: Rating }) {
  return (
    <>
      <p className="tech-line">
        CNN {r.cnn?.toFixed(2) ?? '—'} ({signed(r.cnnPsl ?? 0)} PSL) · feature model {r.geo.toFixed(2)}
        {' '}({signed(r.geoPsl)} PSL) · fused {r.fused.toFixed(2)}/5
      </p>
      <AllFeatures features={r.features} />
    </>
  )
}

export function PersonCard({ p, showId, onToggleSex, onRescan }: {
  p: PersonSnapshot; showId: boolean; onToggleSex: () => void; onRescan: () => void
}) {
  const [view, setView] = useState<'summary' | 'plan' | 'measurements' | 'technical'>('summary')
  const r = p.rating
  const id = showId ? <span className="pid">{p.id}</span> : null
  if (!r) {
    return (
      <article className="card pending">
        <header>{id}<span>{p.frontal ? 'Scanning' : 'Face the camera'}</span></header>
        <div className="progress"><span style={{ width: `${p.progress * 100}%` }} /></div>
        <p className="hint">{p.frontal ? 'Hold still and keep a neutral expression.'
          : 'Look straight at the camera to start the scan.'}</p>
      </article>
    )
  }
  const { strengths, weaknesses } = strengthsWeaknesses(r.features)
  const color = tierColor(r.psl)
  const pillars = r.pillars.filter((pl) => pl.name !== 'Expression')
  return (
    <article className="card" style={{ ['--tier' as string]: color }}>
      <header>{id}</header>
      <div className="score">
        <div className="score-num">{r.psl.toFixed(1)}</div>
        <div className="score-meta">
          <div className="tier">{r.tier}</div>
          <div className="pct">{tierName(r.tier)} · top {Math.max(1, Math.round(100 - r.percentile))}%</div>
        </div>
      </div>

      <div className="pillars">
        {pillars.map((pl) => (
          <div key={pl.name} className="pillar">
            <span className="pillar-name">{pl.name}</span>
            <span className="pillar-bar"><span style={{ width: `${pl.rating * 10}%` }} /></span>
            <span className="pillar-score">{pl.rating.toFixed(1)}</span>
          </div>
        ))}
      </div>

      <FeatureList title="Strengths" items={strengths} kind="good" />
      <FeatureList title="To improve" items={weaknesses} kind="bad" />

      {p.plan && p.plan.potential - r.psl >= 0.05 && (
        <button className="potential" onClick={() => setView(view === 'plan' ? 'summary' : 'plan')}>
          <span>
            <span className="potential-label">Your potential</span>
            <span className="potential-score">{r.psl.toFixed(2)} → <b>{p.plan.potential.toFixed(2)}</b></span>
          </span>
          <span className="potential-cta">{view === 'plan' ? 'Hide plan' : 'See glow-up plan'}</span>
        </button>
      )}
      {view === 'plan' && p.plan && <Plan plan={p.plan} />}

      <nav className="card-tabs">
        <button className={view === 'measurements' ? 'active' : ''}
          onClick={() => setView(view === 'measurements' ? 'summary' : 'measurements')}>All measurements</button>
        <button className={view === 'technical' ? 'active' : ''}
          onClick={() => setView(view === 'technical' ? 'summary' : 'technical')}>Model weights</button>
      </nav>
      {view === 'measurements' && <Measurements features={r.features} />}
      {view === 'technical' && <Technical r={r} />}

      <footer>
        <button className="text" onClick={onToggleSex} title="Scores are calibrated separately for men and women">
          Scored as {p.sex === 'M' ? 'male' : 'female'} · change</button>
        <button className="text" onClick={onRescan}>Rescan</button>
      </footer>
    </article>
  )
}
