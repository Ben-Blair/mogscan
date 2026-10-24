/** One Euro filter (Casiez et al., CHI 2012): a low-pass filter whose cutoff rises with speed, so a
 * still face is smoothed heavily (no jitter) while a moving face is followed with little lag. */
export class OneEuro {
  private x: number | null = null
  private dx = 0
  private t = 0
  private minCutoff: number
  private beta: number
  private dCutoff: number

  /** minCutoff (Hz): smoothing when still. beta: how fast the cutoff grows with speed. */
  constructor(minCutoff = 1.0, beta = 0.02, dCutoff = 1.0) {
    this.minCutoff = minCutoff
    this.beta = beta
    this.dCutoff = dCutoff
  }

  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff)
    return 1 / (1 + tau / dt)
  }

  /** Feed a new value at time `tMs`; returns the filtered value. */
  filter(value: number, tMs: number): number {
    if (this.x === null) { this.x = value; this.t = tMs; return value }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000)
    this.t = tMs
    const rawDx = (value - this.x) / dt
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (rawDx - this.dx)
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx)
    this.x += OneEuro.alpha(cutoff, dt) * (value - this.x)
    return this.x
  }
}
