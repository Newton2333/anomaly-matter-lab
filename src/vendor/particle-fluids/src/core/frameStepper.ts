/** Fraction of a step that still counts as a whole step, absorbing float rounding. */
const STEP_TOLERANCE = 1e-6;

export interface FrameStepperOptions {
  /** Simulation time advanced per step, in seconds. Typically `1 / 60`. */
  readonly fixedDt: number;
  /**
   * Most steps run in one frame when catching up after a stall. Default 4.
   * Time beyond this is dropped, so a slow GPU makes the simulation run
   * slower than real time instead of falling further and further behind.
   */
  readonly maxStepsPerFrame?: number;
}

export interface FrameStepperResult {
  /** Steps run this frame. */
  readonly steps: number;
  /** True if time was dropped this frame because the simulation fell behind. */
  readonly truncated: boolean;
  /** Time carried over to the next frame, in `[0, fixedDt)` seconds. */
  readonly remainderSeconds: number;
}

/**
 * Runs a fixed-timestep simulation at wall-clock speed ("Fix Your Timestep!",
 * Gaffer on Games). Each frame, elapsed time is added to an accumulator and
 * whole steps of `fixedDt` are taken from it, so the solver always sees the
 * timestep it was tuned for no matter the display's frame rate.
 *
 * ```ts
 * const stepper = new FrameStepper({ fixedDt: 1 / 60 });
 * await stepper.pump(performance.now(), (dt) => loop.step(dt));
 * ```
 */
export class FrameStepper {
  readonly fixedDt: number;
  readonly maxStepsPerFrame: number;
  private accumulator = 0;
  private lastMs: number | null = null;

  constructor(options: FrameStepperOptions) {
    if (!Number.isFinite(options.fixedDt) || options.fixedDt <= 0) {
      throw new Error(`FrameStepper: fixedDt must be positive, got ${options.fixedDt}`);
    }
    const cap = options.maxStepsPerFrame ?? 4;
    if (!Number.isInteger(cap) || cap <= 0) {
      throw new Error(`FrameStepper: maxStepsPerFrame must be a positive integer, got ${cap}`);
    }
    this.fixedDt = options.fixedDt;
    this.maxStepsPerFrame = cap;
  }

  /**
   * Take as many fixed steps as the time since the previous call allows.
   * The first call only starts the clock.
   *
   * @param nowMs Current time in milliseconds, e.g. `performance.now()`.
   * @param step Advances the simulation by exactly `dt` seconds.
   */
  async pump(nowMs: number, step: (dt: number) => Promise<void>): Promise<FrameStepperResult> {
    if (this.lastMs === null) {
      this.lastMs = nowMs;
      return { steps: 0, truncated: false, remainderSeconds: 0 };
    }
    this.accumulator += Math.max(0, (nowMs - this.lastMs) / 1000);
    this.lastMs = nowMs;

    // Count whole steps instead of subtracting fixedDt repeatedly, so float
    // rounding can't leave a step's worth of time behind. The tolerance
    // absorbs rounding in the millisecond sums (a millionth of a step).
    const available = this.accumulator / this.fixedDt;
    const truncated = available > this.maxStepsPerFrame + STEP_TOLERANCE;
    const steps = truncated
      ? this.maxStepsPerFrame
      : Math.min(Math.floor(available + STEP_TOLERANCE), this.maxStepsPerFrame);
    // Truncation drops everything past the last step.
    this.accumulator = truncated ? 0 : Math.max(0, this.accumulator - steps * this.fixedDt);

    for (let i = 0; i < steps; i++) await step(this.fixedDt);
    return { steps, truncated, remainderSeconds: this.accumulator };
  }

  /** Forget elapsed time, e.g. after a pause, so the next call doesn't catch up. */
  reset(): void {
    this.accumulator = 0;
    this.lastMs = null;
  }
}
