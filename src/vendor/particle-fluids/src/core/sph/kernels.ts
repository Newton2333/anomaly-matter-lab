import { float, uniform } from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * SPH smoothing kernels from Müller et al. 2003, as used by Position Based
 * Fluids (Macklin & Müller 2013, §3). The `h`-dependent coefficients are
 * computed on the CPU: setting `h.value` recomputes `hSq`, `poly6Coef`, and
 * `spikyCoef`.
 */
export interface SphKernelUniforms {
  /** Smoothing radius. Setting its value updates the other three; it must stay positive. */
  readonly h: UniformNode<'float', number>;
  readonly hSq: UniformNode<'float', number>;
  readonly poly6Coef: UniformNode<'float', number>;
  readonly spikyCoef: UniformNode<'float', number>;
}

/** Uniforms for the Poly6 and Spiky kernels (Müller et al. 2003) with smoothing radius `h`. */
export function createSphKernelUniforms(h: number): SphKernelUniforms {
  assertRadius(h);
  const hSq = uniform(0, 'float');
  const poly6Coef = uniform(0, 'float');
  const spikyCoef = uniform(0, 'float');
  const hNode = uniform(h, 'float');
  // Replace the uniform's value field with an accessor, so writes to it keep
  // the derived coefficients in step. The renderer reads `value` each dispatch.
  let current = h;
  const derive = (value: number): void => {
    hSq.value = value * value;
    poly6Coef.value = 315 / (64 * Math.PI * value ** 9);
    spikyCoef.value = 45 / (Math.PI * value ** 6);
  };
  Object.defineProperty(hNode, 'value', {
    get: () => current,
    set: (value: number) => {
      assertRadius(value);
      current = value;
      derive(value);
    },
    enumerable: true,
    configurable: true,
  });
  derive(h);
  return { h: hNode, hSq, poly6Coef, spikyCoef };
}

function assertRadius(h: number): void {
  if (!Number.isFinite(h) || h <= 0) {
    throw new Error(`createSphKernelUniforms: h must be positive, got ${h}`);
  }
}

export function emitPoly6(r_vec: Any, u: SphKernelUniforms): Any {
  const rSq: Any = r_vec.dot(r_vec);
  return emitPoly6FromRSq(rSq, u);
}

/**
 * {@link emitPoly6} variant that reuses a pre-computed `|r|²` scalar.
 */
export function emitPoly6FromRSq(rSq: Any, u: SphKernelUniforms): Any {
  const d: Any = (u.hSq as Any).sub(rSq).max(float(0.0));
  return d
    .mul(d)
    .mul(d)
    .mul(u.poly6Coef as Any);
}

/**
 * Emit Spiky `∇W(r_vec, h) = −spikyCoef · (h − |r|)² · r̂` as a vec3 TSL
 * node.
 *
 * **Sign convention**: the returned vector is the gradient taken wrt the
 * argument `r_vec`. If a caller uses `r_vec = x_i − x_j`, this returns
 * `∇_{p_i} W(p_i − p_j, h)`. For the `∇_{p_j}` form (as paper eq. 8 writes
 * for the `k = j` case, and paper eq. 15 writes for vorticity), negate the
 * returned vector — or equivalently pass `r_vec = x_j − x_i`.
 *
 * Returns zero-vector for `|r| ≥ h` (via the `(h − r)²` clamp) and for
 * `|r| → 0` (direction is undefined at the origin; the SPH convention is
 * that a particle contributes zero gradient to itself, which is consistent
 * with the gradient at `r = 0` being a point of ambiguity in the kernel
 * itself). The `rSafe = max(r, 1e-20)` guard prevents NaN at true-zero
 * separation.
 */
export function emitSpikyGrad(r_vec: Any, u: SphKernelUniforms): Any {
  const rSq: Any = r_vec.dot(r_vec).toVar();
  const r: Any = rSq.sqrt().toVar();
  // `(h − r)²`, clamped to 0 so outside-radius pairs produce zero.
  const t: Any = (u.h as Any).sub(r).max(float(0.0)).toVar();
  // Guard against `r = 0`: the spiky-gradient magnitude stays finite as
  // `r → 0` (it tends to `spikyCoef · h²`), but the direction `r̂` is
  // undefined. Returning zero there matches the conventional SPH choice
  // (Müller 2003, Monaghan 1992) — a particle contributes no gradient to
  // itself.
  const rSafe: Any = r.max(float(1e-20));
  const rHat: Any = r_vec.div(rSafe);
  // Combine magnitude · direction. `.negate()` applies the paper's
  // explicit minus sign in `−(45/π·h⁶)·(h − r)² · r̂`.
  const mag: Any = t
    .mul(t)
    .mul(u.spikyCoef as Any)
    .negate();
  return rHat.mul(mag);
}
