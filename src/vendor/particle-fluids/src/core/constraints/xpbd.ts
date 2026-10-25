// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Compute the XPBD Gauss-Seidel Lagrange-multiplier update for a single
 * constraint, per Macklin 2016 eq. (18):
 *
 *   Δλ = (−C − α̃ · λ) / (Σ_k w_k · |∇_k C|² + α̃)
 *
 * where:
 *   - `C` is the constraint residual `C(x*)`.
 *   - `sumGradSqInvMass` is `Σ_k (1/m_k) · |∇_k C|²` — the expansion of
 *     `∇C · M⁻¹ · ∇C^T` for a diagonal mass matrix (which we have: M is
 *     `diag(m_1·I₃, m_2·I₃, …)`, so M⁻¹ · ∇C^T is per-particle scalar
 *     multiply).
 *   - `alphaTilde` is `α / dt²` (eq. just above eq. (8)).
 *   - `lambdaCurrent` is `λ_i` — the accumulated multiplier for this
 *     constraint so far this substep. Must start at 0 each substep
 *     (Algorithm 1 line 4) and accumulate Δλ across solver iterations.
 *
 * Returns `Δλ` as a TSL float node. The caller adds it to `λ_c` (once per
 * constraint — see `distance.ts` for the "leader thread" convention that
 * avoids the atomic-free λ write) and applies it per-participant as
 * `Δx_k = w_k · ∇_k C · Δλ` per eq. (17).
 */
export function xpbdDeltaLambda(args: {
  readonly C: Any;
  readonly sumGradSqInvMass: Any;
  readonly alphaTilde: Any;
  readonly lambdaCurrent: Any;
}): Any {
  const { C, sumGradSqInvMass, alphaTilde, lambdaCurrent } = args;
  const numerator: Any = C.negate().sub(alphaTilde.mul(lambdaCurrent));
  const denominator: Any = sumGradSqInvMass.add(alphaTilde);
  return numerator.div(denominator);
}
