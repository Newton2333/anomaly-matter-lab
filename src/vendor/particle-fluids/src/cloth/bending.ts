import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import { Fn, If, Return, atan, cross, float, instanceIndex, instancedArray, vec4 } from 'three/tsl';

import {
  buildConstraintGroups,
  colorConstraints,
  xpbdDeltaLambda,
  type ConstraintType,
  type ParticleSystem,
} from '../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Cloth bending constraints on the dihedral angle between two triangles
 * sharing an edge (Bender et al. 2014 §3.4.2, with the gradients of
 * Bridson et al. 2003 §4).
 *
 * For a tuple `(p1, p2, p3, p4)`, where `(p1, p2)` is the shared edge and
 * `p3`, `p4` are the far vertices of the two triangles:
 *
 *   `E   = p2 − p1`
 *   `N_1 = (p3 − p1) × (p3 − p2)`,  `N_2 = (p4 − p2) × (p4 − p1)`
 *   `θ   = atan2((N_1 × N_2)·E, (N_1·N_2)·|E|)`
 *   `C   = θ − φ_0`
 *
 * `θ` is the signed dihedral angle in (−π, π], zero for a flat pair. A
 * signed angle keeps the gradient consistent on both sides of flat, where
 * `acos(n̂_1·n̂_2)` would fold both bend directions onto the same value.
 * `φ_0` is measured from the input geometry by `createClothGraph`, so the
 * cloth rests in its modelled shape.
 *
 * Gradients (Bridson's closed forms, negated because his `u_k` is the
 * gradient of `π − θ`; `tests/analytical/cloth/bending-gradient.test.ts`
 * checks them against finite differences):
 *
 *   `∇_{p3} C = −|E| · N_1/|N_1|²`
 *   `∇_{p4} C = −|E| · N_2/|N_2|²`
 *   `∇_{p1} C = −((p3−p2)·E/|E| · N_1/|N_1|² + (p4−p2)·E/|E| · N_2/|N_2|²)`
 *   `∇_{p2} C = (p3−p1)·E/|E| · N_1/|N_1|² + (p4−p1)·E/|E| · N_2/|N_2|²`
 *
 * These stay finite at `θ ∈ {0, π}`. The XPBD update is Macklin et al.
 * 2016, eq. 17–18. Tuples are colored so no two in a group share a
 * particle, and each is solved by one thread that moves all four
 * particles. Pinned particles (`invMass = 0`) drop out of both the
 * denominator and their own correction.
 */
export function createClothBendingConstraints(args: {
  readonly particles: ParticleSystem;
  /** Absolute slot offset of this cloth's first particle. */
  readonly particleOffset: number;
  /**
   * Cloth-local bending tuples `[p1, p2, p3, p4]` with
   * `0 ≤ p_k < nClothParticles`. Typically the per-tuple entry from
   * {@link "./graph.js".ClothGraph.bendingTuples}.
   */
  readonly tuples: readonly (readonly [number, number, number, number])[];
  /**
   * Signed rest dihedral angle per tuple, in radians. Typically
   * {@link "./graph.js".ClothGraph.bendingRestAngles}.
   */
  readonly restAngles: readonly number[];
  /**
   * XPBD compliance `α`, the same for every tuple. `C` is an angle, so `α`
   * is the inverse of an angular stiffness: rad²/(N·m) = rad²·s²/(kg·m²).
   */
  readonly compliance: number;
  readonly dt: UniformNode<'float', number>;
}): ConstraintType {
  const { particles, particleOffset, tuples, restAngles, compliance, dt } = args;

  if (!Number.isInteger(particleOffset) || particleOffset < 0) {
    throw new Error(
      `createClothBendingConstraints: particleOffset must be a non-negative integer, got ${particleOffset}`,
    );
  }
  const nConstraints = tuples.length;
  if (nConstraints === 0) throw new Error('createClothBendingConstraints: tuples is empty');
  if (restAngles.length !== nConstraints) {
    throw new Error(
      `createClothBendingConstraints: restAngles.length ${restAngles.length} ≠ tuples.length ${nConstraints}`,
    );
  }
  for (const [a, b, c, d] of tuples) {
    if (
      !Number.isInteger(a) ||
      !Number.isInteger(b) ||
      !Number.isInteger(c) ||
      !Number.isInteger(d) ||
      a < 0 ||
      b < 0 ||
      c < 0 ||
      d < 0
    ) {
      throw new Error(
        `createClothBendingConstraints: tuple (${a},${b},${c},${d}) has a non-integer or negative index`,
      );
    }
    if (a === b || a === c || a === d || b === c || b === d || c === d) {
      throw new Error(
        `createClothBendingConstraints: tuple (${a},${b},${c},${d}) has duplicate indices`,
      );
    }
    const aAbs = particleOffset + a;
    const bAbs = particleOffset + b;
    const cAbs = particleOffset + c;
    const dAbs = particleOffset + d;
    if (
      aAbs >= particles.capacity ||
      bAbs >= particles.capacity ||
      cAbs >= particles.capacity ||
      dAbs >= particles.capacity
    ) {
      throw new Error(
        `createClothBendingConstraints: tuple absolute indices (${aAbs},${bAbs},${cAbs},${dAbs}) exceed capacity ${particles.capacity}`,
      );
    }
  }

  const indices = new Uint32Array(tuples.flat().map((p) => particleOffset + p));
  const particleIndices = instancedArray(indices, 'uint');
  const complianceBuf = instancedArray(nConstraints, 'float');
  const restBuf = instancedArray(Float32Array.from(restAngles), 'float');
  const lambda = instancedArray(nConstraints, 'float');
  (complianceBuf.value.array as Float32Array).fill(compliance);

  const coloring = colorConstraints({ arity: 4, nConstraints, participantsPerConstraint: indices });
  const groups = buildConstraintGroups(
    coloring,
    (c: Any) => {
      const i1: Any = particleIndices.element(c.mul(4)).toVar();
      const i2: Any = particleIndices.element(c.mul(4).add(1)).toVar();
      const i3: Any = particleIndices.element(c.mul(4).add(2)).toVar();
      const i4: Any = particleIndices.element(c.mul(4).add(3)).toVar();

      // Predicted positions (vec3).
      const x1: Any = particles.predictedPositions.element(i1).xyz.toVar();
      const x2: Any = particles.predictedPositions.element(i2).xyz.toVar();
      const x3: Any = particles.predictedPositions.element(i3).xyz.toVar();
      const x4: Any = particles.predictedPositions.element(i4).xyz.toVar();
      const w1: Any = particles.invMass.element(i1).toVar();
      const w2: Any = particles.invMass.element(i2).toVar();
      const w3: Any = particles.invMass.element(i3).toVar();
      const w4: Any = particles.invMass.element(i4).toVar();
      const alpha: Any = complianceBuf.element(c).toVar();
      const phi0: Any = restBuf.element(c).toVar();
      const lamCurrent: Any = lambda.element(c).toVar();

      // E = p2 - p1.
      const E: Any = x2.sub(x1).toVar();
      const eLen2: Any = E.dot(E).toVar();
      // Degenerate edge — skip this thread.
      If(eLen2.lessThan(float(1e-24)), () => {
        Return();
      });
      const eLen: Any = eLen2.sqrt().toVar();

      // N_1 = (p3 - p1) × (p3 - p2)  — Bridson §4 area-weighted normal of triangle 1.
      const a3: Any = x3.sub(x1).toVar();
      const b3: Any = x3.sub(x2).toVar();
      const N1: Any = cross(a3, b3).toVar();
      // N_2 = (p4 - p2) × (p4 - p1)  — Bridson §4 area-weighted normal of triangle 2.
      const a4: Any = x4.sub(x2).toVar();
      const b4: Any = x4.sub(x1).toVar();
      const N2: Any = cross(a4, b4).toVar();
      const n1Sq: Any = N1.dot(N1).toVar();
      const n2Sq: Any = N2.dot(N2).toVar();
      // Degenerate triangle (collapsed) — skip.
      If(n1Sq.lessThan(float(1e-24)), () => {
        Return();
      });
      If(n2Sq.lessThan(float(1e-24)), () => {
        Return();
      });

      // Constraint value C = atan2((N_1×N_2)·E, (N_1·N_2)·|E|) - φ_0.
      // Both atan2 arguments share the factor |N_1|·|N_2|·|E| (positive),
      // so atan2 returns the signed dihedral angle θ ∈ (−π, π] directly
      // without per-pair normalisation. This matches the signed-θ
      // convention Bridson §4 uses for the closed-form gradients below.
      // No clamp needed — atan2 is well-defined for all inputs.
      const cross12: Any = cross(N1, N2);
      const sinTimes: Any = cross12.dot(E);
      const cosTimes: Any = N1.dot(N2).mul(eLen);
      // TSL exports a 2-arg `atan(y, x)` that compiles to WGSL `atan2(y, x)`.
      const theta: Any = atan(sinTimes, cosTimes).toVar();
      const C: Any = theta.sub(phi0).toVar();

      // Bridson §4 closed-form `u_k`, negated to get ∇C = ∇θ (see above).
      //   u_1 = |E| · N_1/|N_1|²
      //   u_2 = |E| · N_2/|N_2|²
      //   u_3 = c31·N_1/|N_1|² + c32·N_2/|N_2|²
      //   u_4 = -(c41·N_1/|N_1|² + c42·N_2/|N_2|²)
      // with cKL = (relevant_diff · E) / |E|.
      const N1Scaled: Any = N1.div(n1Sq).toVar(); // N_1 / |N_1|²
      const N2Scaled: Any = N2.div(n2Sq).toVar(); // N_2 / |N_2|²
      // ∇_{p_3} C = -u_1
      const grad3: Any = N1Scaled.mul(eLen).negate().toVar();
      // ∇_{p_4} C = -u_2
      const grad4: Any = N2Scaled.mul(eLen).negate().toVar();
      // Coefficients for u_3 / u_4 — `(... · E) / |E|`.
      const c31: Any = b3.dot(E).div(eLen).toVar(); // (p3-p2)·E / |E|
      const c32: Any = a4.dot(E).div(eLen).toVar(); // (p4-p2)·E / |E|
      const c41: Any = a3.dot(E).div(eLen).toVar(); // (p3-p1)·E / |E|
      const c42: Any = b4.dot(E).div(eLen).toVar(); // (p4-p1)·E / |E|
      // ∇_{p_1} C = -u_3
      const grad1: Any = N1Scaled.mul(c31).add(N2Scaled.mul(c32)).negate().toVar();
      // ∇_{p_2} C = -u_4 = +(c41·N1Scaled + c42·N2Scaled)  (Bridson's u_4 already carries a leading minus)
      const grad2: Any = N1Scaled.mul(c41).add(N2Scaled.mul(c42)).toVar();

      // Σ w_k |∇_k C|².
      const sumGradSq: Any = grad1
        .dot(grad1)
        .mul(w1)
        .add(grad2.dot(grad2).mul(w2))
        .add(grad3.dot(grad3).mul(w3))
        .add(grad4.dot(grad4).mul(w4))
        .toVar();
      // All four pinned (sum = 0) → no correction.
      If(sumGradSq.lessThanEqual(float(1e-30)), () => {
        Return();
      });

      const dtVal: Any = dt;
      const alphaTilde: Any = alpha.div(dtVal.mul(dtVal));
      const dLambda: Any = xpbdDeltaLambda({
        C,
        sumGradSqInvMass: sumGradSq,
        alphaTilde,
        lambdaCurrent: lamCurrent,
      }).toVar();

      particles.predictedPositions.element(i1).assign(vec4(x1.add(grad1.mul(w1.mul(dLambda))), 0));
      particles.predictedPositions.element(i2).assign(vec4(x2.add(grad2.mul(w2.mul(dLambda))), 0));
      particles.predictedPositions.element(i3).assign(vec4(x3.add(grad3.mul(w3.mul(dLambda))), 0));
      particles.predictedPositions.element(i4).assign(vec4(x4.add(grad4.mul(w4.mul(dLambda))), 0));
      lambda.element(c).assign(lamCurrent.add(dLambda));
    },
    'bending',
  );

  const resetLambdaKernel = Fn(() => {
    lambda.element(instanceIndex).assign(0);
  })()
    .compute(nConstraints)
    .setName('bending.resetLambda');

  return { count: nConstraints, compliance: complianceBuf, lambda, groups, resetLambdaKernel };
}
