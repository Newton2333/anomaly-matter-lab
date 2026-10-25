import { Fn, If, Return, instanceIndex, instancedArray, vec4 } from 'three/tsl';

import type { ParticleSystem } from '../particles.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { xpbdDeltaLambda } from './xpbd.js';
import { buildConstraintGroups, colorConstraints, type ConstraintType } from './types.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * XPBD distance constraints, shared by cloth and direct constraint users.
 *
 * Constraint function (paper Macklin 2016 §6.1 "Spring"):
 *   `C(x_i, x_j) = |x_i − x_j| − L_0`
 *
 * Gradients:
 *   `∇_i C = n`,  `∇_j C = −n`,   where `n = (x_i − x_j) / |x_i − x_j|`.
 * Hence `|∇_i C|² = |∇_j C|² = 1` and
 *   `Σ_k w_k · |∇_k C|² = w_i + w_j`.
 *
 * Constraints are colored so no two in a group share a particle; each
 * group is solved with one thread per constraint, which computes `Δλ`
 * (Macklin et al. 2016, eq. 18) and moves both particles.
 */
export function createDistanceConstraints(args: {
  readonly particles: ParticleSystem;
  readonly pairs: readonly [number, number][];
  /** Compliance α (s²/kg). Scalar broadcasts to every pair; an array or typed array gives one per pair. */
  readonly compliance: number | ArrayLike<number>;
  /** Rest length per pair (metres). Required — must match `pairs.length`. */
  readonly restLength: ArrayLike<number>;
  /** Substep length, usually {@link SolverContext.dt}. */
  readonly dt: UniformNode<'float', number>;
}): ConstraintType {
  const { particles, pairs, compliance, restLength, dt } = args;
  const nConstraints = pairs.length;
  if (nConstraints === 0) throw new Error('createDistanceConstraints: pairs is empty');
  if (restLength.length !== nConstraints) {
    throw new Error(
      `createDistanceConstraints: restLength length ${restLength.length} ≠ pairs.length ${nConstraints}`,
    );
  }
  const complianceArray =
    typeof compliance === 'number'
      ? new Float32Array(nConstraints).fill(compliance)
      : Float32Array.from(compliance);
  if (complianceArray.length !== nConstraints) {
    throw new Error(
      `createDistanceConstraints: compliance length ${complianceArray.length} ≠ pairs.length ${nConstraints}`,
    );
  }
  for (const [i, j] of pairs) {
    if (
      !Number.isInteger(i) ||
      !Number.isInteger(j) ||
      i < 0 ||
      j < 0 ||
      i >= particles.capacity ||
      j >= particles.capacity
    ) {
      throw new Error(
        `createDistanceConstraints: pair (${i}, ${j}) has out-of-range index for capacity ${particles.capacity}`,
      );
    }
    if (i === j) {
      throw new Error(`createDistanceConstraints: pair (${i}, ${j}) references the same particle`);
    }
  }

  const indices = new Uint32Array(pairs.flat());
  const particleIndices = instancedArray(indices, 'uint');
  const complianceBuf = instancedArray(complianceArray, 'float');
  const restBuf = instancedArray(Float32Array.from(restLength), 'float');
  const lambda = instancedArray(nConstraints, 'float');

  const coloring = colorConstraints({ arity: 2, nConstraints, participantsPerConstraint: indices });
  const groups = buildConstraintGroups(
    coloring,
    (c: Any) => {
      const iIdx: Any = particleIndices.element(c.mul(2)).toVar();
      const jIdx: Any = particleIndices.element(c.mul(2).add(1)).toVar();
      const xi: Any = particles.predictedPositions.element(iIdx).xyz.toVar();
      const xj: Any = particles.predictedPositions.element(jIdx).xyz.toVar();
      const wi: Any = particles.invMass.element(iIdx).toVar();
      const wj: Any = particles.invMass.element(jIdx).toVar();
      const wSum: Any = wi.add(wj).toVar();
      const offset: Any = xi.sub(xj).toVar();
      const length: Any = offset.length().toVar();
      If(wSum.lessThanEqual(0).or(length.lessThan(1e-12)), () => {
        Return();
      });
      const n: Any = offset.div(length).toVar();
      const lambdaCurrent: Any = lambda.element(c).toVar();
      const dLambda: Any = xpbdDeltaLambda({
        C: length.sub(restBuf.element(c)),
        sumGradSqInvMass: wSum,
        alphaTilde: complianceBuf.element(c).div(dt.mul(dt)),
        lambdaCurrent,
      }).toVar();
      particles.predictedPositions.element(iIdx).assign(vec4(xi.add(n.mul(wi.mul(dLambda))), 0));
      particles.predictedPositions.element(jIdx).assign(vec4(xj.sub(n.mul(wj.mul(dLambda))), 0));
      lambda.element(c).assign(lambdaCurrent.add(dLambda));
    },
    'distance',
  );

  const resetLambdaKernel = Fn(() => {
    lambda.element(instanceIndex).assign(0);
  })()
    .compute(nConstraints)
    .setName('distance.resetLambda');

  return { count: nConstraints, compliance: complianceBuf, lambda, groups, resetLambdaKernel };
}
