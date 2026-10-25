import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import { Fn, If, Return, float, instanceIndex, instancedArray, vec4 } from 'three/tsl';

import {
  buildConstraintGroups,
  colorConstraints,
  xpbdDeltaLambda,
  type ConstraintType,
  type ParticleSystem,
} from '../core/index.js';

import type { TetherConstraint } from './tetherBuild.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Long-range attachment (LRA) tethers (Kim et al. 2012 §3.1): each free
 * particle `x_i` is kept within `r_i` of a pinned anchor particle `a`.
 *
 *   `C(x_i) = |x_i − a| − r_i`,  active only when `C > 0`
 *   `∇C = n = (x_i − a) / |x_i − a|`
 *
 * `a` is read from the anchor's predicted position every iteration, so
 * tethers follow pinned particles that are moved at runtime. The anchor
 * is pinned, so only `x_i` moves.
 *
 * The tether is unilateral: it stops the cloth over-stretching but never
 * pulls inward, so buckling and wrinkles are left to the distance and
 * bending constraints. When inactive, `λ` is reset to zero so a soft
 * tether does not carry a stale multiplier into the next iteration.
 * Compliance `0` (the default) projects straight onto the sphere, Kim's
 * infinitely stiff tether.
 *
 * A particle tethered to several pinned regions gets one tether per
 * region, in different colors, solved one after another rather than
 * averaged as in Kim §3.4. With a single pinned region the two agree.
 * Pinned particles get no tether at all (`buildTethers` skips them).
 */
export function createClothTetherConstraints(args: {
  readonly particles: ParticleSystem;
  /** Absolute slot offset of this cloth's first particle. */
  readonly particleOffset: number;
  /**
   * Cloth-local LRA constraint list — typically the output of
   * {@link "./tetherBuild.js".buildTethers}. Each entry is one
   * (free-particle, anchor-particle, restRadius) tuple.
   */
  readonly tethers: readonly TetherConstraint[];
  /**
   * XPBD compliance `α` (s²/kg). Default `0` matches Kim 2012's
   * implicit infinite-stiffness behaviour; non-zero produces a
   * "soft" tether that allows some over-stretch under load.
   */
  readonly compliance: number;
  readonly dt: UniformNode<'float', number>;
}): ConstraintType {
  const { particles, particleOffset, tethers, compliance, dt } = args;

  if (!Number.isInteger(particleOffset) || particleOffset < 0) {
    throw new Error(
      `createClothTetherConstraints: particleOffset must be a non-negative integer, got ${particleOffset}`,
    );
  }
  if (!Number.isFinite(compliance) || compliance < 0) {
    throw new Error(
      `createClothTetherConstraints: compliance must be a non-negative finite number, got ${compliance}`,
    );
  }

  const nConstraints = tethers.length;
  if (nConstraints === 0) throw new Error('createClothTetherConstraints: tethers is empty');

  for (const t of tethers) {
    const abs = particleOffset + t.particle;
    if (!Number.isInteger(t.particle) || t.particle < 0 || abs >= particles.capacity) {
      throw new Error(
        `createClothTetherConstraints: tether particle ${t.particle} (absolute ${abs}) out of range for capacity ${particles.capacity}`,
      );
    }
    if (!Number.isFinite(t.restRadius) || t.restRadius < 0) {
      throw new Error(
        `createClothTetherConstraints: tether for particle ${t.particle} has invalid restRadius ${t.restRadius}`,
      );
    }
    const anchor = particleOffset + t.anchor;
    if (!Number.isInteger(t.anchor) || t.anchor < 0 || anchor >= particles.capacity) {
      throw new Error(
        `createClothTetherConstraints: tether anchor ${t.anchor} (absolute ${anchor}) out of range for capacity ${particles.capacity}`,
      );
    }
  }

  const indices = Uint32Array.from(tethers, (t) => particleOffset + t.particle);
  const particleIndices = instancedArray(indices, 'uint');
  const anchorIndices = instancedArray(
    Uint32Array.from(tethers, (t) => particleOffset + t.anchor),
    'uint',
  );
  const complianceBuf = instancedArray(nConstraints, 'float');
  const restBuf = instancedArray(
    Float32Array.from(tethers, (t) => t.restRadius),
    'float',
  );
  const lambda = instancedArray(nConstraints, 'float');
  (complianceBuf.value.array as Float32Array).fill(compliance);

  // Two tethers conflict only when they share a free particle, so a particle
  // with K tethers spreads them over K colors.
  const coloring = colorConstraints({ arity: 1, nConstraints, participantsPerConstraint: indices });
  const groups = buildConstraintGroups(
    coloring,
    (c: Any) => {
      const idx: Any = particleIndices.element(c).toVar();
      const x: Any = particles.predictedPositions.element(idx).xyz.toVar();
      const w: Any = particles.invMass.element(idx).toVar();
      If(w.lessThanEqual(float(0.0)), () => {
        Return();
      });

      const a: Any = particles.predictedPositions.element(anchorIndices.element(c)).xyz.toVar();
      const alpha: Any = complianceBuf.element(c).toVar();
      const rest: Any = restBuf.element(c).toVar();
      const lamCurrent: Any = lambda.element(c).toVar();

      const diff: Any = x.sub(a).toVar();
      const len: Any = diff.length().toVar();

      // Inside the sphere the tether is slack.
      If(len.lessThanEqual(rest), () => {
        lambda.element(c).assign(float(0.0));
        Return();
      });
      // Degenerate (`x ≡ a`) — gradient undefined. Leave λ as-is.
      If(len.lessThan(float(1e-12)), () => {
        Return();
      });

      const C: Any = len.sub(rest);
      const n: Any = diff.div(len).toVar();

      const dtVal: Any = dt;
      const alphaTilde: Any = alpha.div(dtVal.mul(dtVal));
      const dLambda: Any = xpbdDeltaLambda({
        C,
        sumGradSqInvMass: w,
        alphaTilde,
        lambdaCurrent: lamCurrent,
      }).toVar();

      // ∇_i C = n  ⇒  Δx_i = w · n · Δλ
      const dx: Any = n.mul(w.mul(dLambda));
      const newX: Any = x.add(dx);
      particles.predictedPositions.element(idx).assign(vec4(newX, float(0.0)));
      lambda.element(c).assign(lamCurrent.add(dLambda));
    },
    'tether',
  );

  const resetLambdaKernel = Fn(() => {
    lambda.element(instanceIndex).assign(0);
  })()
    .compute(nConstraints)
    .setName('tether.resetLambda');

  return { count: nConstraints, compliance: complianceBuf, lambda, groups, resetLambdaKernel };
}
