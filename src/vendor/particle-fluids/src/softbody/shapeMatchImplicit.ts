import { Fn, If, Loop, float, instanceIndex, int, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { Accumulator, ParticleRange, ParticleSystem } from '../core/index.js';

import { emitPolarDecomposition, type Mat3Nodes } from './polarDecomp.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Local shape matching with oriented particles (Müller & Chentanez 2011,
 * §5.1), used by `SoftbodySystem` with `shapeMatching: 'local'`.
 *
 * Every particle `i` defines a group: itself plus its edge neighbors `N(i)`,
 * stored as compressed rows (`neighborOffsets`, `neighborIndices`). Each
 * solver iteration:
 *
 * 1. finds each group's center `c_i`;
 * 2. fits each group's rotation `R_i` by polar decomposition, including the
 *    particles' own orientations (eq. 7), which keeps chains and single
 *    particles well defined;
 * 3. pulls every group member toward `R_i (x̃_j − c̄_i) + c_i`, scattering
 *    the corrections into an accumulator;
 * 4. stores `R_i` as particle `i`'s predicted orientation (only the group's
 *    center rotates; its members only move).
 *
 * Masses are assumed uniform within a body, so they cancel out of the fit.
 * Each (group, member) pair carries its own Lagrange multiplier, because a
 * particle belongs to several groups at once.
 */
export interface BuildImplicitNeighborhoodCenterKernelArgs {
  readonly particles: ParticleSystem;
  /** Particles the kernel runs over: every body in the system. */
  readonly range: ParticleRange;
  /** Row offsets into `neighborIndices`, length `capacity + 1`. */
  readonly neighborOffsets: StorageBufferNode<'uint'>;
  /** Members of every group (global particle indices), each row starting with the particle itself. */
  readonly neighborIndices: StorageBufferNode<'uint'>;
  /** Each group's current center. */
  readonly particleCenters: StorageBufferNode<'vec4'>;
}

/** Step 1: each group's center `c_i`, the mean of its members' predicted positions. */
export function buildImplicitNeighborhoodCenterKernel(
  args: BuildImplicitNeighborhoodCenterKernelArgs,
): ComputeNode {
  const { particles, range, neighborOffsets, neighborIndices, particleCenters } = args;

  return Fn(() => {
    const i: Any = instanceIndex.add(uint(range.start));
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();
    const count: Any = end.sub(start).toVar();

    const sum: Any = vec3(0.0, 0.0, 0.0).toVar();
    Loop({ start: start, end: end, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const j: Any = neighborIndices.element(k);
      sum.addAssign(particles.predictedPositions.element(j).xyz);
    });
    const safeCount: Any = count.max(uint(1));
    const inv: Any = float(1.0).div(safeCount.toFloat());
    const c: Any = sum.mul(inv);
    const isEmpty: Any = count.equal(uint(0));
    const cx: Any = isEmpty.select(float(0.0), c.x);
    const cy: Any = isEmpty.select(float(0.0), c.y);
    const cz: Any = isEmpty.select(float(0.0), c.z);
    particleCenters.element(i).assign(vec4(cx, cy, cz, float(0.0)));
  })()
    .compute(range.count)
    .setName('shapeMatchImplicit.implicitNeighborhoodCenter');
}

export interface BuildImplicitMomentPolarKernelArgs {
  readonly particles: ParticleSystem;
  /** Particles the kernel runs over: every body in the system. */
  readonly range: ParticleRange;
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly neighborOffsets: StorageBufferNode<'uint'>;
  readonly neighborIndices: StorageBufferNode<'uint'>;
  readonly particleCenters: StorageBufferNode<'vec4'>;
  /** Each group's rest center `c̄_i`, fixed at construction. */
  readonly restNeighborhoodCenters: StorageBufferNode<'vec4'>;
  /** Each group's rotation `R_i`, three row vectors per particle. */
  readonly particleRotations: StorageBufferNode<'vec4'>;
  /** `r²/5`: the orientation term's weight, from a solid sphere's inertia with mass cancelled (eq. 8). */
  readonly aiScalar: UniformNode<'float', number>;
}

/**
 * Step 2: each group's rotation from its moment matrix (eqs. 7–8):
 *
 *   A_i = (r²/5) Σ_j R_j + Σ_j (x*_j − c_i)(x̃_j − c̄_i)ᵀ,   R_i = polar(A_i)
 *
 * The orientation term sums every member's own rotation `R_j`, read from
 * its predicted orientation; including only the center's under-regularizes
 * the fit and lets noise build up. Reading the orientation from the start of
 * the substep instead would pull every fit back toward it and damp spin.
 * Groups outside any body get the identity.
 */
export function buildImplicitMomentPolarKernel(
  args: BuildImplicitMomentPolarKernelArgs,
): ComputeNode {
  const {
    particles,
    range,
    restOffsets,
    neighborOffsets,
    neighborIndices,
    particleCenters,
    restNeighborhoodCenters,
    particleRotations,
    aiScalar,
  } = args;

  return Fn(() => {
    const i: Any = instanceIndex.add(uint(range.start));
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();
    const count: Any = end.sub(start).toVar();

    // Both terms of eq. 7 sum over every member of the group.
    const a00: Any = float(0.0).toVar();
    const a01: Any = float(0.0).toVar();
    const a02: Any = float(0.0).toVar();
    const a10: Any = float(0.0).toVar();
    const a11: Any = float(0.0).toVar();
    const a12: Any = float(0.0).toVar();
    const a20: Any = float(0.0).toVar();
    const a21: Any = float(0.0).toVar();
    const a22: Any = float(0.0).toVar();

    const c: Any = particleCenters.element(i).xyz.toVar();
    const cBar: Any = restNeighborhoodCenters.element(i).xyz.toVar();

    Loop({ start: start, end: end, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const j: Any = neighborIndices.element(k);

      // (r²/5) · R_j — the A_j term from this neighbor (Eq. 8).
      // R_j = mat3FromQuat(particles.predictedRotation[j]). Standard formula
      // for unit quaternion q = (x, y, z, w):
      //   m00 = 1 − 2(y² + z²)     m01 = 2(xy − wz)     m02 = 2(xz + wy)
      //   m10 = 2(xy + wz)         m11 = 1 − 2(x² + z²) m12 = 2(yz − wx)
      //   m20 = 2(xz − wy)         m21 = 2(yz + wx)     m22 = 1 − 2(x² + y²)
      const qj: Any = particles.predictedRotation.element(j).toVar();
      const xx: Any = qj.x.mul(qj.x);
      const yy: Any = qj.y.mul(qj.y);
      const zz: Any = qj.z.mul(qj.z);
      const xy: Any = qj.x.mul(qj.y);
      const xz: Any = qj.x.mul(qj.z);
      const yz: Any = qj.y.mul(qj.z);
      const wx: Any = qj.w.mul(qj.x);
      const wy: Any = qj.w.mul(qj.y);
      const wz: Any = qj.w.mul(qj.z);
      const rj00: Any = float(1.0).sub(yy.add(zz).mul(float(2.0)));
      const rj01: Any = xy.sub(wz).mul(float(2.0));
      const rj02: Any = xz.add(wy).mul(float(2.0));
      const rj10: Any = xy.add(wz).mul(float(2.0));
      const rj11: Any = float(1.0).sub(xx.add(zz).mul(float(2.0)));
      const rj12: Any = yz.sub(wx).mul(float(2.0));
      const rj20: Any = xz.sub(wy).mul(float(2.0));
      const rj21: Any = yz.add(wx).mul(float(2.0));
      const rj22: Any = float(1.0).sub(xx.add(yy).mul(float(2.0)));
      a00.addAssign(aiScalar.mul(rj00));
      a01.addAssign(aiScalar.mul(rj01));
      a02.addAssign(aiScalar.mul(rj02));
      a10.addAssign(aiScalar.mul(rj10));
      a11.addAssign(aiScalar.mul(rj11));
      a12.addAssign(aiScalar.mul(rj12));
      a20.addAssign(aiScalar.mul(rj20));
      a21.addAssign(aiScalar.mul(rj21));
      a22.addAssign(aiScalar.mul(rj22));

      // Outer product accumulate: A += (x*_j - c) (x̃_j - c̄)^T.
      const xj: Any = particles.predictedPositions.element(j).xyz.sub(c);
      const xtj: Any = restOffsets.element(j).xyz.sub(cBar);
      a00.addAssign(xj.x.mul(xtj.x));
      a01.addAssign(xj.x.mul(xtj.y));
      a02.addAssign(xj.x.mul(xtj.z));
      a10.addAssign(xj.y.mul(xtj.x));
      a11.addAssign(xj.y.mul(xtj.y));
      a12.addAssign(xj.y.mul(xtj.z));
      a20.addAssign(xj.z.mul(xtj.x));
      a21.addAssign(xj.z.mul(xtj.y));
      a22.addAssign(xj.z.mul(xtj.z));
    });

    const aPq: Mat3Nodes = {
      m00: a00,
      m01: a01,
      m02: a02,
      m10: a10,
      m11: a11,
      m12: a12,
      m20: a20,
      m21: a21,
      m22: a22,
    };
    const R: Mat3Nodes = emitPolarDecomposition(aPq);

    // Particles outside every body get the identity.
    const isEmpty: Any = count.equal(uint(0));
    const r00: Any = isEmpty.select(float(1.0), R.m00);
    const r01: Any = isEmpty.select(float(0.0), R.m01);
    const r02: Any = isEmpty.select(float(0.0), R.m02);
    const r10: Any = isEmpty.select(float(0.0), R.m10);
    const r11: Any = isEmpty.select(float(1.0), R.m11);
    const r12: Any = isEmpty.select(float(0.0), R.m12);
    const r20: Any = isEmpty.select(float(0.0), R.m20);
    const r21: Any = isEmpty.select(float(0.0), R.m21);
    const r22: Any = isEmpty.select(float(1.0), R.m22);

    const baseSlot: Any = i.mul(uint(3));
    particleRotations.element(baseSlot).assign(vec4(r00, r01, r02, float(0.0)));
    particleRotations.element(baseSlot.add(uint(1))).assign(vec4(r10, r11, r12, float(0.0)));
    particleRotations.element(baseSlot.add(uint(2))).assign(vec4(r20, r21, r22, float(0.0)));
  })()
    .compute(range.count)
    .setName('shapeMatchImplicit.implicitMomentPolar');
}

export interface BuildImplicitShapeMatchScatterKernelArgs {
  readonly particles: ParticleSystem;
  /** Particles the kernel runs over: every body in the system. */
  readonly range: ParticleRange;
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly neighborOffsets: StorageBufferNode<'uint'>;
  readonly neighborIndices: StorageBufferNode<'uint'>;
  readonly particleCenters: StorageBufferNode<'vec4'>;
  readonly restNeighborhoodCenters: StorageBufferNode<'vec4'>;
  readonly particleRotations: StorageBufferNode<'vec4'>;

  /** One multiplier per (group, member) pair, reset each substep. */
  readonly pairLambda: StorageBufferNode<'vec4'>;
  readonly accumulator: Accumulator;
  readonly dt: UniformNode<'float', number>;
}

/**
 * Step 3: pull every member toward its goal in each group it belongs to:
 *
 *   goal = R_i (x̃_j − c̄_i) + c_i,   C = x*_j − goal
 *   Δλ = (−C − α̃ λ) / (w_j + α̃),   α̃ = compliance / dt²
 *
 * A particle belongs to `|N(j)|` groups, so its correction is divided by
 * that count; otherwise the sum overshoots and the body explodes. Each
 * correction is also clamped to a few particle radii, so one bad rotation
 * can't fling a particle. Corrections are scattered into `accumulator`,
 * which the caller applies after the pass.
 */
export function buildImplicitShapeMatchScatterKernel(
  args: BuildImplicitShapeMatchScatterKernelArgs,
): ComputeNode {
  const {
    particles,
    range,
    restOffsets,
    neighborOffsets,
    neighborIndices,
    particleCenters,
    restNeighborhoodCenters,
    particleRotations,
    pairLambda,
    accumulator,
    dt,
  } = args;
  // Clamp on each pair's correction, so one bad rotation can't fling a particle.
  const maxCorrection = 5 * particles.particleRadius;

  return Fn(() => {
    const i: Any = instanceIndex.add(uint(range.start));
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();

    // Particles outside every body have empty rows, so the loop does nothing for them.
    const baseSlot: Any = i.mul(uint(3));
    const R0: Any = particleRotations.element(baseSlot).xyz.toVar();
    const R1: Any = particleRotations.element(baseSlot.add(uint(1))).xyz.toVar();
    const R2: Any = particleRotations.element(baseSlot.add(uint(2))).xyz.toVar();
    const c: Any = particleCenters.element(i).xyz.toVar();
    const cBar: Any = restNeighborhoodCenters.element(i).xyz.toVar();
    // Each body's compliance is stored in the w component of its rest offsets.
    const alphaTilde: Any = restOffsets.element(i).w.div((dt as Any).mul(dt as Any));

    Loop({ start: start, end: end, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const j: Any = neighborIndices.element(k);
      const restRel: Any = restOffsets.element(j).xyz.sub(cBar).toVar();
      // goal = R · restRel + c.
      const goal: Any = vec3(R0.dot(restRel), R1.dot(restRel), R2.dot(restRel)).add(c).toVar();
      const xStar: Any = particles.predictedPositions.element(j).xyz;
      const C: Any = xStar.sub(goal).toVar();
      const wj: Any = particles.invMass.element(j).toVar();
      const denom: Any = wj.add(alphaTilde);

      const lambdaOld: Any = pairLambda.element(k).xyz.toVar();
      const deltaLambda: Any = C.negate().sub(lambdaOld.mul(alphaTilde)).div(denom);
      const newLambda: Any = lambdaOld.add(deltaLambda);
      pairLambda.element(k).assign(vec4(newLambda.x, newLambda.y, newLambda.z, float(0.0)));
      // Average over the groups j belongs to (see above).
      const offJlo: Any = neighborOffsets.element(j).toVar();
      const offJhi: Any = neighborOffsets.element(j.add(uint(1))).toVar();
      const mj: Any = offJhi.sub(offJlo).toFloat().max(float(1.0));
      const deltaXraw: Any = deltaLambda.mul(wj).div(mj).toVar();

      const dxScale: Any = float(maxCorrection * maxCorrection)
        .div(deltaXraw.dot(deltaXraw).max(maxCorrection * maxCorrection))
        .sqrt();
      accumulator.add(j, deltaXraw.mul(dxScale));
    });
  })()
    .compute(range.count)
    .setName('shapeMatchImplicit.implicitShapeMatchScatter');
}

export interface BuildImplicitQpWriteKernelArgs {
  readonly particles: ParticleSystem;
  /** Particles the kernel runs over: every body in the system. */
  readonly range: ParticleRange;
  readonly particleRotations: StorageBufferNode<'vec4'>;
  readonly neighborOffsets: StorageBufferNode<'uint'>;
}

/**
 * Step 4: store each group's rotation `R_i` as its center particle's
 * predicted orientation (as a quaternion, on the shorter arc from the
 * current one). Only the center rotates (Müller & Chentanez 2011, §5.1);
 * the orientation integrator then derives the angular velocity from it.
 * Particles outside every body keep their predicted orientation.
 */
export function buildImplicitQpWriteKernel(args: BuildImplicitQpWriteKernelArgs): ComputeNode {
  const { particles, range, particleRotations, neighborOffsets } = args;

  return Fn(() => {
    const i: Any = instanceIndex.add(uint(range.start));
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();

    const baseSlot: Any = i.mul(uint(3));
    const r0: Any = particleRotations.element(baseSlot).xyz.toVar();
    const r1: Any = particleRotations.element(baseSlot.add(uint(1))).xyz.toVar();
    const r2: Any = particleRotations.element(baseSlot.add(uint(2))).xyz.toVar();

    // quatFromMat3 — branched form to avoid division by small numbers.
    // trace = m00 + m11 + m22; if trace > 0 use the trace-based form;
    // otherwise use the largest-diagonal-element form. See Shoemake 1985.
    const m00: Any = r0.x;
    const m01: Any = r0.y;
    const m02: Any = r0.z;
    const m10: Any = r1.x;
    const m11: Any = r1.y;
    const m12: Any = r1.z;
    const m20: Any = r2.x;
    const m21: Any = r2.y;
    const m22: Any = r2.z;
    const trace: Any = m00.add(m11).add(m22);

    const qx: Any = float(0.0).toVar();
    const qy: Any = float(0.0).toVar();
    const qz: Any = float(0.0).toVar();
    const qw: Any = float(1.0).toVar();

    If(trace.greaterThan(float(0.0)), () => {
      const s: Any = float(0.5).div(trace.add(float(1.0)).sqrt());
      qw.assign(float(0.25).div(s));
      qx.assign(m21.sub(m12).mul(s));
      qy.assign(m02.sub(m20).mul(s));
      qz.assign(m10.sub(m01).mul(s));
    })
      .ElseIf(m00.greaterThan(m11).and(m00.greaterThan(m22)), () => {
        const s: Any = float(2.0).mul(float(1.0).add(m00).sub(m11).sub(m22).sqrt());
        qw.assign(m21.sub(m12).div(s));
        qx.assign(float(0.25).mul(s));
        qy.assign(m01.add(m10).div(s));
        qz.assign(m02.add(m20).div(s));
      })
      .ElseIf(m11.greaterThan(m22), () => {
        const s: Any = float(2.0).mul(float(1.0).add(m11).sub(m00).sub(m22).sqrt());
        qw.assign(m02.sub(m20).div(s));
        qx.assign(m01.add(m10).div(s));
        qy.assign(float(0.25).mul(s));
        qz.assign(m12.add(m21).div(s));
      })
      .Else(() => {
        const s: Any = float(2.0).mul(float(1.0).add(m22).sub(m00).sub(m11).sqrt());
        qw.assign(m10.sub(m01).div(s));
        qx.assign(m02.add(m20).div(s));
        qy.assign(m12.add(m21).div(s));
        qz.assign(float(0.25).mul(s));
      });

    // Shorter-rotation rule against the substep-start `q_i`.
    const qStart: Any = particles.rotation.element(i).toVar();
    const dot: Any = qStart.x
      .mul(qx)
      .add(qStart.y.mul(qy))
      .add(qStart.z.mul(qz))
      .add(qStart.w.mul(qw));
    const flip: Any = dot.lessThan(float(0.0)).select(float(-1.0), float(1.0));

    // Particles outside every body keep their predicted orientation.
    If(end.greaterThan(start), () => {
      particles.predictedRotation
        .element(i)
        .assign(vec4(qx.mul(flip), qy.mul(flip), qz.mul(flip), qw.mul(flip)));
    });
  })()
    .compute(range.count)
    .setName('shapeMatchImplicit.implicitQpWrite');
}

export interface BuildImplicitResetPairLambdaKernelArgs {
  readonly pairLambda: StorageBufferNode<'vec4'>;
  /** Total CSR entries (sum of per-particle degree, including self). */
  readonly totalDegree: number;
}

/** Zero the per-pair multipliers at the start of each substep. */
export function buildImplicitResetPairLambdaKernel(
  args: BuildImplicitResetPairLambdaKernelArgs,
): ComputeNode {
  const { pairLambda, totalDegree } = args;
  if (!Number.isInteger(totalDegree) || totalDegree < 0) {
    throw new Error(
      `buildImplicitResetPairLambdaKernel: totalDegree must be a non-negative integer, got ${totalDegree}`,
    );
  }
  if (totalDegree === 0) {
    // Empty body or no-edge config — emit a one-thread no-op kernel so the
    // caller's pipeline shape is uniform (avoids a conditional dispatch
    // append at the SoftbodySystem call site).
    return Fn(() => {
      // Intentional no-op; one-thread dispatch is cheap and keeps the
      // pipeline shape regular.
      void int(0);
    })()
      .compute(1)
      .setName('shapeMatchImplicit.implicitResetPairLambda');
  }
  return Fn(() => {
    const k: Any = instanceIndex;
    pairLambda.element(k).assign(vec4(0.0, 0.0, 0.0, 0.0));
  })()
    .compute(totalDegree)
    .setName('shapeMatchImplicit.implicitResetPairLambda');
}
