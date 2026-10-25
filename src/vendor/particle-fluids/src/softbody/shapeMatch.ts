import {
  Fn,
  If,
  Loop,
  float,
  instanceIndex,
  localId,
  uint,
  vec3,
  vec4,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { ParticleSystem } from '../core/index.js';

import { emitPolarDecomposition, type Mat3Nodes } from './polarDecomp.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Threads per body in the reductions. Bodies with more particles than this
 * are summed in strides; it must be a power of two for the tree reduction.
 */
const SOFTBODY_WORKGROUP_SIZE = 256;

export interface BuildCenterOfMassKernelArgs {
  readonly particles: ParticleSystem;
  /** Per-body starting index into `particles.predictedPositions`. */
  readonly bodyStart: StorageBufferNode<'uint'>;
  /** Per-body particle count. */
  readonly bodyCount: StorageBufferNode<'uint'>;
  /** Per-particle mass weight `m_i` (any common scale). */
  readonly weights: StorageBufferNode<'float'>;
  /**
   * Per-body output: current center of mass, `xyz` valid, `w` unused.
   * Written once per dispatch (once per substep).
   */
  readonly bodyCenters: StorageBufferNode<'vec4'>;
  /** Number of softbodies — sets the dispatch's workgroup count. */
  readonly numBodies: number;
}

/**
 * Each body's center of mass `c = Σ m_i x*_i / Σ m_i` (Müller et al. 2005,
 * eq. 5): one workgroup per body sums its particles in strides, then a
 * shared-memory tree reduction produces the total.
 */
export function buildCenterOfMassKernel(args: BuildCenterOfMassKernelArgs): ComputeNode {
  const { particles, bodyStart, bodyCount, weights, bodyCenters, numBodies } = args;
  if (!Number.isInteger(numBodies) || numBodies <= 0) {
    throw new Error(
      `buildCenterOfMassKernel: numBodies must be a positive integer, got ${numBodies}`,
    );
  }

  const W = SOFTBODY_WORKGROUP_SIZE;

  return Fn(() => {
    const tid: Any = localId.x;
    const bi: Any = workgroupId.x;
    const start: Any = bodyStart.element(bi).toVar();
    const count: Any = bodyCount.element(bi).toVar();

    // Per-thread partial sums (Σ m x, Σ m) over a stride of the body's particles.
    const localSum: Any = vec4(0.0, 0.0, 0.0, 0.0).toVar();

    // Number of strides = ceil(count / W). Integer div with manual round-up.
    const numStrides: Any = count.add(uint(W - 1)).div(uint(W));
    Loop(
      { start: uint(0), end: numStrides, type: 'uint', condition: '<' },
      ({ i: stride }: { i: Any }) => {
        const localIdx: Any = stride.mul(uint(W)).add(tid).toVar();
        If(localIdx.lessThan(count), () => {
          const globalIdx: Any = start.add(localIdx);
          const xi: Any = particles.predictedPositions.element(globalIdx).xyz;
          const mi: Any = weights.element(globalIdx);
          localSum.addAssign(vec4(xi.mul(mi), mi));
        });
      },
    );

    // Workgroup-shared tree reduction.
    const shared: Any = workgroupArray('vec4', W);
    shared.element(tid).assign(localSum);
    workgroupBarrier();

    // Halve the active range each round. `for` loop runs at shader-build
    // time (TSL unrolls); the `If` inside is the per-thread gate.
    for (let s = W >> 1; s > 0; s >>= 1) {
      const ss = s;
      If(tid.lessThan(uint(ss)), () => {
        const other: Any = shared.element(tid.add(uint(ss)));
        shared.element(tid).assign(shared.element(tid).add(other));
      });
      workgroupBarrier();
    }

    // Thread 0 writes the body centre.
    If(tid.equal(uint(0)), () => {
      const total: Any = shared.element(uint(0));
      bodyCenters.element(bi).assign(vec4(total.xyz.div(total.w), 0));
    });
  })()
    .compute(numBodies * W, [W])
    .setName('shapeMatch.centerOfMass');
}

export interface BuildMomentAndPolarDecompKernelArgs {
  readonly particles: ParticleSystem;
  /** Pre-centered rest offsets `r_i` (per particle, vec3 + pad). */
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly bodyStart: StorageBufferNode<'uint'>;
  readonly bodyCount: StorageBufferNode<'uint'>;
  /** Per-particle mass weight `m_i`, as for {@link buildCenterOfMassKernel}. */
  readonly weights: StorageBufferNode<'float'>;
  /**
   * Per-body output: rotation `R` from the polar decomposition. Stored
   * row-major as three contiguous `vec4` slots per body —
   * `bodyRotations[3·b + row].xyz` holds row `row` of `R_b`, `w` unused.
   * Total length = `3 · numBodies`.
   */
  readonly bodyRotations: StorageBufferNode<'vec4'>;
  readonly numBodies: number;
}

/**
 * Each body's rotation: reduce the moment matrix `A = Σ m_i x*_i r_iᵀ` the
 * same way as the center, then take its polar decomposition (Müller et al.
 * 2005, §3.3). The rest offsets `r_i` are centered on the rest center of
 * mass, so `Σ m_i r_i = 0` and the current center drops out of `A`. Rows of
 * `R` go to `bodyRotations[3b + row]`.
 */
export function buildMomentAndPolarDecompKernel(
  args: BuildMomentAndPolarDecompKernelArgs,
): ComputeNode {
  const { particles, restOffsets, bodyStart, bodyCount, weights, bodyRotations, numBodies } = args;
  if (!Number.isInteger(numBodies) || numBodies <= 0) {
    throw new Error(
      `buildMomentAndPolarDecompKernel: numBodies must be a positive integer, got ${numBodies}`,
    );
  }

  const W = SOFTBODY_WORKGROUP_SIZE;

  return Fn(() => {
    const tid: Any = localId.x;
    const bi: Any = workgroupId.x;
    const start: Any = bodyStart.element(bi).toVar();
    const count: Any = bodyCount.element(bi).toVar();

    // Per-thread partial row sums of A_pq = Σ m_i · x*_i · r_i^T.
    //   row0_i = m_i x*_i.x · r_i,    row1_i = m_i x*_i.y · r_i,    row2_i = m_i x*_i.z · r_i
    const localRow0: Any = vec3(0.0, 0.0, 0.0).toVar();
    const localRow1: Any = vec3(0.0, 0.0, 0.0).toVar();
    const localRow2: Any = vec3(0.0, 0.0, 0.0).toVar();

    const numStrides: Any = count.add(uint(W - 1)).div(uint(W));
    Loop(
      { start: uint(0), end: numStrides, type: 'uint', condition: '<' },
      ({ i: stride }: { i: Any }) => {
        const localIdx: Any = stride.mul(uint(W)).add(tid).toVar();
        If(localIdx.lessThan(count), () => {
          const globalIdx: Any = start.add(localIdx);
          const xi: Any = particles.predictedPositions
            .element(globalIdx)
            .xyz.mul(weights.element(globalIdx));
          const ri: Any = restOffsets.element(globalIdx).xyz;
          localRow0.addAssign(ri.mul(xi.x));
          localRow1.addAssign(ri.mul(xi.y));
          localRow2.addAssign(ri.mul(xi.z));
        });
      },
    );

    // Workgroup-shared accumulators (vec3 has 16-byte stride in WGSL
    // workgroup memory: 3 · 16 · W = 12 KB at W=256, under the 16 KB
    // maxComputeWorkgroupStorageSize WebGPU default).
    const sharedRow0: Any = workgroupArray('vec3', W);
    const sharedRow1: Any = workgroupArray('vec3', W);
    const sharedRow2: Any = workgroupArray('vec3', W);
    sharedRow0.element(tid).assign(localRow0);
    sharedRow1.element(tid).assign(localRow1);
    sharedRow2.element(tid).assign(localRow2);
    workgroupBarrier();

    for (let s = W >> 1; s > 0; s >>= 1) {
      const ss = s;
      If(tid.lessThan(uint(ss)), () => {
        sharedRow0
          .element(tid)
          .assign(sharedRow0.element(tid).add(sharedRow0.element(tid.add(uint(ss)))));
        sharedRow1
          .element(tid)
          .assign(sharedRow1.element(tid).add(sharedRow1.element(tid.add(uint(ss)))));
        sharedRow2
          .element(tid)
          .assign(sharedRow2.element(tid).add(sharedRow2.element(tid.add(uint(ss)))));
      });
      workgroupBarrier();
    }

    // Thread 0 extracts A_pq, runs the polar decomposition, and writes R.
    If(tid.equal(uint(0)), () => {
      const row0: Any = sharedRow0.element(uint(0));
      const row1: Any = sharedRow1.element(uint(0));
      const row2: Any = sharedRow2.element(uint(0));
      const aPq: Mat3Nodes = {
        m00: row0.x,
        m01: row0.y,
        m02: row0.z,
        m10: row1.x,
        m11: row1.y,
        m12: row1.z,
        m20: row2.x,
        m21: row2.y,
        m22: row2.z,
      };
      const R: Mat3Nodes = emitPolarDecomposition(aPq);

      const baseSlot: Any = bi.mul(uint(3));
      bodyRotations.element(baseSlot.add(uint(0))).assign(vec4(R.m00, R.m01, R.m02, float(0.0)));
      bodyRotations.element(baseSlot.add(uint(1))).assign(vec4(R.m10, R.m11, R.m12, float(0.0)));
      bodyRotations.element(baseSlot.add(uint(2))).assign(vec4(R.m20, R.m21, R.m22, float(0.0)));
    });
  })()
    .compute(numBodies * W, [W])
    .setName('shapeMatch.momentAndPolarDecomp');
}

export interface BuildResetLambdaKernelArgs {
  readonly particles: ParticleSystem;
  readonly lambda: StorageBufferNode<'vec4'>;
}

/** Zero the shape-matching multipliers at the start of each substep (Macklin et al. 2016, Algorithm 1). */
export function buildResetLambdaKernel(args: BuildResetLambdaKernelArgs): ComputeNode {
  const { particles, lambda } = args;
  return Fn(() => {
    const i: Any = instanceIndex;
    lambda.element(i).assign(vec4(0.0, 0.0, 0.0, 0.0));
  })()
    .compute(particles.capacity)
    .setName('shapeMatch.resetLambda');
}

export interface BuildShapeMatchDeltaApplyKernelArgs {
  readonly particles: ParticleSystem;
  /** Pre-centered rest offsets `r_i` (per particle, vec3 + pad). */
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly bodyStart: StorageBufferNode<'uint'>;
  readonly bodyCount: StorageBufferNode<'uint'>;
  /** Each body's center, from {@link buildCenterOfMassKernel}. */
  readonly bodyCenters: StorageBufferNode<'vec4'>;
  /** Each body's rotation, from {@link buildMomentAndPolarDecompKernel}. */
  readonly bodyRotations: StorageBufferNode<'vec4'>;
  /**
   * Per-body XPBD compliance `α` in s²/kg. Kernel derives
   * `α̃ = α / dt²` per dispatch.
   */
  readonly bodyCompliance: StorageBufferNode<'float'>;
  /**
   * Per-particle Lagrange multiplier `λ_i` as a 3-vector (xyz used, w
   * padding). Reset to zero at the start of each substep, accumulates
   * Δλ across the iter loop.
   */
  readonly lambda: StorageBufferNode<'vec4'>;
  /** Substep length. */
  readonly dt: UniformNode<'float', number>;
  readonly numBodies: number;
}

/**
 * Pull each particle toward its goal `R r_i + c` as an XPBD constraint with
 * the body's compliance, writing `predictedPositions` in place (each
 * particle's goal depends only on its own body's fit, so there is no race).
 * One workgroup per body, like the reductions.
 */
export function buildShapeMatchDeltaApplyKernel(
  args: BuildShapeMatchDeltaApplyKernelArgs,
): ComputeNode {
  const {
    particles,
    restOffsets,
    bodyStart,
    bodyCount,
    bodyCenters,
    bodyRotations,
    bodyCompliance,
    lambda,
    dt,
    numBodies,
  } = args;
  if (!Number.isInteger(numBodies) || numBodies <= 0) {
    throw new Error(
      `buildShapeMatchDeltaApplyKernel: numBodies must be a positive integer, got ${numBodies}`,
    );
  }

  const W = SOFTBODY_WORKGROUP_SIZE;

  return Fn(() => {
    const tid: Any = localId.x;
    const bi: Any = workgroupId.x;
    const start: Any = bodyStart.element(bi).toVar();
    const count: Any = bodyCount.element(bi).toVar();

    // Read (R, c, α) for this body. Uniform across the workgroup.
    const baseRotSlot: Any = bi.mul(uint(3));
    const R0: Any = bodyRotations.element(baseRotSlot).xyz.toVar();
    const R1: Any = bodyRotations.element(baseRotSlot.add(uint(1))).xyz.toVar();
    const R2: Any = bodyRotations.element(baseRotSlot.add(uint(2))).xyz.toVar();
    const c: Any = bodyCenters.element(bi).xyz.toVar();
    const alpha: Any = bodyCompliance.element(bi).toVar();
    // α̃ = α / dt² (Macklin et al. 2016).
    const alphaTilde: Any = alpha.div((dt as Any).mul(dt as Any));

    const numStrides: Any = count.add(uint(W - 1)).div(uint(W));
    Loop(
      { start: uint(0), end: numStrides, type: 'uint', condition: '<' },
      ({ i: stride }: { i: Any }) => {
        const localIdx: Any = stride.mul(uint(W)).add(tid).toVar();
        If(localIdx.lessThan(count), () => {
          const globalIdx: Any = start.add(localIdx);
          const ri: Any = restOffsets.element(globalIdx).xyz.toVar();

          // goal = R · r + c  (row-major R → three dots against r)
          const goal: Any = vec3(R0.dot(ri), R1.dot(ri), R2.dot(ri)).add(c).toVar();

          const xStar: Any = particles.predictedPositions.element(globalIdx).xyz.toVar();

          // Vector constraint with identity Jacobian:
          //   C_ij = x*_ij − goal_ij
          //   Δλ_ij = (−C_ij − α̃ · λ_ij) / (w_i + α̃)
          //   Δx_ij = w_i · Δλ_ij
          const C: Any = xStar.sub(goal);
          const lambdaOld: Any = lambda.element(globalIdx).xyz.toVar();
          const w: Any = particles.invMass.element(globalIdx).toVar();
          const denom: Any = w.add(alphaTilde);

          const deltaLambda: Any = C.negate().sub(lambdaOld.mul(alphaTilde)).div(denom);
          const newLambda: Any = lambdaOld.add(deltaLambda);
          const deltaX: Any = deltaLambda.mul(w);

          lambda.element(globalIdx).assign(vec4(newLambda.x, newLambda.y, newLambda.z, float(0.0)));
          particles.predictedPositions
            .element(globalIdx)
            .assign(vec4(xStar.add(deltaX), float(0.0)));
        });
      },
    );
  })()
    .compute(numBodies * W, [W])
    .setName('shapeMatch.shapeMatchDeltaApply');
}
