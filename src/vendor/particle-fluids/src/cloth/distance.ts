import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import {
  createDistanceConstraints,
  type ConstraintType,
  type ParticleSystem,
} from '../core/index.js';

/** Translate cloth-local edges into shared particle slots for the core distance solver. */
export function createClothDistanceConstraints(args: {
  readonly particles: ParticleSystem;
  /**
   * Absolute slot offset of this cloth's first particle inside the shared
   * `ParticleSystem`. The cloth-local edge `(i, j)` lands at
   * `(particleOffset + i, particleOffset + j)` in the global slot space.
   */
  readonly particleOffset: number;
  /** Cloth-local edge pairs `[i, j]` with `0 ≤ i, j < nClothParticles`. */
  readonly edges: readonly (readonly [number, number])[];
  /**
   * Rest length per edge (metres). Required — must have the same length as
   * {@link edges}. Typically the per-edge entry from
   * {@link "./graph.js".ClothGraph.distanceRestLengths}.
   */
  readonly restLengths: readonly number[];
  /**
   * XPBD compliance `α` (s²/kg), the same for every edge.
   */
  readonly compliance: number;
  readonly dt: UniformNode<'float', number>;
}): ConstraintType {
  const { particles, particleOffset, edges, restLengths, compliance, dt } = args;
  if (!Number.isInteger(particleOffset) || particleOffset < 0) {
    throw new Error(
      `createClothDistanceConstraints: particleOffset must be a non-negative integer, got ${particleOffset}`,
    );
  }
  if (restLengths.length !== edges.length) {
    throw new Error(
      `createClothDistanceConstraints: restLengths.length ${restLengths.length} ≠ edges.length ${edges.length}`,
    );
  }
  const absolutePairs: [number, number][] = edges.map(([i, j]) => [
    particleOffset + i,
    particleOffset + j,
  ]);
  return createDistanceConstraints({
    particles,
    pairs: absolutePairs,
    compliance,
    restLength: [...restLengths],
    dt,
  });
}
