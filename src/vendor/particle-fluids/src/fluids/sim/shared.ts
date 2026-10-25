import { float, instanceIndex, uint } from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type {
  NeighborList,
  ParticleRange,
  ParticleSystem,
  SphKernelUniforms,
} from '../../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Inputs shared by every fluid kernel. */
export interface FluidKernelContext {
  readonly particles: ParticleSystem;
  /** The fluid's particles. Kernels dispatch one thread per particle in it. */
  readonly range: ParticleRange;
  readonly neighbors: NeighborList;
  readonly sph: SphKernelUniforms;
  readonly restDensity: UniformNode<'float', number>;
  /** Rest volume of one fluid particle (spacing³). */
  readonly particleVolume: UniformNode<'float', number>;
  /** Mass of one fluid particle. */
  readonly mass: UniformNode<'float', number>;
  /** Substep length. */
  readonly dt: UniformNode<'float', number>;
}

/** Emit the global index of the fluid particle this thread handles. */
export function emitFluidIndex(context: FluidKernelContext): Any {
  return instanceIndex.add(uint(context.range.start)).toVar();
}

/** Emit whether particle index `j` lies in any of `ranges`. */
export function emitInRanges(j: Any, ranges: readonly ParticleRange[]): Any {
  return ranges
    .map((range) =>
      j.greaterThanEqual(uint(range.start)).and(j.lessThan(uint(range.start + range.count))),
    )
    .reduce((any: Any, inside: Any) => any.or(inside));
}

/**
 * Emit neighbor `j`'s mass as seen by SPH sums. Boundary particles (non-zero
 * boundary volume) count as `ψ = ρ0 · V` (Akinci et al. 2012, eq. 5), so a
 * sparse layer of solid particles weighs as much as the fluid it displaces.
 * Other particles count with their own mass; pinned ones weigh nothing.
 * That is what keeps two fluids on one particle system apart, and what
 * lets a soft body's interior back up its boundary surface.
 */
export function emitNeighborMass(
  context: FluidKernelContext,
  j: Any,
): { mass: Any; boundaryMass: Any; isBoundary: Any; invMass: Any } {
  const volume: Any = context.particles.boundaryVolume.element(j).toVar();
  const invMass: Any = context.particles.invMass.element(j).toVar();
  const isBoundary: Any = volume.greaterThan(0);
  const boundaryMass: Any = context.restDensity.mul(volume);
  const ownMass: Any = invMass.greaterThan(0).select(float(1).div(invMass), float(0));
  return { mass: isBoundary.select(boundaryMass, ownMass), boundaryMass, isBoundary, invMass };
}

/** Emit neighbor `j`'s volume: its boundary volume, or the fluid particle volume. */
export function emitNeighborVolume(context: FluidKernelContext, j: Any): Any {
  const boundary: Any = context.particles.boundaryVolume.element(j);
  return boundary.greaterThan(0).select(boundary, context.particleVolume);
}
