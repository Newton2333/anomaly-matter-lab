import { Fn, If, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import { emitSpikyGrad, type Accumulator, type ParticleRange } from '../../core/index.js';
import {
  emitFluidIndex,
  emitInRanges,
  emitNeighborMass,
  type FluidKernelContext,
} from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Position correction for each fluid particle (Macklin & Müller 2013, eq. 12,
 * mass-weighted): `Δx_i = (w_i / ρ0) Σ_j (λ_i + λ_j) ψ_j ∇W(x_i − x_j)`.
 * Written to `deltaX` so every particle reads the same positions this pass.
 *
 * With `reaction`, the matching push on each dynamic boundary neighbor in
 * `reaction.ranges` is scattered too, `Δx_j = −(λ_i / ρ0) ψ_j w_j ∇W`, so
 * floating bodies feel the fluid's pressure (buoyancy) with momentum
 * conserved pair by pair. Boundary particles of other fluids get no push.
 */
export function buildPositionDeltaKernel(
  context: FluidKernelContext,
  buffers: {
    readonly lambda: StorageBufferNode<'float'>;
    readonly deltaX: StorageBufferNode<'vec4'>;
    readonly reaction?: {
      readonly accumulator: Accumulator;
      /** This fluid's boundary ranges; only they receive the reaction. */
      readonly ranges: readonly ParticleRange[];
    };
  },
): ComputeNode {
  const { particles, neighbors, sph, restDensity } = context;
  const { lambda, deltaX, reaction } = buffers;

  return Fn(() => {
    const { i, row } = neighbors.emitThread();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const lambdaI: Any = lambda.element(i).toVar();
    const sum: Any = vec3(0).toVar();

    neighbors.forEach(row, (j: Any) => {
      const neighbor = emitNeighborMass(context, j);
      const gradient: Any = emitSpikyGrad(
        xi.sub(particles.predictedPositions.element(j).xyz),
        sph,
      ).toVar();
      sum.addAssign(gradient.mul(lambdaI.add(lambda.element(j)).mul(neighbor.mass)));
      if (reaction) {
        If(
          neighbor.isBoundary
            .and(neighbor.invMass.greaterThan(0))
            .and(lambdaI.notEqual(0))
            .and(emitInRanges(j, reaction.ranges)),
          () => {
            reaction.accumulator.add(
              j,
              gradient.mul(
                lambdaI.div(restDensity).negate().mul(neighbor.boundaryMass).mul(neighbor.invMass),
              ),
            );
          },
        );
      }
    });

    deltaX.element(i).assign(vec4(sum.div(restDensity).mul(particles.invMass.element(i)), 0));
  })()
    .compute(neighbors.threadCount)
    .setName('positionDelta.positionDelta');
}

/** Add `deltaX` to the predicted positions. A separate pass, so no thread reads a moved neighbor. */
export function buildApplyDeltaKernel(
  context: FluidKernelContext,
  deltaX: StorageBufferNode<'vec4'>,
): ComputeNode {
  return Fn(() => {
    const i: Any = emitFluidIndex(context);
    const position: Any = context.particles.predictedPositions.element(i);
    position.assign(vec4(position.xyz.add(deltaX.element(i).xyz), position.w));
  })()
    .compute(context.range.count)
    .setName('positionDelta.applyDelta');
}
