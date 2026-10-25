import { Fn, If, Return, float, vec3 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { emitPoly6FromRSq, emitSpikyGrad } from '../../core/index.js';
import { emitNeighborMass, type FluidKernelContext } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Density and the density-constraint multiplier λ in one neighbor walk
 * (Macklin & Müller 2013, eqs. 2 and 9–11, in XPBD form):
 *
 * - `ρ_i = Σ_j ψ_j W(x_i − x_j)`, where ψ_j is the neighbor's mass or, for
 *   boundary particles, `ρ0 · V_j` (Akinci et al. 2012).
 * - `C_i = ρ_i / ρ0 − 1`. Only compression is corrected, so `λ_i = 0` when
 *   `C_i ≤ 0` (Macklin et al. 2014, eq. 26); this keeps free surfaces from
 *   clumping.
 * - `λ_i = −C_i / (Σ_k w_k |∇_k C_i|² + α / dt²)`, with the inverse masses
 *   `w_k` kept so fluids couple correctly with heavier or lighter solids.
 *
 * ψ_j stays inside the gradient terms as well as the density sum. Dropping it
 * (as the paper can, with unit masses) under-scales the correction by `1/m²`
 * and collapses real-world-scale fluids.
 *
 * Density is written even when λ is zero, because surface tension reads it.
 */
export function buildLambdaKernel(
  context: FluidKernelContext,
  buffers: {
    readonly compliance: UniformNode<'float', number>;
    readonly density: StorageBufferNode<'float'>;
    readonly lambda: StorageBufferNode<'float'>;
  },
): ComputeNode {
  const { particles, neighbors, sph, restDensity, dt } = context;
  const { compliance, density, lambda } = buffers;

  return Fn(() => {
    const { i, row } = neighbors.emitThread();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const rho: Any = float(0).toVar();
    const gradSum: Any = vec3(0).toVar();
    const gradSqSum: Any = float(0).toVar();

    neighbors.forEach(row, (j: Any) => {
      const offset: Any = xi.sub(particles.predictedPositions.element(j).xyz).toVar();
      const neighbor = emitNeighborMass(context, j);
      rho.addAssign(neighbor.mass.mul(emitPoly6FromRSq(offset.dot(offset), sph)));
      const gradient: Any = emitSpikyGrad(offset, sph).mul(neighbor.mass).toVar();
      gradSum.addAssign(gradient);
      gradSqSum.addAssign(gradient.dot(gradient).mul(neighbor.invMass));
    });
    density.element(i).assign(rho);

    const invRho0: Any = float(1).div(restDensity).toVar();
    const constraint: Any = rho.mul(invRho0).sub(1).toVar();
    If(constraint.lessThanEqual(0), () => {
      lambda.element(i).assign(0);
      Return();
    });
    const gradientNorm: Any = gradSum
      .dot(gradSum)
      .mul(particles.invMass.element(i))
      .add(gradSqSum)
      .mul(invRho0.mul(invRho0));
    const alphaTilde: Any = compliance.div(dt.mul(dt));
    lambda.element(i).assign(constraint.negate().div(gradientNorm.add(alphaTilde).max(1e-20)));
  })()
    .compute(neighbors.threadCount)
    .setName('lambda.lambda');
}
