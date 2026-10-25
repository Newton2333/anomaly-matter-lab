import { Continue, Fn, If, float } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { Accumulator } from '../../core/index.js';
import { type FluidKernelContext } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Adhesion spline `A(r)` (Akinci et al. 2013, eq. 7), non-zero only on
 * `[h/2, h]`. Closer boundary contact is already handled by the boundary
 * pressure, so the force would double up there.
 */
function emitAdhesionSpline(r: Any, h: number): Any {
  const coefficient = 0.007 / h ** 3.25;
  const root: Any = r
    .mul(r)
    .mul(-4 / h)
    .add(r.mul(6))
    .sub(2 * h)
    .max(0)
    .sqrt()
    .sqrt();
  return r.mul(2).greaterThan(h).and(r.lessThanEqual(h)).select(root.mul(coefficient), float(0));
}

/**
 * Attract fluid particles toward nearby boundary particles:
 * `F = −β m ψ_k A(r) r̂` (Akinci et al. 2013, eq. 6), with `ψ_k = ρ0 V_k`.
 * Applied to the fluid side only, as a velocity change.
 */
export function buildAdhesionKernel(
  context: FluidKernelContext,
  buffers: { readonly beta: UniformNode<'float', number>; readonly accumulator: Accumulator },
): ComputeNode {
  const { particles, neighbors, sph, restDensity, mass, dt } = context;
  const { beta, accumulator } = buffers;
  const h = sph.h.value;

  return Fn(() => {
    const { i, row } = neighbors.emitThread();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    neighbors.forEach(row, (k: Any) => {
      const volume: Any = particles.boundaryVolume.element(k).toVar();
      If(volume.lessThanEqual(0), () => {
        Continue();
      });
      const offset: Any = xi.sub(particles.predictedPositions.element(k).xyz).toVar();
      const r: Any = offset.length().toVar();
      If(r.lessThan(1e-20), () => {
        Continue();
      });
      const force: Any = offset
        .div(r)
        .mul(beta.mul(mass).mul(restDensity.mul(volume)).mul(emitAdhesionSpline(r, h)).negate());
      accumulator.add(i, force.div(mass).mul(dt));
    });
  })()
    .compute(neighbors.threadCount)
    .setName('adhesion.adhesion');
}
