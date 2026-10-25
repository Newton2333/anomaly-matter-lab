import { Continue, Fn, If, float, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { emitSpikyGrad, type Accumulator } from '../../core/index.js';
import { emitNeighborVolume, type FluidKernelContext } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Surface tension from Akinci et al. 2013, "Versatile Surface Tension and
 * Adhesion for SPH Fluids", §2: a cohesion force that attracts particles
 * at mid range and a curvature force that minimizes surface area.
 *
 * Both are applied pairwise: each pair's force is added to one particle
 * and subtracted from the other in the same step, as the paper requires,
 * so momentum is conserved exactly. The resulting velocity changes are
 * collected in an accumulator and applied once per substep, before the
 * pressure solve (see {@link FluidSystem}).
 */

/** Scaled color-field normal `n_i = h Σ_j V_j ∇W(x_i − x_j)` (Akinci et al. 2013, eq. 3). */
export function buildColorFieldNormalKernel(
  context: FluidKernelContext,
  normal: StorageBufferNode<'vec4'>,
): ComputeNode {
  const { particles, neighbors, sph } = context;
  return Fn(() => {
    const { i, row } = neighbors.emitThread();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const sum: Any = vec3(0).toVar();
    neighbors.forEach(row, (j: Any) => {
      const gradient: Any = emitSpikyGrad(xi.sub(particles.predictedPositions.element(j).xyz), sph);
      sum.addAssign(gradient.mul(emitNeighborVolume(context, j)));
    });
    normal.element(i).assign(vec4(sum.mul(sph.h), 0));
  })()
    .compute(neighbors.threadCount)
    .setName('cohesion.colorFieldNormal');
}

/**
 * Cohesion spline `C(r)` (Akinci et al. 2013, eq. 2): attractive with a peak at
 * `h/2`, zero at `h`, and mildly repulsive at very short range.
 */
function emitCohesionSpline(r: Any, h: number): Any {
  const coefficient = 32 / (Math.PI * h ** 9);
  const gap: Any = float(h).sub(r).max(0);
  const shared: Any = gap.mul(gap).mul(gap).mul(r.mul(r).mul(r));
  const value: Any = r
    .mul(2)
    .greaterThan(h)
    .select(shared, shared.mul(2).sub(h ** 6 / 64));
  return r.greaterThan(0).and(r.lessThanEqual(h)).select(value.mul(coefficient), float(0));
}

/**
 * Scatter each fluid pair's surface tension force as velocity changes:
 * `F = K_ij (−γ m² C(r) r̂ − γ m (n_i − n_j))` with `K_ij = 2ρ0 / (ρ_i + ρ_j)`
 * (Akinci et al. 2013, eqs. 1–5). Only neighbors in the fluid's own range
 * pair up: boundaries and other materials' particles are skipped.
 */
export function buildSurfaceTensionKernel(
  context: FluidKernelContext,
  buffers: {
    readonly gamma: UniformNode<'float', number>;
    readonly normal: StorageBufferNode<'vec4'>;
    readonly density: StorageBufferNode<'float'>;
    readonly accumulator: Accumulator;
  },
): ComputeNode {
  const { particles, neighbors, sph, restDensity, mass, dt } = context;
  const { gamma, normal, density, accumulator } = buffers;
  const h = sph.h.value;
  const end = context.range.start + context.range.count;

  return Fn(() => {
    const { i, row } = neighbors.emitThread();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const ni: Any = normal.element(i).xyz.toVar();
    const rhoI: Any = density.element(i).toVar();

    neighbors.forEach(row, (j: Any) => {
      // Visit each pair once, and only fluid–fluid pairs. `j > i` already
      // puts j past the range's start.
      If(j.lessThanEqual(i).or(j.greaterThanEqual(uint(end))), () => {
        Continue();
      });
      const offset: Any = xi.sub(particles.predictedPositions.element(j).xyz).toVar();
      const r: Any = offset.length().toVar();
      If(r.lessThan(1e-20), () => {
        Continue();
      });
      const correction: Any = restDensity.mul(2).div(rhoI.add(density.element(j)).max(1));
      const cohesion: Any = offset
        .div(r)
        .mul(gamma.mul(mass).mul(mass).mul(emitCohesionSpline(r, h)).negate());
      const curvature: Any = ni.sub(normal.element(j).xyz).mul(gamma.mul(mass)).negate();
      const dv: Any = cohesion.add(curvature).mul(correction).div(mass).mul(dt).toVar();
      accumulator.add(i, dv);
      accumulator.add(j, dv.negate());
    });
  })()
    .compute(neighbors.threadCount)
    .setName('cohesion.surfaceTension');
}
