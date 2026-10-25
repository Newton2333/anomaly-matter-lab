import type { Box3 } from 'three';
import { Fn, If, float, instanceIndex, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import {
  emitForEachNeighbor,
  emitPoly6FromRSq,
  type HashGrid,
  type SphKernelUniforms,
} from '../core/index.js';
import type { FluidSystem } from '../fluids/index.js';
import type { GasSystem } from './GasSystem.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Move each live tracer with the fluid's velocity at its position,
 * `v(x) = Σ_j v_j W(x − x_j) / Σ_j W(x − x_j)` (Macklin et al. 2014, eq. 28),
 * and retire tracers older than `lifetime` or outside `bounds`. Tracers aren't particles in the
 * grid; they only query it.
 */
export function buildSmokeAdvectKernel(args: {
  readonly tracers: GasSystem;
  readonly fluid: FluidSystem;
  readonly hashGrid: HashGrid;
  readonly sph: SphKernelUniforms;
  readonly dt: UniformNode<'float', number>;
  readonly lifetime: number;
  readonly bounds?: Box3 | undefined;
}): ComputeNode {
  const { tracers, fluid, hashGrid, sph, dt, lifetime, bounds } = args;
  const { particles, range } = fluid;
  const end = range.start + range.count;

  return Fn(() => {
    const s: Any = instanceIndex;
    If(tracers.smokeAlive.element(s).greaterThan(uint(0)), () => {
      const position: Any = tracers.smokePositions.element(s).xyz.toVar();
      const age: Any = tracers.smokeAge.element(s).add(dt).toVar();
      let expired: Any = age.greaterThanEqual(lifetime);
      if (bounds) {
        const { min, max } = bounds;
        expired = expired
          .or(position.lessThan(vec3(min.x, min.y, min.z)).any())
          .or(position.greaterThan(vec3(max.x, max.y, max.z)).any());
      }
      If(expired, () => {
        tracers.smokeAlive.element(s).assign(uint(0));
      }).Else(() => {
        const velocitySum: Any = vec3(0).toVar();
        const weightSum: Any = float(0).toVar();
        emitForEachNeighbor(hashGrid, position, (j: Any) => {
          If(j.greaterThanEqual(uint(range.start)).and(j.lessThan(uint(end))), () => {
            const offset: Any = position.sub(particles.positions.element(j).xyz);
            const w: Any = emitPoly6FromRSq(offset.dot(offset), sph);
            velocitySum.addAssign(particles.velocities.element(j).xyz.mul(w));
            weightSum.addAssign(w);
          });
        });
        const velocity: Any = weightSum
          .greaterThan(1e-12)
          .select(velocitySum.div(weightSum.max(1e-12)), vec3(0));
        tracers.smokePositions.element(s).assign(vec4(position.add(velocity.mul(dt)), 0));
        tracers.smokeVelocities.element(s).assign(vec4(velocity, 0));
        tracers.smokeAge.element(s).assign(age);
      });
    });
  })()
    .compute(tracers.capacity)
    .setName('smokeAdvect.smokeAdvect');
}
