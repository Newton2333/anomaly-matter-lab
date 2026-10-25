import { Fn, If, instanceIndex, vec4 } from 'three/tsl';
import type { Vector3 } from 'three';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from './particles.js';

// TSL's type declarations drop the operator methods on many node types.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface IntegrationKernels {
  /** Apply gravity and predict `x* = x + v·dt` (Macklin & Müller 2013, Algorithm 1). */
  readonly predict: ComputeNode;
  /** Derive `v = (x* − x) / dt` from the solved positions and commit `x = x*`. */
  readonly advect: ComputeNode;
}

/** Build the predict and advect kernels. Particles with zero inverse mass stay put. */
export function buildIntegrationKernels(
  particles: ParticleSystem,
  dt: UniformNode<'float', number>,
  gravity: UniformNode<'vec3', Vector3>,
): IntegrationKernels {
  const predict = Fn(() => {
    const i: Any = instanceIndex;
    If(particles.invMass.element(i).greaterThan(0), () => {
      const v: Any = particles.velocities.element(i);
      const velocity: Any = v.xyz.add((gravity as Any).mul(dt));
      v.assign(vec4(velocity, 0));
      const x: Any = particles.positions.element(i).xyz;
      particles.predictedPositions.element(i).assign(vec4(x.add(velocity.mul(dt)), 0));
    });
  })()
    .compute(particles.capacity)
    .setName('integrate.predict');

  const advect = Fn(() => {
    const i: Any = instanceIndex;
    If(particles.invMass.element(i).greaterThan(0), () => {
      const x: Any = particles.positions.element(i);
      const xStar: Any = particles.predictedPositions.element(i);
      particles.velocities.element(i).assign(vec4(xStar.xyz.sub(x.xyz).div(dt), 0));
      x.assign(xStar);
    });
  })()
    .compute(particles.capacity)
    .setName('integrate.advect');

  return { predict, advect };
}
