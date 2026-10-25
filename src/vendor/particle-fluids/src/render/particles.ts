import { Color, InstancedMesh, SphereGeometry } from 'three';
import { MeshPhongNodeMaterial } from 'three/webgpu';
import { instanceIndex, positionLocal, uint } from 'three/tsl';

import { assertRange, type ParticleRange, type ParticleSystem } from '../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface ParticleMeshOptions {
  /** Particles to draw. Default: all of them. */
  readonly range?: ParticleRange;
  /** Sphere radius. Default: 90% of the particle radius, so neighbors read as separate. */
  readonly radius?: number;
  /** Default `0x5fb9ff`. Ignored when `colorNode` is given. */
  readonly color?: number | string;
  /** Per-particle color from its position, as a TSL vec3 node. */
  readonly colorNode?: (position: Any) => Any;
  /** Sphere tessellation. Default 8 × 6. */
  readonly widthSegments?: number;
  readonly heightSegments?: number;
}

/**
 * Draw particles as instanced spheres whose positions are read straight
 * from the simulation's GPU buffer, so there is no per-frame CPU work.
 * Handy for debugging, or for showing the particles under a surface.
 */
export function createParticleMesh(
  particles: ParticleSystem,
  options: ParticleMeshOptions = {},
): InstancedMesh {
  const range = options.range ?? { start: 0, count: particles.capacity };
  assertRange(particles, range, 'createParticleMesh');
  const radius = options.radius ?? particles.particleRadius * 0.9;
  const geometry = new SphereGeometry(
    radius,
    options.widthSegments ?? 8,
    options.heightSegments ?? 6,
  );
  const material = new MeshPhongNodeMaterial({
    color: new Color(options.color ?? 0x5fb9ff),
    shininess: 60,
    specular: new Color(0x334466),
  });
  const center: Any = particles.positions.element(instanceIndex.add(uint(range.start))).xyz;
  material.positionNode = center.add(positionLocal);
  if (options.colorNode) material.colorNode = options.colorNode(center);
  const mesh = new InstancedMesh(geometry, material, range.count);
  mesh.frustumCulled = false;
  return mesh;
}
