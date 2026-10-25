import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute } from 'three';

import type { SoftbodySystem } from './SoftbodySystem.js';

/** Particles each vertex follows. */
const INFLUENCES = 4;
/**
 * Particles with fewer face neighbors than this sit on one-particle-thick
 * chains (the tip of a bunny ear, say), which shape matching can't hold
 * steady. Vertices bind to steadier particles instead, so thin features
 * move rigidly with the body rather than tearing.
 */
const STABLE_DEGREE = 3;

/**
 * Skin `geometry` to a soft body: give every vertex its four nearest stable
 * particles and inverse-distance weights that sum to one, stored as the
 * `influences` and `weights` attributes. The geometry must be positioned
 * like the body's rest shape. Returns the geometry.
 *
 * Only {@link SoftbodyMesh} calls this, so its errors carry that name.
 */
export function bindSoftbodyMesh(
  geometry: BufferGeometry,
  softbody: SoftbodySystem,
  bodyIndex: number,
): BufferGeometry {
  const body = softbody.bodies[bodyIndex];
  if (!body) throw new Error(`SoftbodyMesh: no body ${bodyIndex} (have ${softbody.bodies.length})`);
  const position = geometry.getAttribute('position');
  if (!position) throw new Error('SoftbodyMesh: geometry has no position attribute');
  const { restPositions: rest, range, edges } = body;
  const n = range.count;

  const stable = new Uint8Array(n).fill(1);
  if (edges && edges.length > 0) {
    const degree = new Uint32Array(n);
    for (const i of edges) degree[i]!++;
    for (let i = 0; i < n; i++) stable[i] = degree[i]! >= STABLE_DEGREE ? 1 : 0;
    // A body made only of chains still has to bind to something.
    if (!stable.includes(1)) stable.fill(1);
  }

  const influences = new Uint32Array(INFLUENCES * position.count).fill(range.start);
  const weights = new Float32Array(INFLUENCES * position.count);
  // Keeps a vertex sitting on a particle from getting an infinite weight.
  const minDistance = 0.1 * softbody.particles.particleRadius;
  const bestDistSq = new Float64Array(INFLUENCES);
  const bestIndex = new Int32Array(INFLUENCES);
  for (let v = 0; v < position.count; v++) {
    const x = position.getX(v),
      y = position.getY(v),
      z = position.getZ(v);
    bestDistSq.fill(Infinity);
    bestIndex.fill(-1);
    for (let i = 0; i < n; i++) {
      if (!stable[i]) continue;
      const dx = rest[i * 3]! - x,
        dy = rest[i * 3 + 1]! - y,
        dz = rest[i * 3 + 2]! - z;
      const d = dx * dx + dy * dy + dz * dz;
      let worst = 0;
      for (let k = 1; k < INFLUENCES; k++) if (bestDistSq[k]! > bestDistSq[worst]!) worst = k;
      if (d < bestDistSq[worst]!) {
        bestDistSq[worst] = d;
        bestIndex[worst] = i;
      }
    }
    let total = 0;
    for (let k = 0; k < INFLUENCES; k++) {
      if (bestIndex[k]! < 0) continue;
      total += 1 / Math.max(Math.sqrt(bestDistSq[k]!), minDistance);
    }
    for (let k = 0; k < INFLUENCES; k++) {
      const i = bestIndex[k]!;
      if (i < 0) continue;
      influences[v * INFLUENCES + k] = range.start + i;
      weights[v * INFLUENCES + k] = 1 / Math.max(Math.sqrt(bestDistSq[k]!), minDistance) / total;
    }
  }
  geometry.setAttribute('influences', new Uint32BufferAttribute(influences, INFLUENCES));
  geometry.setAttribute('weights', new Float32BufferAttribute(weights, INFLUENCES));
  return geometry;
}
