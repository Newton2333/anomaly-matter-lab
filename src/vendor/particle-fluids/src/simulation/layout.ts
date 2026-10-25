import { Box3, Matrix4, Quaternion, Vector3, type BufferGeometry, type Mesh } from 'three';

import { sampleSdf } from '../sdf/index.js';
import type { SDFData } from '../core/index.js';

/** Grid points spaced `spacing` apart filling `box`, offset half a spacing from its corner. */
export function fillBox(box: Box3, spacing: number): number[] {
  const points: number[] = [];
  const { min, max } = box;
  for (let y = min.y + spacing / 2; y < max.y; y += spacing)
    for (let z = min.z + spacing / 2; z < max.z; z += spacing)
      for (let x = min.x + spacing / 2; x < max.x; x += spacing) points.push(x, y, z);
  return points;
}

/** A mesh's geometry in world space. */
export function worldGeometry(mesh: Mesh): BufferGeometry {
  mesh.updateWorldMatrix(true, false);
  return mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
}

/**
 * The radius at which `volume` cubic metres filled at spacing `2r`, plus
 * `area` square metres of cloth at spacing `clothSpacing · r`, take `budget`
 * particles.
 */
export function radiusForBudget(
  volume: number,
  area: number,
  clothSpacing: number,
  budget: number,
): number {
  const count = (r: number) => volume / (2 * r) ** 3 + area / (clothSpacing * r) ** 2;
  let low = 1e-5,
    high = 10;
  for (let i = 0; i < 80; i++) {
    const mid = Math.sqrt(low * high);
    if (count(mid) > budget) low = mid;
    else high = mid;
  }
  return high;
}

/** A solid that liquid particles must not start inside. */
export type Obstacle = (p: Vector3) => number;

export function sphereObstacle(center: Vector3, radius: number): Obstacle {
  return (p) => p.distanceTo(center) - radius;
}

export function boxObstacle(center: Vector3, half: Vector3, rotation: Quaternion): Obstacle {
  const inverse = rotation.clone().invert();
  const local = new Vector3();
  return (p) => {
    local.copy(p).sub(center).applyQuaternion(inverse);
    const q = new Vector3(Math.abs(local.x), Math.abs(local.y), Math.abs(local.z)).sub(half);
    const outside = new Vector3(Math.max(q.x, 0), Math.max(q.y, 0), Math.max(q.z, 0)).length();
    return outside + Math.min(Math.max(q.x, q.y, q.z), 0);
  };
}

export function capsuleObstacle(a: Vector3, b: Vector3, radius: number): Obstacle {
  const ab = b.clone().sub(a);
  const lengthSq = Math.max(ab.lengthSq(), 1e-12);
  const closest = new Vector3();
  return (p) => {
    const t = Math.min(1, Math.max(0, p.clone().sub(a).dot(ab) / lengthSq));
    closest.copy(a).addScaledVector(ab, t);
    return p.distanceTo(closest) - radius;
  };
}

export function sdfObstacle(sdf: SDFData, matrixWorld: Matrix4): Obstacle {
  const inverse = matrixWorld.clone().invert();
  const scale = new Vector3().setFromMatrixScale(matrixWorld).x;
  const local = new Vector3();
  return (p) => {
    local.copy(p).applyMatrix4(inverse);
    return sampleSdf(sdf, local.x, local.y, local.z) * scale;
  };
}

/** Points (xyz triples) whose centres are at least `clearance` from every obstacle and every point in `occupied`. */
export function clearOf(
  points: number[],
  clearance: number,
  obstacles: readonly Obstacle[],
  occupied: Float32Array[],
): number[] {
  const cell = clearance;
  const key = (x: number, y: number, z: number) =>
    `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  const grid = new Map<string, number[]>();
  for (const positions of occupied)
    for (let i = 0; i < positions.length; i += 3) {
      const k = key(positions[i]!, positions[i + 1]!, positions[i + 2]!);
      const list = grid.get(k) ?? [];
      list.push(positions[i]!, positions[i + 1]!, positions[i + 2]!);
      grid.set(k, list);
    }
  const kept: number[] = [];
  const p = new Vector3();
  outer: for (let i = 0; i < points.length; i += 3) {
    p.set(points[i]!, points[i + 1]!, points[i + 2]!);
    for (const obstacle of obstacles) if (obstacle(p) < clearance) continue outer;
    const cx = Math.floor(p.x / cell),
      cy = Math.floor(p.y / cell),
      cz = Math.floor(p.z / cell);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const list = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!list) continue;
          for (let j = 0; j < list.length; j += 3) {
            const ddx = list[j]! - p.x,
              ddy = list[j + 1]! - p.y,
              ddz = list[j + 2]! - p.z;
            if (ddx * ddx + ddy * ddy + ddz * ddz < clearance * clearance) continue outer;
          }
        }
    kept.push(p.x, p.y, p.z);
  }
  return kept;
}
