import type { BufferGeometry } from 'three';

import type { SDFData } from '../core/collision/SDFCollider.js';
import { toTriangleMesh, type TriangleMesh } from '../core/mesh.js';

/**
 * Offline SDF baker — brute-force CPU implementation.
 *
 * For each voxel center:
 *   1. Find the closest point on the input mesh (minimum over all triangles
 *      of the point-to-triangle distance; Ericson RTCD §5.1.5).
 *   2. Determine sign via 3-axis ray-casting majority vote (Möller-Trumbore
 *      ray-triangle intersection counts along +X, +Y, +Z; voxel is inside
 *      when ≥ 2 of the 3 counts are odd).
 *   3. φ = sign · minDistance.
 *
 * Complexity: O(V · T) where V = resolution³, T = triangle count; there is
 * no acceleration structure, so bake offline (as `npm run assets:honey`
 * does) and keep collision meshes to a few thousand triangles.
 *
 * Watertightness assumption: the ray-cast sign test is only correct on
 * manifold, watertight meshes. Meshes with holes, T-junctions, or doubled
 * faces produce inconsistent axis votes. We detect this by counting voxels
 * where the three axis votes disagree; above
 * {@link MAX_INCONSISTENT_VOXEL_FRACTION}, `bakeMeshToSdf` throws with
 * diagnostic counts so the caller can re-mesh rather than silently storing
 * a partly-wrong SDF.
 */

export interface BakeOptions {
  /** Voxels along each axis of the cubic grid, at least 4. */
  readonly resolution: number;
  /**
   * Space added around the mesh's bounds on every side, in metres. Outside
   * the grid the field reads its edge voxels, so keep this above the contact
   * distance (particle radius plus any collider thickness). Default: two
   * voxels (one below resolution 8).
   */
  readonly padding?: number;
}

/**
 * Maximum fraction of voxels whose 3-axis sign votes disagree before the
 * baker rejects the input mesh as non-watertight.
 *
 * An odd-vote inconsistency rate above this threshold means the mesh is
 * not manifold and the baked SDF's sign is unreliable. A small amount of
 * noise (< 1%) is expected from voxels whose center lands within f32 epsilon
 * of a triangle edge or vertex; above 1% we treat it as a structural defect.
 */
const MAX_INCONSISTENT_VOXEL_FRACTION = 0.01;

const RAY_EPS = 1e-7;

/** Internal scratch triangle — pre-extracted to avoid re-indexing per voxel. */
interface Triangle {
  readonly ax: number;
  readonly ay: number;
  readonly az: number;
  readonly bx: number;
  readonly by: number;
  readonly bz: number;
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly minX: number;
  readonly minY: number;
  readonly minZ: number;
  readonly maxX: number;
  readonly maxY: number;
  readonly maxZ: number;
}

/**
 * Bake a triangle mesh into a 3D signed-distance field.
 *
 * The mesh must be watertight and manifold — see module doc. A cubic voxel
 * grid is fitted around the mesh's AABB expanded by `padding` on every axis,
 * scaled up so the longest axis fits exactly; the mesh stays centered in the
 * grid.
 */
export function bakeMeshToSdf(mesh: BufferGeometry | TriangleMesh, options: BakeOptions): SDFData {
  const { vertices, indices } = toTriangleMesh(mesh);
  const input = {
    positions: vertices,
    indices,
    resolution: options.resolution,
    padding: options.padding,
  };
  validateInput(input);

  const triangles = extractTriangles(input.positions, input.indices);
  const { origin, voxelSize } = fitCubicGrid(triangles, input.padding, input.resolution);
  const res = input.resolution;
  const data = new Float32Array(res * res * res);

  let inconsistentVotes = 0;
  // Rays start this far off the voxel center, sideways, so they don't run
  // exactly along the edges of meshes that are symmetric about the grid
  // (a sphere's meridians on the grid diagonal), where both neighbouring
  // triangles count a hit.
  const jitter = voxelSize * 1e-4;

  // Per-voxel brute-force loop. Hottest path in the baker; keep simple.
  for (let zi = 0; zi < res; zi++) {
    const pz = origin[2] + (zi + 0.5) * voxelSize;
    for (let yi = 0; yi < res; yi++) {
      const py = origin[1] + (yi + 0.5) * voxelSize;
      const rowBase = zi * res * res + yi * res;
      for (let xi = 0; xi < res; xi++) {
        const px = origin[0] + (xi + 0.5) * voxelSize;

        const minDistSq = closestTriangleDistSq(px, py, pz, triangles);
        const { inside, inconsistent } = insideByRayMajority(px, py, pz, triangles, jitter);
        if (inconsistent) inconsistentVotes++;

        const dist = Math.sqrt(minDistSq);
        data[rowBase + xi] = inside ? -dist : dist;
      }
    }
  }

  const totalVoxels = res * res * res;
  const inconsistentFraction = inconsistentVotes / totalVoxels;
  if (inconsistentFraction > MAX_INCONSISTENT_VOXEL_FRACTION) {
    throw new Error(
      `bakeMeshToSdf: mesh appears non-watertight — ${inconsistentVotes} of ${totalVoxels} ` +
        `voxels (${(inconsistentFraction * 100).toFixed(2)}%) had disagreeing axis sign votes. ` +
        `Threshold is ${(MAX_INCONSISTENT_VOXEL_FRACTION * 100).toFixed(2)}%. ` +
        `Check the input mesh for holes, T-junctions, or non-manifold edges.`,
    );
  }

  return {
    data,
    resolution: [res, res, res],
    origin: [origin[0], origin[1], origin[2]],
    voxelSize: [voxelSize, voxelSize, voxelSize],
  };
}

function validateInput(input: {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
  readonly resolution: number;
  readonly padding: number | undefined;
}): void {
  if (!Number.isInteger(input.resolution) || input.resolution < 4) {
    throw new Error(`bakeMeshToSdf: resolution must be an integer ≥ 4, got ${input.resolution}`);
  }
  if (input.padding !== undefined && !(input.padding >= 0 && Number.isFinite(input.padding))) {
    throw new Error(
      `bakeMeshToSdf: padding must be a non-negative finite number, got ${input.padding}`,
    );
  }
  if (input.positions.length % 3 !== 0) {
    throw new Error(
      `bakeMeshToSdf: positions.length must be a multiple of 3, got ${input.positions.length}`,
    );
  }
  if (input.indices.length % 3 !== 0) {
    throw new Error(
      `bakeMeshToSdf: indices.length must be a multiple of 3, got ${input.indices.length}`,
    );
  }
  if (input.indices.length === 0) {
    throw new Error('bakeMeshToSdf: mesh has zero triangles');
  }
  const maxIdx = input.positions.length / 3;
  for (let i = 0; i < input.indices.length; i++) {
    const idx = input.indices[i]!;
    if (idx >= maxIdx) {
      throw new Error(`bakeMeshToSdf: index ${idx} out of range (positions count ${maxIdx})`);
    }
  }
}

function extractTriangles(positions: Float32Array, indices: Uint32Array): Triangle[] {
  const out: Triangle[] = [];
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t]! * 3;
    const i1 = indices[t + 1]! * 3;
    const i2 = indices[t + 2]! * 3;
    const ax = positions[i0]!,
      ay = positions[i0 + 1]!,
      az = positions[i0 + 2]!;
    const bx = positions[i1]!,
      by = positions[i1 + 1]!,
      bz = positions[i1 + 2]!;
    const cx = positions[i2]!,
      cy = positions[i2 + 1]!,
      cz = positions[i2 + 2]!;
    out.push({
      ax,
      ay,
      az,
      bx,
      by,
      bz,
      cx,
      cy,
      cz,
      minX: Math.min(ax, bx, cx),
      minY: Math.min(ay, by, cy),
      minZ: Math.min(az, bz, cz),
      maxX: Math.max(ax, bx, cx),
      maxY: Math.max(ay, by, cy),
      maxZ: Math.max(az, bz, cz),
    });
  }
  return out;
}

function fitCubicGrid(
  triangles: Triangle[],
  padding: number | undefined,
  resolution: number,
): { origin: [number, number, number]; voxelSize: number } {
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (const tri of triangles) {
    if (tri.minX < minX) minX = tri.minX;
    if (tri.minY < minY) minY = tri.minY;
    if (tri.minZ < minZ) minZ = tri.minZ;
    if (tri.maxX > maxX) maxX = tri.maxX;
    if (tri.maxY > maxY) maxY = tri.maxY;
    if (tri.maxZ > maxZ) maxZ = tri.maxZ;
  }
  // Default: k voxels on every side of the longest axis, where the voxel
  // size itself includes the padding: p = k · (extent + 2p) / resolution.
  if (padding === undefined) {
    const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const k = resolution >= 8 ? 2 : 1;
    padding = (k * extent) / (resolution - 2 * k);
  }
  // Expand by padding on all sides.
  minX -= padding;
  minY -= padding;
  minZ -= padding;
  maxX += padding;
  maxY += padding;
  maxZ += padding;
  // Longest axis determines the cubic side length.
  const side = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
  const voxelSize = side / resolution;
  // Center the mesh in a cubic grid: origin = center - side/2.
  const cx = 0.5 * (minX + maxX);
  const cy = 0.5 * (minY + maxY);
  const cz = 0.5 * (minZ + maxZ);
  return {
    origin: [cx - 0.5 * side, cy - 0.5 * side, cz - 0.5 * side],
    voxelSize,
  };
}

/**
 * Squared distance from point `(px, py, pz)` to the closest triangle. O(T).
 * Wraps {@link pointTriangleDistSq} per triangle with an AABB-reject pre-check
 * keyed off the best-so-far distance to reduce the exact-distance calls on
 * far triangles.
 */
function closestTriangleDistSq(px: number, py: number, pz: number, triangles: Triangle[]): number {
  let best = Infinity;
  for (const t of triangles) {
    // AABB lower-bound: squared distance from point to the triangle's AABB.
    // If that already exceeds the current best, the exact triangle distance
    // can only be larger (triangle ⊆ AABB).
    const dx = clampOutside(px, t.minX, t.maxX);
    const dy = clampOutside(py, t.minY, t.maxY);
    const dz = clampOutside(pz, t.minZ, t.maxZ);
    const aabbDistSq = dx * dx + dy * dy + dz * dz;
    if (aabbDistSq >= best) continue;
    const d = pointTriangleDistSq(px, py, pz, t.ax, t.ay, t.az, t.bx, t.by, t.bz, t.cx, t.cy, t.cz);
    if (d < best) best = d;
  }
  return best;
}

function clampOutside(p: number, lo: number, hi: number): number {
  if (p < lo) return lo - p;
  if (p > hi) return p - hi;
  return 0;
}

/**
 * Squared distance from point P to triangle (A, B, C). Ericson
 * "Real-Time Collision Detection" §5.1.5 — the 7-region Voronoi
 * classification. Clamped barycentric projection; no trig, no square roots.
 */
function pointTriangleDistSq(
  px: number,
  py: number,
  pz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
): number {
  const abx = bx - ax,
    aby = by - ay,
    abz = bz - az;
  const acx = cx - ax,
    acy = cy - ay,
    acz = cz - az;
  const apx = px - ax,
    apy = py - ay,
    apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) {
    return apx * apx + apy * apy + apz * apz;
  }
  const bpx = px - bx,
    bpy = py - by,
    bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) {
    return bpx * bpx + bpy * bpy + bpz * bpz;
  }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const qx = apx - v * abx,
      qy = apy - v * aby,
      qz = apz - v * abz;
    return qx * qx + qy * qy + qz * qz;
  }
  const cpx = px - cx,
    cpy = py - cy,
    cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) {
    return cpx * cpx + cpy * cpy + cpz * cpz;
  }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const qx = apx - w * acx,
      qy = apy - w * acy,
      qz = apz - w * acz;
    return qx * qx + qy * qy + qz * qz;
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    const qx = bpx - w * (cx - bx);
    const qy = bpy - w * (cy - by);
    const qz = bpz - w * (cz - bz);
    return qx * qx + qy * qy + qz * qz;
  }
  // Face interior region.
  const denom = 1.0 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  const qx = apx - v * abx - w * acx;
  const qy = apy - v * aby - w * acy;
  const qz = apz - v * abz - w * acz;
  return qx * qx + qy * qy + qz * qz;
}

/**
 * Sign determination by 3-axis ray-cast majority vote. Cast rays along +X,
 * +Y, +Z from `(px, py, pz)`; count Möller-Trumbore intersections per axis;
 * each axis votes "inside" if its hit count is odd. Each ray starts `jitter`-scaled
 * off the point, sideways to its axis. Majority (≥ 2 of 3)
 * wins. `inconsistent = true` when the three votes split (e.g. two inside
 * and one outside, or vice versa) — the caller uses this to flag
 * non-watertight meshes per module doc.
 */
function insideByRayMajority(
  px: number,
  py: number,
  pz: number,
  triangles: Triangle[],
  jitter: number,
): { inside: boolean; inconsistent: boolean } {
  // Irrational multiples keep the offset ray off rational-slope edges too.
  const a = jitter * Math.SQRT2;
  const b = jitter * Math.PI;
  const hitsX = countHitsAxis(px, py + a, pz + b, 0, triangles);
  const hitsY = countHitsAxis(px + b, py, pz + a, 1, triangles);
  const hitsZ = countHitsAxis(px + a, py + b, pz, 2, triangles);
  const insideX = (hitsX & 1) === 1;
  const insideY = (hitsY & 1) === 1;
  const insideZ = (hitsZ & 1) === 1;
  const trueCount = (insideX ? 1 : 0) + (insideY ? 1 : 0) + (insideZ ? 1 : 0);
  const inside = trueCount >= 2;
  const inconsistent = trueCount === 1 || trueCount === 2;
  return { inside, inconsistent };
}

/**
 * Count triangle intersections along an axis-aligned ray from `(px, py, pz)`.
 * `axis` is 0 (+X), 1 (+Y), or 2 (+Z). Möller-Trumbore specialized per axis
 * to avoid building a `dir` vector.
 */
function countHitsAxis(
  px: number,
  py: number,
  pz: number,
  axis: number,
  triangles: Triangle[],
): number {
  let hits = 0;
  for (const t of triangles) {
    // Cheap AABB cull on the two perpendicular axes.
    if (axis === 0) {
      if (py < t.minY || py > t.maxY) continue;
      if (pz < t.minZ || pz > t.maxZ) continue;
      if (px > t.maxX) continue; // ray goes +X; triangle fully behind
    } else if (axis === 1) {
      if (px < t.minX || px > t.maxX) continue;
      if (pz < t.minZ || pz > t.maxZ) continue;
      if (py > t.maxY) continue;
    } else {
      if (px < t.minX || px > t.maxX) continue;
      if (py < t.minY || py > t.maxY) continue;
      if (pz > t.maxZ) continue;
    }
    if (rayTriHit(px, py, pz, axis, t)) hits++;
  }
  return hits;
}

/**
 * Axis-aligned-ray vs triangle intersection (Möller-Trumbore, direction
 * hard-coded per axis to skip the cross(dir, edge2) and dot(dir, q)
 * operations). Returns `true` if the ray hits the triangle interior
 * (u,v strictly inside [0,1] with u+v ≤ 1) at t > RAY_EPS.
 *
 * Edge/vertex hits (u == 0, v == 0, or u + v == 1) are counted as hits
 * deterministically; the 3-axis majority vote above absorbs the rare
 * double-count cases.
 */
function rayTriHit(px: number, py: number, pz: number, axis: number, t: Triangle): boolean {
  // dir = (axis==0) ? (1,0,0) : (axis==1) ? (0,1,0) : (0,0,1)
  // edge1 = b - a, edge2 = c - a
  const e1x = t.bx - t.ax,
    e1y = t.by - t.ay,
    e1z = t.bz - t.az;
  const e2x = t.cx - t.ax,
    e2y = t.cy - t.ay,
    e2z = t.cz - t.az;
  // h = cross(dir, e2)
  let hx: number, hy: number, hz: number;
  if (axis === 0) {
    hx = 0;
    hy = -e2z;
    hz = e2y;
  } else if (axis === 1) {
    hx = e2z;
    hy = 0;
    hz = -e2x;
  } else {
    hx = -e2y;
    hy = e2x;
    hz = 0;
  }
  const a = e1x * hx + e1y * hy + e1z * hz;
  if (Math.abs(a) < RAY_EPS) return false;
  const invA = 1 / a;
  const sx = px - t.ax,
    sy = py - t.ay,
    sz = pz - t.az;
  const u = invA * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return false;
  // q = cross(s, e1)
  const qx = sy * e1z - sz * e1y;
  const qy = sz * e1x - sx * e1z;
  const qz = sx * e1y - sy * e1x;
  // v = invA · dot(dir, q)
  const v = invA * (axis === 0 ? qx : axis === 1 ? qy : qz);
  if (v < 0 || u + v > 1) return false;
  const tHit = invA * (e2x * qx + e2y * qy + e2z * qz);
  return tHit > RAY_EPS;
}
