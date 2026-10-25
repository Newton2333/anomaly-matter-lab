import type { BufferGeometry } from 'three';

import { sampleSdf } from '../sdf/sample.js';
import type { SDFData } from '../core/collision/SDFCollider.js';
import { toTriangleMesh, type TriangleMesh } from '../core/mesh.js';

export interface VoxelizeOptions {
  /** Radius of the particles. They are placed on a grid with spacing `2 · particleRadius`. */
  readonly particleRadius: number;
  /**
   * Keep only the largest face-connected piece. Thin features can voxelize
   * into islands that no longer belong to the body. Default `false`.
   */
  readonly largestPiece?: boolean;
  /**
   * Distance field input only (mesh input throws): also fill grid points up
   * to this far outside the surface, in metres. A little dilation keeps thin
   * parts attached. Default 0.
   */
  readonly dilation?: number;
}

export interface VoxelizeResult {
  /**
   * xyz per particle, surface particles first, in the input's own space:
   * the geometry's local coordinates for a mesh, the field's coordinates
   * (those of `origin`) for {@link SDFData}.
   */
  readonly positions: Float32Array;
  /** Number of particles. 0 when the shape is too small for the grid. */
  readonly count: number;
  /** Number of leading particles on the surface. */
  readonly surfaceCount: number;
  /** Pairs of face-adjacent particles, `[i0, j0, i1, j1, …]` with `i < j`. */
  readonly edges: Uint32Array;
}

/**
 * Fill a closed shape with particles on a cubic grid, ready to become a soft
 * body. The shape can be a mesh (inside is decided by ray-cast parity, so it
 * must be closed) or a baked signed distance field.
 *
 * Particles are ordered surface first, as {@link SoftbodySystem} expects;
 * surface particles are those with a missing grid neighbor. Positions are in
 * the shape's own space: a `BufferGeometry` is read without any object
 * transform, and a field's positions share its `origin`'s coordinates. Move
 * them into world space before uploading. Check `count`: a shape smaller
 * than the grid spacing gives no particles.
 */
export function voxelize(
  shape: BufferGeometry | TriangleMesh | SDFData,
  options: VoxelizeOptions,
): VoxelizeResult {
  const { particleRadius } = options;
  if (!(particleRadius > 0) || !Number.isFinite(particleRadius)) {
    throw new Error(`voxelize: particleRadius must be positive, got ${particleRadius}`);
  }
  const { dilation } = options;
  const isField = 'data' in shape;
  if (dilation !== undefined) {
    if (!isField) throw new Error('voxelize: dilation applies only to SDFData input');
    if (!Number.isFinite(dilation)) {
      throw new Error(`voxelize: dilation must be a finite number, got ${dilation}`);
    }
  }
  const spacing = 2 * particleRadius;
  const { min, max, inside } = isField
    ? sdfShape(shape, dilation ?? 0)
    : meshShape(toTriangleMesh(shape));
  const dims = [0, 1, 2].map((a) => Math.max(1, Math.ceil((max[a]! - min[a]!) / spacing))) as [
    number,
    number,
    number,
  ];
  const [nx, ny, nz] = dims;
  const cellCount = nx * ny * nz;
  const center = (cell: number, axis: number): number =>
    min[axis]! +
    particleRadius +
    spacing * [cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny))][axis]!;

  const occupied = new Uint8Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    if (inside(center(cell, 0), center(cell, 1), center(cell, 2))) occupied[cell] = 1;
  }

  // Face-adjacent occupied neighbors of a cell.
  const neighbors = (cell: number): number[] => {
    const x = cell % nx,
      y = Math.floor(cell / nx) % ny,
      z = Math.floor(cell / (nx * ny));
    const out: number[] = [];
    if (x > 0) out.push(cell - 1);
    if (x < nx - 1) out.push(cell + 1);
    if (y > 0) out.push(cell - nx);
    if (y < ny - 1) out.push(cell + nx);
    if (z > 0) out.push(cell - nx * ny);
    if (z < nz - 1) out.push(cell + nx * ny);
    return out.filter((n) => occupied[n]);
  };

  if (options.largestPiece) {
    const piece = new Int32Array(cellCount).fill(-1);
    let largest: number[] = [];
    for (let seed = 0; seed < cellCount; seed++) {
      if (!occupied[seed] || piece[seed] !== -1) continue;
      const members = [seed];
      piece[seed] = seed;
      for (let k = 0; k < members.length; k++) {
        for (const n of neighbors(members[k]!)) {
          if (piece[n] === -1) {
            piece[n] = seed;
            members.push(n);
          }
        }
      }
      if (members.length > largest.length) largest = members;
    }
    occupied.fill(0);
    for (const cell of largest) occupied[cell] = 1;
  }

  const cells: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) if (occupied[cell]) cells.push(cell);
  const isSurface = (cell: number): boolean => neighbors(cell).length < 6;
  const surfaceCells = cells.filter(isSurface);
  const ordered = [...surfaceCells, ...cells.filter((cell) => !isSurface(cell))];
  const particleOf = new Map(ordered.map((cell, i) => [cell, i]));

  const positions = new Float32Array(ordered.length * 3);
  ordered.forEach((cell, i) => {
    for (let a = 0; a < 3; a++) positions[i * 3 + a] = center(cell, a);
  });
  const edges: number[] = [];
  for (const cell of ordered) {
    const i = particleOf.get(cell)!;
    for (const n of neighbors(cell)) {
      const j = particleOf.get(n)!;
      if (i < j) edges.push(i, j);
    }
  }
  return {
    positions,
    count: ordered.length,
    surfaceCount: surfaceCells.length,
    edges: Uint32Array.from(edges),
  };
}

interface Shape {
  readonly min: readonly number[];
  readonly max: readonly number[];
  inside(x: number, y: number, z: number): boolean;
}

function meshShape(mesh: TriangleMesh): Shape {
  if (mesh.vertices.length === 0) throw new Error('voxelize: mesh has no vertices');
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < mesh.vertices.length; v += 3) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a]!, mesh.vertices[v + a]!);
      max[a] = Math.max(max[a]!, mesh.vertices[v + a]!);
    }
  }
  return { min, max, inside: (x, y, z) => isInsideMesh(mesh, x, y, z) };
}

function sdfShape(sdf: SDFData, dilation: number): Shape {
  const min = [...sdf.origin];
  const max = sdf.origin.map((o, a) => o + sdf.resolution[a]! * sdf.voxelSize[a]!);
  return { min, max, inside: (x, y, z) => sampleSdf(sdf, x, y, z) < dilation };
}

/** Möller–Trumbore ray–triangle test: the hit's ray parameter `t`, or −1 for a miss. */
const INTERSECT_EPS = 1e-9;

function rayTriangleIntersect(
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
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
  const e1x = bx - ax;
  const e1y = by - ay;
  const e1z = bz - az;
  const e2x = cx - ax;
  const e2y = cy - ay;
  const e2z = cz - az;
  // h = d × e2
  const hx = dy * e2z - dz * e2y;
  const hy = dz * e2x - dx * e2z;
  const hz = dx * e2y - dy * e2x;
  // a = e1 · h
  const a = e1x * hx + e1y * hy + e1z * hz;
  if (a > -INTERSECT_EPS && a < INTERSECT_EPS) return -1; // parallel
  const f = 1 / a;
  // s = origin − a
  const sx = ox - ax;
  const sy = oy - ay;
  const sz = oz - az;
  // u = f · (s · h)
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return -1;
  // q = s × e1
  const qx = sy * e1z - sz * e1y;
  const qy = sz * e1x - sx * e1z;
  const qz = sx * e1y - sy * e1x;
  // v = f · (d · q)
  const v = f * (dx * qx + dy * qy + dz * qz);
  if (v < 0 || u + v > 1) return -1;
  // t = f · (e2 · q)
  const t = f * (e2x * qx + e2y * qy + e2z * qz);
  if (t < INTERSECT_EPS) return -1; // behind origin
  return t;
}

/**
 * Point-in-mesh by ray-cast parity: an odd number of crossings means inside.
 * The ray is slightly off-axis so it doesn't graze axis-aligned edges.
 */
function isInsideMesh(mesh: TriangleMesh, px: number, py: number, pz: number): boolean {
  const { vertices, indices } = mesh;
  let crossings = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]! * 3,
      b = indices[t + 1]! * 3,
      c = indices[t + 2]! * 3;
    const hit = rayTriangleIntersect(
      px,
      py,
      pz,
      1,
      0.00137,
      0.00241,
      vertices[a]!,
      vertices[a + 1]!,
      vertices[a + 2]!,
      vertices[b]!,
      vertices[b + 1]!,
      vertices[b + 2]!,
      vertices[c]!,
      vertices[c + 1]!,
      vertices[c + 2]!,
    );
    if (hit > 0) crossings++;
  }
  return (crossings & 1) === 1;
}
