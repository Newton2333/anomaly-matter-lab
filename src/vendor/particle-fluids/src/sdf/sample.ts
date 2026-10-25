import type { SDFData } from '../core/collision/SDFCollider.js';

/**
 * Sample a distance field on the CPU with trilinear filtering, matching what
 * the GPU sees.
 *
 * Coordinate convention matches the GPU sampler's half-texel-centered UV:
 * voxel center `(i, j, k)` sits at world position
 * `origin + (i + 0.5, j + 0.5, k + 0.5) · voxelSize`. Queries outside
 * `[origin, origin + resolution · voxelSize]` clamp to the edge voxel.
 */
export function sampleSdf(sdf: SDFData, x: number, y: number, z: number): number {
  const { data, resolution, origin, voxelSize } = sdf;
  const [resX, resY, resZ] = resolution;
  if (!(isGridSize(resX) && isGridSize(resY) && isGridSize(resZ))) {
    throw new Error(`sampleSdf: resolution must be integers ≥ 2, got [${resX}, ${resY}, ${resZ}]`);
  }
  // Continuous voxel index — 0.5 offset so voxel (0,0,0) center is at gc = 0.5.
  const gx = (x - origin[0]) / voxelSize[0] - 0.5;
  const gy = (y - origin[1]) / voxelSize[1] - 0.5;
  const gz = (z - origin[2]) / voxelSize[2] - 0.5;
  const i0 = clampInt(Math.floor(gx), 0, resX - 2);
  const j0 = clampInt(Math.floor(gy), 0, resY - 2);
  const k0 = clampInt(Math.floor(gz), 0, resZ - 2);
  const tx = clamp01(gx - i0);
  const ty = clamp01(gy - j0);
  const tz = clamp01(gz - k0);
  const stride = resX;
  const sliceStride = resX * resY;
  const base = i0 + j0 * stride + k0 * sliceStride;
  const c000 = data[base]!;
  const c100 = data[base + 1]!;
  const c010 = data[base + stride]!;
  const c110 = data[base + stride + 1]!;
  const c001 = data[base + sliceStride]!;
  const c101 = data[base + sliceStride + 1]!;
  const c011 = data[base + sliceStride + stride]!;
  const c111 = data[base + sliceStride + stride + 1]!;
  const c00 = lerp(c000, c100, tx);
  const c10 = lerp(c010, c110, tx);
  const c01 = lerp(c001, c101, tx);
  const c11 = lerp(c011, c111, tx);
  const c0 = lerp(c00, c10, ty);
  const c1 = lerp(c01, c11, ty);
  return lerp(c0, c1, tz);
}

/**
 * CPU central-difference gradient — matches the 6-sample pattern in
 * `emitSampleSdf`. Step size is `voxelSize` per axis; returns the raw
 * (un-normalized) `∇φ` so callers can check `|∇φ| ≈ 1`.
 */
export function sampleSdfGradient(
  sdf: SDFData,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const [hx, hy, hz] = sdf.voxelSize;
  const phiXp = sampleSdf(sdf, x + hx, y, z);
  const phiXn = sampleSdf(sdf, x - hx, y, z);
  const phiYp = sampleSdf(sdf, x, y + hy, z);
  const phiYn = sampleSdf(sdf, x, y - hy, z);
  const phiZp = sampleSdf(sdf, x, y, z + hz);
  const phiZn = sampleSdf(sdf, x, y, z - hz);
  return [(phiXp - phiXn) / (2 * hx), (phiYp - phiYn) / (2 * hy), (phiZp - phiZn) / (2 * hz)];
}

function isGridSize(n: number): boolean {
  return Number.isInteger(n) && n >= 2;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

function clampInt(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
