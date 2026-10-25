import { Vector3 } from 'three';
import {
  Fn,
  atomicAdd,
  atomicMax,
  atomicStore,
  instanceIndex,
  int,
  uniform,
  uint,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import { MORTON_BIAS, mortonBucketUnmasked } from './mortonHash.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Outer safety clamp for the f32→i32 cast. Cell coordinates outside this
 * range produce undefined behaviour at the cast (WGSL spec: "implementation-
 * defined"). We clamp `floor((x − origin) / cellSize)` to `[−2^30, 2^30 − 1]`
 * before the cast so the cast is well-defined for any input. At
 * `cellSize = 0.1 m` the safe range is `|x − origin| ≤ ~1.1 × 10⁸ m` —
 * effectively unbounded for any realistic scene; a particle reaching this
 * threshold is a simulation collapse.
 *
 * The Morton encoding has a tighter contract — `[−MORTON_BIAS, +MORTON_BIAS − 1]`
 * — that is enforced separately below via the `overflowFlag`. The CELL_COORD_MAX
 * clamp protects the cast itself; the Morton check protects the encoding's
 * spatial-locality guarantee.
 */
const CELL_COORD_MAX = 1 << 30;

/**
 * Kernels that reset the per-bucket histogram + overflow flag and compute
 * per-particle hash buckets while populating the histogram.
 *
 * Bucket function: 3D Morton (Z-curve) encoding with a fixed positive bias,
 * so spatially adjacent cells land in nearby buckets and the 27-cell
 * neighbor walk's lookups cluster. See `mortonHash.ts` for the encoding and
 * its supported cell range.
 *
 * Hash-collision tax: distinct cells can still map to the same bucket
 * (Morton's lower bits collide for cells differing only in higher bits, plus
 * the standard mask-collision when `hashTableSize` is smaller than the
 * encoding range). Per-bucket sorted-indices lists therefore remain a
 * possibly-mixed union of cells' particles, and every caller's
 * `onCandidate` distance filter rejects the false-positive pairs.
 *
 * Overflow handling: TWO checks. (1) `CELL_COORD_MAX` clamp keeps the
 * f32→i32 cast well-defined for any input. (2) Cell coordinates outside
 * `[−MORTON_BIAS, +MORTON_BIAS − 1]` set the overflow flag. Particles
 * outside the Morton range still simulate (Morton's lower bits wrap-alias)
 * but their neighbour queries lose spatial-locality and may pay extra
 * false-positive walking cost. The flag makes this observable; tests
 * assert the flag stays zero on supported scenes.
 */
export interface CellIndexKernels {
  /** One-thread dispatch; zeroes the overflow flag. */
  readonly resetOverflowFlag: ComputeNode;
  /**
   * Dispatched over `capacity`. Needs `counts` zeroed, as the previous
   * rebuild's finalize pass leaves it. Also records each particle's rank
   * within its bucket, which places it in the sorted order without atomics.
   */
  readonly cellIndexAndHistogram: ComputeNode;
  /** Mutable scene-centering offset subtracted before cell-coord quantization. */
  readonly hashOrigin: UniformNode<'vec3', Vector3>;
  /** Mutable cell edge length. */
  readonly cellSize: UniformNode<'float', number>;
}

/**
 * Build the reset + cell-index + histogram kernels against the given
 * storage. `hashTableSize` is embedded as a build-time constant because
 * changing it requires resizing `counts` / `cellStart` / `cellEnd`
 * anyway; mutating it at runtime would desync them.
 */
export function buildCellIndexKernels(
  positions: StorageBufferNode<'vec4'>,
  cellIndex: StorageBufferNode<'uint'>,
  rank: StorageBufferNode<'uint'>,
  counts: StorageBufferNode<'uint'>,
  overflowFlag: StorageBufferNode<'uint'>,
  capacity: number,
  hashTableSize: number,
  initialHashOrigin: Vector3,
  initialCellSize: number,
): CellIndexKernels {
  if ((hashTableSize & (hashTableSize - 1)) !== 0) {
    throw new Error(
      `buildCellIndexKernels: hashTableSize must be a power of two, got ${hashTableSize}`,
    );
  }
  const hashOrigin = uniform(initialHashOrigin.clone());
  const cellSize = uniform(initialCellSize, 'float');

  const bucketMask = hashTableSize - 1;

  const resetOverflowFlag = Fn(() => {
    atomicStore(overflowFlag.element(uint(0)), uint(0));
  })()
    .compute(1)
    .setName('cellIndex.resetOverflowFlag');

  const cellIndexAndHistogram = Fn(() => {
    const i: Any = instanceIndex;
    const p: Any = positions.element(i);
    // Teschner §3 discretization with an optional origin shift. The
    // subtract-then-divide order keeps `(x − origin)` small enough that
    // f32 mantissa loss around `x ≈ 10⁷` m does not perturb the cell
    // coordinate by more than one unit — the scene-centered default
    // (`hashOrigin = 0`) is the precision-preserving choice for any
    // container whose extent is << 10⁸ m.
    const rel: Any = p.xyz.sub(hashOrigin as Any).div(cellSize as Any);
    // f32 clamp before the i32 cast — keeps the cast well-defined even
    // for simulation-collapse positions (see CELL_COORD_MAX JSDoc above).
    const cxRawF: Any = rel.x.floor();
    const cyRawF: Any = rel.y.floor();
    const czRawF: Any = rel.z.floor();
    const lo: Any = int(-CELL_COORD_MAX);
    const hi: Any = int(CELL_COORD_MAX - 1);
    const cxRaw: Any = cxRawF.toInt();
    const cyRaw: Any = cyRawF.toInt();
    const czRaw: Any = czRawF.toInt();
    const cx: Any = cxRaw.clamp(lo, hi);
    const cy: Any = cyRaw.clamp(lo, hi);
    const cz: Any = czRaw.clamp(lo, hi);
    // Morton (Z-curve) bucket encoding. See `mortonHash.ts` for the bias
    // contract and the `part1by2` bit-spreading derivation. The mask
    // clamps the up-to-30-bit Morton code into the bucket range; the
    // lower bits preserve spatial locality for cells within
    // `±2^(log2(hashTableSize)/3)` of the bias center.
    const mortonCode: Any = mortonBucketUnmasked(cx, cy, cz);
    const bucket: Any = mortonCode.bitAnd(uint(bucketMask));
    cellIndex.element(i).assign(bucket);
    rank.element(i).assign(atomicAdd(counts.element(bucket), uint(1)));
    // Overflow check 1: f32→i32 saturation (CELL_COORD_MAX clamp fired).
    // Overflow check 2: cell coord outside the Morton range. The Morton
    // encoding wrap-aliases for out-of-range cells; the flag makes this
    // observable from CPU. The `notEqual(cx)` short-circuits to false for
    // the common in-range case.
    const overSatX: Any = cxRaw.notEqual(cx);
    const overSatY: Any = cyRaw.notEqual(cy);
    const overSatZ: Any = czRaw.notEqual(cz);
    const overMortonX: Any = cx
      .lessThan(int(-MORTON_BIAS))
      .or(cx.greaterThanEqual(int(MORTON_BIAS)));
    const overMortonY: Any = cy
      .lessThan(int(-MORTON_BIAS))
      .or(cy.greaterThanEqual(int(MORTON_BIAS)));
    const overMortonZ: Any = cz
      .lessThan(int(-MORTON_BIAS))
      .or(cz.greaterThanEqual(int(MORTON_BIAS)));
    const anyOver: Any = overSatX
      .or(overSatY)
      .or(overSatZ)
      .or(overMortonX)
      .or(overMortonY)
      .or(overMortonZ);
    const flagVal: Any = anyOver.select(uint(1), uint(0));
    atomicMax(overflowFlag.element(uint(0)), flagVal);
  })()
    .compute(capacity)
    .setName('cellIndex.cellIndexAndHistogram');

  return {
    resetOverflowFlag,
    cellIndexAndHistogram,
    hashOrigin,
    cellSize,
  };
}
