import { Vector3 } from 'three';
import { instancedArray } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import { releaseStorageBuffers, type ParticleSystem } from '../particles.js';
import { buildCellIndexKernels } from './cellIndex.js';
import {
  MAX_CELLS_SINGLE_LEVEL_SCAN,
  SCAN_WORKGROUP_SIZE,
  buildCountSortKernels,
  padToScanWorkgroup,
} from './sort.js';

/**
 * Round `n` up to the next power of two. Used to pick the default
 * `hashTableSize` from `2 · capacity`.
 */
function nextPow2(n: number): number {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export interface HashGridOptions {
  /**
   * Cell edge length. A neighbor query visits the 27 cells around a point,
   * so this must be at least the largest query radius.
   */
  readonly cellSize: number;
  /**
   * Hash buckets; a power of two. Default: the next power of two above
   * `2 × capacity`, which keeps collisions between distant cells rare,
   * capped at 1,048,576.
   */
  readonly hashTableSize?: number;
  /**
   * Offset subtracted from positions before they are quantized to cells.
   * Cells keep their spatial locality within ±512 cells of it on each axis,
   * so set it near the middle of scenes far from the world origin. Default
   * `(0, 0, 0)`.
   */
  readonly hashOrigin?: Vector3;
}

/**
 * The grid's buffers read back to the CPU, for tests and debugging. `counts`,
 * `cellStart`, and `cellEnd` are padded past `hashTableSize` with zeros.
 * Particle order within a bucket may differ from run to run.
 */
export interface HashGridSnapshot {
  readonly capacity: number;
  readonly hashTableSize: number;
  readonly hashTableSizePadded: number;
  readonly cellIndex: Uint32Array;
  readonly counts: Uint32Array;
  readonly cellStart: Uint32Array;
  readonly cellEnd: Uint32Array;
  readonly sortedIndices: Uint32Array;
}

/**
 * Spatial hash for neighbor search (Teschner et al. 2003), rebuilt on the GPU
 * with a counting sort:
 *
 * 1. hash each particle's cell (Morton order), count particles per bucket,
 *    and note each particle's rank within its bucket;
 * 2. prefix-sum the counts (Blelloch scan) into each bucket's start and end;
 * 3. scatter particle indices into bucket order at start + rank, and copy
 *    the predicted positions into that order too, so neighbor walks read
 *    memory contiguously.
 *
 * The space is unbounded: distant cells share buckets, and queries filter
 * the extra candidates by distance. {@link SimLoop} owns and rebuilds a grid
 * for you; build one directly only for custom tools.
 */
export class HashGrid {
  readonly renderer: WebGPURenderer;
  readonly particles: ParticleSystem;
  readonly cellSize: number;
  /** Number of hash buckets, a power of two. */
  readonly hashTableSize: number;
  /** `hashTableSize` rounded up to a multiple of `SCAN_WORKGROUP_SIZE`. */
  readonly hashTableSizePadded: number;

  readonly cellIndex: StorageBufferNode<'uint'>;
  /** Per-bucket histogram, used during a rebuild and zero outside one. */
  readonly counts: StorageBufferNode<'uint'>;
  readonly cellStart: StorageBufferNode<'uint'>;
  readonly cellEnd: StorageBufferNode<'uint'>;
  readonly sortedIndices: StorageBufferNode<'uint'>;
  /** Each particle's slot in the sorted order: the inverse of `sortedIndices`. */
  readonly slotOf: StorageBufferNode<'uint'>;
  /** `predictedPositions` in bucket order, refreshed by every rebuild. */
  readonly sortedPredictedPositions: StorageBufferNode<'vec4'>;
  /**
   * Atomic flag set to 1 during a rebuild when a particle's cell coordinate
   * is outside ±512 cells of {@link hashOrigin} on some axis (or so far out
   * that it had to be clamped). Such particles still find their neighbors,
   * but through buckets shared with distant cells, so queries do extra work.
   */
  readonly overflowFlag: StorageBufferNode<'uint'>;

  readonly hashOriginUniform: UniformNode<'vec3', Vector3>;
  readonly cellSizeUniform: UniformNode<'float', number>;

  private readonly blockSums: StorageBufferNode<'uint'>;
  /** Each particle's position within its bucket, from the histogram's atomic add. */
  private readonly rank: StorageBufferNode<'uint'>;
  private readonly pipeline: ComputeNode[];

  private disposed = false;

  constructor(particles: ParticleSystem, options: HashGridOptions) {
    const { cellSize } = options;

    if (!Number.isFinite(cellSize) || cellSize <= 0) {
      throw new Error(`HashGrid: cellSize must be a positive finite number, got ${cellSize}`);
    }

    // Past the scan's limit, more particles share each bucket, which costs
    // query time but stays correct.
    const hashTableSize =
      options.hashTableSize ??
      Math.min(nextPow2(2 * particles.capacity), MAX_CELLS_SINGLE_LEVEL_SCAN);
    if (
      !Number.isInteger(hashTableSize) ||
      hashTableSize < 1 ||
      (hashTableSize & (hashTableSize - 1)) !== 0
    ) {
      throw new Error(
        `HashGrid: hashTableSize must be a positive power of two, got ${hashTableSize}`,
      );
    }
    const hashTableSizePadded = padToScanWorkgroup(hashTableSize);

    if (hashTableSizePadded > MAX_CELLS_SINGLE_LEVEL_SCAN) {
      throw new Error(
        `HashGrid: hashTableSize=${hashTableSize} (capacity ${particles.capacity}) exceeds ` +
          `the ${MAX_CELLS_SINGLE_LEVEL_SCAN}-bucket limit of the prefix scan. ` +
          `Use a smaller hashTableSize.`,
      );
    }

    const hashOrigin = options.hashOrigin ?? new Vector3(0, 0, 0);

    this.renderer = particles.renderer;
    this.particles = particles;
    this.cellSize = cellSize;
    this.hashTableSize = hashTableSize;
    this.hashTableSizePadded = hashTableSizePadded;

    this.cellIndex = instancedArray(particles.capacity, 'uint');
    this.counts = instancedArray(hashTableSizePadded, 'uint').toAtomic();
    this.cellStart = instancedArray(hashTableSizePadded, 'uint');
    this.cellEnd = instancedArray(hashTableSizePadded, 'uint');
    this.sortedIndices = instancedArray(particles.capacity, 'uint');
    this.slotOf = instancedArray(particles.capacity, 'uint');
    this.sortedPredictedPositions = instancedArray(particles.capacity, 'vec4');
    this.overflowFlag = instancedArray(1, 'uint').toAtomic();
    this.blockSums = instancedArray(SCAN_WORKGROUP_SIZE, 'uint');
    this.rank = instancedArray(particles.capacity, 'uint');

    // Bin by predicted positions: every query runs after prediction, at predicted positions.
    const cellIndexKernels = buildCellIndexKernels(
      particles.predictedPositions,
      this.cellIndex,
      this.rank,
      this.counts,
      this.overflowFlag,
      particles.capacity,
      hashTableSize,
      hashOrigin,
      cellSize,
    );
    this.hashOriginUniform = cellIndexKernels.hashOrigin;
    this.cellSizeUniform = cellIndexKernels.cellSize;

    const sort = buildCountSortKernels(
      this.counts,
      this.cellStart,
      this.cellEnd,
      this.blockSums,
      this.cellIndex,
      this.rank,
      this.sortedIndices,
      this.slotOf,
      particles.predictedPositions,
      this.sortedPredictedPositions,
      particles.capacity,
      hashTableSizePadded,
    );

    this.pipeline = [
      cellIndexKernels.resetOverflowFlag,
      cellIndexKernels.cellIndexAndHistogram,
      sort.blockScan,
      sort.blockSumScan,
      sort.finalizeCellRanges,
      sort.scatter,
    ];
  }

  /**
   * Origin the cells are measured from, in metres: the kernels' uniform
   * itself, so mutate it in place to move the grid. Move it only right
   * before a rebuild: queries read it too, and look in the wrong cells of a
   * grid built around the old origin. Particles more than 512 cells from it
   * on an axis set {@link overflowFlag}.
   */
  get hashOrigin(): Vector3 {
    return this.hashOriginUniform.value;
  }

  /** Rebuild the grid from the current `particles.predictedPositions`. */
  async rebuild(): Promise<void> {
    this.assertAlive();
    await this.renderer.computeAsync(this.pipeline);
  }

  /** The rebuild's kernels, for batching into a larger dispatch. Don't modify it. */
  get rebuildPipeline(): readonly ComputeNode[] {
    this.assertAlive();
    return this.pipeline;
  }

  async readback(): Promise<HashGridSnapshot> {
    this.assertAlive();
    const [cellIndex, cellStart, cellEnd, sortedIndices] = await Promise.all([
      this.renderer.getArrayBufferAsync(this.cellIndex.value),
      this.renderer.getArrayBufferAsync(this.cellStart.value),
      this.renderer.getArrayBufferAsync(this.cellEnd.value),
      this.renderer.getArrayBufferAsync(this.sortedIndices.value),
    ]);
    const starts = new Uint32Array(cellStart);
    const ends = new Uint32Array(cellEnd);
    return {
      capacity: this.particles.capacity,
      hashTableSize: this.hashTableSize,
      hashTableSizePadded: this.hashTableSizePadded,
      cellIndex: new Uint32Array(cellIndex),
      // The rebuild clears the counts buffer for the next one, so recover them.
      counts: ends.map((end, b) => end - starts[b]!),
      cellStart: starts,
      cellEnd: ends,
      sortedIndices: new Uint32Array(sortedIndices),
    };
  }

  /**
   * True if {@link overflowFlag} was set by the last rebuild: some particle
   * was more than 512 cells from {@link hashOrigin} on an axis. Stalls on
   * the GPU.
   */
  async readbackOverflow(): Promise<boolean> {
    this.assertAlive();
    const buf = await this.renderer.getArrayBufferAsync(this.overflowFlag.value);
    return new Uint32Array(buf)[0] !== 0;
  }

  /**
   * Free the grid's GPU buffers. Later rebuilds, readbacks, and reads of
   * {@link rebuildPipeline} throw; kernels that query the grid can't run
   * afterwards.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    releaseStorageBuffers(this.renderer, [
      this.cellIndex,
      this.counts,
      this.cellStart,
      this.cellEnd,
      this.sortedIndices,
      this.slotOf,
      this.sortedPredictedPositions,
      this.overflowFlag,
      this.blockSums,
      this.rank,
    ]);
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('HashGrid has been disposed');
  }
}
