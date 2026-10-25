import { Fn, If, Loop, Return, atomicStore, instanceIndex, instancedArray, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { assertRange, type ParticleRange, type ParticleSystem } from '../particles.js';
import type { HashGrid } from './HashGrid.js';
import { emitForEachNeighbor } from './query.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Most neighbors stored per particle. Extra neighbors are dropped and flagged. */
export const MAX_NEIGHBORS = 64;

/**
 * Neighbors within a radius for every particle in a range, gathered once per
 * substep so kernels that visit neighbors several times (density, pressure,
 * viscosity, …) don't each walk the grid (Macklin & Müller 2013, §6).
 *
 * Lists are kept per grid slot rather than per particle: the particle in
 * slot `s` of the grid's sorted order has its list in row `s`. Kernels that
 * read the lists run one thread per slot (see {@link emitThread}), so a
 * workgroup handles particles that are close together, whose neighbors
 * overlap and stay in cache. Rows are stored column-major: neighbor `k` of
 * row `s` lives at `k · capacity + s`, so a warp reading its `k`th
 * neighbors reads contiguously. Rows of particles outside the range are
 * never written.
 */
export class NeighborList {
  readonly particles: ParticleSystem;
  readonly range: ParticleRange;
  /** Neighbor particle indices, `MAX_NEIGHBORS` rows of `capacity`. */
  readonly indices: StorageBufferNode<'uint'>;
  /** Neighbor count per row. */
  readonly counts: StorageBufferNode<'uint'>;
  /** Atomic flag set when some particle had more than {@link MAX_NEIGHBORS} neighbors. */
  readonly overflowFlag: StorageBufferNode<'uint'>;
  private builtFor: HashGrid | undefined;

  constructor(particles: ParticleSystem, range: ParticleRange) {
    assertRange(particles, range, 'NeighborList');
    this.particles = particles;
    this.range = range;
    this.indices = instancedArray(particles.capacity * MAX_NEIGHBORS, 'uint');
    this.counts = instancedArray(particles.capacity, 'uint');
    this.overflowFlag = instancedArray(1, 'uint').toAtomic();
  }

  /** The grid the list is built from, whose sorted order its rows follow. Set by {@link buildKernels}. */
  get grid(): HashGrid {
    if (!this.builtFor) throw new Error('NeighborList: call buildKernels first');
    return this.builtFor;
  }

  /** Threads to dispatch for a kernel that uses {@link emitThread}: one per grid slot. */
  get threadCount(): number {
    return this.particles.capacity;
  }

  /**
   * Kernels that rebuild the list from `grid`, keeping neighbors closer than
   * `sqrt(radiusSq)`: one clears the overflow flag, the next fills the list.
   */
  buildKernels(grid: HashGrid, radiusSq: UniformNode<'float', number>): ComputeNode[] {
    this.builtFor = grid;
    // A separate dispatch, so no thread can clear a flag another has already set.
    const resetOverflow = Fn(() => {
      atomicStore(this.overflowFlag.element(uint(0)), uint(0));
    })()
      .compute(1)
      .setName('neighborList.resetOverflow');
    const build = Fn(() => {
      const { row } = this.emitThread();
      const xi: Any = grid.sortedPredictedPositions.element(row).xyz.toVar();
      const count: Any = uint(0).toVar();
      emitForEachNeighbor(grid, xi, (j: Any, candidate: Any) => {
        // The grid's sorted copy of the positions keeps these reads coalesced.
        const offset: Any = xi.sub(grid.sortedPredictedPositions.element(candidate).xyz);
        If(offset.dot(offset).lessThan(radiusSq), () => {
          If(count.lessThan(uint(MAX_NEIGHBORS)), () => {
            this.indices.element(count.mul(uint(this.threadCount)).add(row)).assign(j);
            count.addAssign(uint(1));
          }).Else(() => {
            atomicStore(this.overflowFlag.element(uint(0)), uint(1));
          });
        });
      });
      this.counts.element(row).assign(count);
    })()
      .compute(this.threadCount)
      .setName('neighborList.build');
    return [resetOverflow, build];
  }

  /**
   * Emit the start of a kernel dispatched with {@link threadCount} threads:
   * the thread's grid slot (its row in the list) and the particle in it.
   * Threads whose particle is outside the range return. Valid after the
   * list's grid has been rebuilt in the same substep.
   */
  emitThread(): { readonly i: Any; readonly row: Any } {
    const grid = this.grid;
    const row: Any = instanceIndex;
    const i: Any = grid.sortedIndices.element(row).toVar();
    const { start, count } = this.range;
    If(i.lessThan(uint(start)).or(i.greaterThanEqual(uint(start + count))), () => {
      Return();
    });
    return { i, row };
  }

  /** Emit TSL that calls `onNeighbor(j)` for every stored neighbor in `row`. */
  forEach(row: Any, onNeighbor: (j: Any) => void): void {
    Loop(
      { start: uint(0), end: this.counts.element(row), type: 'uint', condition: '<' },
      ({ i: k }: { i: Any }) => {
        onNeighbor(this.indices.element(k.mul(uint(this.threadCount)).add(row)));
      },
    );
  }

  /** True if some particle's list was truncated during the last rebuild. Stalls on the GPU. */
  async readbackOverflow(): Promise<boolean> {
    const buffer = await this.particles.renderer.getArrayBufferAsync(this.overflowFlag.value);
    return new Uint32Array(buffer)[0] !== 0;
  }
}
