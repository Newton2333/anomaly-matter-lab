import { Fn, atomicLoad, atomicStore, instancedArray, storage, struct, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import { IndirectStorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';

import { releaseStorageBuffers } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Fixed-point scale for the accumulated Lagrange multipliers λ_n and λ_t, in
 * ticks per kg·m. 2^20 gives ~1e-6 kg·m resolution and ±2048 kg·m of range,
 * far beyond the per-substep impulses of particle scenes.
 */
export const LAMBDA_SCALE = 1 << 20;

/**
 * One particle–particle contact. `lambdaN`/`lambdaT` accumulate the normal and
 * tangential multipliers across solver iterations (Macklin et al. 2020 §3.5);
 * `normal` is the last normal the solve used, read by the friction pass.
 *
 * Layout: i (0), j (4), lambdaN (8), lambdaT (12), normal (16), stride 32 bytes.
 */
export const ContactRecord = struct(
  {
    i: 'uint',
    j: 'uint',
    lambdaN: { type: 'int', atomic: true },
    lambdaT: { type: 'int', atomic: true },
    normal: 'vec3',
  },
  'ContactRecord',
);
const RECORD_STRIDE_WORDS = 8;

/** Threads per workgroup of the per-contact kernels (three's default). */
const WORKGROUP_SIZE = 64;
/** WebGPU's guaranteed `maxComputeWorkgroupsPerDimension`. */
const MAX_WORKGROUPS_PER_DIMENSION = 65535;

/** Word offsets in {@link ContactBuffer.dispatch}. */
const DISPATCH_STORED = 3;
const DISPATCH_EMITTED = 4;

/**
 * Contact pairs found each substep, plus their accumulated multipliers.
 *
 * The per-contact kernels are dispatched indirectly, with one thread per
 * stored pair: after the pairs are generated, {@link prepareKernel} turns
 * the pair count into workgroup counts on the GPU.
 */
export class ContactBuffer {
  readonly renderer: WebGPURenderer;
  /** Pairs past this count are dropped. The counter still counts them. */
  readonly maxContacts: number;
  readonly records: StorageBufferNode<'struct'>;
  /** Atomic count of pairs emitted so far this substep. */
  readonly counter: StorageBufferNode<'uint'>;
  /**
   * Indirect dispatch arguments for the per-contact kernels: workgroup
   * counts (x, y, z), then the number of stored pairs and the number
   * emitted, including dropped ones.
   */
  readonly dispatchArgs: IndirectStorageBufferAttribute;
  /** {@link dispatchArgs} as a storage node, written by {@link prepareKernel}. */
  readonly dispatch: StorageBufferNode<'uint'>;
  /**
   * Read-only view of {@link dispatchArgs}. The per-contact kernels read it
   * while it drives their own dispatch, which WebGPU allows only read-only.
   */
  private readonly dispatchReadOnly: StorageBufferNode<'uint'>;
  /**
   * One thread, run after pair generation: writes {@link dispatchArgs} from
   * the counter and zeroes the counter for the next substep.
   */
  readonly prepareKernel: ComputeNode;

  constructor(renderer: WebGPURenderer, maxContacts: number) {
    if (!Number.isInteger(maxContacts) || maxContacts <= 0) {
      throw new Error(`ContactBuffer: maxContacts must be a positive integer, got ${maxContacts}`);
    }
    this.renderer = renderer;
    this.maxContacts = maxContacts;
    // three r184 declares a one-element struct buffer as a bare struct, which
    // can't be indexed, so always allocate at least two records.
    this.records = instancedArray(Math.max(maxContacts, 2), ContactRecord as Any) as Any;
    this.counter = instancedArray(1, 'uint').toAtomic();
    this.dispatchArgs = new IndirectStorageBufferAttribute(new Uint32Array(5), 1);
    this.dispatch = storage(this.dispatchArgs, 'uint', 5) as Any;
    this.dispatchReadOnly = (storage(this.dispatchArgs, 'uint', 5) as Any).toReadOnly();

    const { counter, dispatch } = this;
    this.prepareKernel = Fn(() => {
      const emitted: Any = (atomicLoad(counter.element(uint(0))) as Any).toVar();
      const stored: Any = emitted.min(uint(maxContacts)).toVar();
      const groups: Any = stored
        .add(uint(WORKGROUP_SIZE - 1))
        .div(uint(WORKGROUP_SIZE))
        .toVar();
      dispatch.element(uint(0)).assign(groups.min(uint(MAX_WORKGROUPS_PER_DIMENSION)));
      dispatch
        .element(uint(1))
        .assign(
          groups
            .add(uint(MAX_WORKGROUPS_PER_DIMENSION - 1))
            .div(uint(MAX_WORKGROUPS_PER_DIMENSION)),
        );
      dispatch.element(uint(2)).assign(uint(1));
      dispatch.element(uint(DISPATCH_STORED)).assign(stored);
      dispatch.element(uint(DISPATCH_EMITTED)).assign(emitted);
      atomicStore(counter.element(uint(0)), uint(0));
    })()
      .compute(1)
      .setName('ContactBuffer.prepare');
  }

  /** Emit the number of pairs stored this substep. Valid after {@link prepareKernel}. */
  emitStoredCount(): Any {
    return this.dispatchReadOnly.element(uint(DISPATCH_STORED));
  }

  /** Number of pairs emitted in the last substep, including dropped ones. */
  async readbackCount(): Promise<number> {
    const buffer = await this.renderer.getArrayBufferAsync(this.dispatchArgs);
    return new Uint32Array(buffer)[DISPATCH_EMITTED]!;
  }

  /** Free the contact storage on the GPU. Kernels built on it can't run afterwards. */
  dispose(): void {
    releaseStorageBuffers(this.renderer, [this.records, this.counter, this.dispatch]);
  }

  /** The stored `(i, j)` pairs, flattened. For tests and debugging. */
  async readbackPairs(): Promise<{ readonly nContacts: number; readonly pairs: Uint32Array }> {
    const [dispatchBuffer, recordsBuffer] = await Promise.all([
      this.renderer.getArrayBufferAsync(this.dispatchArgs),
      this.renderer.getArrayBufferAsync(this.records.value),
    ]);
    const nContacts = new Uint32Array(dispatchBuffer)[DISPATCH_STORED]!;
    const words = new Uint32Array(recordsBuffer);
    const pairs = new Uint32Array(2 * nContacts);
    for (let c = 0; c < nContacts; c++) {
      pairs[2 * c] = words[c * RECORD_STRIDE_WORDS]!;
      pairs[2 * c + 1] = words[c * RECORD_STRIDE_WORDS + 1]!;
    }
    return { nContacts, pairs };
  }
}
