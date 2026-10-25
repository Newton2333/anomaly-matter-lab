import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

/**
 * Tracer buffers the smoke renderers draw. {@link GasSystem} provides them,
 * and so can your own emitter, as long as it fills the same buffers.
 */
export interface SmokeTracers {
  readonly capacity: number;
  /** Seconds a tracer lives. Renderers fade tracers out with age. */
  readonly lifetime: number;
  readonly smokePositions: StorageBufferNode<'vec4'>;
  /** Seconds since each tracer was released. */
  readonly smokeAge: StorageBufferNode<'float'>;
  /** 1 for live tracers, 0 for free slots. */
  readonly smokeAlive: StorageBufferNode<'uint'>;
  /**
   * Optional velocity per tracer. {@link GasSpriteRenderer} passes it to its
   * `colorNode`, or zero when absent.
   */
  readonly smokeVelocities?: StorageBufferNode<'vec4'>;
}
