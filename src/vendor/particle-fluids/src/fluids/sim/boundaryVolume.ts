import { Continue, Fn, If, float, instanceIndex, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import {
  emitForEachNeighbor,
  emitPoly6FromRSq,
  type HashGrid,
  type ParticleRange,
  type ParticleSystem,
  type SphKernelUniforms,
} from '../../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Boundary volume `V_i = 1 / Σ_k W(x_i − x_k)` for each particle in `ranges`
 * (Akinci et al. 2012, eq. 4), summing only over the particle's own range.
 * Densely packed boundary particles get smaller volumes, so an unevenly
 * sampled solid still pushes on the fluid evenly.
 *
 * One dispatch covers every range: thread `t` handles the `t`-th boundary
 * particle counting through the ranges in order.
 *
 * @param positions Which positions to use: the committed ones for a one-time
 *   setup pass, or the predicted ones when the boundary moves every substep.
 */
export function buildBoundaryVolumeKernel(args: {
  readonly particles: ParticleSystem;
  readonly grid: HashGrid;
  readonly sph: SphKernelUniforms;
  readonly ranges: readonly ParticleRange[];
  readonly positions: StorageBufferNode<'vec4'>;
}): ComputeNode {
  const { particles, grid, sph, ranges, positions } = args;
  const total = ranges.reduce((sum, range) => sum + range.count, 0);

  return Fn(() => {
    const t: Any = instanceIndex;
    // Find the thread's range: the last one whose first thread is ≤ t.
    const start: Any = uint(ranges[0]!.start).toVar();
    const end: Any = uint(ranges[0]!.start + ranges[0]!.count).toVar();
    const first: Any = uint(0).toVar();
    let offset = 0;
    for (const range of ranges) {
      const from = offset;
      If(t.greaterThanEqual(uint(from)), () => {
        start.assign(uint(range.start));
        end.assign(uint(range.start + range.count));
        first.assign(uint(from));
      });
      offset += range.count;
    }
    const i: Any = start.add(t.sub(first)).toVar();
    const xi: Any = positions.element(i).xyz.toVar();
    const sum: Any = float(0).toVar();
    emitForEachNeighbor(grid, xi, (j: Any) => {
      If(j.lessThan(start).or(j.greaterThanEqual(end)), () => {
        Continue();
      });
      const offset: Any = xi.sub(positions.element(j).xyz);
      const rSq: Any = offset.dot(offset);
      If(rSq.greaterThanEqual(sph.hSq), () => {
        Continue();
      });
      sum.addAssign(emitPoly6FromRSq(rSq, sph));
    });
    particles.boundaryVolume.element(i).assign(float(1).div(sum.max(1e-12)));
  })()
    .compute(total)
    .setName('boundaryVolume.boundaryVolume');
}
