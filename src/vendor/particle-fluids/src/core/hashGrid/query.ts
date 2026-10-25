import { If, Loop, int, uint } from 'three/tsl';

import type { HashGrid } from './HashGrid.js';
import { MORTON_BIAS, part1by2 } from './mortonHash.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Emit TSL that visits every particle in the 27 grid cells around `position`
 * and calls `onCandidate` with each particle's index and its slot in the
 * grid's sorted order. Candidates can be up to two cells away, so callers
 * must still filter by actual distance.
 *
 * Cells are bucketed with a 3D Morton code, so neighboring cells land in
 * nearby buckets and their lookups stay cache friendly. Two cells can still
 * share a bucket; buckets already visited for this query are skipped so no
 * particle is reported twice.
 *
 * `onCandidate` runs while the shader is being built and must emit TSL.
 * Use `Continue()` inside it to skip a candidate; `Return()` would end the
 * whole kernel invocation.
 */
export function emitForEachNeighbor(
  grid: HashGrid,
  position: Any,
  onCandidate: (neighborIndex: Any, sortedSlot: Any) => void,
): void {
  const bucketMask = grid.hashTableSize - 1;
  const cell: Any = position.sub(grid.hashOriginUniform).div(grid.cellSizeUniform).floor();
  // Spread each axis's three cell coordinates once; a cell's Morton code is
  // then three ORs (see `mortonBucketUnmasked`).
  const spread = (c: Any, shift: number): Any[] =>
    [-1, 0, 1].map((d) =>
      part1by2(c.add(int(d + MORTON_BIAS)).toUint())
        .shiftLeft(uint(shift))
        .toVar(),
    );
  const sx = spread(cell.x.toInt().toVar(), 0);
  const sy = spread(cell.y.toInt().toVar(), 1);
  const sz = spread(cell.z.toInt().toVar(), 2);

  const visited: Any[] = [];
  // `let` in each header gives every callback its own (dx, dy, dz) binding.
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const code: Any = sx[dx + 1].bitOr(sy[dy + 1]).bitOr(sz[dz + 1]);
        const bucket: Any = code.bitAnd(uint(bucketMask)).toVar();
        const walk = (): void => {
          Loop(
            {
              start: grid.cellStart.element(bucket),
              end: grid.cellEnd.element(bucket),
              type: 'uint',
              condition: '<',
            },
            ({ i }: { i: Any }) => onCandidate(grid.sortedIndices.element(i), i),
          );
        };
        if (visited.length === 0) walk();
        else
          If(
            visited
              .map((b) => bucket.equal(b))
              .reduce((a, b) => a.or(b))
              .not(),
            walk,
          );
        visited.push(bucket);
      }
    }
  }
}
