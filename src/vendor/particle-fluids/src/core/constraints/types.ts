import { Fn, instanceIndex, instancedArray } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * One color class of a constraint set: constraints that share no particle,
 * so one thread per constraint can solve them all at once and write every
 * particle it touches without racing another thread.
 */
export interface ConstraintGroup {
  /** Indices of the group's constraints. */
  readonly constraints: StorageBufferNode<'uint'>;
  readonly count: number;
  /** Solves every constraint in the group, one thread each. */
  readonly solveKernel: ComputeNode;
}

/**
 * A set of XPBD constraints of one kind (distance, bending, tether, …),
 * ready to schedule with {@link constraintKernels}.
 */
export interface ConstraintType {
  readonly count: number;
  /** Compliance `α` per constraint (inverse stiffness; units depend on the kind). */
  readonly compliance: StorageBufferNode<'float'>;
  /**
   * Accumulated Lagrange multiplier per constraint (Macklin et al. 2016,
   * eq. 13), zeroed each substep by `resetLambdaKernel`.
   */
  readonly lambda: StorageBufferNode<'float'>;
  /** Color groups, solved in order every iteration. */
  readonly groups: readonly ConstraintGroup[];
  readonly resetLambdaKernel: ComputeNode;
}

/**
 * Greedy graph coloring — partitions a set of constraints into groups such
 * that no two constraints in the same group share a particle.
 *
 * Input: flat `participantsPerConstraint[c·arity + k]` = particle index for
 * participant `k` of constraint `c`, plus `arity` and `nConstraints`.
 *
 * Output: `groupOf[c]` = color index assigned to constraint `c` (0-indexed),
 * plus `numGroups` = one past the largest color used.
 *
 * The greedy algorithm is O(nConstraints · arity · maxDegree) worst-case; for
 * structured topologies (cloth lattice, chains) it produces an optimal or
 * near-optimal coloring.
 *
 * Bounds: `numGroups ≥ maxDegree`, the largest number of constraints any
 * single particle participates in, and `numGroups ≤ 1 + arity · (maxDegree − 1)`,
 * since a constraint conflicts with at most that many others. Greedy
 * coloring can exceed `maxDegree + 1`. Tested by
 * `tests/analytical/xpbd/xpbd-group-partition.test.ts`.
 */
export function colorConstraints(args: {
  readonly arity: number;
  readonly nConstraints: number;
  readonly participantsPerConstraint: readonly number[] | Uint32Array;
}): { groupOf: Uint32Array; numGroups: number } {
  const { arity, nConstraints, participantsPerConstraint } = args;
  if (!Number.isInteger(arity) || arity < 1) {
    throw new Error(`colorConstraints: arity must be a positive integer, got ${arity}`);
  }
  if (!Number.isInteger(nConstraints) || nConstraints < 0) {
    throw new Error(`colorConstraints: nConstraints must be an integer ≥ 0, got ${nConstraints}`);
  }
  if (participantsPerConstraint.length !== arity * nConstraints) {
    throw new Error(
      `colorConstraints: participantsPerConstraint length ${participantsPerConstraint.length} ≠ arity × nConstraints (${arity * nConstraints})`,
    );
  }
  if (nConstraints === 0) return { groupOf: new Uint32Array(0), numGroups: 0 };

  const groupOf = new Uint32Array(nConstraints);
  // Colors already used by each particle.
  const particleGroups: Map<number, Set<number>> = new Map();

  let numGroups = 0;
  for (let c = 0; c < nConstraints; c++) {
    let g = 0;
    // Find the lowest-indexed group that conflicts with none of this
    // constraint's participants.
    while (true) {
      let conflict = false;
      for (let k = 0; k < arity; k++) {
        const p = participantsPerConstraint[c * arity + k]!;
        const owned = particleGroups.get(p);
        if (owned !== undefined && owned.has(g)) {
          conflict = true;
          break;
        }
      }
      if (!conflict) break;
      g++;
    }
    groupOf[c] = g;
    if (g + 1 > numGroups) numGroups = g + 1;
    for (let k = 0; k < arity; k++) {
      const p = participantsPerConstraint[c * arity + k]!;
      let owned = particleGroups.get(p);
      if (owned === undefined) {
        owned = new Set();
        particleGroups.set(p, owned);
      }
      owned.add(g);
    }
  }

  return { groupOf, numGroups };
}

/**
 * Schedule constraint types as material kernels: every multiplier is reset
 * once per substep, then each type's color groups are solved in order every
 * iteration (Macklin et al. 2016, Algorithm 1).
 */
export function constraintKernels(types: readonly ConstraintType[]): {
  readonly preSolve: ComputeNode[];
  readonly solve: ComputeNode[];
} {
  return {
    preSolve: types.map((type) => type.resetLambdaKernel),
    solve: types.flatMap((type) => type.groups.map((group) => group.solveKernel)),
  };
}

/**
 * Build one solve kernel per color of `coloring`. `solve(constraint)` emits
 * TSL that projects one constraint and writes all of its particles; it may
 * `Return()` early. `name` labels the kernels in GPU profiles.
 */
export function buildConstraintGroups(
  coloring: { readonly groupOf: Uint32Array; readonly numGroups: number },
  solve: (constraint: Any) => void,
  name = 'constraints',
): ConstraintGroup[] {
  const members: number[][] = Array.from({ length: coloring.numGroups }, () => []);
  coloring.groupOf.forEach((group, constraint) => members[group]!.push(constraint));
  return members.map((list) => {
    const constraints = instancedArray(Uint32Array.from(list), 'uint');
    const solveKernel = Fn(() => {
      solve(constraints.element(instanceIndex).toVar());
    })()
      .compute(list.length)
      .setName(`${name}.solve`);
    return { constraints, count: list.length, solveKernel };
  });
}
