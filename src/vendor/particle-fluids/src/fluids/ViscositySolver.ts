import {
  Fn,
  If,
  Loop,
  Return,
  float,
  instanceIndex,
  instancedArray,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import {
  MAX_NEIGHBORS,
  emitPoly6FromRSq,
  type Material,
  type MaterialKernels,
  type ParticleSystem,
} from '../core/index.js';
import type { FluidSystem } from './FluidSystem.js';
import { emitFluidIndex } from './sim/shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface ViscositySolverOptions {
  /**
   * Kinematic viscosity ν, m²/s, as the discrete solve sees it: the
   * Laplacian is approximated as `10 / smoothingRadius²` times the
   * difference from the neighbors' weighted mean velocity. Tune it by eye;
   * honey is around 20. Not the same scale as {@link FluidSystem}'s XSPH
   * `viscosity`, which is a unitless blend factor. Must be ≥ 0.
   */
  readonly viscosity: number;
  /** Jacobi sweeps per substep, 1–64. Default 12. Finer particles need more. */
  readonly iterations?: number;
}

/**
 * Implicit viscosity for thick liquids such as honey. Each substep solves
 * `(I − dt ν L) v = v₀` for the fluid's velocities with Jacobi sweeps, which
 * stays stable at viscosities where {@link FluidSystem}'s explicit
 * `viscosity` would blow up.
 *
 * List it after its fluid in the {@link SimLoop}'s `materials`.
 */
export class ViscositySolver implements Material {
  readonly fluid: FluidSystem;

  /** The fluid's particle storage. */
  get particles(): ParticleSystem {
    return this.fluid.particles;
  }
  private readonly viscosityUniform: UniformNode<'float', number>;
  private readonly iterations: number;

  constructor(fluid: FluidSystem, options: ViscositySolverOptions) {
    const iterations = options.iterations ?? 12;
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 64) {
      throw new Error(
        `ViscositySolver: iterations must be an integer from 1 to 64, got ${iterations}`,
      );
    }
    this.fluid = fluid;
    this.viscosityUniform = uniform(nonNegative(options.viscosity), 'float');
    this.iterations = iterations;
  }

  /** Kinematic viscosity ν; see {@link ViscositySolverOptions.viscosity}. */
  get viscosity(): number {
    return this.viscosityUniform.value;
  }
  set viscosity(value: number) {
    this.viscosityUniform.value = nonNegative(value);
  }

  build(): MaterialKernels {
    let context;
    try {
      context = this.fluid.kernelContext;
    } catch {
      throw new Error('ViscositySolver: list it after its FluidSystem in `materials`');
    }
    const { particles, range, neighbors, sph, particleVolume, dt } = context;
    const { grid } = neighbors;
    const h = this.fluid.smoothingRadius;
    // Everything here is indexed by grid slot (the neighbor list's rows), so
    // the sweeps read and write contiguously. `rhs.w` holds a row's weight
    // sum, or −1 for rows whose particle is not in the fluid. The iterates
    // have one extra row, never written, that stays zero.
    const rows = neighbors.threadCount;
    const zeroRow = rows;
    const rhs = instancedArray(rows, 'vec4');
    const ping = instancedArray(rows + 1, 'vec4');
    const pong = instancedArray(rows + 1, 'vec4');
    // Per stored neighbor: its weight and its row, laid out like the list.
    const weights = instancedArray(rows * MAX_NEIGHBORS, 'float');
    const neighborRows = instancedArray(rows * MAX_NEIGHBORS, 'uint');
    const { start, count } = range;

    const inFluid = (index: Any): Any =>
      index.greaterThanEqual(uint(start)).and(index.lessThan(uint(start + count)));
    // Neighbors outside the fluid (boundaries) enter the sums at zero
    // velocity, through the zero row.
    const prepare = Fn(() => {
      const row: Any = instanceIndex;
      const i: Any = grid.sortedIndices.element(row).toVar();
      If(inFluid(i).not(), () => {
        rhs.element(row).assign(vec4(0, 0, 0, -1));
        Return();
      });
      const position: Any = particles.positions.element(i).xyz.toVar();
      const weight: Any = float(0).toVar();
      Loop(
        { start: uint(0), end: neighbors.counts.element(row), type: 'uint', condition: '<' },
        ({ i: k }: { i: Any }) => {
          const entry: Any = k.mul(uint(rows)).add(row).toVar();
          const j: Any = neighbors.indices.element(entry).toVar();
          const offset: Any = position.sub(particles.positions.element(j).xyz);
          const w: Any = emitPoly6FromRSq(offset.dot(offset), sph).mul(particleVolume).toVar();
          weights.element(entry).assign(w);
          neighborRows
            .element(entry)
            .assign(inFluid(j).select(grid.slotOf.element(j), uint(zeroRow)));
          weight.addAssign(w);
        },
      );
      const v: Any = particles.velocities.element(i).xyz;
      rhs.element(row).assign(vec4(v, weight));
      ping.element(row).assign(vec4(v, 0));
    })()
      .compute(rows)
      .setName('ViscositySolver.prepare');

    // Each sweep reads one iterate and writes the other, so a sweep is one dispatch.
    const rate: Any = this.viscosityUniform.mul(dt).mul(10 / (h * h));
    const sweep = (source: Any, target: Any): ComputeNode =>
      Fn(() => {
        const row: Any = instanceIndex;
        const b: Any = rhs.element(row).toVar();
        If(b.w.lessThan(0), () => {
          Return();
        });
        const sum: Any = vec3(0).toVar();
        Loop(
          { start: uint(0), end: neighbors.counts.element(row), type: 'uint', condition: '<' },
          ({ i: k }: { i: Any }) => {
            const entry: Any = k.mul(uint(rows)).add(row);
            sum.addAssign(
              source.element(neighborRows.element(entry)).xyz.mul(weights.element(entry)),
            );
          },
        );
        target.element(row).assign(vec4(b.xyz.add(sum.mul(rate)).div(b.w.mul(rate).add(1)), 0));
      })()
        .compute(rows)
        .setName('ViscositySolver.sweep');
    const forward = sweep(ping, pong);
    const backward = sweep(pong, ping);
    const result = this.iterations % 2 === 1 ? pong : ping;

    const apply = Fn(() => {
      const i: Any = emitFluidIndex(context);
      const velocity: Any = particles.velocities.element(i);
      velocity.assign(vec4(result.element(grid.slotOf.element(i)).xyz, velocity.w));
    })()
      .compute(range.count)
      .setName('ViscositySolver.apply');

    return {
      postSolve: [
        prepare,
        ...Array.from({ length: this.iterations }, (_, k) => (k % 2 === 0 ? forward : backward)),
        apply,
      ],
    };
  }
}

function nonNegative(viscosity: number): number {
  if (!(viscosity >= 0) || !Number.isFinite(viscosity)) {
    throw new Error(`ViscositySolver: viscosity must be ≥ 0, got ${viscosity}`);
  }
  return viscosity;
}
