import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { HashGrid } from './hashGrid/HashGrid.js';
import type { ParticleRange, ParticleSystem } from './particles.js';

/** Shared solver state that {@link SimLoop} passes to {@link Material.build}. */
export interface SolverContext {
  readonly particles: ParticleSystem;
  /** Substep length in seconds, updated by {@link SimLoop.step}. */
  readonly dt: UniformNode<'float', number>;
  /**
   * Neighbor grid over every particle, rebuilt at the start of each substep.
   * Only usable when a material declares a {@link Material.neighborRadius}
   * or particle contacts are enabled. Otherwise it is a placeholder that
   * throws when any of its properties is read (so destructuring it is safe).
   */
  readonly hashGrid: HashGrid;
  /**
   * Reserve a collision group that no other material, and no particle
   * uploaded before the loop was created, uses. Groups start above the
   * highest group in the CPU copy of `particles.collisionGroup` when the
   * loop is constructed; groups written to particles after that can clash.
   */
  allocateCollisionGroup(): number;
}

/** The kernels a material adds to the solver. Every list is optional. */
export interface MaterialKernels {
  /** Run once, before the first step, after an initial grid rebuild. */
  readonly init?: readonly ComputeNode[];
  /** Run at the start of every step, before the first substep. */
  readonly beforeStep?: readonly ComputeNode[];
  /** Run once per substep, after positions are predicted and the grid is rebuilt. */
  readonly preSolve?: readonly ComputeNode[];
  /** Run every solver iteration, before contacts and colliders. */
  readonly solve?: readonly ComputeNode[];
  /** Run once per substep, after velocities are updated from the solved positions. */
  readonly postSolve?: readonly ComputeNode[];
  /**
   * Particles in this range never get particle–particle contacts with each
   * other, because the material keeps them apart itself (a fluid's pressure
   * solve, for example). They still collide with every other particle.
   */
  readonly noSelfContacts?: ParticleRange;
}

/**
 * Physics that runs inside a {@link SimLoop}: fluids, soft bodies, cloth, or
 * your own kernels. The loop calls {@link build} once, in the order materials
 * are listed, and dispatches the returned kernels every substep.
 *
 * A minimal custom material:
 *
 * ```ts
 * const drag: Material = {
 *   build: ({ particles, dt }) => ({
 *     postSolve: [Fn(() => { ... })().compute(particles.capacity)],
 *   }),
 * };
 * ```
 */
export interface Material {
  /**
   * Farthest distance, in metres, this material looks for neighbors through
   * the grid. Finite and ≥ 0; `0` or absent means no neighbor queries.
   */
  readonly neighborRadius?: number;
  /** Particles the material was built for. When set, {@link SimLoop} checks it matches its own. */
  readonly particles?: ParticleSystem;
  build(context: SolverContext): MaterialKernels;
  /** Called by {@link SimLoop.step} before each step's GPU work, with the step length. */
  update?(dt: number): void;
}
