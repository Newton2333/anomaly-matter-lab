import { Vector3 } from 'three';
import { instancedArray, uniform } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { Accumulator } from './accumulator.js';
import type { Collider } from './collision/collider.js';
import { resolveFriction } from './collision/PrimitiveSet.js';
import {
  CONTACT_RADIUS_EXPANSION,
  ContactBuffer,
  buildContactFrictionKernel,
  buildContactGenerateKernel,
  buildContactSolveKernel,
  buildContactStabilizeKernel,
  type ContactEmitters,
} from './contact/index.js';
import { HashGrid } from './hashGrid/HashGrid.js';
import { buildIntegrationKernels } from './integrate.js';
import type { Material, MaterialKernels, SolverContext } from './materials.js';
import { releaseStorageBuffers, type ParticleRange, type ParticleSystem } from './particles.js';

/** Particle–particle contact settings. */
export interface ContactOptions {
  /** Static friction coefficient between particles. Default 0.5. */
  readonly muS?: number;
  /** Kinetic friction coefficient between particles. Default 0.4. */
  readonly muK?: number;
  /** Most contact pairs kept per substep; extras are dropped. Default `8 × capacity`. */
  readonly maxContacts?: number;
}

export interface SimLoopOptions {
  /** Substeps per {@link SimLoop.step}. Default 4. More substeps are stiffer and more stable. */
  readonly substeps?: number;
  /** Constraint iterations per substep. Default 2. */
  readonly iterations?: number;
  /** Gravity in m/s². Default `(0, -9.81, 0)`. Also settable through {@link SimLoop.gravity}. */
  readonly gravity?: Vector3;
  /** Physics to run, in order. */
  readonly materials?: readonly Material[];
  /** Shapes particles collide with. */
  readonly colliders?: readonly Collider[];
  /**
   * Collide particles with each other. Needed for soft bodies and cloth to
   * touch one another; fluids keep their own particles apart without it.
   * Default off.
   */
  readonly contact?: boolean | ContactOptions;
  /**
   * Origin of the neighbor grid's cells, in metres. Neighbor queries slow
   * down for particles more than 512 cells from it on any axis, so set it
   * near the middle of scenes far from the world origin. Copied; move it
   * later through `hashGrid.hashOrigin`. Default `(0, 0, 0)`.
   */
  readonly hashOrigin?: Vector3;
}

/** Fixed-point and capacity overflows seen during the last {@link SimLoop.step}. */
export interface SimLoopOverflow {
  /** A collider or contact position correction saturated its fixed-point sum (> 10 m per axis). */
  readonly positions: boolean;
  /** A collider or contact friction correction saturated its fixed-point sum (> 50 m/s per axis). */
  readonly velocities: boolean;
  /** At the last grid rebuild, a particle was more than 512 cells from the grid's origin. */
  readonly grid: boolean;
  /** The last substep found more than `maxContacts` contact pairs and dropped the rest. */
  readonly contacts: boolean;
}

/**
 * Advances a {@link ParticleSystem} and every material and collider attached
 * to it, using extended position-based dynamics (XPBD) with substepping.
 *
 * Each substep:
 * 1. predict positions from velocity and gravity;
 * 2. rebuild the neighbor grid, find contacts, run materials' pre-solve kernels;
 * 3. iterate: material constraints, then contacts and colliders;
 * 4. derive velocities from the solved positions;
 * 5. run materials' post-solve kernels, then friction.
 *
 * The whole frame is submitted to the GPU in one batch.
 *
 * Create the loop after uploading particles: the collision groups it hands
 * to materials start above the highest group uploaded by then.
 *
 * ```ts
 * const loop = new SimLoop(particles, { materials: [fluid], colliders: [walls] });
 * await loop.step(1 / 60);
 * ```
 */
export class SimLoop {
  readonly particles: ParticleSystem;
  readonly iterations: number;
  /** Neighbor grid, present when a material needs neighbors or contacts are on. */
  readonly hashGrid: HashGrid | undefined;
  /** Contact storage, present when contacts are on. */
  readonly contacts: ContactBuffer | undefined = undefined;
  /** Substep length in seconds, shared with every kernel. */
  readonly dt: UniformNode<'float', number> = uniform(1 / 240, 'float');

  private readonly gravityUniform: UniformNode<'vec3', Vector3>;
  private readonly colliders: readonly Collider[];
  private readonly materials: readonly Material[];
  private readonly accumulators: { positions: Accumulator; velocities: Accumulator } | undefined;
  /** GPU buffers the loop created itself, freed by {@link dispose}. */
  private readonly ownBuffers: { readonly value: object }[] = [];
  private disposed = false;
  private readonly substepsUniform = uniform(4, 'float');
  private substepCount: number;
  private readonly frameStart: ComputeNode[] = [];
  private readonly substep: ComputeNode[] = [];
  private readonly init: ComputeNode[] = [];
  private pipeline: ComputeNode[] = [];
  private initialized = false;

  constructor(particles: ParticleSystem, options: SimLoopOptions = {}) {
    const substeps = options.substeps ?? 4;
    const iterations = options.iterations ?? 2;
    assertCount(substeps, 1, 'substeps');
    assertCount(iterations, 0, 'iterations');
    if (particles.disposed) throw new Error('SimLoop: the ParticleSystem has been disposed');
    const materials = options.materials ?? [];
    const colliders = options.colliders ?? [];
    for (const collider of colliders) {
      if (collider.particles !== particles) {
        throw new Error('SimLoop: every collider must be built for the same ParticleSystem');
      }
    }
    for (const material of materials) {
      if (material.particles !== undefined && material.particles !== particles) {
        throw new Error('SimLoop: every material must be built for the same ParticleSystem');
      }
      const radius = material.neighborRadius;
      if (radius !== undefined && !(radius >= 0 && Number.isFinite(radius))) {
        throw new Error(
          `SimLoop: a material's neighborRadius must be finite and ≥ 0, got ${radius}`,
        );
      }
    }

    this.particles = particles;
    this.iterations = iterations;
    this.colliders = colliders;
    this.materials = materials;
    this.substepCount = substeps;
    this.gravityUniform = uniform((options.gravity ?? new Vector3(0, -9.81, 0)).clone());
    const contact =
      options.contact === true ? {} : options.contact === false ? undefined : options.contact;

    // One grid serves every neighbor query, so size its cells for the widest one.
    const neighborRadius = Math.max(
      contact ? 2 * particles.particleRadius * CONTACT_RADIUS_EXPANSION : 0,
      ...materials.map((material) => material.neighborRadius ?? 0),
    );
    this.hashGrid =
      neighborRadius > 0
        ? new HashGrid(particles, {
            cellSize: neighborRadius,
            ...(options.hashOrigin && { hashOrigin: options.hashOrigin }),
          })
        : undefined;

    // Hand out collision groups above any the particles were uploaded with.
    // Groups written to the particles after this can clash with them.
    let nextCollisionGroup =
      (particles.collisionGroup.value.array as Uint32Array).reduce((a, b) => Math.max(a, b), 0) + 1;
    const grid = this.hashGrid;
    const context: SolverContext = {
      particles,
      dt: this.dt,
      hashGrid: grid ?? missingHashGrid(),
      allocateCollisionGroup: () => nextCollisionGroup++,
    };
    const built: MaterialKernels[] = materials.map((material) => material.build(context));

    const { predict, advect } = buildIntegrationKernels(particles, this.dt, this.gravityUniform);
    const solve = built.flatMap((kernels) => kernels.solve ?? []);
    const postSolve = built.flatMap((kernels) => kernels.postSolve ?? []);
    const preSolve: ComputeNode[] = [];
    if (grid) preSolve.push(...grid.rebuildPipeline);

    // Contacts and colliders scatter corrections into shared accumulators:
    // positions are applied after every iteration, velocities after friction.
    // Applying a sum clears it, so the sums start every substep at zero.
    let scatter: ComputeNode[] = [];
    let friction: ComputeNode[] = [];
    const substepEnd: ComputeNode[] = [];
    let applyPositions: ComputeNode[] = [];
    if (contact || colliders.length > 0) {
      const positions = new Accumulator(particles, 10, 'positionCorrections');
      const velocities = new Accumulator(particles, 50, 'velocityCorrections');
      this.accumulators = { positions, velocities };
      // Overflow flags are cleared once per step, so they cover every substep.
      this.frameStart.push(
        positions.buildResetOverflowKernel(),
        velocities.buildResetOverflowKernel(),
      );
      if (contact && grid) {
        const { buffer, kernels } = buildContacts(particles, contact, grid, this.dt, {
          positions,
          velocities,
          quiet: built.flatMap((k) => (k.noSelfContacts ? [k.noSelfContacts] : [])),
          ownBuffers: this.ownBuffers,
        });
        this.contacts = buffer;
        preSolve.push(...kernels.preSolve);
        scatter = [...kernels.solve];
        friction = [...kernels.postSolve];
      }
      for (const collider of colliders) {
        const kernels = collider.buildKernels({
          particles,
          dt: this.dt,
          substeps: this.substepsUniform,
          positions,
          velocities,
        });
        this.frameStart.push(...(kernels.frameStart ?? []));
        preSolve.push(...kernels.preSolve);
        scatter.push(...kernels.solve);
        friction.push(...kernels.postSolve);
        substepEnd.push(...(kernels.substepEnd ?? []));
      }
      applyPositions = [positions.buildApplyKernel([particles.predictedPositions])];
      friction = [...friction, velocities.buildApplyKernel([particles.velocities])];
    }
    for (const kernels of built) {
      preSolve.push(...(kernels.preSolve ?? []));
      this.init.push(...(kernels.init ?? []));
      this.frameStart.push(...(kernels.beforeStep ?? []));
    }
    if (grid && this.init.length > 0) this.init.unshift(...grid.rebuildPipeline);

    this.substep.push(predict, ...preSolve);
    for (let i = 0; i < iterations; i++) this.substep.push(...solve, ...scatter, ...applyPositions);
    this.substep.push(advect, ...postSolve, ...friction, ...substepEnd);
    this.rebuildPipeline();
  }

  /** Substeps per {@link step}. Changing it takes effect immediately without recompiling. */
  get substeps(): number {
    return this.substepCount;
  }

  set substeps(value: number) {
    this.assertAlive();
    assertCount(value, 1, 'substeps');
    this.substepCount = value;
    this.rebuildPipeline();
  }

  /** Gravity in m/s². Mutate it in place to change gravity live. */
  get gravity(): Vector3 {
    return this.gravityUniform.value;
  }

  /**
   * Advance the simulation by `dt` seconds: let materials and colliders
   * upload their changes (colliders follow any objects they're attached
   * to), then run every substep on the GPU.
   */
  async step(dt: number): Promise<void> {
    this.assertAlive();
    if (this.particles.disposed) throw new Error('SimLoop: the ParticleSystem has been disposed');
    if (!(dt > 0) || !Number.isFinite(dt)) {
      throw new Error(`SimLoop: step dt must be a positive finite number, got ${dt}`);
    }
    for (const material of this.materials) material.update?.(dt);
    for (const collider of this.colliders) collider.update(dt);
    this.dt.value = dt / this.substepCount;
    const pipeline = this.initialized ? this.pipeline : [...this.init, ...this.pipeline];
    this.initialized = true;
    await this.particles.renderer.computeAsync(pipeline);
  }

  /**
   * Report fixed-point and capacity overflows from the last {@link step}.
   * Any `true` means some corrections or contacts were lost that step.
   * Stalls on the GPU, so use it for debugging and tests.
   */
  async readbackOverflow(): Promise<SimLoopOverflow> {
    this.assertAlive();
    const [positions, velocities, grid, contacts] = await Promise.all([
      this.accumulators?.positions.readbackOverflow() ?? false,
      this.accumulators?.velocities.readbackOverflow() ?? false,
      this.hashGrid?.readbackOverflow() ?? false,
      this.contacts?.readbackCount().then((count) => count > this.contacts!.maxContacts) ?? false,
    ]);
    return { positions, velocities, grid, contacts };
  }

  /**
   * Free the GPU buffers the loop created: its neighbor grid, contact
   * storage, and correction sums. Later calls to `step`, `readbackOverflow`,
   * and the `substeps` setter throw. Materials, colliders, and the
   * `ParticleSystem` are disposed separately.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.hashGrid?.dispose();
    this.contacts?.dispose();
    this.accumulators?.positions.dispose();
    this.accumulators?.velocities.dispose();
    releaseStorageBuffers(this.particles.renderer, this.ownBuffers);
    this.pipeline = [];
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('SimLoop: the loop has been disposed');
  }

  private rebuildPipeline(): void {
    this.substepsUniform.value = this.substepCount;
    const pipeline = [...this.frameStart];
    for (let s = 0; s < this.substepCount; s++) pipeline.push(...this.substep);
    this.pipeline = pipeline;
  }
}

/**
 * Stand-in for {@link SolverContext.hashGrid} when the loop has no grid. It
 * throws only when used, so materials can destructure the context freely.
 */
function missingHashGrid(): HashGrid {
  return new Proxy({} as HashGrid, {
    get() {
      throw new Error(
        'SimLoop: a material used the neighbor grid without declaring a neighborRadius',
      );
    },
  });
}

function assertCount(value: number, min: number, name: string): void {
  if (!Number.isInteger(value) || value < min) {
    throw new Error(`SimLoop: ${name} must be an integer ≥ ${min}, got ${value}`);
  }
}

/** Build particle–particle contact storage and its kernels. */
function buildContacts(
  particles: ParticleSystem,
  options: ContactOptions,
  grid: HashGrid,
  dt: UniformNode<'float', number>,
  shared: {
    readonly positions: Accumulator;
    readonly velocities: Accumulator;
    /** Ranges whose particles never contact each other. */
    readonly quiet: readonly ParticleRange[];
    /** Receives the buffers created here, so the loop can free them. */
    readonly ownBuffers: { readonly value: object }[];
  },
): {
  buffer: ContactBuffer;
  kernels: { preSolve: ComputeNode[]; solve: ComputeNode[]; postSolve: ComputeNode[] };
} {
  const { positions, velocities } = shared;
  const { muS, muK } = resolveFriction(options, 'SimLoop');
  const contacts = new ContactBuffer(
    particles.renderer,
    options.maxContacts ?? 8 * particles.capacity,
  );
  const preSolve: ComputeNode[] = [];

  // Pairs are found from the particles outside the quiet ranges; pairs that
  // include a quiet particle are found from the other side.
  const isEmitter = new Uint32Array(particles.capacity).fill(1);
  for (const range of shared.quiet) isEmitter.fill(0, range.start, range.start + range.count);
  const emitterCount = isEmitter.reduce((sum, flag) => sum + flag, 0);
  if (emitterCount === particles.capacity) {
    preSolve.push(buildContactGenerateKernel({ particles, hashGrid: grid, contacts }));
  } else if (emitterCount > 0) {
    const emitters: ContactEmitters = { isEmitter: instancedArray(isEmitter, 'uint') };
    shared.ownBuffers.push(emitters.isEmitter);
    preSolve.push(buildContactGenerateKernel({ particles, hashGrid: grid, contacts, emitters }));
  }

  preSolve.push(
    contacts.prepareKernel,
    buildContactStabilizeKernel({ particles, contacts, accumulator: positions }),
    positions.buildApplyKernel([particles.positions, particles.predictedPositions]),
  );
  return {
    buffer: contacts,
    kernels: {
      preSolve,
      solve: [
        buildContactSolveKernel({
          particles,
          contacts,
          accumulator: positions,
          muS: uniform(muS, 'float'),
        }),
      ],
      postSolve: [
        buildContactFrictionKernel({
          particles,
          contacts,
          accumulator: velocities,
          muK: uniform(muK, 'float'),
          dt,
        }),
      ],
    },
  };
}
