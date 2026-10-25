import type { Box3, Vector3 } from 'three';
import { Fn, If, instanceIndex, instancedArray, uint, uniform, vec4 } from 'three/tsl';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import {
  createSphKernelUniforms,
  type Material,
  type MaterialKernels,
  type ParticleSystem,
  type SolverContext,
} from '../core/index.js';
import { releaseStorageBuffers } from '../core/particles.js';
import type { FluidSystem } from '../fluids/index.js';
import { AirHeat, type HeatSource } from './heat.js';
import { buildSmokeAdvectKernel } from './smokeAdvect.js';
import type { SmokeTracers } from './render/types.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface GasSystemOptions {
  /** Most tracers alive at once. */
  readonly capacity: number;
  /** Seconds each tracer lives. Default 5. */
  readonly lifetime?: number;
  /**
   * Tracers that leave this box are retired early, so smoke can vent out of
   * a scene. A retired tracer's slot is reused once its lifetime is up.
   */
  readonly bounds?: Box3;
  /**
   * Regions that heat the air to temperature 1. Hot air rises and cools as it
   * goes, carrying the smoke with it. Giving any turns on air temperature.
   */
  readonly heatSources?: readonly HeatSource[];
  /**
   * Upward acceleration, in m/s², of air at temperature 1 relative to the
   * average air temperature. Default 3. Requires `heatSources`.
   */
  readonly buoyancy?: number;
  /** How fast air cools, as an exponential rate per second. Default 0.8. Requires `heatSources`. */
  readonly cooling?: number;
}

/**
 * Smoke: massless tracer particles carried by a fluid's velocity field
 * (Macklin et al. 2014, §7.2.1). The fluid is the air; tracers make its
 * motion visible. Draw them with {@link GasVolumeRenderer} or
 * {@link GasSpriteRenderer}.
 *
 * List the gas before its fluid in the same {@link SimLoop}'s `materials`,
 * so tracers follow the solved velocities before vorticity and viscosity
 * adjust them.
 *
 * ```ts
 * const smoke = new GasSystem(air, { capacity: 5000, lifetime: 8 });
 * const loop = new SimLoop(particles, { materials: [smoke, air] });
 * smoke.emit([0, 0.1, 0]);
 * ```
 *
 * With `heatSources`, the gas also tracks air temperature and hot air rises
 * (Boussinesq buoyancy). The density solve only resists compression, so
 * rising air can leave gaps nothing refills; a little gravity on the air
 * (1 m/s² is plenty) keeps it settled while the hot air still rises.
 *
 * ```ts
 * const smoke = new GasSystem(air, {
 *   capacity: 8000,
 *   heatSources: [{ position: new Vector3(0, 0, 0), radius: 0.15 }],
 * });
 * ```
 */
export class GasSystem implements Material, SmokeTracers {
  readonly fluid: FluidSystem;

  /** The fluid's particle storage. */
  get particles(): ParticleSystem {
    return this.fluid.particles;
  }
  readonly capacity: number;
  readonly lifetime: number;
  readonly bounds: Box3 | undefined;
  readonly smokePositions: StorageBufferNode<'vec4'>;
  /** Velocity each tracer moved with during the last substep. */
  readonly smokeVelocities: StorageBufferNode<'vec4'>;
  readonly smokeAge: StorageBufferNode<'float'>;
  readonly smokeAlive: StorageBufferNode<'uint'>;
  /**
   * Temperature of each air particle, indexed from the start of the fluid's
   * range. Only present with `heatSources`.
   */
  readonly temperature: StorageBufferNode<'float'> | undefined;

  private readonly heat: AirHeat | undefined;
  /** Tracers are recycled in order, so the oldest slot is always the next free one. */
  private cursor = 0;
  private time = 0;
  private readonly bornAt: Float64Array;
  private readonly pending: number[] = [];
  private readonly spawns: StorageBufferNode<'vec4'>;
  private readonly spawnStart: UniformNode<'uint', number>;
  private readonly spawnCount: UniformNode<'uint', number>;
  /** Substep uniform of the loop that last built this gas; identifies that loop. */
  private loopDt: UniformNode<'float', number> | undefined;
  private fluidChecked = false;
  private readonly kernels: ComputeNode[] = [];
  private disposed = false;

  constructor(fluid: FluidSystem, options: GasSystemOptions) {
    const { capacity } = options;
    const lifetime = options.lifetime ?? 5;
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`GasSystem: capacity must be a positive integer, got ${capacity}`);
    }
    if (!(lifetime > 0) || !Number.isFinite(lifetime)) {
      throw new Error(`GasSystem: lifetime must be positive, got ${lifetime}`);
    }
    this.fluid = fluid;
    this.capacity = capacity;
    this.lifetime = lifetime;
    this.bounds = options.bounds?.clone();
    if (options.heatSources?.length) {
      this.heat = new AirHeat(
        fluid,
        options.heatSources,
        checkBuoyancy(options.buoyancy ?? 3),
        checkCooling(options.cooling ?? 0.8),
      );
    } else if (options.buoyancy !== undefined || options.cooling !== undefined) {
      throw new Error('GasSystem: buoyancy and cooling need heatSources');
    }
    this.temperature = this.heat?.temperature;
    this.bornAt = new Float64Array(capacity).fill(-Infinity);
    this.smokePositions = instancedArray(capacity, 'vec4');
    this.smokeVelocities = instancedArray(capacity, 'vec4');
    this.smokeAge = instancedArray(capacity, 'float');
    this.smokeAlive = instancedArray(capacity, 'uint');
    this.spawns = instancedArray(capacity, 'vec4');
    // r184's typings only accept 'float' here; 'uint' works at runtime.
    this.spawnStart = uniform(0, 'uint' as 'float') as unknown as UniformNode<'uint', number>;
    this.spawnCount = uniform(0, 'uint' as 'float') as unknown as UniformNode<'uint', number>;
  }

  /** Upward acceleration of air at temperature 1, in m/s². Requires `heatSources`. */
  get buoyancy(): number {
    return this.requireHeat().buoyancy.value;
  }
  set buoyancy(value: number) {
    this.requireHeat().buoyancy.value = checkBuoyancy(value);
  }

  /** Exponential cooling rate of the air, per second. Requires `heatSources`. */
  get cooling(): number {
    return this.requireHeat().cooling.value;
  }
  set cooling(value: number) {
    this.requireHeat().cooling.value = checkCooling(value);
  }

  private requireHeat(): AirHeat {
    if (!this.heat) throw new Error('GasSystem: give heatSources to use air temperature');
    return this.heat;
  }

  get neighborRadius(): number {
    return this.fluid.smoothingRadius;
  }

  /** Tracers currently alive. */
  get aliveCount(): number {
    let alive = this.pending.length / 3;
    for (const born of this.bornAt) if (this.time - born < this.lifetime) alive++;
    return Math.min(alive, this.capacity);
  }

  /**
   * Release a tracer at `position` at the start of the next step. Returns
   * `false` if every tracer is still alive.
   */
  emit(position: Vector3 | readonly [number, number, number]): boolean {
    this.assertAlive();
    const slot = (this.cursor + this.pending.length / 3) % this.capacity;
    if (
      this.pending.length / 3 >= this.capacity ||
      this.time - this.bornAt[slot]! < this.lifetime
    ) {
      return false;
    }
    const [x, y, z] = 'x' in position ? [position.x, position.y, position.z] : position;
    this.pending.push(x, y, z);
    return true;
  }

  update(dt: number): void {
    this.assertAlive();
    // The fluid is built after the gas, so the first step is the earliest
    // point to check it was built by the same loop.
    if (this.loopDt && !this.fluidChecked) {
      if (this.fluidLoopDt() !== this.loopDt) {
        throw new Error("GasSystem: its FluidSystem must be in the same SimLoop's `materials`");
      }
      this.fluidChecked = true;
    }
    const count = this.pending.length / 3;
    const spawns = this.spawns.value.array as Float32Array;
    for (let k = 0; k < count; k++) {
      spawns.set(
        [this.pending[k * 3]!, this.pending[k * 3 + 1]!, this.pending[k * 3 + 2]!, 1],
        k * 4,
      );
      this.bornAt[(this.cursor + k) % this.capacity] = this.time;
    }
    if (count > 0) this.spawns.value.needsUpdate = true;
    this.spawnStart.value = this.cursor;
    this.spawnCount.value = count;
    this.cursor = (this.cursor + count) % this.capacity;
    this.pending.length = 0;
    this.time += dt;
  }

  build({ hashGrid, dt }: SolverContext): MaterialKernels {
    if (this.fluidLoopDt() === dt) {
      throw new Error('GasSystem: list it before its FluidSystem in `materials`');
    }
    this.loopDt = dt;
    this.fluidChecked = false;

    const spawn = Fn(() => {
      const k: Any = instanceIndex;
      If(k.lessThan(this.spawnCount), () => {
        const slot: Any = this.spawnStart.add(k).mod(uint(this.capacity));
        this.smokePositions.element(slot).assign(vec4(this.spawns.element(k).xyz, 0));
        this.smokeVelocities.element(slot).assign(vec4(0));
        this.smokeAge.element(slot).assign(0);
        this.smokeAlive.element(slot).assign(uint(1));
      });
    })()
      .compute(this.capacity)
      .setName('GasSystem.spawn');

    const postSolve = [
      buildSmokeAdvectKernel({
        tracers: this,
        fluid: this.fluid,
        hashGrid,
        sph: createSphKernelUniforms(this.fluid.smoothingRadius),
        dt,
        lifetime: this.lifetime,
        bounds: this.bounds,
      }),
      ...(this.heat?.build(dt) ?? []),
    ];
    this.kernels.push(spawn, ...postSolve);
    return { beforeStep: [spawn], postSolve };
  }

  /**
   * Free the tracer and temperature buffers and the compiled kernels. Dispose
   * the {@link SimLoop} that runs this gas first; the gas can't be used after.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const kernel of this.kernels) kernel.dispose();
    this.kernels.length = 0;
    releaseStorageBuffers(this.fluid.particles.renderer, [
      this.smokePositions,
      this.smokeVelocities,
      this.smokeAge,
      this.smokeAlive,
      this.spawns,
      ...(this.heat?.buffers ?? []),
    ]);
  }

  /** Substep uniform of the loop that built the fluid, if it has been built. */
  private fluidLoopDt(): UniformNode<'float', number> | undefined {
    try {
      return this.fluid.kernelContext.dt;
    } catch {
      return undefined;
    }
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('GasSystem: already disposed');
  }
}

function checkBuoyancy(value: number): number {
  if (!Number.isFinite(value)) throw new Error(`GasSystem: buoyancy must be finite, got ${value}`);
  return value;
}

function checkCooling(value: number): number {
  if (!(value >= 0) || !Number.isFinite(value)) {
    throw new Error(`GasSystem: cooling must be ≥ 0, got ${value}`);
  }
  return value;
}
