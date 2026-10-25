import { Vector3, type Mesh } from 'three';

import type { ClothSystem } from '../cloth/index.js';
import type { FluidAppearance, FluidSurfaceRenderer, FluidSystem } from '../fluids/index.js';
import type { ViscositySolver } from '../fluids/index.js';
import type { GasSystem, GasVolumeRenderer } from '../gas/index.js';
import type { SoftbodyMesh, SoftbodySystem } from '../softbody/index.js';

/** Throws until the simulation has started. */
function built<T>(value: T | undefined, what: string): T {
  if (value === undefined) {
    throw new Error(`${what} is created when the simulation starts (on the first step)`);
  }
  return value;
}

type FluidSetting = 'viscosity' | 'surfaceTension' | 'vorticity' | 'adhesion';

/** A liquid added with {@link Simulation.addFluid}. Settings can change at any time. */
export class Fluid {
  /** @internal */ system: FluidSystem | undefined;
  /** @internal */ thick: ViscositySolver | undefined;
  /** @internal */ renderer: FluidSurfaceRenderer | undefined;
  /** @internal */ count = 0;

  /** @internal */
  constructor(
    /** @internal */ readonly settings: Record<FluidSetting, number> & { thickness: number },
    /** @internal */ readonly look: Partial<FluidAppearance>,
    /** @internal Whether `thickness` was given, which adds the thick-liquid solver. */
    readonly thickEnabled: boolean,
  ) {}

  /** How much the liquid resists flowing. About 0.01 for water, up to 0.3 for syrup. */
  get viscosity(): number {
    return this.settings.viscosity;
  }
  set viscosity(value: number) {
    this.set('viscosity', value);
  }
  /** Pulls the liquid into round drops and smooth sheets. About 0.1; above 0.25 streams break up. */
  get surfaceTension(): number {
    return this.settings.surfaceTension;
  }
  set surfaceTension(value: number) {
    this.set('surfaceTension', value);
  }
  /** Keeps swirls alive. About 0.02. */
  get vorticity(): number {
    return this.settings.vorticity;
  }
  set vorticity(value: number) {
    this.set('vorticity', value);
  }
  /** How much the liquid clings to soft bodies and cloth. About 0.1. */
  get adhesion(): number {
    return this.settings.adhesion;
  }
  set adhesion(value: number) {
    this.set('adhesion', value);
  }
  /**
   * Extra thickness for honey-like liquids, around 20 for honey. Pass
   * `thickness` to `addFluid`, even as 0, to change it after the simulation starts.
   */
  get thickness(): number {
    return this.settings.thickness;
  }
  set thickness(value: number) {
    if (!this.thickEnabled) {
      throw new Error(
        'Fluid.thickness: pass `thickness` to addFluid, even as 0, to be able to change it',
      );
    }
    this.settings.thickness = value;
    if (this.thick) this.thick.viscosity = value;
  }

  /** Particles this liquid uses. 0 until the simulation starts. */
  get particleCount(): number {
    return this.count;
  }

  /** Change how the liquid looks: `color`, `roughness`, `ior`, and more. */
  setAppearance(appearance: Partial<FluidAppearance>): void {
    Object.assign(this.look, appearance);
    this.renderer?.setAppearance(appearance);
  }

  /** The underlying {@link FluidSystem}, for advanced use. Available once the simulation starts. */
  get fluidSystem(): FluidSystem {
    return built(this.system, 'Fluid.fluidSystem');
  }
  /** The {@link FluidSurfaceRenderer} drawing it. Available once the simulation starts. */
  get surface(): FluidSurfaceRenderer {
    return built(this.renderer, 'Fluid.surface');
  }
  /** The liquid surface mesh in the scene. Available once the simulation starts. */
  get mesh(): Mesh {
    return this.surface.mesh;
  }

  private set(key: FluidSetting, value: number): void {
    this.settings[key] = value;
    if (this.system) this.system[key] = value;
  }
}

/** Smoke added with {@link Simulation.addSmoke}. Settings can change at any time. */
export class Smoke {
  /** @internal */ system: GasSystem | undefined;
  /** @internal */ renderer: GasVolumeRenderer | undefined;
  /** @internal */ carry = 0;
  /** @internal */ count = 0;
  private warned = false;

  /** @internal */
  constructor(
    /** @internal */ readonly settings: {
      rate: number;
      readonly maxRate: number;
      heat: number;
      cooling: number;
      opacity: number;
      readonly source: Vector3;
      readonly sourceRadius: number;
    },
  ) {}

  /**
   * Tracers released at the source per second. More makes denser, finer
   * smoke. Capped at the `maxRate` given to `addSmoke`.
   */
  get rate(): number {
    return this.settings.rate;
  }
  set rate(value: number) {
    if (value > this.settings.maxRate && !this.warned) {
      this.warned = true;
      console.warn(
        `Smoke.rate: ${value} is above maxRate (${this.settings.maxRate}), so it's capped. Pass a higher maxRate to addSmoke.`,
      );
    }
    this.settings.rate = value;
  }
  /** How strongly the source heats the air, as upward acceleration in m/s². 0 stops it rising. */
  get heat(): number {
    return this.settings.heat;
  }
  set heat(value: number) {
    if (!Number.isFinite(value)) throw new Error(`Smoke.heat: must be finite, got ${value}`);
    this.settings.heat = value;
    if (this.system) this.system.buoyancy = value;
  }
  /** How fast the air cools, per second. Faster cooling stops the smoke rising sooner. */
  get cooling(): number {
    return this.settings.cooling;
  }
  set cooling(value: number) {
    if (!(value >= 0) || !Number.isFinite(value)) {
      throw new Error(`Smoke.cooling: must be finite and ≥ 0, got ${value}`);
    }
    this.settings.cooling = value;
    if (this.system) this.system.cooling = value;
  }
  /** How opaque the smoke looks. */
  get opacity(): number {
    return this.settings.opacity;
  }
  set opacity(value: number) {
    this.settings.opacity = value;
    if (this.renderer) this.renderer.density = value;
  }
  /** Air particles filling the container. 0 until the simulation starts. */
  get particleCount(): number {
    return this.count;
  }
  /** Where smoke is released and air is heated. Mutate it to move the source. */
  get source(): Vector3 {
    return this.settings.source;
  }

  /**
   * Release one extra tracer at `position`. Returns `false` when every
   * tracer is still alive, so nothing was released.
   */
  emit(position: Vector3 | readonly [number, number, number]): boolean {
    return built(this.system, 'Smoke.emit').emit(position);
  }

  /** The underlying {@link GasSystem}, for advanced use. Available once the simulation starts. */
  get gasSystem(): GasSystem {
    return built(this.system, 'Smoke.gasSystem');
  }
}

/** A soft body added with {@link Simulation.addSoftbody}. */
export class Softbody {
  /** @internal */ system: SoftbodySystem | undefined;
  /** @internal */ index = -1;
  /** @internal */ count = 0;
  /** @internal */ skinned: SoftbodyMesh | undefined;

  /** @internal */
  constructor(
    /** The mesh the body was made from. It's hidden once the simulation starts. */
    readonly source: Mesh,
    /** @internal */ readonly settings: { softness: number; readonly density: number },
  ) {}

  /** Particles this body uses. 0 until the simulation starts. */
  get particleCount(): number {
    return this.count;
  }
  /** Density in kg/m³, as given to `addSoftbody`. */
  get density(): number {
    return this.settings.density;
  }
  /** 0 is firm, 1 is loose jelly. */
  get softness(): number {
    return this.settings.softness;
  }
  set softness(value: number) {
    this.settings.softness = value;
    if (this.system) this.system.setCompliance(this.index, softbodyCompliance(value, this.count));
  }

  /** The deforming mesh drawn in the scene. Available once the simulation starts. */
  get mesh(): SoftbodyMesh {
    return built(this.skinned, 'Softbody.mesh');
  }
  /**
   * The {@link SoftbodySystem} underneath, for advanced use; this body is
   * `bodyIndex` in it. Available once the simulation starts.
   */
  get softbodySystem(): SoftbodySystem {
    return built(this.system, 'Softbody.softbodySystem');
  }
  /** This body's index in {@link softbodySystem}. */
  get bodyIndex(): number {
    return this.index;
  }
}

/** Shape-matching compliance for local shape matching: 0 is firm rubber, 1 is loose jelly. */
export function softbodyCompliance(softness: number, count: number): number {
  return 10 ** (-6 + softness * 3) * (count / 200);
}

/** A cloth added with {@link Simulation.addCloth}. */
export class Cloth {
  /** @internal */ system: ClothSystem | undefined;
  /** @internal */ surfaceMesh: Mesh | undefined;
  /** @internal */ segments = 0;
  /** @internal */ count = 0;

  /** @internal */
  constructor(
    /** @internal */ readonly settings: {
      softness: number;
      readonly weight: number;
      readonly wind: Vector3;
    },
  ) {}

  /** Particles this cloth uses. 0 until the simulation starts. */
  get particleCount(): number {
    return this.count;
  }
  /** Wind velocity in m/s. Mutate it to change the wind. */
  get wind(): Vector3 {
    return this.settings.wind;
  }
  /** 0 is stiff canvas, 1 drapes like silk. */
  get softness(): number {
    return this.settings.softness;
  }
  set softness(value: number) {
    this.settings.softness = value;
    if (this.system)
      this.system.bendCompliance = clothBendCompliance(value, this.segments, this.settings.weight);
  }

  /** The cloth surface drawn in the scene. Available once the simulation starts. */
  get mesh(): Mesh {
    return built(this.surfaceMesh, 'Cloth.mesh');
  }
  /** The underlying {@link ClothSystem}, for advanced use. Available once the simulation starts. */
  get clothSystem(): ClothSystem {
    return built(this.system, 'Cloth.clothSystem');
  }
}

/**
 * Bending compliance that gives the same drape at every grid resolution:
 * bending gradients grow as the grid refines and particle masses shrink.
 */
export function clothBendCompliance(softness: number, segments: number, weight: number): number {
  return 10 ** (-1 + softness * 5) * (segments / 30) ** 4 * (0.35 / weight);
}
