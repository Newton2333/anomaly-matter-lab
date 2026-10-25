import { BackSide, BoxGeometry, HalfFloatType, Mesh, Vector3 } from 'three';
import { MeshBasicNodeMaterial, Storage3DTexture, type WebGPURenderer } from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicLoad,
  atomicStore,
  cameraPosition,
  cameraProjectionMatrixInverse,
  color,
  float,
  getViewPosition,
  instanceIndex,
  instancedArray,
  ivec3,
  mix,
  positionWorld,
  screenCoordinate,
  screenUV,
  texture3D,
  textureStore,
  uint,
  uniform,
  vec3,
  vec4,
  viewportDepthTexture,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { SmokeTracers } from './types.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface GasVolumeRendererOptions {
  readonly renderer: WebGPURenderer;
  /** Corners of the box the smoke is drawn in. */
  readonly min: Vector3;
  readonly max: Vector3;
  /** Density grid size, 4–128 per axis. Default `[32, 56, 32]`. */
  readonly resolution?: readonly [number, number, number];
  /** Ray-march steps, 8–128. Default 56. */
  readonly steps?: number;
  /** Smoke opacity multiplier. Default 1. */
  readonly density?: number;
  /** Color of fully lit smoke. Default `0xd8dfe6`. */
  readonly color?: number;
  /** Color of smoke in its own shadow. Default `0x3b4758`. */
  readonly shadowColor?: number;
  /** Direction toward the light. Default `(-0.35, 0.8, 0.4)`. */
  readonly lightDirection?: Vector3;
  /**
   * Density fades smoothly to zero near the box's walls, so smoke doesn't end
   * in a hard edge. Each value is the fade's width as a fraction of the box:
   * `sides` for the four vertical walls, `bottom` and `top` for the floor and
   * ceiling. 0 turns that fade off. Default
   * `{ sides: 0.07, bottom: 0.025, top: 0.24 }`.
   */
  readonly edgeFade?: {
    readonly sides?: number;
    readonly bottom?: number;
    readonly top?: number;
  };
}

/**
 * Draws smoke tracers as lit, volumetric smoke: tracers are splatted into a
 * density grid, blurred, lit with single scattering from one light
 * direction, and ray marched inside a box that respects scene depth.
 * Call {@link update} once per frame before rendering.
 */
export class GasVolumeRenderer {
  readonly object: Mesh;
  private readonly renderer: WebGPURenderer;
  private readonly densityUniform: UniformNode<'float', number>;
  private readonly kernels: ComputeNode[];
  private readonly textures: Storage3DTexture[];

  constructor(gas: SmokeTracers, options: GasVolumeRendererOptions) {
    const { min, max, resolution = [32, 56, 32], steps = 56 } = options;
    this.renderer = options.renderer;
    const [nx, ny, nz] = resolution;
    if (resolution.some((n) => !Number.isInteger(n) || n < 4 || n > 128)) {
      throw new Error(
        `GasVolumeRenderer: resolution must be integers in 4–128, got [${resolution.join(', ')}]`,
      );
    }
    if (!Number.isInteger(steps) || steps < 8 || steps > 128) {
      throw new Error(`GasVolumeRenderer: steps must be an integer in 8–128, got ${steps}`);
    }
    const size = max.clone().sub(min);
    if (!(Math.min(size.x, size.y, size.z) > 0)) {
      throw new Error('GasVolumeRenderer: max must exceed min on every axis');
    }
    const { sides = 0.07, bottom = 0.025, top = 0.24 } = options.edgeFade ?? {};
    for (const [name, value, limit] of [
      ['sides', sides, 0.5],
      ['bottom', bottom, 1],
      ['top', top, 1],
    ] as const) {
      if (!(value >= 0 && value <= limit)) {
        throw new Error(
          `GasVolumeRenderer: edgeFade.${name} must be in [0, ${limit}], got ${value}`,
        );
      }
    }
    const count = nx * ny * nz;
    const ticks = (instancedArray(count, 'uint') as Any).setAtomic(true);
    const a = instancedArray(count, 'float');
    const b = instancedArray(count, 'float');
    const minimum = uniform(min.clone());
    const extent = uniform(size);
    this.densityUniform = uniform(options.density ?? 1, 'float');
    const scale = vec3(nx - 1, ny - 1, nz - 1);
    const indexOf = (p: Any): Any =>
      p.z
        .mul(nx * ny)
        .add(p.y.mul(nx))
        .add(p.x)
        .toUint();
    const cellOf = (i: Any): Any =>
      vec3(i.mod(nx).toFloat(), i.div(nx).mod(ny).toFloat(), i.div(nx * ny).toFloat());
    const clear = Fn(() => {
      atomicStore(ticks.element(instanceIndex), uint(0));
    })()
      .compute(count)
      .setName('GasVolumeRenderer.clear');
    const splat = Fn(() => {
      const i: Any = instanceIndex;
      If(gas.smokeAlive.element(i).greaterThan(uint(0)), () => {
        const p: Any = gas.smokePositions
          .element(i)
          .xyz.sub(minimum)
          .div(extent)
          .mul(scale)
          .toVar();
        If(
          p
            .greaterThanEqual(vec3(1))
            .all()
            .and(p.lessThan(scale.sub(1)).all()),
          () => {
            const base = p.floor(),
              f = p.fract();
            // Each tracer carries a fixed random weight, so the smoke's grain is
            // attached to the smoke and moves with it.
            const weight: Any = i
              .toFloat()
              .mul(12.9898)
              .sin()
              .mul(43758.5453)
              .fract()
              .mul(1.2)
              .add(0.4);
            const fade: Any = float(1)
              .sub(gas.smokeAge.element(i).div(gas.lifetime))
              .max(0)
              .mul(weight);
            for (let z = 0; z < 2; z++)
              for (let y = 0; y < 2; y++)
                for (let x = 0; x < 2; x++) {
                  const w: Any = (x ? f.x : float(1).sub(f.x))
                    .mul(y ? f.y : float(1).sub(f.y))
                    .mul(z ? f.z : float(1).sub(f.z))
                    .mul(fade);
                  atomicAdd(ticks.element(indexOf(base.add(vec3(x, y, z)))), w.mul(65536).toUint());
                }
          },
        );
      });
    })()
      .compute(gas.capacity)
      .setName('GasVolumeRenderer.splat');
    const blur = (axis: number, source: Any, target: Any, atomic = false): ComputeNode =>
      Fn(() => {
        const cell: Any = cellOf(instanceIndex).toVar();
        const sum: Any = float(0).toVar();
        const weights = [0.006, 0.0606, 0.2417, 0.3829, 0.2417, 0.0606, 0.006];
        for (let k = -3; k <= 3; k++) {
          const offset = [0, 0, 0];
          offset[axis] = k;
          const neighbor: Any = cell
            .add(vec3(...(offset as [number, number, number])))
            .clamp(vec3(0), scale);
          const value: Any = atomic
            ? (atomicLoad(source.element(indexOf(neighbor))) as Any).toFloat().div(65536)
            : source.element(indexOf(neighbor));
          sum.addAssign(value.mul(weights[k + 3]!));
        }
        target.element(instanceIndex).assign(sum);
      })()
        .compute(count)
        .setName('GasVolumeRenderer.blur');
    const extinction = 0.00015 / ((size.x * size.y * size.z) / count);
    // 0 → 1 over `width` from a wall; smoothstep with equal edges is undefined, so skip it.
    const ramp = (distance: Any, width: number): Any =>
      width > 0 ? distance.smoothstep(0, width) : float(1);
    const sampleParticles = Fn(([world]: Any[]) => {
      const unit: Any = world.sub(minimum).div(extent).toVar();
      const edge: Any = unit.min(vec3(1).sub(unit));
      const fade: Any = ramp(edge.x.min(edge.z), sides)
        .mul(ramp(unit.y, bottom))
        .mul(ramp(float(1).sub(unit.y), top));
      const p: Any = unit.mul(scale).clamp(vec3(0), scale.sub(0.001)).toVar();
      const cell = p.floor(),
        f = p.fract();
      const sum: Any = float(0).toVar();
      for (let z = 0; z < 2; z++)
        for (let y = 0; y < 2; y++)
          for (let x = 0; x < 2; x++) {
            const w: Any = (x ? f.x : float(1).sub(f.x))
              .mul(y ? f.y : float(1).sub(f.y))
              .mul(z ? f.z : float(1).sub(f.z));
            sum.addAssign(a.element(indexOf(cell.add(vec3(x, y, z)))).mul(w));
          }
      return sum.mul(extinction).mul(this.densityUniform).mul(fade);
    });
    const densityTexture = new Storage3DTexture(nx, ny, nz);
    const volumeTexture = new Storage3DTexture(nx, ny, nz);
    for (const texture of [densityTexture, volumeTexture]) texture.type = HalfFloatType;
    this.textures = [densityTexture, volumeTexture];
    const densityAt = (world: Any): Any =>
      (texture3D(densityTexture, world.sub(minimum).div(extent)) as Any).level(0).r;
    // Resolve density and incident light once per voxel. The ray marcher then
    // uses one hardware-filtered 3D sample per step rather than dozens of buffer reads.
    const resolve = Fn(() => {
      const cell: Any = cellOf(instanceIndex);
      const p: Any = minimum.add(
        cell
          .add(0.5)
          .div(vec3(nx, ny, nz))
          .mul(extent),
      );
      const density: Any = sampleParticles(p);
      textureStore(densityTexture, ivec3(cell), vec4(density, 0, 0, 1)).toWriteOnly();
    })()
      .compute(count)
      .setName('GasVolumeRenderer.resolve');
    const sun = vec3((options.lightDirection ?? new Vector3(-0.35, 0.8, 0.4)).clone().normalize());
    const lighting = Fn(() => {
      const cell: Any = cellOf(instanceIndex);
      const p: Any = minimum.add(
        cell
          .add(0.5)
          .div(vec3(nx, ny, nz))
          .mul(extent),
      );
      const shadow: Any = float(0).toVar();
      const taps = [0.03, 0.07, 0.13, 0.22, 0.36, 0.58];
      taps.forEach((distance, k) => {
        const length = distance - (taps[k - 1] ?? 0);
        shadow.addAssign(densityAt(p.add(sun.mul(distance))).mul(length));
      });
      // Beer–Lambert with a softer second lobe standing in for multiple scattering,
      // so shadowed smoke darkens gradually instead of going flat.
      const light: Any = shadow.mul(-2.5).exp().mul(0.7).add(shadow.mul(-0.35).exp().mul(0.3));
      textureStore(volumeTexture, ivec3(cell), vec4(densityAt(p), light, 0, 1)).toWriteOnly();
    })()
      .compute(count)
      .setName('GasVolumeRenderer.lighting');
    this.kernels = [
      clear,
      splat,
      blur(0, ticks, a, true),
      blur(1, a, b),
      blur(2, b, a),
      resolve,
      lighting,
    ];
    const material = new MeshBasicNodeMaterial({
      transparent: true,
      side: BackSide,
      depthWrite: false,
      // Depth-testing the box rejects smoke in front of opaque geometry whenever
      // its exit face is behind that geometry. Clip each ray to scene depth instead.
      depthTest: false,
      fog: false,
    });
    material.fragmentNode = Fn(() => {
      const direction: Any = positionWorld.sub(cameraPosition).normalize().toVar();
      const inv: Any = vec3(1).div(direction.add(0.000001));
      const near: Any = minimum.sub(cameraPosition).mul(inv);
      const far: Any = minimum.add(extent).sub(cameraPosition).mul(inv);
      const lower = near.min(far),
        upper = near.max(far);
      const start: Any = lower.x.max(lower.y).max(lower.z).max(0).toVar();
      const end: Any = upper.x.min(upper.y).min(upper.z).toVar();
      const sceneDepth: Any = viewportDepthTexture();
      const opaqueDistance: Any = getViewPosition(
        screenUV,
        sceneDepth.r,
        cameraProjectionMatrixInverse,
      ).length();
      end.assign(end.min(opaqueDistance));
      const step: Any = end.sub(start).max(0).div(steps).toVar();
      const jitter: Any = screenCoordinate
        .dot(vec3(12.9898, 78.233, 0).xy)
        .sin()
        .mul(43758.5453)
        .fract();
      const t: Any = start.add(step.mul(jitter)).toVar();
      const transmittance: Any = float(1).toVar();
      const radiance: Any = vec3(0).toVar();
      Loop(steps, () => {
        const p: Any = cameraPosition.add(direction.mul(t));
        const field: Any = (texture3D(volumeTexture, p.sub(minimum).div(extent)) as Any)
          .level(0)
          .toVar();
        const density: Any = field.r;
        If(density.greaterThan(0.005), () => {
          const shade: Any = mix(
            color(options.shadowColor ?? 0x3b4758),
            color(options.color ?? 0xd8dfe6),
            field.g,
          );
          const alpha: Any = float(1).sub(density.mul(step).negate().exp());
          radiance.addAssign(shade.mul(alpha).mul(transmittance));
          transmittance.mulAssign(float(1).sub(alpha));
        });
        If(transmittance.lessThan(0.015), () => {
          Break();
        });
        t.addAssign(step);
      });
      const alpha: Any = float(1).sub(transmittance);
      return vec4(radiance.div(alpha.max(0.0001)), alpha);
    })();
    this.object = new Mesh(new BoxGeometry(size.x, size.y, size.z), material);
    this.object.position.copy(min).addScaledVector(size, 0.5);
    this.object.name = 'GasVolume';
    this.object.frustumCulled = false;
  }

  /** Smoke opacity multiplier. */
  get density(): number {
    return this.densityUniform.value;
  }
  set density(value: number) {
    this.densityUniform.value = value;
  }

  /** Rebuild the density volume from the tracers. Call once per frame before rendering. */
  async update(): Promise<void> {
    if (this.object.visible) await this.renderer.computeAsync(this.kernels);
  }

  dispose(): void {
    this.object.geometry.dispose();
    (this.object.material as MeshBasicNodeMaterial).dispose();
    for (const texture of this.textures) texture.dispose();
  }
}
