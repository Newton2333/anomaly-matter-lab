import { Fn, If, float, instanceIndex, instancedArray, texture3D, uniform, vec3 } from 'three/tsl';
import {
  ClampToEdgeWrapping,
  Data3DTexture,
  DataUtils,
  HalfFloatType,
  LinearFilter,
  Matrix3,
  Matrix4,
  Quaternion,
  RedFormat,
  Vector3,
} from 'three';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { releaseStorageBuffers, type ParticleSystem } from '../particles.js';
import {
  emitColliderContact,
  emitColliderFriction,
  type Collider,
  type ColliderContext,
  type ColliderKernels,
} from './collider.js';
import { normalized, resolveFriction } from './PrimitiveSet.js';
import { emitSampleSdf, type SdfFields } from './sdf.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** A signed distance field sampled on a regular grid; negative inside. */
export interface SDFData {
  /** One distance per voxel, x fastest, then y, then z. */
  readonly data: Float32Array;
  readonly resolution: readonly [number, number, number];
  /** Corner of the first voxel, in the mesh's local space. */
  readonly origin: readonly [number, number, number];
  readonly voxelSize: readonly [number, number, number];
}

export interface SDFColliderOptions {
  /** Static friction coefficient. Default 0.5. */
  readonly muS?: number;
  /** Kinetic friction coefficient. Default 0.4. */
  readonly muK?: number;
  /** World position of the mesh's local origin. Default `(0, 0, 0)`. */
  readonly position?: Vector3;
  /** Orientation about the local origin. Normalized. Default identity. */
  readonly rotation?: Quaternion;
  /** Uniform scale about the local origin. Default `1`. */
  readonly scale?: number;
  /**
   * Extra contact distance beyond the particle radius, in metres. Default
   * `0`. Sparse particle surfaces such as cloth leave gaps that thin
   * features (ears, fins) can slip through; about half the particle spacing
   * closes them. It belongs to the particles, so it does not scale with
   * {@link SDFCollider.setScale}.
   */
  readonly thickness?: number;
}

/**
 * Collision with an arbitrary mesh through a baked signed distance field
 * (see `bakeMeshToSdf`). The field is stored as a half-float 3D texture and
 * sampled with hardware trilinear filtering. The collider can be placed,
 * rotated, and uniformly scaled at any time; friction treats it as static.
 */
export class SDFCollider implements Collider {
  readonly particles: ParticleSystem;
  readonly texture: Data3DTexture;
  /** Per-loop solver buffers, freed on dispose. */
  private readonly lambdas: { readonly value: object }[] = [];
  /** @internal Sampling inputs, shared with the fluid surface renderer. */
  readonly fields: SdfFields;

  private readonly muSUniform: UniformNode<'float', number>;
  private readonly muKUniform: UniformNode<'float', number>;
  private readonly thicknessUniform: UniformNode<'float', number>;
  private readonly positionUniform: UniformNode<'vec3', Vector3>;
  private readonly rotationUniform: UniformNode<'mat3', Matrix3>;
  private readonly invRotationUniform: UniformNode<'mat3', Matrix3>;
  private readonly scaleUniform: UniformNode<'float', number>;
  private readonly invScaleUniform: UniformNode<'float', number>;
  private readonly orientation = new Quaternion();
  private readonly kernels: ComputeNode[] = [];
  private changes = 0;

  constructor(particles: ParticleSystem, sdf: SDFData, options: SDFColliderOptions = {}) {
    const [nx, ny, nz] = sdf.resolution;
    if (![nx, ny, nz].every((n) => Number.isInteger(n) && n >= 4)) {
      throw new Error(`SDFCollider: resolution must be integers ≥ 4, got [${nx}, ${ny}, ${nz}]`);
    }
    if (sdf.data.length !== nx * ny * nz) {
      throw new Error(
        `SDFCollider: ${sdf.data.length} values do not match resolution ${nx}×${ny}×${nz}`,
      );
    }
    const thickness = options.thickness ?? 0;
    assertThickness(thickness);
    const { muS, muK } = resolveFriction(options, 'SDFCollider');
    this.particles = particles;

    // Half floats filter on every adapter; 32-bit float filtering is optional in WebGPU.
    const half = new Uint16Array(sdf.data.length);
    for (let i = 0; i < half.length; i++) half[i] = DataUtils.toHalfFloat(sdf.data[i]!);
    const texture = new Data3DTexture(half, nx, ny, nz);
    texture.format = RedFormat;
    texture.type = HalfFloatType;
    texture.minFilter = texture.magFilter = LinearFilter;
    texture.wrapS = texture.wrapT = texture.wrapR = ClampToEdgeWrapping;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;
    this.texture = texture;

    this.muSUniform = uniform(muS, 'float');
    this.muKUniform = uniform(muK, 'float');
    this.thicknessUniform = uniform(thickness, 'float');
    this.positionUniform = uniform(new Vector3());
    this.rotationUniform = uniform(new Matrix3());
    this.invRotationUniform = uniform(new Matrix3());
    this.scaleUniform = uniform(1, 'float');
    this.invScaleUniform = uniform(1, 'float');
    const [vx, vy, vz] = sdf.voxelSize;
    this.fields = {
      texture: texture3D(texture),
      position: this.positionUniform,
      bakedOrigin: uniform(new Vector3(...sdf.origin)),
      voxelSize: uniform(new Vector3(vx, vy, vz)),
      invVoxelSize: uniform(new Vector3(1 / vx, 1 / vy, 1 / vz)),
      resolution: uniform(new Vector3(nx, ny, nz)),
      rotation: this.rotationUniform,
      invRotation: this.invRotationUniform,
      scale: this.scaleUniform,
      invScale: this.invScaleUniform,
    };
    this.setPosition(options.position ?? new Vector3());
    this.setRotation(options.rotation ?? new Quaternion());
    this.setScale(options.scale ?? 1);
    this.changes = 0;
  }

  /**
   * Increments whenever the placement (position, rotation, or scale) changes,
   * so renderers can refresh anything cached against it.
   */
  get version(): number {
    return this.changes;
  }

  get muS(): number {
    return this.muSUniform.value;
  }
  set muS(value: number) {
    this.muSUniform.value = resolveFriction({ muS: value }, 'SDFCollider').muS;
  }
  get muK(): number {
    return this.muKUniform.value;
  }
  set muK(value: number) {
    this.muKUniform.value = resolveFriction({ muK: value }, 'SDFCollider').muK;
  }

  /** Extra contact distance beyond the particle radius, in metres. Not scaled by `scale`. */
  get thickness(): number {
    return this.thicknessUniform.value;
  }
  set thickness(value: number) {
    assertThickness(value);
    this.thicknessUniform.value = value;
  }

  get position(): Vector3 {
    return this.positionUniform.value.clone();
  }
  get rotation(): Quaternion {
    return this.orientation.clone();
  }
  get scale(): number {
    return this.scaleUniform.value;
  }

  setPosition(position: Vector3): void {
    if (this.positionUniform.value.equals(position)) return;
    this.positionUniform.value.copy(position);
    this.changes++;
  }

  /** Set the orientation. `rotation` is normalized. */
  setRotation(rotation: Quaternion): void {
    const unit = normalized(rotation, 'SDFCollider.setRotation');
    if (this.orientation.equals(unit)) return;
    this.orientation.copy(unit);
    this.rotationUniform.value.setFromMatrix4(new Matrix4().makeRotationFromQuaternion(unit));
    this.invRotationUniform.value.copy(this.rotationUniform.value).transpose();
    this.changes++;
  }

  setScale(scale: number): void {
    if (!(scale > 0) || !Number.isFinite(scale)) {
      throw new Error(`SDFCollider.setScale: scale must be positive, got ${scale}`);
    }
    if (this.scaleUniform.value === scale) return;
    this.scaleUniform.value = scale;
    this.invScaleUniform.value = 1 / scale;
    this.changes++;
  }

  /** Place the collider from a matrix. Scale must be uniform. */
  setTransform(transform: Matrix4): void {
    // Matrix4.decompose reads a zero scale as 1 and folds a mirror into a
    // negative x scale, so both are caught here, before it runs.
    const determinant = transform.determinant();
    if (!Number.isFinite(determinant)) {
      throw new Error('SDFCollider.setTransform: the matrix has NaN or infinite values');
    }
    if (Math.abs(determinant) < 1e-18) {
      throw new Error('SDFCollider.setTransform: scale must be non-zero');
    }
    if (determinant < 0) {
      throw new Error('SDFCollider.setTransform: mirrored transforms aren’t supported');
    }
    const position = new Vector3();
    const rotation = new Quaternion();
    const scale = new Vector3();
    transform.decompose(position, rotation, scale);
    if (Math.abs(scale.x - scale.y) + Math.abs(scale.x - scale.z) > 1e-4 * Math.abs(scale.x)) {
      throw new Error('SDFCollider.setTransform: scale must be uniform');
    }
    this.setPosition(position);
    this.setRotation(rotation);
    this.setScale(scale.x);
  }

  update(): void {}

  /** @internal */
  buildKernels({ particles, dt, positions, velocities }: ColliderContext): ColliderKernels {
    // (λ_n, λ_t) per particle.
    const lambda = instancedArray(2 * particles.capacity, 'float');
    this.lambdas.push(lambda);
    const reach: Any = float(particles.particleRadius).add(this.thicknessUniform);

    const resetLambda = Fn(() => {
      lambda.element(instanceIndex).assign(0);
    })()
      .compute(2 * particles.capacity)
      .setName('SDFCollider.resetLambda');

    /** Emit `body(normal, phi)` when the SDF gradient at `x` is usable. */
    const sample = (x: Any, body: (normal: Any, phi: Any) => void): void => {
      const phi: Any = float(0).toVar();
      const gradient: Any = vec3(0).toVar();
      emitSampleSdf(this.fields, x, phi, gradient);
      const length: Any = gradient.length().toVar();
      If(length.greaterThan(1e-6), () => body(gradient.div(length), phi));
    };

    const solve = Fn(() => {
      const p: Any = instanceIndex;
      const w: Any = particles.invMass.element(p).toVar();
      If(w.greaterThan(0), () => {
        const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
        sample(xStar, (normal, phi) => {
          If(phi.lessThan(reach), () => {
            emitColliderContact({
              particle: p,
              invMass: w,
              displacement: xStar.sub(particles.positions.element(p).xyz),
              normal,
              depth: reach.sub(phi),
              muS: this.muSUniform,
              lambda,
              lambdaIndex: p.mul(2),
              accumulator: positions,
            });
          });
        });
      });
    })()
      .compute(particles.capacity)
      .setName('SDFCollider.solve');

    const friction = Fn(() => {
      const p: Any = instanceIndex;
      const w: Any = particles.invMass.element(p).toVar();
      const lambdaN: Any = lambda.element(p.mul(2)).toVar();
      If(w.greaterThan(0).and(lambdaN.greaterThan(0)), () => {
        sample(particles.predictedPositions.element(p).xyz, (normal) => {
          emitColliderFriction({
            particle: p,
            invMass: w,
            velocity: particles.velocities.element(p).xyz,
            normal,
            lambdaN,
            muK: this.muKUniform,
            dt,
            accumulator: velocities,
          });
        });
      });
    })()
      .compute(particles.capacity)
      .setName('SDFCollider.friction');

    this.kernels.push(resetLambda, solve, friction);
    return { preSolve: [resetLambda], solve: [solve], postSolve: [friction] };
  }

  /** Dispose `texture` and release the kernels' pipelines and bindings. */
  dispose(): void {
    this.texture.dispose();
    for (const kernel of this.kernels) kernel.dispose();
    this.kernels.length = 0;
    releaseStorageBuffers(this.particles.renderer, this.lambdas);
    this.lambdas.length = 0;
  }
}

function assertThickness(value: number): void {
  if (!(value >= 0) || !Number.isFinite(value)) {
    throw new Error(`SDFCollider: thickness must be non-negative, got ${value}`);
  }
}
