import {
  Fn,
  If,
  Loop,
  float,
  instanceIndex,
  instancedArray,
  uint,
  uniform,
  uniformArray,
  vec3,
} from 'three/tsl';
import { Quaternion, Vector3, Vector4, type Object3D } from 'three';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { releaseStorageBuffers, type ParticleSystem } from '../particles.js';
import {
  emitColliderContact,
  emitColliderFriction,
  type Collider,
  type ColliderContext,
  type ColliderKernels,
} from './collider.js';
import {
  FLAG_INVERT,
  FLAG_SWEEP,
  KIND_BOX,
  KIND_CAPSULE,
  KIND_PLANE,
  KIND_SPHERE,
  emitColliderSdf,
  emitColliderSurfaceVelocity,
} from './primitives.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Friction and motion shared by every primitive. */
export interface PrimitiveOptions {
  /** Static friction coefficient. Default 0.5. */
  readonly muS?: number;
  /** Kinetic friction coefficient. Default 0.4. */
  readonly muK?: number;
  /**
   * Surface velocity in m/s, used for friction (a conveyor belt, for example).
   * The primitive itself stays put. Primitives attached to an object derive
   * their velocity from the object's motion instead.
   */
  readonly velocity?: Vector3;
}

export interface SolidPrimitiveOptions extends PrimitiveOptions {
  /** Keep particles inside the shape instead of outside it. Default `false`. */
  readonly invert?: boolean;
}

export interface BoxOptions extends SolidPrimitiveOptions {
  /** Orientation. Normalized. Default identity. */
  readonly rotation?: Quaternion;
}

interface Primitive {
  kind: number;
  flags: number;
  /** Plane: normal. Sphere and box: center (w = radius). Capsule: a (w = radius). */
  data0: [number, number, number, number];
  /** Plane: point. Box: half extents. Capsule: b. w = μ_s. */
  data1: [number, number, number, number];
  /** Velocity, w = μ_k. */
  velocity: [number, number, number, number];
  /** Angular velocity about the pivot, rad/s. */
  spin: [number, number, number];
  rotation: Quaternion;
}

interface Attachment {
  readonly slot: number;
  readonly object: Object3D;
  readonly previous: Vector3;
  readonly previousRotation: Quaternion;
  /** Applied after the object's rotation: identity for boxes, the inverse attach-time rotation otherwise. */
  readonly offset: Quaternion;
  /** Plane normal or capsule half axis when attached. */
  readonly direction: Vector3;
}

/** GPU copies of the primitives, allocated the first time a kernel needs them. */
interface PrimitiveBuffers {
  readonly capacity: number;
  /** `(kind << 16) | flags` per slot. */
  readonly packed: StorageBufferNode<'uint'>;
  readonly data0: StorageBufferNode<'vec4'>;
  readonly data1: StorageBufferNode<'vec4'>;
  readonly velocity: StorageBufferNode<'vec4'>;
  /** Orientation per slot, stored in uniform memory to save storage bindings. */
  readonly rotation: Any;
  readonly rotations: Quaternion[];
  /** Angular velocity per slot, uniform memory like `rotation`. */
  readonly spin: Any;
  readonly spins: Vector4[];
  readonly count: UniformNode<'uint', number>;
}

/**
 * Planes, spheres, boxes, and capsules that particles collide with.
 *
 * Add shapes, then pass the set to {@link SimLoop}'s `colliders`. To move a
 * shape, {@link attach} it to an `Object3D` and move or turn the object; the
 * set follows it every step, sweeps it across the substeps, and derives its
 * surface velocity for friction.
 *
 * ```ts
 * const colliders = new PrimitiveSet(particles);
 * colliders.addPlane(new Vector3(0, 1, 0), new Vector3());
 * const paddle = colliders.addBox(new Vector3(0, 0.5, 0), new Vector3(0.2, 0.05, 0.01));
 * colliders.attach(paddle, paddleMesh);
 * ```
 */
export class PrimitiveSet implements Collider {
  readonly particles: ParticleSystem;

  private readonly primitives: Primitive[] = [];
  private readonly attachments: Attachment[] = [];
  private readonly reserved: number | undefined;
  private buffers: PrimitiveBuffers | undefined;
  private readonly kernels: ComputeNode[] = [];
  private dirty = true;
  private disposed = false;
  private changes = 0;
  /**
   * Seconds left in the current frame. The solver rewinds moving primitives by
   * this much, so a once-per-frame update becomes a smooth sweep.
   */
  private readonly motionClock = instancedArray(1, 'float');
  /** Per-loop solver buffers, freed on dispose. */
  private readonly lambdas: { readonly value: object }[] = [];

  /**
   * @param options.capacity Slots to reserve for primitives added after the
   *   simulation starts. Defaults to the number added before then.
   */
  constructor(particles: ParticleSystem, options: { readonly capacity?: number } = {}) {
    if (
      options.capacity !== undefined &&
      !(Number.isInteger(options.capacity) && options.capacity > 0)
    ) {
      throw new Error(`PrimitiveSet: capacity must be a positive integer, got ${options.capacity}`);
    }
    this.particles = particles;
    this.reserved = options.capacity;
  }

  /** Number of primitives. */
  get count(): number {
    return this.primitives.length;
  }

  /** Increments whenever primitives change, so renderers can refresh cached shapes. */
  get version(): number {
    return this.changes;
  }

  /** Add a plane through `point`; particles stay on the side `normal` points to. */
  addPlane(normal: Vector3, point: Vector3, options: PrimitiveOptions = {}): number {
    if (!(normal.lengthSq() > 0)) throw new Error('PrimitiveSet.addPlane: normal must be non-zero');
    const n = normal.clone().normalize();
    return this.add(KIND_PLANE, 0, [n.x, n.y, n.z, 0], [point.x, point.y, point.z, 0], options);
  }

  addSphere(center: Vector3, radius: number, options: SolidPrimitiveOptions = {}): number {
    assertPositive(radius, 'PrimitiveSet.addSphere: radius');
    return this.add(
      KIND_SPHERE,
      options.invert ? FLAG_INVERT : 0,
      [center.x, center.y, center.z, radius],
      [0, 0, 0, 0],
      options,
    );
  }

  addBox(center: Vector3, halfExtents: Vector3, options: BoxOptions = {}): number {
    if (!(halfExtents.x > 0 && halfExtents.y > 0 && halfExtents.z > 0)) {
      throw new Error('PrimitiveSet.addBox: halfExtents must be positive');
    }
    const rotation = options.rotation && normalized(options.rotation, 'PrimitiveSet.addBox');
    return this.add(
      KIND_BOX,
      options.invert ? FLAG_INVERT : 0,
      [center.x, center.y, center.z, 0],
      [halfExtents.x, halfExtents.y, halfExtents.z, 0],
      options,
      rotation,
    );
  }

  /** Add a capsule between `a` and `b`. */
  addCapsule(a: Vector3, b: Vector3, radius: number, options: PrimitiveOptions = {}): number {
    assertPositive(radius, 'PrimitiveSet.addCapsule: radius');
    return this.add(KIND_CAPSULE, 0, [a.x, a.y, a.z, radius], [b.x, b.y, b.z, 0], options);
  }

  /**
   * Make a primitive follow `object`'s world position and orientation. The
   * primitive's reference point (plane point, center, or capsule midpoint)
   * moves to the object's position. A box takes the object's orientation; a
   * plane's normal and a capsule's axis keep their current direction and turn
   * with the object from now on. Object scale is ignored.
   */
  attach(slot: number, object: Object3D): void {
    this.assertAlive();
    const primitive = this.primitiveAt(slot);
    const index = this.attachments.findIndex((a) => a.slot === slot);
    if (index >= 0) this.attachments.splice(index, 1);
    const position = object.getWorldPosition(new Vector3());
    const rotation = object.getWorldQuaternion(new Quaternion());
    const [x0, y0, z0] = primitive.data0;
    const [x1, y1, z1] = primitive.data1;
    const direction =
      primitive.kind === KIND_PLANE
        ? new Vector3(x0, y0, z0)
        : primitive.kind === KIND_CAPSULE
          ? new Vector3(x1 - x0, y1 - y0, z1 - z0).multiplyScalar(0.5)
          : new Vector3();
    const offset = primitive.kind === KIND_BOX ? new Quaternion() : rotation.clone().invert();
    const attachment = {
      slot,
      object,
      previous: position.clone(),
      previousRotation: rotation.clone(),
      offset,
      direction,
    };
    this.attachments.push(attachment);
    primitive.flags |= FLAG_SWEEP;
    this.place(attachment, position, rotation);
    this.dirty = true;
  }

  /** Move and resize a sphere. `velocity` feeds friction and sweeps the sphere within a frame. */
  setSphere(slot: number, center: Vector3, radius: number, velocity?: Vector3): void {
    this.assertAlive();
    const primitive = this.primitiveAt(slot);
    if (primitive.kind !== KIND_SPHERE)
      throw new Error(`PrimitiveSet.setSphere: slot ${slot} is not a sphere`);
    assertPositive(radius, 'PrimitiveSet.setSphere: radius');
    primitive.data0 = [center.x, center.y, center.z, radius];
    primitive.flags |= FLAG_SWEEP;
    primitive.velocity = [
      velocity?.x ?? 0,
      velocity?.y ?? 0,
      velocity?.z ?? 0,
      primitive.velocity[3],
    ];
    this.dirty = true;
  }

  /** Follow attached objects and upload any changes. {@link SimLoop.step} calls this. */
  update(dt: number): void {
    if (this.disposed) return;
    if (this.attachments.length > 0 && dt > 0) {
      const position = new Vector3();
      const rotation = new Quaternion();
      const turn = new Quaternion();
      for (const attachment of this.attachments) {
        const { object, previous, previousRotation } = attachment;
        const primitive = this.primitives[attachment.slot]!;
        object.getWorldPosition(position);
        object.getWorldQuaternion(rotation);
        const v = position.clone().sub(previous).divideScalar(dt);
        primitive.velocity = [v.x, v.y, v.z, primitive.velocity[3]];
        // ω = axis · angle / dt of the shortest turn since the last step.
        turn.copy(previousRotation).invert().premultiply(rotation);
        if (turn.w < 0) turn.set(-turn.x, -turn.y, -turn.z, -turn.w);
        const s = Math.hypot(turn.x, turn.y, turn.z);
        const rate = s > 1e-12 ? (2 * Math.atan2(s, turn.w)) / (s * dt) : 0;
        primitive.spin = [turn.x * rate, turn.y * rate, turn.z * rate];
        this.place(attachment, position, rotation);
        previous.copy(position);
        previousRotation.copy(rotation);
      }
      this.dirty = true;
    }
    if (this.dirty && this.buffers) this.upload(this.buffers);
  }

  /** @internal GPU buffers read by collider kernels and the fluid surface renderer. */
  get gpu(): PrimitiveBuffers {
    this.assertAlive();
    if (!this.buffers) {
      const capacity = Math.max(1, this.reserved ?? this.primitives.length);
      const rotations = Array.from({ length: capacity }, () => new Quaternion());
      const spins = Array.from({ length: capacity }, () => new Vector4());
      this.buffers = {
        capacity,
        packed: instancedArray(capacity, 'uint'),
        data0: instancedArray(capacity, 'vec4'),
        data1: instancedArray(capacity, 'vec4'),
        velocity: instancedArray(capacity, 'vec4'),
        rotation: uniformArray(rotations, 'vec4'),
        rotations,
        spin: uniformArray(spins, 'vec4'),
        spins,
        // r184's typings only accept 'float' here; 'uint' works at runtime.
        count: uniform(0, 'uint' as 'float') as unknown as UniformNode<'uint', number>,
      };
      this.upload(this.buffers);
    }
    return this.buffers;
  }

  /** @internal */
  buildKernels({
    particles,
    dt,
    substeps,
    positions,
    velocities,
  }: ColliderContext): ColliderKernels {
    const gpu = this.gpu;
    const radius = particles.particleRadius;
    // (λ_n, λ_t) per particle–primitive pair.
    const lambda = instancedArray(2 * particles.capacity * gpu.capacity, 'float');
    this.lambdas.push(lambda);
    const clock: Any = this.motionClock.element(0);

    const resetLambda = Fn(() => {
      lambda.element(instanceIndex).assign(0);
    })()
      .compute(2 * particles.capacity * gpu.capacity)
      .setName('PrimitiveSet.resetLambda');

    const forEachPrimitive = (body: (slot: Any) => void): void => {
      Loop(
        { start: uint(0), end: gpu.count as Any, type: 'uint', condition: '<' },
        ({ i }: { i: Any }) => body(i),
      );
    };

    const solve = Fn(() => {
      const p: Any = instanceIndex;
      const w: Any = particles.invMass.element(p).toVar();
      If(w.greaterThan(0), () => {
        const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
        const displacement: Any = xStar.sub(particles.positions.element(p).xyz).toVar();
        forEachPrimitive((slot) => {
          const phi: Any = float(0).toVar();
          const normal: Any = vec3(0).toVar();
          emitColliderSdf(gpu, slot, xStar, phi, normal, clock);
          If(phi.lessThan(radius), () => {
            emitColliderContact({
              particle: p,
              invMass: w,
              displacement,
              colliderDisplacement: emitColliderSurfaceVelocity(gpu, slot, xStar, clock).mul(dt),
              normal,
              depth: float(radius).sub(phi),
              muS: gpu.data1.element(slot).w,
              lambda,
              lambdaIndex: p.mul(uint(gpu.capacity)).add(slot).mul(uint(2)),
              accumulator: positions,
            });
          });
        });
      });
    })()
      .compute(particles.capacity)
      .setName('PrimitiveSet.solve');

    const friction = Fn(() => {
      const p: Any = instanceIndex;
      const w: Any = particles.invMass.element(p).toVar();
      If(w.greaterThan(0), () => {
        const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
        const velocity: Any = particles.velocities.element(p).xyz.toVar();
        forEachPrimitive((slot) => {
          const lambdaN: Any = lambda
            .element(p.mul(uint(gpu.capacity)).add(slot).mul(uint(2)))
            .toVar();
          If(lambdaN.greaterThan(0), () => {
            const phi: Any = float(0).toVar();
            const gradient: Any = vec3(0).toVar();
            emitColliderSdf(gpu, slot, xStar, phi, gradient, clock);
            const length: Any = gradient.length().toVar();
            If(length.greaterThan(1e-6), () => {
              emitColliderFriction({
                particle: p,
                invMass: w,
                velocity,
                colliderVelocity: emitColliderSurfaceVelocity(gpu, slot, xStar, clock),
                normal: gradient.div(length),
                lambdaN,
                muK: gpu.velocity.element(slot).w,
                dt,
                accumulator: velocities,
              });
            });
          });
        });
      });
    })()
      .compute(particles.capacity)
      .setName('PrimitiveSet.friction');

    const frameStart = Fn(() => {
      clock.assign(dt.mul(substeps.sub(1)));
    })()
      .compute(1)
      .setName('PrimitiveSet.frameStart');
    const tick = Fn(() => {
      clock.assign(clock.sub(dt).max(0));
    })()
      .compute(1)
      .setName('PrimitiveSet.tick');
    this.kernels.push(resetLambda, solve, friction, frameStart, tick);
    return {
      // The clock starts at the time left after the first substep and ticks
      // down once per substep.
      frameStart: [frameStart],
      preSolve: [resetLambda],
      solve: [solve],
      postSolve: [friction],
      substepEnd: [tick],
    };
  }

  /**
   * Remove all attachments and release the kernels' pipelines and bindings.
   * three.js has no call to free a storage buffer, so the GPU buffers are
   * dropped and freed once garbage-collected. The set cannot be used again.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.attachments.length = 0;
    for (const kernel of this.kernels) kernel.dispose();
    this.kernels.length = 0;
    const { buffers } = this;
    releaseStorageBuffers(this.particles.renderer, [
      this.motionClock,
      ...this.lambdas,
      ...(buffers ? [buffers.packed, buffers.data0, buffers.data1, buffers.velocity] : []),
    ]);
    this.lambdas.length = 0;
    this.buffers = undefined;
  }

  private add(
    kind: number,
    flags: number,
    data0: [number, number, number, number],
    data1: [number, number, number, number],
    options: PrimitiveOptions,
    rotation?: Quaternion,
  ): number {
    this.assertAlive();
    const capacity = this.reserved ?? this.buffers?.capacity;
    if (capacity !== undefined && this.primitives.length >= capacity) {
      throw new Error(
        `PrimitiveSet: capacity ${capacity} is full. Pass a larger \`capacity\` to reserve slots for primitives added after the simulation starts.`,
      );
    }
    const { muS, muK } = resolveFriction(options, 'PrimitiveSet');
    const v = options.velocity;
    this.primitives.push({
      kind,
      flags,
      data0,
      data1: [data1[0], data1[1], data1[2], muS],
      velocity: [v?.x ?? 0, v?.y ?? 0, v?.z ?? 0, muK],
      spin: [0, 0, 0],
      rotation: rotation?.clone() ?? new Quaternion(),
    });
    this.dirty = true;
    return this.primitives.length - 1;
  }

  private primitiveAt(slot: number): Primitive {
    const primitive = this.primitives[slot];
    if (!primitive) throw new Error(`PrimitiveSet: no primitive in slot ${slot}`);
    return primitive;
  }

  /** Place an attached primitive from its object's world position and rotation. */
  private place(attachment: Attachment, position: Vector3, rotation: Quaternion): void {
    const primitive = this.primitives[attachment.slot]!;
    primitive.rotation.copy(rotation).multiply(attachment.offset);
    const d = attachment.direction.clone().applyQuaternion(primitive.rotation);
    if (primitive.kind === KIND_PLANE) {
      primitive.data0 = [d.x, d.y, d.z, primitive.data0[3]];
      primitive.data1 = [position.x, position.y, position.z, primitive.data1[3]];
    } else if (primitive.kind === KIND_CAPSULE) {
      const radius = primitive.data0[3];
      const muS = primitive.data1[3];
      primitive.data0 = [position.x - d.x, position.y - d.y, position.z - d.z, radius];
      primitive.data1 = [position.x + d.x, position.y + d.y, position.z + d.z, muS];
    } else {
      primitive.data0 = [position.x, position.y, position.z, primitive.data0[3]];
    }
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('PrimitiveSet has been disposed');
  }

  private upload(gpu: PrimitiveBuffers): void {
    const packed = gpu.packed.value.array as Uint32Array;
    const data0 = gpu.data0.value.array as Float32Array;
    const data1 = gpu.data1.value.array as Float32Array;
    const velocity = gpu.velocity.value.array as Float32Array;
    this.primitives.forEach((primitive, slot) => {
      packed[slot] = (primitive.kind << 16) | primitive.flags;
      data0.set(primitive.data0, slot * 4);
      data1.set(primitive.data1, slot * 4);
      velocity.set(primitive.velocity, slot * 4);
      gpu.rotations[slot]!.copy(primitive.rotation);
      gpu.spins[slot]!.set(...primitive.spin, 0);
    });
    gpu.packed.value.needsUpdate = true;
    gpu.data0.value.needsUpdate = true;
    gpu.data1.value.needsUpdate = true;
    gpu.velocity.value.needsUpdate = true;
    gpu.count.value = this.primitives.length;
    this.dirty = false;
    this.changes++;
  }
}

function assertPositive(value: number, name: string): void {
  if (!(value > 0) || !Number.isFinite(value))
    throw new Error(`${name} must be positive, got ${value}`);
}

/**
 * Validate friction options against the defaults μ_s = 0.5, μ_k = 0.4.
 * `owner` prefixes the error message.
 */
export function resolveFriction(
  options: { readonly muS?: number; readonly muK?: number },
  owner = 'resolveFriction',
): {
  readonly muS: number;
  readonly muK: number;
} {
  const muS = options.muS ?? 0.5;
  const muK = options.muK ?? 0.4;
  if (!(muS >= 0) || !(muK >= 0)) {
    throw new Error(
      `${owner}: friction coefficients must be non-negative (got muS=${muS}, muK=${muK})`,
    );
  }
  return { muS, muK };
}

/** A normalized copy of `q`, which must be finite and non-zero. */
export function normalized(q: Quaternion, owner: string): Quaternion {
  const length = q.length();
  if (!(length > 0) || !Number.isFinite(length)) {
    throw new Error(`${owner}: rotation must be a finite, non-zero quaternion`);
  }
  return q.clone().normalize();
}
