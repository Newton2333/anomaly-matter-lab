import { instancedArray, uniform } from 'three/tsl';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import {
  Accumulator,
  assertRange,
  type Material,
  type MaterialKernels,
  type ParticleRange,
  type ParticleSystem,
  type SolverContext,
} from '../core/index.js';
import { buildRotationKernels } from './rotation.js';
import {
  buildCenterOfMassKernel,
  buildMomentAndPolarDecompKernel,
  buildResetLambdaKernel,
  buildShapeMatchDeltaApplyKernel,
} from './shapeMatch.js';
import {
  buildImplicitMomentPolarKernel,
  buildImplicitNeighborhoodCenterKernel,
  buildImplicitQpWriteKernel,
  buildImplicitResetPairLambdaKernel,
  buildImplicitShapeMatchScatterKernel,
} from './shapeMatchImplicit.js';

/** One soft body: a block of particles that tries to keep its rest shape. */
export interface SoftbodyDef {
  /** The body's particles. Surface particles must come first. */
  readonly range: ParticleRange;
  /**
   * How many of the body's first particles lie on its surface. Fluids push
   * on these (see {@link SoftbodySystem.surfaceRange}). Default: all of them.
   */
  readonly surfaceCount?: number;
  /**
   * Shape-matching compliance in s²/kg; 0 is rigid. Default 0.
   *
   * Every particle carries its own constraint, so at a fixed compliance a
   * body with more particles is stiffer: scale compliance roughly with the
   * particle count to keep the same feel. Global matching stays nearly rigid
   * at any compliance and mostly controls how much the body wobbles, so try
   * 1e-7 to 1e-5. Local matching is where compliance shapes the material:
   * about `1e-6 · count / 200` is firm rubber and `1e-3 · count / 200` loose
   * jelly (the mapping {@link Simulation.addSoftbody} uses).
   */
  readonly compliance?: number;
  /**
   * Rest shape, xyz per particle. Default: the positions the particles were
   * uploaded with, so upload them before creating the system.
   */
  readonly restPositions?: Float32Array;
  /**
   * Pairs of neighboring particles `[i0, j0, i1, j1, …]`, as indices local to
   * the body. Required for local shape matching; under global matching they
   * only guide {@link SoftbodyMesh} binding. {@link voxelize} produces them.
   */
  readonly edges?: Uint32Array;
}

export interface SoftbodySystemOptions {
  readonly bodies: readonly SoftbodyDef[];
  /**
   * - `'global'` (default): each body matches its rest shape as a whole.
   *   Cheap and stiff; bodies wobble but don't bend much.
   * - `'local'`: every particle matches the shape of its own neighborhood
   *   (Müller & Chentanez 2011, §5.1), so bodies bend and fold. Needs `edges`.
   */
  readonly shapeMatching?: 'global' | 'local';
  /** Let a body's own particles collide with each other, e.g. so a ring can't fold through itself. Default `false`. */
  readonly selfCollision?: boolean;
}

/** A body's settings after defaults are applied. */
export interface SoftbodyBody {
  readonly range: ParticleRange;
  readonly surfaceCount: number;
  readonly restPositions: Float32Array;
  /** Center of mass of the rest shape. */
  readonly restCenter: readonly [number, number, number];
  readonly edges: Uint32Array | undefined;
}

/**
 * Soft and near-rigid bodies made of particles, simulated with shape
 * matching (Müller et al. 2005; Müller & Chentanez 2011) as XPBD
 * constraints. Add it to a {@link SimLoop}'s `materials`; enable the loop's
 * `contact` option so bodies collide with each other.
 *
 * Global shape matching weights particles by the masses they have when the
 * system is created, so a body with a heavy base settles base-down in water.
 * These weights (also used for {@link bodyCenters}) and the rest centers are
 * fixed at construction: a later {@link ParticleSystem.setInvMass} changes
 * how the particles move but not how shape matching weighs them.
 *
 * ```ts
 * const shape = voxelize(mesh, { particleRadius });
 * particles.uploadParticles(pointsOf(shape));
 * const bodies = new SoftbodySystem(particles, {
 *   bodies: [{ range: { start: 0, count: shape.count }, surfaceCount: shape.surfaceCount, compliance: 1e-6 }],
 * });
 * ```
 */
export class SoftbodySystem implements Material {
  readonly particles: ParticleSystem;
  readonly bodies: readonly SoftbodyBody[];
  readonly shapeMatching: 'global' | 'local';
  /**
   * Best-fit rotation of each body relative to its rest shape, as three row
   * vectors: rows `3b`, `3b + 1`, `3b + 2` belong to body `b`. Identity
   * until the first step. Under local matching the body can bend, so this is
   * the rotation that best fits the whole body.
   */
  readonly bodyRotations: StorageBufferNode<'vec4'>;
  /** Current mass-weighted center of each body in `.xyz`. Zero until the first step. */
  readonly bodyCenters: StorageBufferNode<'vec4'>;
  /**
   * @internal Rest position of each particle relative to its body's rest
   * center; `w` holds the body's compliance.
   */
  readonly restOffsets: StorageBufferNode<'vec4'>;

  private readonly defs: readonly SoftbodyDef[];
  private readonly selfCollision: boolean;
  private readonly bodyCompliance: StorageBufferNode<'float'>;
  /** Each particle's mass relative to the average of its body. */
  private readonly weights: StorageBufferNode<'float'>;
  /** Smallest range covering every body. */
  private readonly span: ParticleRange;

  constructor(particles: ParticleSystem, options: SoftbodySystemOptions) {
    const { bodies } = options;
    if (bodies.length === 0) throw new Error('SoftbodySystem: at least one body is required');
    this.particles = particles;
    this.defs = bodies;
    this.shapeMatching = options.shapeMatching ?? 'global';
    this.selfCollision = options.selfCollision ?? false;

    this.restOffsets = instancedArray(particles.capacity, 'vec4');
    this.bodyCenters = instancedArray(bodies.length, 'vec4');
    this.bodyRotations = instancedArray(bodies.length * 3, 'vec4');
    this.bodyCompliance = instancedArray(bodies.length, 'float');
    // Start every body at the identity rotation, which skinning reads before the first step.
    const rotations = this.bodyRotations.value.array as Float32Array;
    for (let b = 0; b < bodies.length; b++) {
      rotations[b * 12 + 0] = rotations[b * 12 + 5] = rotations[b * 12 + 10] = 1;
    }

    const uploaded = particles.positions.value.array as Float32Array;
    const invMass = particles.invMass.value.array as Float32Array;
    const rest = this.restOffsets.value.array as Float32Array;
    const weights = new Float32Array(particles.capacity);
    const used = new Uint8Array(particles.capacity);
    const resolved: SoftbodyBody[] = [];
    let spanStart = Infinity;
    let spanEnd = 0;
    bodies.forEach((body, b) => {
      const { range } = body;
      if (range.count === 0) throw new Error(`SoftbodySystem: body ${b} has no particles`);
      assertRange(particles, range, `SoftbodySystem body ${b}`);
      for (let i = range.start; i < range.start + range.count; i++) {
        if (used[i]) throw new Error(`SoftbodySystem: particle ${i} belongs to two bodies`);
        used[i] = 1;
      }
      const restPositions =
        body.restPositions ??
        Float32Array.from({ length: range.count * 3 }, (_, k) => {
          return uploaded[(range.start + Math.floor(k / 3)) * 4 + (k % 3)]!;
        });
      if (restPositions.length !== range.count * 3) {
        throw new Error(`SoftbodySystem: body ${b} needs ${range.count * 3} rest coordinates`);
      }
      const surfaceCount = body.surfaceCount ?? range.count;
      if (!Number.isInteger(surfaceCount) || surfaceCount < 0 || surfaceCount > range.count) {
        throw new Error(`SoftbodySystem: body ${b} surfaceCount ${surfaceCount} is out of range`);
      }
      if (body.edges) assertEdges(body.edges, range.count, b);
      else if (this.shapeMatching === 'local') {
        throw new Error(`SoftbodySystem: body ${b} needs edges for local shape matching`);
      }
      const bodyWeights = massWeights(invMass.subarray(range.start, range.start + range.count));
      weights.set(bodyWeights, range.start);
      const center = centroid(restPositions, bodyWeights);
      assertVolumetric(restPositions, center, particles.particleRadius, b);
      for (let i = 0; i < range.count; i++) {
        rest[(range.start + i) * 4 + 0] = restPositions[i * 3]! - center[0];
        rest[(range.start + i) * 4 + 1] = restPositions[i * 3 + 1]! - center[1];
        rest[(range.start + i) * 4 + 2] = restPositions[i * 3 + 2]! - center[2];
      }
      resolved.push({ range, surfaceCount, restPositions, restCenter: center, edges: body.edges });
      spanStart = Math.min(spanStart, range.start);
      spanEnd = Math.max(spanEnd, range.start + range.count);
      this.writeCompliance(b, body.compliance ?? 0);
    });
    this.bodies = resolved;
    this.span = { start: spanStart, count: spanEnd - spanStart };
    this.weights = instancedArray(weights, 'float');
  }

  /** The particles of body `index`. */
  particleRange(index: number): ParticleRange {
    return this.body(index).range;
  }

  /** The surface particles of body `index`, e.g. for {@link FluidSystem.addBoundary}. */
  surfaceRange(index: number): ParticleRange {
    const body = this.body(index);
    return { start: body.range.start, count: body.surfaceCount };
  }

  /** Change a body's shape-matching compliance. Takes effect on the next step. */
  setCompliance(index: number, compliance: number): void {
    this.body(index);
    this.writeCompliance(index, compliance);
  }

  build({ particles, dt, allocateCollisionGroup }: SolverContext): MaterialKernels {
    if (!this.selfCollision) {
      const groups = particles.collisionGroup.value.array as Uint32Array;
      this.bodies.forEach((body, b) => {
        const { start, count } = body.range;
        const group = groups[start]!;
        const shared = groups.subarray(start, start + count).every((g) => g === group);
        // A group the user gave the whole body already keeps it from colliding with itself.
        if (shared && group !== 0) return;
        if (!shared) {
          throw new Error(
            `SoftbodySystem: body ${b} has mixed collision groups; give all its particles one group, or none`,
          );
        }
        particles.setCollisionGroup(body.range, allocateCollisionGroup());
      });
    }
    return this.shapeMatching === 'global' ? this.buildGlobal(dt) : this.buildLocal(dt);
  }

  /** Kernels that fit each body's center and rotation to its current particles. */
  private buildBodyFrameKernels() {
    const { particles, bodies, weights } = this;
    const bodyStart = instancedArray(new Uint32Array(bodies.map((b) => b.range.start)), 'uint');
    const bodyCount = instancedArray(new Uint32Array(bodies.map((b) => b.range.count)), 'uint');
    const shared = { particles, bodyStart, bodyCount, numBodies: bodies.length };
    const center = buildCenterOfMassKernel({ ...shared, weights, bodyCenters: this.bodyCenters });
    const rotation = buildMomentAndPolarDecompKernel({
      ...shared,
      weights,
      restOffsets: this.restOffsets,
      bodyRotations: this.bodyRotations,
    });
    return { shared, center, rotation };
  }

  private buildGlobal(dt: SolverContext['dt']): MaterialKernels {
    const { particles } = this;
    const lambda = instancedArray(particles.capacity, 'vec4');
    const { shared, center, rotation } = this.buildBodyFrameKernels();
    return {
      preSolve: [buildResetLambdaKernel({ particles, lambda })],
      // Refit the body frame every iteration so other constraints' pushes carry the body along.
      solve: [
        center,
        rotation,
        buildShapeMatchDeltaApplyKernel({
          ...shared,
          restOffsets: this.restOffsets,
          bodyCenters: this.bodyCenters,
          bodyRotations: this.bodyRotations,
          bodyCompliance: this.bodyCompliance,
          lambda,
          dt,
        }),
      ],
    };
  }

  private buildLocal(dt: SolverContext['dt']): MaterialKernels {
    const { particles, bodies, span } = this;
    const graph = buildNeighborGraph(this.defs, particles.capacity);
    const rest = this.restOffsets.value.array as Float32Array;

    // Rest centroid of each particle's neighborhood (the particle and its edge neighbors).
    const restCenters = instancedArray(particles.capacity, 'vec4');
    const restCentersArray = restCenters.value.array as Float32Array;
    for (const body of bodies) {
      for (let i = body.range.start; i < body.range.start + body.range.count; i++) {
        const [from, to] = [graph.offsets[i]!, graph.offsets[i + 1]!];
        for (let k = from; k < to; k++) {
          const j = graph.indices[k]!;
          for (let axis = 0; axis < 3; axis++) {
            restCentersArray[i * 4 + axis]! += rest[j * 4 + axis]! / (to - from);
          }
        }
      }
    }

    const neighborOffsets = instancedArray(graph.offsets, 'uint');
    const neighborIndices = instancedArray(graph.indices, 'uint');
    const particleCenters = instancedArray(particles.capacity, 'vec4');
    const particleRotations = instancedArray(3 * particles.capacity, 'vec4');
    const pairLambda = instancedArray(graph.indices.length, 'vec4');
    const accumulator = new Accumulator(particles, 10, 'shapeMatchCorrections');
    const shared = {
      particles,
      range: span,
      neighborOffsets,
      neighborIndices,
      particleCenters,
    };
    const neighborhood = {
      ...shared,
      restOffsets: this.restOffsets,
      restNeighborhoodCenters: restCenters,
      particleRotations,
    };
    const center = buildImplicitNeighborhoodCenterKernel(shared);
    const rotation = buildImplicitMomentPolarKernel({
      ...neighborhood,
      aiScalar: uniform((particles.particleRadius * particles.particleRadius) / 5, 'float'),
    });
    const orientation = buildRotationKernels(particles, span, dt);
    // Not used by the solve; keeps bodyCenters and bodyRotations current for callers.
    const frame = this.buildBodyFrameKernels();

    return {
      preSolve: [
        orientation.predict,
        buildImplicitResetPairLambdaKernel({ pairLambda, totalDegree: graph.indices.length }),
      ],
      solve: [
        center,
        rotation,
        buildImplicitShapeMatchScatterKernel({ ...neighborhood, pairLambda, accumulator, dt }),
        accumulator.buildApplyKernel([particles.predictedPositions], span),
        buildImplicitQpWriteKernel({ particles, range: span, particleRotations, neighborOffsets }),
      ],
      postSolve: [orientation.advect, frame.center, frame.rotation],
    };
  }

  private body(index: number): SoftbodyBody {
    const body = this.bodies[index];
    if (!body) throw new Error(`SoftbodySystem: no body ${index} (have ${this.bodies.length})`);
    return body;
  }

  private writeCompliance(index: number, compliance: number): void {
    if (!(compliance >= 0) || !Number.isFinite(compliance)) {
      throw new Error(`SoftbodySystem: compliance must be ≥ 0, got ${compliance}`);
    }
    const range = this.defs[index]!.range;
    (this.bodyCompliance.value.array as Float32Array)[index] = compliance;
    const rest = this.restOffsets.value.array as Float32Array;
    for (let i = range.start; i < range.start + range.count; i++) rest[i * 4 + 3] = compliance;
    this.bodyCompliance.value.needsUpdate = true;
    this.restOffsets.value.needsUpdate = true;
  }
}

/**
 * Each particle's mass relative to the average of its body, from the body's
 * inverse masses. Pinned particles (inverse mass 0) count as average, and a
 * body of equal masses gets weight 1 throughout.
 */
function massWeights(invMass: Float32Array): Float32Array {
  const masses = Array.from(invMass, (w) => (w > 0 ? 1 / w : 0));
  const dynamic = masses.filter((m) => m > 0);
  if (dynamic.every((m) => m === dynamic[0])) return new Float32Array(invMass.length).fill(1);
  const mean = dynamic.reduce((sum, m) => sum + m, 0) / dynamic.length;
  return Float32Array.from(masses, (m) => (m > 0 ? m / mean : 1));
}

function centroid(positions: Float32Array, weights: Float32Array): [number, number, number] {
  let x = 0,
    y = 0,
    z = 0,
    total = 0;
  for (let i = 0; i < weights.length; i++) {
    x += positions[i * 3]! * weights[i]!;
    y += positions[i * 3 + 1]! * weights[i]!;
    z += positions[i * 3 + 2]! * weights[i]!;
    total += weights[i]!;
  }
  return [x / total, y / total, z / total];
}

/**
 * Shape matching needs a rotation from the particles' spread, which is only
 * defined for a volume: reject flat or collinear rest shapes.
 */
function assertVolumetric(
  positions: Float32Array,
  center: readonly number[],
  particleRadius: number,
  body: number,
): void {
  let xx = 0,
    xy = 0,
    xz = 0,
    yy = 0,
    yz = 0,
    zz = 0;
  for (let i = 0; i < positions.length / 3; i++) {
    const x = positions[i * 3]! - center[0]!;
    const y = positions[i * 3 + 1]! - center[1]!;
    const z = positions[i * 3 + 2]! - center[2]!;
    xx += x * x;
    xy += x * y;
    xz += x * z;
    yy += y * y;
    yz += y * z;
    zz += z * z;
  }
  const smallest = Math.sqrt(Math.max(0, smallestEigenvalue([xx, xy, xz, yy, yz, zz])));
  if (smallest < particleRadius * 1e-3) {
    throw new Error(
      `SoftbodySystem: body ${body} is flat; shape matching needs a 3D particle cloud`,
    );
  }
}

/** Smallest eigenvalue of a symmetric 3×3 matrix (upper triangle xx, xy, xz, yy, yz, zz), by Jacobi rotations. */
function smallestEigenvalue(m: [number, number, number, number, number, number]): number {
  let [a00, a01, a02, a11, a12, a22] = m;
  for (let sweep = 0; sweep < 30; sweep++) {
    const off = Math.abs(a01) + Math.abs(a02) + Math.abs(a12);
    if (off < 1e-14) break;
    // Zero the largest off-diagonal entry.
    const pivot =
      Math.abs(a01) >= Math.abs(a02) && Math.abs(a01) >= Math.abs(a12)
        ? 0
        : Math.abs(a02) >= Math.abs(a12)
          ? 1
          : 2;
    const [apq, app, aqq] =
      pivot === 0 ? [a01, a00, a11] : pivot === 1 ? [a02, a00, a22] : [a12, a11, a22];
    if (Math.abs(apq) < 1e-20) break;
    const theta = (aqq - app) / (2 * apq);
    const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;
    if (pivot === 0) {
      [a00, a11, a01, a02, a12] = [
        app - t * apq,
        aqq + t * apq,
        0,
        c * a02 - s * a12,
        s * a02 + c * a12,
      ];
    } else if (pivot === 1) {
      [a00, a22, a02, a01, a12] = [
        app - t * apq,
        aqq + t * apq,
        0,
        c * a01 - s * a12,
        s * a01 + c * a12,
      ];
    } else {
      [a11, a22, a12, a01, a02] = [
        app - t * apq,
        aqq + t * apq,
        0,
        c * a01 - s * a02,
        s * a01 + c * a02,
      ];
    }
  }
  return Math.min(a00, a11, a22);
}

function assertEdges(edges: Uint32Array, count: number, body: number): void {
  if (edges.length % 2 !== 0) {
    throw new Error(`SoftbodySystem: body ${body} has an odd number of edge indices`);
  }
  for (let e = 0; e < edges.length; e += 2) {
    const [i, j] = [edges[e]!, edges[e + 1]!];
    if (!(i < count) || !(j < count) || i === j) {
      throw new Error(`SoftbodySystem: body ${body} has an invalid edge (${i}, ${j})`);
    }
  }
}

/**
 * Compressed neighbor lists over global particle indices: row `i` holds `i`
 * itself followed by its edge neighbors (Müller & Chentanez 2011, §5.1).
 * Edges are validated by the constructor.
 */
function buildNeighborGraph(
  bodies: readonly SoftbodyDef[],
  capacity: number,
): { offsets: Uint32Array; indices: Uint32Array } {
  const degree = new Uint32Array(capacity);
  for (const { range, edges } of bodies) {
    for (let i = 0; i < range.count; i++) degree[range.start + i]! += 1;
    for (let e = 0; e < edges!.length; e += 2) {
      degree[range.start + edges![e]!]! += 1;
      degree[range.start + edges![e + 1]!]! += 1;
    }
  }
  const offsets = new Uint32Array(capacity + 1);
  for (let i = 0; i < capacity; i++) offsets[i + 1] = offsets[i]! + degree[i]!;
  const indices = new Uint32Array(Math.max(1, offsets[capacity]!));
  const cursor = offsets.slice(0, capacity);
  for (const { range, edges } of bodies) {
    for (let i = range.start; i < range.start + range.count; i++) indices[cursor[i]!++] = i;
    for (let e = 0; e < edges!.length; e += 2) {
      const [i, j] = [range.start + edges![e]!, range.start + edges![e + 1]!];
      indices[cursor[i]!++] = j;
      indices[cursor[j]!++] = i;
    }
  }
  return { offsets, indices };
}
