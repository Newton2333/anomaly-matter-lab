import { Vector3 } from 'three';
import {
  Fn,
  If,
  Loop,
  instanceIndex,
  instancedArray,
  mix,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import {
  assertRange,
  constraintKernels,
  type ConstraintType,
  type Material,
  type MaterialKernels,
  type ParticleRange,
  type ParticleSystem,
  type SolverContext,
} from '../core/index.js';
import { createClothAeroKernel } from './aero.js';
import { createClothBendingConstraints } from './bending.js';
import { createClothDistanceConstraints } from './distance.js';
import type { ClothGraph } from './graph.js';
import { createClothTetherConstraints } from './tether.js';
import { buildTethers, type TetherConstraint } from './tetherBuild.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface ClothSystemOptions {
  /** Shape, masses, and pins; see {@link createClothGraph}. */
  readonly graph: ClothGraph;
  /** First particle slot the cloth occupies. Default 0. */
  readonly offset?: number;
  /** Stretch compliance in s²/kg. Default 1e-7 (barely stretches). */
  readonly stretchCompliance?: number;
  /**
   * Bending compliance in rad²/(N·m). Default 1e-5. Higher values drape
   * more loosely.
   */
  readonly bendCompliance?: number;
  /**
   * Compliance of the long-range attachments that stop pinned cloth from
   * stretching under its own weight (Kim et al. 2012). Default 0.
   */
  readonly tetherCompliance?: number;
  /** How far past its rest distance a particle may drift from its pins, as a fraction ≥ 0. Default 0. */
  readonly stretchTolerance?: number;
  /** Wind velocity in m/s. Default none. */
  readonly wind?: Vector3;
  /** Air drag, `½ · C_D · ρ_air` in kg/m³. Default 0.6125. */
  readonly drag?: number;
  /** Air lift, `½ · C_L · ρ_air` in kg/m³. Default 0.3. */
  readonly lift?: number;
  /**
   * Blend each particle's velocity toward its neighbors' after every
   * substep, from 0 (off) to 1. Calms high-frequency ripples, such as
   * stretched cloth chattering against a collider, without slowing the
   * cloth's overall motion. Default off. The smoothing pass is only built
   * when this option is given, so pass `0` to change it later.
   */
  readonly damping?: number;
}

/**
 * Cloth made of particles joined by stretch, bending (Bergou et al. 2006),
 * and long-range attachment constraints (Kim et al. 2012), with aerodynamic
 * drag and lift per triangle (Keckeisen et al. 2004).
 *
 * The cloth writes its particles' positions and masses into the
 * {@link ParticleSystem} from its graph, so there is no need to upload them.
 * Their velocities are set to zero; their collision groups are kept.
 *
 * ```ts
 * const graph = createClothGraph(new PlaneGeometry(1, 1, 40, 40), { pinnedIndices: [0, 40] });
 * const particles = new ParticleSystem(renderer, graph.positions.length, 0.01);
 * const cloth = new ClothSystem(particles, { graph, wind: new Vector3(0, 0, 2) });
 * ```
 */
export class ClothSystem implements Material {
  readonly particles: ParticleSystem;
  readonly graph: ClothGraph;
  readonly range: ParticleRange;
  /** Long-range attachments built from the pins, with cloth-local particle indices. */
  readonly tethers: readonly TetherConstraint[];

  private readonly options: ClothSystemOptions;
  private readonly windUniform: UniformNode<'vec3', Vector3>;
  private readonly dragUniform: UniformNode<'float', number>;
  private readonly liftUniform: UniformNode<'float', number>;
  private readonly dampingUniform: UniformNode<'float', number>;
  private bendComplianceValue: number;
  private bending: ConstraintType | undefined;

  constructor(particles: ParticleSystem, options: ClothSystemOptions) {
    const { graph } = options;
    this.range = { start: options.offset ?? 0, count: graph.positions.length };
    assertRange(particles, this.range, 'ClothSystem');
    for (const key of [
      'stretchCompliance',
      'bendCompliance',
      'tetherCompliance',
      'stretchTolerance',
      'drag',
      'lift',
    ] as const) {
      const value = options[key];
      if (value !== undefined) nonNegative(value, key);
    }
    if (options.damping !== undefined) unitInterval(options.damping, 'damping');
    this.particles = particles;
    this.graph = graph;
    this.options = options;
    this.bendComplianceValue = options.bendCompliance ?? 1e-5;
    this.windUniform = uniform((options.wind ?? new Vector3()).clone());
    this.dragUniform = uniform(options.drag ?? 0.6125, 'float');
    this.liftUniform = uniform(options.lift ?? 0.3, 'float');
    this.dampingUniform = uniform(options.damping ?? 0, 'float');
    this.tethers = buildTethers({
      graph,
      options: { stretchTolerance: options.stretchTolerance ?? 0 },
    });

    // Keep any collision group already set on the range.
    const groups = particles.collisionGroup.value.array as Uint32Array;
    particles.uploadParticles(
      graph.positions.map((position, i) => ({
        position,
        invMass: graph.invMass[i]!,
        collisionGroup: groups[this.range.start + i]!,
      })),
      this.range.start,
    );
  }

  /** Wind velocity in m/s. Mutate it to change the wind. */
  get wind(): Vector3 {
    return this.windUniform.value;
  }

  /** Air drag, `½ · C_D · ρ_air` in kg/m³. */
  get drag(): number {
    return this.dragUniform.value;
  }
  set drag(value: number) {
    this.dragUniform.value = nonNegative(value, 'drag');
  }

  /** Air lift, `½ · C_L · ρ_air` in kg/m³. */
  get lift(): number {
    return this.liftUniform.value;
  }
  set lift(value: number) {
    this.liftUniform.value = nonNegative(value, 'lift');
  }

  /**
   * Velocity smoothing, 0 to 1. `0` when the `damping` option was not
   * given; setting it then throws, since the smoothing pass was not built.
   */
  get damping(): number {
    return this.dampingUniform.value;
  }
  set damping(value: number) {
    if (this.options.damping === undefined) {
      throw new Error('ClothSystem: pass `damping` in the options to enable it before changing it');
    }
    this.dampingUniform.value = unitInterval(value, 'damping');
  }

  /** Bending compliance. Changes take effect on the next step. */
  get bendCompliance(): number {
    return this.bendComplianceValue;
  }
  set bendCompliance(value: number) {
    this.bendComplianceValue = nonNegative(value, 'bendCompliance');
    if (this.bending) {
      (this.bending.compliance.value.array as Float32Array).fill(value);
      this.bending.compliance.value.needsUpdate = true;
    }
  }

  build({ particles, dt }: SolverContext): MaterialKernels {
    const { graph, options } = this;
    const offset = this.range.start;
    const constraints: ConstraintType[] = [
      createClothDistanceConstraints({
        particles,
        particleOffset: offset,
        edges: graph.distancePairs,
        restLengths: graph.distanceRestLengths,
        compliance: options.stretchCompliance ?? 1e-7,
        dt,
      }),
    ];
    if (graph.bendingTuples.length > 0) {
      this.bending = createClothBendingConstraints({
        particles,
        particleOffset: offset,
        tuples: graph.bendingTuples,
        restAngles: graph.bendingRestAngles,
        compliance: this.bendComplianceValue,
        dt,
      });
      constraints.push(this.bending);
    }
    const { tethers } = this;
    if (tethers.length > 0) {
      constraints.push(
        createClothTetherConstraints({
          particles,
          particleOffset: offset,
          tethers,
          compliance: options.tetherCompliance ?? 0,
          dt,
        }),
      );
    }
    const { preSolve, solve } = constraintKernels(constraints);
    // Wind acts on the predicted positions before the constraints solve.
    preSolve.push(
      createClothAeroKernel({
        particles,
        particleOffset: offset,
        nClothParticles: this.range.count,
        triangles: graph.triangles,
        dt,
        wind: this.windUniform,
        dragCoeff: this.dragUniform,
        liftCoeff: this.liftUniform,
      }),
    );
    return {
      preSolve,
      solve,
      postSolve: options.damping === undefined ? [] : this.buildDamping(particles),
    };
  }

  /** Blend velocities toward the average of each particle's edge neighbors. */
  private buildDamping(particles: ParticleSystem): ComputeNode[] {
    const { graph, range } = this;
    const count = range.count;
    const degree = new Uint32Array(count);
    for (const [i, j] of graph.distancePairs) {
      degree[i]!++;
      degree[j]!++;
    }
    const offsets = new Uint32Array(count + 1);
    for (let i = 0; i < count; i++) offsets[i + 1] = offsets[i]! + degree[i]!;
    const neighbors = new Uint32Array(Math.max(1, offsets[count]!));
    const cursor = offsets.slice(0, count);
    for (const [i, j] of graph.distancePairs) {
      neighbors[cursor[i]!++] = j;
      neighbors[cursor[j]!++] = i;
    }
    const offsetBuffer = instancedArray(offsets, 'uint');
    const neighborBuffer = instancedArray(neighbors, 'uint');
    const smoothed = instancedArray(count, 'vec4');

    const average = Fn(() => {
      const local: Any = instanceIndex;
      const v: Any = particles.velocities.element(local.add(uint(range.start)));
      const sum: Any = vec3(0).toVar();
      const from: Any = offsetBuffer.element(local);
      const to: Any = offsetBuffer.element(local.add(uint(1)));
      Loop({ start: from, end: to, type: 'uint', condition: '<' }, ({ i }: { i: Any }) => {
        sum.addAssign(
          particles.velocities.element(neighborBuffer.element(i).add(uint(range.start))).xyz,
        );
      });
      const mean: Any = sum.div(to.sub(from).max(uint(1)).toFloat());
      smoothed.element(local).assign(vec4(mix(v.xyz, mean, this.dampingUniform), v.w));
    })()
      .compute(count)
      .setName('ClothSystem.average');
    const apply = Fn(() => {
      const i: Any = instanceIndex.add(uint(range.start));
      If(particles.invMass.element(i).greaterThan(0), () => {
        particles.velocities.element(i).assign(smoothed.element(instanceIndex));
      });
    })()
      .compute(count)
      .setName('ClothSystem.apply');
    return [average, apply];
  }
}

function nonNegative(value: number, name: string): number {
  if (!(value >= 0) || !Number.isFinite(value)) {
    throw new Error(`ClothSystem: ${name} must be ≥ 0, got ${value}`);
  }
  return value;
}

function unitInterval(value: number, name: string): number {
  if (!(value >= 0 && value <= 1)) {
    throw new Error(`ClothSystem: ${name} must be between 0 and 1, got ${value}`);
  }
  return value;
}
