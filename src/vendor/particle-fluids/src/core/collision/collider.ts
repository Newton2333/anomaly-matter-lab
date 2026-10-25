import { If, float, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { Accumulator } from '../accumulator.js';
import type { ParticleSystem } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Solver state {@link SimLoop} passes to a collider when it builds its kernels. */
export interface ColliderContext {
  readonly particles: ParticleSystem;
  /** Substep length in seconds. */
  readonly dt: UniformNode<'float', number>;
  /** Substeps per frame. */
  readonly substeps: UniformNode<'float', number>;
  /** Position corrections, applied after every solver iteration. */
  readonly positions: Accumulator;
  /** Velocity corrections, applied once per substep after friction. */
  readonly velocities: Accumulator;
}

/** Kernels a collider adds to each substep. */
export interface ColliderKernels {
  readonly frameStart?: readonly ComputeNode[];
  readonly preSolve: readonly ComputeNode[];
  readonly solve: readonly ComputeNode[];
  readonly postSolve: readonly ComputeNode[];
  readonly substepEnd?: readonly ComputeNode[];
}

/** Static or moving geometry that particles collide with. See {@link PrimitiveSet} and {@link SDFCollider}. */
export interface Collider {
  readonly particles: ParticleSystem;
  /** Called by {@link SimLoop.step} before every step with the step length. */
  update(dt: number): void;
  /** @internal Called once by {@link SimLoop}. */
  buildKernels(context: ColliderContext): ColliderKernels;
  dispose(): void;
}

/**
 * Emit TSL for one particle touching one collider (Macklin et al. 2020 §3.5,
 * as for particle contacts): push the particle out along `normal` by `depth`,
 * and cancel its slip relative to the collider while the accumulated
 * tangential multiplier stays inside the static friction cone.
 *
 * `lambda` holds `(λ_n, λ_t)` for this particle–collider pair at
 * `lambdaIndex` and `lambdaIndex + 1`, reset every substep.
 */
export function emitColliderContact(args: {
  readonly particle: Any;
  readonly invMass: Any;
  /** Particle displacement since the start of the substep. */
  readonly displacement: Any;
  /** Collider displacement over the substep, or omit for a static collider. */
  readonly colliderDisplacement?: Any;
  /** Unit normal pointing out of the collider. */
  readonly normal: Any;
  /** Penetration depth, > 0. */
  readonly depth: Any;
  readonly muS: Any;
  readonly lambda: StorageBufferNode<'float'>;
  readonly lambdaIndex: Any;
  readonly accumulator: Accumulator;
}): void {
  const { particle, invMass, normal, depth, muS, lambda, lambdaIndex, accumulator } = args;
  const lambdaN: Any = lambda.element(lambdaIndex).add(depth.div(invMass)).toVar();
  lambda.element(lambdaIndex).assign(lambdaN);

  const slip: Any = args.colliderDisplacement
    ? args.displacement.sub(args.colliderDisplacement)
    : args.displacement;
  const tangential: Any = slip.sub(normal.mul(slip.dot(normal))).toVar();
  const tangentialLength: Any = tangential.length().toVar();
  const lambdaTIndex: Any = lambdaIndex.add(uint(1));
  const lambdaT: Any = lambda.element(lambdaTIndex).toVar();
  const sticks: Any = tangentialLength
    .greaterThan(1e-10)
    .and(tangentialLength.div(invMass).lessThanEqual(muS.mul(lambdaN).sub(lambdaT)));

  const push: Any = normal.mul(depth);
  accumulator.add(particle, sticks.select(push.sub(tangential), push));
  lambda
    .element(lambdaTIndex)
    .assign(lambdaT.add(sticks.select(tangentialLength.div(invMass), float(0))));
}

/**
 * Emit TSL for kinetic friction against a collider (Macklin et al. 2020 §3.6):
 * reduce the particle's tangential velocity relative to the collider by at
 * most `μ_k · λ_n · w / dt`.
 */
export function emitColliderFriction(args: {
  readonly particle: Any;
  readonly invMass: Any;
  readonly velocity: Any;
  /** Omit for a static collider. */
  readonly colliderVelocity?: Any;
  readonly normal: Any;
  readonly lambdaN: Any;
  readonly muK: Any;
  readonly dt: Any;
  readonly accumulator: Accumulator;
}): void {
  const { particle, invMass, normal, lambdaN, muK, dt, accumulator } = args;
  const relative: Any = args.colliderVelocity
    ? args.velocity.sub(args.colliderVelocity)
    : args.velocity;
  const vT: Any = relative.sub(normal.mul(normal.dot(relative))).toVar();
  const vTLength: Any = vT.length().toVar();
  If(vTLength.greaterThan(1e-6), () => {
    const change: Any = muK.mul(lambdaN).mul(invMass).div(dt).min(vTLength);
    accumulator.add(particle, vT.div(vTLength).mul(change).negate());
  });
}
