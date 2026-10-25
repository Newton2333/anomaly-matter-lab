import { Fn, If, Return, atomicAdd, atomicLoad, float, instanceIndex } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { Accumulator } from '../accumulator.js';
import type { ParticleSystem } from '../particles.js';
import { LAMBDA_SCALE, type ContactBuffer } from './ContactBuffer.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Emit TSL that returns early for threads past the stored contacts. */
function emitSkipUnusedSlot(contacts: ContactBuffer, c: Any): void {
  If(c.greaterThanEqual(contacts.emitStoredCount()), () => {
    Return();
  });
}

/**
 * Solve every contact once, one thread per stored contact, scattering corrections
 * into `accumulator`:
 *
 * 1. Non-penetration (Macklin et al. 2014, eq. 22): push the pair apart along
 *    the contact normal, split by inverse mass.
 * 2. Static friction (Macklin et al. 2020, §3.5): cancel the pair's
 *    tangential slip this substep, but only while the tangential multiplier
 *    stays inside the friction cone `λ_t ≤ μ_s · λ_n`. The check is made
 *    against the correction about to be applied, because a parallel scatter
 *    cannot see other threads' same-iteration updates. Slip outside the cone
 *    is left to the kinetic friction pass.
 */
export function buildContactSolveKernel(args: {
  readonly particles: ParticleSystem;
  readonly contacts: ContactBuffer;
  readonly accumulator: Accumulator;
  readonly muS: UniformNode<'float', number>;
}): ComputeNode {
  const { particles, contacts, accumulator, muS } = args;
  const contactDistance = 2 * particles.particleRadius;

  return Fn(() => {
    const c: Any = instanceIndex;
    emitSkipUnusedSlot(contacts, c);
    const record: Any = contacts.records.element(c);
    const i: Any = record.get('i').toVar();
    const j: Any = record.get('j').toVar();

    const xiStar: Any = particles.predictedPositions.element(i).xyz.toVar();
    const xjStar: Any = particles.predictedPositions.element(j).xyz.toVar();
    const offset: Any = xiStar.sub(xjStar).toVar();
    const distance: Any = offset.length().toVar();
    const wi: Any = particles.invMass.element(i).toVar();
    const wj: Any = particles.invMass.element(j).toVar();
    const wSum: Any = wi.add(wj).toVar();
    If(
      distance
        .lessThanEqual(1e-8)
        .or(distance.greaterThanEqual(contactDistance))
        .or(wSum.lessThanEqual(0)),
      () => {
        Return();
      },
    );
    const n: Any = offset.div(distance).toVar();
    const depth: Any = float(contactDistance).sub(distance);

    // Normal correction, and the normal multiplier accumulated for friction.
    const dLambdaN: Any = depth.div(wSum).toVar();
    atomicAdd(record.get('lambdaN'), dLambdaN.mul(LAMBDA_SCALE).toInt());
    record.get('normal').assign(n);

    // Tangential slip since the start of the substep: Δp_t.
    const slip: Any = xiStar
      .sub(particles.positions.element(i).xyz)
      .sub(xjStar.sub(particles.positions.element(j).xyz))
      .toVar();
    const tangential: Any = slip.sub(n.mul(slip.dot(n))).toVar();
    const tangentialLength: Any = tangential.length().toVar();

    // λ_n here already includes this iteration's contribution.
    const lambdaN: Any = (atomicLoad(record.get('lambdaN')) as Any).toFloat().div(LAMBDA_SCALE);
    const lambdaT: Any = (atomicLoad(record.get('lambdaT')) as Any).toFloat().div(LAMBDA_SCALE);
    const dLambdaT: Any = tangentialLength.div(wSum).toVar();
    const sticks: Any = tangentialLength
      .greaterThan(1e-10)
      .and(dLambdaT.lessThanEqual(muS.mul(lambdaN).sub(lambdaT)));

    const normalI: Any = n.mul(wi.mul(dLambdaN));
    const normalJ: Any = n.mul(wj.mul(dLambdaN)).negate();
    const frictionI: Any = tangential.mul(wi.div(wSum)).negate();
    const frictionJ: Any = tangential.mul(wj.div(wSum));
    accumulator.add(i, sticks.select(normalI.add(frictionI), normalI));
    accumulator.add(j, sticks.select(normalJ.add(frictionJ), normalJ));
    atomicAdd(record.get('lambdaT'), sticks.select(dLambdaT, float(0)).mul(LAMBDA_SCALE).toInt());
  })()
    .compute(contacts.dispatchArgs as Any)
    .setName('solve.contactSolve');
}

/**
 * Separate pairs that already overlap at the start of the substep, moving
 * both the current and predicted positions (Macklin et al. 2014, §4.4). This
 * keeps initial overlaps from turning into velocity. Uses the same
 * accumulator; the caller applies it to both position buffers.
 */
export function buildContactStabilizeKernel(args: {
  readonly particles: ParticleSystem;
  readonly contacts: ContactBuffer;
  readonly accumulator: Accumulator;
}): ComputeNode {
  const { particles, contacts, accumulator } = args;
  const contactDistance = 2 * particles.particleRadius;

  return Fn(() => {
    const c: Any = instanceIndex;
    emitSkipUnusedSlot(contacts, c);
    const record: Any = contacts.records.element(c);
    const i: Any = record.get('i').toVar();
    const j: Any = record.get('j').toVar();

    const offset: Any = particles.positions
      .element(i)
      .xyz.sub(particles.positions.element(j).xyz)
      .toVar();
    const distance: Any = offset.length().toVar();
    const wi: Any = particles.invMass.element(i).toVar();
    const wj: Any = particles.invMass.element(j).toVar();
    const wSum: Any = wi.add(wj).toVar();
    If(
      distance
        .lessThanEqual(1e-8)
        .or(distance.greaterThanEqual(contactDistance))
        .or(wSum.lessThanEqual(0)),
      () => {
        Return();
      },
    );
    const n: Any = offset.div(distance);
    const dLambda: Any = float(contactDistance).sub(distance).div(wSum).toVar();
    accumulator.add(i, n.mul(wi.mul(dLambda)));
    accumulator.add(j, n.mul(wj.mul(dLambda)).negate());
  })()
    .compute(contacts.dispatchArgs as Any)
    .setName('solve.contactStabilize');
}

/**
 * Kinetic friction as a velocity change after the position solve (Macklin et
 * al. 2020, §3.6, eq. 30): reduce each pair's tangential relative velocity by
 * at most `μ_k · λ_n / dt`, split by inverse mass.
 */
export function buildContactFrictionKernel(args: {
  readonly particles: ParticleSystem;
  readonly contacts: ContactBuffer;
  readonly accumulator: Accumulator;
  readonly muK: UniformNode<'float', number>;
  readonly dt: UniformNode<'float', number>;
}): ComputeNode {
  const { particles, contacts, accumulator, muK, dt } = args;

  return Fn(() => {
    const c: Any = instanceIndex;
    emitSkipUnusedSlot(contacts, c);
    const record: Any = contacts.records.element(c);
    const i: Any = record.get('i').toVar();
    const j: Any = record.get('j').toVar();
    const wi: Any = particles.invMass.element(i).toVar();
    const wj: Any = particles.invMass.element(j).toVar();
    const wSum: Any = wi.add(wj).toVar();
    const lambdaN: Any = (atomicLoad(record.get('lambdaN')) as Any)
      .toFloat()
      .div(LAMBDA_SCALE)
      .toVar();
    If(wSum.lessThanEqual(0).or(lambdaN.lessThanEqual(0)), () => {
      Return();
    });

    const n: Any = record.get('normal').toVar();
    const v: Any = particles.velocities
      .element(i)
      .xyz.sub(particles.velocities.element(j).xyz)
      .toVar();
    const vT: Any = v.sub(n.mul(n.dot(v))).toVar();
    const vTLength: Any = vT.length().toVar();
    If(vTLength.lessThan(1e-6), () => {
      Return();
    });
    const change: Any = muK.mul(lambdaN).mul(wSum).div(dt).min(vTLength);
    const impulse: Any = vT.div(vTLength).mul(change).negate().div(wSum).toVar();
    accumulator.add(i, impulse.mul(wi));
    accumulator.add(j, impulse.mul(wj).negate());
  })()
    .compute(contacts.dispatchArgs as Any)
    .setName('solve.contactFriction');
}
