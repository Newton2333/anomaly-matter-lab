import { Fn, If, atan, cos, float, instanceIndex, length, sin, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleRange, ParticleSystem } from '../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Below this angular speed a substep's rotation is under f32 quaternion precision. */
const OMEGA_EPS = 1e-8;
const ANGLE_HALF_EPS = 1e-8;
/**
 * Cap on committed angular speed (rad/s). A noisy rotation from an
 * ill-conditioned polar decomposition would otherwise become a huge ω that
 * the next substep applies as an over-rotation.
 */
const OMEGA_MAX = 100;

/**
 * Integrate per-particle orientation for oriented particles (Müller &
 * Chentanez 2011, §4.1).
 *
 * - `predict` (eq. 11): `qp = rotate(ω, dt) · q`.
 * - `advect` (eqs. 12–15): `ω = axis(r) · angle(r) / dt` with `r = qp · q⁻¹`,
 *   taking the shorter arc, then `q = qp`.
 */
export function buildRotationKernels(
  particles: ParticleSystem,
  range: ParticleRange,
  dt: UniformNode<'float', number>,
): { predict: ComputeNode; advect: ComputeNode } {
  const predict = Fn(() => {
    const i: Any = instanceIndex.add(range.start);
    const q: Any = particles.rotation.element(i).toVar();
    const omega: Any = particles.angularVelocity.element(i).xyz.toVar();
    const omegaMag: Any = length(omega);
    If(omegaMag.lessThan(float(OMEGA_EPS)), () => {
      particles.predictedRotation.element(i).assign(q);
    }).Else(() => {
      const halfAngle: Any = omegaMag.mul(dt).mul(0.5);
      const spinXYZ: Any = omega.mul(sin(halfAngle).div(omegaMag));
      const spinW: Any = cos(halfAngle);
      // spin · q
      const qXYZ: Any = q.xyz;
      const newXYZ: Any = qXYZ.mul(spinW).add(spinXYZ.mul(q.w)).add(spinXYZ.cross(qXYZ));
      const newW: Any = spinW.mul(q.w).sub(spinXYZ.dot(qXYZ));
      particles.predictedRotation.element(i).assign(vec4(newXYZ, newW));
    });
  })()
    .compute(range.count)
    .setName('rotation.predict');

  const advect = Fn(() => {
    const i: Any = instanceIndex.add(range.start);
    const q: Any = particles.rotation.element(i).toVar();
    const qpRaw: Any = particles.predictedRotation.element(i).toVar();
    const qp: Any = qpRaw.div(qpRaw.dot(qpRaw).max(1e-12).sqrt()).toVar();

    // r = qp · conj(q), flipped onto the shorter arc.
    const qInvXYZ: Any = q.xyz.negate();
    const rXYZpre: Any = qp.xyz.mul(q.w).add(qInvXYZ.mul(qp.w)).add(qp.xyz.cross(qInvXYZ));
    const rWpre: Any = qp.w.mul(q.w).sub(qp.xyz.dot(qInvXYZ));
    const flip: Any = rWpre.lessThan(0).select(float(-1), float(1));
    const rXYZ: Any = rXYZpre.mul(flip).toVar();
    const rW: Any = rWpre.mul(flip).toVar();

    const halfSinMag: Any = length(rXYZ);
    If(halfSinMag.lessThan(float(ANGLE_HALF_EPS)), () => {
      particles.angularVelocity.element(i).assign(vec4(0, 0, 0, 0));
    }).Else(() => {
      const angle: Any = atan(halfSinMag, rW).mul(2);
      const omega: Any = rXYZ.mul(angle.div(halfSinMag).div(dt));
      const omegaScale: Any = float(OMEGA_MAX * OMEGA_MAX)
        .div(omega.dot(omega).max(OMEGA_MAX * OMEGA_MAX))
        .sqrt();
      particles.angularVelocity.element(i).assign(vec4(omega.mul(omegaScale), 0));
    });

    particles.rotation.element(i).assign(qp);
  })()
    .compute(range.count)
    .setName('rotation.advect');

  return { predict, advect };
}
