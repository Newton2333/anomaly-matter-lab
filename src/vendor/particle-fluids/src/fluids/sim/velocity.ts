import { Continue, Fn, If, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { emitPoly6FromRSq, emitSpikyGrad } from '../../core/index.js';
import { emitFluidIndex, emitNeighborVolume, type FluidKernelContext } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Buffers for vorticity confinement (Macklin & Müller 2013, §5). */
export interface VorticityBuffers {
  readonly strength: UniformNode<'float', number>;
  /** Curl of the velocity field, ω. */
  readonly omega: StorageBufferNode<'vec4'>;
  readonly omegaLength: StorageBufferNode<'float'>;
  /** Gradient of |ω|, η. */
  readonly eta: StorageBufferNode<'vec4'>;
}

/** Buffers for XSPH viscosity (Macklin & Müller 2013, eq. 17). */
export interface ViscosityBuffers {
  /** Blend factor c: 0 leaves velocities alone, larger values average them with neighbors. */
  readonly c: UniformNode<'float', number>;
  readonly deltaV: StorageBufferNode<'vec4'>;
}

/**
 * One neighbor walk that gathers both the vorticity `ω_i = Σ_j V_j (v_i − v_j) × ∇W`
 * (eq. 15) and the XSPH velocity change `Δv_i = c Σ_j V_j (v_j − v_i) W` (eq. 17).
 * Neighbor volumes `V_j` stand in for the paper's unit masses, so the result
 * doesn't depend on particle size. Velocities are only read; see
 * {@link buildVelocityApplyKernel}.
 */
export function buildVelocityWalkKernel(
  context: FluidKernelContext,
  passes: { readonly vorticity?: VorticityBuffers; readonly viscosity?: ViscosityBuffers },
): ComputeNode {
  const { particles, neighbors, sph } = context;
  const { vorticity, viscosity } = passes;

  return Fn(() => {
    const { i, row } = neighbors.emitThread();
    const xi: Any = particles.positions.element(i).xyz.toVar();
    const vi: Any = particles.velocities.element(i).xyz.toVar();
    const curl: Any = vec3(0).toVar();
    const smoothing: Any = vec3(0).toVar();

    neighbors.forEach(row, (j: Any) => {
      const offset: Any = xi.sub(particles.positions.element(j).xyz).toVar();
      const vj: Any = particles.velocities.element(j).xyz;
      const volume: Any = emitNeighborVolume(context, j);
      if (vorticity) curl.addAssign(vi.sub(vj).cross(emitSpikyGrad(offset, sph)).mul(volume));
      if (viscosity) {
        smoothing.addAssign(
          vj
            .sub(vi)
            .mul(emitPoly6FromRSq(offset.dot(offset), sph))
            .mul(volume),
        );
      }
    });

    if (vorticity) {
      vorticity.omega.element(i).assign(vec4(curl, 0));
      vorticity.omegaLength.element(i).assign(curl.length());
    }
    if (viscosity) viscosity.deltaV.element(i).assign(vec4(smoothing.mul(viscosity.c), 0));
  })()
    .compute(neighbors.threadCount)
    .setName('velocity.velocityWalk');
}

/** The location vector `η = ∇|ω|` for vorticity confinement, from fluid neighbors only. */
export function buildVorticityGradientKernel(
  context: FluidKernelContext,
  vorticity: VorticityBuffers,
): ComputeNode {
  const { particles, neighbors, sph, particleVolume } = context;
  return Fn(() => {
    const { i, row } = neighbors.emitThread();
    const xi: Any = particles.positions.element(i).xyz.toVar();
    const omegaI: Any = vorticity.omegaLength.element(i).toVar();
    const eta: Any = vec3(0).toVar();
    neighbors.forEach(row, (j: Any) => {
      If(particles.boundaryVolume.element(j).greaterThan(0), () => {
        Continue();
      });
      const gradient: Any = emitSpikyGrad(xi.sub(particles.positions.element(j).xyz), sph);
      eta.addAssign(gradient.mul(vorticity.omegaLength.element(j).sub(omegaI).mul(particleVolume)));
    });
    vorticity.eta.element(i).assign(vec4(eta, 0));
  })()
    .compute(neighbors.threadCount)
    .setName('velocity.vorticityGradient');
}

/**
 * Add the vorticity confinement force `ε (N × ω)` with `N = η / |η|` (eq. 16)
 * and the XSPH velocity change to each particle's velocity.
 */
export function buildVelocityApplyKernel(
  context: FluidKernelContext,
  passes: { readonly vorticity?: VorticityBuffers; readonly viscosity?: ViscosityBuffers },
): ComputeNode {
  const { particles, dt } = context;
  const { vorticity, viscosity } = passes;
  return Fn(() => {
    const i: Any = emitFluidIndex(context);
    const change: Any = vec3(0).toVar();
    if (vorticity) {
      const eta: Any = vorticity.eta.element(i).xyz.toVar();
      const etaLength: Any = eta.length().toVar();
      If(etaLength.greaterThanEqual(1e-12), () => {
        const force: Any = eta.div(etaLength).cross(vorticity.omega.element(i).xyz);
        change.addAssign(force.mul(vorticity.strength).mul(dt));
      });
    }
    if (viscosity) change.addAssign(viscosity.deltaV.element(i).xyz);
    const velocity: Any = particles.velocities.element(i);
    velocity.assign(vec4(velocity.xyz.add(change), velocity.w));
  })()
    .compute(context.range.count)
    .setName('velocity.velocityApply');
}
