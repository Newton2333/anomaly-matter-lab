import { WebGPURenderer } from 'three/webgpu';
import type { WebGPURendererParameters } from 'three/src/renderers/webgpu/WebGPURenderer.js';

/** Device limits the solver's scan and contact kernels need. */
const REQUIRED_LIMITS: Record<string, number> = {
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxStorageBuffersPerShaderStage: 10,
};

/**
 * Create and initialize a `WebGPURenderer` with the device limits the solver
 * needs. Throws if WebGPU is unavailable, because the simulation has no
 * WebGL fallback.
 */
export async function createParticleRenderer(
  options: Partial<WebGPURendererParameters> = {},
): Promise<WebGPURenderer> {
  const requiredLimits: Record<string, number> = { ...options.requiredLimits };
  for (const [name, value] of Object.entries(REQUIRED_LIMITS)) {
    requiredLimits[name] = Math.max(requiredLimits[name] ?? 0, value);
  }
  const renderer = new WebGPURenderer({ ...options, requiredLimits });
  await renderer.init();
  if ((renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend !== true) {
    renderer.dispose();
    throw new Error(
      'createParticleRenderer: WebGPU is unavailable, and this library has no WebGL fallback. ' +
        'Use a browser with WebGPU enabled (on Linux Chromium, pass --enable-unsafe-webgpu).',
    );
  }
  return renderer;
}
