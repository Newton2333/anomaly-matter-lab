/// <reference types="@webgpu/types" />
/**
 * ANOMALY — GPU fluid renderer.
 * PBF solver and reconstructed liquid surface: Daniel Greenheck, MIT.
 * See vendor/particle-fluids/LICENSE and THIRD_PARTY_NOTICES.md.
 */
import {
  AdditiveBlending, Box3, Color, InstancedMesh, SphereGeometry, Vector3,
  type PerspectiveCamera, type Scene,
} from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { Fn, If, float, instanceIndex, mix, positionLocal, sin, uniform, vec3, vec4 } from 'three/tsl';
import {
  createParticleRenderer, FluidSurfaceRenderer, FluidSystem, ParticleSystem,
  PrimitiveSet, SimLoop, type ParticleInit,
} from './vendor/particle-fluids/src/index.js';

import { framePercentile, nextGpuProfile, gpuProfileChanged, calibrationPressure } from './gpu-quality.js';
import { stagePixelRatio } from './render-resolution.js';
import { GpuCompositor } from './gpu-compositor';

type MaterialName = 'metal' | 'glass' | 'plasma';
interface WarmupOptions {
  onProgress?: (mode: MaterialName, fraction: number) => void;
  beforeRender?: (mode: MaterialName) => void;
}
type AnyNode = any;
interface GpuEngineOptions {
  canvas: HTMLCanvasElement;
  scene: Scene;
  camera: PerspectiveCamera;
  particleCount?: number;
  onStats?: (stats: Record<string, unknown>) => void;
  onError?: (error: Error) => void;
}
const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number(v) || 0));
const CENTER = new Vector3(0, 0.85, 0);
const DROP_RADIUS = 0.30;
// These controls change forces, not particle positions. The soft core leaves
// room for the incompressible drop instead of continually crushing its centre.
export const GPU_INTERACTION = Object.freeze({
  pointerX: .36, pointerY: .30,
  holdCore: .21, holdStiffness: 28, holdMaxAcceleration: 6, holdDamping: 10.6,
});

function seedDrop(radius: number): ParticleInit[] {
  const points: ParticleInit[] = [], spacing = radius * 2;
  for (let y = -DROP_RADIUS + spacing / 2; y < DROP_RADIUS; y += spacing)
    for (let z = -DROP_RADIUS + spacing / 2; z < DROP_RADIUS; z += spacing)
      for (let x = -DROP_RADIUS + spacing / 2; x < DROP_RADIUS; x += spacing) {
        if (x * x + y * y + z * z >= DROP_RADIUS * DROP_RADIUS) continue;
        points.push({
          // A slight anisotropic starting drop settles into the solved fluid.
          position: [x * 1.10, CENTER.y + y * 0.96, z / 1.056],
          velocity: [-z * 1.15, Math.sin(x * 9) * 0.045, x * 1.15],
        });
      }
  return points;
}
function fitRadius(target: number): number {
  let radius = 0.018;
  for (let i = 0; i < 5; i++) radius *= Math.cbrt(Math.max(1, seedDrop(radius).length) / target);
  return radius;
}

export async function createGpuEngine(options: GpuEngineOptions) {
  const { canvas, scene, camera } = options;
  if (!navigator.gpu) throw new Error('WebGPU is unavailable.');
  const renderer = await createParticleRenderer({ canvas, antialias: true, alpha: false });
  let compositor: GpuCompositor | undefined;
  try {
  compositor = new GpuCompositor({ renderer, scene, camera });
  const mobile = matchMedia('(pointer: coarse)').matches || innerWidth < 700;
  const targetCount = options.particleCount ?? (mobile ? 3200 : 6500);
  const radius = fitRadius(targetCount);
  const initial = seedDrop(radius);
  const count = initial.length;
  const particles = new ParticleSystem(renderer, count, radius);
  particles.uploadParticles(initial);
  const fluid = new FluidSystem(particles, { viscosity: 0.075, surfaceTension: 0.16 });
  const colliders = new PrimitiveSet(particles);
  const friction = { muS: 0.035, muK: 0.02 };
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), friction);
  colliders.addPlane(new Vector3(0, -1, 0), new Vector3(0, 1.85, 0), friction);
  colliders.addPlane(new Vector3(1, 0, 0), new Vector3(-0.9, 0, 0), friction);
  colliders.addPlane(new Vector3(-1, 0, 0), new Vector3(0.9, 0, 0), friction);
  colliders.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -0.9), friction);
  colliders.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, 0.9), friction);
  const loop = new SimLoop(particles, {
    substeps: 3, iterations: 2, gravity: new Vector3(), materials: [fluid], colliders: [colliders],
  });
  // Keep the fluid-owned mass and material collision metadata when resetting.
  const resetData = initial.map((p, i) => ({ ...p,
    invMass: (particles.invMass.value.array as Float32Array)[i],
    collisionGroup: (particles.collisionGroup.value.array as Uint32Array)[i],
  }));
  const surface = new FluidSurfaceRenderer(fluid, {
    renderer, scene, camera,
    externalScene: { color: compositor.stageColor, depth: compositor.stageDepth },
    bounds: new Box3(new Vector3(-0.92, -0.02, -0.92), new Vector3(0.92, 1.87, 0.92)),
    colliders: [colliders], motionStretch: 0.012,
    appearance: { metalness: 1, metalColor: 0xd7e0e3, roughness: 0.065, envIntensity: 1.35,
      color: 0xb5dde5, attenuationDistance: 0.8, scattering: 0.015, ior: 1.46 },
  });
  surface.mesh.name = 'ANOMALY — solved liquid surface';
  scene.add(surface.mesh);
  compositor.setSurface(surface.mesh);

  // Both star cores and soft halos follow the SAME solved GPU particle buffer.
  const starGeometry = new SphereGeometry(radius * 0.30, 5, 3);
  const starMaterial = new MeshBasicNodeMaterial({ color: 0xffc476, toneMapped: false });
  const particleCenter: AnyNode = particles.positions.element(instanceIndex).xyz;
  const starSeed: AnyNode = sin(float(instanceIndex).mul(12.9898)).mul(43758.5453).fract();
  starMaterial.positionNode = particleCenter.add(positionLocal);
  starMaterial.colorNode = mix(vec3(1.7, 0.38, 0.06), vec3(3.2, 2.4, 1.4), starSeed.pow(3));
  const stars = new InstancedMesh(starGeometry, starMaterial, count);
  stars.frustumCulled = false; stars.visible = false;
  stars.name = 'ANOMALY — live fluid particles';
  const haloGeometry = new SphereGeometry(radius * 0.9, 5, 3);
  const haloMaterial = new MeshBasicNodeMaterial({ color: new Color(0xff962a), transparent: true,
    opacity: 0.085, blending: AdditiveBlending, depthWrite: false, toneMapped: false });
  haloMaterial.positionNode = particleCenter.add(positionLocal);
  const halos = new InstancedMesh(haloGeometry, haloMaterial, count);
  halos.frustumCulled = false; halos.visible = false; halos.renderOrder = 5;
  scene.add(stars, halos);

  const attractor = uniform(CENTER.clone());
  const pointerTarget = CENTER.clone();
  const pull = uniform(2.6), swirl = uniform(0.45), chargeUniform = uniform(0), timeUniform = uniform(0);
  const turbulence = uniform(0.018), airDamping = uniform(0.10);
  const stepDt = uniform(1 / 60);
  const force = Fn(() => {
    const velocity: AnyNode = particles.velocities.element(instanceIndex);
    const position: AnyNode = particles.positions.element(instanceIndex).xyz;
    const offset: AnyNode = attractor.sub(position);
    const distance: AnyNode = offset.length();
    const normalInward: AnyNode = offset.div(distance.max(0.18)).mul(pull);
    const heldAcceleration: AnyNode = distance.sub(GPU_INTERACTION.holdCore).max(0)
      .mul(GPU_INTERACTION.holdStiffness).min(GPU_INTERACTION.holdMaxAcceleration);
    const heldInward: AnyNode = offset.div(distance.max(1e-4)).mul(heldAcceleration);
    const inward: AnyNode = mix(normalInward, heldInward, chargeUniform);
    const relative: AnyNode = position.sub(attractor);
    const vortex: AnyNode = vec3(relative.z.negate(),
      sin(relative.x.mul(8).add(timeUniform)).mul(0.085), relative.x).mul(swirl);
    const eddies: AnyNode = vec3(
      sin(relative.y.mul(6).add(timeUniform.mul(0.7))),
      sin(relative.z.mul(7).sub(timeUniform.mul(0.83))),
      sin(relative.x.mul(6).add(timeUniform.mul(0.63))),
    ).mul(turbulence).mul(float(1).sub(chargeUniform.mul(.98)));
    // Integrate drag exponentially. At full hold the damping is near critical
    // for the soft-core spring, so pressure corrections shed energy instead of
    // building an underdamped compression/rebound cycle over a long press.
    const drag: AnyNode = airDamping.add(chargeUniform.mul(GPU_INTERACTION.holdDamping));
    const damp: AnyNode = stepDt.mul(drag).negate().exp();
    const forceDt: AnyNode = float(1).sub(damp).div(drag.max(1e-4));
    velocity.assign(vec4(velocity.xyz.mul(damp).add(inward.add(vortex).add(eddies).mul(forceDt)), velocity.w));
  })().compute(count);

  const burstCenter = uniform(CENTER.clone()), burstStrength = uniform(3.2);
  const burstKernel = Fn(() => {
    const position: AnyNode = particles.positions.element(instanceIndex).xyz;
    const delta: AnyNode = position.sub(burstCenter);
    const distance: AnyNode = delta.length();
    If(distance.lessThan(0.82), () => {
      const velocity: AnyNode = particles.velocities.element(instanceIndex);
      const falloff: AnyNode = float(1).sub(distance.div(0.82).pow(2)).pow(2);
      const direction: AnyNode = delta.div(distance.max(1e-4));
      velocity.assign(vec4(velocity.xyz.add(direction.mul(falloff.mul(burstStrength))), velocity.w));
    });
  })().compute(count);

  let material: MaterialName = 'metal';
  let energy = .42, charge = 0, paused = false, destroyed = false, resetPending = false;
  let burstPending = 0, time = 0, reportTime = 0, dirty = true;
  let pointerX = 0, pointerY = 0;
  let lostError: Error | undefined;
  let warming = false;
  let warmupPromise: Promise<void> | undefined;
  // Expensive liquid rays have their own budget. The ring and scene keep a
  // CSS-native antialiased target even when liquid shading scales down.
  const pixelBudget = mobile ? 1_300_000 : 2_400_000;
  const profiles = {
    metal: { quality: 1, reflections: true, canReflect: true },
    glass: { quality: 1, reflections: true, canReflect: true },
    plasma: { quality: 1, reflections: false, canReflect: false },
  };
  let viewportWidth = Math.max(1, canvas.clientWidth || innerWidth);
  let viewportHeight = Math.max(1, canvas.clientHeight || innerHeight);
  let appliedPixelRatio = 0, liquidPixelRatio = 0, resolutionDirty = true;
  let qualityChangedAt = performance.now(), frameLatencyMs = 0;
  let lastSubmittedAt = 0;
  const completedFrames: number[] = [];
  const pendingFrames: Promise<void>[] = [];

  function applyResolution() {
    const nativeRatio = Math.min(window.devicePixelRatio || 1, 1.6);
    const ratio = stagePixelRatio(viewportWidth, viewportHeight, window.devicePixelRatio || 1);
    liquidPixelRatio = Math.min(nativeRatio, Math.sqrt(pixelBudget / (viewportWidth * viewportHeight))) *
      profiles[material].quality;
    surface.reflections = profiles[material].reflections;
    if (Math.abs(ratio - appliedPixelRatio) > 0.005) {
      renderer.setPixelRatio(ratio);
      appliedPixelRatio = ratio;
    }
    renderer.setSize(viewportWidth, viewportHeight, false);
    compositor!.resize(Math.max(1, Math.floor(viewportWidth * ratio)),
      Math.max(1, Math.floor(viewportHeight * ratio)),
      Math.max(1, Math.floor(viewportWidth * liquidPixelRatio)),
      Math.max(1, Math.floor(viewportHeight * liquidPixelRatio)));
    resolutionDirty = false;
  }

  function resetTimingWindow() {
    completedFrames.length = 0;
    frameLatencyMs = 0;
    qualityChangedAt = performance.now();
    lastSubmittedAt = 0;
  }

  function observeCompletedFrame(mode: MaterialName, submittedAt: number, gap: number) {
    if (destroyed || warming || paused || document.hidden || material !== mode) return;
    if (mode === 'plasma') return; // Stars use the native scene pass, not liquid rays.
    const now = performance.now();
    // Exclude tab suspension, first frame/shader compilation and mode switches.
    if (gap <= 0 || gap > 1500 || now - qualityChangedAt < 1200) return;
    const latency = now - submittedAt;
    frameLatencyMs = frameLatencyMs ? frameLatencyMs * 0.85 + latency * 0.15 : latency;
    completedFrames.push(now);
    while (completedFrames.length > 1 && completedFrames[0]! < now - 1500) completedFrames.shift();
    const span = now - completedFrames[0]!;
    if (completedFrames.length < 3 || (span < 650 && !(frameLatencyMs > 80 && span >= 200))) return;
    const completionFps = (completedFrames.length - 1) * 1000 / span;
    let next = profiles[mode];
    // Require both slow completed frames and an occupied GPU queue. A 30 Hz
    // monitor or a CPU-bound tab should not lose image quality for no benefit.
    if (completionFps < 52 && frameLatencyMs > 24)
      next = nextGpuProfile(profiles[mode], Math.max(1000 / completionFps, frameLatencyMs / 2));
    else if (completionFps > 57 && frameLatencyMs < 12 && now - qualityChangedAt > 12000)
      next = nextGpuProfile(profiles[mode], frameLatencyMs, { recover: true });
    if (gpuProfileChanged(profiles[mode], next)) {
      profiles[mode] = next;
      resolutionDirty = true;
      resetTimingWindow();
    }
  }
  function applyPhysics() {
    // All modes evolve the actual solved buffer. Changing a material changes its
    // physical response as well as its appearance; no procedural shape override.
    if (material === 'metal') {
      fluid.viscosity = 0.12 - energy * 0.085;
      fluid.surfaceTension = 0.24 - energy * 0.10;
      airDamping.value = 0.14;
      turbulence.value = 0.012 + energy * 0.025;
    } else if (material === 'glass') {
      fluid.viscosity = 0.06 - energy * 0.045;
      fluid.surfaceTension = 0.12 - energy * 0.07;
      airDamping.value = 0.09;
      turbulence.value = 0.02 + energy * 0.045;
    } else {
      fluid.viscosity = 0.006 - energy * 0.004;
      fluid.surfaceTension = 0.008 - energy * 0.006;
      airDamping.value = 0.30;
      turbulence.value = 0.24 + energy * 0.38;
    }
    const held = charge * charge * (3 - 2 * charge);
    // XSPH dissipates relative particle motion while the actual PBF solve keeps
    // the fluid volume. Normal material viscosity returns immediately on release.
    fluid.viscosity += (.16 - fluid.viscosity) * held;
  }
  const device = (renderer.backend as unknown as { device?: GPUDevice }).device;
  void device?.lost.then(info => {
    if (destroyed || info.reason === 'destroyed') return;
    lostError = new Error('The WebGPU device was interrupted.');
    options.onError?.(lostError);
  });

  const api = {
    kind: 'webgpu' as const, renderer, scene, camera, particleCount: count,
    render() { compositor!.render(); },
    async verifyFrame() {
      // The first settled drop is centered here. Verify its covered draw before
      // exposing the page; a driver/buffer failure takes the existing fallback.
      camera.updateMatrixWorld();
      const point = CENTER.clone().project(camera);
      await compositor!.validateSurface(point.x * .5 + .5, .5 - point.y * .5);
    },
    waitForIdle,
    warmup(options: WarmupOptions = {}) {
      // A second caller joins the same preparation instead of running another
      // simulation/render sequence against the renderer concurrently.
      return warmupPromise ??= warmup(options);
    },
    setMaterial(name: MaterialName) {
      if (!['metal', 'glass', 'plasma'].includes(name)) return;
      if (name === material) return;
      resolutionDirty = true; resetTimingWindow();
      if (name === 'plasma' && material !== 'plasma') burstPending = Math.max(burstPending, 0.24);
      material = name; surface.mesh.visible = name !== 'plasma';
      stars.visible = halos.visible = name === 'plasma';
      if (name === 'metal') surface.setAppearance({ metalness: 1, metalColor: 0xd7e0e3,
        roughness: 0.065, envIntensity: 1.35, color: 0xb5dde5, attenuationDistance: 0.8, scattering: 0.015 });
      if (name === 'glass') surface.setAppearance({ metalness: 0, color: 0x94e3ec,
        ior: 1.46, attenuationDistance: 0.65, scattering: 0.018, roughness: 0.035, envIntensity: 1.1 });
      applyPhysics();
      dirty = true;
    },
    setEnergy(value: number) {
      energy = clamp(value); applyPhysics();
      dirty = true;
    },
    setPointer(x: number, y: number) { pointerX = clamp(x, -1, 1); pointerY = clamp(y, -1, 1); },
    clearPointer() { pointerX = pointerY = 0; },
    setCharge(value: number) {
      // Start/continue/cancel a hold before any queued release is submitted.
      // The caller releases explicitly with setCharge(0), then burst(strength).
      burstPending = 0;
      const next = clamp(value);
      if (next !== charge) { charge = next; applyPhysics(); }
      if (charge > 0) paused = false;
    },
    burst(strength = 1) {
      burstPending = clamp(strength, 0.15, 1); charge = 0; paused = false; applyPhysics();
    },
    setPaused(value: boolean) { paused = !!value; resetTimingWindow(); },
    reset() { resetPending = true; burstPending = 0; charge = 0; energy = .42; pointerX = pointerY = 0;
      time = reportTime = 0; timeUniform.value = 0; attractor.value.copy(CENTER); applyPhysics();
      resetTimingWindow(); dirty = true; },
    async getParticles() {
      await waitForIdle();
      const packed = new Float32Array(await renderer.getArrayBufferAsync(particles.positions.value));
      const xyz = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        xyz[i * 3] = packed[i * 4]!; xyz[i * 3 + 1] = packed[i * 4 + 1]!; xyz[i * 3 + 2] = packed[i * 4 + 2]!;
      }
      return xyz;
    },
    resize(width: number, height: number) {
      viewportWidth = Math.max(1, width); viewportHeight = Math.max(1, height);
      applyResolution(); resetTimingWindow(); dirty = true;
    },
    // Call immediately after renderer.render/renderAsync. Three's computeAsync
    // resolves after submission, not GPU completion: without a fence the next
    // frame can enqueue more expensive surface work indefinitely during a burst.
    // Two in-flight frames retain CPU/GPU overlap while bounding interaction lag.
    afterRender() {
      if (destroyed || !device) return;
      try {
        const submittedAt = performance.now();
        const gap = lastSubmittedAt ? submittedAt - lastSubmittedAt : 0;
        lastSubmittedAt = submittedAt;
        const submittedMode = material;
        const completion = device.queue.onSubmittedWorkDone().then(() => {
          observeCompletedFrame(submittedMode, submittedAt, gap);
        }).catch(() => {
          // device.lost above owns recovery; fences must never reject the RAF loop.
        }).finally(() => {
          const index = pendingFrames.indexOf(completion);
          if (index !== -1) pendingFrames.splice(index, 1);
        });
        pendingFrames.push(completion);
      } catch {
        // A device being disposed/lost is reported by its existing error handler.
      }
    },
    async step(dt: number) {
      if (destroyed) return;
      if (lostError) throw lostError;
      while (pendingFrames.length >= 2) await pendingFrames[0];
      if (destroyed) return;
      if (lostError) throw lostError;
      if (resolutionDirty) applyResolution();
      const started = performance.now();
      if (resetPending) { particles.uploadParticles(resetData); resetPending = false; dirty = true; }
      if (!paused && dt > 0) {
        const step = clamp(dt, 1 / 240, 1 / 30);
        time += step; stepDt.value = step; timeUniform.value = time;
        pointerTarget.set(pointerX * GPU_INTERACTION.pointerX, CENTER.y + pointerY * GPU_INTERACTION.pointerY, 0);
        attractor.value.lerp(pointerTarget, 1 - Math.exp(-step * 5));
        const held = charge * charge * (3 - 2 * charge);
        chargeUniform.value = held;
        const centerPull = material === 'plasma' ? 0.72 : material === 'glass' ? 2.3 : 2.8;
        pull.value = centerPull;
        const spin = material === 'plasma' ? 2.1 + energy * 1.8 :
          material === 'glass' ? 0.28 + energy * 1.35 : 0.12 + energy * 1.1;
        swirl.value = spin * (1 - held * .96);
        await renderer.computeAsync(force);
        if (burstPending > 0 && charge === 0) {
          burstCenter.value.copy(attractor.value);
          burstStrength.value = 2.2 + burstPending * 2.6;
          await renderer.computeAsync(burstKernel); burstPending = 0;
        }
        await loop.step(step); dirty = true;
      }
      if (material !== 'plasma' && dirty) { await surface.update(); dirty = false; }
      if (time - reportTime > 0.5) { reportTime = time;
        options.onStats?.({ kind: 'webgpu', particles: count, stepMs: performance.now() - started,
          substeps: loop.substeps, material, pixelRatio: appliedPixelRatio,
          renderPixels: Math.round(viewportWidth * viewportHeight * appliedPixelRatio ** 2),
          liquidPixels: Math.round(viewportWidth * viewportHeight * liquidPixelRatio ** 2),
          gpuQueueLatencyMs: frameLatencyMs, quality: profiles[material].quality,
          screenSpaceReflections: profiles[material].reflections }); }
    },
    destroy() {
      if (destroyed) return; destroyed = true;
      scene.remove(surface.mesh, stars, halos);
      surface.dispose(); compositor!.dispose(); loop.dispose(); colliders.dispose(); particles.dispose();
      starGeometry.dispose(); starMaterial.dispose(); haloGeometry.dispose(); haloMaterial.dispose();
      force.dispose(); burstKernel.dispose(); renderer.dispose();
    },
  };
  async function withTimeout<T>(task: Promise<T>, milliseconds: number, label: string): Promise<T> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([task, new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`GPU ${label} took too long.`)), milliseconds);
      })]);
    } finally { if (timer) clearTimeout(timer); }
  }

  async function waitForIdle(): Promise<void> {
    if (lostError) throw lostError;
    if (destroyed) throw new Error('GPU renderer was disposed.');
    if (device) await withTimeout(device.queue.onSubmittedWorkDone(), 15000, 'preparation');
    if (lostError) throw lostError;
  }

  async function warmup({onProgress, beforeRender}: WarmupOptions): Promise<void> {
    const savedEnergy = energy, savedPaused = paused;
    const modes: MaterialName[] = ['metal', 'glass', 'plasma'];
    let preparationTime = 0, visibilityEpoch = 0;
    const visibilityChanged = () => { visibilityEpoch++; };
    document.addEventListener('visibilitychange', visibilityChanged);
    warming = true;
    const visible = async () => {
      if (!document.hidden) return;
      await new Promise<void>(resolve => {
        const changed = () => {
          if (document.hidden) return;
          document.removeEventListener('visibilitychange', changed);
          resolve();
        };
        document.addEventListener('visibilitychange', changed);
        changed();
      });
    };
    try {
      await waitForIdle();
      for (let index = 0; index < modes.length; index++) {
        const mode = modes[index]!;
        let work = 0;
        const report = () => onProgress?.(mode, (index + Math.min(.98, ++work / 22)) / modes.length);
        const draw = async (): Promise<number> => {
          await visible();
          const started = performance.now(), sampleEpoch = visibilityEpoch;
          beforeRender?.(mode);
          await api.step(1 / 60);
          // Startup can afford a fence here. It separates solver/reconstruction
          // time from shading so a compute bottleneck does not blur the image.
          await waitForIdle();
          const drawStarted = performance.now();
          api.render();
          await waitForIdle();
          const ended = performance.now();
          const duration = ended - started;
          const pressure = calibrationPressure(drawStarted - started, ended - drawStarted);
          // A hidden tab can suspend GPU work or timer delivery. Exclude the
          // entire transition even when the tab became visible before the fence.
          const valid = !document.hidden && visibilityEpoch === sampleEpoch;
          if (valid) preparationTime += duration;
          if (preparationTime > 90000) throw new Error('GPU preparation exceeded its time budget.');
          report();
          return valid ? pressure : 0;
        };
        api.reset(); api.setMaterial(mode); api.setEnergy(savedEnergy); api.setPaused(false);
        await visible();
        beforeRender?.(mode);
        await api.step(1 / 60);
        // The real covered draw compiles each visible shader in its actual
        // render context, including viewport depth/color texture dependencies.
        // compileAsync runs updateBefore outside that context in Three r184,
        // which is unsafe for this liquid's framebuffer-copy nodes.
        await draw(); // Discard first allocation/compilation timing.
        const compact: number[] = [];
        for (let sample = 0; sample < 3; sample++) compact.push(await draw());
        api.burst(.85);
        const scattered: number[] = [];
        for (let sample = 0; sample < 6; sample++) {
          const duration = await draw();
          if (sample >= 3) scattered.push(duration);
        }
        let frameMs = Math.max(framePercentile(compact, .5), framePercentile(scattered, .5));
        // At most three adjustments: omit SSR, tune pixels, then verify/tune
        // once more. Every selected profile is sampled after its resize work.
        for (let adjustment = 0; mode !== 'plasma' && adjustment < 3; adjustment++) {
          const next = nextGpuProfile(profiles[mode], frameMs);
          if (!gpuProfileChanged(profiles[mode], next)) break;
          profiles[mode] = next; resolutionDirty = true;
          await draw(); // New target allocation is not a steady frame sample.
          const samples: number[] = [];
          for (let sample = 0; sample < 3; sample++) samples.push(await draw());
          frameMs = framePercentile(samples, .5);
        }
        options.onStats?.({ kind: 'webgpu', preparation: true, material: mode,
          calibrationPressureMs: frameMs, quality: profiles[mode].quality,
          screenSpaceReflections: profiles[mode].reflections });
        onProgress?.(mode, Math.min(.98, (index + 1) / modes.length));
      }
      // Warm-up impulses never leak into the opening scene. Reset the storage
      // buffers and rebuild a metal surface without advancing simulation time.
      api.reset(); api.setMaterial('metal'); api.setEnergy(savedEnergy); api.setPaused(savedPaused);
      await api.step(0);
      await waitForIdle();
      beforeRender?.('metal');
      onProgress?.('metal', 1);
    } finally {
      document.removeEventListener('visibilitychange', visibilityChanged);
      warming = false;
      resetTimingWindow();
    }
  }

  api.setEnergy(energy);
  options.onStats?.({ kind: 'webgpu', particles: count, substeps: 3, material });
  return api;
  } catch (error) {
    // Initialization failures must not keep a second GPU device alive while
    // the caller falls back to another rendering backend.
    compositor?.dispose(); renderer.dispose();
    throw error;
  }
}
