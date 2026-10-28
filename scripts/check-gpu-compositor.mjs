import assert from 'node:assert/strict';
import fs from 'node:fs';
import { registerHooks } from 'node:module';
import ts from 'typescript';

// Load the real TypeScript modules, including upstream .js specifiers that point
// at TypeScript sources. No browser or GPU is emulated by this loader.
const hooks = registerHooks({
  resolve(specifier, context, next) {
    const source = {'three': 'Three.Core.js', 'three/webgpu': 'Three.WebGPU.js', 'three/tsl': 'Three.TSL.js'}[specifier];
    if (source) return next(new URL('../node_modules/three/src/' + source, import.meta.url).href, context);
    try { return next(specifier, context); }
    catch (error) {
      if (specifier.startsWith('.') && specifier.endsWith('.js') && context.parentURL) {
        const alternate = new URL(specifier.slice(0, -3) + '.ts', context.parentURL);
        if (fs.existsSync(alternate)) return next(alternate.href, context);
      }
      throw error;
    }
  },
  load(url, context, next) {
    if (url.startsWith('file:') && url.endsWith('.ts')) {
      return {format: 'module', shortCircuit: true, source: ts.transpileModule(fs.readFileSync(new URL(url), 'utf8'), {
        compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext},
      }).outputText};
    }
    return next(url, context);
  },
});
const { Color, Scene, PerspectiveCamera, Mesh, BoxGeometry, MeshBasicMaterial, RenderTarget, Box3, Vector3,
  ACESFilmicToneMapping, SRGBColorSpace, LinearSRGBColorSpace, NoToneMapping, DataUtils } = await import('three');
const { WebGPURenderer } = await import('three/webgpu');
const { default: WGSLNodeBuilder } = await import('three/src/renderers/webgpu/nodes/WGSLNodeBuilder.js');
const { GpuCompositor } = await import('../src/gpu-compositor.ts');
const { ParticleSystem, FluidSystem, FluidSurfaceRenderer } = await import('../src/vendor/particle-fluids/src/index.ts');

function fixture(samples = 4) {
  const scene = new Scene(), camera = new PerspectiveCamera(), surface = new Mesh(new BoxGeometry(), new MeshBasicMaterial());
  scene.background = new Color('#123456');
  scene.add(surface);
  const output = new RenderTarget(23, 17);
  const calls = [];
  let current = output, face = 2, mip = 1, clearAlpha = .7, color = new Color('#abcdef');
  const renderer = {
    samples, autoClear: false, throwAt: 0, depth: true,
    toneMapping: ACESFilmicToneMapping, outputColorSpace: SRGBColorSpace,
    getRenderTarget: () => current,
    getActiveCubeFace: () => face,
    getActiveMipmapLevel: () => mip,
    setRenderTarget(value, nextFace = 0, nextMip = 0) { current = value; face = nextFace; mip = nextMip; },
    getClearColor: target => target.copy(color),
    getClearAlpha: () => clearAlpha,
    setClearColor(value, alpha = 1) { color.set(value); clearAlpha = alpha; },
    render(object, view) {
      calls.push({object, material: object.material, view, target: current, background: scene.background,
        surfaceVisible: surface.visible, cameraLayers: camera.layers.mask,
        surfaceLayers: surface.layers.mask, clearAlpha, autoClear: this.autoClear,
        depth: this.depth, toneMapping: this.toneMapping, outputColorSpace: this.outputColorSpace});
      if (this.throwAt === calls.length) throw new Error('injected render failure');
    },
  };
  const compositor = new GpuCompositor({renderer, scene, camera});
  compositor.setSurface(surface);
  const original = {background: scene.background, cameraMask: camera.layers.mask, surfaceMask: surface.layers.mask};
  const restored = (surfaceVisible = true) => {
    assert.equal(scene.background, original.background);
    assert.equal(camera.layers.mask, original.cameraMask);
    assert.equal(surface.layers.mask, original.surfaceMask);
    assert.equal(surface.visible, surfaceVisible);
    assert.equal(renderer.autoClear, false);
    assert.equal(current, output); assert.equal(face, 2); assert.equal(mip, 1);
    assert.equal(clearAlpha, .7); assert.equal(color.getHexString(), 'abcdef');
    assert.equal(renderer.depth, true);
    assert.equal(renderer.toneMapping, ACESFilmicToneMapping);
    assert.equal(renderer.outputColorSpace, SRGBColorSpace);
  };
  return {compositor, renderer, scene, camera, surface, calls, output, restored};
}

for (const samples of [0, 4]) {
  const f = fixture(samples);
  f.compositor.resize(1920, 1080, 864, 486);
  assert.equal(f.compositor.stageTarget.width, 1920);
  assert.equal(f.compositor.stageTarget.height, 1080);
  assert.equal(f.compositor.fluidTarget.width, 864);
  assert.equal(f.compositor.fluidTarget.height, 486);
  assert.equal(f.compositor.stageTarget.samples, samples);
  assert.equal(f.compositor.fluidTarget.samples, 0);
  f.compositor.render();
  assert.equal(f.calls.length, 3);
  assert.equal(f.calls[0].target, f.compositor.stageTarget);
  assert.equal(f.calls[0].surfaceVisible, false);
  assert.equal(f.calls[1].target, f.compositor.fluidTarget);
  assert.equal(f.calls[1].background, null); assert.equal(f.calls[1].clearAlpha, 0);
  assert.equal(f.calls[1].cameraLayers, f.calls[1].surfaceLayers);
  assert.equal(f.calls[1].cameraLayers & 1, 0);
  assert.equal(f.calls[2].target, f.output); assert(f.calls[2].object.isQuadMesh);
  assert.equal(f.calls[2].depth, false);
  assert.equal(f.calls[2].toneMapping, NoToneMapping);
  assert.equal(f.calls[2].outputColorSpace, LinearSRGBColorSpace);
  assert(f.calls.every(call => call.autoClear));
  f.restored();
  // Particle mode uses native stage + explicit output conversion. It never
  // samples stale liquid or creates Three's hidden full-resolution HDR target.
  f.calls.length = 0; f.surface.visible = false; f.compositor.render();
  assert.equal(f.calls.length, 2);
  assert.equal(f.calls[0].target, f.compositor.stageTarget);
  assert.equal(f.calls[1].target, f.output);
  assert.equal(f.calls[1].material, f.compositor.stageMaterial);
  assert.equal(f.calls[1].depth, false);
  assert.equal(f.calls[1].toneMapping, NoToneMapping);
  assert.equal(f.calls[1].outputColorSpace, LinearSRGBColorSpace);
  f.restored(false);
  f.surface.visible = true;
  const disposals = [0, 0, 0, 0];
  [f.compositor.stageTarget, f.compositor.fluidTarget, f.compositor.material, f.compositor.stageMaterial].forEach((o, i) => o.addEventListener('dispose', () => disposals[i]++));
  f.compositor.dispose(); f.compositor.dispose();
  assert.deepEqual(disposals, [1, 1, 1, 1]);
  assert.throws(() => f.compositor.render(), /disposed/);
}
for (const pass of [1, 2, 3]) {
  const f = fixture(); f.renderer.throwAt = pass;
  assert.throws(() => f.compositor.render(), /injected render failure/);
  f.restored(); f.compositor.dispose();
}
for (const pass of [1, 2]) {
  const f = fixture(); f.surface.visible = false; f.renderer.throwAt = pass;
  assert.throws(() => f.compositor.render(), /injected render failure/);
  f.restored(false); f.compositor.dispose();
}
{
  const f = fixture(4);
  const oldVersion = f.surface.material.version;
  f.compositor.resize(3840, 2160, 1248, 702);
  assert.equal(f.compositor.stageTarget.width, 3840);
  assert.equal(f.compositor.stageTarget.samples, 0);
  assert(f.surface.material.version > oldVersion);
  f.compositor.resize(1920, 1080, 864, 486);
  assert.equal(f.compositor.stageTarget.samples, 4);
  f.compositor.dispose();
}
{
  const f = fixture(); f.compositor.resize(1920, 1080, 864, 486);
  const stride = 256 / Uint16Array.BYTES_PER_ELEMENT;
  const pixels = new Uint16Array(stride * 2 + 12);
  let args;
  f.renderer.readRenderTargetPixelsAsync = async (...value) => { args = value; return pixels; };
  // The sole visible sample lives in the last padded row: an incorrect packed
  // read would reject a valid surface. RGB and alpha must both be present.
  pixels[stride * 2 + 4] = DataUtils.toHalfFloat(.25);
  pixels[stride * 2 + 7] = DataUtils.toHalfFloat(1);
  await f.compositor.validateSurface(.5, .5);
  assert.equal(args[0], f.compositor.fluidTarget);
  assert.deepEqual(args.slice(1), [431, 242, 3, 3]);
  pixels.fill(0);
  await assert.rejects(f.compositor.validateSurface(.5, .5), /did not render/);
  pixels[3] = DataUtils.toHalfFloat(1);
  await assert.rejects(f.compositor.validateSurface(.5, .5), /did not render/);
  pixels[0] = 0x7c00; // Half-float positive infinity.
  await assert.rejects(f.compositor.validateSurface(.5, .5), /did not render/);
  pixels[0] = DataUtils.toHalfFloat(.2);
  await f.compositor.validateSurface(0, 0);
  assert.deepEqual(args.slice(1), [0, 0, 3, 3]);
  f.renderer.readRenderTargetPixelsAsync = async () => { throw new Error('readback unavailable'); };
  await assert.rejects(f.compositor.validateSurface(.5, .5), /readback unavailable/);
  f.compositor.dispose();
}

// Generate actual Three WGSL, using only feature declarations in place of an
// initialized device. This catches TSL/type/binding mistakes, not driver errors.
function shaderRenderer(samples) {
  const renderer = new WebGPURenderer({canvas: {width: 1, height: 1, style: {}}, antialias: samples > 0});
  renderer.backend.renderer = renderer;
  renderer.backend.compatibilityMode = false;
  renderer.backend.device = {features: new Set(['float32-filterable']), limits: {maxUniformBufferBindingSize: 65536}};
  renderer.hasFeature = feature => renderer.backend.device.features.has(feature);
  renderer.toneMapping = ACESFilmicToneMapping;
  return renderer;
}
for (const samples of [0, 4]) {
  const renderer = shaderRenderer(samples), scene = new Scene(), camera = new PerspectiveCamera();
  const compositor = new GpuCompositor({renderer, scene, camera});
  const builder = new WGSLNodeBuilder(compositor.quad, renderer);
  builder.scene = scene; builder.camera = compositor.quad.camera;
  builder.build();
  assert(builder.fragmentShader.includes('textureLoad'));
  assert(builder.fragmentShader.includes('textureSample'));
  assert.equal(builder.fragmentShader.includes('texture_depth_multisampled_2d'), samples > 0);
  assert(!builder.fragmentShader.includes('undefined'));
  compositor.quad.material = compositor.stageMaterial;
  const stageBuilder = new WGSLNodeBuilder(compositor.quad, renderer);
  stageBuilder.scene = scene; stageBuilder.camera = compositor.quad.camera;
  stageBuilder.build();
  assert(stageBuilder.uniforms.fragment.some(uniform => uniform.node.value === compositor.stageTarget.texture));
  assert(!stageBuilder.uniforms.fragment.some(uniform =>
    uniform.node.value === compositor.fluidTarget.texture || uniform.node.value === compositor.fluidTarget.depthTexture));
  assert(!stageBuilder.fragmentShader.includes('texture_depth'));
  const particles = new ParticleSystem(renderer, 8, .05);
  const fluid = new FluidSystem(particles);
  const surface = new FluidSurfaceRenderer(fluid, {
    renderer, scene, camera, voxelBudget: 512,
    bounds: new Box3(new Vector3(-.5, 0, -.5), new Vector3(.5, 1, .5)),
    externalScene: {color: compositor.stageColor, depth: compositor.stageDepth},
  });
  compositor.setSurface(surface.mesh);
  scene.add(surface.mesh);
  const liquidBuilder = new WGSLNodeBuilder(surface.mesh, renderer);
  liquidBuilder.scene = scene; liquidBuilder.camera = camera;
  liquidBuilder.build();
  assert(liquidBuilder.fragmentShader.includes('texture_3d'));
  assert.equal(liquidBuilder.fragmentShader.includes('texture_depth_multisampled_2d'), samples > 0);
  assert(!liquidBuilder.fragmentShader.includes('undefined'));
  assert(!liquidBuilder.updateBeforeNodes.some(node => node.isViewportTextureNode));
  surface.dispose(); particles.dispose();
  compositor.dispose();
}
hooks.deregister();
console.log('GPU compositor: separate target sizes, pass isolation, particle output, state/error/color restoration, disposal, MSAA memory bounds, startup readback and composite/stage/liquid WGSL generation passed.');
