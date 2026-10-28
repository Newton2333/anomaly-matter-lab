import {
  Color, ColorManagement, DataUtils, DepthTexture, FloatType, HalfFloatType, LinearFilter, NearestFilter, NoToneMapping,
  RenderTarget, Vector2, type Camera, type Mesh, type Scene,
} from 'three';
import { MeshBasicNodeMaterial, QuadMesh, type WebGPURenderer } from 'three/webgpu';
import { Fn, float, renderOutput, screenUV, select, texture, uniform, vec2, vec4 } from 'three/tsl';

type AnyNode = any;
interface CompositorOptions {
  renderer: WebGPURenderer;
  scene: Scene;
  camera: Camera;
}

/** Keep the inexpensive stage sharp without increasing the liquid ray count. */
export class GpuCompositor {
  readonly stageTarget: RenderTarget;
  readonly fluidTarget: RenderTarget;
  readonly stageColor: AnyNode;
  readonly stageDepth: AnyNode;
  private readonly renderer: WebGPURenderer;
  private readonly scene: Scene;
  private readonly camera: Camera;
  private readonly material: MeshBasicNodeMaterial;
  private readonly stageMaterial: MeshBasicNodeMaterial;
  private readonly quad: QuadMesh;
  private readonly stageTexel = uniform(new Vector2(1, 1));
  private readonly fluidTexel = uniform(new Vector2(1, 1));
  private readonly maxSamples: number;
  private outputTransform = '';
  private surface: Mesh | null = null;
  private disposed = false;

  constructor({renderer, scene, camera}: CompositorOptions) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    // Match the actual device, including Three's non-MSAA compatibility path.
    const samples = renderer.samples >= 4 ? 4 : 0;
    this.maxSamples = samples;
    const target = (msaa: number) => new RenderTarget(1, 1, {
      type: HalfFloatType, minFilter: LinearFilter, magFilter: LinearFilter,
      generateMipmaps: false, depthBuffer: true, stencilBuffer: false,
      depthTexture: new DepthTexture(1, 1, FloatType), samples: msaa,
    });
    this.stageTarget = target(samples);
    this.fluidTarget = target(0);
    this.stageTarget.texture.name = 'ANOMALY stage color';
    this.fluidTarget.texture.name = 'ANOMALY liquid color';
    for (const buffer of [this.stageTarget, this.fluidTarget]) {
      buffer.depthTexture!.minFilter = NearestFilter;
      buffer.depthTexture!.magFilter = NearestFilter;
    }
    this.stageColor = texture(this.stageTarget.texture, screenUV);
    this.stageDepth = texture(this.stageTarget.depthTexture!, screenUV);
    this.material = new MeshBasicNodeMaterial({depthTest: false, depthWrite: false});
    this.material.name = 'ANOMALY depth-aware liquid composite';
    this.stageMaterial = new MeshBasicNodeMaterial({depthTest: false, depthWrite: false});
    this.stageMaterial.name = 'ANOMALY stage output';
    this.updateOutput(renderer.toneMapping, renderer.outputColorSpace, true);
    this.quad = new QuadMesh(this.material);
  }

  private updateOutput(toneMapping: WebGPURenderer['toneMapping'], colorSpace: string, force = false): void {
    const key = `${toneMapping}:${colorSpace}`;
    if (!force && key === this.outputTransform) return;
    this.material.fragmentNode = renderOutput(this.compositeNode(), toneMapping, colorSpace);
    this.material.needsUpdate = true;
    this.stageMaterial.fragmentNode = renderOutput(vec4(this.stageColor.sample(screenUV).rgb, 1), toneMapping, colorSpace);
    this.stageMaterial.needsUpdate = true;
    this.outputTransform = key;
  }

  private compositeNode(): AnyNode {
    const fluidColor: AnyNode = texture(this.fluidTarget.texture, screenUV);
    const fluidDepth: AnyNode = texture(this.fluidTarget.depthTexture!, screenUV);
    const samples = this.stageTarget.samples;
    return Fn(() => {
      const at: AnyNode = screenUV;
      const stage: AnyNode = this.stageColor.sample(at).rgb;
      // All targets contain linear light. The output pass applies the renderer's
      // normal tone mapping and color conversion exactly once.
      const liquid: AnyNode = vec4(0).toVar();
      // A small subpixel footprint smooths only the liquid coverage. Clear black
      // plus alpha zero makes filtered RGB premultiplied, avoiding dark fringes.
      for (const [x, y] of [[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]]) {
        liquid.addAssign(fluidColor.sample(at.add(this.stageTexel.mul(vec2(x, y)))).mul(.25));
      }
      // A bilinear color footprint can straddle a discarded ray. Use the nearest
      // liquid depth among that footprint instead of its cleared far-plane texel.
      let liquidZ: AnyNode = float(1);
      for (const [x, y] of [[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]]) {
        liquidZ = liquidZ.min(fluidDepth.sample(at.add(this.fluidTexel.mul(vec2(x, y)))).r);
      }
      let stageZ: AnyNode = this.stageDepth.sample(at).r;
      // Multisampled depth textures are loaded by sample index in Three's WGSL
      // backend. Conservatively preserve even a thin ring's partially covered pixel.
      for (let sample = 1; sample < samples; sample++) {
        stageZ = stageZ.min(this.stageDepth.sample(at).level(sample).r);
      }
      const visible: AnyNode = select(liquidZ.lessThanEqual(stageZ.add(1e-5)), 1, 0);
      const alpha: AnyNode = liquid.a.clamp(0, 1).mul(visible);
      return vec4(liquid.rgb.mul(visible).add(stage.mul(alpha.oneMinus())), 1);
    })();
  }

  setSurface(surface: Mesh): void {
    this.surface = surface;
  }

  /** Physical pixel dimensions; the caller owns display and liquid budgets. */
  resize(stageWidth: number, stageHeight: number, fluidWidth: number, fluidHeight: number): void {
    const integer = (value: number) => Math.max(1, Math.floor(Number.isFinite(value) ? value : 1));
    const sw = integer(stageWidth), sh = integer(stageHeight);
    const fw = integer(fluidWidth), fh = integer(fluidHeight);
    // Native 4K plus four HDR/depth samples would consume roughly 460 MB for
    // the stage alone. At that density keep native pixels and omit extra MSAA.
    const samples = sw * sh <= 4_000_000 ? this.maxSamples : 0;
    if (this.stageTarget.samples !== samples) {
      this.stageTarget.samples = samples;
      this.stageTarget.dispose();
      this.updateOutput(this.renderer.toneMapping, this.renderer.outputColorSpace, true);
      if (this.surface) {
        for (const material of Array.isArray(this.surface.material) ? this.surface.material : [this.surface.material]) {
          material.needsUpdate = true;
        }
      }
    }
    this.stageTarget.setSize(sw, sh);
    this.fluidTarget.setSize(fw, fh);
    this.stageTexel.value.set(1 / sw, 1 / sh);
    this.fluidTexel.value.set(1 / fw, 1 / fh);
  }

  render(): void {
    if (this.disposed) throw new Error('The liquid compositor was disposed.');
    const {renderer, scene, camera, surface} = this;
    if (!surface) throw new Error('The liquid compositor has no surface.');
    const output = renderer.getRenderTarget();
    const face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
    const autoClear = renderer.autoClear;
    const toneMapping = renderer.toneMapping, outputColorSpace = renderer.outputColorSpace;
    const depth = renderer.depth;
    const background = scene.background;
    const clearColor = renderer.getClearColor(new Color());
    const clearAlpha = renderer.getClearAlpha();
    const cameraLayers = camera.layers.mask, surfaceLayers = surface.layers.mask;
    const visible = surface.visible;
    const quadMaterial = this.quad.material;
    try {
      renderer.autoClear = true;
      surface.visible = false;
      renderer.setRenderTarget(this.stageTarget);
      renderer.render(scene, camera);

      if (visible) {
        // The scene and liquid keep their world transforms and environment. A
        // temporary reserved layer isolates only the liquid for the second pass.
        surface.visible = true;
        surface.layers.set(30);
        camera.layers.set(30);
        scene.background = null;
        renderer.setClearColor(0x000000, 0);
        renderer.setRenderTarget(this.fluidTarget);
        renderer.render(scene, camera);
      }

      renderer.setRenderTarget(output, face, mip);
      this.updateOutput(output ? NoToneMapping : toneMapping,
        output ? ColorManagement.workingColorSpace : outputColorSpace);
      // Like Three's RenderPipeline, transform color in this final shader so
      // render() does not allocate another full-size HDR/MSAA output target.
      renderer.toneMapping = NoToneMapping;
      renderer.outputColorSpace = ColorManagement.workingColorSpace;
      renderer.depth = false;
      // Particle mode still uses the explicit output transform, avoiding a
      // hidden HDR framebuffer. Its shader never reads the previous liquid frame.
      this.quad.material = visible ? this.material : this.stageMaterial;
      this.quad.render(renderer);
    } finally {
      surface.visible = visible;
      surface.layers.mask = surfaceLayers;
      camera.layers.mask = cameraLayers;
      scene.background = background;
      renderer.setClearColor(clearColor, clearAlpha);
      renderer.autoClear = autoClear;
      renderer.toneMapping = toneMapping;
      renderer.outputColorSpace = outputColorSpace;
      renderer.depth = depth;
      this.quad.material = quadMaterial;
      renderer.setRenderTarget(output, face, mip);
    }
  }

  /** One covered startup readback; callers must supply top-left texture UVs. */
  async validateSurface(u: number, v: number): Promise<void> {
    if (this.disposed) throw new Error('The liquid compositor was disposed.');
    if (!Number.isFinite(u) || !Number.isFinite(v)) throw new Error('Invalid liquid verification position.');
    const target = this.fluidTarget;
    const width = Math.min(3, target.width), height = Math.min(3, target.height);
    const x = Math.max(0, Math.min(target.width - width, Math.floor(u * target.width) - 1));
    const y = Math.max(0, Math.min(target.height - height, Math.floor(v * target.height) - 1));
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const pixels = await Promise.race([
        this.renderer.readRenderTargetPixelsAsync(target, x, y, width, height),
        new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('Liquid verification timed out.')), 10000); }),
      ]);
      // r184's WebGPU readback retains 256-byte row padding. Do not mistake
      // padding for pixels or assume a packed width*height*4 result.
      const stride = pixels.length === width * height * 4 ? width * 4 :
        Math.ceil(width * 4 * pixels.BYTES_PER_ELEMENT / 256) * 256 / pixels.BYTES_PER_ELEMENT;
      const channel = (index: number) => pixels instanceof Uint16Array ?
        DataUtils.fromHalfFloat(pixels[index]!) : pixels[index]!;
      for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
        const index = row * stride + column * 4;
        const values = [channel(index), channel(index + 1), channel(index + 2), channel(index + 3)];
        if (values.every(Number.isFinite) && values[3]! > .01 &&
          Math.max(values[0]!, values[1]!, values[2]!) > 1e-5) return;
      }
      throw new Error('The liquid surface did not render.');
    } finally {
      clearTimeout(timer);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.stageTarget.dispose();
    this.fluidTarget.dispose();
    this.material.dispose();
    this.stageMaterial.dispose();
    // QuadMesh's triangle geometry is shared globally and is not ours to free.
  }
}
