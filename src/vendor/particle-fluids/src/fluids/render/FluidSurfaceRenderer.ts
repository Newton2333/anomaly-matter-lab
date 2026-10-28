import {
  BackSide,
  BoxGeometry,
  Color,
  DirectionalLight,
  Mesh,
  Vector2,
  Vector3,
  type Box3,
  type Camera,
  type Scene,
  type Texture,
} from 'three';
import { MeshBasicNodeMaterial, type WebGPURenderer } from 'three/webgpu';
import {
  Break,
  Discard,
  Fn,
  If,
  Loop,
  cameraPosition,
  cameraProjectionMatrix,
  cameraProjectionMatrixInverse,
  cameraViewMatrix,
  cameraWorldMatrix,
  float,
  getViewPosition,
  instancedArray,
  mix,
  pmremTexture,
  positionWorld,
  reflect,
  refract,
  screenUV,
  select,
  texture3D,
  uniform,
  vec2,
  vec3,
  vec4,
  viewportDepthTexture,
  viewportSharedTexture,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import { PrimitiveSet, SDFCollider, type ParticleRange } from '../../core/index.js';
import type { FluidSystem } from '../FluidSystem.js';
import { releaseStorageBuffers } from '../../core/particles.js';
import { FIELD_BAND, SurfaceField } from './SurfaceField.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Look of the liquid. Colours are sRGB hex; distances are metres. */
export interface FluidAppearance {
  /** Colour white light takes on after crossing `attenuationDistance` of liquid. */
  readonly color: number;
  readonly attenuationDistance: number;
  /** Light scattered back out of the body (0 = clear water, 1 = milky/honey). */
  readonly scattering: number;
  /** Index of refraction (water is 1.333). */
  readonly ior: number;
  /** Blurs reflections and widens highlights, 0–1. */
  readonly roughness: number;
  /** Strength of the environment reflection. */
  readonly envIntensity: number;
  /** 0 = dielectric liquid, 1 = liquid metal tinted by `metalColor`. */
  readonly metalness: number;
  readonly metalColor: number;
}

const DEFAULT_APPEARANCE: FluidAppearance = {
  color: 0x2a8fb0,
  attenuationDistance: 0.6,
  scattering: 0.08,
  ior: 1.333,
  roughness: 0.04,
  envIntensity: 1,
  metalness: 0,
  metalColor: 0xc8d2da,
};

export interface FluidSurfaceRendererOptions {
  readonly renderer: WebGPURenderer;
  /** Optional opaque scene buffers for rendering liquid at an independent resolution. */
  readonly externalScene?: { readonly color: Any; readonly depth: Any };
  /**
   * Scene the surface is drawn in. Its environment map and its main
   * directional light shade the liquid; both are re-read on each
   * {@link FluidSurfaceRenderer.update}.
   */
  readonly scene: Scene;
  /**
   * Used only by {@link FluidSurfaceRenderer.pick}, which casts rays from it.
   * Drawing uses whichever camera renders the mesh.
   */
  readonly camera: Camera;
  /** Box the liquid can reach. Leave a few centimetres of margin around walls. */
  readonly bounds: Box3;
  /** Colliders the liquid wets, drawing a meniscus where it meets them. */
  readonly colliders?: readonly (PrimitiveSet | SDFCollider)[];
  /** Moving shapes cut out of the liquid every frame. */
  readonly carve?: PrimitiveSet;
  /** Particles of floating or submerged solids the liquid wets. */
  readonly solids?: ParticleRange;
  /**
   * Stretch fast particles along their velocity, in seconds, to smooth thin
   * streams. Default 0; must be ≥ 0.
   */
  readonly motionStretch?: number;
  /**
   * Voxels in the surface grid, which sets its resolution. Each voxel costs
   * about 76 bytes. Default: {@link FluidSurfaceRenderer.defaultVoxelBudget}.
   */
  readonly voxelBudget?: number;
  readonly appearance?: Partial<FluidAppearance>;
  /**
   * Draw pockets cut into the liquid (see `carve`) with a bright rim
   * and an optional smoke fill. Makes the transmitted-light march longer.
   * `smokeDensity` is extinction per metre, ≥ 0.
   */
  readonly cavities?: { readonly smokeColor: number; readonly smokeDensity: number };
  /**
   * Bend light passing through the surface. Default `true`. Turn it off to
   * show the scene behind unbent, which avoids dark smears where objects
   * cross the surface.
   */
  readonly refraction?: boolean;
}

const MAX_STEPS = 128;
const THICKNESS_STEPS = 32;
const SSR_STEPS = 24;

/**
 * Ray-marched liquid surface.
 *
 * `SurfaceField` turns the particles into a smooth signed field each frame;
 * this class draws it with a back-faced proxy box whose fragment shader
 * sphere-traces the field, refracts the opaque scene behind it, absorbs
 * light along the refracted path (Beer–Lambert), and adds Fresnel-weighted
 * environment reflection plus a GGX key-light highlight. The hit depth is
 * written so later transparent objects, ambient occlusion, and fog see the
 * real surface.
 */
export class FluidSurfaceRenderer {
  readonly mesh: Mesh;
  readonly fluid: FluidSystem;
  /** @internal */
  readonly field: SurfaceField;
  private readonly reflectionsUniform = uniform(1, 'float');
  private readonly cavitySmokeDensity = uniform(0, 'float');
  private readonly appearance: {
    readonly color: ReturnType<typeof uniform<'color', Color>>;
    readonly attenuationDistance: ReturnType<typeof uniform<'float', number>>;
    readonly scattering: ReturnType<typeof uniform<'float', number>>;
    readonly ior: ReturnType<typeof uniform<'float', number>>;
    readonly roughness: ReturnType<typeof uniform<'float', number>>;
    readonly envIntensity: ReturnType<typeof uniform<'float', number>>;
    readonly metalness: ReturnType<typeof uniform<'float', number>>;
    readonly metalColor: ReturnType<typeof uniform<'color', Color>>;
  };

  private readonly opts: FluidSurfaceRendererOptions;
  private readonly primitiveSets: readonly PrimitiveSet[];
  private readonly sdfColliders: readonly SDFCollider[];
  /** Collider versions the wetted walls were last computed for; unset until the first update. */
  private wallVersions: string | undefined;
  /** Builds the fragment shader for the current environment map. */
  private readonly buildShade: () => Any;
  private environment: Texture | null;
  /** Environment samplers in the current shader, retargeted when the map changes. */
  private pmremNodes: Any[] = [];
  private disposed = false;
  private readonly sunDirection = uniform(new Vector3(0.35, 0.9, 0.45).normalize());
  private readonly sunColor = uniform(new Color(0, 0, 0));
  private readonly environmentIntensity = uniform(1, 'float');
  private sun: DirectionalLight | undefined;
  private readonly pickOrigin = uniform(new Vector3());
  private readonly pickDirection = uniform(new Vector3(0, 0, -1));
  private readonly pickResult = instancedArray(1, 'vec4');
  private readonly pickKernel: ComputeNode;

  constructor(fluid: FluidSystem, options: FluidSurfaceRendererOptions) {
    const motionStretch = options.motionStretch ?? 0;
    if (!(motionStretch >= 0) || !Number.isFinite(motionStretch)) {
      throw new Error(`FluidSurfaceRenderer: motionStretch must be ≥ 0, got ${motionStretch}`);
    }
    const voxelBudget =
      options.voxelBudget ?? FluidSurfaceRenderer.defaultVoxelBudget(fluid.range.count);
    if (!(voxelBudget >= 1) || !Number.isFinite(voxelBudget)) {
      throw new Error(`FluidSurfaceRenderer: voxelBudget must be ≥ 1, got ${voxelBudget}`);
    }
    this.fluid = fluid;
    this.opts = options;
    const colliders = options.colliders ?? [];
    this.primitiveSets = colliders.filter((c): c is PrimitiveSet => c instanceof PrimitiveSet);
    this.sdfColliders = colliders.filter((c): c is SDFCollider => c instanceof SDFCollider);
    const look: FluidAppearance = { ...DEFAULT_APPEARANCE };
    for (const [key, value] of Object.entries(options.appearance ?? {}))
      if (value !== undefined) Object.assign(look, { [key]: finiteAppearance(key, value) });
    this.appearance = {
      color: uniform(new Color(look.color)),
      attenuationDistance: uniform(look.attenuationDistance, 'float'),
      scattering: uniform(look.scattering, 'float'),
      ior: uniform(look.ior, 'float'),
      roughness: uniform(look.roughness, 'float'),
      envIntensity: uniform(look.envIntensity, 'float'),
      metalness: uniform(look.metalness, 'float'),
      metalColor: uniform(new Color(look.metalColor)),
    };

    const bounds = options.bounds;
    this.field = new SurfaceField({
      fluidSystem: fluid,
      min: bounds.min,
      max: bounds.max,
      voxelBudget,
      colliders: this.primitiveSets,
      carve: options.carve,
      sdfColliders: this.sdfColliders,
      solids: options.solids,
      motionStretch,
    });
    const field = this.field;
    const origin = uniform(field.origin.clone());
    const extent = uniform(field.extent.clone());
    const voxel = field.voxel;

    const sample = (p: Any): Any =>
      (texture3D(field.texture, p.sub(origin).div(extent)) as Any).level(0);
    const distanceAt = (p: Any): Any => sample(p).r.mul(voxel);

    /** Sphere-trace [start, end] along a ray. Returns (t, hit ? 1 : 0). */
    const trace = (rayOrigin: Any, direction: Any, start: Any, end: Any): Any => {
      const t: Any = start.toVar();
      const previousT: Any = start.toVar();
      const previous: Any = float(FIELD_BAND * voxel).toVar();
      const hit: Any = float(0).toVar();
      Loop(MAX_STEPS, () => {
        const d: Any = distanceAt(rayOrigin.add(direction.mul(t))).toVar();
        If(d.lessThan(0), () => {
          hit.assign(1);
          Break();
        });
        previousT.assign(t);
        previous.assign(d);
        t.addAssign(d.mul(0.8).max(voxel * 0.3));
        If(t.greaterThan(end), () => {
          Break();
        });
      });
      // Two secant refinements between the last outside and first inside sample.
      If(hit.greaterThan(0).and(t.greaterThan(previousT)), () => {
        const lo: Any = previousT.toVar();
        const hi: Any = t.toVar();
        const dLo: Any = previous.toVar();
        const dHi: Any = distanceAt(rayOrigin.add(direction.mul(hi))).toVar();
        for (let k = 0; k < 2; k++) {
          const mid: Any = lo.add(hi.sub(lo).mul(dLo.div(dLo.sub(dHi).max(1e-6)))).toVar();
          const dMid: Any = distanceAt(rayOrigin.add(direction.mul(mid))).toVar();
          If(dMid.lessThan(0), () => {
            hi.assign(mid);
            dHi.assign(dMid);
          }).Else(() => {
            lo.assign(mid);
            dLo.assign(dMid);
          });
        }
        t.assign(lo.add(hi.sub(lo).mul(dLo.div(dLo.sub(dHi).max(1e-6)))));
      });
      return vec2(t, hit);
    };

    const boxInterval = (rayOrigin: Any, direction: Any): Any => {
      const safe: Any = select(direction.abs().lessThan(1e-6), vec3(1e-6), direction);
      const inv: Any = vec3(1).div(safe);
      const a: Any = origin.sub(rayOrigin).mul(inv);
      const b: Any = origin.add(extent).sub(rayOrigin).mul(inv);
      const lo: Any = a.min(b),
        hi: Any = a.max(b);
      return vec2(lo.x.max(lo.y).max(lo.z).max(0), hi.x.min(hi.y).min(hi.z));
    };

    // ---- Surface hit (shared by colour and depth outputs) ----------------
    const viewDirection: Any = positionWorld.sub(cameraPosition).normalize().toVar();
    const sceneDepthAt = (uv: Any = screenUV): Any => options.externalScene
      ? options.externalScene.depth.sample(uv) : viewportDepthTexture(uv);
    const sceneColorAt = (uv: Any): Any => options.externalScene
      ? options.externalScene.color.sample(uv) : viewportSharedTexture(uv);
    const opaqueDistance: Any = getViewPosition(
      screenUV,
      sceneDepthAt().r,
      cameraProjectionMatrixInverse,
    )
      .length()
      .toVar();
    const surface: Any = Fn(() => {
      const interval: Any = boxInterval(cameraPosition, viewDirection).toVar();
      const end: Any = interval.y.min(opaqueDistance);
      Discard(end.lessThanEqual(interval.x));
      const result: Any = trace(cameraPosition, viewDirection, interval.x, end).toVar();
      Discard(result.y.lessThan(0.5).or(result.x.greaterThan(end)));
      return result.x;
    })().toVar();
    const hitPoint: Any = cameraPosition.add(viewDirection.mul(surface)).toVar();

    const project = (world: Any): Any => {
      const clip: Any = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(world, 1));
      return clip.xyz.div(clip.w);
    };

    // A missing environment falls back to a sky gradient. The two need
    // different shaders, so `update()` rebuilds this when the scene gains or
    // loses its map; swapping one map for another only retargets the samplers.
    this.environment = options.scene.environment;
    const envSample = (direction: Any, roughness: Any): Any => {
      if (!this.environment)
        return mix(vec3(0.05, 0.06, 0.08), vec3(0.5, 0.55, 0.6), direction.y.mul(0.5).add(0.5));
      const node: Any = pmremTexture(this.environment, direction, roughness);
      this.pmremNodes.push(node);
      return node.rgb.mul(this.environmentIntensity);
    };

    const a = this.appearance as Any;
    const cavities = options.cavities;
    const smokeColor: Any = uniform(new Color(cavities?.smokeColor ?? 0x808080));
    if (cavities) this.smokeDensity = cavities.smokeDensity;
    this.buildShade = () =>
      Fn(() => {
        // Normal from the field gradient over one voxel; the field is already
        // smooth, so this stays stable across frames without extra filtering.
        const e = voxel;
        const n: Any = vec3(
          distanceAt(hitPoint.add(vec3(e, 0, 0))).sub(distanceAt(hitPoint.sub(vec3(e, 0, 0)))),
          distanceAt(hitPoint.add(vec3(0, e, 0))).sub(distanceAt(hitPoint.sub(vec3(0, e, 0)))),
          distanceAt(hitPoint.add(vec3(0, 0, e))).sub(distanceAt(hitPoint.sub(vec3(0, 0, e)))),
        )
          .normalize()
          .toVar();
        const v: Any = viewDirection.negate();
        // Grazing normals from a coarse field can face away; keep them visible.
        n.assign(n.add(v.mul(n.dot(v).negate().max(0).mul(1.02))).normalize());
        const cosV: Any = n.dot(v).clamp(1e-4, 1).toVar();
        const roughness: Any = a.roughness.clamp(0.02, 1);

        const toUV = (world: Any): Any => {
          const ndc: Any = project(world);
          return vec2(ndc.x.mul(0.5).add(0.5), ndc.y.mul(-0.5).add(0.5)).clamp(0.001, 0.999);
        };
        const opaqueAt = (uv: Any): Any =>
          cameraWorldMatrix.mul(
            vec4(getViewPosition(uv, sceneDepthAt(uv).r, cameraProjectionMatrixInverse), 1),
          ).xyz;
        const body: Any = vec3(0).toVar();
        // ANOMALY optimization: a fully metallic surface never transmits light.
        // Branch uniformly before the thickness march/depth reprojections instead
        // of calculating a dielectric result that mix(..., metal, 1) discards.
        // Intermediate metalness still executes the unchanged dielectric path.
        If(a.metalness.lessThan(1), () => {
          // Refraction and absorption along the transmitted ray.
          const eta: Any = float(1).div(a.ior);
          const transmitted: Any = (
            options.refraction === false ? viewDirection : refract(viewDirection, n, eta)
          ).toVar();
          const optical: Any = float(0).toVar();
          const travel: Any = float(voxel * 0.5).toVar();
          const exitInterval: Any = boxInterval(hitPoint, transmitted);
          const stepLength = voxel * 1.5;
          const tint: Any = a.color.max(vec3(1e-3));
          const attenuation: Any = a.attenuationDistance.max(1e-4);
          // Cavity state: smoke path length and the rim reflection of the pocket
          // being crossed. Both count only once the ray re-enters liquid, so the
          // body's own back face never registers as a bubble.
          const smoke: Any = float(0).toVar();
          const airRun: Any = float(0).toVar();
          const rim: Any = vec3(0).toVar();
          const pendingRim: Any = vec3(0).toVar();
          const wasInside: Any = float(1).toVar();
          Loop(cavities ? THICKNESS_STEPS + 16 : THICKNESS_STEPS, () => {
            const q: Any = hitPoint.add(transmitted.mul(travel));
            const s: Any = sample(q).toVar();
            optical.addAssign(s.g.clamp(0, 1).mul(stepLength));
            travel.addAssign(stepLength);
            if (cavities) {
              If(s.r.lessThan(0), () => {
                smoke.addAssign(airRun);
                rim.addAssign(pendingRim);
                airRun.assign(0);
                pendingRim.assign(vec3(0));
                wasInside.assign(1);
              }).Else(() => {
                If(wasInside.greaterThan(0.5), () => {
                  // Leaving liquid into air: reflect the environment, with total
                  // internal reflection past the critical angle.
                  const g: Any = vec3(
                    distanceAt(q.add(vec3(voxel, 0, 0))).sub(distanceAt(q.sub(vec3(voxel, 0, 0)))),
                    distanceAt(q.add(vec3(0, voxel, 0))).sub(distanceAt(q.sub(vec3(0, voxel, 0)))),
                    distanceAt(q.add(vec3(0, 0, voxel))).sub(distanceAt(q.sub(vec3(0, 0, voxel)))),
                  ).normalize();
                  const cosI: Any = transmitted.dot(g).clamp(0, 1);
                  const sinT: Any = cosI.mul(cosI).oneMinus().sqrt().mul(a.ior);
                  const f: Any = select(
                    sinT.greaterThanEqual(1),
                    float(1),
                    float(0.02).add(float(0.98).mul(cosI.oneMinus().pow(5))),
                  );
                  const seen: Any = tint.pow(vec3(optical.div(attenuation)));
                  pendingRim.assign(
                    envSample(reflect(transmitted, g.negate()), float(0.05))
                      .mul(a.envIntensity)
                      .mul(f)
                      .mul(seen),
                  );
                });
                airRun.addAssign(stepLength);
                wasInside.assign(0);
              });
            }
            // In cavity mode keep marching through bubbles as large as the field band.
            If(
              s.r
                .greaterThan(cavities ? FIELD_BAND - 0.5 : 1.5)
                .or(travel.greaterThan(exitInterval.y)),
              () => {
                Break();
              },
            );
          });
          // Find where the bent ray meets the opaque scene: start from the
          // straight-through distance, then re-project against the depth buffer
          // twice. Grazing rays bend steeply, so the first guess overshoots.
          // The march already found where the bent ray leaves the liquid (at the
          // floor of a pool, say); the straight-through distance can overshoot a
          // container wall into the far background.
          const reach: Any = opaqueDistance
            .sub(surface)
            .clamp(0, 1.5)
            .min(travel.add(voxel * 2))
            .toVar();
          for (let k = 0; k < 2; k++) {
            const guess: Any = opaqueAt(toUV(hitPoint.add(transmitted.mul(reach))));
            const along: Any = guess.sub(hitPoint).dot(transmitted);
            // A step that lands on something in front of the surface (a floating
            // duck, say) says nothing about the bent ray; keep the last estimate.
            const behindSurface: Any = guess.sub(cameraPosition).length().greaterThan(surface);
            reach.assign(
              select(
                behindSurface.and(along.greaterThan(0)),
                along.min(travel.add(voxel * 4)),
                reach,
              ),
            );
          }
          // Prefer the refined point, then the liquid exit point; never pull colour
          // from geometry in front of the surface.
          const occluded = (uv: Any): Any =>
            opaqueAt(uv).sub(cameraPosition).length().lessThan(surface);
          const bentUV: Any = toUV(hitPoint.add(transmitted.mul(reach))).toVar();
          const exitUV: Any = toUV(hitPoint.add(transmitted.mul(travel))).toVar();
          const refractUV: Any = select(
            occluded(bentUV).not(),
            bentUV,
            select(occluded(exitUV).not(), exitUV, screenUV),
          );
          const behind: Any = sceneColorAt(refractUV).rgb;
          const depth: Any = optical.div(attenuation);
          const transmittance: Any = tint.pow(vec3(depth));
          // Light scattered back toward the eye travels about half the path, so it
          // takes the colour of the medium at half depth rather than the colour
          // the medium removed.
          const ambient: Any = envSample(n, float(1)).add(
            this.sunColor.mul(n.dot(this.sunDirection).mul(0.5).add(0.5).mul(0.3)),
          );
          const opacity: Any = float(1).sub(transmittance.dot(vec3(1 / 3)));
          const scattered: Any = ambient
            .mul(tint.pow(vec3(depth.mul(0.5))))
            .mul(a.scattering)
            .mul(opacity);
          body.assign(behind.mul(transmittance).add(scattered));
          if (cavities) {
            const smokeClear: Any = smoke.mul(this.cavitySmokeDensity.negate()).exp();
            const smokeLit: Any = envSample(vec3(0, 1, 0), float(1))
              .add(this.sunColor.mul(0.15))
              .mul(smokeColor);
            body.assign(mix(smokeLit, behind, smokeClear).mul(transmittance).add(scattered).add(rim));
          }
        });

        // Surface reflection: environment plus a GGX highlight from the key light.
        const f0: Any = a.ior.sub(1).div(a.ior.add(1)).pow(2);
        const fresnel: Any = f0.add(f0.oneMinus().mul(cosV.oneMinus().pow(5)));
        const reflectDirection: Any = reflect(viewDirection, n).toVar();
        const reflected: Any = envSample(reflectDirection, roughness).mul(a.envIntensity).toVar();
        // Screen-space reflections: march the reflected ray against the depth
        // buffer with growing steps, refine the crossing, and fade toward the
        // environment where the ray leaves the screen or finds nothing.
        If(this.reflectionsUniform.greaterThan(0.5), () => {
          const t: Any = float(voxel).toVar();
          const previous: Any = float(0).toVar();
          const found: Any = float(0).toVar();
          const hitUV: Any = vec2(0).toVar();
          Loop(SSR_STEPS, () => {
            const p: Any = hitPoint.add(reflectDirection.mul(t));
            const clip: Any = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(p, 1));
            If(clip.w.lessThanEqual(0), () => {
              Break();
            });
            const uv: Any = vec2(
              clip.x.div(clip.w).mul(0.5).add(0.5),
              clip.y.div(clip.w).mul(-0.5).add(0.5),
            );
            If(
              uv.x.lessThan(0).or(uv.x.greaterThan(1)).or(uv.y.lessThan(0)).or(uv.y.greaterThan(1)),
              () => {
                Break();
              },
            );
            const sceneDistance: Any = opaqueAt(uv).sub(cameraPosition).length();
            const depth: Any = p.sub(cameraPosition).length().sub(sceneDistance);
            If(depth.greaterThan(0).and(depth.lessThan(t.mul(0.35).add(0.05))), () => {
              // Bisect between the last point in front and this one behind.
              const lo: Any = previous.toVar();
              const hi: Any = t.toVar();
              for (let k = 0; k < 4; k++) {
                const mid: Any = lo.add(hi).mul(0.5);
                const q: Any = hitPoint.add(reflectDirection.mul(mid));
                const qUV: Any = toUV(q);
                const behindScene: Any = q
                  .sub(cameraPosition)
                  .length()
                  .greaterThan(opaqueAt(qUV).sub(cameraPosition).length());
                hi.assign(select(behindScene, mid, hi));
                lo.assign(select(behindScene, lo, mid));
              }
              hitUV.assign(toUV(hitPoint.add(reflectDirection.mul(hi))));
              found.assign(1);
              Break();
            });
            previous.assign(t);
            t.mulAssign(1.3);
          });
          const edge: Any = hitUV.min(vec2(1).sub(hitUV)).mul(12).clamp(0, 1);
          const confidence: Any = found
            .mul(edge.x.mul(edge.y))
            .mul(roughness.mul(-2.5).add(1).clamp(0, 1));
          reflected.assign(mix(reflected, sceneColorAt(hitUV).rgb, confidence));
        });
        const l: Any = this.sunDirection;
        const h: Any = l.add(v).normalize();
        const nl: Any = n.dot(l).max(0);
        const nh: Any = n.dot(h).max(0);
        const alpha: Any = roughness.mul(roughness);
        const a2: Any = alpha.mul(alpha);
        const denom: Any = nh.mul(nh).mul(a2.sub(1)).add(1);
        const distribution: Any = a2.div(denom.mul(denom).mul(Math.PI));
        const visibility: Any = float(0.5).div(
          nl
            .mul(cosV.mul(cosV).mul(a2.oneMinus()).add(a2).sqrt())
            .add(cosV.mul(nl.mul(nl).mul(a2.oneMinus()).add(a2).sqrt()))
            .max(1e-5),
        );
        const specular: Any = this.sunColor.mul(distribution.mul(visibility).mul(nl)).min(vec3(64));

        const dielectric: Any = mix(body, reflected, fresnel).add(specular.mul(fresnel));
        const metalF: Any = a.metalColor.add(a.metalColor.oneMinus().mul(cosV.oneMinus().pow(5)));
        const metal: Any = reflected.add(specular).mul(metalF);
        return vec4(mix(dielectric, metal, a.metalness) as Any, 1);
      })();

    const material = new MeshBasicNodeMaterial({
      transparent: true,
      side: BackSide,
      depthTest: false,
      depthWrite: true,
      fog: false,
    });
    material.fragmentNode = this.buildShade();
    const hitClip: Any = project(hitPoint);
    material.depthNode = hitClip.z.clamp(0, 1);

    const size = field.extent;
    this.mesh = new Mesh(new BoxGeometry(size.x, size.y, size.z), material);
    this.mesh.position.copy(field.origin).addScaledVector(size, 0.5);
    this.mesh.name = 'FluidSurface';
    this.mesh.frustumCulled = false;
    // Draw before other transparent objects (e.g. glass walls) so they layer on top.
    this.mesh.renderOrder = -1;

    // Single-ray pick for interaction, traced against the same field.
    this.pickKernel = Fn(() => {
      const interval: Any = boxInterval(this.pickOrigin, this.pickDirection).toVar();
      const result: Any = trace(this.pickOrigin, this.pickDirection, interval.x, interval.y);
      const hit: Any = result.y.greaterThan(0.5).and(result.x.lessThanEqual(interval.y));
      this.pickResult
        .element(0)
        .assign(vec4(this.pickOrigin.add(this.pickDirection.mul(result.x)), select(hit, 1, 0)));
    })()
      .compute(1)
      .setName('FluidSurfaceRenderer.pick');

    this.findSun();
  }

  /** The voxel budget used when none is given: finer particles earn a finer grid, within a memory ceiling. */
  static defaultVoxelBudget(particleCount: number): number {
    return particleCount >= 50_000 ? 1_200_000 : particleCount >= 20_000 ? 900_000 : 600_000;
  }

  /** Screen-space reflections of the scene. When off, the surface reflects only the environment. */
  get reflections(): boolean {
    return this.reflectionsUniform.value > 0.5;
  }
  set reflections(enabled: boolean) {
    this.reflectionsUniform.value = enabled ? 1 : 0;
  }

  /**
   * Extinction of the smoke inside cavities, 1/m. Requires `cavities` in the
   * options.
   */
  get smokeDensity(): number {
    this.assertCavities();
    return this.cavitySmokeDensity.value;
  }
  set smokeDensity(value: number) {
    this.assertCavities();
    if (!(value >= 0) || !Number.isFinite(value)) {
      throw new Error(`FluidSurfaceRenderer: smokeDensity must be ≥ 0, got ${value}`);
    }
    this.cavitySmokeDensity.value = value;
  }

  /** Change how the liquid looks. */
  setAppearance(appearance: Partial<FluidAppearance>): void {
    // Validate everything first, so a bad field leaves the look unchanged.
    const entries = (Object.entries(appearance) as [keyof FluidAppearance, number][])
      .filter(([, value]) => value !== undefined)
      .map(([key, value]) => [key, finiteAppearance(key, value)] as const);
    for (const [key, value] of entries) {
      const node = this.appearance[key] as Any;
      if (node.value instanceof Color) node.value.set(value);
      else node.value = value;
    }
  }

  /** Rebuild the surface from the particles. Call once per frame, before rendering. */
  async update(): Promise<void> {
    this.assertAlive();
    if (!this.sun) this.findSun();
    if (this.sun) {
      this.sun.updateMatrixWorld();
      this.sun.target.updateMatrixWorld();
      this.sunDirection.value
        .setFromMatrixPosition(this.sun.matrixWorld)
        .sub(new Vector3().setFromMatrixPosition(this.sun.target.matrixWorld))
        .normalize();
      this.sunColor.value.copy(this.sun.color).multiplyScalar(this.sun.intensity);
    }
    this.environmentIntensity.value = this.opts.scene.environmentIntensity ?? 1;
    this.refreshEnvironment();
    // Wetted walls are computed on the first update, then only after a collider moves.
    const versions = [
      ...this.primitiveSets.map((set) => set.version),
      ...this.sdfColliders.map((collider) => collider.version),
    ].join();
    const kernels =
      versions === this.wallVersions
        ? this.field.kernels
        : [this.field.wallKernel, ...this.field.kernels];
    this.wallVersions = versions;
    await this.opts.renderer.computeAsync(kernels as ComputeNode[]);
  }

  /** World-space point where a viewport ray (uv in [0, 1], y down) meets the liquid. */
  async pick(uv: Vector2): Promise<Vector3 | null> {
    this.assertAlive();
    const camera = this.opts.camera;
    camera.updateMatrixWorld();
    const ndc = new Vector3(uv.x * 2 - 1, 1 - uv.y * 2, 0.5);
    const origin = new Vector3().setFromMatrixPosition(camera.matrixWorld);
    const direction = ndc.unproject(camera).sub(origin).normalize();
    this.pickOrigin.value.copy(origin);
    this.pickDirection.value.copy(direction);
    await this.opts.renderer.computeAsync(this.pickKernel);
    const data = new Float32Array(
      await this.opts.renderer.getArrayBufferAsync(this.pickResult.value),
    );
    if (data[3]! < 0.5) return null;
    return new Vector3(data[0], data[1], data[2]);
  }

  /**
   * Remove the mesh from its parent and free the geometry, material, field
   * texture, and GPU buffers. The renderer can't be used after.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    (this.mesh.material as MeshBasicNodeMaterial).dispose();
    for (const node of this.pmremNodes) node.dispose();
    this.pmremNodes = [];
    this.pickKernel.dispose();
    this.field.dispose(this.opts.renderer);
    releaseStorageBuffers(this.opts.renderer, [this.pickResult]);
  }

  /**
   * Follow `scene.environment`: retarget the samplers when one map replaces
   * another, and rebuild the shader when the map appears or goes away.
   */
  private refreshEnvironment(): void {
    const environment = this.opts.scene.environment;
    if (environment === this.environment) return;
    const rebuild = !environment !== !this.environment;
    this.environment = environment;
    if (!rebuild) {
      for (const node of this.pmremNodes) node.value = environment;
      return;
    }
    const stale = this.pmremNodes;
    this.pmremNodes = [];
    const material = this.mesh.material as MeshBasicNodeMaterial;
    material.fragmentNode = this.buildShade();
    material.needsUpdate = true;
    for (const node of stale) node.dispose();
  }

  private assertCavities(): void {
    if (!this.opts.cavities) {
      throw new Error(
        'FluidSurfaceRenderer: pass `cavities` in the options to enable smokeDensity before changing it',
      );
    }
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('FluidSurfaceRenderer: already disposed');
  }

  private findSun(): void {
    let best: DirectionalLight | undefined;
    this.opts.scene.traverse((object) => {
      if (object instanceof DirectionalLight && (!best || object.castShadow)) best = object;
    });
    this.sun = best;
  }
}

function finiteAppearance(key: string, value: number): number {
  if (!Number.isFinite(value)) {
    throw new Error(`FluidSurfaceRenderer: appearance.${key} must be finite, got ${value}`);
  }
  return value;
}
