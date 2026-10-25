import { Color, Vector3 } from 'three';
import type { MeshStandardMaterial } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  If,
  Fn,
  cameraViewMatrix,
  cross,
  dFdx,
  dFdy,
  faceDirection,
  float,
  normalize,
  positionView,
  texture,
  uint,
  uniform,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { SoftbodySystem } from './SoftbodySystem.js';
import { buildSkinNormalFn, buildSkinPositionFn } from './dlb.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * A node material that skins a mesh to a soft body by dual-quaternion
 * blending (Kavan et al. 2008), reading particle rotations on the GPU:
 *
 * - Local shape matching: each influence uses its particle's own orientation.
 * - Global shape matching: all particles share the body's rotation, which is
 *   converted to a quaternion once; the blend then reduces to linear blend
 *   skinning.
 */
export interface CreateSoftbodySkinMaterialOptions {
  readonly softbody: SoftbodySystem;
  readonly bodyIndex: number;
  /**
   * Material whose color, maps, roughness, and metalness are copied onto
   * the skin material. The geometry needs uvs for the maps to show.
   */
  readonly sourceMaterial?: MeshStandardMaterial;
}

export function createSoftbodySkinMaterial(
  options: CreateSoftbodySkinMaterialOptions,
): MeshStandardNodeMaterial {
  const { softbody, bodyIndex, sourceMaterial } = options;
  const body = softbody.bodies[bodyIndex];
  if (!body) throw new Error(`createSoftbodySkinMaterial: no body ${bodyIndex}`);
  // The skinning shader works relative to the body's rest center.
  const cBarUniform: UniformNode<'vec3', Vector3> = uniform(new Vector3(...body.restCenter));

  let createGetRotationQuat: () => (idx: Any) => Any;
  if (softbody.shapeMatching === 'local') {
    // Local shape matching: each particle has its own orientation.
    const rotation = softbody.particles.rotation;
    createGetRotationQuat = () => (idx: Any) => rotation.element(idx);
  } else {
    // §5.3: shared body rotation. Convert R → q ONCE in the vertex
    // shader and return the same value for every influence. Identical
    // Shoemake branching to `shapeMatchImplicit.ts` Pass 4 — kept inline
    // because the only sharable abstraction would be a helper that
    // takes nine TSL scalars and returns a vec4, and that's no clearer
    // than the inline form.
    const bodyRotations = softbody.bodyRotations;
    const baseRow: Any = uint(3 * bodyIndex);
    const sharedQuat = (): Any => {
      const r0: Any = bodyRotations.element(baseRow).xyz.toVar();
      const r1: Any = bodyRotations.element(baseRow.add(uint(1))).xyz.toVar();
      const r2: Any = bodyRotations.element(baseRow.add(uint(2))).xyz.toVar();
      const m00: Any = r0.x;
      const m01: Any = r0.y;
      const m02: Any = r0.z;
      const m10: Any = r1.x;
      const m11: Any = r1.y;
      const m12: Any = r1.z;
      const m20: Any = r2.x;
      const m21: Any = r2.y;
      const m22: Any = r2.z;
      const trace: Any = m00.add(m11).add(m22);
      const qx: Any = float(0.0).toVar();
      const qy: Any = float(0.0).toVar();
      const qz: Any = float(0.0).toVar();
      const qw: Any = float(1.0).toVar();
      If(trace.greaterThan(float(0.0)), () => {
        // s = 0.5 / sqrt(trace+1) — same formulation as
        // shapeMatchImplicit.ts qpWriteKernel.
        const s: Any = float(0.5).div(trace.add(float(1.0)).sqrt());
        qw.assign(float(0.25).div(s));
        qx.assign(m21.sub(m12).mul(s));
        qy.assign(m02.sub(m20).mul(s));
        qz.assign(m10.sub(m01).mul(s));
      })
        .ElseIf(m00.greaterThan(m11).and(m00.greaterThan(m22)), () => {
          const s: Any = float(2.0).mul(float(1.0).add(m00).sub(m11).sub(m22).sqrt());
          qw.assign(m21.sub(m12).div(s));
          qx.assign(float(0.25).mul(s));
          qy.assign(m01.add(m10).div(s));
          qz.assign(m02.add(m20).div(s));
        })
        .ElseIf(m11.greaterThan(m22), () => {
          const s: Any = float(2.0).mul(float(1.0).add(m11).sub(m00).sub(m22).sqrt());
          qw.assign(m02.sub(m20).div(s));
          qx.assign(m01.add(m10).div(s));
          qy.assign(float(0.25).mul(s));
          qz.assign(m12.add(m21).div(s));
        })
        .Else(() => {
          const s: Any = float(2.0).mul(float(1.0).add(m22).sub(m00).sub(m11).sqrt());
          qw.assign(m10.sub(m01).div(s));
          qx.assign(m02.add(m20).div(s));
          qy.assign(m12.add(m21).div(s));
          qz.assign(float(0.25).mul(s));
        });
      return vec4(qx, qy, qz, qw);
    };
    // Per-`Fn`-invocation closure. The first lane materialises the
    // shared quaternion as a `.toVar()` inside the current Fn body;
    // subsequent lanes reuse the same node reference. The closure is
    // re-created for the position and normal Fns separately, so the
    // memoised node never crosses Fn boundaries.
    createGetRotationQuat = () => {
      let cached: Any | null = null;
      return (_idx: Any): Any => {
        if (cached === null) cached = sharedQuat().toVar();
        return cached;
      };
    };
  }

  const material = new MeshStandardNodeMaterial({
    color: new Color(0xd07030),
    roughness: 0.6,
    metalness: 0,
  });

  // Honour a glTF source material — copy PBR scalars + every texture
  // map slot the upstream `MeshStandardMaterial` exposes. Three.js's
  // node material extends the legacy material so direct property
  // assignment works; the standard PBR shader graph reads these without
  // any explicit `colorNode` / `mapNode` wiring on our side.
  if (sourceMaterial) {
    material.side = sourceMaterial.side;
    if (sourceMaterial.color) material.color.copy(sourceMaterial.color);
    if (sourceMaterial.emissive) material.emissive.copy(sourceMaterial.emissive);
    material.roughness = sourceMaterial.roughness;
    material.metalness = sourceMaterial.metalness;
    material.map = sourceMaterial.map;
    material.normalMap = sourceMaterial.normalMap;
    material.roughnessMap = sourceMaterial.roughnessMap;
    material.metalnessMap = sourceMaterial.metalnessMap;
    material.aoMap = sourceMaterial.aoMap;
    material.aoMapIntensity = sourceMaterial.aoMapIntensity;
    material.emissiveMap = sourceMaterial.emissiveMap;
    material.emissiveIntensity = sourceMaterial.emissiveIntensity;
    if (sourceMaterial.normalScale) {
      material.normalScale.copy(sourceMaterial.normalScale);
    }
  }

  const skinArgs = {
    particles: softbody.particles,
    restOffsets: softbody.restOffsets,
    cBar: cBarUniform,
    createGetRotationQuat,
  };
  // `Fn` returns a "callable" node — invoke once with no args to get
  // the actual TSL node graph. Three.js sees the call site as a single
  // node; the body is inlined into the vertex shader.
  (material as Any).positionNode = buildSkinPositionFn(skinArgs)();
  // Skin at vertices, then interpolate the resulting normal. Integer influence
  // indices are flat varyings: reading them in the fragment stage applies one
  // corner's skin transform to the whole triangle and creates visible facets.
  const worldNormal = buildSkinNormalFn(skinArgs)().toVarying();
  // NodeMaterial.normalNode expects view space; the DQB helper returns world space.
  const viewNormal = normalize(cameraViewMatrix.mul(vec4(worldNormal, 0)).xyz).mul(
    faceDirection as Any,
  );
  (material as Any).normalNode = viewNormal;
  if (sourceMaterial?.normalMap) {
    const map = sourceMaterial.normalMap;
    const scale = uniform(sourceMaterial.normalScale.clone());
    // Build the tangent frame from the deformed surface, so the source normal
    // map remains attached to the skin as the body rotates, bends, and stretches.
    (material as Any).normalNode = Fn(() => {
      const dp1: Any = dFdx(positionView),
        dp2: Any = dFdy(positionView);
      const duv1: Any = dFdx(uv(map.channel)),
        duv2: Any = dFdy(uv(map.channel));
      const p2: Any = cross(dp2, viewNormal),
        p1: Any = cross(viewNormal, dp1);
      const tangent: Any = p2.mul(duv1.x).add(p1.mul(duv2.x));
      const bitangent: Any = p2.mul(duv1.y).add(p1.mul(duv2.y));
      const inverseScale: Any = tangent
        .dot(tangent)
        .max(bitangent.dot(bitangent))
        .max(1e-12)
        .inverseSqrt();
      const sampled: Any = texture(map).xyz.mul(2).sub(1);
      const mapped: Any = vec3(sampled.xy.mul(scale), sampled.z);
      return normalize(
        tangent
          .mul(mapped.x)
          .add(bitangent.mul(mapped.y))
          .mul(inverseScale)
          .add(viewNormal.mul(mapped.z)),
      );
    })();
  }

  return material;
}
