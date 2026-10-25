import { Fn, If, attribute, float, positionLocal, vec3, vec4 } from 'three/tsl';
import type { Vector3 } from 'three';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { ParticleSystem } from '../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * For K influences with weights `w_j` and per-influence rigid-motion
 * dual quaternions `q̂_j = q_j + ε · ½ · t_j · q_j`:
 *   1. Antipodality flip vs influence 0 (Algorithm 2): negate any `q̂_j`
 *      whose real part has negative dot product with `q̂_0`'s real part.
 *   2. DLB blend (Eq. 11): `q̂_blend = Σ w_j · q̂_j`.
 *   3. Normalization by `‖q̂_blend.real‖` (Algorithm 1 step `b₀ ← ‖b̂‖`).
 *   4. Apply rigid-motion dual quaternion to a body-local rest vertex:
 *      `v' = R(q̂.real) · v + 2 · (a₀·d_ε − a_ε·d₀ + d₀ × d_ε)` where
 *      `(a₀, d₀) = q̂.real` and `(a_ε, d_ε) = q̂.dual` (Lemma 4 + Algo 1).
 *
 * Each vertex has four influences (Müller & Chentanez 2011, §7). Fewer than 4
 * influences are zero-padded by {@link bindSoftbodyMesh}; the unused
 * lanes contribute weight 0 and vanish in the sum.
 *
 * Coordinate frame contract.
 *  - `vRestLocal` (input) — pre-centered body-local: `vertex_mesh − c̄_body`.
 *  - `restOffsets[k]` — same pre-centered body-local frame.
 *  - `particles.positions[k]` — world space.
 *  - `getRotationQuat(k)` — unit quaternion mapping body-rest → world.
 * Output is a world-space position.
 */

/**
 * Standard quaternion-rotates-vector formula (Kavan 2008 Lemma 4 / well-
 * known cross-product identity):
 *   `v' = v + 2 q.xyz × (q.xyz × v + q.w · v)`.
 * Cheaper than building the 3x3 matrix; identical algebra.
 */
function quatRotate(q: Any, v: Any): Any {
  const qv = q.xyz;
  return v.add(qv.cross(qv.cross(v).add(v.mul(q.w))).mul(float(2.0)));
}

/**
 * Hamilton quaternion product `(a ⊗ b)` with the `(x, y, z, w)` layout.
 *   `(a ⊗ b).xyz = a.w·b.xyz + b.w·a.xyz + a.xyz × b.xyz`
 *   `(a ⊗ b).w   = a.w·b.w   − dot(a.xyz, b.xyz)`
 */
function quatMul(a: Any, b: Any): Any {
  const v = a.w.mul(b.xyz).add(b.w.mul(a.xyz)).add(a.xyz.cross(b.xyz));
  const w = a.w.mul(b.w).sub(a.xyz.dot(b.xyz));
  return vec4(v.x, v.y, v.z, w);
}

/**
 * Construct the rigid-motion dual quaternion `q̂_k = q_k + ε · ½ · t_k · q_k`
 * (Kavan 2008 §3.2, with `t_k` treated as a pure-imaginary quaternion
 * `(t_k, 0)` for the multiplication).
 *
 * The translation `t_k` is the body-rest-local-to-world offset such that
 * applying `q̂_k` to a rest-frame point `p` yields `R(q_k)·p + t_k`. For
 * a particle with current world position `x*_k` and rest body-local
 * position `r_k`, this is `t_k = x*_k − R(q_k) · r_k`.
 *
 * Returns the dual part as a vec4 (xyz = vector, w = scalar) — the real
 * part is the input quaternion `q`, returned by the caller separately to
 * keep this helper a pure function.
 */
function rigidDualPart(q: Any, t: Any): Any {
  // tQuat = (t.xyz, 0) — pure-imaginary quaternion.
  const tQuat = vec4(t.x, t.y, t.z, float(0.0));
  return quatMul(tQuat, q).mul(float(0.5));
}

export interface BuildSkinPositionFnArgs {
  /** Owning particle system; used for `positions.element(idx)` reads. */
  readonly particles: ParticleSystem;
  /**
   * Per-particle pre-centered body-local rest offsets (the
   * {@link SoftbodySystem.restOffsets} buffer). Indexed by global
   * particle slot.
   */
  readonly restOffsets: StorageBufferNode<'vec4'>;
  /**
   * Pre-centering offset for the bound body's mesh: `c̄_body`. Subtracted
   * from `positionLocal` before any rest-frame math runs.
   */
  readonly cBar: UniformNode<'vec3', Vector3>;
  /**
   * Per-`Fn`-invocation factory for the rotation source. The skin shader
   * builds the position and normal `Fn`s separately; each invocation
   * gets a FRESH `(idx) => quat` closure so per-Fn caching state cannot
   * leak between them. With local shape matching the closure returns
   * `particles.rotation.element(idx)`. With global shape matching it computes the body-shared quaternion once (memoised inside
   * the closure) and returns the same node reference for every lane;
   * DLB then collapses to LBS as Kavan 2008 Eq. 11 specifies.
   */
  readonly createGetRotationQuat: () => (particleIndex: Any) => Any;
}

/**
 * Build a TSL `Fn` returning the skinned **world-space** position for
 * the current vertex, using the DLB blend on K=4 influences.
 *
 * The function reads two per-vertex attributes set by
 * {@link bindSoftbodyMesh}:
 *  - `influences` (uvec4): 4 global particle slots.
 *  - `weights` (vec4): 4 inverse-distance weights, partition-of-unity.
 *
 * The shader is intentionally K=4 unrolled — runtime-indexed vec4-
 * component access is awkward in TSL, and 4 iterations is cheap enough
 * to unroll on any GPU (about 12–16 FMAs per vertex).
 */
export function buildSkinPositionFn(args: BuildSkinPositionFnArgs): Any {
  const { particles, restOffsets, cBar, createGetRotationQuat } = args;

  return Fn(() => {
    const getRotationQuat = createGetRotationQuat();
    // ---- Read per-vertex bind data ----
    // `attribute` defaults to `vec4` for 4-wide buffer attributes; the
    // Uint32BufferAttribute we wrote in bindMesh is exposed as a uvec4
    // (three.js TSL infers from the attribute's `array.constructor`).
    const inf: Any = attribute('influences');
    const wts: Any = attribute('weights');

    // Pre-centred body-local rest position of THIS vertex.
    const vRest: Any = positionLocal.xyz.sub(cBar).toVar();

    // ---- Per-influence dual-quaternion construction ----
    // Lane indices and weights, materialised as fresh `var`s so the
    // codegen doesn't re-read the attribute four times.
    const i0: Any = inf.x;
    const i1: Any = inf.y;
    const i2: Any = inf.z;
    const i3: Any = inf.w;
    const w0: Any = wts.x;
    const w1: Any = wts.y;
    const w2: Any = wts.z;
    const w3: Any = wts.w;

    // Helper closure — builds `(real_k, dual_k)` for one lane and
    // returns them as a `[vec4, vec4]` tuple. Inlines particle reads
    // and dual-part construction.
    const laneDQ = (idx: Any): { real: Any; dual: Any } => {
      const q = getRotationQuat(idx).toVar();
      const xStar = particles.positions.element(idx).xyz;
      const r = restOffsets.element(idx).xyz;
      // t_k = x*_k − R(q_k) · r_k
      const t = xStar.sub(quatRotate(q, r));
      const dual = rigidDualPart(q, t);
      return { real: q, dual };
    };

    const dq0 = laneDQ(i0);
    const dq1 = laneDQ(i1);
    const dq2 = laneDQ(i2);
    const dq3 = laneDQ(i3);

    // ---- Antipodality (Algorithm 2): flip lanes whose real part has
    //      negative dot with the pivot (lane 0). `select(cond, a, b)`
    //      returns a if cond else b.
    const flipSign = (qPivot: Any, q: Any): Any => {
      const dot = qPivot.x
        .mul(q.x)
        .add(qPivot.y.mul(q.y))
        .add(qPivot.z.mul(q.z))
        .add(qPivot.w.mul(q.w));
      return dot.lessThan(float(0.0)).select(float(-1.0), float(1.0));
    };
    const s1 = flipSign(dq0.real, dq1.real);
    const s2 = flipSign(dq0.real, dq2.real);
    const s3 = flipSign(dq0.real, dq3.real);

    const r0 = dq0.real;
    const r1 = dq1.real.mul(s1);
    const r2 = dq2.real.mul(s2);
    const r3 = dq3.real.mul(s3);
    const d0 = dq0.dual;
    const d1 = dq1.dual.mul(s1);
    const d2 = dq2.dual.mul(s2);
    const d3 = dq3.dual.mul(s3);

    // ---- Eq. 11: weighted sum (real and dual independently) ----
    const realSum: Any = r0.mul(w0).add(r1.mul(w1)).add(r2.mul(w2)).add(r3.mul(w3)).toVar();
    const dualSum: Any = d0.mul(w0).add(d1.mul(w1)).add(d2.mul(w2)).add(d3.mul(w3)).toVar();

    // ---- Normalize: divide by ‖real_sum‖ (Algorithm 1 `b₀ ← ‖b̂‖`).
    //      The dual part divides by the SAME norm — preserves the
    //      unit-DQ manifold under the standard rigid-motion algebra. ----
    const norm: Any = realSum.length().max(float(1e-20));
    const invNorm: Any = float(1.0).div(norm);
    const realN: Any = realSum.mul(invNorm).toVar();
    const dualN: Any = dualSum.mul(invNorm).toVar();

    // ---- Apply to vRest:
    //        v_rotated = R(realN) · vRest             (Lemma 4)
    //        translation = 2·(a₀·d_ε − a_ε·d₀ + d₀ × d_ε)   (Algorithm 1)
    //        v_world = v_rotated + translation
    // ---------------------------------------------------------------
    const vRotated: Any = quatRotate(realN, vRest);
    // (a₀, d₀) = (realN.w, realN.xyz); (a_ε, d_ε) = (dualN.w, dualN.xyz).
    const translation: Any = realN.w
      .mul(dualN.xyz)
      .sub(dualN.w.mul(realN.xyz))
      .add(realN.xyz.cross(dualN.xyz))
      .mul(float(2.0));
    return vRotated.add(translation);
  });
}

/**
 * Build a TSL `Fn` returning the skinned **world-space** normal.
 *
 * Normals transform by the rotation part of the (post-normalization)
 * blended dual quaternion: `n' = R(realN) · n`. DQB preserves unit length under pure rotation; the
 * inverse-transpose collapses because rotation matrices are orthogonal.
 *
 * Same antipodality-and-blend code as the position path, factored out
 * so the position function stays a pure expression (no side-effect
 * register sharing). Three.js codegen will CSE the shared sub-graphs
 * across the two `Fn`s.
 */
export function buildSkinNormalFn(args: BuildSkinPositionFnArgs): Any {
  const { particles, restOffsets, createGetRotationQuat } = args;

  return Fn(() => {
    const getRotationQuat = createGetRotationQuat();
    const inf: Any = attribute('influences');
    const wts: Any = attribute('weights');

    const i0: Any = inf.x;
    const i1: Any = inf.y;
    const i2: Any = inf.z;
    const i3: Any = inf.w;
    const w0: Any = wts.x;
    const w1: Any = wts.y;
    const w2: Any = wts.z;
    const w3: Any = wts.w;

    // For normal we only need the real part — the translation drops out.
    // We still need the antipodality flip to keep the rotation blend
    // sensible across opposite-hemisphere quaternions.
    const q0 = getRotationQuat(i0).toVar();
    const q1 = getRotationQuat(i1).toVar();
    const q2 = getRotationQuat(i2).toVar();
    const q3 = getRotationQuat(i3).toVar();
    void particles; // restOffsets/particles unused on the normal path
    void restOffsets;

    const flipSign = (qPivot: Any, q: Any): Any => {
      const dot = qPivot.x
        .mul(q.x)
        .add(qPivot.y.mul(q.y))
        .add(qPivot.z.mul(q.z))
        .add(qPivot.w.mul(q.w));
      return dot.lessThan(float(0.0)).select(float(-1.0), float(1.0));
    };
    const s1 = flipSign(q0, q1);
    const s2 = flipSign(q0, q2);
    const s3 = flipSign(q0, q3);

    const realSum: Any = q0
      .mul(w0)
      .add(q1.mul(s1).mul(w1))
      .add(q2.mul(s2).mul(w2))
      .add(q3.mul(s3).mul(w3))
      .toVar();
    const norm: Any = realSum.length().max(float(1e-20));
    const realN: Any = realSum.div(norm).toVar();

    const normalIn: Any = attribute('normal');
    return quatRotate(realN, normalIn.xyz);
  });
}

// Avoid an "unused" lint hit on the local helper; it's intentionally
// kept module-private but referenced via the closure inside the Fn.
void If;
void vec3;
