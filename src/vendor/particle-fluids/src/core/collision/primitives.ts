import { If, float, max, uint, vec3, vec4 } from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Primitive kinds, stored in the high 16 bits of each slot's packed word. */
export const KIND_PLANE = 0;
export const KIND_SPHERE = 1;
export const KIND_BOX = 2;
export const KIND_CAPSULE = 3;

/** Flag bit (low 16 bits of the packed word): the inside is the valid region. */
export const FLAG_INVERT = 1 << 0;
/**
 * Flag bit: the primitive moves (attached or placed each frame), so the solver
 * sweeps it back along its velocity and spin within a frame. Without it,
 * `velocity` is surface velocity only (a conveyor belt).
 */
export const FLAG_SWEEP = 1 << 1;

/** Rotate `v` by unit quaternion `q`: `v + 2w(q × v) + q × (2(q × v))`. */
function emitQuatRotate(q: Any, v: Any): Any {
  const qxyz: Any = q.xyz;
  const t: Any = float(2.0).mul(qxyz.cross(v)).toVar();
  return v.add(t.mul(q.w)).add(qxyz.cross(t));
}

function emitQuatConjugate(q: Any): Any {
  return vec4(q.x.negate(), q.y.negate(), q.z.negate(), q.w);
}

/** Unit quaternion for a rotation by `|ω|·t` about `ω`. */
function emitQuatFromSpin(spin: Any, t: Any): Any {
  const rate: Any = spin.length().toVar();
  const half: Any = rate.mul(t).mul(0.5).toVar();
  const axis: Any = spin.div(rate.max(float(EPSILON_GRADIENT)));
  return vec4(axis.mul(half.sin()), half.cos());
}

/** The point a primitive turns about: the plane point, a capsule's midpoint, or the center. */
function emitPivot(kind: Any, data0: Any, data1: Any): Any {
  return kind
    .equal(uint(KIND_PLANE))
    .select(
      data1.xyz,
      kind.equal(uint(KIND_CAPSULE)).select(data0.xyz.add(data1.xyz).mul(0.5), data0.xyz),
    );
}

/** `rewind` for swept primitives, 0 for the rest. */
function emitSweepTime(flags: Any, rewind: Any): Any {
  return flags.bitAnd(uint(FLAG_SWEEP)).notEqual(uint(0)).select(rewind, float(0));
}

/** Lengths below this are treated as zero when normalizing a gradient. */
const EPSILON_GRADIENT = 1e-8;

/** GPU buffers {@link emitColliderSdf} reads; see {@link PrimitiveSet}. */
export interface ColliderFields {
  readonly packed: Any;
  readonly data0: Any;
  readonly data1: Any;
  readonly rotation: Any;
  /** Linear velocity per slot, `w` unused here. */
  readonly velocity?: Any;
  /** Angular velocity per slot in rad/s about the pivot, `w` unused. */
  readonly spin?: Any;
}

/**
 * Emit TSL that evaluates the signed distance `phi` to the primitive in
 * `colliderSlot` and its outward unit gradient, writing both into the given
 * vars. Distances are positive outside the shape; inverted primitives flip
 * both signs.
 *
 * - Plane: `phi = n · (x − p)`
 * - Sphere: `phi = |x − c| − r`
 * - Box: `q = |x_local| − e`, `phi = |max(q, 0)| + min(max(q.x, q.y, q.z), 0)`
 * - Capsule: sphere around the closest point on the segment
 *
 * `rewind` (seconds) evaluates a primitive flagged {@link FLAG_SWEEP} as it
 * was that long ago, by moving the query point forward along the primitive's
 * velocity and spin instead of moving the primitive back.
 */
export function emitColliderSdf(
  fields: ColliderFields,
  colliderSlot: Any,
  xWorld: Any,
  phiVar: Any,
  gradientVar: Any,
  rewind?: Any,
): void {
  const packedVal: Any = fields.packed.element(colliderSlot);
  const kind: Any = packedVal.shiftRight(uint(16)).bitAnd(uint(0xffff));
  const flags: Any = packedVal.bitAnd(uint(0xffff));
  const data0: Any = fields.data0.element(colliderSlot);
  const data1: Any = fields.data1.element(colliderSlot);

  // t seconds ago the primitive sat v·t back and turned R(−ωt) about its
  // pivot c, so its field then at x is its current field at
  // x' = c + R(ωt)(x − c + v·t), with the gradient turned back by R(−ωt).
  let x: Any = xWorld;
  let unturn: Any;
  if (rewind !== undefined && fields.velocity) {
    const t: Any = emitSweepTime(flags, rewind).toVar();
    const shifted: Any = xWorld.add(fields.velocity.element(colliderSlot).xyz.mul(t));
    if (fields.spin) {
      const pivot: Any = emitPivot(kind, data0, data1).toVar();
      const turn: Any = emitQuatFromSpin(fields.spin.element(colliderSlot).xyz, t).toVar();
      x = pivot.add(emitQuatRotate(turn, shifted.sub(pivot))).toVar();
      unturn = emitQuatConjugate(turn);
    } else {
      x = shifted.toVar();
    }
  }

  // -------- Plane --------
  // data0.xyz = unit normal n, data1.xyz = reference point p.
  If(kind.equal(uint(KIND_PLANE)), () => {
    const n: Any = data0.xyz;
    const p: Any = data1.xyz;
    phiVar.assign(n.dot(x.sub(p)));
    gradientVar.assign(n);
  });

  // -------- Sphere --------
  // data0.xyz = center c, data0.w = radius R.
  If(kind.equal(uint(KIND_SPHERE)), () => {
    const c: Any = data0.xyz;
    const R: Any = data0.w;
    const diff: Any = x.sub(c).toVar();
    const len: Any = diff.length().toVar();
    phiVar.assign(len.sub(R));
    // Guarded divide — see EPSILON_GRADIENT doc above.
    const safeLen: Any = len.max(float(EPSILON_GRADIENT));
    gradientVar.assign(diff.div(safeLen));
  });

  // -------- Oriented Box --------
  // data0.xyz = center c, data1.xyz = half-extents e, rotation = unit
  // quaternion defining the box's world orientation (identity = AABB).
  // Rotation is applied in two places: (1) the particle position is
  // rotated into the box's local frame via the conjugate quaternion
  // before the axis-aligned SDF, (2) the resulting gradient is rotated
  // back to world space by the forward quaternion.
  If(kind.equal(uint(KIND_BOX)), () => {
    const c: Any = data0.xyz;
    const e: Any = data1.xyz;
    const rot: Any = fields.rotation.element(colliderSlot);
    // World → box-local. Identity quaternion passes `x - c` through
    // unchanged, so the axis-aligned case costs two cross products
    // but returns the same answer it would have without rotation.
    const xWorldOffset: Any = x.sub(c).toVar();
    const xLocal: Any = emitQuatRotate(emitQuatConjugate(rot), xWorldOffset).toVar();
    const q: Any = xLocal.abs().sub(e).toVar();
    // Exterior term: length of positive components of q.
    const qPos: Any = max(q, vec3(float(0.0))).toVar();
    const outside: Any = qPos.length();
    // Interior term: max of q's components, clamped at zero.
    const inside: Any = q.x.max(q.y).max(q.z).min(float(0.0));
    phiVar.assign(outside.add(inside));
    // Gradient — closed-form in the box's LOCAL frame, then rotated back
    // to world space below.
    //   Outside (any q > 0): unit(max(q, 0)) with per-axis sign from
    //     xLocal (local-space position).
    //   Inside (all q ≤ 0): outward direction is the axis with the
    //     greatest q (closest to escaping); sign taken from xLocal.
    const signs: Any = vec3(
      xLocal.x.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0)),
      xLocal.y.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0)),
      xLocal.z.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0)),
    ).toVar();
    const outsideLen: Any = qPos.length().toVar();
    const isOutside: Any = outsideLen.greaterThan(float(EPSILON_GRADIENT));
    const domX: Any = q.x.greaterThanEqual(q.y).and(q.x.greaterThanEqual(q.z));
    const domY: Any = q.y.greaterThanEqual(q.x).and(q.y.greaterThanEqual(q.z));
    const insideDir: Any = vec3(
      domX.select(signs.x, float(0.0)),
      domX.not().and(domY).select(signs.y, float(0.0)),
      domX.not().and(domY.not()).select(signs.z, float(0.0)),
    ).toVar();
    const localGrad: Any = isOutside
      .select(qPos.mul(signs).div(outsideLen.max(float(EPSILON_GRADIENT))), insideDir)
      .toVar();
    // Rotate the local-frame gradient back to world. For identity
    // quaternion this is a no-op (passes `localGrad` through).
    gradientVar.assign(emitQuatRotate(rot, localGrad));
  });

  // -------- Capsule --------
  // data0.xyz = endpoint a, data0.w = radius R, data1.xyz = endpoint b.
  If(kind.equal(uint(KIND_CAPSULE)), () => {
    const a: Any = data0.xyz;
    const R: Any = data0.w;
    const b: Any = data1.xyz;
    const ab: Any = b.sub(a).toVar();
    const ax: Any = x.sub(a).toVar();
    // Project x onto the segment, clamp to [0, 1].
    const abLenSq: Any = ab.dot(ab).max(float(EPSILON_GRADIENT));
    const t: Any = ax.dot(ab).div(abLenSq).clamp(float(0.0), float(1.0));
    const closest: Any = a.add(ab.mul(t)).toVar();
    const diff: Any = x.sub(closest).toVar();
    const len: Any = diff.length().toVar();
    phiVar.assign(len.sub(R));
    const safeLen: Any = len.max(float(EPSILON_GRADIENT));
    gradientVar.assign(diff.div(safeLen));
  });

  // -------- Invert --------
  // Flip sign of both phi and gradient. The collider's "outside" becomes
  // the interior half-space. Plan §"Inside-flip for containers".
  const invertBit: Any = flags.bitAnd(uint(FLAG_INVERT));
  const isInverted: Any = invertBit.notEqual(uint(0));
  phiVar.assign(isInverted.select(phiVar.negate(), phiVar));
  gradientVar.assign(isInverted.select(gradientVar.negate(), gradientVar));
  if (unturn !== undefined) gradientVar.assign(emitQuatRotate(unturn, gradientVar));
}

/**
 * Emit TSL for the velocity of the primitive in `colliderSlot`'s surface at
 * `xWorld`: `v + ω × (x − c)`, with the pivot `c` taken `rewind` seconds ago
 * for swept primitives, as in {@link emitColliderSdf}.
 */
export function emitColliderSurfaceVelocity(
  fields: ColliderFields & { readonly velocity: Any },
  colliderSlot: Any,
  xWorld: Any,
  rewind?: Any,
): Any {
  const velocity: Any = fields.velocity.element(colliderSlot).xyz.toVar();
  if (!fields.spin) return velocity;
  const packedVal: Any = fields.packed.element(colliderSlot);
  const kind: Any = packedVal.shiftRight(uint(16)).bitAnd(uint(0xffff));
  const pivot: Any = emitPivot(
    kind,
    fields.data0.element(colliderSlot),
    fields.data1.element(colliderSlot),
  );
  const t: Any = rewind !== undefined ? emitSweepTime(packedVal.bitAnd(uint(0xffff)), rewind) : 0;
  const arm: Any = xWorld.sub(pivot).add(velocity.mul(t));
  return velocity.add(fields.spin.element(colliderSlot).xyz.cross(arm));
}
