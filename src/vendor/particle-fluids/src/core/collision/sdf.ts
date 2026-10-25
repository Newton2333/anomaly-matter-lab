import { float, vec3 } from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * TSL-side view of the uniforms and 3D-texture binding an SDF collider
 * exposes to its kernels. See {@link SDFCollider} for the CPU-side owner
 * that populates these fields.
 *
 * All uniforms are per-collider. Unlike the analytic primitives (packed
 * into a shared `PrimitiveSet`), each SDF collider is its own TSL node
 * graph because 3D-texture bindings cannot be indexed by a runtime slot
 * — `texture_3d<f32>` bindings are compile-time in WGSL, so a scene
 * with N SDFs compiles N distinct kernels (one per collider).
 *
 * Coordinate pipeline:
 *   - `position`   — world-space location of the mesh's LOCAL origin.
 *   - `rotation` / `invRotation` — forward and inverse of the mesh's
 *                   world rotation. `invRotation` maps world → local at
 *                   sample time; `rotation` maps the local-frame
 *                   gradient back to world at return.
 *   - `scale` / `invScale` — uniform scale factor. Non-uniform scale
 *                   would break the `|∇φ| = 1` eikonal invariant and is
 *                   rejected by `SDFCollider.setTransform`.
 *   - `bakedOrigin` — mesh-local coordinate of the voxel grid's (0,0,0)
 *                   corner, as produced by the baker. Immutable.
 *
 * The `resolution` uniform is stored as a `vec3` of floats (not `uvec3`)
 * because the only place the kernel uses it is as a divisor on the
 * f32 grid coordinate and for the central-difference step size — both
 * arithmetic paths want float.
 */
export interface SdfFields {
  /** `Texture3DNode` wrapping an `r16float` `Data3DTexture`. */
  readonly texture: Any;
  /** World-space position of the mesh's local origin (`vec3` uniform). */
  readonly position: Any;
  /** Mesh-local coordinate of the grid's `(0, 0, 0)` corner. */
  readonly bakedOrigin: Any;
  /** World-space size of one voxel along each axis (same value per axis in v1). */
  readonly voxelSize: Any;
  /** `1 / voxelSize` per axis. Pre-computed at upload. */
  readonly invVoxelSize: Any;
  /** Voxel count per axis, as a float `vec3`. */
  readonly resolution: Any;
  /** Forward rotation `R` — used to map local-space gradient back to world. */
  readonly rotation: Any;
  /** Inverse rotation `R^T` — used to map world position into local space. */
  readonly invRotation: Any;
  /** Uniform scale factor (scalar float uniform). */
  readonly scale: Any;
  /** `1 / scale` — pre-computed so the kernel avoids a per-sample divide. */
  readonly invScale: Any;
}

/**
 * Evaluate φ(x) and its outward-unit gradient ∇φ(x) for a single SDF
 * collider, given the particle world-space position `x`. Signature mirrors
 * `emitColliderSdf` so the caller can thread `(phi, grad)` into the same
 * Macklin 2014 §6.1 eq. (22) projection arithmetic that the analytic
 * primitives feed into.
 *
 * Sampling model:
 *   - World → local:
 *       `x_local = invRotation · (x − position) · invScale`
 *     For an identity rotation + unit scale, `x_local = x − position`.
 *     Under rotation, the particle position is
 *     mapped into the box's local frame before the grid lookup.
 *   - Grid coordinate:
 *       `g = (x_local − bakedOrigin) / voxelSize`
 *     A voxel center (i, j, k) sits at `g = (i + 0.5, j + 0.5, k + 0.5)`.
 *   - Normalized UV for `texture.sample` is `g / resolution`, so the
 *     voxel center is at uv = (i + 0.5) / res — the standard half-texel
 *     centered convention.
 *   - φ is trilinearly interpolated by the hardware sampler.
 *   - ∇φ is reconstructed by six auxiliary samples at `uv ± 1/resolution`
 *     along each axis, central differences divided by `2 · voxelSize`.
 *     Seven texture samples total per particle per collider (one φ, six
 *     gradient).
 *
 * Transform back:
 *   - `φ_world = scale · φ_local` — scaling the SDF value preserves the
 *     eikonal property `|∇φ| = 1`: under the chain rule the scale cancels
 *     exactly because `∂x_local/∂x = invScale · invRotation` and
 *     `scale · invScale = 1`.
 *   - `∇φ_world = rotation · ∇φ_local` — rotate the local gradient back
 *     into world coordinates. Magnitude is preserved because `rotation`
 *     is orthogonal.
 *
 * The gradient is NOT re-normalized here. For a signed-distance function
 * `|∇φ| ≈ 1` everywhere away from the medial axis; callers that need a
 * unit vector (`SDFCollider` solve + friction kernels) do the divide
 * inline.
 *
 * Boundary behaviour: `ClampToEdge` wrap on all three axes (set by the
 * loader) means samples outside the grid return the padded boundary
 * value. The baker writes positive (exterior) distances in the padding
 * region, so a particle far from the collider sees `φ > 0` and the
 * projection is a no-op.
 *
 * @param fields  SDF binding group; usually one `SDFCollider.fields`.
 * @param x       `vec3` TSL node — particle position in world space.
 * @param phiVar  `float` TSL var the kernel writes φ(x) into.
 * @param gradientVar  `vec3` TSL var the kernel writes ∇φ(x) into.
 */
export function emitSampleSdf(fields: SdfFields, x: Any, phiVar: Any, gradientVar: Any): void {
  // World → local (subtract mesh origin, rotate into local frame, scale down).
  const worldOffset: Any = x.sub(fields.position).toVar();
  const xLocal: Any = fields.invRotation.mul(worldOffset).mul(fields.invScale).toVar();

  // Local → voxel grid coord → normalized UV. Voxel (0,0,0) centre sits
  // at uv = 0.5/res.
  const gridCoord: Any = xLocal.sub(fields.bakedOrigin).mul(fields.invVoxelSize).toVar();
  const uv: Any = gridCoord.div(fields.resolution).toVar();

  // φ at the particle position — trilinear via the hardware sampler.
  const phiLocal: Any = fields.texture.sample(uv).r.toVar();

  // Central-difference gradient in local space. One voxel in UV space is
  // `1 / resolution`. Step in UV, sample, then compute `Δφ / (2·h)`
  // where `h = voxelSize` in local-space units.
  const invRes: Any = vec3(float(1.0)).div(fields.resolution).toVar();
  const stepX: Any = vec3(invRes.x, float(0.0), float(0.0));
  const stepY: Any = vec3(float(0.0), invRes.y, float(0.0));
  const stepZ: Any = vec3(float(0.0), float(0.0), invRes.z);

  const phiXp: Any = fields.texture.sample(uv.add(stepX)).r;
  const phiXn: Any = fields.texture.sample(uv.sub(stepX)).r;
  const phiYp: Any = fields.texture.sample(uv.add(stepY)).r;
  const phiYn: Any = fields.texture.sample(uv.sub(stepY)).r;
  const phiZp: Any = fields.texture.sample(uv.add(stepZ)).r;
  const phiZn: Any = fields.texture.sample(uv.sub(stepZ)).r;

  const two: Any = float(2.0);
  const gradLocal: Any = vec3(
    phiXp.sub(phiXn).div(two.mul(fields.voxelSize.x)),
    phiYp.sub(phiYn).div(two.mul(fields.voxelSize.y)),
    phiZp.sub(phiZn).div(two.mul(fields.voxelSize.z)),
  ).toVar();

  // Transform back to world. `scale · φ_local` restores distance units;
  // `rotation · ∇φ_local` rotates the gradient from local to world
  // (orthogonal rotation preserves magnitude).
  phiVar.assign(phiLocal.mul(fields.scale));
  gradientVar.assign(fields.rotation.mul(gradLocal));
}
