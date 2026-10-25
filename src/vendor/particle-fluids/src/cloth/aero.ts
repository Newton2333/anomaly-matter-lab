import type { Vector3 } from 'three';
import {
  Fn,
  If,
  Loop,
  Return,
  cross,
  float,
  instanceIndex,
  instancedArray,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from '../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Aerodynamic drag and lift on cloth (Keckeisen et al. 2004, §3).
 *
 * **Kernel shape — per-vertex gather (not per-triangle scatter).**
 * The kernel dispatches one thread per cloth particle. Each thread
 * walks its incident triangles via a pre-built CSR table, computes
 * the per-triangle force `F_t = F_drag + F_lift`, accumulates
 * `F_sum += F_t / 3` (paper §3 — each triangle distributes its force
 * equally to its three vertices), then writes its own
 * `predictedPositions[v]` slot. No scatter, no atomics, no race on
 * shared vertices: each thread reads many other threads' positions
 * + velocities (read-only) but writes only its own slot.
 *
 * **Per-vertex force formulas (Keckeisen 2004 §3, eqs. 3.1 & §3
 * drag / lift).** For triangle `t` with vertices `(x_1, x_2, x_3)`,
 * area `A`, face normal `n = normalize((x_2-x_1)×(x_3-x_1))`:
 *
 *   `v_avg   = (v_1 + v_2 + v_3) / 3`
 *   `v_rel   = v_avg - v_wind(centroid)`              (object-relative wind)
 *   `v̂_rel   = v_rel / |v_rel|`
 *   `n̄       = sign(n · v_rel) · n`                   (eq. 3.1: normal flips toward flow)
 *   `F_drag  = -dragCoeff · |v_rel|² · A · (n̄·v̂_rel) · v̂_rel`
 *   `u_raw   = (n̄ × v̂_rel) × v̂_rel`                  (perpendicular to v_rel, in (n̄,v̂_rel) plane)
 *   `F_lift  = liftCoeff · |v_rel|² · A · u_raw`
 *
 * `dragCoeff = 0.5 · C_D · ρ` and `liftCoeff = 0.5 · C_L · ρ`: the
 * constant `0.5 · ρ` is folded into the tunable coefficients.
 *
 * **Why `F_lift = liftCoeff · |v_rel|² · A · u_raw` (no extra
 * `cos(θ)` factor).** The paper writes
 *   `F_lift = (1/2)·C_L·ρ·|v_rel|² · A · cos(θ) · û`
 * where `û` is unit and `cos(θ) = |v̂_rel × n̄|` peaks parallel to
 * the face. The unnormalized cross-product chain has magnitude
 *   `|u_raw| = |(n̄ × v̂_rel) × v̂_rel| = |n̄ × v̂_rel| = sin(angle(n̄,v̂_rel)) = cos(θ)`
 * (the second equality follows from `(a×b)⊥c` for unit `c`).
 * So `cos(θ) · û = cos(θ) · (u_raw / |u_raw|) = u_raw`, and the
 * normalize + multiply collapses to one cross-product chain — same
 * physics, fewer ops, and no division-by-zero at perpendicular wind
 * (where `cos(θ) = 0`, `u_raw = 0`, and lift correctly vanishes).
 *
 * **Position-only update (not velocity).** The kernel adds
 * `Δx* = dt² · invMass · F_sum` to `predictedPositions` and does
 * NOT touch `velocities`. The substep's `advect` re-derives
 * `v = (x* − x) / dt` after the iter loop, which correctly captures
 * the aero contribution as `dt · invMass · F_sum` in the final
 * velocity. This avoids a write-conflict on `velocities[v]` between
 * threads (each cloth vertex thread reads its triangles' other
 * vertices' velocities; if those threads also wrote `velocities`
 * we'd hit a read/write race) and is consistent with the
 * semi-implicit-Euler integration the rest of the pipeline uses
 * (`x* = x + dt · (v + dt · a) = x + dt·v + dt²·a` for any external
 * acceleration `a = F · invMass`).
 *
 * **Pipeline placement.** Runs once per substep before the constraint
 * iterations, so `velocities` and `positions` are at the substep start
 * and `predictedPositions` holds the gravity-only prediction. The aero
 * `Δx*` adds onto that, so the constraints see the wind-perturbed
 * prediction.
 *
 * **Wind.** One uniform velocity across the whole cloth, a steady
 * directional wind (paper Fig. 2 "Directional").
 *
 * **Skipping degenerate triangles.** Per-triangle inside the loop
 * we guard against `|v_rel|² < 1e-12` (no relative motion → no
 * force) and `|n_raw|² < 1e-12` (collapsed triangle → no area, no
 * normal). The `If` blocks gate the force accumulation; the loop
 * body always completes uniformly so the per-thread iteration
 * count stays `incidenceCount[v]` regardless of triangle quality.
 */
export function createClothAeroKernel(args: {
  readonly particles: ParticleSystem;
  /** Absolute slot offset of this cloth's first particle. */
  readonly particleOffset: number;
  /** Number of cloth particles ( = `graph.positions.length`). */
  readonly nClothParticles: number;
  /**
   * Cloth-local triangle list `[i_1, i_2, i_3]` per triangle, with
   * `0 ≤ i_k < nClothParticles`. Typically `graph.triangles`.
   */
  readonly triangles: readonly (readonly [number, number, number])[];
  readonly dt: UniformNode<'float', number>;
  /** Wind velocity, m/s. */
  readonly wind: UniformNode<'vec3', Vector3>;
  /** `½ · C_D · ρ_air`. */
  readonly dragCoeff: UniformNode<'float', number>;
  /** `½ · C_L · ρ_air`. */
  readonly liftCoeff: UniformNode<'float', number>;
}): ComputeNode {
  const { particles, particleOffset, nClothParticles, triangles, dt, wind, dragCoeff, liftCoeff } =
    args;

  if (!Number.isInteger(particleOffset) || particleOffset < 0) {
    throw new Error(
      `createClothAeroKernel: particleOffset must be a non-negative integer, got ${particleOffset}`,
    );
  }
  if (!Number.isInteger(nClothParticles) || nClothParticles <= 0) {
    throw new Error(
      `createClothAeroKernel: nClothParticles must be a positive integer, got ${nClothParticles}`,
    );
  }
  if (particleOffset + nClothParticles > particles.capacity) {
    throw new Error(
      `createClothAeroKernel: particleOffset (${particleOffset}) + nClothParticles (${nClothParticles}) exceeds capacity ${particles.capacity}`,
    );
  }

  const nTri = triangles.length;

  // Validate triangle indices.
  for (let t = 0; t < nTri; t++) {
    const tri = triangles[t]!;
    for (const idx of tri) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= nClothParticles) {
        throw new Error(
          `createClothAeroKernel: triangle ${t} has out-of-range vertex ${idx} (nClothParticles = ${nClothParticles})`,
        );
      }
    }
  }

  // ---- Build per-vertex incidence CSR ----
  const incidenceCount = new Uint32Array(nClothParticles);
  for (let t = 0; t < nTri; t++) {
    const [a, b, c] = triangles[t]!;
    incidenceCount[a]!++;
    incidenceCount[b]!++;
    incidenceCount[c]!++;
  }
  const offsets = new Uint32Array(nClothParticles + 1);
  let cumulative = 0;
  for (let v = 0; v < nClothParticles; v++) {
    offsets[v] = cumulative;
    cumulative += incidenceCount[v]!;
  }
  offsets[nClothParticles] = cumulative;
  const incidenceData = new Uint32Array(cumulative);
  // Re-walk triangles, filling the CSR using a write cursor per vertex.
  const cursor = new Uint32Array(nClothParticles);
  for (let t = 0; t < nTri; t++) {
    const [a, b, c] = triangles[t]!;
    incidenceData[offsets[a]! + cursor[a]!++] = t;
    incidenceData[offsets[b]! + cursor[b]!++] = t;
    incidenceData[offsets[c]! + cursor[c]!++] = t;
  }

  // ---- GPU storage ----
  const triangleVerts = instancedArray(Math.max(1, nTri * 3), 'uint');
  const incidenceOffsets = instancedArray(nClothParticles + 1, 'uint');
  const incidenceDataBuf = instancedArray(Math.max(1, cumulative), 'uint');

  if (nTri > 0) {
    const triArr = triangleVerts.value.array as Uint32Array;
    for (let t = 0; t < nTri; t++) {
      const [a, b, c] = triangles[t]!;
      triArr[t * 3 + 0] = a;
      triArr[t * 3 + 1] = b;
      triArr[t * 3 + 2] = c;
    }
    triangleVerts.value.needsUpdate = true;
  }
  {
    const offArr = incidenceOffsets.value.array as Uint32Array;
    offArr.set(offsets);
    incidenceOffsets.value.needsUpdate = true;
  }
  if (cumulative > 0) {
    const inArr = incidenceDataBuf.value.array as Uint32Array;
    inArr.set(incidenceData);
    incidenceDataBuf.value.needsUpdate = true;
  }

  // ---- Kernel ----
  const kernel = Fn(() => {
    const v: Any = instanceIndex;
    const absSlot: Any = v.add(uint(particleOffset)).toVar();
    const w: Any = particles.invMass.element(absSlot).toVar();
    // Pinned vertices receive no aero displacement (Δx = dt²·w·F = 0).
    // Early-return saves the incident-triangle walk for pinned slots.
    If(w.lessThanEqual(float(0.0)), () => {
      Return();
    });

    const start: Any = incidenceOffsets.element(v).toVar();
    const end: Any = incidenceOffsets.element(v.add(uint(1))).toVar();
    const fSum: Any = vec3(float(0.0), float(0.0), float(0.0)).toVar();

    Loop({ start, end, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const tri: Any = incidenceDataBuf.element(k).toVar();
      const triBase: Any = tri.mul(uint(3));
      const i1Local: Any = triangleVerts.element(triBase).toVar();
      const i2Local: Any = triangleVerts.element(triBase.add(uint(1))).toVar();
      const i3Local: Any = triangleVerts.element(triBase.add(uint(2))).toVar();
      const i1: Any = i1Local.add(uint(particleOffset)).toVar();
      const i2: Any = i2Local.add(uint(particleOffset)).toVar();
      const i3: Any = i3Local.add(uint(particleOffset)).toVar();

      const x1: Any = particles.positions.element(i1).xyz.toVar();
      const x2: Any = particles.positions.element(i2).xyz.toVar();
      const x3: Any = particles.positions.element(i3).xyz.toVar();
      const v1: Any = particles.velocities.element(i1).xyz.toVar();
      const v2: Any = particles.velocities.element(i2).xyz.toVar();
      const v3: Any = particles.velocities.element(i3).xyz.toVar();

      // Average velocity, relative to wind.
      const vAvg: Any = v1.add(v2).add(v3).div(float(3.0)).toVar();
      const vRel: Any = vAvg.sub(wind).toVar();
      const vRelSq: Any = vRel.dot(vRel).toVar();

      // Skip if no relative wind motion.
      If(vRelSq.greaterThan(float(1e-12)), () => {
        // Triangle area-weighted normal (Keckeisen 2004 §3, also
        // Bender 2014 §3.4.2 for the same triangle-area formula).
        //   n_raw = (x_2 - x_1) × (x_3 - x_1),  |n_raw| = 2·A
        const e12: Any = x2.sub(x1).toVar();
        const e13: Any = x3.sub(x1).toVar();
        const nRaw: Any = cross(e12, e13).toVar();
        const nMagSq: Any = nRaw.dot(nRaw).toVar();
        // Skip degenerate triangles (collinear / zero area).
        If(nMagSq.greaterThan(float(1e-12)), () => {
          const nMag: Any = nMagSq.sqrt().toVar();
          const area: Any = nMag.mul(float(0.5)).toVar();
          const nUnit: Any = nRaw.div(nMag).toVar();

          // n̄ = sign(n · v_rel) · n  per eq. 3.1.
          const cosNV: Any = nUnit.dot(vRel).toVar(); // |v_rel|·cos(angle(n,v))
          const flipSign: Any = cosNV.lessThan(float(0.0)).select(float(-1.0), float(1.0));
          const nBar: Any = nUnit.mul(flipSign).toVar();

          const vRelMag: Any = vRelSq.sqrt().toVar();
          const vRelHat: Any = vRel.div(vRelMag).toVar();

          // Drag: F_drag = -dragCoeff · |v_rel|² · A · (n̄·v̂_rel) · v̂_rel.
          // (n̄·v̂_rel) = |cos(angle(n,v))| ≥ 0 because of the n̄ flip.
          const adjNV: Any = nBar.dot(vRelHat).toVar();
          const dragMag: Any = dragCoeff.mul(vRelSq).mul(area).mul(adjNV).toVar();
          const fDrag: Any = vRelHat.mul(dragMag).negate().toVar();

          // Lift: F_lift = liftCoeff · |v_rel|² · A · ((n̄ × v̂_rel) × v̂_rel).
          // See block comment above for why the unnormalized chain
          // already carries the cos(θ) factor.
          const nBarCrossV: Any = cross(nBar, vRelHat).toVar();
          const uRaw: Any = cross(nBarCrossV, vRelHat).toVar();
          const liftMag: Any = liftCoeff.mul(vRelSq).mul(area).toVar();
          const fLift: Any = uRaw.mul(liftMag).toVar();

          // Per-vertex contribution = F_total / 3 (paper §3, distributes
          // triangle force equally to its three vertices).
          const fThird: Any = fDrag.add(fLift).div(float(3.0)).toVar();
          fSum.assign(fSum.add(fThird));
        });
      });
    });

    // Δx* = dt² · invMass · F_sum  (semi-implicit Euler external-force step).
    const dtVal: Any = dt;
    const dtSq: Any = dtVal.mul(dtVal).toVar();
    const dxStar: Any = fSum.mul(dtSq.mul(w)).toVar();
    const xStar: Any = particles.predictedPositions.element(absSlot).xyz.toVar();
    particles.predictedPositions.element(absSlot).assign(vec4(xStar.add(dxStar), float(0.0)));
  })()
    .compute(nClothParticles)
    .setName('aero.wind');

  return kernel;
}
