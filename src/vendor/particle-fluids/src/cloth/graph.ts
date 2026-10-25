import type { BufferGeometry } from 'three';

/**
 * Output of {@link createClothGraph}: per-particle state plus the two
 * constraint topology lists `ClothSystem` consumes (distance pairs,
 * bending tuples).
 *
 * Coordinate convention: positions are taken verbatim from the input
 * `BufferGeometry`. `createClothGraph` does not transform vertices —
 * apply any world-space transform to the geometry's attribute or pass
 * a pre-transformed clone before calling.
 */
export interface ClothGraph {
  /** Per-particle world position, length `nParticles`. */
  readonly positions: readonly (readonly [number, number, number])[];
  /**
   * Per-particle inverse mass (kg⁻¹). Pinned vertices return `0`; every
   * other vertex is in at least one triangle, so its mass is positive.
   * Unpinned vertices: `1 / (vertexArea · surfaceDensity)`.
   * Vertex area is `(1/3) · Σ A_t` over incident triangles `t`
   * (lumped-mass convention; see Grinspun 2003 §2 "Dynamics" — the
   * mass at a vertex is a third of the total area of the incident
   * triangles, scaled by the area mass density).
   */
  readonly invMass: readonly number[];
  /**
   * Distance constraint participants — every unique unordered triangle
   * edge `(i, j)` with `i < j`. Length is the number of edges; each
   * entry is exactly two indices into {@link positions}.
   */
  readonly distancePairs: readonly (readonly [number, number])[];
  /**
   * Rest length per distance pair (metres), measured at construction
   * from `positions[i]` ↔ `positions[j]`. Same length as
   * {@link distancePairs}.
   */
  readonly distanceRestLengths: readonly number[];
  /**
   * Triangle list — one entry per non-degenerate triangle in the
   * input geometry, expressed as a triple of dedup'd vertex indices.
   * Order matches the input index buffer (degenerate triangles are
   * skipped, so the count may be less than `indexBuffer.count / 3`).
   * Used for per-triangle forces such as wind (Keckeisen et al. 2004).
   */
  readonly triangles: readonly (readonly [number, number, number])[];
  /**
   * Bending constraint tuples — one per shared edge (an edge incident
   * to exactly two triangles). Each tuple is `[p1, p2, p3, p4]` per
   * Bender 2014 §3.4.2: `p1, p2` are the shared-edge endpoints,
   * `p3` is the far vertex of one incident triangle, `p4` the far
   * vertex of the other. Boundary edges (incident to one triangle)
   * and non-manifold edges (>2 triangles) are skipped.
   */
  readonly bendingTuples: readonly (readonly [number, number, number, number])[];
  /**
   * Rest dihedral angle per bending tuple, in radians ∈ (−π, π].
   * Measured at construction as `atan2((N_1×N_2)·E, (N_1·N_2)·|E|)`
   * with `N_1 = (p3-p1) × (p3-p2)`, `N_2 = (p4-p2) × (p4-p1)`,
   * `E = p2 - p1` — Bridson 2003 §4 SIGNED-angle convention (flat = 0,
   * valley vs mountain fold = signed). Same convention as
   * `bending.ts` so the §4 closed-form gradients apply directly.
   *
   * **Why signed (atan2), not unsigned (acos).** Bridson 2003 §4 (top
   * of page 5) computes `sin(θ/2) = ±√((1−n̂_1·n̂_2)/2)` with the sign
   * from `(n̂_1 × n̂_2) · ê`, so his closed-form `u_k` gradients are for
   * the signed dihedral. Using `acos` (unsigned θ) folds positive- and
   * negative-side bends onto the same value, causing the gradient
   * sign to be wrong on one half — the cloth amplifies any
   * out-of-plane perturbation instead of damping it. See `bending.ts`
   * for the longer rationale.
   *
   * A flat planar mesh gives `restAngle = 0`; a valley fold has
   * positive rest angle, a mountain fold negative.
   */
  readonly bendingRestAngles: readonly number[];
  /**
   * Diagnostic counts. `nonManifoldEdgesSkipped > 0` means the input
   * mesh has edges shared by 3+ triangles (rare for cloth meshes but
   * possible in poorly-authored geometry); those edges contribute
   * distance constraints but not bending constraints. `boundaryEdges`
   * is also informational — every boundary edge yields a distance
   * constraint but no bending tuple, which is correct (no second
   * triangle to bend against).
   */
  readonly diagnostics: {
    readonly boundaryEdges: number;
    readonly nonManifoldEdgesSkipped: number;
    readonly degenerateTrianglesSkipped: number;
  };
}

export interface ClothGraphOptions {
  /**
   * Surface density in kg/m². Default `0.2`, about light cotton.
   */
  readonly surfaceDensity?: number;
  /**
   * Vertex indices to pin (set `invMass = 0`). Each must be an integer in
   * `[0, positions.length)`. Indices reference the
   * **deduplicated** vertex list returned in {@link ClothGraph.positions} —
   * call {@link createClothGraph} once and inspect the output before
   * deciding which indices to pin if you need to map from raw geometry
   * indices.
   */
  readonly pinnedIndices?: readonly number[];
}

const DEFAULT_SURFACE_DENSITY = 0.2;
/** Vertices within this distance (metres) are merged, so seams in the input become connected cloth. */
const WELD_EPSILON = 1e-6;

/**
 * Build a {@link ClothGraph} from a `THREE.BufferGeometry`.
 *
 * Input requirements:
 *   - Indexed triangle geometry — `geom.getIndex()` must be non-null.
 *     Non-indexed geometry is rejected (cloth bending requires explicit
 *     edge sharing across triangles, which non-indexed meshes do not
 *     express). Convert via `geom.toIndexed()` or
 *     `BufferGeometryUtils.mergeVertices` upstream.
 *   - A `position` attribute with `itemSize = 3`.
 *   - Every vertex, after welding, in at least one non-degenerate
 *     triangle. A vertex outside every triangle would have no mass.
 *
 * Vertices within 1e-6 m of an earlier vertex are welded into it.
 *
 * Pure-CPU; no GPU or renderer dependency. Safe to call at scene load
 * time.
 */
export function createClothGraph(
  geom: BufferGeometry,
  options: ClothGraphOptions = {},
): ClothGraph {
  const surfaceDensity = options.surfaceDensity ?? DEFAULT_SURFACE_DENSITY;
  const pinnedIndices = options.pinnedIndices ?? [];

  if (!Number.isFinite(surfaceDensity) || surfaceDensity <= 0) {
    throw new Error(
      `createClothGraph: surfaceDensity must be a positive finite number, got ${surfaceDensity}`,
    );
  }

  const indexAttr = geom.getIndex();
  if (indexAttr === null) {
    throw new Error(
      'createClothGraph: input BufferGeometry must be indexed (call geom.toIndexed() or merge vertices upstream)',
    );
  }
  const positionAttr = geom.getAttribute('position');
  if (!positionAttr) {
    throw new Error('createClothGraph: input BufferGeometry has no position attribute');
  }
  if (positionAttr.itemSize !== 3) {
    throw new Error(
      `createClothGraph: position attribute itemSize must be 3, got ${positionAttr.itemSize}`,
    );
  }

  const rawPositions: [number, number, number][] = [];
  const rawCount = positionAttr.count;
  for (let i = 0; i < rawCount; i++) {
    rawPositions.push([positionAttr.getX(i), positionAttr.getY(i), positionAttr.getZ(i)]);
  }

  // Vertex weld. Map old → new index, merging each vertex into the first
  // kept vertex within WELD_EPSILON. Kept vertices are binned in a grid of
  // WELD_EPSILON cells, so any match lies in the vertex's cell or one of its
  // 26 neighbors. Cells are keyed by a hash; a collision only adds
  // candidates, which the distance test rejects.
  const oldToNew = new Uint32Array(rawCount);
  const positions: [number, number, number][] = [];
  const cells = new Map<number, number[]>();
  const cellHash = (x: number, y: number, z: number): number =>
    Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791);
  for (let i = 0; i < rawCount; i++) {
    const p = rawPositions[i]!;
    const cx = Math.floor(p[0] / WELD_EPSILON);
    const cy = Math.floor(p[1] / WELD_EPSILON);
    const cz = Math.floor(p[2] / WELD_EPSILON);
    let match = -1;
    search: for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          for (const k of cells.get(cellHash(cx + dx, cy + dy, cz + dz)) ?? []) {
            const q = positions[k]!;
            const ex = p[0] - q[0];
            const ey = p[1] - q[1];
            const ez = p[2] - q[2];
            if (ex * ex + ey * ey + ez * ez <= WELD_EPSILON * WELD_EPSILON) {
              match = k;
              break search;
            }
          }
        }
    if (match === -1) {
      match = positions.length;
      positions.push(p);
      const key = cellHash(cx, cy, cz);
      const cell = cells.get(key);
      if (cell) cell.push(match);
      else cells.set(key, [match]);
    }
    oldToNew[i] = match;
  }
  const nParticles = positions.length;
  for (let k = 0; k < pinnedIndices.length; k++) {
    const index = pinnedIndices[k]!;
    if (!Number.isInteger(index) || index < 0 || index >= nParticles) {
      throw new Error(
        `createClothGraph: pinnedIndices[${k}] is ${index}, not a vertex index in [0, ${nParticles})`,
      );
    }
  }

  // Walk triangles. Each triangle contributes (a) up to 3 edges to the
  // edge map and (b) up to 3 face areas to the per-vertex area.
  const triCount = (indexAttr.count / 3) | 0;
  if (indexAttr.count !== triCount * 3) {
    throw new Error(`createClothGraph: index count ${indexAttr.count} is not a multiple of 3`);
  }
  const vertexArea = new Float64Array(nParticles);
  /**
   * Edge map: key `min,max` → list of triangles (each triangle is
   * stored as the index of its third "far" vertex). For a manifold
   * cloth edge this list has length 1 (boundary) or 2 (interior).
   * Length 3+ → non-manifold; we record but don't emit a bending tuple.
   */
  const edgeMap = new Map<string, { a: number; b: number; far: number[] }>();
  const edgeKey = (i: number, j: number): string => (i < j ? `${i}-${j}` : `${j}-${i}`);

  let degenerateSkipped = 0;
  const triangles: [number, number, number][] = [];
  for (let t = 0; t < triCount; t++) {
    const ia = oldToNew[indexAttr.getX(t * 3 + 0)]!;
    const ib = oldToNew[indexAttr.getX(t * 3 + 1)]!;
    const ic = oldToNew[indexAttr.getX(t * 3 + 2)]!;
    if (ia === ib || ib === ic || ia === ic) {
      degenerateSkipped++;
      continue;
    }
    const A = positions[ia]!;
    const B = positions[ib]!;
    const C = positions[ic]!;
    const ABx = B[0] - A[0];
    const ABy = B[1] - A[1];
    const ABz = B[2] - A[2];
    const ACx = C[0] - A[0];
    const ACy = C[1] - A[1];
    const ACz = C[2] - A[2];
    // Triangle area = 0.5 · |AB × AC|.
    const nx = ABy * ACz - ABz * ACy;
    const ny = ABz * ACx - ABx * ACz;
    const nz = ABx * ACy - ABy * ACx;
    const area = 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (area <= 0 || !Number.isFinite(area)) {
      degenerateSkipped++;
      continue;
    }
    triangles.push([ia, ib, ic]);
    // Lumped mass: each vertex gets a third of each incident triangle's area.
    const third = area / 3;
    vertexArea[ia]! += third;
    vertexArea[ib]! += third;
    vertexArea[ic]! += third;

    // Add the three edges. Each edge's "far" vertex is the third triangle vertex.
    const addEdge = (u: number, v: number, far: number): void => {
      const k = edgeKey(u, v);
      let rec = edgeMap.get(k);
      if (rec === undefined) {
        rec = { a: Math.min(u, v), b: Math.max(u, v), far: [] };
        edgeMap.set(k, rec);
      }
      rec.far.push(far);
    };
    addEdge(ia, ib, ic);
    addEdge(ib, ic, ia);
    addEdge(ic, ia, ib);
  }

  // A vertex outside every triangle would have no mass and no edges, so it
  // would act as an unattached pin.
  const isolated = vertexArea.findIndex((area) => !(area > 0));
  if (isolated !== -1) {
    throw new Error(
      `createClothGraph: vertex ${isolated} is in no non-degenerate triangle; remove unused vertices from the geometry`,
    );
  }

  // Inverse-mass per vertex — pinned slots forced to 0.
  const pinnedSet = new Set<number>(pinnedIndices);
  const invMass: number[] = new Array(nParticles);
  for (let i = 0; i < nParticles; i++) {
    invMass[i] = pinnedSet.has(i) ? 0 : 1 / (vertexArea[i]! * surfaceDensity);
  }

  // Distance pairs — every unique edge.
  const distancePairs: [number, number][] = [];
  const distanceRestLengths: number[] = [];
  // Bending tuples — every edge with exactly two incident triangles.
  const bendingTuples: [number, number, number, number][] = [];
  const bendingRestAngles: number[] = [];
  let boundaryEdges = 0;
  let nonManifoldSkipped = 0;
  for (const rec of edgeMap.values()) {
    distancePairs.push([rec.a, rec.b]);
    const pa = positions[rec.a]!;
    const pb = positions[rec.b]!;
    const dx = pa[0] - pb[0];
    const dy = pa[1] - pb[1];
    const dz = pa[2] - pb[2];
    distanceRestLengths.push(Math.sqrt(dx * dx + dy * dy + dz * dz));

    if (rec.far.length === 1) {
      boundaryEdges++;
    } else if (rec.far.length === 2) {
      const p1 = rec.a;
      const p2 = rec.b;
      const p3 = rec.far[0]!;
      const p4 = rec.far[1]!;
      bendingTuples.push([p1, p2, p3, p4]);
      bendingRestAngles.push(measureDihedralAngle(positions, p1, p2, p3, p4));
    } else {
      nonManifoldSkipped++;
    }
  }

  return {
    positions,
    invMass,
    triangles,
    distancePairs,
    distanceRestLengths,
    bendingTuples,
    bendingRestAngles,
    diagnostics: {
      boundaryEdges,
      nonManifoldEdgesSkipped: nonManifoldSkipped,
      degenerateTrianglesSkipped: degenerateSkipped,
    },
  };
}

/**
 * Compute the SIGNED dihedral angle
 * `atan2((N_1×N_2)·E, (N_1·N_2)·|E|)` per Bridson 2003 §4 with
 * `N_1 = (p3-p1) × (p3-p2)`, `N_2 = (p4-p2) × (p4-p1)`, `E = p2-p1`.
 * Returns 0 when either `N_k` has zero magnitude (degenerate
 * triangle pair) or when the edge is degenerate; the bending kernel
 * also guards against these at solve time.
 *
 * The atan2 form gives flat = 0 (valley vs mountain encoded by sign)
 * and is consistent with the closed-form gradients in `bending.ts`.
 */
function measureDihedralAngle(
  positions: readonly (readonly [number, number, number])[],
  i1: number,
  i2: number,
  i3: number,
  i4: number,
): number {
  const p1 = positions[i1]!;
  const p2 = positions[i2]!;
  const p3 = positions[i3]!;
  const p4 = positions[i4]!;
  // E = p2 - p1
  const ex = p2[0] - p1[0];
  const ey = p2[1] - p1[1];
  const ez = p2[2] - p1[2];
  const eLen = Math.sqrt(ex * ex + ey * ey + ez * ez);
  if (eLen < 1e-12) return 0;
  // N_1 = (p3 - p1) × (p3 - p2)
  const a3x = p3[0] - p1[0];
  const a3y = p3[1] - p1[1];
  const a3z = p3[2] - p1[2];
  const b3x = p3[0] - p2[0];
  const b3y = p3[1] - p2[1];
  const b3z = p3[2] - p2[2];
  const n1x = a3y * b3z - a3z * b3y;
  const n1y = a3z * b3x - a3x * b3z;
  const n1z = a3x * b3y - a3y * b3x;
  // N_2 = (p4 - p2) × (p4 - p1)
  const a4x = p4[0] - p2[0];
  const a4y = p4[1] - p2[1];
  const a4z = p4[2] - p2[2];
  const b4x = p4[0] - p1[0];
  const b4y = p4[1] - p1[1];
  const b4z = p4[2] - p1[2];
  const n2x = a4y * b4z - a4z * b4y;
  const n2y = a4z * b4x - a4x * b4z;
  const n2z = a4x * b4y - a4y * b4x;
  const n1Sq = n1x * n1x + n1y * n1y + n1z * n1z;
  const n2Sq = n2x * n2x + n2y * n2y + n2z * n2z;
  if (n1Sq < 1e-24 || n2Sq < 1e-24) return 0;
  // (N_1 × N_2)
  const cx = n1y * n2z - n1z * n2y;
  const cy = n1z * n2x - n1x * n2z;
  const cz = n1x * n2y - n1y * n2x;
  const sinTimes = cx * ex + cy * ey + cz * ez; // = |N1|·|N2|·|E|·sin(θ_signed)
  const cosTimes = (n1x * n2x + n1y * n2y + n1z * n2z) * eLen; // = |N1|·|N2|·|E|·cos(θ)
  return Math.atan2(sinTimes, cosTimes);
}
