import type { ClothGraph } from './graph.js';

/**
 * One Long-Range-Attachment (LRA) constraint per Kim, Chentanez,
 * Müller-Fischer 2012 §3.1 — a unilateral distance constraint
 * between a free cloth particle and a pinned attachment particle.
 *
 *   `C(x_i) = |x_i - x_a| - r_i ≤ 0`
 *
 * `restRadius` is the **geodesic** (along-surface) shortest path
 * from the attachment to the particle at rest, per §3.2 — Euclidean
 * distance fails for pre-sculpted curved cloth where straightening
 * geometry would exceed straight-line distance (paper Fig. 4).
 *
 * The anchor is stored as a particle index, not a position, so the
 * solver reads the pinned particle's current position every substep
 * and the tethers follow pins that are moved at runtime.
 */
export interface TetherConstraint {
  /** Cloth-local index of the constrained free particle (`invMass > 0`). */
  readonly particle: number;
  /** Cloth-local index of the pinned particle the tether hangs from. */
  readonly anchor: number;
  /**
   * Geodesic rest distance from `particle` to `anchor` along the cloth
   * surface, in metres. Inflated by `(1 + stretchTolerance)` per
   * paper §3.5 if the build was given a non-zero tolerance.
   */
  readonly restRadius: number;
}

export interface BuildTethersOptions {
  /**
   * Max number of LRA constraints to assign to any one free particle —
   * Kim 2012 §3.4 "Pruning Attachment Points". Default `4` matching the
   * paper's typical-N. The build picks up to `N` islands per particle,
   * one constraint per island (the closest pinned vertex *within* that
   * island). Scenes with fewer islands than `N` produce fewer
   * constraints per particle.
   */
  readonly maxAttachmentsPerParticle?: number;
  /**
   * Per Kim 2012 §3.5 "Controlled Stretchiness" — every `r_i` is
   * multiplied by `(1 + stretchTolerance)`. Default `0` (strict
   * inextensibility). The paper shows `0.1`–`0.2` produces more
   * natural-looking folds on over-constrained scenes (their Fig. 5).
   */
  readonly stretchTolerance?: number;
}

const DEFAULT_MAX_ATTACHMENTS = 4;
const DEFAULT_STRETCH_TOLERANCE = 0;

/**
 * Build LRA constraints for every free particle in the cloth, per
 * Kim 2012 Algorithm 2 ("Assigning attachment to each particle").
 *
 * Algorithm:
 *
 *   1. Identify pinned particles (`graph.invMass[i] === 0`).
 *   2. Connected components: BFS over the **pinned-only** subgraph
 *      of `graph.distancePairs` to partition pinned particles into
 *      islands. Two pinned vertices are in the same island iff they
 *      are connected via cloth edges that pass only through pinned
 *      vertices (paper §3.4 — "a set of connected islands (e.g.
 *      waistline for a skirt, shoulder for sleeves, a group of
 *      pinned points)").
 *   3. For each island: run multi-source Dijkstra over the **full**
 *      edge graph with edge weights = rest length, sources = every
 *      pinned vertex in the island. The result gives, for every
 *      free particle, the geodesic distance to the nearest pinned
 *      vertex in that island.
 *   4. For each free particle: sort islands by geodesic distance,
 *      take the closest `N`, emit one LRA constraint per island
 *      anchored at the **nearest pinned vertex**
 *      (paper §3.4 — "the closest attachment point from each
 *      island"). Apply `(1 + stretchTolerance)` to each `r_i`.
 *
 * Edge weights are rest lengths — the geodesic distance is along
 * the cloth surface, not Euclidean. Paper §3.2 motivates this:
 * Euclidean distance under-estimates path length on pre-sculpted
 * curved cloth, so straightening geometry can exceed the rest
 * radius and falsely trigger the LRA projection.
 *
 * Pure CPU; no GPU dependency. Safe to call at scene-load time.
 *
 * Returns an empty array when there are no pinned vertices (a
 * cloth with no attachments has no LRA constraints to enforce).
 */
export function buildTethers(args: {
  readonly graph: ClothGraph;
  readonly options?: BuildTethersOptions;
}): readonly TetherConstraint[] {
  const { graph, options = {} } = args;
  const maxAttachments = options.maxAttachmentsPerParticle ?? DEFAULT_MAX_ATTACHMENTS;
  const stretchTolerance = options.stretchTolerance ?? DEFAULT_STRETCH_TOLERANCE;

  if (!Number.isInteger(maxAttachments) || maxAttachments <= 0) {
    throw new Error(
      `buildTethers: maxAttachmentsPerParticle must be a positive integer, got ${maxAttachments}`,
    );
  }
  if (!Number.isFinite(stretchTolerance) || stretchTolerance < 0) {
    throw new Error(
      `buildTethers: stretchTolerance must be a non-negative finite number, got ${stretchTolerance}`,
    );
  }

  const n = graph.positions.length;
  const pinned: boolean[] = new Array(n);
  let pinnedCount = 0;
  for (let i = 0; i < n; i++) {
    pinned[i] = graph.invMass[i] === 0;
    if (pinned[i]) pinnedCount++;
  }
  if (pinnedCount === 0) return [];

  // Build adjacency list with edge weights = rest length.
  type AdjEntry = { to: number; weight: number };
  const adj: AdjEntry[][] = Array.from({ length: n }, () => []);
  for (let e = 0; e < graph.distancePairs.length; e++) {
    const [a, b] = graph.distancePairs[e]!;
    const w = graph.distanceRestLengths[e]!;
    adj[a]!.push({ to: b, weight: w });
    adj[b]!.push({ to: a, weight: w });
  }

  // Step 1: Connected components on the pinned-only subgraph (Kim
  // 2012 Algorithm 2 island identification). Two pinned vertices
  // are in the same island iff connected via pinned-only edges.
  const island: number[] = new Array(n).fill(-1);
  let nIslands = 0;
  for (let s = 0; s < n; s++) {
    if (!pinned[s] || island[s] !== -1) continue;
    const id = nIslands++;
    const stack: number[] = [s];
    island[s] = id;
    while (stack.length > 0) {
      const u = stack.pop()!;
      for (const { to } of adj[u]!) {
        if (pinned[to] && island[to] === -1) {
          island[to] = id;
          stack.push(to);
        }
      }
    }
  }

  // Step 2: For each island, run multi-source Dijkstra over the
  // full edge graph with sources = pinned vertices in that island.
  // Result: per (free particle, island), the geodesic distance to
  // the nearest pinned vertex in that island, plus the index of
  // that pinned vertex (the anchor).
  const INF = Number.POSITIVE_INFINITY;
  // Per-island per-particle: distance + nearest-pinned-vertex index.
  // Stored as flat arrays to avoid per-iteration allocations.
  type Attachment = { distance: number; anchorVertex: number };
  // perParticleAttachments[v] = list of (island, distance, anchorVertex)
  // for every island that reached this free particle.
  const perParticleAttachments: { islandId: number; attachment: Attachment }[][] = Array.from(
    { length: n },
    () => [],
  );

  for (let id = 0; id < nIslands; id++) {
    const dist = new Float64Array(n);
    const nearest = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      dist[i] = INF;
      nearest[i] = -1;
    }
    // Initialize sources.
    const heap = new BinaryHeap<[number, number]>(([d1], [d2]) => d1 - d2);
    for (let i = 0; i < n; i++) {
      if (pinned[i] && island[i] === id) {
        dist[i] = 0;
        nearest[i] = i;
        heap.push([0, i]);
      }
    }
    while (heap.size > 0) {
      const top = heap.pop()!;
      const [d, u] = top;
      if (d > dist[u]!) continue; // stale heap entry
      for (const { to, weight } of adj[u]!) {
        const nd = d + weight;
        if (nd < dist[to]!) {
          dist[to] = nd;
          nearest[to] = nearest[u]!;
          heap.push([nd, to]);
        }
      }
    }
    // Record reach for free particles only — pinned ones don't get tethers.
    for (let v = 0; v < n; v++) {
      if (pinned[v]) continue;
      if (!Number.isFinite(dist[v]!)) continue;
      perParticleAttachments[v]!.push({
        islandId: id,
        attachment: { distance: dist[v]!, anchorVertex: nearest[v]! },
      });
    }
  }

  // Step 3: For each free particle, pick the `maxAttachments`
  // closest islands and emit one LRA per island.
  const radiusScale = 1 + stretchTolerance;
  const out: TetherConstraint[] = [];
  for (let v = 0; v < n; v++) {
    if (pinned[v]) continue;
    const list = perParticleAttachments[v]!;
    if (list.length === 0) continue;
    // Sort ascending by distance; deterministic tie-breaker by islandId.
    list.sort((a, b) => a.attachment.distance - b.attachment.distance || a.islandId - b.islandId);
    const take = Math.min(maxAttachments, list.length);
    for (let k = 0; k < take; k++) {
      const { attachment } = list[k]!;
      out.push({
        particle: v,
        anchor: attachment.anchorVertex,
        restRadius: attachment.distance * radiusScale,
      });
    }
  }
  return out;
}

// -- Internal: minimal binary min-heap used for Dijkstra above. --
//
// The cloth-tether graph is small (typical scene: a few thousand
// vertices, a few thousand edges). A heap is overkill for that size
// but keeps Dijkstra strictly O((V+E) log V) instead of relying on
// linear scans.
class BinaryHeap<T> {
  private readonly data: T[] = [];
  constructor(private readonly cmp: (a: T, b: T) => number) {}
  get size(): number {
    return this.data.length;
  }
  push(v: T): void {
    this.data.push(v);
    this.siftUp(this.data.length - 1);
  }
  pop(): T | undefined {
    if (this.data.length === 0) return undefined;
    const top = this.data[0]!;
    const last = this.data.pop()!;
    if (this.data.length > 0) {
      this.data[0] = last;
      this.siftDown(0);
    }
    return top;
  }
  private siftUp(i: number): void {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.cmp(this.data[i]!, this.data[parent]!) < 0) {
        [this.data[i], this.data[parent]] = [this.data[parent]!, this.data[i]!];
        i = parent;
      } else break;
    }
  }
  private siftDown(i: number): void {
    const n = this.data.length;
    while (true) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let smallest = i;
      if (l < n && this.cmp(this.data[l]!, this.data[smallest]!) < 0) smallest = l;
      if (r < n && this.cmp(this.data[r]!, this.data[smallest]!) < 0) smallest = r;
      if (smallest === i) break;
      [this.data[i], this.data[smallest]] = [this.data[smallest]!, this.data[i]!];
      i = smallest;
    }
  }
}
