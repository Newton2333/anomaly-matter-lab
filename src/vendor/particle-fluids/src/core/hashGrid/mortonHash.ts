import { int, uint } from 'three/tsl';

// TSL `@types` strip many proxy methods from GPGPU nodes. Same loose
// alias used in `cellIndex.ts`, `query.ts`, etc.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Bias added to every cell coordinate before Morton encoding so the
 * unsigned bit-interleaving pattern works for negative cell coords. The
 * encoding supports cell coords in `[-MORTON_BIAS, +MORTON_BIAS - 1]` per
 * axis without aliasing.
 *
 * With 4 cm cells that is about ±20 m per axis. Larger scenes need a
 * wider bias and more Morton bits per axis.
 *
 * Particles outside this range do NOT crash. Their Morton-encoded bucket
 * indices wrap due to the unsigned arithmetic + bit-mask, which can cause
 * them to alias with in-range cells. The narrow-phase distance filter in
 * every neighbor-walk caller rejects the resulting false-positive
 * candidates at a per-pair cost. Performance degrades; correctness is
 * preserved. The `overflowFlag` mechanism in `cellIndex.ts` flags this
 * for tests so the situation is observable rather than silent.
 */
export const MORTON_BIAS = 512;

/**
 * Spread the lower 10 bits of `n` over every third bit position, leaving
 * zeros in between. Standard Morton "part1by2" pattern with magic-number
 * bit masks. Implemented inline (not as a WGSL function) because TSL does
 * not expose user-defined helper functions outside `Fn(...)` bodies and
 * this routine must be callable from inside both the cellIndex and walk
 * bucket-computation contexts.
 *
 * Input contract: `n` should hold a value in `[0, 1023]` (10 bits). Higher
 * bits are masked off in the first step. Output is a u32 with the input's
 * lower 10 bits placed at bit positions `0, 3, 6, 9, 12, 15, 18, 21, 24, 27`.
 */
export function part1by2(n: Any): Any {
  const m0: Any = n.bitAnd(uint(0x000003ff));
  const m1a: Any = m0.bitOr(m0.shiftLeft(uint(16)));
  const m1: Any = m1a.bitAnd(uint(0xff0000ff));
  const m2a: Any = m1.bitOr(m1.shiftLeft(uint(8)));
  const m2: Any = m2a.bitAnd(uint(0x0300f00f));
  const m3a: Any = m2.bitOr(m2.shiftLeft(uint(4)));
  const m3: Any = m3a.bitAnd(uint(0x030c30c3));
  const m4a: Any = m3.bitOr(m3.shiftLeft(uint(2)));
  const m4: Any = m4a.bitAnd(uint(0x09249249));
  return m4;
}

/**
 * Compute the unmasked Morton (Z-curve) bucket code for the given signed
 * cell coordinates. The caller masks the result with `(hashTableSize − 1)`
 * to clamp into the bucket range.
 *
 * The bias is added in signed-int space, then `.toUint()` reinterprets the
 * result as unsigned. For coords in `[-MORTON_BIAS, +MORTON_BIAS - 1]` the
 * biased value lands in `[0, 1023]`, which `part1by2` accepts. Coords
 * outside that range still produce a u32 bucket but lose Morton's
 * spatial-locality property — see `MORTON_BIAS` JSDoc.
 *
 * Output is a u32 with up to 30 bits of Morton code, layout
 * `(z bits) (y bits) (x bits)` interleaved so adjacent cells in 3D space
 * have nearby codes (with octant discontinuities every 8, 64, 512, ...
 * cells where the Z-curve folds).
 *
 * Compared with an XOR-mixing hash, Morton order made neighbor walks about
 * 2–2.8× faster on Apple Silicon because nearby cells' lookups stay cached.
 */
export function mortonBucketUnmasked(cx: Any, cy: Any, cz: Any): Any {
  const ux: Any = cx.add(int(MORTON_BIAS)).toUint();
  const uy: Any = cy.add(int(MORTON_BIAS)).toUint();
  const uz: Any = cz.add(int(MORTON_BIAS)).toUint();
  const sx: Any = part1by2(ux);
  const sy: Any = part1by2(uy);
  const sz: Any = part1by2(uz);
  return sx.bitOr(sy.shiftLeft(uint(1))).bitOr(sz.shiftLeft(uint(2)));
}
