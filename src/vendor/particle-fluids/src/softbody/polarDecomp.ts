import { If, float, sqrt } from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Nine scalar TSL nodes forming a 3x3 matrix in row-major order:
 * `m[row][col]` = `m${row}${col}`.
 */
export interface Mat3Nodes {
  readonly m00: Any;
  readonly m01: Any;
  readonly m02: Any;
  readonly m10: Any;
  readonly m11: Any;
  readonly m12: Any;
  readonly m20: Any;
  readonly m21: Any;
  readonly m22: Any;
}

/**
 * TSL emitter for the 3x3 polar decomposition `A = R · S` via Jacobi
 * diagonalization of the symmetric PSD matrix `M = A^T · A`.
 *
 * Used by both global and local shape matching.
 *
 * Input / output are `Mat3Nodes` objects (nine float nodes in row-major
 * order). The caller is responsible for driving this inside an `If` /
 * workgroup gate (the math executes unconditionally in the emitted
 * shader; `SoftbodySystem`'s Pass 2 wraps it in `If(tid == 0, ...)` so
 * only one thread per workgroup runs it).
 */
export function emitPolarDecomposition(a: Mat3Nodes): Mat3Nodes {
  // --- Step 1: M = A^T · A (symmetric, 6 scalars) ---
  // A is row-major: a.m{row}{col}. A^T is column-major (same nine values
  // permuted). (A^T · A)[i][j] = Σ_k A^T[i][k] · A[k][j] = Σ_k A[k][i] · A[k][j].
  const m00 = a.m00.mul(a.m00).add(a.m10.mul(a.m10)).add(a.m20.mul(a.m20)).toVar();
  const m01 = a.m00.mul(a.m01).add(a.m10.mul(a.m11)).add(a.m20.mul(a.m21)).toVar();
  const m02 = a.m00.mul(a.m02).add(a.m10.mul(a.m12)).add(a.m20.mul(a.m22)).toVar();
  const m11 = a.m01.mul(a.m01).add(a.m11.mul(a.m11)).add(a.m21.mul(a.m21)).toVar();
  const m12 = a.m01.mul(a.m02).add(a.m11.mul(a.m12)).add(a.m21.mul(a.m22)).toVar();
  const m22 = a.m02.mul(a.m02).add(a.m12.mul(a.m12)).add(a.m22.mul(a.m22)).toVar();

  // --- Step 2: Jacobi-diagonalize M, accumulating Q ---
  // Q starts as identity. 9 mutable scalars, row-major.
  const q00 = float(1.0).toVar();
  const q01 = float(0.0).toVar();
  const q02 = float(0.0).toVar();
  const q10 = float(0.0).toVar();
  const q11 = float(1.0).toVar();
  const q12 = float(0.0).toVar();
  const q20 = float(0.0).toVar();
  const q21 = float(0.0).toVar();
  const q22 = float(1.0).toVar();

  const eps = float(1e-12);

  // Each sweep performs three Jacobi Givens rotations (plane pivots
  // (0,1), (0,2), (1,2)). The mutation helpers below encapsulate the
  // stable Rutishauser formulation:
  //   τ = (m_qq − m_pp) / (2 · m_pq)
  //   t = sign(τ) / (|τ| + √(1 + τ²))
  //   c = 1 / √(1 + t²),  s = t · c
  //   m_pp' = m_pp − t · m_pq,  m_qq' = m_qq + t · m_pq,  m_pq' = 0
  //   (other off-diagonals mix according to the rotation)
  //   Q' = Q · R_pq  (updates Q's columns p and q)
  //
  // Guarded by |m_pq| > ε so near-diagonal planes are no-ops.

  const applyRotation01 = (): void => {
    If(m01.abs().greaterThan(eps), () => {
      const apq: Any = m01;
      const app: Any = m00;
      const aqq: Any = m11;
      const tau: Any = aqq.sub(app).div(apq.mul(float(2.0)));
      const tauSign: Any = tau.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0));
      const tauAbs: Any = tau.abs();
      const t: Any = tauSign.div(tauAbs.add(sqrt(tauAbs.mul(tauAbs).add(float(1.0)))));
      const cc: Any = float(1.0).div(sqrt(t.mul(t).add(float(1.0))));
      const ss: Any = t.mul(cc);

      // Snapshot rows/cols we read-after-write within this rotation.
      const oldM02: Any = m02.toVar();
      const oldM12: Any = m12.toVar();
      const oldQ00: Any = q00.toVar();
      const oldQ01: Any = q01.toVar();
      const oldQ10: Any = q10.toVar();
      const oldQ11: Any = q11.toVar();
      const oldQ20: Any = q20.toVar();
      const oldQ21: Any = q21.toVar();

      // M diagonal + zeroed off-diagonal
      m00.assign(app.sub(t.mul(apq)));
      m11.assign(aqq.add(t.mul(apq)));
      m01.assign(float(0.0));
      // Other off-diagonals mix (k = 2):
      //   m_pk' = c · m_pk − s · m_qk,  m_qk' = s · m_pk + c · m_qk
      m02.assign(cc.mul(oldM02).sub(ss.mul(oldM12)));
      m12.assign(ss.mul(oldM02).add(cc.mul(oldM12)));

      // Q columns p = 0, q = 1. Givens R_pq = [[c, s], [−s, c]] in the
      // (p, q) plane (matches the M updates above). Q' = Q · R_pq gives:
      //   Q'[:, 0] = c · Q[:, 0] − s · Q[:, 1]
      //   Q'[:, 1] = s · Q[:, 0] + c · Q[:, 1]
      q00.assign(cc.mul(oldQ00).sub(ss.mul(oldQ01)));
      q01.assign(ss.mul(oldQ00).add(cc.mul(oldQ01)));
      q10.assign(cc.mul(oldQ10).sub(ss.mul(oldQ11)));
      q11.assign(ss.mul(oldQ10).add(cc.mul(oldQ11)));
      q20.assign(cc.mul(oldQ20).sub(ss.mul(oldQ21)));
      q21.assign(ss.mul(oldQ20).add(cc.mul(oldQ21)));
    });
  };

  const applyRotation02 = (): void => {
    If(m02.abs().greaterThan(eps), () => {
      const apq: Any = m02;
      const app: Any = m00;
      const aqq: Any = m22;
      const tau: Any = aqq.sub(app).div(apq.mul(float(2.0)));
      const tauSign: Any = tau.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0));
      const tauAbs: Any = tau.abs();
      const t: Any = tauSign.div(tauAbs.add(sqrt(tauAbs.mul(tauAbs).add(float(1.0)))));
      const cc: Any = float(1.0).div(sqrt(t.mul(t).add(float(1.0))));
      const ss: Any = t.mul(cc);

      const oldM01: Any = m01.toVar();
      const oldM12: Any = m12.toVar();
      const oldQ00: Any = q00.toVar();
      const oldQ02: Any = q02.toVar();
      const oldQ10: Any = q10.toVar();
      const oldQ12: Any = q12.toVar();
      const oldQ20: Any = q20.toVar();
      const oldQ22: Any = q22.toVar();

      m00.assign(app.sub(t.mul(apq)));
      m22.assign(aqq.add(t.mul(apq)));
      m02.assign(float(0.0));
      // Other off-diagonals (k = 1):
      m01.assign(cc.mul(oldM01).sub(ss.mul(oldM12)));
      m12.assign(ss.mul(oldM01).add(cc.mul(oldM12)));

      // Q columns p = 0, q = 2 (see applyRotation01 for the convention).
      q00.assign(cc.mul(oldQ00).sub(ss.mul(oldQ02)));
      q02.assign(ss.mul(oldQ00).add(cc.mul(oldQ02)));
      q10.assign(cc.mul(oldQ10).sub(ss.mul(oldQ12)));
      q12.assign(ss.mul(oldQ10).add(cc.mul(oldQ12)));
      q20.assign(cc.mul(oldQ20).sub(ss.mul(oldQ22)));
      q22.assign(ss.mul(oldQ20).add(cc.mul(oldQ22)));
    });
  };

  const applyRotation12 = (): void => {
    If(m12.abs().greaterThan(eps), () => {
      const apq: Any = m12;
      const app: Any = m11;
      const aqq: Any = m22;
      const tau: Any = aqq.sub(app).div(apq.mul(float(2.0)));
      const tauSign: Any = tau.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0));
      const tauAbs: Any = tau.abs();
      const t: Any = tauSign.div(tauAbs.add(sqrt(tauAbs.mul(tauAbs).add(float(1.0)))));
      const cc: Any = float(1.0).div(sqrt(t.mul(t).add(float(1.0))));
      const ss: Any = t.mul(cc);

      const oldM01: Any = m01.toVar();
      const oldM02: Any = m02.toVar();
      const oldQ01: Any = q01.toVar();
      const oldQ02: Any = q02.toVar();
      const oldQ11: Any = q11.toVar();
      const oldQ12: Any = q12.toVar();
      const oldQ21: Any = q21.toVar();
      const oldQ22: Any = q22.toVar();

      m11.assign(app.sub(t.mul(apq)));
      m22.assign(aqq.add(t.mul(apq)));
      m12.assign(float(0.0));
      // Other off-diagonals (k = 0):
      m01.assign(cc.mul(oldM01).sub(ss.mul(oldM02)));
      m02.assign(ss.mul(oldM01).add(cc.mul(oldM02)));

      // Q columns p = 1, q = 2 (see applyRotation01 for the convention).
      q01.assign(cc.mul(oldQ01).sub(ss.mul(oldQ02)));
      q02.assign(ss.mul(oldQ01).add(cc.mul(oldQ02)));
      q11.assign(cc.mul(oldQ11).sub(ss.mul(oldQ12)));
      q12.assign(ss.mul(oldQ11).add(cc.mul(oldQ12)));
      q21.assign(cc.mul(oldQ21).sub(ss.mul(oldQ22)));
      q22.assign(ss.mul(oldQ21).add(cc.mul(oldQ22)));
    });
  };

  for (let sweep = 0; sweep < 8; sweep++) {
    applyRotation01();
    applyRotation02();
    applyRotation12();
  }

  // --- Step 3: S^-1 = Q · diag(1/√λ_i) · Q^T ---
  // After Jacobi, the diagonal of M holds the eigenvalues λ_i. Floor at
  // 1e-20 to keep the inverse finite in degenerate edge cases (the
  // rank-3 precondition at SoftbodySystem construction ensures this
  // floor never clips in practice).
  const invSqrtL0: Any = float(1.0).div(sqrt(m00.max(float(1e-20))));
  const invSqrtL1: Any = float(1.0).div(sqrt(m11.max(float(1e-20))));
  const invSqrtL2: Any = float(1.0).div(sqrt(m22.max(float(1e-20))));

  // Q is stored with columns being the eigenvectors. Expand the
  // sum-of-outer-products form `S^-1 = Σ_k (1/√λ_k) q_k q_k^T`:
  //   S^-1[i][j] = Σ_k (1/√λ_k) · Q[i][k] · Q[j][k]
  const si00: Any = q00
    .mul(q00)
    .mul(invSqrtL0)
    .add(q01.mul(q01).mul(invSqrtL1))
    .add(q02.mul(q02).mul(invSqrtL2));
  const si01: Any = q00
    .mul(q10)
    .mul(invSqrtL0)
    .add(q01.mul(q11).mul(invSqrtL1))
    .add(q02.mul(q12).mul(invSqrtL2));
  const si02: Any = q00
    .mul(q20)
    .mul(invSqrtL0)
    .add(q01.mul(q21).mul(invSqrtL1))
    .add(q02.mul(q22).mul(invSqrtL2));
  const si11: Any = q10
    .mul(q10)
    .mul(invSqrtL0)
    .add(q11.mul(q11).mul(invSqrtL1))
    .add(q12.mul(q12).mul(invSqrtL2));
  const si12: Any = q10
    .mul(q20)
    .mul(invSqrtL0)
    .add(q11.mul(q21).mul(invSqrtL1))
    .add(q12.mul(q22).mul(invSqrtL2));
  const si22: Any = q20
    .mul(q20)
    .mul(invSqrtL0)
    .add(q21.mul(q21).mul(invSqrtL1))
    .add(q22.mul(q22).mul(invSqrtL2));
  // S^-1 is symmetric: si10 = si01, si20 = si02, si21 = si12.

  // --- Step 4: R = A · S^-1 ---
  // R[i][j] = Σ_k A[i][k] · S^-1[k][j]
  const r00: Any = a.m00.mul(si00).add(a.m01.mul(si01)).add(a.m02.mul(si02));
  const r01: Any = a.m00.mul(si01).add(a.m01.mul(si11)).add(a.m02.mul(si12));
  const r02: Any = a.m00.mul(si02).add(a.m01.mul(si12)).add(a.m02.mul(si22));
  const r10: Any = a.m10.mul(si00).add(a.m11.mul(si01)).add(a.m12.mul(si02));
  const r11: Any = a.m10.mul(si01).add(a.m11.mul(si11)).add(a.m12.mul(si12));
  const r12: Any = a.m10.mul(si02).add(a.m11.mul(si12)).add(a.m12.mul(si22));
  const r20: Any = a.m20.mul(si00).add(a.m21.mul(si01)).add(a.m22.mul(si02));
  const r21: Any = a.m20.mul(si01).add(a.m21.mul(si11)).add(a.m22.mul(si12));
  const r22: Any = a.m20.mul(si02).add(a.m21.mul(si12)).add(a.m22.mul(si22));

  return {
    m00: r00,
    m01: r01,
    m02: r02,
    m10: r10,
    m11: r11,
    m12: r12,
    m20: r20,
    m21: r21,
    m22: r22,
  };
}
