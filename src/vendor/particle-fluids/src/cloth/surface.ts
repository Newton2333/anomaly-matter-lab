import { DoubleSide, Mesh, PlaneGeometry } from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
  Fn,
  cross,
  faceDirection,
  float,
  mat3,
  normalize,
  transformNormalToView,
  uv,
  vec3,
} from 'three/tsl';
import type { ClothSystem } from './ClothSystem.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface ClothSurfaceOptions {
  /** Grid size of the cloth: particles per row and number of rows, in row-major order. */
  readonly columns: number;
  readonly rows: number;
  /** Mesh vertices per grid cell along each axis. Default 3. */
  readonly subdivisions?: number;
  /**
   * Material to draw with. Its `positionNode` and `normalNode` are
   * overwritten, so don't share it with another mesh. Default: a
   * double-sided sheen material.
   */
  readonly material?: MeshPhysicalNodeMaterial;
  /**
   * Fit a cubic B-spline near the particles (default) instead of a
   * Catmull-Rom spline through them. Catmull-Rom shows grid-scale buckling
   * as lumps; the B-spline filters it out while keeping larger folds.
   */
  readonly smooth?: boolean;
}

/**
 * A mesh that follows a grid-shaped cloth on the GPU, as a bicubic surface
 * through the particles with analytic normals. The cloth must come from a
 * grid (such as a `PlaneGeometry`) whose vertices are in row-major order.
 *
 * The particles' simulation positions become the mesh's local positions,
 * as with `createParticleMesh`, so the mesh's world transform (its own and
 * its parents') is applied on top of them. Leave it at the identity to draw
 * the cloth where it is simulated; colliders and other materials don't see
 * the transform.
 */
export function createClothSurface(cloth: ClothSystem, options: ClothSurfaceOptions): Mesh {
  const { particles } = cloth;
  const { columns, rows, subdivisions = 3, smooth = true } = options;
  const offset = cloth.range.start;
  if (columns < 2 || rows < 2 || columns * rows !== cloth.range.count) {
    throw new Error(
      `createClothSurface: a ${columns}×${rows} grid does not match the cloth's ${cloth.range.count} particles`,
    );
  }
  const catmullRom = (t: Any) => {
    const t2 = t.mul(t),
      t3 = t2.mul(t);
    return [
      t.mul(-0.5).add(t2).sub(t3.mul(0.5)),
      t2.mul(-2.5).add(t3.mul(1.5)).add(1),
      t.mul(0.5).add(t2.mul(2)).sub(t3.mul(1.5)),
      t3.sub(t2).mul(0.5),
    ];
  };
  const catmullRomDerivatives = (t: Any) => [
    t.mul(2).sub(t.mul(t).mul(1.5)).sub(0.5),
    t.mul(-5).add(t.mul(t).mul(4.5)),
    t.mul(4).sub(t.mul(t).mul(4.5)).add(0.5),
    t.mul(t).mul(1.5).sub(t),
  ];
  // Uniform cubic B-spline basis and its derivative.
  const bSpline = (t: Any) => {
    const t2 = t.mul(t),
      t3 = t2.mul(t),
      s = float(1).sub(t);
    return [
      s.mul(s).mul(s).div(6),
      t3.mul(3).sub(t2.mul(6)).add(4).div(6),
      t3.mul(-3).add(t2.mul(3)).add(t.mul(3)).add(1).div(6),
      t3.div(6),
    ];
  };
  const bSplineDerivatives = (t: Any) => {
    const t2 = t.mul(t),
      s = float(1).sub(t);
    return [
      s.mul(s).mul(-0.5),
      t2.mul(1.5).sub(t.mul(2)),
      t2.mul(-1.5).add(t).add(0.5),
      t2.mul(0.5),
    ];
  };
  const weights = smooth ? bSpline : catmullRom;
  const derivatives = smooth ? bSplineDerivatives : catmullRomDerivatives;
  const frame: Any = Fn(() => {
    const x: Any = uv()
      .x.mul(columns - 1)
      .min(columns - 1.0001)
      .toVar();
    const y: Any = float(1)
      .sub(uv().y)
      .mul(rows - 1)
      .min(rows - 1.0001)
      .toVar();
    const ix: Any = x.floor(),
      iy: Any = y.floor();
    const wx = weights(x.fract()),
      wy = weights(y.fract());
    const dx = derivatives(x.fract()),
      dy = derivatives(y.fract());
    const p: Any = vec3(0).toVar(),
      tu: Any = vec3(0).toVar(),
      tv: Any = vec3(0).toVar();
    for (let v = 0; v < 4; v++)
      for (let u = 0; u < 4; u++) {
        const index = iy
          .add(v - 1)
          .clamp(0, rows - 1)
          .mul(columns)
          .add(ix.add(u - 1).clamp(0, columns - 1))
          .add(offset)
          .toUint();
        const sample: Any = particles.positions.element(index).xyz.toVar();
        p.addAssign(sample.mul(wx[u]).mul(wy[v]));
        tu.addAssign(sample.mul(dx[u]).mul(wy[v]));
        tv.addAssign(sample.mul(wx[u]).mul(dy[v]));
      }
    return mat3(p, tu, tv);
  })();
  const material =
    options.material ??
    new MeshPhysicalNodeMaterial({ side: DoubleSide, roughness: 0.4, sheen: 1 });
  material.positionNode = frame.element(0);
  const normal = normalize(cross(frame.element(2), frame.element(1))).toVarying();
  // Custom normalNode bypasses the built-in two-sided normal adjustment.
  material.normalNode = normalize(transformNormalToView(normal) as Any).mul(faceDirection as Any);
  const geometry = new PlaneGeometry(1, 1, (columns - 1) * subdivisions, (rows - 1) * subdivisions);
  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}
