import { Mesh, type BufferGeometry, type MeshStandardMaterial } from 'three';
import type { MeshStandardNodeMaterial } from 'three/webgpu';

import { bindSoftbodyMesh } from './bindMesh.js';
import { createSoftbodySkinMaterial } from './skinMaterial.js';
import type { SoftbodySystem } from './SoftbodySystem.js';

/**
 * A mesh that deforms with a soft body. Each vertex follows its nearest
 * particles by dual-quaternion blending on the GPU.
 *
 * `geometry` must be placed exactly where the body's rest shape is (in
 * world space), and the mesh itself must keep an identity transform: the
 * skinning writes world positions.
 */
export class SoftbodyMesh extends Mesh<BufferGeometry, MeshStandardNodeMaterial> {
  readonly softbody: SoftbodySystem;
  readonly bodyIndex: number;

  /**
   * @param material Colors and maps to copy onto the skinned material.
   *   Default: a plain orange-brown standard material.
   */
  constructor(
    softbody: SoftbodySystem,
    bodyIndex: number,
    geometry: BufferGeometry,
    material?: MeshStandardMaterial,
  ) {
    bindSoftbodyMesh(geometry, softbody, bodyIndex);
    super(
      geometry,
      createSoftbodySkinMaterial({
        softbody,
        bodyIndex,
        ...(material ? { sourceMaterial: material } : {}),
      }),
    );
    this.softbody = softbody;
    this.bodyIndex = bodyIndex;
    this.frustumCulled = false;
    this.castShadow = true;
    this.receiveShadow = true;
  }
}
