import type { BufferGeometry } from 'three';

/** An indexed triangle mesh as flat arrays. */
export interface TriangleMesh {
  /** xyz per vertex. */
  readonly vertices: Float32Array;
  /** Three vertex indices per triangle. */
  readonly indices: Uint32Array;
}

/**
 * Read a geometry's positions and triangles into new arrays. Non-indexed
 * geometry is treated as a triangle soup. A `TriangleMesh` is validated and
 * copied. Throws on malformed input.
 */
export function toTriangleMesh(mesh: BufferGeometry | TriangleMesh): TriangleMesh {
  if ('vertices' in mesh) {
    const { vertices, indices } = mesh;
    if (vertices.length % 3 !== 0) {
      throw new Error(`TriangleMesh: vertices length ${vertices.length} is not a multiple of 3`);
    }
    return validated(Float32Array.from(vertices), Uint32Array.from(indices));
  }
  if (typeof (mesh as Partial<BufferGeometry>).getAttribute !== 'function') {
    throw new Error('toTriangleMesh: expected a BufferGeometry or a TriangleMesh');
  }
  const position = mesh.getAttribute('position');
  if (!position) throw new Error('toTriangleMesh: geometry has no position attribute');
  if (position.itemSize < 3) {
    throw new Error(`toTriangleMesh: position attribute has itemSize ${position.itemSize}, need 3`);
  }
  const vertices = new Float32Array(position.count * 3);
  for (let i = 0; i < position.count; i++) {
    vertices[i * 3] = position.getX(i);
    vertices[i * 3 + 1] = position.getY(i);
    vertices[i * 3 + 2] = position.getZ(i);
  }
  const index = mesh.getIndex();
  const indices = index
    ? Uint32Array.from(index.array)
    : Uint32Array.from({ length: position.count }, (_, i) => i);
  return validated(vertices, indices);
}

function validated(vertices: Float32Array, indices: Uint32Array): TriangleMesh {
  if (indices.length % 3 !== 0) {
    throw new Error(`TriangleMesh: indices length ${indices.length} is not a multiple of 3`);
  }
  const count = vertices.length / 3;
  for (const index of indices) {
    if (index >= count) throw new Error(`TriangleMesh: index ${index} is out of range`);
  }
  return { vertices, indices };
}
