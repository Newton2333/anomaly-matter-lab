import type { SDFData } from '../core/collision/SDFCollider.js';

/**
 * Custom `.sdf.bin` binary format — v1.
 *
 * Little-endian. Header is 48 bytes, followed by raw f32 voxel data in
 * z-major order: `voxel[x + y·resX + z·resX·resY]`.
 *
 *   offset  size  field
 *   ------  ----  -----
 *        0     4  magic "PSDF" (0x50, 0x53, 0x44, 0x46)
 *        4     4  version u32 = 1
 *        8     4  resolutionX u32
 *       12     4  resolutionY u32
 *       16     4  resolutionZ u32
 *       20     4  reserved u32 (0)
 *       24    12  origin vec3 f32
 *       36    12  voxelSize vec3 f32
 *       48   ...  raw voxel data (f32 × resX·resY·resZ)
 */
export const SDF_MAGIC = 0x46445350; // "PSDF" read as little-endian u32
export const SDF_VERSION = 1;
const HEADER_BYTES = 48;

/** Serialize an in-memory `SDFData` into the `.sdf.bin` wire format. */
export function encodeSdfBinary(sdf: SDFData): ArrayBuffer {
  const [resX, resY, resZ] = sdf.resolution;
  const voxelCount = resX * resY * resZ;
  if (voxelCount !== sdf.data.length) {
    throw new Error(
      `encodeSdfBinary: resolution ${resX}·${resY}·${resZ}=${voxelCount} does not match data length ${sdf.data.length}`,
    );
  }
  const buffer = new ArrayBuffer(HEADER_BYTES + voxelCount * 4);
  const header = new DataView(buffer);
  header.setUint32(0, SDF_MAGIC, true);
  header.setUint32(4, SDF_VERSION, true);
  header.setUint32(8, resX, true);
  header.setUint32(12, resY, true);
  header.setUint32(16, resZ, true);
  header.setUint32(20, 0, true);
  header.setFloat32(24, sdf.origin[0], true);
  header.setFloat32(28, sdf.origin[1], true);
  header.setFloat32(32, sdf.origin[2], true);
  header.setFloat32(36, sdf.voxelSize[0], true);
  header.setFloat32(40, sdf.voxelSize[1], true);
  header.setFloat32(44, sdf.voxelSize[2], true);
  const voxels = new Float32Array(buffer, HEADER_BYTES, voxelCount);
  voxels.set(sdf.data);
  return buffer;
}

/** Deserialize a `.sdf.bin` buffer into an `SDFData` — inverse of encode. */
export function decodeSdfBinary(buffer: ArrayBuffer): SDFData {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error(
      `decodeSdfBinary: buffer too small for header (${buffer.byteLength} < ${HEADER_BYTES})`,
    );
  }
  const header = new DataView(buffer);
  const magic = header.getUint32(0, true);
  if (magic !== SDF_MAGIC) {
    throw new Error(
      `decodeSdfBinary: bad magic 0x${magic.toString(16)} (expected 0x${SDF_MAGIC.toString(16)} "PSDF")`,
    );
  }
  const version = header.getUint32(4, true);
  if (version !== SDF_VERSION) {
    throw new Error(`decodeSdfBinary: unsupported version ${version} (expected ${SDF_VERSION})`);
  }
  const resX = header.getUint32(8, true);
  const resY = header.getUint32(12, true);
  const resZ = header.getUint32(16, true);
  const originX = header.getFloat32(24, true);
  const originY = header.getFloat32(28, true);
  const originZ = header.getFloat32(32, true);
  const voxelX = header.getFloat32(36, true);
  const voxelY = header.getFloat32(40, true);
  const voxelZ = header.getFloat32(44, true);
  if (resX < 2 || resY < 2 || resZ < 2) {
    throw new Error(`decodeSdfBinary: resolution must be ≥ 2, got [${resX}, ${resY}, ${resZ}]`);
  }
  if (![voxelX, voxelY, voxelZ].every((v) => v > 0 && Number.isFinite(v))) {
    throw new Error(
      `decodeSdfBinary: voxelSize must be positive, got [${voxelX}, ${voxelY}, ${voxelZ}]`,
    );
  }
  const voxelCount = resX * resY * resZ;
  const expectedBytes = HEADER_BYTES + voxelCount * 4;
  if (buffer.byteLength < expectedBytes) {
    throw new Error(
      `decodeSdfBinary: buffer too small (${buffer.byteLength} < ${expectedBytes} for ${resX}·${resY}·${resZ} voxels)`,
    );
  }
  // Copy out of the source buffer so the returned Float32Array owns its
  // storage — otherwise a caller that holds onto the decoded data would
  // pin the much-larger source buffer.
  const voxels = new Float32Array(voxelCount);
  voxels.set(new Float32Array(buffer, HEADER_BYTES, voxelCount));
  return {
    data: voxels,
    resolution: [resX, resY, resZ],
    origin: [originX, originY, originZ],
    voxelSize: [voxelX, voxelY, voxelZ],
  };
}
