import { Color, InstancedMesh, PlaneGeometry } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  cameraWorldMatrix,
  exp,
  float,
  instanceIndex,
  positionLocal,
  uniform,
  uv,
  vec3,
} from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { SmokeTracers } from './types.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface GasSpriteRendererOptions {
  /** Default `0xeeeeee`. Ignored when `colorNode` is given. */
  readonly color?: number | string;
  /** Sprite width in metres. Default 0.08. */
  readonly size?: number;
  /** Opacity of a new tracer, in (0, 1]. Default 0.6. */
  readonly initialOpacity?: number;
  /** Seconds for opacity to fall by a factor of e. Default a third of the lifetime. */
  readonly opacityTau?: number;
  /** Per-sprite color from the tracer's position and velocity, as TSL vec3 nodes. */
  readonly colorNode?: (position: Any, velocity: Any) => Any;
}

/**
 * Draws smoke tracers as soft camera-facing sprites that fade with age.
 * Cheap and good for seeing individual tracers; for convincing smoke use
 * {@link GasVolumeRenderer}. Sprites aren't depth sorted.
 */
export class GasSpriteRenderer {
  readonly object: InstancedMesh;
  private readonly sizeUniform: UniformNode<'float', number>;

  constructor(gas: SmokeTracers, options: GasSpriteRendererOptions = {}) {
    const { color = 0xeeeeee, initialOpacity = 0.6, opacityTau = gas.lifetime / 3 } = options;
    const size = checkSize(options.size ?? 0.08);
    if (!(initialOpacity > 0 && initialOpacity <= 1)) {
      throw new Error(`GasSpriteRenderer: initialOpacity must be in (0, 1], got ${initialOpacity}`);
    }
    if (!(opacityTau > 0))
      throw new Error(`GasSpriteRenderer: opacityTau must be positive, got ${opacityTau}`);

    this.sizeUniform = uniform(size, 'float');
    const material = new MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
    material.color = new Color(color);
    const center: Any = gas.smokePositions.element(instanceIndex).xyz;
    const alive: Any = gas.smokeAlive.element(instanceIndex).toFloat();
    // Dead tracers collapse to a point.
    const extent: Any = this.sizeUniform.mul(alive);
    const right: Any = (cameraWorldMatrix as Any).element(0).xyz;
    const up: Any = (cameraWorldMatrix as Any).element(1).xyz;
    material.positionNode = center
      .add(right.mul((positionLocal as Any).x.mul(extent)))
      .add(up.mul((positionLocal as Any).y.mul(extent)));
    if (options.colorNode) {
      const velocity: Any = gas.smokeVelocities?.element(instanceIndex).xyz ?? vec3(0);
      material.colorNode = options.colorNode(center, velocity);
    }
    const disk: Any = (uv() as Any).mul(2).sub(1);
    const falloff: Any = float(1).sub(disk.dot(disk)).max(0);
    const fade: Any = exp(gas.smokeAge.element(instanceIndex).div(opacityTau).negate())
      .mul(initialOpacity)
      .mul(alive);
    material.opacityNode = fade.mul(falloff.mul(falloff));

    this.object = new InstancedMesh(new PlaneGeometry(1, 1), material, gas.capacity);
    this.object.frustumCulled = false;
  }

  /** Sprite width in metres. */
  get size(): number {
    return this.sizeUniform.value;
  }
  set size(value: number) {
    this.sizeUniform.value = checkSize(value);
  }

  dispose(): void {
    this.object.geometry.dispose();
    (this.object.material as MeshBasicNodeMaterial).dispose();
  }
}

function checkSize(size: number): number {
  if (!(size > 0) || !Number.isFinite(size)) {
    throw new Error(`GasSpriteRenderer: size must be positive, got ${size}`);
  }
  return size;
}
