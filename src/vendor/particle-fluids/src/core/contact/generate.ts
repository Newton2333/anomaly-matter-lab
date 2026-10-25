import {
  Continue,
  Fn,
  If,
  Return,
  atomicAdd,
  atomicStore,
  instanceIndex,
  int,
  uint,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import type { HashGrid } from '../hashGrid/HashGrid.js';
import { emitForEachNeighbor } from '../hashGrid/query.js';
import type { ParticleSystem } from '../particles.js';
import type { ContactBuffer } from './ContactBuffer.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Candidate pairs are gathered within `2r · RADIUS_EXPANSION` so particles
 * that move into contact during the solve are still caught (Macklin et al.
 * 2014, §9). The solve only corrects pairs that actually overlap.
 */
export const CONTACT_RADIUS_EXPANSION = 1.1;

/**
 * The subset of particles that search for contacts. Pairs where neither
 * particle is an emitter are never generated.
 */
export interface ContactEmitters {
  /** 1 for emitters, 0 otherwise, per particle. */
  readonly isEmitter: StorageBufferNode<'uint'>;
}

/**
 * Find overlapping particle pairs and append them to `contacts`, with their
 * multipliers zeroed. One thread
 * per emitting particle walks its grid neighborhood. Each pair is emitted
 * once: by the lower index when both ends are emitters, otherwise by the
 * emitter. Pairs in the same non-zero collision group, and pairs of two
 * pinned particles, are skipped.
 */
export function buildContactGenerateKernel(args: {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  readonly contacts: ContactBuffer;
  /** Omit to let every particle emit. */
  readonly emitters?: ContactEmitters;
}): ComputeNode {
  const { particles, hashGrid, contacts, emitters } = args;
  const candidateRadius = 2 * particles.particleRadius * CONTACT_RADIUS_EXPANSION;
  const maxContacts = contacts.maxContacts;

  // Threads run in the grid's sorted order, so a workgroup handles nearby
  // particles whose walks share cells and stay in cache.
  return Fn(() => {
    const slot: Any = instanceIndex;
    const i: Any = hashGrid.sortedIndices.element(slot).toVar();
    if (emitters) {
      If(emitters.isEmitter.element(i).equal(uint(0)), () => {
        Return();
      });
    }
    const xi: Any = hashGrid.sortedPredictedPositions.element(slot).xyz.toVar();
    const groupI: Any = particles.collisionGroup.element(i).toVar();
    const wi: Any = particles.invMass.element(i).toVar();

    emitForEachNeighbor(hashGrid, xi, (j: Any, candidate: Any) => {
      if (emitters) {
        If(j.equal(i), () => {
          Continue();
        });
        If(emitters.isEmitter.element(j).equal(uint(1)).and(j.lessThan(i)), () => {
          Continue();
        });
      } else {
        If(j.lessThanEqual(i), () => {
          Continue();
        });
      }
      const groupJ: Any = particles.collisionGroup.element(j);
      If(groupI.notEqual(uint(0)).and(groupI.equal(groupJ)), () => {
        Continue();
      });
      If(wi.lessThanEqual(0).and(particles.invMass.element(j).lessThanEqual(0)), () => {
        Continue();
      });
      const offset: Any = xi.sub(hashGrid.sortedPredictedPositions.element(candidate).xyz);
      If(offset.dot(offset).greaterThanEqual(candidateRadius * candidateRadius), () => {
        Continue();
      });

      const slot: Any = atomicAdd(contacts.counter.element(uint(0)), uint(1));
      If(slot.lessThan(uint(maxContacts)), () => {
        const record: Any = contacts.records.element(slot);
        record.get('i').assign(i);
        record.get('j').assign(j);
        atomicStore(record.get('lambdaN'), int(0));
        atomicStore(record.get('lambdaT'), int(0));
      });
    });
  })()
    .compute(particles.capacity)
    .setName('generate.contactGenerate');
}
