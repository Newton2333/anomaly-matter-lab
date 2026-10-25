import assert from 'node:assert/strict';
import {framePercentile,nextGpuProfile,gpuProfileChanged,MIN_GPU_QUALITY} from '../src/gpu-quality.js';

const full={quality:1,reflections:true,canReflect:true};
assert.deepEqual(nextGpuProfile(full,16.7),full);
assert.equal(framePercentile([NaN,0,16,18,17,19]),19);
assert.equal(framePercentile([16,180,17],.5),17); // One allocation/GC spike is not a steady rate.

// A saturated renderer first keeps image resolution and drops the extra SSR
// march; if that is insufficient, repeated observations converge to the floor.
let profile=nextGpuProfile(full,40);
assert.equal(profile.quality,1);assert.equal(profile.reflections,false);
assert.equal(nextGpuProfile(full,20).reflections,false);
assert(nextGpuProfile(profile,20).quality<profile.quality);
profile=nextGpuProfile(profile,30);
assert(profile.quality<1);assert(profile.quality>=MIN_GPU_QUALITY);
for(let i=0;i<30;i++)profile=nextGpuProfile(profile,90);
assert.equal(profile.quality,MIN_GPU_QUALITY);
assert(!gpuProfileChanged(profile,nextGpuProfile(profile,90)));

// Borderline frame times do not undo a downgrade. Recovery requires explicit
// permission from the runtime's long stable-time window and advances slowly.
for(const ms of [18,19,18.5,17,19])assert.deepEqual(nextGpuProfile(profile,ms),profile);
assert.deepEqual(nextGpuProfile(profile,8),profile);
const recovered=nextGpuProfile(profile,8,{recover:true});
assert.equal(recovered.quality,MIN_GPU_QUALITY+.025);assert.equal(recovered.reflections,false);
assert.equal(nextGpuProfile({...profile,quality:1},8,{recover:true}).reflections,false);
assert.equal(nextGpuProfile({...profile,quality:1},5,{recover:true}).reflections,true);

// Particle mode also has a pixel budget, but can never acquire an SSR pass.
const particles={quality:1,reflections:false,canReflect:false};
assert(nextGpuProfile(particles,40).quality<1);
assert.equal(nextGpuProfile(particles,3,{recover:true}).reflections,false);
for(const ms of [NaN,Infinity,0,-5])assert.deepEqual(nextGpuProfile(full,ms),full);
console.log('GPU quality: saturation, floor, hysteresis, recovery and invalid-sample checks passed.');
