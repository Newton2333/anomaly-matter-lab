import assert from 'node:assert/strict';
import * as THREE from 'three';
import {stagePixelRatio} from '../src/render-resolution.js';
import {createCpuEngine} from '../src/cpu-engine.js';

// Dense and large desktop displays must not magnify a smaller scene image.
// Extra retina samples stay bounded; the native viewport takes priority when
// it alone exceeds the supersampling budget.
const viewports=[[850,680],[1366,768],[1920,1080],[2560,1440],[3440,1440],[3840,2160]];
for(const [w,h] of viewports)for(const dpr of [.8,1,1.25,1.5,2,3]){
  const ratio=stagePixelRatio(w,h,dpr),pixels=w*h*ratio*ratio;
  assert(ratio>=1,'a scene pixel is never enlarged beyond one CSS pixel');
  assert(ratio<=Math.max(1,Math.min(dpr,1.6)),'no unnecessary retina samples');
  assert(pixels<=Math.max(w*h,4_000_000)+1e-6,'extra samples fit the stage budget');
}
assert.equal(stagePixelRatio(3840,2160,2),1,'4K is rendered natively instead of upscaled');
assert.equal(stagePixelRatio(850,680,2),1.6,'small retina windows retain antialiasing headroom');
for(const invalid of [0,-1,NaN,Infinity,undefined]){
  assert(Number.isFinite(stagePixelRatio(invalid,1080,2)));
  assert.equal(stagePixelRatio(1920,1080,invalid),1);
}

// Exercise actual WebGL-engine resize calls without needing a GPU context.
// Its lower budget preserves the previous transmission cost at 1080p/HiDPI.
const dprDescriptor=Object.getOwnPropertyDescriptor(globalThis,'devicePixelRatio');
const renderer={shadowMap:{},ratio:1,frames:[],
  setPixelRatio(value){this.ratio=value;},
  setSize(width,height,updateStyle){this.frames.push({width,height,ratio:this.ratio,updateStyle});}};
const engine=await createCpuEngine({scene:new THREE.Scene(),camera:new THREE.PerspectiveCamera(),renderer});
const particles=engine.getParticles(),surface=engine.mesh.geometry.attributes.position.array.slice();
try{
  for(const [w,h] of viewports)for(const dpr of [1,2]){
    Object.defineProperty(globalThis,'devicePixelRatio',{configurable:true,value:dpr});
    engine.resize(w,h);
    const frame=renderer.frames.at(-1),pixels=w*h*frame.ratio*frame.ratio;
    assert.deepEqual([frame.width,frame.height,frame.updateStyle],[w,h,false]);
    assert(frame.ratio>=1,'WebGL resize preserves the whole scene at native CSS resolution');
    assert(pixels<=Math.max(w*h,2_400_000)+1e-6,'WebGL keeps its lower supersampling budget');
    if(w*h<=2_400_000){
      const previous=Math.min(dpr,1.6,Math.sqrt(2_400_000/(w*h)));
      assert.equal(frame.ratio,previous,'small-window rendering cost is unchanged');
    }
  }
  assert.deepEqual(engine.getParticles(),particles,'resolution changes never step the fluid');
  assert.deepEqual(engine.mesh.geometry.attributes.position.array,surface,'resolution changes preserve reconstructed geometry');
}finally{
  engine.destroy();
  if(dprDescriptor)Object.defineProperty(globalThis,'devicePixelRatio',dprDescriptor);
  else delete globalThis.devicePixelRatio;
}
console.log('Render resolution: native desktop silhouettes, bounded retina samples, WebGL resize and unchanged fluid geometry passed.');
