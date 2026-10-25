import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';

// Read the actual API methods/constants without constructing a GPU device.
// The browser pressure test separately exercises the full PBF solver/readback.
const source=fs.readFileSync(new URL('../src/gpu-engine.ts',import.meta.url),'utf8');
const file=ts.createSourceFile('gpu-engine.ts',source,ts.ScriptTarget.Latest,true);
const variables=new Map();
function visit(node){if(ts.isVariableDeclaration(node)&&ts.isIdentifier(node.name))variables.set(node.name.text,node);ts.forEachChild(node,visit);}
visit(file);
const evaluate=expression=>new Function(`return (${expression});`)();
const constants=evaluate(variables.get('GPU_INTERACTION').initializer.getText(file));
const clampSource=variables.get('clamp').initializer.getText(file);
const methods=variables.get('api').initializer.properties.filter(property=>
  ['setCharge','burst','reset','setPointer','clearPointer'].includes(property.name?.getText(file))).map(property=>property.getText(file)).join(',\n');
const snippet=ts.transpileModule(`const clamp=${clampSource}; const api={${methods}};`,
  {compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
const harness=new Function(`
  let charge=0,burstPending=0,paused=false,energy=.8,pointerX=0,pointerY=0,
    resetPending=false,time=4,reportTime=3,dirty=false;
  const timeUniform={value:4},CENTER={},attractor={value:{copy(){}}};
  const applyPhysics=()=>{},resetTimingWindow=()=>{};
  ${snippet}
  return {api,state:()=>({charge,burstPending,paused,energy,pointerX,pointerY,resetPending})};
`)();

// Starting a new hold must cancel an old queued release even before its first
// non-zero progress frame. A release can then queue exactly its explicit impulse.
harness.api.burst(.8);assert.equal(harness.state().burstPending,.8);
harness.api.setCharge(0);assert.equal(harness.state().burstPending,0);
for(const charge of [.01,.25,.5,1,1,1]){
  harness.api.setCharge(charge);assert.equal(harness.state().charge,charge);
  assert.equal(harness.state().burstPending,0);
}
harness.api.setCharge(0);harness.api.burst(.7);
assert.equal(harness.state().charge,0);assert.equal(harness.state().burstPending,.7);
harness.api.setCharge(.2);assert.equal(harness.state().burstPending,0);
harness.api.setPointer(1,-1);assert.equal(harness.state().pointerX,1);
harness.api.setPointer(0,0);assert.equal(harness.state().pointerX,0);
harness.api.setPointer(-.6,.8);harness.api.clearPointer();
assert.equal(harness.state().pointerX,0);assert.equal(harness.state().pointerY,0);
harness.api.reset();assert.equal(harness.state().energy,0);assert.equal(harness.state().charge,0);
assert.equal(harness.state().burstPending,0);assert.equal(harness.state().resetPending,true);

// Check the held-force integrator over all supported frame steps, including an
// outward-moving droplet and a fast inward crossing. This tests damping and the
// bounded restoring field; it is not a substitute for the actual GPU PBF test.
const summaries=[];
for(const dt of [1/240,1/120,1/60,1/30])for(const baseDrag of [.09,.14,.30]){
  const drag=baseDrag+constants.holdDamping;
  assert(drag>=2*Math.sqrt(constants.holdStiffness));
  for(const [initialPosition,initialVelocity] of [[.6,0],[.55,1.8],[.21,-2]]){
    let position=initialPosition,velocity=initialVelocity,maxDistance=Math.abs(position);
    const damp=Math.exp(-dt*drag),forceDt=(1-damp)/drag;
    for(let step=0;step<Math.ceil(8/dt);step++){
      const distance=Math.abs(position);
      const acceleration=-Math.sign(position)*Math.min(constants.holdMaxAcceleration,
        Math.max(0,distance-constants.holdCore)*constants.holdStiffness);
      assert(Math.abs(acceleration)<=constants.holdMaxAcceleration);
      velocity=velocity*damp+acceleration*forceDt;position+=velocity*dt;
      assert(Number.isFinite(position)&&Number.isFinite(velocity));
      maxDistance=Math.max(maxDistance,Math.abs(position));
    }
    assert(Math.abs(velocity)<.002);
    assert(Math.abs(position)<=constants.holdCore+.002);
    assert(maxDistance<.8);
    summaries.push({dt,baseDrag,maxDistance});
  }
}
assert(Math.abs(constants.pointerX/.24-1.5)<1e-12);assert(Math.abs(constants.pointerY/.20-1.5)<1e-12);
// The regular drop still fits inside the real ±.9 m collision box at full pull.
assert(constants.pointerX+.33<.9);assert(.85-constants.pointerY-.30>0);
assert(.85+constants.pointerY+.30<1.85);
console.log(`GPU interaction: charge/release/reset/pointer contracts and ${summaries.length} bounded held-force trajectories passed.`);
