import assert from 'node:assert/strict';
import * as THREE from 'three';
import {CpuFluidSolver,createCpuEngine} from '../src/cpu-engine.js';

function measure(fluid){
  const p=fluid.positions,v=fluid.velocities,centroid=[0,0,0];
  for(let k=0;k<p.length;k+=3)for(let axis=0;axis<3;axis++)centroid[axis]+=p[k+axis]/fluid.count;
  let variance=0,maxSpeed=0,maxRadius=0;
  for(let k=0;k<p.length;k+=3){
    const radius=Math.hypot(p[k]-centroid[0],p[k+1]-centroid[1],p[k+2]-centroid[2]);
    const speed=Math.hypot(v[k],v[k+1],v[k+2]);
    assert.ok(Number.isFinite(radius)&&Number.isFinite(speed),'finite particles and velocities');
    assert.ok(Math.abs(p[k])<=.9&&p[k+1]>=0&&p[k+1]<=1.85&&Math.abs(p[k+2])<=.9,'container bounds');
    variance+=radius*radius;maxSpeed=Math.max(maxSpeed,speed);maxRadius=Math.max(maxRadius,radius);
  }
  return {centroid,radius:Math.sqrt(variance/fluid.count),maxSpeed,maxRadius};
}
function advance(fluid,duration,dt=1/60,beforeStep=()=>{}){
  const end=fluid.time+duration;
  while(fluid.time<end-1e-9){beforeStep(fluid);fluid.step(dt);}
}

const results=[];
for(const spacing of [.064,.09])for(const energy of [0,1])for(const dt of [1/60,1/15]){
  const fluid=new CpuFluidSolver({spacing});
  assert.equal(fluid.energy,0,'default energy is zero');
  assert.equal(fluid.count,spacing===.064?485:179,'particle quality is unchanged');
  fluid.energy=energy;
  // One pointer event must remain effective throughout a stationary long hold.
  const target=[.30,.99,-.08];fluid.setPointer(target);
  target[0]=100;assert.equal(fluid.pointer[0],.30,'copy caller values');
  let minRadius=Infinity,maxRadius=0,maxSpeed=0;
  advance(fluid,15,dt,()=>{
    fluid.charge=Math.min(1,fluid.time/1.6);
    if(fluid.time>3){
      const m=measure(fluid);minRadius=Math.min(minRadius,m.radius);
      maxRadius=Math.max(maxRadius,m.radius);maxSpeed=Math.max(maxSpeed,m.maxSpeed);
    }
  });
  const held=measure(fluid);
  assert.ok(maxRadius<.27,'long hold stays compact');
  assert.ok(maxRadius-minRadius<.025,'no spontaneous expansion during hold');
  assert.ok(maxSpeed<1,'no spontaneous hold impulse');
  assert.ok(Math.hypot(held.centroid[0]-.30,held.centroid[1]-.99,held.centroid[2]+.08)<.02,'whole fluid follows the persistent center');
  const heldPositions=fluid.positions.slice();advance(fluid,.2,dt);
  assert.notDeepEqual(fluid.positions,heldPositions,'held fluid remains simulated');
  fluid.burst(1);advance(fluid,.2,dt);const released=measure(fluid);
  assert.ok(released.radius>held.radius*2.5,'explicit release remains visibly expansive');
  results.push({particles:fluid.count,energy,inputHz:Math.round(1/dt),holdMin:+minRadius.toFixed(4),
    holdMax:+maxRadius.toFixed(4),peakHoldSpeed:+maxSpeed.toFixed(4),releaseRatio:+(released.radius/held.radius).toFixed(2)});
  fluid.reset();assert.equal(fluid.energy,0);assert.equal(fluid.charge,0);
  assert.deepEqual(fluid.center,[0,.89,0]);assert.deepEqual(fluid.pointer,[0,.89,0]);
}

// Position constraint corrections must never be divided by microscopic dt.
const tiny=new CpuFluidSolver();tiny.energy=1;tiny.charge=1;advance(tiny,5,.04);
const settledTime=tiny.time,settledPositions=tiny.positions.slice(),settledVelocities=tiny.velocities.slice();
tiny.step(1e-6);tiny.step(1e-6);
assert.equal(tiny.time,settledTime);assert.deepEqual(tiny.positions,settledPositions);assert.deepEqual(tiny.velocities,settledVelocities);
let peakTinySpeed=0;
advance(tiny,10,.04,()=>{tiny.step(1e-6);peakTinySpeed=Math.max(peakTinySpeed,measure(tiny).maxSpeed);});
assert.ok(peakTinySpeed<1,'tiny frame deltas do not turn corrections into an explosion');
const split=new CpuFluidSolver(),whole=new CpuFluidSolver();
split.step(.002);split.step(.002);split.step(.001);whole.step(.005);
assert.deepEqual(split.positions,whole.positions,'small elapsed times accumulate without speeding up simulation');
assert.deepEqual(split.velocities,whole.velocities);

// Moving and cleared targets move the field continuously; neither operation
// teleports particles or changes the radial origin of a release impulse.
const moving=new CpuFluidSolver();moving.energy=1;moving.charge=1;
advance(moving,6,1/60,()=>moving.setPointer([Math.sin(moving.time*1.2)*.36,.89+Math.cos(moving.time*1.2)*.30,0]));
assert.ok(measure(moving).radius<.27);
moving.setPointer([.36,.99,0]);advance(moving,3);
assert.ok(measure(moving).centroid[0]>.32,'translation is clearly visible');
const beforeClear=moving.positions.slice(),oldCenter=moving.center.slice();
moving.clearPointer();assert.deepEqual(moving.positions,beforeClear);assert.deepEqual(moving.center,oldCenter);
moving.burst(1);advance(moving,.2);assert.ok(measure(moving).radius>.5);
advance(moving,4);assert.ok(Math.hypot(moving.center[0],moving.center[1]-.89,moving.center[2])<.001);

// Exercise the real engine mapping without creating a browser graphics context.
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(36,16/9,.025,40);
camera.position.set(1.8,1.48,3.05);camera.lookAt(-.2,.64,0);
const originalPointer=CpuFluidSolver.prototype.setPointer;
let mapped,lastStats;
CpuFluidSolver.prototype.setPointer=function(target){mapped=target.slice();return originalPointer.call(this,target);};
const engine=await createCpuEngine({scene,camera,renderer:{isSoftwareRenderer:true,setSize(){}},onStats:stats=>lastStats=stats});
try{
  engine.setPointer(1,0);
  const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0);
  const displacement=new THREE.Vector3(...mapped).sub(new THREE.Vector3(0,.89,0));
  assert.ok(Math.abs(displacement.dot(right)-.36)<1e-9,'screen-right traction has the intended bounded travel');
  engine.setPointer(0,0);assert.deepEqual(mapped,[0,.89,0],'screen center is a valid pointer sample');
  camera.position.set(-3,1.48,1.8);camera.lookAt(-.2,.64,0);engine.setPointer(1,0);
  right.setFromMatrixColumn(camera.matrixWorld,0);
  assert.ok(new THREE.Vector3(...mapped).sub(new THREE.Vector3(0,.89,0)).dot(right)>.35,'mapping follows camera orbit');
  engine.setPaused(true);engine.setEnergy(1);engine.step(1);assert.equal(lastStats.energy,1);
  engine.reset();engine.step(1);assert.equal(lastStats.energy,0,'engine reset energy is zero');
  assert.equal(engine.mesh.resolution,48,'surface resolution is unchanged');
}finally{CpuFluidSolver.prototype.setPointer=originalPointer;engine.destroy();}

console.table(results);
console.log(`CPU interaction: 15 s holds, low frame rate, energy 0/1, persistent/moving center, explicit release, tiny dt and reset passed (tiny-dt peak speed ${peakTinySpeed.toFixed(3)}).`);
