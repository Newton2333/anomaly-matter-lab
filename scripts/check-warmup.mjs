import assert from 'node:assert/strict';
import {warmupExperience} from '../src/warmup.js';

const scene={},camera={},calls=[],progress=[];let mode='metal',fenced=true;
const engine={
  renderer:{async compileAsync(s,c){assert.equal(s,scene);assert.equal(c,camera);calls.push(['compile',mode]);},
    render(s,c){assert.equal(s,scene);assert.equal(c,camera);calls.push(['render',mode]);fenced=false;}},
  setPaused(value){calls.push(['pause',value]);},reset(){calls.push(['reset']);mode='metal';},
  setEnergy(value){assert.equal(value,.42);},setMaterial(value){mode=value;},
  burst(){calls.push(['burst',mode]);},async step(dt){assert.ok(fenced);assert.equal(dt,1/60);calls.push(['step',mode]);},
  afterRender(){calls.push(['submit',mode]);},async waitForIdle(){fenced=true;calls.push(['fence',mode]);}
};
await warmupExperience(engine,{scene,camera,beforeRender:value=>assert.equal(value,mode),
  onProgress:(value,fraction)=>{assert.ok(fenced);progress.push({value,fraction});},yieldFrame:async()=>{}});
assert.deepEqual(calls.filter(c=>c[0]==='compile').map(c=>c[1]),['metal','glass','plasma']);
assert.deepEqual(calls.filter(c=>c[0]==='burst').map(c=>c[1]),['metal','glass','plasma']);
assert.equal(calls.filter(c=>c[0]==='render').length,9);
assert.equal(calls.filter(c=>c[0]==='reset').length,4);
assert.equal(mode,'metal');assert.equal(progress.at(-1).fraction,1);
assert.ok(progress.every((value,i)=>!i||value.fraction>progress[i-1].fraction));
let delegated=0;
await warmupExperience({async warmup({beforeRender,onProgress}){delegated++;beforeRender('glass');onProgress('glass',1);}},
  {beforeRender:value=>assert.equal(value,'glass'),onProgress:(value,fraction)=>assert.equal(fraction,1)});
assert.equal(delegated,1);
await assert.rejects(()=>warmupExperience({warmup:async()=>{throw new Error('lost');}}),/lost/);
console.log('Warmup: material compilation, burst coverage, GPU fences, progress and delegation checks passed.');
