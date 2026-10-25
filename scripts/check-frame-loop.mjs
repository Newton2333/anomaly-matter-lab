import assert from 'node:assert/strict';
import {createFrameLoop} from '../src/frame-loop.js';

function harness(render){
  const queue=new Map(),frames=[],errors=[];
  let id=0,animated=false,visible=true;
  const loop=createFrameLoop({render:async(now,dt)=>{frames.push({now,dt});await render?.(now,dt);},
    shouldAnimate:()=>animated,isVisible:()=>visible,onError:error=>errors.push(error),
    requestFrame:fn=>{queue.set(++id,fn);return id;},cancelFrame:key=>queue.delete(key)});
  return {loop,queue,frames,errors,animate:value=>animated=value,visible:value=>visible=value,
    async tick(now){const next=queue.entries().next().value;assert.ok(next,'expected a scheduled frame');queue.delete(next[0]);await next[1](now);}};
}

// Paused scenes render exactly once per invalidation; repeated requests coalesce.
const idle=harness();idle.loop.request();idle.loop.request();assert.equal(idle.queue.size,1);
await idle.tick(100);assert.equal(idle.queue.size,0);assert.equal(idle.frames.length,1);
idle.loop.request();await idle.tick(10000);assert.equal(idle.frames[1].dt,1/60);

// Active scenes continue, but a hidden/background transition cancels their RAF.
const active=harness();active.animate(true);active.loop.request();await active.tick(100);
assert.equal(active.queue.size,1);await active.tick(116);assert.equal(active.frames[1].dt,.016);
active.visible(false);active.loop.suspend();assert.equal(active.queue.size,0);
active.loop.request();assert.equal(active.queue.size,0);
active.visible(true);active.loop.resume();await active.tick(30000);
assert.equal(active.frames.at(-1).dt,1/60);active.animate(false);await active.tick(30016);
assert.equal(active.queue.size,0);

// Requests and suspend/resume during an async GPU step never overlap renders.
let release;const blocked=harness(()=>new Promise(resolve=>release=resolve));
blocked.loop.request();const pending=blocked.tick(100);blocked.loop.request();
assert.equal(blocked.queue.size,0);blocked.loop.suspend();blocked.loop.resume();
assert.equal(blocked.queue.size,0);release();await pending;assert.equal(blocked.queue.size,1);
blocked.loop.stop();assert.equal(blocked.queue.size,0);blocked.loop.request();assert.equal(blocked.queue.size,0);

// A render failure is reported once and stops automatic retries.
const failed=harness(()=>{throw new Error('device lost');});failed.animate(true);
failed.loop.request();await failed.tick(100);assert.equal(failed.errors.length,1);
assert.equal(failed.queue.size,0);failed.loop.request();assert.equal(failed.queue.size,0);
console.log('Frame scheduling: idle, active, background, async overlap and error checks passed.');
