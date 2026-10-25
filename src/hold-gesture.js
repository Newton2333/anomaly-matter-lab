// The input that began a hold is the only input allowed to release it.
// Reaching full charge never releases; cancelling focus/capture never bursts.
export function createHoldGesture({duration=1600,now=()=>performance.now()}={}) {
  let owner=null,started=0;
  const value=()=>owner===null?0:Math.max(0,Math.min(1,(now()-started)/duration));
  return {
    get active(){return owner!==null;},
    owns(source){return owner===source;},
    start(source){if(owner!==null)return false;owner=source;started=now();return true;},
    value,
    release(source){if(owner!==source)return null;const strength=value();owner=null;return strength;},
    cancel(){owner=null;},
  };
}
