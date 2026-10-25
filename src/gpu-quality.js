// Controller shared by startup calibration and steady-state GPU observations.
// Simulation resolution is deliberately independent of this rendering budget.
export const MIN_GPU_QUALITY = .45;

export function framePercentile(samples, fraction=.75) {
  const sorted=samples.filter(value=>Number.isFinite(value)&&value>0).sort((a,b)=>a-b);
  return sorted.length?sorted[Math.ceil((sorted.length-1)*fraction)]:0;
}

export function nextGpuProfile(profile, frameMs, {recover=false}={}) {
  const next={...profile};
  if(!Number.isFinite(frameMs)||frameMs<=0)return next;
  if(frameMs>19) {
    // The environment reflection remains; only the additional screen-space
    // scene-reflection march is skipped when it cannot meet the frame budget.
    if(next.reflections)next.reflections=false;
    else next.quality=Math.max(MIN_GPU_QUALITY,
      Math.min(next.quality-.03,next.quality*Math.sqrt(16.7/frameMs)));
  } else if(recover&&frameMs<12) {
    // Recover pixels gently. Scene reflections require substantially more
    // spare time and are restored only after full image resolution is back.
    if(next.quality<1)next.quality=Math.min(1,next.quality+.025);
    else if(frameMs<6&&next.canReflect)next.reflections=true;
  }
  return next;
}

export function gpuProfileChanged(a,b) {
  return a.reflections!==b.reflections||Math.abs(a.quality-b.quality)>.005;
}
