// Exercise the actual materials and first burst while the loading cover is up.
// There is no artificial wait: each increment represents completed render work.
export async function warmupExperience(engine,{scene=engine.scene,camera=engine.camera,beforeRender,onProgress=()=>{},energy=0,
  yieldFrame=()=>new Promise(resolve=>setTimeout(resolve,0))}={}) {
  if(engine.warmup){await engine.warmup({beforeRender,onProgress});return;}
  const modes=['metal','glass','plasma'];
  const passes=3;
  engine.setPaused(false);
  for(let index=0;index<modes.length;index++){
    const mode=modes[index];
    engine.reset();engine.setEnergy(energy);engine.setMaterial(mode);
    for(let pass=0;pass<passes;pass++){
      if(pass===1)engine.burst(.25);
      await engine.step(1/60);
      beforeRender?.(mode);
      if(pass===0&&engine.renderer.compileAsync)
        await engine.renderer.compileAsync(scene,camera);
      if(engine.renderer.renderAsync)await engine.renderer.renderAsync(scene,camera);
      else engine.renderer.render(scene,camera);
      engine.afterRender?.();
      await engine.waitForIdle?.();
      onProgress(mode,(index*passes+pass+1)/(modes.length*passes));
      await yieldFrame();
    }
  }
  engine.reset();engine.setEnergy(energy);
}
