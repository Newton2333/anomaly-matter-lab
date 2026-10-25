// Keep one asynchronous render in flight. Paused and hidden views sleep until
// an interaction invalidates them, rather than drawing the same frame forever.
export function createFrameLoop({render,shouldAnimate,isVisible=()=>!document.hidden,
  onError=console.error,requestFrame=requestAnimationFrame,cancelFrame=cancelAnimationFrame}) {
  let queued=0,running=false,dirty=false,suspended=false,stopped=false,last=0;
  function schedule(){
    if(!queued&&!running&&!suspended&&!stopped&&isVisible())queued=requestFrame(tick);
  }
  async function tick(now){
    queued=0;
    if(suspended||stopped||!isVisible()){last=0;return;}
    const dt=Math.min(last?(now-last)/1000:1/60,.05);last=now;
    running=true;dirty=false;
    try{await render(now,dt);}
    catch(error){stopped=true;onError(error);}
    finally{
      running=false;
      if(!suspended&&!stopped&&isVisible()&&(dirty||shouldAnimate()))schedule();
      else last=0;
    }
  }
  return {
    request(){dirty=true;schedule();},
    suspend(){suspended=true;if(queued)cancelFrame(queued);queued=0;last=0;},
    resume(){suspended=false;dirty=true;schedule();},
    stop(){stopped=true;if(queued)cancelFrame(queued);queued=0;last=0;},
  };
}
