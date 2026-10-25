import './style.css';
import { createStage } from './stage.js';
import { translate, initialLanguage } from './i18n.js';
import { createFrameLoop } from './frame-loop.js';
import { warmupExperience } from './warmup.js';
import { createHoldGesture } from './hold-gesture.js';

const $=id=>document.getElementById(id);
const root=$('experience');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
let canvas=$('scene'),stage,engine,material='metal',energy=0,paused=reduced,orbit=false,immersive=false;
let charge=0,pointerId=null,ready=false,time=0;
const hold=createHoldGesture();
let toastTimer,toastKey=null,toastValues={},language=initialLanguage();
let loadingKey='loading';
const t=(key,values)=>translate(language,key,values);
const sampleIds={metal:'01',glass:'02',plasma:'03'};
const frameLoop=createFrameLoop({render:frame,shouldAnimate:()=>ready&&!paused,
  onError:error=>{console.error('ANOMALY render',error);ready=false;toast('renderError');}});
function requestRender(){if(ready)frameLoop.request();}
document.body.classList.add('loading');
root.setAttribute('aria-busy','true');
function loadingProgress(mode,fraction=0){
  loadingKey=({metal:'loadingMetal',glass:'loadingGlass',plasma:'loadingPlasma',ready:'loadingReady'})[mode]||'loading';
  $('loader-message').textContent=t(loadingKey);
  const progress=Math.round(Math.max(0,Math.min(1,fraction))*100);
  $('loader-progress').value=progress;$('loader-percent').textContent=`${progress}%`;
}

function toast(key,values={}){clearTimeout(toastTimer);toastKey=key;toastValues=values;$('toast').textContent=t(key,values);$('toast').classList.add('visible');toastTimer=setTimeout(()=>{$('toast').classList.remove('visible');toastKey=null;},2800);}
function replaceCanvas(){const fresh=canvas.cloneNode(false);canvas.replaceWith(fresh);canvas=fresh;}
function applyState(){engine.setMaterial(material);engine.setEnergy(energy);engine.setPaused(paused);stage.setOrbit(orbit);}
function updateLabels(){root.dataset.material=material;$('sample-id').textContent=sampleIds[material];$('sample-name').textContent=t(material+'Name');$('headline').textContent=t(material+'Title');$('description').textContent=t(material+'Description');document.querySelectorAll('.materials button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.material===material)));}
function updatePauseLabel(){$('pause').firstElementChild.textContent=t(paused?'resume':'pause');$('pause').querySelector('.setting-state').textContent=paused?'▶':'Ⅱ';}
function updateRendererNote(){const mode=root.dataset.renderer;$('renderer-note').textContent=mode?t(mode==='webgpu'?'rendererGpu':mode==='webgl'?'rendererGl':'rendererSoftware'):'';}
function applyLanguage(){
  document.documentElement.lang=language==='en'?'en':'zh-CN';document.title=t('pageTitle');document.querySelector('meta[name="description"]').content=t('metaDescription');
  document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));
  document.querySelectorAll('[data-i18n-aria]').forEach(el=>el.setAttribute('aria-label',t(el.dataset.i18nAria)));
  $('language').textContent=language==='zh'?'EN':'中';$('language').setAttribute('aria-label',t('languageLabel'));$('language').title=t('languageLabel');$('language').lang=language==='zh'?'en':'zh-CN';
  $('orbit').querySelector('.setting-state').textContent=t(orbit?'on':'off');
  updateLabels();updatePauseLabel();updateRendererNote();resetPrompt();if(hold.active)updateChargePrompt();if(toastKey)$('toast').textContent=t(toastKey,toastValues);
  if(!ready)$('loader-message').textContent=t(loadingKey);
}
$('language').addEventListener('click',()=>{language=language==='zh'?'en':'zh';applyLanguage();});

async function initialize(){
  const errors=[];
  for(const mode of ['webgpu','webgl','software']){
    loadingProgress('loading');
    stage=createStage();
    try{
      if(mode==='webgpu'){
        const {createGpuEngine}=await import('./gpu-engine.ts');
        engine=await createGpuEngine({canvas,scene:stage.scene,camera:stage.camera,onError:error=>{toast('deviceLost');console.warn(error);}});
      }else if(mode==='webgl'){
        const {createCpuEngine}=await import('./cpu-engine.js');
        engine=await createCpuEngine({canvas,scene:stage.scene,camera:stage.camera});
      }else{
        const [{createCpuEngine},{SoftwareRenderer}]=await Promise.all([import('./cpu-engine.js'),import('./software-renderer.js')]);
        const renderer=new SoftwareRenderer({canvas});
        engine=await createCpuEngine({canvas,scene:stage.scene,camera:stage.camera,renderer});
      }
      resize();
      if(mode!=='software')await stage.setupEnvironment(engine.renderer,mode==='webgpu');
      applyState();
      await warmupExperience(engine,{energy,scene:stage.scene,camera:stage.camera,
        beforeRender:mode=>stage.update(0,{material:mode,charge:0,dt:0}),
        onProgress:(mode,fraction)=>loadingProgress(mode,fraction*.95)});
      loadingProgress('ready',.95);
      engine.reset();applyState();
      await engine.step(0);
      stage.update(0,{material,charge:0,dt:0});
      if(mode==='webgpu'&&engine.renderer.renderAsync)await engine.renderer.renderAsync(stage.scene,stage.camera);
      else engine.renderer.render(stage.scene,stage.camera);
      engine.afterRender?.();
      await engine.waitForIdle?.();
      loadingProgress('ready',1);
      root.dataset.renderer=mode;
      updateRendererNote();
      ready=true;bindCanvas();
      root.setAttribute('aria-busy','false');
      $('loader').classList.add('done');document.body.classList.remove('loading');document.body.classList.add('ready');
      setTimeout(()=>$('loader').hidden=true,700);
      updatePauseLabel();
      requestRender();
      return;
    }catch(error){
      errors.push({mode,message:String(error?.message||error)});
      console.info(`ANOMALY ${mode} unavailable:`,error?.message||error);
      try{engine?.destroy();}catch{}engine=null;
      try{stage.dispose();}catch{}
      replaceCanvas();
    }
  }
  $('loader').hidden=true;root.setAttribute('aria-busy','false');$('fatal').hidden=false;console.error('ANOMALY initialization failed',errors);
}
function resize(){const r=$('canvas-mount').getBoundingClientRect();stage?.resize(r.width,r.height);engine?.resize(r.width,r.height);requestRender();}
window.addEventListener('resize',resize);
async function frame(now,dt){
  if(!ready)return;
  if(!paused)time+=dt;
  if(hold.active){charge=hold.value();engine.setCharge(charge);$('charge-number').textContent=Math.round(charge*100);$('charge-cursor').querySelector('circle').style.strokeDashoffset=220*(1-charge);updateChargePrompt();}
  await engine.step(dt);
  stage.update(time,{charge,material,dt:paused?0:dt});
  engine.renderer.render(stage.scene,stage.camera);
  engine.afterRender?.();
}
function selectMaterial(next){if(!ready)return;cancelGesture();material=next;engine.setMaterial(next);updateLabels();requestRender();}
document.querySelectorAll('.materials button').forEach(b=>b.addEventListener('click',()=>selectMaterial(b.dataset.material)));

function resetPrompt(){
  $('gesture-move').textContent=t(matchMedia('(pointer:coarse)').matches?'swipe':'move');
  const hold=document.createElement('span'),arrow=document.createElement('span'),release=document.createElement('span');
  hold.textContent=t('hold');release.textContent=t('release');arrow.textContent='→';arrow.className='gesture-arrow';arrow.setAttribute('aria-hidden','true');
  $('gesture-copy').replaceChildren(hold,arrow,release);$('gesture').classList.remove('is-charging');
}
let chargePromptValue=-1;
function updateChargePrompt(){const value=Math.round(charge*100);if(value!==chargePromptValue||!$('gesture').classList.contains('is-charging')){$('gesture-copy').textContent=charge>=1?t('releaseNow'):t('charging',{value});chargePromptValue=value;}$('gesture').classList.add('is-charging');}
function startCharge(source){if(!ready||!hold.start(source))return;engine.setMaterial(material);engine.setCharge(0);if(paused){paused=false;engine.setPaused(false);updatePauseLabel();}charge=0;$('charge-cursor').classList.add('active');requestRender();}
function endCharge(release,source){if(!hold.active)return;const strength=release?hold.release(source):null;if(release&&strength===null)return;if(!release)hold.cancel();engine?.setCharge(0);if(release)engine.burst(Math.max(.18,strength));$('charge-cursor').classList.remove('active');charge=0;resetPrompt();}
function cancelGesture(){endCharge(false);if(pointerId!==null&&canvas.hasPointerCapture(pointerId))canvas.releasePointerCapture(pointerId);pointerId=null;engine?.clearPointer();stage?.setPointer(0,0);}
function pointer(event){const r=canvas.getBoundingClientRect();const x=(event.clientX-r.left)/r.width*2-1,y=1-(event.clientY-r.top)/r.height*2;engine?.setPointer(x,y);stage?.setPointer(x,y);$('charge-cursor').style.left=`${event.clientX-r.left}px`;$('charge-cursor').style.top=`${event.clientY-r.top}px`;}
function bindCanvas(){
  canvas.addEventListener('pointermove',pointer);
  canvas.addEventListener('pointerdown',e=>{if(e.button!==0||pointerId!==null||hold.active)return;pointerId=e.pointerId;canvas.setPointerCapture(pointerId);pointer(e);closeSettings();startCharge(`pointer:${pointerId}`);});
  canvas.addEventListener('pointerup',e=>{if(e.button!==0||e.pointerId!==pointerId)return;endCharge(true,`pointer:${pointerId}`);if(canvas.hasPointerCapture(pointerId))canvas.releasePointerCapture(pointerId);pointerId=null;});
  canvas.addEventListener('pointercancel',e=>{if(e.pointerId===pointerId)cancelGesture();});canvas.addEventListener('lostpointercapture',e=>{if(e.pointerId===pointerId&&hold.owns(`pointer:${pointerId}`))cancelGesture();});
  canvas.addEventListener('pointerleave',()=>{if(!hold.active){engine.clearPointer();stage.setPointer(0,0);}});
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
}
window.addEventListener('blur',cancelGesture);document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelGesture();frameLoop.suspend();}else frameLoop.resume();});
$('energy').addEventListener('input',e=>{energy=+e.target.value/100;engine?.setEnergy(energy);$('energy-value').textContent=`${e.target.value}%`;e.target.style.backgroundImage=`linear-gradient(90deg,var(--accent) ${e.target.value}%,#404850 ${e.target.value}%)`;});
function togglePause(force){cancelGesture();paused=typeof force==='boolean'?force:!paused;engine?.setPaused(paused);updatePauseLabel();requestRender();}
$('pause').addEventListener('click',()=>togglePause());
$('orbit').addEventListener('click',()=>{orbit=!orbit;stage?.setOrbit(orbit);if(paused&&orbit)togglePause(false);$('orbit').querySelector('.setting-state').textContent=t(orbit?'on':'off');});
function closeSettings(){$('settings').hidden=true;$('menu').setAttribute('aria-expanded','false');}
$('menu').addEventListener('click',()=>{const show=$('settings').hidden;$('settings').hidden=!show;$('menu').setAttribute('aria-expanded',String(show));});$('close-settings').addEventListener('click',closeSettings);
function setImmersive(value){immersive=value;closeSettings();root.classList.toggle('focus-mode',immersive);$('exit-immersive').hidden=!immersive;document.querySelectorAll('.chrome').forEach(el=>el.inert=immersive);if(immersive)canvas.focus({preventScroll:true});else $('immersive').focus({preventScroll:true});}
$('immersive').addEventListener('click',()=>setImmersive(true));$('exit-immersive').addEventListener('click',()=>setImmersive(false));
$('fullscreen').addEventListener('click',async()=>{closeSettings();try{if(document.fullscreenElement)await document.exitFullscreen();else if(root.requestFullscreen)await root.requestFullscreen();else setImmersive(true);}catch{setImmersive(true);}});
function reset(){if(!ready)return;cancelGesture();engine.reset();stage.reset();orbit=false;energy=0;paused=reduced;material='metal';applyState();updateLabels();$('energy').value=0;$('energy').style.backgroundImage='linear-gradient(90deg,var(--accent) 0%,#404850 0%)';$('energy-value').textContent='0%';$('orbit').querySelector('.setting-state').textContent=t('off');updatePauseLabel();requestRender();toast('resetDone');}
$('reset').addEventListener('click',reset);
async function capture(){if(!ready)return;const snapshotMaterial=material;$('capture').disabled=true;closeSettings();try{if(engine.kind==='webgpu'&&engine.renderer.renderAsync)await engine.renderer.renderAsync(stage.scene,stage.camera);else engine.renderer.render(stage.scene,stage.camera);engine.afterRender?.();const shot=document.createElement('canvas');shot.width=canvas.width;shot.height=canvas.height;const ctx=shot.getContext('2d');ctx.drawImage(canvas,0,0);const size=Math.max(16,shot.width*.021);ctx.fillStyle='#e9e5dc';ctx.font=`${size}px Arial`;ctx.fillText('ANOMALY',shot.width*.05,shot.height*.08);const blob=await new Promise(r=>shot.toBlob(r,'image/png'));if(!blob)throw Error('Snapshot unavailable');const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`ANOMALY-${snapshotMaterial}-${Date.now()}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);toast('captured');}catch{toast('captureFailed');}finally{$('capture').disabled=false;}}
$('capture').addEventListener('click',capture);
$('about-open').addEventListener('click',()=>{closeSettings();$('about').showModal();});$('about-close').addEventListener('click',()=>$('about').close());$('reload').addEventListener('click',()=>location.reload());
document.addEventListener('keydown',e=>{if(e.repeat||e.ctrlKey||e.metaKey||e.altKey||$('about').open)return;if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;const key=e.key.toLowerCase();if(e.code==='Space'&&!['BUTTON','A'].includes(e.target.tagName)){e.preventDefault();$('charge-cursor').style.left='58%';$('charge-cursor').style.top='48%';startCharge('keyboard');}else if(key==='1')selectMaterial('metal');else if(key==='2')selectMaterial('glass');else if(key==='3')selectMaterial('plasma');else if(key==='h')setImmersive(!immersive);else if(key==='escape'){if(immersive)setImmersive(false);closeSettings();}else if(key==='r')reset();else if(key==='p')togglePause();});
document.addEventListener('keyup',e=>{if(e.code==='Space'&&hold.owns('keyboard')){e.preventDefault();endCharge(true,'keyboard');}});
window.addEventListener('pagehide',()=>{cancelGesture();frameLoop.suspend();});
window.addEventListener('pageshow',()=>frameLoop.resume());
applyLanguage();initialize();
