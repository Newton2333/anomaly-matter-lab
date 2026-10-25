import './style.css';
import { isHandheld } from './device.js';
import { translate } from './i18n.js';

export function launchExperience(device) {
const handheld=isHandheld(device);

if(handheld){
  // Stop before importing Three.js, the GPU solver, or any rendering backend.
  document.getElementById('experience').hidden=true;
  document.getElementById('about').hidden=true;
  document.getElementById('desktop-notice').hidden=false;
  document.body.classList.add('desktop-only');
  let language='en';
  const button=document.getElementById('mobile-language'),link=document.getElementById('desktop-link');
  link.value=location.origin+'/';
  function apply(){
    document.documentElement.lang=language==='en'?'en':'zh-CN';
    document.title=translate(language,'pageTitle');
    document.querySelectorAll('[data-mobile-i18n]').forEach(el=>el.textContent=translate(language,el.dataset.mobileI18n));
    button.textContent=language==='en'?'中':'EN';button.setAttribute('aria-label',translate(language,'languageLabel'));
    link.setAttribute('aria-label',translate(language,'desktopLink'));
    document.getElementById('desktop-notice').setAttribute('aria-label',translate(language,'desktopTitle'));
    document.getElementById('mobile-status').textContent='';
  }
  button.addEventListener('click',()=>{language=language==='en'?'zh':'en';apply();});
  document.getElementById('copy-desktop-link').addEventListener('click',async()=>{
    let key='linkCopied';
    try{await navigator.clipboard.writeText(link.value);}catch{key='copyManually';link.focus();link.select();}
    document.getElementById('mobile-status').textContent=translate(language,key);
  });
  apply();
}else{
  import('./main.js').catch(error=>{
    document.getElementById('loader').hidden=true;
    document.getElementById('fatal').hidden=false;
    document.getElementById('reload').onclick=()=>location.reload();
    console.error('ANOMALY could not start',error);
  });
}

}
