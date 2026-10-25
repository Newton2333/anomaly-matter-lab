import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export function createStage() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#101318');
  scene.fog = new THREE.FogExp2('#101318',.075);
  const camera = new THREE.PerspectiveCamera(36,innerWidth/innerHeight,.025,40);
  camera.position.set(1.8,1.48,3.05);
  const target = new THREE.Vector3(-.20,.64,0);
  camera.lookAt(target);

  scene.add(new THREE.HemisphereLight(0xc3d9e9,0x2f2118,.65));
  const key = new THREE.DirectionalLight(0xfff5e5,3.1); key.position.set(-2,4,3); key.castShadow=true;
  key.shadow.mapSize.set(1024,1024); key.shadow.camera.left=-2;key.shadow.camera.right=2;key.shadow.camera.top=2.5;key.shadow.camera.bottom=-1.5;key.shadow.bias=-.001;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x91cbd8,2.2);rim.position.set(3,1,-2);scene.add(rim);
  const amber = new THREE.PointLight(0xffb45e,5.5,5,2);amber.position.set(-1.1,.55,.5);scene.add(amber);

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:0x12161b,roughness:.7,metalness:.15}));
  floor.rotation.x=-Math.PI/2;floor.position.y=-.17;floor.receiveShadow=true;scene.add(floor);
  const pedestalMaterial = new THREE.MeshStandardMaterial({color:0x252a30,metalness:.82,roughness:.25});
  const profile = [[0,-.13],[.76,-.13],[.84,-.115],[.87,-.085],[.87,-.045],[.84,-.018],[.77,-.008],[0,-.008]].map(([r,y])=>new THREE.Vector2(r,y));
  const plinth = new THREE.Mesh(new THREE.LatheGeometry(profile,128),pedestalMaterial);plinth.receiveShadow=true;plinth.castShadow=true;scene.add(plinth);
  const rimMaterial = new THREE.MeshStandardMaterial({color:0xc7bba6,metalness:1,roughness:.22});
  const trim = new THREE.Mesh(new THREE.TorusGeometry(.854,.004,8,160),rimMaterial);trim.rotation.x=Math.PI/2;trim.position.y=-.047;scene.add(trim);
  const deck = new THREE.Mesh(new THREE.CircleGeometry(.78,128),new THREE.MeshStandardMaterial({color:0x161b20,metalness:.72,roughness:.34}));deck.rotation.x=-Math.PI/2;deck.position.y=-.006;deck.receiveShadow=true;scene.add(deck);
  for(const radius of [.57,.66,.75]) {
    const engraving=new THREE.Mesh(new THREE.TorusGeometry(radius,.0008,4,160),new THREE.MeshBasicMaterial({color:0x41464a,transparent:true,opacity:.45}));
    engraving.rotation.x=Math.PI/2;engraving.position.y=-.004;scene.add(engraving);
  }
  // A floating ring gives the fluid scale and occlusion.
  const apparatus=new THREE.Group();apparatus.position.set(0,.80,-.10);apparatus.rotation.set(.08,.68,.38);scene.add(apparatus);
  const ring=new THREE.Mesh(new THREE.TorusGeometry(.63,.016,16,180),new THREE.MeshStandardMaterial({color:0xc3c7cd,metalness:.95,roughness:.2}));apparatus.add(ring);
  const glowMaterial=new THREE.MeshBasicMaterial({color:0xe1b475,transparent:true,opacity:.82});
  const lightArc=new THREE.Mesh(new THREE.TorusGeometry(.648,.0028,6,140,Math.PI*1.32),glowMaterial);lightArc.rotation.z=.8;apparatus.add(lightArc);
  // Keep the two upper details; leave the lower arc free of any supporting tab.
  for(let i=0;i<2;i++){
    const anchor=new THREE.Mesh(new THREE.BoxGeometry(.065,.028,.035),new THREE.MeshStandardMaterial({color:0x313943,metalness:.85,roughness:.24}));
    const a=i*Math.PI*2/3;anchor.position.set(Math.cos(a)*.63,Math.sin(a)*.63,0);anchor.rotation.z=a;apparatus.add(anchor);
  }
  const shadowCanvas=document.createElement('canvas');shadowCanvas.width=256;shadowCanvas.height=256;
  const ctx=shadowCanvas.getContext('2d'),gradient=ctx.createRadialGradient(128,128,0,128,128,128);gradient.addColorStop(0,'rgba(0,0,0,.7)');gradient.addColorStop(.5,'rgba(0,0,0,.25)');gradient.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=gradient;ctx.fillRect(0,0,256,256);
  const shadowTexture=new THREE.CanvasTexture(shadowCanvas);
  const shadow=new THREE.Mesh(new THREE.PlaneGeometry(.95,.95),new THREE.MeshBasicMaterial({map:shadowTexture,transparent:true,depthWrite:false}));shadow.rotation.x=-Math.PI/2;shadow.position.y=.001;scene.add(shadow);
  let environment=null;
  let orbitAngle=0;
  let cameraOrbit=false;
  let zoom=1;
  let mat='metal';
  const pointer=new THREE.Vector2();
  const desired=new THREE.Vector2();
  const accentColors={metal:new THREE.Color(0xe1b475),glass:new THREE.Color(0x94d7e9),plasma:new THREE.Color(0xff7a25)};

  async function setupEnvironment(renderer,isWebGPU) {
    const room=new RoomEnvironment();
    // Reflection cards live in the environment, so their reflections follow the surface.
    const card=new THREE.Mesh(new THREE.PlaneGeometry(2.0,4.0),new THREE.MeshBasicMaterial({color:0xffcf92,side:THREE.DoubleSide}));card.position.set(-3,1.6,1);card.rotation.y=Math.PI/2;room.add(card);
    let Generator;
    if(isWebGPU)Generator=(await import('three/webgpu')).PMREMGenerator;
    else Generator=THREE.PMREMGenerator;
    const pmrem=new Generator(renderer);
    if(pmrem.compileCubemapShader)pmrem.compileCubemapShader();
    environment=pmrem.fromScene(room,.025,.1,100);
    scene.environment=environment.texture;
    scene.environmentIntensity=1.05;
    room.dispose();pmrem.dispose();
    renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.06;
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    if(renderer.shadowMap){renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;}
  }
  function resize(w,h) {
    camera.aspect=w/h;
    const mobile=w<720;
    camera.fov=mobile?44:36;
    target.set(mobile?0:-.22,mobile?.77:.64,0);
    zoom=mobile?1.05:1;
    camera.updateProjectionMatrix();
  }
  function update(time,{charge=0,material='metal',dt=.016}={}) {
    mat=material;
    pointer.lerp(desired,1-Math.exp(-dt*2));
    if(cameraOrbit)orbitAngle+=dt*.12;
    const angle=.47+orbitAngle+pointer.x*.045;
    const radius=3.5*zoom;
    camera.position.set(Math.sin(angle)*radius,1.42+pointer.y*.07,Math.cos(angle)*radius);
    camera.lookAt(target);
    const targetColor=accentColors[mat]||accentColors.metal;
    glowMaterial.color.lerp(targetColor,dt===0?1:1-Math.exp(-dt*5));
    amber.color.lerp(targetColor,dt===0?1:1-Math.exp(-dt*3));amber.intensity=5.5+charge*14;
    lightArc.scale.setScalar(1+charge*.006);
    glowMaterial.opacity=.65+charge*.35;
    apparatus.rotation.y=.68+Math.sin(time*.15)*.05;
    shadow.material.opacity=.64+charge*.2;
    shadow.scale.setScalar(1-charge*.3);
  }
  return {scene,camera,setupEnvironment,resize,update,setPointer(x,y){desired.set(x,y);},setOrbit(v){cameraOrbit=!!v;},reset(){orbitAngle=0;cameraOrbit=false;desired.set(0,0);},dispose(){environment?.dispose();shadowTexture.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of(Array.isArray(o.material)?o.material:[o.material]))m.dispose();}});}};
}
