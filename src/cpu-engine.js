import * as THREE from 'three';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import { stagePixelRatio } from './render-resolution.js';

const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

/** A compact position-based fluid solver. Positions are integrated, never prescribed. */
export class CpuFluidSolver {
  constructor({ spacing=.064 }={}) {
    this.spacing=spacing; this.h=spacing*2; this.h2=this.h*this.h;
    this.restDensity=5.9; this.center=[0,.89,0]; this.energy=.42; this.charge=0;
    this.pointer=[0,.89,0]; this.pendingTime=0;
    const seedPositions=[]; const n=Math.ceil(.318/spacing);
    let seed=1234567; const rnd=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296;};
    for(let x=-n;x<=n;x++)for(let y=-n;y<=n;y++)for(let z=-n;z<=n;z++){
      if((x*x+y*y+z*z)*spacing*spacing>.318*.318)continue;
      seedPositions.push(x*spacing+(rnd()-.5)*.003,.89+y*spacing+(rnd()-.5)*.003,z*spacing+(rnd()-.5)*.003);
    }
    this.initial=new Float32Array(seedPositions); this.count=seedPositions.length/3;
    this.positions=new Float32Array(this.initial);this.velocities=new Float32Array(this.initial.length);
    this.previous=new Float32Array(this.initial.length);this.predicted=new Float32Array(this.initial.length);
    this.corrections=new Float32Array(this.initial.length);this.lambdas=new Float32Array(this.count);
    this.densities=new Float32Array(this.count);this.next=new Int32Array(this.count);
    this.gridSize=Math.ceil(1.9/this.h)+4;this.gridPlane=this.gridSize*this.gridSize;
    this.heads=new Int32Array(this.gridSize**3);this.time=0;
    this.cells=new Int32Array(this.count);this.gradientSums=new Float32Array(this.initial.length);this.gradientNorms=new Float32Array(this.count);this.neighborWeights=new Float32Array(this.count);
    this.neighborOffsets=new Int32Array(27);let stencil=0;
    for(let z=-1;z<=1;z++)for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++)this.neighborOffsets[stencil++]=x+y*this.gridSize+z*this.gridPlane;
    this.pairCapacity=0;this.reservePairs(this.count*48);
    this.reset();
  }
  reset(){
    this.positions.set(this.initial);this.velocities.fill(0);this.energy=.42;this.charge=0;this.time=0;this.pendingTime=0;
    this.center[0]=0;this.center[1]=.89;this.center[2]=0;this.clearPointer();
    for(let i=0;i<this.count;i++){const k=i*3;this.velocities[k]=this.positions[k+2]*.45;this.velocities[k+2]=-this.positions[k]*.45;}
    this.predicted.set(this.positions);
  }
  gridCoordinate(v,axis){return clamp(Math.floor((v+(axis===1?.03:.95))/this.h)+1,1,this.gridSize-2);}
  reservePairs(capacity){
    const oldCapacity=this.pairCapacity;this.pairCapacity=capacity;
    for(const name of ['pairA','pairB']){const next=new Uint16Array(capacity);if(oldCapacity)next.set(this[name]);this[name]=next;}
    for(const name of ['pairGX','pairGY','pairGZ','pairWeight','pairAntiClump']){const next=new Float32Array(capacity);if(oldCapacity)next.set(this[name]);this[name]=next;}
  }
  buildPairs(invDensity){
    const p=this.predicted,g=this.gridSize,n=this.count,h2=this.h2,heads=this.heads,next=this.next,cells=this.cells;
    heads.fill(-1);
    for(let i=0;i<n;i++){const k=i*3,x=this.gridCoordinate(p[k],0),y=this.gridCoordinate(p[k+1],1),z=this.gridCoordinate(p[k+2],2),id=x+y*g+z*this.gridPlane;cells[i]=id;next[i]=heads[id];heads[id]=i;}
    let count=0;const coefficient=-6/h2*invDensity;
    // Each undirected neighbor pair is located once. Both particles then share
    // its kernel/gradient instead of scanning 27 cells separately for each pass.
    for(let i=0;i<n;i++){
      const k=i*3,x=p[k],y=p[k+1],z=p[k+2],cell=cells[i];
      for(let o=0;o<27;o++){
        let j=heads[cell+this.neighborOffsets[o]];
        // Each cell list is descending by particle index. Once j <= i,
        // all remaining entries are self/duplicate pairs. Stop walking here;
        // the accepted pairs and their accumulation order stay identical.
        while(j>i){
          const q=j*3,rx=x-p[q],ry=y-p[q+1],rz=z-p[q+2],r2=rx*rx+ry*ry+rz*rz;
          if(r2<h2&&r2>1e-10){
            if(count===this.pairCapacity)this.reservePairs(this.pairCapacity*2);
            const a=1-r2/h2,w=a*a*a;this.pairA[count]=i;this.pairB[count]=j;this.pairWeight[count]=w;
            if(invDensity){const grad=coefficient*a*a,ratio=w/.753571,square=ratio*ratio;this.pairGX[count]=grad*rx;this.pairGY[count]=grad*ry;this.pairGZ[count]=grad*rz;this.pairAntiClump[count]=-.000014*square*square;}
            count++;
          }
          j=next[j];
        }
      }
    }
    this.pairCount=count;
  }
  bound(k){
    const p=this.predicted,r=this.spacing*.28;
    p[k]=clamp(p[k],-.9+r,.9-r);p[k+1]=clamp(p[k+1],r,1.85-r);p[k+2]=clamp(p[k+2],-.9+r,.9-r);
  }
  setPointer(target){
    this.pointer[0]=clamp(Number(target[0])||0,-.44,.44);
    this.pointer[1]=clamp(Number.isFinite(target[1])?target[1]:.89,.53,1.25);
    this.pointer[2]=clamp(Number(target[2])||0,-.44,.44);
  }
  clearPointer(){this.pointer[0]=0;this.pointer[1]=.89;this.pointer[2]=0;}
  burst(strength=1){
    strength=clamp(strength,.1,1);this.charge=0;
    for(let i=0;i<this.count;i++){
      const k=i*3,x=this.positions[k]-this.center[0],y=this.positions[k+1]-this.center[1],z=this.positions[k+2]-this.center[2],length=Math.max(.04,Math.hypot(x,y,z));
      // Initial impulse is applied to velocities; subsequent separation is physical.
      const speed=(1.1+strength*2.45)*(0.75+.25*Math.min(1,length/.3));
      this.velocities[k]+=x/length*speed;this.velocities[k+1]+=y/length*speed+.25*strength;this.velocities[k+2]+=z/length*speed;
    }
  }
  substep(dt){
    const p=this.positions,v=this.velocities,q=this.predicted,n=this.count,h2=this.h2;
    this.previous.set(p);this.time+=dt;
    // Move the containment field, instead of adding a short-lived velocity kick
    // to a few nearby particles. A held pointer remains effective and release
    // impulses use the same moving center as the compressed liquid.
    const follow=1-Math.exp(-dt*5);
    for(let axis=0;axis<3;axis++)this.center[axis]+=(this.pointer[axis]-this.center[axis])*follow;
    const pull=4.0+this.energy*4.0+this.charge*20,spin=.30+this.energy*1.6,damping=Math.exp(-dt*(.85+this.energy*.20+this.charge*1.7));
    for(let i=0;i<n;i++){
      const k=i*3,rx=p[k]-this.center[0],ry=p[k+1]-this.center[1],rz=p[k+2]-this.center[2];
      // A bounded harmonic containment field lets individual droplets escape,
      // strike the floor and return; it does not constrain the shape geometrically.
      let ax=-rx*pull+rz*spin,ay=-ry*pull,az=-rz*pull-rx*spin;
      v[k]=(v[k]+ax*dt)*damping;v[k+1]=(v[k+1]+ay*dt)*damping;v[k+2]=(v[k+2]+az*dt)*damping;
      q[k]=p[k]+v[k]*dt;q[k+1]=p[k+1]+v[k+1]*dt;q[k+2]=p[k+2]+v[k+2]*dt;this.bound(k);
    }
    const densityTarget=this.restDensity*(1+this.charge*.38),invDensity=1/densityTarget;
    const sums=this.gradientSums,norms=this.gradientNorms,densities=this.densities,corrections=this.corrections,lambdas=this.lambdas;
    for(let iteration=0;iteration<3;iteration++){
      this.buildPairs(invDensity);densities.fill(1);sums.fill(0);norms.fill(0);
      const {pairA,pairB,pairWeight,pairGX,pairGY,pairGZ,pairAntiClump,pairCount}=this;
      for(let e=0;e<pairCount;e++){
        const i=pairA[e],j=pairB[e],ik=i*3,jk=j*3,w=pairWeight[e],gx=pairGX[e],gy=pairGY[e],gz=pairGZ[e],squared=gx*gx+gy*gy+gz*gz;
        densities[i]+=w;densities[j]+=w;norms[i]+=squared;norms[j]+=squared;
        sums[ik]+=gx;sums[ik+1]+=gy;sums[ik+2]+=gz;sums[jk]-=gx;sums[jk+1]-=gy;sums[jk+2]-=gz;
      }
      for(let i=0;i<n;i++){const k=i*3,gx=sums[k],gy=sums[k+1],gz=sums[k+2],constraint=Math.max(0,densities[i]*invDensity-1);lambdas[i]=-constraint/(norms[i]+gx*gx+gy*gy+gz*gz+35);}
      corrections.fill(0);
      for(let e=0;e<pairCount;e++){
        const i=pairA[e],j=pairB[e],ik=i*3,jk=j*3,scale=lambdas[i]+lambdas[j]+pairAntiClump[e],dx=scale*pairGX[e],dy=scale*pairGY[e],dz=scale*pairGZ[e];
        corrections[ik]+=dx;corrections[ik+1]+=dy;corrections[ik+2]+=dz;corrections[jk]-=dx;corrections[jk+1]-=dy;corrections[jk+2]-=dz;
      }
      const limit=this.spacing*.32,limit2=limit*limit;
      for(let k=0;k<q.length;k+=3){const dx=corrections[k],dy=corrections[k+1],dz=corrections[k+2],length2=dx*dx+dy*dy+dz*dz,factor=length2>limit2?limit/Math.sqrt(length2):1;q[k]+=dx*factor;q[k+1]+=dy*factor;q[k+2]+=dz*factor;this.bound(k);}
    }
    for(let k=0;k<q.length;k++){v[k]=(q[k]-this.previous[k])/dt;p[k]=q[k];}
    // Symmetric XSPH exchanges use one cached pair traversal as well.
    this.buildPairs(0);corrections.fill(0);const weights=this.neighborWeights;weights.fill(0);
    const cohesion=dt*3.0;
    for(let e=0;e<this.pairCount;e++){
      const i=this.pairA[e],j=this.pairB[e],ik=i*3,jk=j*3,w=this.pairWeight[e];weights[i]+=w;weights[j]+=w;
      for(let axis=0;axis<3;axis++){const delta=((v[jk+axis]-v[ik+axis])*.055+(q[jk+axis]-q[ik+axis])*cohesion)*w;corrections[ik+axis]+=delta;corrections[jk+axis]-=delta;}
    }
    for(let i=0;i<n;i++)if(weights[i]>.01){const k=i*3,invWeight=1/weights[i];v[k]+=corrections[k]*invWeight;v[k+1]+=corrections[k+1]*invWeight;v[k+2]+=corrections[k+2]*invWeight;}

  }
  step(dt){
    if(!Number.isFinite(dt)||dt<=0)return;
    this.pendingTime+=Math.min(dt,.04);
    // PBD derives velocity from corrected displacement / dt. Accumulate tiny
    // invalidations rather than magnifying a density correction into an impulse.
    if(this.pendingTime<1/240)return;
    const duration=this.pendingTime;this.pendingTime=0;
    const substeps=Math.max(1,Math.ceil(duration/.010));for(let i=0;i<substeps;i++)this.substep(duration/substeps);
  }
}

export async function createCpuEngine({canvas,scene,camera,onStats,renderer:providedRenderer}={}) {
  const renderer=providedRenderer||new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'high-performance'});
  const software=!!renderer.isSoftwareRenderer;
  if(!software){renderer.setPixelRatio(stagePixelRatio(1,1,globalThis.devicePixelRatio));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;}
  const fluid=new CpuFluidSolver({spacing:software?.09:.064});
  const materialMetal=new THREE.MeshStandardMaterial({color:0xcbd2d5,metalness:1,roughness:.105,envMapIntensity:1.55});
  const materialGlass=new THREE.MeshPhysicalMaterial({color:0xafd9e0,metalness:0,roughness:.055,transmission:.94,thickness:.32,ior:1.46,clearcoat:1,clearcoatRoughness:.04,attenuationColor:0x75b5c5,attenuationDistance:1.2,envMapIntensity:1.5});
  const mesh=new MarchingCubes(48,materialMetal,false,false,55000);
  const surfaceIndices=new Uint32Array(mesh.positionArray.length/3),edgeVertices=new Int32Array(mesh.resolution**3*3);
  const indexAttribute=new THREE.BufferAttribute(surfaceIndices,1).setUsage(THREE.DynamicDrawUsage);mesh.geometry.setIndex(indexAttribute);
  const compactSurface=()=>{
    // MarchingCubes emits each grid-edge intersection once per adjacent face.
    // The intersection is identical, so share it instead of transforming/uploading
    // the same position and smooth normal up to six times. No simplification.
    const size=mesh.resolution,size2=size*size,size3=size2*size,half=size/2,p=mesh.positionArray,n=mesh.normalArray;
    edgeVertices.fill(-1);let unique=0;
    for(let i=0;i<mesh.count;i++){
      const k=i*3,x=(p[k]+1)*half,y=(p[k+1]+1)*half,z=(p[k+2]+1)*half;
      const ix=Math.round(x),iy=Math.round(y),iz=Math.round(z),dx=Math.abs(x-ix),dy=Math.abs(y-iy),dz=Math.abs(z-iz);
      let key;if(dx>=dy&&dx>=dz)key=Math.floor(x)+iy*size+iz*size2;else if(dy>=dz)key=size3+ix+Math.floor(y)*size+iz*size2;else key=size3*2+ix+iy*size+Math.floor(z)*size2;
      let vertex=edgeVertices[key];
      if(vertex===-1){vertex=unique++;edgeVertices[key]=vertex;const dest=vertex*3;p[dest]=p[k];p[dest+1]=p[k+1];p[dest+2]=p[k+2];n[dest]=n[k];n[dest+1]=n[k+1];n[dest+2]=n[k+2];}
      surfaceIndices[i]=vertex;
    }
    for(const name of ['position','normal']){const attribute=mesh.geometry.getAttribute(name);attribute.count=unique;attribute.clearUpdateRanges();if(unique)attribute.addUpdateRange(0,unique*3);}
    indexAttribute.count=mesh.count;indexAttribute.clearUpdateRanges();if(mesh.count)indexAttribute.addUpdateRange(0,mesh.count);indexAttribute.needsUpdate=true;
  };
  // MarchingCubes local coordinates span [-1,1]. A .96 scale gives a 1.92 m field.
  mesh.position.set(0,.925,0);mesh.scale.setScalar(.96);mesh.isolation=80;
  mesh.castShadow=true;mesh.receiveShadow=true;mesh.frustumCulled=false;mesh.name='simulated-fluid-surface';scene.add(mesh);
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(fluid.positions,3).setUsage(THREE.DynamicDrawUsage));
  const ids=new Float32Array(fluid.count);for(let i=0;i<ids.length;i++)ids[i]=i;geometry.setAttribute('aId',new THREE.BufferAttribute(ids,1));
  // Standard PointsMaterial keeps the same simulated particles available to all renderers.
  const spritePixels=new Uint8Array(32*32*4);
  for(let y=0;y<32;y++)for(let x=0;x<32;x++){const r=Math.hypot((x-15.5)/15.5,(y-15.5)/15.5),i=(y*32+x)*4;spritePixels[i]=spritePixels[i+1]=spritePixels[i+2]=255;spritePixels[i+3]=Math.round(255*Math.exp(-r*r*4)*Math.max(0,1-r));}
  const sprite=new THREE.DataTexture(spritePixels,32,32,THREE.RGBAFormat);sprite.magFilter=THREE.LinearFilter;sprite.minFilter=THREE.LinearFilter;sprite.needsUpdate=true;
  const pointMaterial=new THREE.PointsMaterial({color:0xffb855,map:sprite,size:.055,sizeAttenuation:true,transparent:true,opacity:.96,blending:THREE.AdditiveBlending,depthWrite:false});
  const points=new THREE.Points(geometry,pointMaterial);points.visible=false;points.frustumCulled=false;points.name='simulated-fluid-particles';scene.add(points);
  let paused=false,material='metal',dirty=true,elapsed=0,frames=0;
  const target=new THREE.Vector3(),center=new THREE.Vector3(0,.89,0);
  const pointerRight=new THREE.Vector3(),pointerUp=new THREE.Vector3(),pointerPositionArray=[0,0,0];
  const updateSurface=()=>{
    if(material==='plasma'){geometry.attributes.position.needsUpdate=true;return;}
    // UVs/colors are disabled, so resetting palette data is unnecessary.
    mesh.field.fill(0);mesh.normal_cache.fill(0);mesh.count=0;
    const p=fluid.positions,strength=.035*(fluid.spacing/.064)**2;
    if(software){
      // A bounded C2 SPH density kernel avoids the singular metaball peaks
      // that produce angular finite-difference normals at sparse resolutions.
      const size=mesh.resolution,planeSize=size*size,radius=fluid.h/1.92*size,radius2=radius*radius,field=mesh.field;
      for(let i=0;i<fluid.count;i++){
        const k=i*3,cx=(p[k]/1.92+.5)*size,cy=((p[k+1]-.925)/1.92+.5)*size,cz=(p[k+2]/1.92+.5)*size;
        const x0=Math.max(1,Math.floor(cx-radius)),x1=Math.min(size-2,Math.ceil(cx+radius)),y0=Math.max(1,Math.floor(cy-radius)),y1=Math.min(size-2,Math.ceil(cy+radius)),z0=Math.max(1,Math.floor(cz-radius)),z1=Math.min(size-2,Math.ceil(cz+radius));
        for(let z=z0;z<=z1;z++)for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
          const dx=x-cx,dy=y-cy,dz=z-cz,a=1-(dx*dx+dy*dy+dz*dz)/radius2;
          if(a>0)field[x+y*size+z*planeSize]+=125*a*a*a;
        }
      }
    }else{
      for(let i=0;i<fluid.count;i++){const k=i*3;mesh.addBall(p[k]/1.92+.5,(p[k+1]-.925)/1.92+.5,p[k+2]/1.92+.5,strength,12);}
    }
    mesh.update();
    compactSurface();
  };
  // Settle initial overlaps through the same solver used during interaction.
  for(let i=0;i<8;i++)fluid.step(1/120);updateSurface();
  const api={
    renderer,kind:software?'software':'webgl',mesh,points,particleCount:fluid.count,
    setMaterial(name){
      const nextMaterial=['metal','glass','plasma'].includes(name)?name:'metal';if(nextMaterial===material)return;
      const previousMaterial=material;material=nextMaterial;mesh.material=material==='glass'?materialGlass:materialMetal;mesh.visible=material!=='plasma';points.visible=material==='plasma';
      // Both liquid materials use the same surface. Rebuild only if its particle
      // data changed, or it was not maintained while the particle view was active.
      if(material==='plasma'||previousMaterial==='plasma'||dirty)updateSurface();
      dirty=false;
    },
    setEnergy(value){fluid.energy=clamp(Number(value)||0,0,1);},
    clearPointer(){fluid.clearPointer();},
    setPointer(x,y){
      camera.updateMatrixWorld();
      pointerRight.setFromMatrixColumn(camera.matrixWorld,0);pointerUp.setFromMatrixColumn(camera.matrixWorld,1);
      // Screen-relative, bounded travel remains intuitive while orbiting. The
      // center is a valid target; only clearPointer() ends the interaction.
      target.copy(center).addScaledVector(pointerRight,clamp(Number(x)||0,-1,1)*.36)
        .addScaledVector(pointerUp,clamp(Number(y)||0,-1,1)*.30);
      fluid.setPointer(target.toArray(pointerPositionArray));
    },
    setCharge(value){fluid.charge=clamp(Number(value)||0,0,1);},
    burst(strength=1){fluid.burst(strength);dirty=true;},
    setPaused(value){paused=!!value;},
    reset(){fluid.reset();material='metal';mesh.material=materialMetal;mesh.visible=true;points.visible=false;dirty=false;updateSurface();},
    step(dt){if(!paused){fluid.step(dt);dirty=true;}if(dirty){updateSurface();dirty=false;}elapsed+=dt;frames++;if(elapsed>.75){onStats?.({fps:Math.round(frames/elapsed),particles:fluid.count,backend:software?'Software 3D':'WebGL2',material,energy:fluid.energy});frames=0;elapsed=0;}},
    resize(w,h){
      // Budget retina supersampling, not the base scene. Rendering below one
      // pixel per CSS pixel makes the ring and liquid outlines visibly blocky.
      if(!software)renderer.setPixelRatio(stagePixelRatio(w,h,globalThis.devicePixelRatio,{pixelBudget:2_400_000}));
      renderer.setSize(w,h,false);
    },
    getParticles(){return new Float32Array(fluid.positions);},
    destroy(){scene.remove(mesh,points);mesh.geometry.dispose();geometry.dispose();materialMetal.dispose();materialGlass.dispose();pointMaterial.dispose();sprite.dispose();if(!providedRenderer)renderer.dispose();}
  };
  return api;
}
