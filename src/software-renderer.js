/* Original geometry-preserving software renderer for browsers without GPU access.
 * The simulation, meshes and camera are shared with the WebGL experience.
 * Reflection shading is an approximation; this is not a second simulation.
 */
import { Matrix4, Matrix3, Vector3, Color } from 'three';

const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const smooth=(a,b,v)=>{const t=clamp((v-a)/(b-a));return t*t*(3-2*t);};
const toneMap=new Uint8Array(65536);
for(let i=0;i<toneMap.length;i++){const v=i/8192;toneMap[i]=Math.round(255*Math.pow(clamp((v*(2.51*v+.03))/(v*(2.43*v+.59)+.14)),1/2.2));}
const encode=v=>toneMap[Math.max(0,Math.min(65535,(v*8192)|0))];
const MATCAP_SIZE=256;
let reflectionMap;

// Filter the final image, not individual triangles. Alpha coverage per triangle
// would leave seams where adjacent fluid faces share the same silhouette pixel.
// This short directional filter touches contrast edges only; depth stays intact.
export function smoothRasterEdges(source,target,depth,width,height){
  target.set(source);
  const stride=width*4;
  const luma=i=>source[i]*.299+source[i+1]*.587+source[i+2]*.114;
  for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
    const pixel=y*width+x;
    // Only silhouettes need this fallback pass. Interior highlights keep their
    // original contrast, and the smooth backdrop does not need colour samples.
    const occupied=depth[pixel]!==Infinity;
    if((depth[pixel-width]!==Infinity)===occupied&&(depth[pixel+width]!==Infinity)===occupied&&
      (depth[pixel-1]!==Infinity)===occupied&&(depth[pixel+1]!==Infinity)===occupied)continue;
    const i=pixel*4,n=luma(i-stride),s=luma(i+stride),w=luma(i-4),e=luma(i+4),c=luma(i);
    const low=Math.min(c,n,s,w,e),high=Math.max(c,n,s,w,e);
    if(high-low<Math.max(12,high*.125))continue;
    // Smooth along an edge, not across it. Straight edges and flat interiors
    // remain exact; the little steps where an edge changes rows get coverage.
    const step=Math.abs(n+s-2*c)<Math.abs(w+e-2*c)?stride:4;
    for(let channel=0;channel<3;channel++)target[i+channel]=source[i+channel]*.5+
      (source[i-step+channel]+source[i+step+channel])*.25;
  }
  return target;
}

function createReflectionMap(){
  if(reflectionMap)return reflectionMap;
  const a=new Float32Array(MATCAP_SIZE*MATCAP_SIZE*4);
  const rect=(x,y,w,h,s)=>1-smooth(-s,s,Math.max(Math.abs(x)-w,Math.abs(y)-h));
  for(let iy=0;iy<MATCAP_SIZE;iy++)for(let ix=0;ix<MATCAP_SIZE;ix++){
    const nx=ix/(MATCAP_SIZE-1)*2-1,ny=iy/(MATCAP_SIZE-1)*2-1;
    const nz=Math.sqrt(Math.max(0,1-nx*nx-ny*ny));
    const rx=2*nx*nz,ry=2*ny*nz,rz=2*nz*nz-1;
    const az=Math.atan2(rx,rz),el=Math.asin(clamp(ry,-1,1));
    const key=rect(az+.93,el-.53,.49,.64,.17)*2.1;
    const strip=rect(az-1.48,el-.22,.16,.93,.075)*3.2;
    const top=rect(az-.12,el-1.06,.78,.19,.13)*1.3;
    const rim=rect(Math.abs(az)-2.73,el-.18,.23,.80,.12)*1.8;
    const gold=rect(az+.07,el+.22,.055,.63,.06)*.85;
    const floor=(1-smooth(-.85,-.32,ry))*.14;
    const haze=Math.exp(-Math.pow((el+.12)/.10,2))*.055;
    const off=(iy*MATCAP_SIZE+ix)*4;
    a[off]=.008+key*.90+strip*.98+top+rim*.52+gold+floor+haze;
    a[off+1]=.011+key*.95+strip+top+rim*.72+gold*.46+floor*1.12+haze*1.2;
    a[off+2]=.015+key+strip+top+rim*.91+gold*.12+floor*1.23+haze*1.4;
    a[off+3]=Math.pow(1-nz,3.2);
  }
  reflectionMap=a;return a;
}

export class SoftwareRenderer {
  constructor({canvas}={}){
    this.domElement=canvas||document.createElement('canvas');
    this.context=this.domElement.getContext('2d',{alpha:false});
    if(!this.context)throw new Error('Canvas 2D is unavailable');
    this.isSoftwareRenderer=true;this.isWebGLRenderer=false;this.shadowMap={enabled:false};
    this.toneMapping=0;this.toneMappingExposure=1;this.outputColorSpace='srgb';
    this.pixelRatio=1;this.width=1;this.height=1;this.disposed=false;
    this.surface=document.createElement('canvas');this.surfaceContext=this.surface.getContext('2d');
    this.matcap=createReflectionMap();this.modelView=new Matrix4();this.mvp=new Matrix4();this.normalMatrix=new Matrix3();
    this.instanceMatrix=new Matrix4();this.worldMatrix=new Matrix4();this.tmp=new Vector3();this.color=new Color();
    this.vertexCache=new WeakMap();this.backgroundKey='';this.background=null;
    this.info={render:{triangles:0,points:0,calls:0},memory:{geometries:0,textures:0}};
    this.setSize(this.domElement.clientWidth||800,this.domElement.clientHeight||600);
  }
  setPixelRatio(r){this.pixelRatio=Math.max(.5,Math.min(2,Number(r)||1));}
  getPixelRatio(){return this.pixelRatio;}
  getSize(target){return target.set(this.width,this.height);}
  setSize(w,h,updateStyle=true){
    this.width=Math.max(1,Math.round(w));this.height=Math.max(1,Math.round(h));
    // Keep software work bounded even on 4K displays. The final silhouette
    // filter smooths the sampling steps without multiplying the shading cost.
    const scale=Math.min(1,760/this.width,620/this.height,Math.sqrt(310000/(this.width*this.height)));
    this.rw=Math.max(1,Math.round(this.width*scale));this.rh=Math.max(1,Math.round(this.height*scale));
    this.domElement.width=this.width;this.domElement.height=this.height;
    if(updateStyle){this.domElement.style.width=`${this.width}px`;this.domElement.style.height=`${this.height}px`;}
    this.surface.width=this.rw;this.surface.height=this.rh;
    this.image=this.surfaceContext.createImageData(this.rw,this.rh);this.filteredImage=this.surfaceContext.createImageData(this.rw,this.rh);this.depth=new Float32Array(this.rw*this.rh);
    this.backgroundKey='';
  }
  _background(scene){
    const c=scene.background?.isColor?scene.background:this.color.set('#101318');
    const key=`${this.rw}:${this.rh}:${c.getHexString()}`;
    if(key!==this.backgroundKey){
      this.backgroundKey=key;this.background=new Uint8ClampedArray(this.image.data.length);
      const srgb=c.clone().convertLinearToSRGB(),base=[srgb.r*255,srgb.g*255,srgb.b*255];
      for(let y=0;y<this.rh;y++)for(let x=0;x<this.rw;x++){
        const xx=(x/this.rw-.58)*1.5,yy=(y/this.rh-.46),halo=Math.exp(-(xx*xx+yy*yy)*5.2);
        const floor=smooth(.50,.97,y/this.rh)*4,shade=Math.exp(-Math.pow((y/this.rh-.77)/.085,2)-Math.pow((x/this.rw-.56)/.27,2))*8;
        const i=(y*this.rw+x)*4;
        this.background[i]=base[0]+halo*4+floor-shade;
        this.background[i+1]=base[1]+halo*5+floor-shade;
        this.background[i+2]=base[2]+halo*6+floor-shade;
        this.background[i+3]=255;
      }
    }
    this.image.data.set(this.background);this.depth.fill(Infinity);
  }
  _vertices(geometry,matrix,camera,count){
    const p=geometry.attributes.position,n=geometry.attributes.normal,uv=geometry.attributes.uv;
    let v=this.vertexCache.get(geometry);
    if(!v||v.length<count*10){v=new Float32Array(count*10);this.vertexCache.set(geometry,v);}
    this.modelView.multiplyMatrices(camera.matrixWorldInverse,matrix);this.mvp.multiplyMatrices(camera.projectionMatrix,this.modelView);this.normalMatrix.getNormalMatrix(this.modelView);
    const m=this.mvp.elements,nm=this.normalMatrix.elements,pa=p.array,na=n?.array,ua=uv?.array;
    for(let i=0;i<count;i++){
      const k=i*p.itemSize,x=pa[k],y=pa[k+1],z=pa[k+2],o=i*10;
      const cw=m[3]*x+m[7]*y+m[11]*z+m[15],iw=1/cw;
      v[o]=(m[0]*x+m[4]*y+m[8]*z+m[12])*iw*this.rw*.5+this.rw*.5;
      v[o+1]=-(m[1]*x+m[5]*y+m[9]*z+m[13])*iw*this.rh*.5+this.rh*.5;
      v[o+2]=(m[2]*x+m[6]*y+m[10]*z+m[14])*iw;
      v[o+3]=cw;
      if(na){const nn=i*n.itemSize,nx=na[nn],ny=na[nn+1],nz=na[nn+2];v[o+4]=nm[0]*nx+nm[3]*ny+nm[6]*nz;v[o+5]=nm[1]*nx+nm[4]*ny+nm[7]*nz;v[o+6]=nm[2]*nx+nm[5]*ny+nm[8]*nz;}
      else{v[o+4]=0;v[o+5]=0;v[o+6]=1;}
      if(ua){v[o+7]=ua[i*uv.itemSize];v[o+8]=ua[i*uv.itemSize+1];}
    }
    return v;
  }
  _material(material){
    const c=material.color||new Color(1,1,1),e=material.emissive||new Color(0,0,0);
    const metal=material.metalness||0,roughness=material.roughness??1,reflection=(.035+.965*metal)*(1-roughness*.30),emissiveIntensity=material.emissiveIntensity||1;
    // These factors are uniform over a mesh; computing them per fragment is
    // especially wasteful when a burst covers much more of the viewport.
    return {color:[c.r,c.g,c.b],emissive:[e.r*emissiveIntensity,e.g*emissiveIntensity,e.b*emissiveIntensity],diffuse:[c.r*(1-metal),c.g*(1-metal),c.b*(1-metal)],reflection:[reflection*(.25+.75*c.r),reflection*(.25+.75*c.g),reflection*(.25+.75*c.b)],transmission:material.transmission||0,basic:material.isMeshBasicMaterial,alpha:material.opacity??1,shadow:!!material.map&&material.isMeshBasicMaterial,side:material.side||0,depthWrite:material.depthWrite!==false};
  }
  _triangle(v,ia,ib,ic,m){
    ia*=10;ib*=10;ic*=10;
    if(v[ia+3]<=.025||v[ib+3]<=.025||v[ic+3]<=.025)return;
    const ax=v[ia],ay=v[ia+1],bx=v[ib],by=v[ib+1],cx=v[ic],cy=v[ic+1];
    const area=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
    if(Math.abs(area)<.015||(m.side===0&&area>0)||(m.side===1&&area<0))return;
    const minX=Math.max(0,Math.floor(Math.min(ax,bx,cx))),maxX=Math.min(this.rw-1,Math.ceil(Math.max(ax,bx,cx)));
    const minY=Math.max(0,Math.floor(Math.min(ay,by,cy))),maxY=Math.min(this.rh-1,Math.ceil(Math.max(ay,by,cy)));
    if(maxX<minX||maxY<minY)return;
    const inv=1/((by-cy)*(ax-cx)+(cx-bx)*(ay-cy)),dax=(by-cy)*inv,dbx=(cy-ay)*inv;
    const data=this.image.data,depth=this.depth,map=this.matcap,ms=MATCAP_SIZE-1;
    const nsign=area>0?-1:1,alpha=m.transmission>.05?1:m.alpha,opaque=alpha===1;
    const span=maxX-minX,clipRows=span>8&&(span+1)*(maxY-minY+1)>Math.abs(area)*2;
    const dcx=-dax-dbx;
    for(let y=minY;y<=maxY;y++){
      let wa=((by-cy)*(minX+.5-cx)+(cx-bx)*(y+.5-cy))*inv,wb=((cy-ay)*(minX+.5-cx)+(ax-cx)*(y+.5-cy))*inv;
      let rowStart=minX,rowEnd=maxX;
      // Bound long, narrow triangles to their actual covered scanline. This
      // keeps the same barycentric tolerance while avoiding empty box pixels.
      if(clipRows){
        let lo=0,hi=span;
        if(dax>0)lo=Math.max(lo,(-.00001-wa)/dax);else if(dax<0)hi=Math.min(hi,(-.00001-wa)/dax);
        if(dbx>0)lo=Math.max(lo,(-.00001-wb)/dbx);else if(dbx<0)hi=Math.min(hi,(-.00001-wb)/dbx);
        const wc0=1-wa-wb;
        if(dcx>0)lo=Math.max(lo,(-.00001-wc0)/dcx);else if(dcx<0)hi=Math.min(hi,(-.00001-wc0)/dcx);
        rowStart+=Math.max(0,Math.ceil(lo));rowEnd=minX+Math.min(span,Math.floor(hi));
        const skip=rowStart-minX;wa+=skip*dax;wb+=skip*dbx;
      }
      for(let x=rowStart;x<=rowEnd;x++,wa+=dax,wb+=dbx){
        const wc=1-wa-wb;if(wa<-.00001||wb<-.00001||wc<-.00001)continue;
        const z=wa*v[ia+2]+wb*v[ib+2]+wc*v[ic+2],idx=y*this.rw+x;if(z>1||z< -1||z>=depth[idx])continue;
        const off=idx*4;
        if(m.shadow){
          const u=wa*v[ia+7]+wb*v[ib+7]+wc*v[ic+7]-.5,vt=wa*v[ia+8]+wb*v[ib+8]+wc*v[ic+8]-.5;
          const a=Math.exp(-(u*u+vt*vt)*14)*.62*m.alpha;
          data[off]*=1-a;data[off+1]*=1-a;data[off+2]*=1-a;continue;
        }
        let r,g,b;
        if(m.basic){r=m.color[0];g=m.color[1];b=m.color[2];}
        else {
          let nx=(wa*v[ia+4]+wb*v[ib+4]+wc*v[ic+4])*nsign,ny=(wa*v[ia+5]+wb*v[ib+5]+wc*v[ic+5])*nsign,nz=(wa*v[ia+6]+wb*v[ib+6]+wc*v[ic+6])*nsign;
          const ni=1/Math.sqrt(Math.max(.000001,nx*nx+ny*ny+nz*nz));nx*=ni;ny*=ni;nz*=ni;
          const mx=clamp((nx*.5+.5)*ms,0,ms-.001),my=clamp((ny*.5+.5)*ms,0,ms-.001),ix=mx|0,iy=my|0,fx=mx-ix,fy=my-iy;
          const p=(iy*MATCAP_SIZE+ix)*4,p1=p+4,p2=p+MATCAP_SIZE*4,p3=p2+4;
          const w0=(1-fx)*(1-fy),w1=fx*(1-fy),w2=(1-fx)*fy,w3=fx*fy;
          const er=map[p]*w0+map[p1]*w1+map[p2]*w2+map[p3]*w3,eg=map[p+1]*w0+map[p1+1]*w1+map[p2+1]*w2+map[p3+1]*w3,eb=map[p+2]*w0+map[p1+2]*w1+map[p2+2]*w2+map[p3+2]*w3;
          const fr=1-Math.max(0,nz),fres=fr*fr*fr,diff=.12+Math.max(0,nx*-.45+ny*.70+nz*.55)*.50;
          if(m.transmission>.05){
            const refl=.08+.82*fres,s=Math.max(0,Math.sin(nx*4-ny*5)),s2=s*s,s4=s2*s2,s8=s4*s4,caustic=s8*s8*s2*.09;
            r=er*refl+.007+caustic*.15;g=eg*refl+.038+caustic*.7;b=eb*refl+.053+caustic;
          }else{
            r=diff*m.diffuse[0]+er*m.reflection[0];
            g=diff*m.diffuse[1]+eg*m.reflection[1];
            b=diff*m.diffuse[2]+eb*m.reflection[2];
          }
        }
        r+=m.emissive[0];g+=m.emissive[1];b+=m.emissive[2];
        if(opaque){data[off]=encode(r);data[off+1]=encode(g);data[off+2]=encode(b);}
        else{data[off]=data[off]*(1-alpha)+encode(r)*alpha;data[off+1]=data[off+1]*(1-alpha)+encode(g)*alpha;data[off+2]=data[off+2]*(1-alpha)+encode(b)*alpha;}
        if(m.depthWrite)depth[idx]=z;
      }
    }
    this.info.render.triangles++;
  }
  _mesh(mesh,matrix,camera){
    const geo=mesh.geometry,p=geo?.attributes?.position;if(!p)return;
    // Infinite ground is represented by the backdrop, avoiding clipping giant triangles.
    if(geo.type==='PlaneGeometry'&&(geo.parameters.width>20||geo.parameters.height>20))return;
    const index=geo.index?.array,available=index?index.length:p.count;
    const start=Math.max(0,geo.drawRange?.start||0),end=Math.min(available,start+(geo.drawRange?.count??available));
    if(end<=start)return;
    const vertexCount=index?p.count:end,v=this._vertices(geo,matrix,camera,vertexCount);
    const mat=Array.isArray(mesh.material)?mesh.material[0]:mesh.material;if(!mat||!mat.visible||mat.opacity===0)return;
    const m=this._material(mat);
    for(let i=start;i+2<end;i+=3)this._triangle(v,index?index[i]:i,index?index[i+1]:i+1,index?index[i+2]:i+2,m);
    this.info.render.calls++;
  }
  _point(x,y,z,r,color,alpha=1){
    if(z< -1||z>1)return;
    const data=this.image.data;
    for(let yy=Math.max(0,Math.floor(y-r*2));yy<=Math.min(this.rh-1,Math.ceil(y+r*2));yy++)for(let xx=Math.max(0,Math.floor(x-r*2));xx<=Math.min(this.rw-1,Math.ceil(x+r*2));xx++){
      const d=((xx-x)**2+(yy-y)**2)/(r*r),a=Math.exp(-d*2.4)*alpha;if(a<.01)continue;
      const idx=yy*this.rw+xx;if(z>this.depth[idx])continue;const o=idx*4;
      data[o]=Math.min(255,data[o]+color[0]*a);data[o+1]=Math.min(255,data[o+1]+color[1]*a);data[o+2]=Math.min(255,data[o+2]+color[2]*a);
    }
    this.info.render.points++;
  }
  _points(mesh,camera){
    const geo=mesh.geometry,p=geo.attributes.position;if(!p)return;
    const count=Math.min(p.count,geo.drawRange?.count??p.count),v=this._vertices(geo,mesh.matrixWorld,camera,count),c=mesh.material.color||new Color(0xffb76c),cols=geo.attributes.color;
    for(let i=0;i<count;i++){
      const o=i*10;if(v[o+3]<=.025)continue;
      const size=mesh.material.size||.015,r=clamp(size*this.rh*1.6/v[o+3],.7,4);
      const color=cols?[Math.sqrt(cols.array[i*3])*255,Math.sqrt(cols.array[i*3+1])*255,Math.sqrt(cols.array[i*3+2])*255]:[Math.sqrt(c.r)*255,Math.sqrt(c.g)*255,Math.sqrt(c.b)*255];
      this._point(v[o],v[o+1],v[o+2],r,color,mesh.material.opacity??1);
    }
  }
  render(scene,camera){
    if(this.disposed)return;
    scene.updateMatrixWorld(true);camera.updateMatrixWorld();camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    this.info.render={triangles:0,points:0,calls:0};this._background(scene);
    const opaque=[],transparent=[],points=[];
    scene.traverseVisible(o=>{if(o.isPoints)points.push(o);else if(o.isMesh){const m=Array.isArray(o.material)?o.material[0]:o.material;(m?.transparent&&!m.transmission?transparent:opaque).push(o);}});
    const draw=o=>{
      if(o.isInstancedMesh){
        for(let i=0;i<o.count;i++){
          o.getMatrixAt(i,this.instanceMatrix);this.worldMatrix.multiplyMatrices(o.matrixWorld,this.instanceMatrix);
          // Particle displays use a luminous projected sprite rather than tiny sphere triangles.
          if(o.count>50){this.modelView.multiplyMatrices(camera.matrixWorldInverse,this.worldMatrix);this.mvp.multiplyMatrices(camera.projectionMatrix,this.modelView);const e=this.mvp.elements,w=e[15];if(w<=.025)continue;const scale=Math.hypot(this.worldMatrix.elements[0],this.worldMatrix.elements[1],this.worldMatrix.elements[2]);const c=o.material.color||new Color(0xffb767);this._point((e[12]/w+1)*this.rw*.5,(1-e[13]/w)*this.rh*.5,e[14]/w,clamp(scale*this.rh*.55/w,.8,4),[Math.sqrt(c.r)*255,Math.sqrt(c.g)*255,Math.sqrt(c.b)*255],o.material.opacity??1);}
          else this._mesh(o,this.worldMatrix,camera);
        }
      }else this._mesh(o,o.matrixWorld,camera);
    };
    opaque.forEach(draw);transparent.forEach(draw);points.forEach(o=>this._points(o,camera));
    smoothRasterEdges(this.image.data,this.filteredImage.data,this.depth,this.rw,this.rh);
    this.surfaceContext.putImageData(this.filteredImage,0,0);
    this.context.imageSmoothingEnabled=true;this.context.imageSmoothingQuality='high';
    this.context.drawImage(this.surface,0,0,this.width,this.height);
  }
  clear(){this.context.clearRect(0,0,this.width,this.height);}
  dispose(){this.disposed=true;this.image=null;this.filteredImage=null;this.depth=null;this.background=null;this.vertexCache=new WeakMap();}
  forceContextLoss(){}
}
