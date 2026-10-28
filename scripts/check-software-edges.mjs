import assert from 'node:assert/strict';
import {smoothRasterEdges} from '../src/software-renderer.js';

const width=48,height=40;
function image(inside){
  const pixels=new Uint8ClampedArray(width*height*4),depth=new Float32Array(width*height).fill(Infinity);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const p=y*width+x,i=p*4,value=inside(x,y)?224:16;
    pixels[i]=pixels[i+1]=pixels[i+2]=value;pixels[i+3]=255;
    if(value===224)depth[p]=.5;
  }
  return {pixels,depth};
}
function filter(input){
  const source=input.pixels.slice(),depth=input.depth.slice();
  const result=smoothRasterEdges(input.pixels,new Uint8ClampedArray(input.pixels.length),input.depth,width,height);
  assert.deepEqual(input.pixels,source,'filter reads an immutable source image');
  assert.deepEqual(input.depth,depth,'antialiasing never changes visibility/depth');
  for(let i=3;i<result.length;i+=4)assert.equal(result[i],255,'the scene remains opaque');
  return result;
}

for(const inside of [()=>false,()=>true,(x)=>x>=24,(x,y)=>y>=20]){
  const input=image(inside);
  assert.deepEqual(filter(input),input.pixels,'flat images and straight silhouettes stay sharp');
}

const slope=image((x,y)=>x>=8+Math.floor(y*.5)),smoothed=filter(slope);
let coverage=0;
for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
  const i=(y*width+x)*4;
  if(smoothed[i]>16&&smoothed[i]<224)coverage++;
  if(x<5||x>30)assert.equal(smoothed[i],slope.pixels[i],'flat interiors/background remain exact');
}
assert(coverage>20,'diagonal staircase gets intermediate edge coverage');

// Strong interior highlights must not receive a whole-image blur.
const highlighted=image((x,y)=>(x+y)%2===0);
highlighted.depth.fill(.5);
assert.deepEqual(filter(highlighted),highlighted.pixels,'interior contrast remains unchanged');
console.log('Software edges: diagonal coverage, sharp straight edges, interior contrast and unchanged depth passed.');
