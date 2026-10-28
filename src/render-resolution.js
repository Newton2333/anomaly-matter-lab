/** Keep scene silhouettes at CSS-native resolution; budget only extra samples. */
export function stagePixelRatio(width,height,dpr=1,{pixelBudget=4_000_000,maxPixelRatio=1.6}={}) {
  const w=Number.isFinite(width)&&width>0?width:1;
  const h=Number.isFinite(height)&&height>0?height:1;
  const density=Number.isFinite(dpr)&&dpr>0?dpr:1;
  const ceiling=Number.isFinite(maxPixelRatio)&&maxPixelRatio>=1?maxPixelRatio:1.6;
  const budget=Number.isFinite(pixelBudget)&&pixelBudget>0?pixelBudget:4_000_000;
  return Math.max(1,Math.min(density,ceiling,Math.sqrt(budget/(w*h))));
}
