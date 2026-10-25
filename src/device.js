// Test device identity, not viewport width: a narrow desktop window still runs.
export function isHandheld({userAgent='',platform='',maxTouchPoints=0,mobileHint=false}={}) {
  return mobileHint || /Android|iPhone|iPad|iPod|IEMobile|Opera Mini/i.test(userAgent)
    || (/Mac/.test(platform) && maxTouchPoints > 1);
}
