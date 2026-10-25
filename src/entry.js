import { launchExperience } from './launch.js';
launchExperience({userAgent:navigator.userAgent,platform:navigator.platform,
  maxTouchPoints:navigator.maxTouchPoints,mobileHint:navigator.userAgentData?.mobile});
