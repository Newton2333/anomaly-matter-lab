# ANOMALY

A small experiment in fluid, light and motion by **Newton Quan**.

[Live demo](https://anomaly-matter-lab.vercel.app) · Desktop browser · English / 中文

Pull a little, hold to gather, then let go. Switch between mercury, glass and stardust, adjust the activity, or pause to save a frame.

一个可以随手玩的流体小实验。水银流动、玻璃折光、星尘聚散：移动鼠标牵引，长按聚拢，松手让它散开。也可以调整活跃程度，或暂停留下一张截图。

## Run locally

```sh
npm ci
npm run dev
```

Build with `npm run build`. Run the interaction checks with `npm run check`.

## Credits

[Three.js](https://threejs.org/) · [Three.js Particle Fluids](https://github.com/dgreenheck/threejs-particle-fluids) by Daniel Greenheck · [Position Based Fluids](https://blog.mmacklin.com/project/pbf/) by Miles Macklin and Matthias Müller

The GPU fluid code is adapted from Three.js Particle Fluids. Third-party license notices are in `public/licenses/`.
