# ANOMALY

A browser-based fluid experiment by **Newton Quan**.

[Live demo](https://anomaly-matter-lab.vercel.app) · Desktop browser · English / 中文

Three visual styles: mercury, glass and stardust. Move the pointer to attract, hold to gather, release to scatter.

流体交互实验，支持水银、玻璃和星尘三种效果。移动鼠标牵引，长按聚合，松手散开。

## Run locally

```sh
npm ci
npm run dev
```

Build with `npm run build`. Run the interaction checks with `npm run check`.

## Credits

[Three.js](https://threejs.org/) · [Three.js Particle Fluids](https://github.com/dgreenheck/threejs-particle-fluids) by Daniel Greenheck · [Position Based Fluids](https://blog.mmacklin.com/project/pbf/) by Miles Macklin and Matthias Müller

The GPU fluid code is adapted from Three.js Particle Fluids. Third-party license notices are in `public/licenses/`.
