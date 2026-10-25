# ANOMALY

An interactive experiment in matter, light and motion by **Newton Quan**.

[Open the experiment](https://anomaly-matter-lab.vercel.app) · [Source](https://github.com/Newton2333/anomaly-matter-lab)

![ANOMALY](preview.jpg)

Move your cursor to draw the matter closer. Hold the left mouse button to gather it; release to scatter. Mercury, liquid glass and stardust share a simulated particle system, with a floating ring and a procedural studio environment.

The experience opens in English and offers a complete Chinese translation. Activity starts at 0%. Materials and the first scatter are prepared before the loading cover lifts. Rendering detail adapts to device load. Phones and tablets show a lightweight desktop notice.

## Controls

| Action | Control |
| --- | --- |
| Attract matter | Move the cursor over the scene |
| Gather and release | Hold and release the left mouse button or Space |
| Change material | 1 / 2 / 3 |
| Pause or resume | P |
| Regroup | R |
| Immersive view | H; Escape to leave |

Settings include activity, camera orbit, pause, image capture and fullscreen.

## Run locally

```sh
npm ci
npm run dev
```

```sh
npm run check
npm run build
```

Vite writes the static website to `dist/`. The included `vercel.json` supports a source deployment to Vercel. Hardware WebGPU is preferred, with WebGL and software compatibility paths.

## Credits and sources

| Contribution | Source |
| --- | --- |
| Project creation | Newton Quan |
| GPU particle simulation and reconstructed liquid surface | [Three.js Particle Fluids](https://github.com/dgreenheck/threejs-particle-fluids), Daniel Greenheck — MIT |
| Rendering and CPU Marching Cubes surface | [Three.js r184](https://threejs.org/), three.js authors — MIT |
| Simulation foundations | [Position Based Fluids](https://blog.mmacklin.com/project/pbf/), Miles Macklin and Matthias Müller, 2013 |
| Build tools | Vite and TypeScript; versions in `package-lock.json` |

The CPU compatibility solver, software renderer, interaction controls, bilingual interface and procedural scene are maintained in this project. The vendored GPU library is adapted for this experiment; its original copyright, license and research references are retained. No models, textures or fonts from the upstream demonstration are included.

Third-party licenses are included in `public/licenses/` and `src/vendor/particle-fluids/`. The upstream notices also describe assets used by its own demo; those assets are not part of ANOMALY. Third-party licenses apply to their respective components.

This is an artistic simulation, not a scientifically calibrated material model.

## 中文

ANOMALY 是 Newton Quan 创作的交互物质实验。移动鼠标牵引，长按聚拢，松手扩散；在水银、玻璃与星尘之间切换。默认活跃度为 0%，右上角可切换中英文。请使用电脑浏览器打开。
