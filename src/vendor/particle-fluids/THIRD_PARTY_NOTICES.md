# Third-party notices

The MIT license at the repository root applies to original project code. It does not replace dependency licenses.

The demo uses Three.js (MIT). The lighting environment, icon, and preset preview images are generated locally. No external fonts or research papers are bundled. Model textures are included under the asset licenses below.

Vite, TypeScript, Vitest, Playwright, ESLint, and the other development dependencies retain the notices shipped with their packages. Exact versions are recorded in `package-lock.json`. Preserve applicable dependency notices when redistributing a bundled application.

Research cited in source comments belongs to the respective authors and publishers. Citations explain the algorithms; the papers themselves are not redistributed under this project's license.

## Elastic study meshes

The doughnut, croissant, banana, and gingerbread meshes come from **Kenney Food Kit 2.0**, created by [Kenney](https://kenney.nl/assets/food-kit) and released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). The original asset license is included at `demo/assets/elastic/LICENSE.txt`.

The four source OBJ/MTL files and original color atlas are in `demo/assets/elastic/`. The demo versions in `public/models/elastic/` remove duplicate faces, weld and subdivide the surface, adjust thickness on the banana and gingerbread, normalize scale, preserve the texture coordinates and color atlas, and add connected particle templates. Run `npm run assets:elastic` to reproduce them. The source and derived meshes retain their CC0 status.

## Buoyancy rubber ducks

`public/models/buoyancy/rubber-duck.glb` is the [Rubber Duck Toy](https://polyhaven.com/a/rubber_duck_toy) by **Plat251**, distributed by Poly Haven under [CC0 1.0 Universal](https://polyhaven.com/license). The embedded 1K color, normal, and material textures are retained. The demo scales and rotates the model and samples its volume for buoyancy; the asset remains CC0.

## Honey bunny

`demo/assets/honey/bunny.drc` is the **Stanford Bunny** from the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) (Stanford University Computer Graphics Laboratory), in the Draco-compressed copy distributed with the three.js examples. Stanford permits free use of the model and asks that it be credited. Run `npm run assets:honey` to rebuild `public/models/honey/`: the script decodes the mesh, caps the five holes in its base, scales it to 42 cm, writes a quantized display mesh, and bakes a collision distance field from a simplified copy.
