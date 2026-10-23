# Rendering

## Render panel

The Render panel (`public/js/render.js`, hidden by default, saved in this
browser) trades drawing quality for GPU load, applied at once: **Preset**
(High, Medium (the default; Low on a phone or tablet), Low, Ultra low; Custom once a field differs),
**Resolution** (canvas pixels per CSS pixel, at most the screen's; the
presets keep the screen's, which on an iPhone 15 made little difference),
**Anti-alias** (off, FXAA, MSAA 4×; High uses MSAA, Medium FXAA, Low and Ultra low none), **Occlusion** (SSAO off, on at half
resolution, high at full), **Shadows** (the sun's, on or off: this Filament
build binds no shadow-map options, so their size is fixed) and **Textures**
(64, 512, 1024 or 2048 px on the longer side; a change decodes the scene's
images again, so it can be compared on the spot: a kitchen's 18 textures
are 63 MB at 1024 and 0.3 MB at 64). Ultra low is Low with 64 px textures,
to see what textures cost. The WebXR eyes leave these out: no FXAA, SSAO
or shadows (they come out different in each eye and shimmer as the head
moves), MSAA instead (4× unless the VR menu says otherwise), and shadows
only if the VR menu turns them on; Filament renders them straight into the
session's layer, and renders every session frame. **Enter AR** (where
`immersive-ar` is supported, e.g. Quest 3) is the same with passthrough:
no sky and no world floor, so the robots stand in your room, on its floor.

Below the quality settings (`public/js/render.js` too):

- **Floor** (`public/js/floor.js`, saved in this browser, in network mode
  too since it is only this page's view): how the world floor looks, not its
  physics. Checker, grid, tiles, wood and concrete (textures drawn in code,
  tileable), plain (one color) or none. Each pattern has two colors, side
  by side on the Color row (checker's cells, grid's base and lines, tiles
  and grout, light wood and grain, light and dark concrete). The pattern is worked out once per style as a mix
  between the two and a shade per pixel, so a new color only mixes the
  texture again (a few ms, live while the picker moves). Size is one cell,
  tile, plank width or pattern repeat; picking a style (or Default) starts
  from its own size, colors and shine. A floor saved with the older single
  tint loads as the two colors it looked like. **Shine** takes the floor from its style's roughness toward
  polished (0.05; checker starts at 100 %): it then reflects the sky, and the robots with
  Reflections on. Robot cameras see the floor too.
- **Lighting** (`public/js/lighting.js`, saved in this browser like the
  floor): the sun's brightness, azimuth, elevation and color, and the ambient
  (sky) light's brightness, live while a slider moves; Default resets them.
- **Reflections**: the sky is also the scene's reflection map (IBL), its mip
  levels prefiltered in code for each roughness, so metal and shiny surfaces
  reflect it everywhere (VR and robot cameras included) at no extra cost.
  Screen-space reflections (mirroring the robots on a shiny floor) are the
  Reflections setting above, on in the High preset and off in VR;
  only the opaque lit materials take part (`reflections : screenspace` in
  `lit.mat` and `lit_textured.mat`).
- **Camera** (last, saved in this browser): *Follow selection*, off by
  default. A double-click selects a body and the camera turns to it once;
  with this on it keeps following it (orbit, zoom and pan go on, a pan
  offsetting the view) until the selection is cleared or changes, and holds
  still while something is dragged through the pointer (a pulled body, an IK
  or goal marker).

## Textures

Geoms whose material has a 2D RGB texture (`mat_texid` role
`mjTEXROLE_RGB`) use `lit_textured`: an sRGB Filament texture with mipmaps,
multiplied by the material rgba as in MuJoCo.

Image files never reach MuJoCo: the `@mujoco/mujoco` WASM heap is capped at
2 GB and MuJoCo would decode every PNG into it (large scenes then fail with
"Could not allocate memory"), although physics never reads them. So the
loader rewrites each plain `<texture type="2d" file=…>` into a 1×1 builtin of
the same name, and the renderer decodes the image itself with
`createImageBitmap` and swaps it in when ready. Builtin, cube and
flipped/grid textures still come from `tex_data`. Meshes use their `mesh_texcoord`s (v
flipped: MuJoCo counts it from the image's top row, Filament's materials flip it); primitives
(e.g. a textured table box, the groundplane) get an object-space planar
projection per face. `texrepeat`/`texuniform` are baked into the UVs. Planes
without any material get the built-in checker; otherwise the material color.

Surfaces are PBR (Filament's lit model): a material's `metallic` and
`roughness` (MuJoCo 3.2+) are used as they are; a material without them gets
a roughness from its classic attributes, `shininess` (0.5 by default: 0.45)
smoother, a `specular` under 0.5 more matte, `reflectance` toward polished,
and is not metallic. Texture layers other than `rgb` (orm, normal, …) are not
drawn yet. In network mode the server sends these fields with the render
data (`server/render_asset.py`).

GPU memory is what limits a scene, and textures take most of it (RGBA8 plus
mipmaps: a 2048² image is 21 MB). So images are scaled down on their
longer side (to 1024 px by default: the Render panel's Textures), one
texture per image file is shared by every model that uses it (a RoboCasa kitchen names its 19 images 69 times), and a model's
buffers and textures are freed once no spawned copy uses it. One kitchen went
from 452 to 84 MB of textures; two kitchens, the Garmi apartment and an
ANYmal from 1.9 GB to 0.5 GB. If the page still loses its WebGL context, it
says so (reload); Chrome may then keep WebGL off for the page for a while,
which the page reports at start-up instead of failing in Filament.

Geoms whose rgba alpha is below 1 (glass doors, …) use the alpha-blended
variants `lit_transparent` / `lit_textured_transparent` (the texture's own
alpha multiplies in) and cast no shadow; alpha 0 hides a geom, as in MuJoCo.
Back faces are culled like MuJoCo does, except in environments
(`mjcf_assets/environments/`), which draw both sides: game-engine scenes
model walls one-sided, facing in, so from outside they would vanish (the lit
materials are compiled `doubleSided`, their instances set one-sided by default). Filament sorts transparent
renderables back to front per renderable, so a self-overlapping or
interpenetrating transparent mesh can blend in the wrong order.

## How it works

`public/js/main.js`: for each visible geom (group < 3) of a robot's own model a
Filament renderable is built once, with vertices baked from `geom_size`
(plane, sphere, capsule, ellipsoid, cylinder, box, mesh; hfield/sdf skipped).
Meshes use MuJoCo's per-corner normals (`mesh_facenormal` → `mesh_normal`),
so they shade smoothly with creases where MuJoCo (or the OBJ file) put them.
Each frame it runs `mj_step` on the world until sim time catches up with wall
time, then composes body poses with the geoms' local poses into the Filament
transforms (see [Memory](spawning.md#memory-physics-only-world-render-data-per-robot-file)). The scene stays in MuJoCo's Z-up frame (camera
`up = +Z`).

## Materials

Filament.js can only load precompiled `.filamat` files, and they must match the
runtime version exactly — that is why `filament` is pinned to `1.53.4` (the
latest on npm). Sources are in `materials/`; to rebuild, use `matc` from the
same Filament release
(`https://github.com/google/filament/releases/download/v1.53.4/filament-v1.53.4-linux.tgz`):

```sh
matc -a opengl -p mobile -o public/materials/lit.filamat     materials/lit.mat
matc -a opengl -p mobile -o public/materials/lit_textured.filamat materials/lit_textured.mat
matc -a opengl -p mobile -o public/materials/lit_transparent.filamat materials/lit_transparent.mat
matc -a opengl -p mobile -o public/materials/lit_textured_transparent.filamat materials/lit_textured_transparent.mat
matc -a opengl -p mobile -o public/materials/unlit_textured.filamat materials/unlit_textured.mat
```

## Relation to MuJoCo's own Filament renderer

MuJoCo's native Filament backend (`mjrfilament`, `src/render/filament/`) is
not part of the `@mujoco/mujoco` npm package (3.15.0). Its CMake has an
Emscripten branch, but it requires a native build first (to compile the
materials with matc) and then a full Emscripten build of Filament + MuJoCo.
This example instead drives Filament.js directly from JavaScript.
