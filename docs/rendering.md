# Rendering

## Render panel

The Render panel (`public/js/render/render.js`, hidden by default, saved in this
browser) trades drawing quality for GPU load, applied at once: **Preset**
(High, Medium (the default; Low on a phone or tablet), Low, Ultra low; Custom once a field differs),
**Resolution** (canvas pixels per CSS pixel, at most the screen's; the
presets keep the screen's, which on an iPhone 15 made little difference),
**Anti-alias** (off, FXAA, MSAA 4×; High uses MSAA, Medium FXAA, Low and Ultra low none), **Occlusion** (SSAO off, on at half
resolution, high at full; a radius of 8 cm, of the order of a Microduck's parts: a larger one darkened a
halo round every silhouette), **Shadows** (the sun's and the scene lights', on or off; the sun's map
2048² in 3 cascades) and **Textures**
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

Below the quality settings (`public/js/render/render.js` too):

- **Look** (`public/js/render/looks.js`): **Materials**, *MuJoCo* (the
  default, saved in this browser) draws geoms as MuJoCo 4's own Filament
  renderer does (see [Materials](#materials-and-mujocos-look)), *PBR* the
  page's older way, every geom PBR with an estimated roughness; switching
  swaps the material instances at once. **Model's look**: whether the
  scene's own render settings (`filament.*`, below) apply; it names the
  model they come from, and is greyed out when no model has any.
- **Floor** (`public/js/render/floor.js`, saved in this browser, in network mode
  too since it is only this page's view): how the world floor looks, not its
  physics. Grid (the default), checker, tiles, wood and concrete (textures
  drawn in code, tileable), plain (one color) or none. Each pattern has two colors, side
  by side on the Color row (grid's base and lines, checker's cells, tiles
  and grout, light wood and grain, light and dark concrete). The pattern is worked out once per style as a mix
  between the two and a shade per pixel, so a new color only mixes the
  texture again (a few ms, live while the picker moves). Size is one cell,
  tile, plank width or pattern repeat; picking a style (or Default) starts
  from its own size, colors and shine. A floor saved with the older single
  tint loads as the two colors it looked like. **Shine** takes the floor from its style's roughness toward
  polished (0.05; checker starts at 100 %): it then reflects the sky, and the robots with
  Reflections on. Robot cameras see the floor too.
- **Lighting** (`public/js/render/lighting.js`, saved in this browser like the
  floor): the sun's brightness, azimuth, elevation and color, and the ambient
  (sky) light's brightness, live while a slider moves; Default resets them.
  **Light shadows**: how many of the scene's own lights cast shadows (none,
  1, 2 (the default), 4 or 8): the brightest of those with `castshadow`
  (MuJoCo's default), each a shadow pass more; none while Shadows is off,
  and none in VR. Their maps are mjrf's: `filament.shadows.map_size`
  (2048 by default), four cascades for a directional light.
  The ambient light is brighter from above than from below (spherical
  harmonics bands 0 and 1: the sky over a darker floor), so a side the sun
  does not reach (in shadow) still shows its shape rather than one flat grey.
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

Surfaces are drawn as MuJoCo 4 draws them (Materials: MuJoCo; see
[Materials and MuJoCo's look](#materials-and-mujocos-look)): Phong for a
material without `metallic` / `roughness`, PBR (Filament's lit model) with
them or with PBR texture layers. With Materials on PBR every geom is PBR,
a material without them getting a roughness from its classic attributes,
`shininess` (0.5 by default: 0.45) smoother, a `specular` under 0.5 more
matte, `reflectance` toward polished, and not metallic. In network mode the
server sends these fields with the render data (`server/render_asset.py`).

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

## Spawn effect

With **Spawn effect** on (the Render panel's Show, saved in this browser, off
by default), a robot or object that is spawned or reset materialises from its feet up
(`public/js/render/reveal.js`, after the Microduck simulator's scan-up): an
orange hologram of its triangles' edges rises with a scan line, and its own
surfaces follow 0.3 s behind, with a hot orange edge at the line. It takes
0.9 s for a Microduck; a taller robot's takes longer (with the square root of
its height: a G1 1.8 s), its bands wider. Environments are left out.

While it runs, the robot's parts are drawn with their own material instances
of `lit_reveal` / `lit_textured_reveal` (masked: cut above the line, their
shadow too; the shared instances of the robot file stay as they are, so its
other copies are untouched) and an additive `unlit_reveal_wire` entity each,
a child of the part's entity; transparent parts are hidden until it ends.
The lines ride with the robot's root body. In network mode Reset shows it on
the page that pressed it; a spawn shows it on every page.

## How it works

`public/js/app/main.js`: for each visible geom (group < 3) of a robot's own model a
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
runtime version exactly. npm's `filament` stopped at 1.53.4, so
`scripts/vendor.mjs` takes Filament.js from a GitHub release's web build
instead (`filament-v1.80.0-web.tgz`, checked against its hash; to move to
another version change FILAMENT there and recompile every material).
Sources are in `materials/`; to rebuild, use `matc` from the
same Filament release
(`https://github.com/google/filament/releases/download/v1.80.0/filament-v1.80.0-linux.tgz`):

```sh
matc -a opengl -p mobile -o public/materials/lit.filamat     materials/lit.mat
matc -a opengl -p mobile -o public/materials/lit_textured.filamat materials/lit_textured.mat
matc -a opengl -p mobile -o public/materials/lit_transparent.filamat materials/lit_transparent.mat
matc -a opengl -p mobile -o public/materials/lit_textured_transparent.filamat materials/lit_textured_transparent.mat
matc -a opengl -p mobile -o public/materials/unlit_textured.filamat materials/unlit_textured.mat
matc -a opengl -p mobile -o public/materials/unlit_lines.filamat materials/unlit_lines.mat
matc -a opengl -p mobile -o public/materials/lit_reveal.filamat materials/lit_reveal.mat
matc -a opengl -p mobile -o public/materials/lit_textured_reveal.filamat materials/lit_textured_reveal.mat
matc -a opengl -p mobile -o public/materials/unlit_reveal_wire.filamat materials/unlit_reveal_wire.mat
```

or all of them at once (also `phong*`, `lit_maps*`, `unlit_depth`,
`unlit_segmentation`): `scripts/build_materials.sh path/to/matc`.

## Materials and MuJoCo's look

MuJoCo 4's own Filament renderer (mjrf) draws a material that sets neither
`metallic` nor `roughness` (most MJCF) with Filament's specular-glossiness
model, MuJoCo's classic look, and PBR otherwise. With Materials on *MuJoCo*
the page does the same (`render/looks.js` geomLook):

- **Phong** (`phong*.mat`): specular = `specular` × 0.2, glossiness =
  `shininess` × 0.1 (a geom without a material: 0.5 each, as in mjrf), so
  most robots come out more matte than with the estimated PBR roughness.
  The multipliers are mjrf's defaults; a model can set its own
  (`filament.phong.specular_multiplier`, `…shininess_multiplier`).
- **PBR** (`lit*.mat`): metallic and roughness as the material sets them.
- **Texture layers** (`<material><layer texture=… role=…/>`, `lit_maps.mat`):
  normal (tangent space, OpenGL convention; the tangent frames follow the
  UVs), ORM (occlusion, roughness, metallic in r, g, b), separate
  occlusion / roughness / metallic layers (packed into one ORM image on the
  CPU, each layer's red channel) and emissive. Data layers (normal, ORM)
  are uploaded linear, colors sRGB. A material with any of them is PBR.
- **Emission**: `emission` × 0.3 (`filament.phong.emissive_multiplier`)
  times the color, as an emissive color, in every material.

A model's **render settings** are `<custom>` elements named `filament.*`,
the names mjrf reads, e.g.

```xml
<custom>
  <text name="filament.cg.tone_mapping" data="filmic"/>
  <numeric name="filament.cg.exposure" data="0.5"/>
  <numeric name="filament.fog.enabled" data="1"/>
  <numeric name="filament.fog.color" data="0.8 0.85 0.9"/>
</custom>
```

The scene takes them from an environment in it, else from the first robot
that has any: color grading (`filament.cg.*`: tone mapping `aces`,
`aces_legacy`, `filmic` or `linear`, exposure, contrast, vibrance,
saturation, temperature, tint, slope / offset / power,
shadows / midtones / highlights, curves), fog (`filament.fog.*`), bloom
(`filament.bloom.*`), vignette (`filament.vignette.*`), the AO's finer
options (`filament.ao.*`: it may turn the Render panel's AO off, not on),
the scene lights' shadow maps (`filament.shadows.map_size`).
They apply to the window, the robots' cameras and
(color grading only) the VR eyes. In network mode the server sends them
with the render data (`render_asset.py` render_settings).

What Filament.js 1.80 cannot follow: the PBR neutral tone mapper (mjrf's
default; its ColorGrading binding has no PBR_NEUTRAL, so `pbr_neutral` keeps
Filament's), shadow types other than PCF (`filament.shadows.type`: VSM and
PCSS, so soft shadows and `bulbradius`, draw no shadows at all in its WebGL
backend, even in a minimal page of its own), image lights, mjrf's planar mirror reflections (`reflectance`:
here the sky and SSR) and its object outlines.

## Relation to MuJoCo's own Filament renderer

MuJoCo's native Filament backend (`mjrfilament`, `src/render/filament/`) is
not part of the `@mujoco/mujoco` npm package (4.0.0). Its CMake has an
Emscripten branch, but it requires a native build first (to compile the
materials with matc) and then a full Emscripten build of Filament + MuJoCo.
This example instead drives Filament.js directly from JavaScript.

## Flex bodies

Flex (soft) bodies, MuJoCo's cables, cloth and soft solids, are drawn by
`public/js/render/flex.js`, ported from mjrf (`mesh_util.cc`
FillFlexVertices, `model_renderables.cc`):
- 2D: each triangle twice, front and back offset by the flex radius along
  the vertex normals, and side faces along the boundary.
- 3D: the boundary triangles, offset by the radius.
- 1D: a rod per edge and a ball per vertex (mjrf draws capsules; a box
  cross-section is drawn round).

Normals are smooth (summed around each vertex) or flat (`flex_flatskin`).
Each spawned copy has its own vertex buffer, rewritten every frame from
`flexvert_xpos`. A flex gets one material: its first element's, else
`flex_rgba`, with its texture where the flex has texture coordinates.
They follow the Materials switch, cast shadows, and are not in the robots'
depth and segmentation cameras yet.
