# MuJoCo + Filament on the web

Minimal example: physics from the official MuJoCo WASM bindings
(`@mujoco/mujoco`), rendering with Filament.js (WebGL2, PBR, sun shadows, SSAO).
No bundler — `public/` is a static site.

```sh
npm install      # postinstall copies the wasm/js runtimes into public/vendor/
npm start        # python3 scripts/serve.py 4000: public/ with Cache-Control: no-store
```

Open http://localhost:4000 (in JupyterHub: `.../user/<name>/proxy/4000/` — all
URLs are relative, so the path proxy works). Left-drag orbits, right-drag pans,
wheel zooms. The gizmo in the bottom-left corner shows the world axes (Z up);
click an axis to look along it (`public/gizmo.js`). The meter in the
bottom-right corner is stats.js: click it to cycle FPS / ms per frame / MB.

## Spawning robots

The scene starts as an empty floor (`BASE_XML` in `public/world.js`). The
Assets panel lists every model under three tabs — Environments, Robots,
Objects (`public/assets.js`); drag one onto the floor to spawn it there. While
dragging, a marker on the floor shows the spawn point and heading (the model's
+x, which starts out along the world +x axis); the mouse wheel or Q/E turns it in
15° steps, Shift+wheel or R/F raises or lowers it in 5 cm steps (a pole
shows the height; it is kept for the next drag), and Esc cancels. The drag
label shows the spawn position and heading. The toolbar (top left) runs the
simulation: Undo removes the last one, Clear all of them, Reset puts every
robot back in its spawn pose. The toggles at the top right show or hide the
Assets, Stage and Control panels.

Like MuJoCo's simulate: double-click a body to select it (double-click the
floor to deselect) and Shift+drag to pull it with a spring
on `xfrc_applied`. The Stage panel lists everything spawned, each with its
joint positions (settable while paused), actuator controls, a Reset button
(back to its spawn pose, at rest, while the rest keeps going) and a remove
button (`public/inspect.js`). Save stage remembers what is spawned, with
its spawn position and heading (not its joint state), in the browser's
localStorage; the next page load spawns it all again in one compile (Forget
starts from an empty floor again).
Picking uses `mj_ray` on the physics world, so it hits collision geoms.

- **Asset list** — one entry per model directory with a `scene.xml` in each
  library checked out next to `package.json`: `my_robot/` (Robots, listed
  first under "My robots"), `mujoco_menagerie/` (Robots),
  `my_environment/` (Robots if its `scene.xml` includes a model file,
  Environments otherwise) and `my_objects/` (Objects). `npm install` (or
  `node scripts/vendor.mjs`) symlinks each into `public/` and writes
  `public/<name>.json`, whose `main` is the model file `scene.xml` includes;
  that file is what gets spawned, so the scene's own floor and lights stay
  out, while a `scene.xml` without an `<include>` (a whole environment) is
  spawned as is. An `asset.json` in a model directory, e.g.
  `{"category": "object", "main": "mug.xml"}`, overrides the tab and entry
  file. Menagerie robots are grouped (Humanoids, Quadrupeds, Arms, …) as in
  the "Menagerie Models" section of `mujoco_menagerie/README.md`, read by
  `vendor.mjs`; `"group": "…"` in `asset.json` sets one for any model, and
  ungrouped models in a grouped tab go under "Other". Re-run
  `node scripts/vendor.mjs` after adding models.
- **Thumbnails** — `python3 scripts/thumbnails.py` (after `vendor.mjs`)
  compresses each model's screenshot (the image in its directory named like
  its entry file, or `"thumbnail"` in `asset.json`) into
  `public/thumbs/<library>/<model>.webp` (≤ 320×240; the 4K menagerie renders
  go from ~84 MB to ~0.4 MB). The originals are untouched; models without a
  screenshot show their initials.
- **Objects from RoboCasa** — `python3 scripts/import_robocasa.py
  [/path/to/robocasa]` copies one instance of each picked kitchen category
  (`PICKS` in the script) into `my_objects/`, adding a free joint (raised so
  the object rests on the floor) and moving collision geoms to group 3. Assets
  without a screenshot get one from `node scripts/render_previews.mjs
  [library…]`, which spawns each alone in headless Chrome (needs the page
  served, e.g. `npm start`) and writes `preview.png` next to it.
- **Loading** — Spawning fetches only the files it
  references — the `<include>` tree, then the mesh/texture/hfield/skin files
  resolved via `<compiler meshdir/texturedir/assetdir>` — into MuJoCo's
  in-memory filesystem (`public/vfs.js`).

A compiled `MjModel` cannot grow, so each spawn rebuilds the world: parse
`BASE_XML` into an `MjSpec`, `mjs_attach` every robot's spec under a frame at
its click point with a unique prefix (`r1/`, `r2/`, …), `mj_compile`. Robots
are attached in order, so the existing ones keep their state (each robot's
qpos/qvel/act/ctrl/mocap entries are a contiguous range sized by its own
model, copied over to wherever they land when one is removed); the new one
starts from
its own first keyframe (qpos + ctrl, free joints shifted to the click point)
or `qpos0`. Compile time grows with the number of robots (~3 s per spawn
with a dozen menagerie robots). Physics `<option>`s come from `BASE_XML`: a
robot's own (e.g. the quadrupeds' elliptic cones / `impratio`) is dropped
with an "Attach conflict" warning in the console.

### Memory: physics-only world, render data per robot file

The `@mujoco/mujoco` WASM heap is capped at 2 GB and never shrinks, and
compiling menagerie's dense visual meshes (normals, BVHs, hulls) is
expensive — Apollo alone peaks at ~1.5 GB, G1 at ~450 MB. On top of that,
`MjSpec.delete()` does not return what a spec and its compile used, so
recompiling the world in one instance leaked hundreds of MB per spawn
("Cannot enlarge memory … limit is 2147483648 bytes" after 4–5 robots).
Hence:

- **Render data** is built once per robot file: it is compiled on its own in
  a throwaway MuJoCo instance, its visible geoms become Filament buffers and
  material instances (with their pose in their body), plus its body
  inertias, first keyframe and referenced names are kept. Spawned copies
  share all of it.
- **The physics world** is rebuilt in a fresh MuJoCo instance on every spawn
  (the old one is garbage-collected as a whole). Before attaching, each
  robot's spec loses its visual geoms (`contype`/`conaffinity` 0, unless
  something refers to them by name) and the meshes only they use; every
  body gets the full model's mass/inertia as an explicit `<inertial>`, so
  the dynamics are unchanged (checked for all menagerie robots). The world
  is typically 10–250 MB per robot instead of up to 1.5 GB.
- **Each frame** a geom's transform is its world body pose (`xpos`/`xmat`)
  times its pose in the body. A robot's body k > 0 is world body
  `bodyOffset + k − 1` (robots are attached in order).
- Fetched files are kept in JS memory and shared (not copied) with every
  instance's filesystem (`FS.writeFile(…, {canOwn: true})`).

Side effect: visual geoms are not in the physics model, so ray-based
sensors (rangefinders) no longer hit them.

Known gaps: `shadow_dexee` needs the `mujoco.pid` plugin, which the WASM build
does not ship; `iit_softfoot` (an `<asset><model>` sub-model) fails to attach. Skyboxes, cube textures and hfields are not drawn.

## Walking G1 (GR00T policy)

If the GR00T locomotion policies are checked out next to `package.json`
(`git clone https://huggingface.co/nepyope/GR00T-WholeBodyControl_g1`),
`npm install` links them as `public/groot`, and every spawned `unitree_g1`
walks under them. The Control panel (`public/control.js`) picks which one
is driven (the newest by default, or the one double-clicked; double-clicking
anything else releases it; the others get a zero command and stand)
and, later, by which controller and input; with the keyboard: **WASD** walk,
**QE** turn, hold **R**/**F** to raise or lower the base (keys typed into a
text field are ignored). `public/groot.js` is a port of lerobot's
`GrootLocomotionController` running on onnxruntime-web (WASM, one thread,
~0.2 ms per inference): two MLPs (Balance when the command is ~0, Walk
otherwise) at 50 Hz on 6 stacked 86-D observations, driving legs and waist,
with the arms held at the default pose. The policy's PD gains are written
into the menagerie position actuators (`gainprm`/`biasprm`), and the IMU is
the floating base (`qpos[3:7]`, `qvel[3:6]`), as in lerobot's G1 sim.
Inference is async, so `App.advance()` awaits it every 0.02 s of sim time.

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
flipped/grid textures still come from `tex_data`. Meshes use their `mesh_texcoord`s; primitives
(e.g. a textured table box, the groundplane) get an object-space planar
projection per face. `texrepeat`/`texuniform` are baked into the UVs. Planes
without any material get the built-in checker; otherwise the material color.

## How it works

`public/main.js`: for each visible geom (group < 3) of a robot's own model a
Filament renderable is built once, with vertices baked from `geom_size`
(plane, sphere, capsule, ellipsoid, cylinder, box, mesh; hfield/sdf skipped).
Meshes use MuJoCo's per-corner normals (`mesh_facenormal` → `mesh_normal`),
so they shade smoothly with creases where MuJoCo (or the OBJ file) put them.
Each frame it runs `mj_step` on the world until sim time catches up with wall
time, then composes body poses with the geoms' local poses into the Filament
transforms (see above). The scene stays in MuJoCo's Z-up frame (camera
`up = +Z`).

## Materials

Filament.js can only load precompiled `.filamat` files, and they must match the
runtime version exactly — that is why `filament` is pinned to `1.53.4` (the
latest on npm). Sources are in `materials/`; to rebuild, use `matc` from the
same Filament release
(`https://github.com/google/filament/releases/download/v1.53.4/filament-v1.53.4-linux.tgz`):

```sh
matc -a opengl -p mobile -o public/materials/lit.filamat     materials/lit.mat
matc -a opengl -p mobile -o public/materials/checker.filamat materials/checker.mat
matc -a opengl -p mobile -o public/materials/lit_textured.filamat materials/lit_textured.mat
```

## Relation to MuJoCo's own Filament renderer

MuJoCo's native Filament backend (`mjrfilament`, `src/render/filament/`) is
not part of the `@mujoco/mujoco` npm package (3.15.0). Its CMake has an
Emscripten branch, but it requires a native build first (to compile the
materials with matc) and then a full Emscripten build of Filament + MuJoCo.
This example instead drives Filament.js directly from JavaScript.
