# MuJoCo + Filament on the web

Minimal example: physics from the official MuJoCo WASM bindings
(`@mujoco/mujoco`), rendering with Filament.js (WebGL2, PBR, sun shadows, SSAO).
No bundler — `public/` is a static site.

```sh
npm install      # postinstall copies the wasm/js runtimes into public/vendor/
npm start        # python3 -m http.server 8000 -d public
```

Open http://localhost:8000 (in JupyterHub: `.../user/<name>/proxy/8000/` — all
URLs are relative, so the path proxy works). Left-drag orbits, right-drag pans,
wheel zooms.

## Spawning robots

The scene starts as an empty floor (`BASE_XML` in `public/world.js`). Pick a
robot in the drop-down and click the floor (a click, not a drag) to drop it
there; Undo removes the last one, Clear all of them, Reset puts every robot
back in its spawn pose.

- **Robot list** — one entry per `mujoco_menagerie/` robot that has a
  `scene.xml` (if menagerie is checked out next to `package.json`).
  `npm install` (or `node scripts/vendor.mjs`) symlinks it to
  `public/menagerie` and writes `public/menagerie.json`, whose `main` is the
  robot file `scene.xml` includes; that file is what gets spawned, so the
  scene's own floor and lights stay out. Spawning fetches only the files it
  references — the `<include>` tree, then the mesh/texture/hfield/skin files
  resolved via `<compiler meshdir/texturedir/assetdir>` — into MuJoCo's
  in-memory filesystem (`public/vfs.js`).

A compiled `MjModel` cannot grow, so each spawn rebuilds the world: parse
`BASE_XML` into an `MjSpec`, `mjs_attach` every robot's spec under a frame at
its click point with a unique prefix (`r1/`, `r2/`, …), `mj_compile`. Robots
are attached in order, so the existing ones keep their state (their
qpos/qvel/act/ctrl are a prefix of the new model's); the new one starts from
its own first keyframe (qpos + ctrl, free joints shifted to the click point)
or `qpos0`. Compile time grows with the number of robots (~1–2 s each
time for a handful of menagerie robots). Physics `<option>`s come from
`BASE_XML`: a robot's own (e.g. the quadrupeds' elliptic cones / `impratio`)
is dropped with an "Attach conflict" warning in the console.

Known gaps: `shadow_dexee` needs the `mujoco.pid` plugin, which the WASM build
does not ship. Skyboxes, cube textures and hfields are not drawn.

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

`public/main.js`: for each visible geom (group < 3) a Filament renderable is
built once, with vertices baked from `geom_size` (plane, sphere, capsule,
ellipsoid, cylinder, box, mesh; hfield/sdf skipped). Meshes use MuJoCo's
per-corner normals (`mesh_facenormal` → `mesh_normal`), so they shade smoothly
with creases where MuJoCo (or the OBJ file) put them. Each frame it runs
`mj_step` until sim time catches up with wall time, then copies
`geom_xpos`/`geom_xmat` into the Filament transforms. The scene stays in
MuJoCo's Z-up frame (camera `up = +Z`).

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
