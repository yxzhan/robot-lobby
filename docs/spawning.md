# Spawning robots

The scene starts as an empty floor (`BASE_XML` in `public/js/world.js`). The
Assets panel lists every model under three tabs — Scenes (`environments/`), Robots,
Objects (`public/js/assets.js`) — after a **Recent** tab (the default): the
last 30 models spawned from the panel, newest first, kept in the browser's
localStorage; a browser without that list starts from `public/json/recent.json`
(model files relative to `public/`, edited by hand). Drag one onto the
floor to spawn it there. While
dragging, a marker on the floor shows the spawn point and heading (the model's
+x, which starts out along the world +x axis); the mouse wheel or Q/E turns it in
15° steps, Shift+wheel or R/F raises or lowers it in 5 cm steps (a pole
shows the height; it is kept for the next drag), and Esc cancels. The drag
label shows the spawn position and heading. The toolbar (top middle) runs the
simulation: Clear removes every spawned item (the Stage panel's ✕ one of
them), Reset puts every robot back in its spawn pose. In a shared world, **Full restart**
(after a confirmation, for everyone) has the server read the model files
again and start anew: a fresh MjData at time 0, every robot at its spawn
point with fresh controllers (the same ones switched on), so edits to the
MJCF files take effect without restarting the server.

**Record** (red; Stop while it records) keeps the simulation's state, 30
times a second of simulated time (qpos and the mocap poses, up to 5 minutes:
the oldest goes), and the bar under the view replays it: ▶ / ❚❚, the
timeline to drag, a speed (0.1× to 2×) and **Live**, back to the
simulation. A recording is of this tab only and in memory (`recorder.js`):
reloading drops it, and so does a spawn or removal (the world it fits is
gone); a Reset while recording goes on in the same recording. Replaying
places the bodies only (no physics, no controllers): locally the simulation
is paused meanwhile and Live puts back its state as it was; in a shared
world the server goes on for everyone and Live shows it again.

The toggles above the
left column show or hide the Assets, Stage and Physics panels (in it), those at
the top right Teleop and the other panels of the right column.

On a touch screen (phone, tablet): one finger orbits the camera, two pinch
to zoom and move together to pan; double tap selects a body, press and hold
on one, then drag, pulls it; an asset is dragged sideways out of its list
(swiping up or down scrolls it; the panel fades so it can be dropped on the
floor behind it); a second finger twisted around the dragged goal marker
turns it; and in keyboard mode two sticks at the bottom drive (left walks,
right turns). On a narrow screen the panels start hidden and show one at a
time, across the width.

Like MuJoCo's simulate: double-click a body to select it (double-click the
floor to deselect) and Shift+drag to pull it with a spring
on `xfrc_applied`. The Stage panel lists everything spawned, each with its
joint positions (settable while paused), actuator controls, a Reset button
(back to its spawn pose, at rest, while the rest keeps going) and a remove
button (`public/js/inspect.js`). Its **Pose** section places it: position (m)
and roll / pitch / yaw (deg) of its root. On a free joint that is its
qpos, set at once with no rebuild (Reset and a saved stage then use the new
pose); a fixed base (an arm, an environment) is placed by its attach frame,
heading only, so the world is rebuilt with its state kept and its mocap
bodies carried along.

**Stages** (`public/js/stage.js`): Save asks for a name and which parts
to keep, and saves what is spawned as a new stage: each item's spawn pose, joint positions,
actuator controls and mocap bodies (by name, so a changed model loads what
still matches), the controllers switched on with their IK and base setups,
the camera setups and which cameras are on, the Physics panel's physics, the
view, and a 256×256 picture of it. Each of joint state, controllers on,
IK and base setups, cameras, physics and view can be left out (the dialog
remembers the choice; Publish asks the same): a stage loads without them,
leaving the physics, the view and the setups as they are and the models in
their spawn pose. A stage is a folder named by a random id
(the name is inside, so names may repeat): `stage.json`, `thumbnail.jpg`,
`ik.json`, `base.json`, `cameras.json` (keyed like `ik_defaults.json`,
`base_defaults.json`, `camera_defaults.json`). Saved stages are **My
stages**: an IndexedDB record of those files in this browser, in either
mode (on a server, the server makes the stage of its world and the page
keeps it). **Publish** (on a server only) shares what is spawned, or one of
My stages, as a **Community** stage: it asks for the name and, optionally,
an author, and the server writes it into its library,
`mjcf_assets/saved_stage/<id>/` (a new id: never over another's), with the
author's name (`server.py` `publish_stage`). It is not committed: a new
`saved_stage/<id>/` folder in the library's checkout, committed (and
pushed) from there by hand. In local mode
Community is the library's own `saved_stage/` (read-only;
`scripts/vendor.mjs` indexes it into `public/json/stages.json`). Either list
loads a stage (it replaces what is spawned, in one compile, each item in
its saved state, which Reset then goes back to; in a shared world for
everyone: one of My stages is sent to the server whole) and downloads it as
one `.stage.json` file, which **Import** adds to My stages; Community stages
also copy a link (`?stage=<id>`), and My stages can be deleted. A page
load starts from an empty floor (in local mode; a server keeps its world),
unless a link names a stage.
Picking uses `mj_ray` on the physics world, so it hits collision geoms.

- **Asset list** — the model library is `mjcf_assets/` next to
  `package.json`, a git repository of its own (the library repository: the
  models, saved stages, menagerie as a submodule, and in its `scripts/` the
  conversions that made them, `urdf2mjcf.py`, `usd2mjcf.py`,
  `basic_scenes.py`, …; see its README), laid out as `<category>/<source>/<model>/`: the category
  (`environments`, `robots`, `objects`) is the panel tab, the source
  (`aicor`, `mujoco_menagerie`, `robocasa`, …) where the models come
  from. Naming convention: a model directory `<name>/` holds `<name>.xml`,
  the file that gets spawned, and optionally `<name>.jpg`, its screenshot (500 px wide; a `.png` also does);
  directories without `<name>.xml` are skipped with a warning. Upstream
  `mujoco_menagerie` keeps its own names, so it is indexed from its README's
  "Menagerie Models" table: one model per row, spawning the model file the
  row's scene includes (so the scene's floor and lights stay out), with the
  row's image as screenshot. Each further `scene_<suffix>.xml` of a model
  directory (left/right hands, `_with_hands`, actuator variants, …)
  adds the model it includes as `<dir>_<suffix>` in the same group, with the
  matching `.png` if there is one, else the directory's only `.png`, else the
  base model's. Directories the table does not list go under "Others". MJX
  models (scene or model file named `*mjx*`) are left out: they collide only
  through contact pairs with their scene's own floor, so they would fall
  through the viewer's.
  `npm install` (or `node scripts/vendor.mjs`) links the library as
  `public/mjcf_assets` and indexes it into `public/assets.json`. Within a tab, models are listed under their
  source, `aicor` first and the others by name; menagerie robots are
  split by the table's sections (Humanoids, Quadrupeds, Arms, …); `{"group": "…"}` in an `asset.json` in
  the model directory overrides the group. Re-run
  `node scripts/vendor.mjs` after adding models.
- **Thumbnails** — `python3 scripts/thumbnails.py` (after `vendor.mjs`)
  compresses each model's screenshot (`<name>.jpg`) into
  `mjcf_assets/thumbs/<category>/<source>/<model>.webp` (≤ 320×240; the 4K
  menagerie renders go from ~84 MB to ~0.4 MB), in the library so that they
  go wherever it goes (a Binder session's volume keeps them). The originals are untouched;
  models without a screenshot show their initials. A `<name>.jpg` added after
  the index was written is picked up too, and models with no image at all
  are listed at the end.
- **Objects from RoboCasa** — `python3 scripts/import_robocasa.py
  [/path/to/robocasa]`, in the library repository (below), copies the picked
  objaverse instances (`PICKS` in the script, named `<category>_<n>` like
  RoboCasa's directories) into `mjcf_assets/objects/robocasa/<name>/`, writing `<name>.xml` with a free
  joint (raised so the object rests on the floor) and collision geoms moved
  to group 3. Models without a screenshot get one from
  `node scripts/render_previews.mjs [category[/source[/name or group]]…]`,
  which spawns each alone in headless Chrome (needs the page served, e.g.
  `npm start`) and writes `<name>.jpg` (500 px wide) next to it.
- **Loading** — Spawning fetches only the files it
  references — the `<include>` tree, then the mesh/texture/hfield/skin files
  resolved via `<compiler meshdir/texturedir/assetdir>` — into MuJoCo's
  in-memory filesystem (`public/js/vfs.js`).

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
with an "Attach conflict" warning in the console; set them in the Physics
panel (a policy that needs them offers to).

## Memory: physics-only world, render data per robot file

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

A keyframe made for the robot's own scene can put it below our floor
(TIAGo's sits at z = -0.985 on a floor at -0.99); a spawn whose collision
geoms reach more than 10 cm below the spawn height is raised to clear it.

Known gaps: `shadow_dexee` needs the `mujoco.pid` plugin, which the WASM build
does not ship; `iit_softfoot` (an `<asset><model>` sub-model) fails to attach;
`tetheria_aero_hand_open` loses 6 of its 7 tendon actuators in `mjs_attach`
(spawning reports the mismatch). Skyboxes, cube textures and hfields are not
drawn.
