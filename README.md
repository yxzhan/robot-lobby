# Robot Lobby

An unofficial multiplayer MuJoCo sandbox in the browser: spawn robots,
drive them, pull them around, in one physics world shared by everyone
connected (a Python server, e.g. on Binder), or alone with the physics in
the page.

Physics from the official MuJoCo WASM bindings
(`@mujoco/mujoco`), rendering with Filament.js (WebGL2, PBR, sun shadows, SSAO).
No bundler — `public/` is a static site.

```sh
npm install      # postinstall copies the wasm/js runtimes into public/vendor/
npm start        # python3 scripts/serve.py 4000: public/ with Cache-Control: no-store
```

Open http://localhost:4000 (in JupyterHub: `.../user/<name>/proxy/4000/` — all
URLs are relative, so the path proxy works). Left-drag orbits, right-drag pans,
wheel zooms. The gizmo in the bottom-left corner shows the world axes (Z up);
click an axis to look along it (`public/gizmo.js`); the same colored axes
(30 cm) mark the world origin on the floor. The meter in the
bottom-right corner is stats.js: click it to cycle FPS / ms per frame / MB.

## Spawning robots

The scene starts as an empty floor (`BASE_XML` in `public/world.js`). The
Assets panel lists every model under three tabs — Environments, Robots,
Objects (`public/assets.js`) — after a **Recent** tab (the default): the
last 30 models spawned from the panel, newest first, kept in the browser's
localStorage; a browser without that list starts from `public/recent.json`
(model files relative to `public/`, edited by hand). Drag one onto the
floor to spawn it there. While
dragging, a marker on the floor shows the spawn point and heading (the model's
+x, which starts out along the world +x axis); the mouse wheel or Q/E turns it in
15° steps, Shift+wheel or R/F raises or lowers it in 5 cm steps (a pole
shows the height; it is kept for the next drag), and Esc cancels. The drag
label shows the spawn position and heading. The toolbar (top left) runs the
simulation: Undo removes the last one, Clear all of them, Reset puts every
robot back in its spawn pose. The toggles at the top right show or hide the
Assets, Stage and Teleop panels.

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
button (`public/inspect.js`). Its **Pose** section places it: position (m)
and roll / pitch / yaw (deg) of its root. On a free joint that is its
qpos, set at once with no rebuild (Reset and a saved stage then use the new
pose); a fixed base (an arm, an environment) is placed by its attach frame,
heading only, so the world is rebuilt with its state kept and its mocap
bodies carried along. Save stage remembers what is spawned, with
its spawn position and heading (not its joint state), in the browser's
localStorage; the next page load spawns it all again in one compile (Forget
starts from an empty floor again).
Picking uses `mj_ray` on the physics world, so it hits collision geoms.

- **Asset list** — the model library is `mjcf_assets/` next to
  `package.json`, laid out as `<category>/<source>/<model>/`: the category
  (`environments`, `robots`, `objects`) is the panel tab, the source
  (`digital_twin`, `mujoco_menagerie`, `robocasa`, …) where the models come
  from. Naming convention: a model directory `<name>/` holds `<name>.xml`,
  the file that gets spawned, and optionally `<name>.png`, its screenshot;
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
  source, `digital_twin` first and the others by name; menagerie robots are
  split by the table's sections (Humanoids, Quadrupeds, Arms, …); `{"group": "…"}` in an `asset.json` in
  the model directory overrides the group. Re-run
  `node scripts/vendor.mjs` after adding models.
- **Thumbnails** — `python3 scripts/thumbnails.py` (after `vendor.mjs`)
  compresses each model's screenshot (`<name>.png`) into
  `public/thumbs/<category>/<source>/<model>.webp` (≤ 320×240; the 4K
  menagerie renders go from ~84 MB to ~0.4 MB). The originals are untouched;
  models without a screenshot show their initials. A `<name>.png` added after
  the index was written is picked up too, and models with no image at all
  are listed at the end.
- **Objects from RoboCasa** — `python3 scripts/import_robocasa.py
  [/path/to/robocasa]` copies one instance of each picked kitchen category
  (`PICKS` in the script) into `mjcf_assets/objects/robocasa/`, adding a free
  joint (raised so the object rests on the floor) and moving collision geoms
  to group 3. Models without a screenshot get one from
  `node scripts/render_previews.mjs [category[/source]…]`, which spawns each
  alone in headless Chrome (needs the page served, e.g. `npm start`) and
  writes `<name>.png` next to it.
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
with an "Attach conflict" warning in the console; set them in the World
panel (a policy that needs them offers to).

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

A keyframe made for the robot's own scene can put it below our floor
(TIAGo's sits at z = -0.985 on a floor at -0.99); a spawn whose collision
geoms reach more than 10 cm below the spawn height is raised to clear it.

Known gaps: `shadow_dexee` needs the `mujoco.pid` plugin, which the WASM build
does not ship; `iit_softfoot` (an `<asset><model>` sub-model) fails to attach;
`tetheria_aero_hand_open` loses 6 of its 7 tendon actuators in `mjs_attach`
(spawning reports the mismatch). Skyboxes, cube textures and hfields are not
drawn.

## World physics

The World panel (`public/worldpanel.js`, settings in `public/physics.js`,
saved in this browser) sets MuJoCo's global options and default geom
properties:
- **Gravity, wind, magnetic field, medium density and viscosity,
  impratio, friction cone** go into the world spec's `<option>` at every rebuild and onto
  the live `model.opt`, so they act at once. Wind and fluid drag need a
  medium (density or viscosity > 0).
- **Default geom density, mass and friction** apply to geoms that do not
  set them themselves or through their own default classes, as a top-level
  `<default><geom/></default>` would: density and friction are written into
  each spawned model's root default class (only what it does not set), and
  into the world's, so the floor takes the friction; a default mass is
  written onto each geom whose element and class chain set neither mass
  nor density (an inherited mass would override a geom's own density).
  They change masses, so **Apply** recompiles the spawned models' inertias
  (`withPhysics` in main.js) and rebuilds the world, keeping its state.
  Most models set their own masses (menagerie robots have explicit
  inertials), so these mostly reach plain objects and the floor.
- **Sleep** (on by default, unlike MuJoCo): MuJoCo's island sleeping
  (`<flag sleep="enable"/>`). Bodies at rest stop being simulated until
  something touches, pulls or moves them; robots with actuators never
  sleep, nor does what rests on them. Nearly all of a large scene rests (a
  RoboCasa kitchen: 44 of its 45 trees), and in WASM physics was most of the
  frame: with two kitchens, the Garmi apartment, a G1 and the duck a step
  went from 3.8 to 1.0 ms, and the world from 40 % of real time to real
  time. It is set at compile time (turning it on in a live model crashes
  MuJoCo), so changing it rebuilds the world. The server does the same.
  Fixed and sleeping bodies also cost no drawing work: `syncTransforms`
  only updates the parts of bodies that moved (3.6 → 0.5 ms a frame there).

## Render panel

The Render panel (`public/render.js`, hidden by default, saved in this
browser) trades drawing quality for GPU load, applied at once: **Preset**
(High, Medium (the default), Low, Ultra low; Custom once a field differs),
**Resolution** (canvas pixels per CSS pixel, at most the screen's; the
presets keep the screen's, which on an iPhone 15 made little difference),
**Anti-alias** (off, FXAA, MSAA 4×), **Occlusion** (SSAO off, on at half
resolution, high at full), **Shadows** (the sun's, on or off: this Filament
build binds no shadow-map options, so their size is fixed) and **Textures**
(64, 512, 1024 or 2048 px on the longer side; a change decodes the scene's
images again, so it can be compared on the spot: a kitchen's 18 textures
are 63 MB at 1024 and 0.3 MB at 64). Ultra low is Low with 64 px textures,
to see what textures cost. The WebXR eyes take the anti-aliasing and
occlusion settings too.

## Teleop panel and controllers

The Teleop panel (`public/teleop.js`, open by default) picks which robot is
driven (the newest by default, or the one double-clicked; double-clicking
anything else releases it; the others get a zero command and stand or
brake) and which of its controllers are on: **Policy**, **Base**, **IK**,
any of them or none. The choice is saved per robot file
(`mujoco-filament/controllers/<path>`); by default a robot runs the policy
made for it (GR00T on every menagerie G1 variant, no compatibility check:
switched on elsewhere, a policy reports the missing joint), Base when a
mobile base was found, IK always. Switching a policy off gives the
actuators their own gains back, and an automatic IK setup is found again
around whatever the remaining controllers drive (a G1 without its policy
gets leg chains). Keyboard: **WASD** walk / drive, **QE** turn, hold
**R**/**F** to raise or lower a locomotion policy's base height (keys typed
into a text field are ignored); the sections fold.

The **Input** select picks the keys or a **goal marker** (the default,
`public/goal.js`): an orange ring with a heading arrow, like the spawn
marker, that sits under the driven robot until it is dragged along the
floor (the mouse wheel or **Q**/**E** turn it while dragging, **Esc** puts
it back; a label by the pointer shows its position and heading). It is
drawn over everything, so a robot's base does not hide it. The robot then drives itself there, with the same twist command
the keys give and within their speeds, so a walking policy and a wheeled
base both follow it, in local and network mode: far off it turns towards
the goal and goes forward (a robot that cannot strafe backs up to a goal
just behind it), near it a mecanum base or a walking robot closes the rest
sideways, and at the point it turns to the goal heading (6 cm, 3.5°, with
hysteresis). A robot keeps going to its goal while another one is driven;
Reaching the goal lets it go (the readout says *arrived*; pushed or carried
off afterwards, the robot stays put). **Stop** forgets the goal, and so does Reset or driving it with the keys.
In network mode a robot has one goal for everyone: the server follows it
(`server/goal.py`), every client shows it, and the last one set wins.

Goals can also come from a script, for several robots at once:

- **In the page** (devtools console or page automation, local and network
  mode): `goals.move(which, {forward, left, turn})` sets each robot's goal
  relative to its own pose (m, rad), `goals.set(which, {x, y, yaw})` an
  absolute one, `goals.clear(which)` stops them, `await goals.wait(which)`
  resolves once all have arrived, `goals.list()` shows names, poses and
  goals. `which` is `'all'` (every robot whose walking policy or base is
  on), a name (`'r3'`), a label, or a list of those. E.g.
  `goals.move('all', {forward: 1}); await goals.wait('all'); goals.move('all', {forward: -1})`.
- **On the server** (HTTP): `GET /api/robots` lists the robots
  (`name`, `pose`, `goal`, `drives`, `arrived`); `POST /api/goal` with JSON
  `{robots, forward, left, turn}` or `{robots, goal: {x, y, yaw}}` sets
  goals, `{robots}` alone clears them. `scripts/goal_demo.py` drives robots
  back and forth with it:
  `python3 scripts/goal_demo.py --robots r1,r2 --distance 2 --times 3`.

A robot that is being driven with the keys in the Teleop panel ignores its
goal, as with the marker.

Pretrained policies live in `pretrained_policies/<dir>/`, one checkout per
model (not tracked by git). `npm install` (or `node scripts/vendor.mjs`)
links the folder as `public/pretrained_policies` and lists what is there in
`public/policies.json`; a registered policy whose directory is missing is
shown as "not installed" and never switched on by default. Policies are
registered in `POLICIES` (`public/policy.js`): each entry has an id, a
name, its `dir`, the robot files it starts on for (`defaultFor`),
`load(file)` for its weights (`file(name)` is the URL of a file in its
directory; shared, loaded on first use; onnxruntime-web is there as
`onnxRuntime()`) and `create(weights, prefix)` for a controller with
`bind` / `unbind` / `owns` / `pose` / `reset` / async `step`, plus `cmd` /
`drive` (and `height`) for a locomotion policy driven by the keys, and an
optional `substep(data)` run before every physics step (actuator models).
An entry's optional `physics` names World settings it needs (e.g.
`{cone: 'elliptic', impratio: 100}`): the Policy section shows them when
the world differs, with a button that applies them. The panel's Policy
section picks the model; a hand or quadruped policy is one more entry.
`public/torchscript.js` reads the tensors of a TorchScript `.pt` (a zip
of raw storages) in the browser, so small exported MLPs/RNNs need no
conversion.

## Walking G1 (GR00T policy)

If the GR00T locomotion policies are checked out into `pretrained_policies/`
(`git clone https://huggingface.co/nepyope/GR00T-WholeBodyControl_g1
pretrained_policies/GR00T-WholeBodyControl_g1`), they load the first time a
robot runs the `groot-g1` policy. `public/groot.js` is a port of lerobot's
`GrootLocomotionController` running on onnxruntime-web (WASM, one thread,
~0.2 ms per inference): two MLPs (Balance when the command is ~0, Walk
otherwise) at 50 Hz on 6 stacked 86-D observations, driving legs and waist,
while the arms hold whatever target their actuators have (the default pose
at spawn; set them with IK chains or the Stage panel's control sliders, as
in g1_mujoco). The policy's PD gains are written
into the menagerie position actuators (`gainprm`/`biasprm`), and the IMU is
the floating base (`qpos[3:7]`, `qvel[3:6]`), as in lerobot's G1 sim.
Inference is async, so `App.advance()` awaits it every 0.02 s of sim time.
`g1.xml` and `g1_with_hands.xml` both walk.

## Walking Spot and Go2 (trained in MuJoCo Playground)

`training/` trains locomotion policies on the GPU with MuJoCo Playground
(MJX / MuJoCo Warp + Brax PPO) and exports them for the viewer; see
`training/README.md`. `uv run train.py` then `uv run export_onnx.py
runs/<name>` put `policy.onnx` and `policy.json` into
`pretrained_policies/playground_spot/` (or `playground_go2/`), and the
menagerie Spot (`boston_dynamics_spot/spot.xml`) and Go2
(`unitree_go2/go2.xml`) start with the `playground-spot` /
`playground-go2` policy on. Playground has no Go2 task: `training/envs/go2.py`
runs its Go1 task on a Go2 built from `go2.xml`; the Go2 policy walks
without falling, the Spot one (Playground's own config) still tips over
under quickly changing commands. `public/playground.js` (and `PlaygroundController` in
`server/policies.py`) runs any such policy from its `policy.json`: it writes
the trained PD gains and joint damping / armature / friction loss into the
robot, builds the observation (body gyro and up vector, joint positions,
a history of tracking errors, foot positions, last action, command, or
for Go2 body velocity, gravity and joint velocities) and sets the targets
`default + action * scale` every 0.02 s; on `go2.xml`'s torque motors it
computes the trained PD torque itself before every physics step.

## Walking Microduck (trained in microduck_rl)

Pollen Robotics' [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
(mjlab, MuJoCo Warp, rsl_rl PPO) trains its 14-servo bipedal duck; with a
clone next to `package.json`, `scripts/import_microduck.py` copies the
robot (`digital_twin/microduck/microduck.xml`) and two of its scenes into
the library, and after training `scripts/export_microduck.py` (run in the
clone's environment, see its docstring) exports the walking policy to
`pretrained_policies/microduck_rl/` (`walk.onnx`, `policy.json`). The duck
starts with `microduck-walk` on (keys and goal marker).

It was trained on the BAM model of the XL330 servos (firmware P loop,
DC motor with back-EMF and battery sag, load-dependent friction), not on
plain position actuators, and does not stand without it. `public/microduck.js`
(and `MicroduckController` / `BamServos` in `server/policies.py`) turns the
robot's position actuators into motors on bind, computes the servo torques
and writes the friction budget into the joints before every physics step,
delays the targets by the training's command delay, and builds mjlab's 61-D
observation (trunk angular velocity and gravity, joint positions and
velocities, last action, command). In the viewer's world it walks without
falling at ~2/3 of the commanded speed forward and back, turns at the
commanded rate, but sidesteps slowly (0.2 m/s → 0.05); standing still it
does not start below ~0.2 m/s or ~0.6 rad/s (`min_command`, so the goal
marker commands at least that) and often does not turn right in place, as
in microduck_rl's own CPU rehearsal (`scripts/infer_policy.py`).

## Mobile bases

Every robot gets a `BaseController` (`public/base.js`), driven
from the Teleop panel like the G1 (**WASD**, **QE**), with a PI loop on the
floating base's measured twist (ground truth as odometry) and an
acceleration limit on the command (a step to full speed tips a tall robot
over). With no command the wheels brake. The panel shows command and
measurement and edits the setup; **Save** keeps it per robot file
(`mujoco-filament/base/<path>`), **Auto** / **Defaults** goes back to the
found / built-in one, **Export** copies JSON for `BASE_DEFAULTS`.

Kinds:
- **wheels**: a free root joint on driven wheels (velocity servos, or
  motors with a P loop on wheel speed). Wheel positions and spin axes come
  from the model; each wheel turns at `ω = g·v / (r g·d)`, `v` the wheel
  center's velocity for the commanded twist, `d` its rolling direction and
  `g` the direction it pushes: `d` for a plain wheel (differential, skid
  steer; no sideways motion), `d ± axis` for mecanum rollers at ±45°.
  A wheel given as a joint instead of an actuator runs on a software
  velocity servo (every physics step, a joint force with feedforward
  against its damping and friction loss), and an actuator on that joint is
  switched off meanwhile; when no wheel has an actuator, `autoConfig()`
  takes the unactuated level hinges hanging right off the base this way.
  Configurable: base joint, wheels (actuator or joint, roller), radius, key
  speeds, feedforward / P / I / integral limit / acceleration per axis.
- **planar**: x / y / yaw position servos on the root body (TidyBot): the
  pose target is integrated from the command.

`autoConfig()` finds them: wheels are hinge joints with a velocity servo
or motor, a level axis and a low axle; a wheel carrying unactuated roller
hinges is mecanum, the side its lowest roller leans to giving ±45°; the
radius is the axle height in the default pose. Garmi has a built-in setup
(its mecanum base: forward and sideways track open-loop; its rollers give
yaw a lever of only |x − y| = 4 cm, hence a yaw feedforward of 3). Checked:
Garmi, TIAGo (position / velocity / motor variants: forward to 0.01 m/s),
TidyBot (planar). Model limits: RB-Y1's base mesh rests on the floor (no
casters), so its wheels cannot move it; TIAGo++'s omni base has no rollers
(no sideways motion, a skid-steer box). Stretch 3's wheel velocity servos
chatter (the left one flips ±2.3 rad/s every step, in plain MuJoCo too), so
its built-in setup drives the wheel joints with software servos (0.3 m/s
and 1 rad/s tracked).

## IK chains

Every robot under `mjcf_assets/robots/` gets an `IkController`
(`public/ik.js`), on by default: IK chains whose targets are RViz-style
interactive markers (`public/marker.js`, a fixed 0.15 m in the world):
drag an arrow to move the target along that axis, a ring to turn it about
that axis, the center ball (colored by chain) to move it in the plane
facing the camera. Axes are the target's own (red x, green y, blue z); the
part under the pointer turns yellow, and the camera does not move while a
marker is held. **G** opens or closes the gripper of the chain dragged
last. Until a marker moves, a chain just holds its joints where they are.

A chain is only `{root, tip}`, plus the tip frame: a site rigidly attached
to the tip body (a TCP) or the body moved by an offset (Franka's 0.1034 m):
its joints are the ones on the body path from the tip up to the root,
root's own joints included, so moving the root to the elbow or down to
Garmi's lift changes which joints take part. What moves a joint, in
order: its own position servo; a position servo on a fixed tendon through
it (Stretch 3's telescoping arm: one IK dof for the tendon's length, the
joints sharing the motion); else, with no actuator at all, a software
servo of IK's own (every physics step, a critically damped PD force on the
joint scaled by its inertia, plus gravity compensation, in
`qfrc_applied`). Joints another controller drives (G1's waist and legs
under GR00T), free joints and joints the user unticks stay still; a joint
coupled to another by a joint equality (Garmi's second lift stage,
Stretch's telescope) adds its motion to its leader's Jacobian column. On a
robot with no position servos at all, automatic chains also take its
joints that have a range.

Configs come from, in order: the one saved in the panel, `IK_DEFAULTS`
(Garmi: both arms, root `*_fr3_link1`, tip `*_fr3_hand` + 0.1034 m), or
`autoConfig()`, which looks at the model: joints with a position servo,
not driven by another controller, not on the robot's root body (a mobile
base) and not named like hand/gripper joints (`finger`, `jaw`, `thumb`,
`hand_`, …) are movable; every serial run of at least 3 of them that ends a
branch of the body tree becomes a chain, from where the tree branches (or
the base) to the last movable body, carried on through fixed single
children (flange to hand). Shorter end runs (a pan-tilt head, fingers) are
dropped first. The tip frame is the model's TCP site when one is fixed to
the tip, by name, best first: `tcp`, `pinch`, `grasp`, `gripper` /
`end_effector` / `ee`, `hand_center` / `palm`, then the `attachment_site`
flange (i2rt YAM `tcp_site`, Kinova `pinch_site`, ViperX `pinch`, ALOHA
`*/gripper`, SO-101 `gripperframe`, TidyBot `pinch_site`, UR5e / iiwa /
Lite6 `attachment_site`). MJCF has no TCP element, so a model without such
a site (menagerie's `panda.xml`, G1) keeps the body origin;
`IK_DEFAULTS` gives Panda and Garmi the Franka TCP offset instead. The one actuator driving joints below the tip that no chain
uses becomes the gripper. Examples: Panda `link1 → hand` (gripper
`actuator8`), UR5e `shoulder_link → wrist_3_link`, ALOHA and TIAGo dual
both arms, Apollo arms, legs and neck, Spot four legs, G1 both arms (legs
and waist are GR00T's), TIAGo `torso_lift_link → arm_7_link` (lift
included); Go2 and H1 have torque motors, so none.

Each tick (50 Hz) every chain takes one damped-least-squares step
(`dq = Jᵀ(JJᵀ+λ²I)⁻¹e`, Jacobian from `xaxis`/`xanchor`), with a pull
towards the starting posture in the null space, joint speed limits and
joint/ctrl range clamps, tracking position and orientation (orientation
weighted 0.3 m/rad; a chain with fewer than 6 joints gets the least-squares
compromise). A joint the step would push past its range is taken out and
the step solved again, so the rest do not chase what it cannot do. IK runs
on the *commanded* joint positions (forward kinematics in a scratch
`MjData`), so gravity sag does not wind the targets up. The target lives in
the frame of the parent of the first body the chain moves: it travels with
the base (and the lift, when the chain starts above it), and a chain rooted
at G1's pelvis keeps its target on the torso while it walks.

The Teleop panel edits the driven robot's chains: tip and root (select, or
double-click a body and press ◎), TCP site or offset, joints, gripper actuator;
**+ Chain** starts one at the selected body. **Save** keeps the setup per
robot file in localStorage (`mujoco-filament/ik/<path>`), **Auto** /
**Defaults** goes back to the found / built-in one, **Export** copies it as
JSON for `IK_DEFAULTS`. While IK is on it owns its joints' controls, so the
Stage panel's sliders for them are overridden; untick **IK chains** to set
them by hand. Known limits: a straight, singular arm (RB-Y1 at rest) cannot
move along itself until moved another way first; joints driven through a
tendon (Stretch's telescoping arm) are not IK joints.

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
Back faces are culled like MuJoCo does. Filament sorts transparent
renderables back to front per renderable, so a self-overlapping or
interpenetrating transparent mesh can blend in the wrong order.

## Shared world (Python server)

```sh
pip install mujoco numpy aiohttp onnxruntime
npm run server   # python3 server/server.py 4007
```

`server/server.py` steps one MuJoCo world in Python and serves `public/`
too; every browser that opens it (http://localhost:4007, or through the
JupyterHub proxy) sees and changes the same world. The page notices the
server (`server.json`) and switches to network mode (`public/net.js`), in
which it loads no MuJoCo at all:

- **Drawing.** The server compiles each model file once and serves what is
  drawn at `/render?rel=…` (`server/render_asset.py`): visible geoms,
  materials, the meshes they use (compiled: MuJoCo recenters meshes), the
  pixels of MuJoCo's builtin textures, image textures by path, and the
  kinematic tree. Gzipped, cached by the browser (ETag); a G1 is ~7 MB. The
  arrays keep MjModel's field names, so `buildVisual` turns them into
  Filament parts like it does a WASM model.
- **State.** 30 times a second the server sends qpos and mocap poses as
  float32 (ctrl at 5 Hz, for the panels): ten G1s are ~1.7 KB per state,
  ~50 KB/s. The page interpolates between states, a little behind (see
  the network meter; quaternions on the short arc), and places the bodies itself
  (`forwardKinematics`, MuJoCo's `mj_kinematics`, matching it to 1e-15).
- **Requests.** Spawn, Undo / Clear / ✕, Reset, Pause, the Stage panel's
  poses and sliders, Shift+drag, double-click picking (the server casts the
  ray) and the World panel's options go to the server, which applies them
  and tells every browser. Robots are attached in the same order on both
  sides, so bodies, joints and actuators are numbered the same.
- **Controllers on the server.** Policies (`server/policies.py`: GR00T
  and Playground policies on onnxruntime, same ids as `policy.js`), mobile
  bases (`server/base.py`) and IK (`server/ik.py`) are ports of `groot.js`,
  `playground.js`, `base.js` and `ik.js`,
  ticked at 50 Hz with the physics; ten G1s walking under GR00T keep real
  time. The Teleop panel works as in local mode through `public/remote.js`.
  The keys' drive command goes to the server when it changes and is
  repeated while held; the server stops a command not repeated for 1 s
  (a closed or frozen tab), and the base's measured twist comes back for
  the readout. Base setups saved from the panel go to
  `server/saved/base.json`; `base_defaults.json` holds the built-in ones.
- **IK markers.** They work as in local mode:
  dragging a marker moves it at once and sends its target (in the chain's
  reference body frame, so it rides with the robot); the server sends
  targets and tracking errors back 5 times a second, so every browser sees
  the markers move. Chain edits, Save (to `server/saved/ik.json`, for
  everyone) and Defaults are requests too. `ik_defaults.json` holds the
  built-in setups for both sides.
- **Network meter.** Bottom right, next to the frame meter: round trip
  (ping, includes the page's own frame time), states per second, download
  rate, and how far the view trails the newest state: one state interval
  plus three times the arrival jitter (40–250 ms).
- **Reconnecting.** When the connection drops (the server stopped, Wi-Fi
  changed, a phone slept: pings unanswered for 5 s count too), a banner
  says so and the page tries again after 1, 2, 4, 8, then every 10 s, at
  once on Retry now, when the network comes back or the tab is shown
  again. The server sends its world on reconnecting; a restarted server's
  (empty) world replaces the old one.
- **Memory.** Copies of a model share its meshes in the server's world:
  ten G1s compile to ~75 MB instead of ~750 MB.

Not yet in network mode: the World panel's default geom properties. The scene lives as long as
the server process. Local mode keeps everything in the browser as before.

`?local` in the URL keeps the page fully local (WASM physics) even when the
server serves it.

## GitHub Pages and Binder

The page and the server can also live apart: the page as a static site on
GitHub Pages with a part of the library, running locally, and the server on
Binder with the whole library, joined from the page's **Server** panel.
`public/endpoint.js` decides where the server is: `?server=<url>`, else
the last one joined while it still answers, else the page's own origin
(server.py serving the page, as above), else none. With a server, every
data file (assets.json, thumbnails, `mjcf_assets/`, `pretrained_policies/`)
and `/render` come from it. The server answers any origin (CORS); on Binder
the session's token, which the page puts on every request, WebSockets
included, is what guards it.

**The static site.** `pages.json` lists the model and policy directories
it carries (the default: Go2, Spot, Panda, G1, the microduck, Garmi, the
custom environments, the digital_twin objects, the banana and the four
policies; ~290 MB). Then:

```sh
node scripts/vendor.mjs && python3 scripts/thumbnails.py
node scripts/build_pages.mjs                 # into dist/
node scripts/build_pages.mjs --push origin   # and force-push it as branch gh-pages
```

Set the repository's Pages source to the `gh-pages` branch. It is one
commit each time, so model files never pile up in its history. The build
fails on a file over 100 MB and warns over 1 GB, GitHub's limits.
`deploy.json` on the site names the Binder repository the panel starts:
`pages.json`'s `binder`, by default this checkout's GitHub remote and branch
on mybinder.org (`--binder gh/<owner>/<repo>/<ref>`, `--hub <url>` to
change it).

**The Binder repository** is this one (`binder/`): `environment.yml`
(Python, MuJoCo, onnxruntime, Node), `postBuild`, which clones
mujoco_menagerie, runs `scripts/basic_scenes.py`, downloads what
`binder/library.txt` lists (GR00T's policies from Hugging Face by default)
and indexes it all (`npm ci`), and `jupyter_server_config.py`, which puts
server.py behind the session's jupyter-server-proxy at `<session>/mujoco/`,
started on the first request. The rest of a local library (digital_twin,
robocasa, your own policies) has to be downloadable for the build:
`python3 scripts/pack_library.py` packs it as one .tar.gz per source (e.g.
for GitHub release assets, 2 GB each at most) and prints the lines for
`library.txt`.

**Joining.** *Start a server on Binder* in the Server panel calls
BinderHub's build API, shows the build and launch, and reloads the page on
the new session's `<url>mujoco/?token=…`. *Join a server* takes such a URL
directly (or any server.py's). *Copy link* gives a link to the page on that
server, token included, so others join the same world; *Go local* leaves
it. A Binder session ends after a while without anyone connected (on
mybinder.org about 10 minutes); the page then starts locally again.

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
matc -a opengl -p mobile -o public/materials/lit_transparent.filamat materials/lit_transparent.mat
matc -a opengl -p mobile -o public/materials/lit_textured_transparent.filamat materials/lit_textured_transparent.mat
```

## Relation to MuJoCo's own Filament renderer

MuJoCo's native Filament backend (`mjrfilament`, `src/render/filament/`) is
not part of the `@mujoco/mujoco` npm package (3.15.0). Its CMake has an
Emscripten branch, but it requires a native build first (to compile the
materials with matc) and then a full Emscripten build of Filament + MuJoCo.
This example instead drives Filament.js directly from JavaScript.
