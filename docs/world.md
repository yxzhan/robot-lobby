# Physics

The Physics panel (`public/js/world/worldpanel.js`, settings in `public/js/world/physics.js`,
not saved: every load starts from the defaults) sets MuJoCo's global options and default geom
properties (how the floor and the lighting look, and whether the camera
follows the selection, are the Render panel's: rendering.md):
- **Gravity, wind, magnetic field, medium density and viscosity,
  impratio, friction cone** go into the world spec's `<option>` at every rebuild and onto
  the live `model.opt`, so they act at once. Wind and fluid drag need a
  medium (density or viscosity > 0).
  Contacts default to an **elliptic cone with impratio 1** (in the browser
  and on the server; MuJoCo's own default is pyramidal): the more accurate
  cone, which the menagerie's legged robots ask for. Impratio 10 on top made
  friction too strong and costs about 5% more step time. Friction is softer this way,
  and a Panda's fingers let a 100 g box creep out within seconds; for
  grasping, elliptic with impratio 10 holds the box through a lift and a
  swing to within 2 mm.
- **Noslip** (Contacts; off by default, as in MuJoCo and in the policies'
  training: with it the Microduck's kick did not kick again after a first
  one; for grasping, 100 iterations, tolerance 1e-10): a post-pass of the
  solver that takes the slip out of frictional contacts. MuJoCo's friction is soft: it pushes back only
  while things slide, so under a steady load (a held box's weight) they
  creep at a constant rate whatever the friction coefficient. Without
  noslip, a Panda holding a 100 g box lets it creep 8 mm in 20 000 steps
  even with elliptic and impratio 10 (out of the fingers within 10^5);
  with it, 0.05 mm in 10^6 steps. Its tolerance, not its iterations,
  decides: at MuJoCo's 1e-6 it stops early and the creep is barely
  smaller. It costs step time where there are many contacts (that grasp:
  25 → ~100 µs a step); 0 iterations turns it off.
- **Solver** (Solver): the integrator (implicitfast by default, as the
  menagerie models set it; Euler, RK4, implicit, and MuJoCo 4's discrete:
  passive damping and stiffness implicit in the solver, so stiff things
  stay stable at much larger steps, as the dry-stone arches below at 6–8 ms), the constraint solver
  (Newton, CG, PGS), its most iterations and its tolerance, live like the
  options above.
- **Timestep and speed** (Simulation): the physics step (0.005 s by default,
  changed live; larger steps cost less per simulated second but are less
  stable, and walking policies were trained at 0.002–0.005 s) and the most
  simulated time per second of wall time, a slider of steps: 0.1× to 10×
  or **max**, as fast as the computer goes (in the browser up to 20 ms of
  stepping per frame; on the server without a pause between rounds). A
  world that cannot keep up runs slower whatever the setting; the meter next
  to the frame meter shows the speed reached (`sim 1.00× · 500 steps/s`).
  In network mode the page's interpolation follows the server's rate.
- **Sync rate** (network mode only): the server's state broadcasts per
  second, 10–60 Hz (30 by default): more is smoother and fresher, and more
  traffic for every page.
- **Default geom density, mass, friction, condim, solref and solimp** apply to geoms that do not
  set them themselves or through their own default classes, as a top-level
  `<default><geom/></default>` would: all but mass are written into
  each spawned model's root default class (only what it does not set), and
  into the world's, so the floor takes them too; a default mass is
  written onto each geom whose element and class chain set neither mass
  nor density (an inherited mass would override a geom's own density).
  Condim 4 adds torsional friction (a held thing does not spin in the
  fingers); a smaller solref time constant makes contacts stiffer (keep it
  above two timesteps). They change masses or contacts, so **Apply** recompiles the spawned models' inertias
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
- **IPC** (Contacts; off, as in MuJoCo): MuJoCo 4's incremental
  potential contacts for flex bodies (`<flag ipc="enable"/>`): cloth never
  passes through itself or other things (the bag example), at a large cost
  in step time. Set at compile time, so changing it rebuilds the world.
- **Speculative** (Contacts; off, as in MuJoCo): MuJoCo 4's speculative
  contacts (`<flag speculative="enable"/>`), found ahead of time from the
  geoms' velocities, so that fast things do not pass through thin ones at
  large time steps (the MuJoCo examples' bat hitting a ball at a 10 ms
  step). Live, no rebuild; the server does the same.
- **A scene's own options**: spawning a model whose `asset.json` says
  `"physics": "own"` (the MuJoCo examples below) takes the options its
  `<option>` writes out (time step, integrator, cone, impratio, medium,
  noslip, the sleep / speculative / ipc flags, no gravity for
  `gravity="disable"`) into these settings, on top of MuJoCo's own defaults
  (sceneBase: Euler, 2 ms, pyramidal, no sleep; not the viewer's, nor what
  an earlier one set: an arch's impratio 100 would stay for the next
  scene), so that it runs as in MuJoCo (the flex examples that set no
  integrator need Euler: flex elasticity refuses implicitfast), and the
  status line names those that changed (`sceneOptions` in physics.js, `scene_options` in server.py).
  Every other model's `<option>` loses against these settings, as before:
  RoboCasa's kitchens set a 2 ms step (and impratio 20), which would make
  every robot's physics cost 2.5 times as much. They stay until changed,
  also after the scene is removed; a stage keeps its own.

## MuJoCo 4 examples

The Scenes tab's **mujoco_examples** are example models of MuJoCo 4.0 that
show its new physics, copied by the library's
`scripts/import_mujoco_examples.py` (from a google-deepmind/mujoco checkout
at tag 4.0.0) without their floors (the lowest plane facing up; a scene
whose floor was below z = 0 is moved up onto ours), each with its own
options (above): surface velocity (a treadmill sweeping a passive humanoid
off, a carousel of conveyor ramps), speculative contacts (a bat and ball, a
dropped chain caught by a thin bar, a cradle of 3 mm plates), adhesion (a
fridge door held shut by magnets, Shift+drag it open; a sand castle),
plasticity (wires that stay bent; clay, foil, paper, a roller and a twisted
bar that stay deformed), sleep (dominos), rigid bodies (a house of cards, a
deck of cards, three dry-stone arches, balloons) and flex (soft) bodies:
cables, coils and rods, cloth (a flag, a poncho, drapes, a hammock with a
humanoid, a trampoline, a bag with IPC), shells and soft solids (bunnies,
jelly, a soft box, spheres, grippers squeezing soft things, cantilevers).
All of MuJoCo's flex examples but `cloth_sdf` (its SDF plugin is not in the
WASM build); the actuator- or plugin-driven ones (active adhesion,
tactile) are left out. Some are heavy: in the browser the bag steps at
about 1/25 of real time, drape, poncho, cantilever and basket at 0.1–0.5×
(on the server, natively, about twice as fast). Tendons are not drawn (the
balloons' strings). A scene whose `<attach>`ed humanoid brought its own
keyframes (treadmill, hammock) has them left out: the viewer starts a model
in its first keyframe, which would put the humanoid at the origin.

Flex bodies are drawn by `render/flex.js` as MuJoCo 4's own renderer
draws them (2D: both faces offset by the flex radius and closed along the
boundary; 3D: the boundary; 1D: rods and balls), from mjData's
`flexvert_xpos` every frame; in network mode the server sends the flex
vertices with every state (server.md). Not yet: flexes in the robots'
depth / segmentation cameras, and Shift+drag on flex vertices.
