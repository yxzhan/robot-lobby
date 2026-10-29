# Physics

The Physics panel (`public/js/world/worldpanel.js`, settings in `public/js/world/physics.js`,
not saved: every load starts from the defaults) sets MuJoCo's global options and default geom
properties (how the floor and the lighting look, and whether the camera
follows the selection, are the Render panel's: rendering.md):
- **Gravity, wind, magnetic field, medium density and viscosity,
  impratio, friction cone** go into the world spec's `<option>` at every rebuild and onto
  the live `model.opt`, so they act at once. Wind and fluid drag need a
  medium (density or viscosity > 0).
  Contacts default to MuJoCo's **pyramidal cone with impratio 1** (in the
  browser and on the server): an elliptic cone with impratio 10 made friction
  too strong and costs about 5% more step time. Friction is softer this way,
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
  menagerie models set it; Euler, RK4, implicit), the constraint solver
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
