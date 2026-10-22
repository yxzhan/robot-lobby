# World physics

The World panel (`public/js/worldpanel.js`, settings in `public/js/physics.js`,
not saved: every load starts from the defaults) sets MuJoCo's global options and default geom
properties:
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
- **Timestep and speed** (Simulation): the physics step (0.005 s by default,
  changed live; larger steps cost less per simulated second but are less
  stable, and walking policies were trained at 0.002–0.005 s) and the most
  simulated time per second of wall time: 0.1× to 10× or **max**, as fast as
  the computer goes (in the browser up to 20 ms of stepping per frame; on the
  server without a pause between rounds). A world that cannot keep up runs
  slower whatever the setting; the meter next to the frame meter shows the
  speed reached (`sim 1.00× · 500 steps/s`). In network mode the page's
  interpolation follows the server's rate.
- **Sync rate** (network mode only): the server's state broadcasts per
  second, 10–60 Hz (30 by default): more is smoother and fresher, and more
  traffic for every page.
- **Floor** (`public/js/floor.js`, saved in this browser, in network mode
  too since it is only this page's view): how the world floor looks, not its
  physics. Checker, grid, tiles, wood and concrete (textures drawn in code,
  tileable, tinted by the color), plain (one color) or none. Size is one cell,
  tile, plank width or pattern repeat; picking a style starts from its own
  size and color. **Shine** takes the floor from its style's roughness toward
  polished (0.05; checker starts at 100 %): it then reflects the sky, and the robots with the Render
  panel's Reflections on. Robot cameras see the floor too.
- **Lighting** (`public/js/lighting.js`, saved in this browser like the
  floor): the sun's brightness, azimuth, elevation and color, and the ambient
  (sky) light's brightness, live while a slider moves; Default resets them.
  Shadows stay in the Render panel.
- **Reflections**: the sky is also the scene's reflection map (IBL), its mip
  levels prefiltered in code for each roughness, so metal and shiny surfaces
  reflect it everywhere (VR and robot cameras included) at no extra cost.
  Screen-space reflections (mirroring the robots on a shiny floor) are the
  Render panel's Reflections setting, on in the High preset and off in VR;
  only the opaque lit materials take part (`reflections : screenspace` in
  `lit.mat` and `lit_textured.mat`).
- **Camera** (last, saved in this browser): *Follow selection*, off by
  default. A double-click selects a body and the camera turns to it once;
  with this on it keeps following it (orbit, zoom and pan go on, a pan
  offsetting the view) until the selection is cleared or changes, and holds
  still while something is dragged through the pointer (a pulled body, an IK
  or goal marker).
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
