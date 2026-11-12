// Physics settings (the Physics panel), for this page only (not saved):
//
// - Global options (MuJoCo <option>): gravity, wind, magnetic field, medium
//   density and viscosity (both 0: no fluid forces, so wind does nothing),
//   impratio (frictional-to-normal constraint impedance ratio), friction
//   cone (pyramidal or elliptic). Elliptic with impratio 1 by default (MuJoCo's
//   own is pyramidal): the more accurate cone, as the menagerie's legged
//   robots use; impratio 10 on top made friction too strong (things stick)
//   and costs more step time; friction is softer at 1, though, and held
//   things can creep (a Panda lets a 100 g box slide out of its fingers in
//   seconds; elliptic with impratio 10 holds it, swinging, within 2 mm),
//   so grasping may want those back in the Physics panel. Noslip (a
//   post-pass of the solver that takes the slip out of frictional contacts)
//   is off by default, as in MuJoCo (and in the training of the policies:
//   with it the Microduck's kick did not kick after a first one), and costs
//   step time with many contacts. For grasping, turn it on (100 iterations,
//   tolerance 1e-10): without it soft friction creeps steadily (a held box
//   drops out after some 10^5 steps even with elliptic and impratio 10;
//   with noslip it moves 0.1 mm in 2·10^5). Its tolerance, not its
//   iterations, decides: at MuJoCo's 1e-6 it stops early and the creep is
//   barely smaller.
//   Solver (Newton, CG, PGS), its iterations and tolerance, and the
//   integrator are global options too. They go into
//   the world spec at every rebuild (world.js) and are also set on the live
//   model, so a change needs no rebuild.
// - Default geom properties: density, mass, friction, condim (1 frictionless,
//   3 sliding, 4 also torsional: a held thing does not spin in the fingers,
//   6 also rolling), solref (contact time constant and damping ratio: a
//   smaller time constant is stiffer, it should stay above 2 timesteps) and
//   solimp (dmin, dmax, width: how soft and how deep) for every geom that
//   does not set them itself or through its own default classes, like a
//   top-level <default><geom …/></default> in each model: they are written
//   into each spawned model's root default class (only the attributes it
//   does not set) before it is compiled, and into the world's (the floor).
//   They change masses and inertias, so they need a rebuild.
// - Timestep and speed: the physics step (s; live, no rebuild: larger
//   steps cost less per simulated second but are less stable, and walking
//   policies were trained at 0.002-0.005 s) and the most simulated time per
//   second of wall time (1: real time; 0: as fast as it goes). A world that
//   cannot keep up runs slower whatever the speed.
// - Sync rate (network mode only): how many states a second the server
//   sends each page (more: smoother and fresher, more traffic).
// - Sleep (MuJoCo's island sleeping, <flag sleep>): bodies at rest stop being
//   simulated until something touches, pulls or moves them; robots with
//   actuators never sleep. Most of a large scene (a kitchen's drawers, doors
//   and props) rests, so this saves most of its physics time. On here, unlike
//   MuJoCo's default; it is set when the model is compiled, so a change
//   rebuilds the world.
// - Speculative contacts (MuJoCo 4, <flag speculative>): contacts found
//   ahead of time from the geoms' velocities, so that fast things do not
//   pass through thin ones at large time steps. Off, as in MuJoCo; live.
// - A model's own options: spawning one whose asset.json says "physics":
//   "own" (the MuJoCo examples, environments/mujoco_examples) takes the
//   options its <option> writes out (sceneOptions; not MuJoCo's defaults
//   for the rest) into these settings, as if set in the panel, so that a
//   scene made to show, say, speculative contacts at a 10 ms step runs as
//   made: on top of the defaults (sceneBase), not of what an earlier one
//   set. Other models' options lose against these, and so do those later
//   (a stage keeps its own settings).

// MuJoCo's own defaults, but for sleep.
export const PHYSICS_DEFAULTS = Object.freeze({
  gravity: [0, 0, -9.81],
  wind: [0, 0, 0],
  magnetic: [0, -0.5, 0],
  density: 0,
  viscosity: 0,
  impratio: 1,
  cone: 'elliptic',
  noslip_iterations: 0,
  noslip_tolerance: 1e-10,
  integrator: 'implicitfast',  // the menagerie models' (MuJoCo's own is Euler)
  solver: 'Newton',
  iterations: 100,
  tolerance: 1e-8,
  timestep: 0.005,
  speed: 1,
  stateHz: 30,  // network mode: the server's state broadcasts per second
  sleep: true,
  speculative: false,
  geom: { density: 1000, mass: null, friction: [1, 0.005, 0.0001], condim: 3, solref: [0.02, 1], solimp: [0.9, 0.95, 0.001] },
});

// The Physics panel's settings for this page: the defaults at every load,
// not kept in the browser.
export function loadPhysics() {
  return structuredClone(PHYSICS_DEFAULTS);
}

const OPTION_KEYS = ['timestep', 'gravity', 'wind', 'magnetic', 'density', 'viscosity', 'impratio', 'cone',
  'noslip_iterations', 'noslip_tolerance', 'integrator', 'solver', 'iterations', 'tolerance'];
export const CONES = ['pyramidal', 'elliptic'];  // mjtCone order
export const INTEGRATORS = ['Euler', 'RK4', 'implicit', 'implicitfast', 'discrete'];  // mjtIntegrator order
export const SOLVERS = ['PGS', 'CG', 'Newton'];  // mjtSolver order
export const CONDIMS = [1, 3, 4, 6];
export const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 10, 0];  // × real time; 0: as fast as it goes
export const STATE_RATES = [10, 15, 20, 30, 60];  // Hz
const fmt = (v) => (Array.isArray(v) ? v.join(' ') : String(v));

// The <option> element of the world spec.
export function optionXml(s) {
  const attrs = OPTION_KEYS.map((k) => `${k}="${fmt(s[k])}"`).join(' ');
  const flags = [s.sleep && 'sleep', s.speculative && 'speculative'].filter(Boolean).map((f) => `${f}="enable"`);
  return `<option ${attrs}>${flags.length ? `<flag ${flags.join(' ')}/>` : ''}</option>`;
}

const ENABLE_SPECULATIVE = 1 << 7;  // mjtEnableBit

// What a model's own options go on top of: the defaults of the global
// options and flags (not speed, sync rate or default geom properties).
export function sceneBase() {
  const out = {};
  for (const k of [...OPTION_KEYS, 'sleep', 'speculative']) out[k] = structuredClone(PHYSICS_DEFAULTS[k]);
  return out;
}

// The settings a model's own <option> elements set (in its parsed include
// tree, `docs`): {key: value} for the attributes it writes out, and sleep /
// speculative from its <flag>. An environment's, taken into the settings
// when it is spawned.
export function sceneOptions(docs) {
  const out = {};
  for (const doc of docs) {
    for (const opt of doc.documentElement.children) {
      if (opt.tagName !== 'option') continue;
      for (const k of OPTION_KEYS) {
        const v = opt.getAttribute(k);
        if (v === null) continue;
        if (k === 'cone' || k === 'integrator' || k === 'solver') {
          const names = { cone: CONES, integrator: INTEGRATORS, solver: SOLVERS }[k];
          const name = names.find((n) => n.toLowerCase() === v.trim().toLowerCase());
          if (name) out[k] = name;
        } else {
          const nums = v.trim().split(/\s+/).map(Number);
          if (nums.every(Number.isFinite)) out[k] = Array.isArray(PHYSICS_DEFAULTS[k]) ? nums : nums[0];
        }
      }
      for (const flag of opt.children) {
        if (flag.tagName !== 'flag') continue;
        for (const f of ['sleep', 'speculative']) {
          const v = flag.getAttribute(f);
          if (v !== null) out[f] = v === 'enable';
        }
      }
    }
  }
  return out;
}

// Sets the global options on a compiled model (takes effect at the next step).
export function applyOptions(model, s) {
  const o = model.opt;
  o.timestep = s.timestep;
  for (const k of ['gravity', 'wind', 'magnetic']) o[k].set(s[k]);
  o.density = s.density;
  o.viscosity = s.viscosity;
  o.impratio = s.impratio;
  o.cone = Math.max(0, CONES.indexOf(s.cone));
  o.integrator = Math.max(0, INTEGRATORS.indexOf(s.integrator));
  o.solver = Math.max(0, SOLVERS.indexOf(s.solver));
  o.iterations = s.iterations;
  o.tolerance = s.tolerance;
  o.noslip_iterations = s.noslip_iterations;
  o.noslip_tolerance = s.noslip_tolerance;
  o.enableflags = s.speculative ? o.enableflags | ENABLE_SPECULATIVE : o.enableflags & ~ENABLE_SPECULATIVE;
}

// The default geom attributes that differ from MuJoCo's ({name: value}).
export function geomOverrides(s) {
  const d = PHYSICS_DEFAULTS.geom, g = { ...d, ...s.geom }, out = {};  // a stage saved before some of them: the defaults
  if (g.density !== d.density) out.density = fmt(g.density);
  if (g.mass != null) out.mass = fmt(g.mass);
  for (const k of ['friction', 'condim', 'solref', 'solimp']) {
    if (JSON.stringify(g[k]) !== JSON.stringify(d[k])) out[k] = fmt(g[k]);
  }
  return out;
}

// Identifies the default geom attributes in caches of compiled inertias.
export const geomSignature = (s) => JSON.stringify(geomOverrides(s));

// Whether going from settings a to b needs the world compiled again.
export const needsRebuild = (a, b) => geomSignature(a) !== geomSignature(b) || !!a.sleep !== !!b.sleep;

// Writes `attrs` into the root default class of a model's XML documents
// (the parsed include tree, entry file first), keeping what it sets itself.
// A default mass cannot go there: an inherited mass overrides a geom's own
// density, so it is written on each geom that has neither (geomsWithoutMass).
export function injectGeomDefaults(docs, attrs) {
  const { mass, ...rest } = attrs;
  if (mass !== undefined) for (const g of geomsWithoutMass(docs)) g.setAttribute('mass', mass);
  attrs = rest;
  if (!Object.keys(attrs).length) return;
  const isRootDefault = (el) => el.parentNode === el.ownerDocument.documentElement
    && (!el.hasAttribute('class') || el.getAttribute('class') === 'main');
  let def = null;
  for (const doc of docs) {
    def = [...doc.getElementsByTagName('default')].find(isRootDefault) ?? null;
    if (def) break;
  }
  if (!def) {
    const doc = docs[0], root = doc.documentElement;
    def = doc.createElementNS(root.namespaceURI, 'default');
    root.insertBefore(def, root.firstChild);
  }
  const doc = def.ownerDocument;
  let geom = [...def.children].find((c) => c.tagName === 'geom');
  if (!geom) {
    geom = doc.createElementNS(def.namespaceURI, 'geom');
    def.insertBefore(geom, def.firstChild);
  }
  const own = new Set(geom.getAttributeNames());
  for (const [k, v] of Object.entries(attrs)) {
    // A model's own mass or density wins over either of ours.
    if (own.has(k) || ((k === 'density' || k === 'mass') && (own.has('density') || own.has('mass')))) continue;
    geom.setAttribute(k, v);
  }
}

// The <geom> elements that get neither mass nor density from themselves or
// their default class chain (their class, else the nearest childclass of an
// enclosing body or frame, else the root class; each class inherits from
// the one it is nested in).
function geomsWithoutMass(docs) {
  const parent = new Map(), sets = new Map();  // class -> parent class, class -> sets mass/density
  const isMassy = (el) => el.hasAttribute('mass') || el.hasAttribute('density');
  const visit = (def, up) => {
    const name = def.getAttribute('class') ?? 'main';
    if (up !== null) parent.set(name, up);
    const g = [...def.children].find((c) => c.tagName === 'geom');
    if (g && isMassy(g)) sets.set(name, true);
    for (const c of def.children) if (c.tagName === 'default') visit(c, name);
  };
  for (const doc of docs) {
    for (const d of doc.documentElement.children) if (d.tagName === 'default') visit(d, null);
  }
  const classMassy = (name) => {
    for (let c = name; c; c = parent.get(c)) if (sets.get(c)) return true;
    return false;
  };
  const out = [];
  for (const doc of docs) {
    for (const g of doc.getElementsByTagName('geom')) {
      if (g.closest('default') || isMassy(g)) continue;
      let cls = g.getAttribute('class');
      for (let e = g.parentElement; !cls && e; e = e.parentElement) {
        if ((e.tagName === 'body' || e.tagName === 'frame') && e.hasAttribute('childclass')) cls = e.getAttribute('childclass');
      }
      if (!classMassy(cls ?? 'main')) out.push(g);
    }
  }
  return out;
}
