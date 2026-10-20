// World physics settings (the World panel), for this page only (not saved):
//
// - Global options (MuJoCo <option>): gravity, wind, magnetic field, medium
//   density and viscosity (both 0: no fluid forces, so wind does nothing),
//   impratio (frictional-to-normal constraint impedance ratio), friction
//   cone (pyramidal or elliptic). Elliptic with impratio 10 by default, not
//   MuJoCo's pyramidal and 1: with those, friction is soft and things held
//   creep (a Panda lets a 100 g box slide out of its fingers in seconds;
//   with these it holds it, swinging, within 2 mm), at ~5% more step time;
//   legged robots' feet creep less too. They go into
//   the world spec at every rebuild (world.js) and are also set on the live
//   model, so a change needs no rebuild.
// - Default geom properties: density, mass and friction for every geom that
//   does not set them itself or through its own default classes, like a
//   top-level <default><geom …/></default> in each model: they are written
//   into each spawned model's root default class (only the attributes it
//   does not set) before it is compiled, and into the world's (the floor).
//   They change masses and inertias, so they need a rebuild.
// - Sleep (MuJoCo's island sleeping, <flag sleep>): bodies at rest stop being
//   simulated until something touches, pulls or moves them; robots with
//   actuators never sleep. Most of a large scene (a kitchen's drawers, doors
//   and props) rests, so this saves most of its physics time. On here, unlike
//   MuJoCo's default; it is set when the model is compiled, so a change
//   rebuilds the world.

// MuJoCo's own defaults, but for the contacts and sleep.
export const PHYSICS_DEFAULTS = Object.freeze({
  gravity: [0, 0, -9.81],
  wind: [0, 0, 0],
  magnetic: [0, -0.5, 0],
  density: 0,
  viscosity: 0,
  impratio: 10,
  cone: 'elliptic',
  sleep: true,
  geom: { density: 1000, mass: null, friction: [1, 0.005, 0.0001] },
});

// The World panel's settings for this page: the defaults at every load,
// not kept in the browser.
export function loadPhysics() {
  return structuredClone(PHYSICS_DEFAULTS);
}

const OPTION_KEYS = ['gravity', 'wind', 'magnetic', 'density', 'viscosity', 'impratio', 'cone'];
export const CONES = ['pyramidal', 'elliptic'];  // mjtCone order
const fmt = (v) => (Array.isArray(v) ? v.join(' ') : String(v));

// The <option> element of the world spec.
export function optionXml(s) {
  const attrs = OPTION_KEYS.map((k) => `${k}="${fmt(s[k])}"`).join(' ');
  return `<option timestep="0.002" integrator="implicitfast" ${attrs}>${s.sleep ? '<flag sleep="enable"/>' : ''}</option>`;
}

// Sets the global options on a compiled model (takes effect at the next step).
export function applyOptions(model, s) {
  const o = model.opt;
  for (const k of ['gravity', 'wind', 'magnetic']) o[k].set(s[k]);
  o.density = s.density;
  o.viscosity = s.viscosity;
  o.impratio = s.impratio;
  o.cone = Math.max(0, CONES.indexOf(s.cone));
}

// The default geom attributes that differ from MuJoCo's ({name: value}).
export function geomOverrides(s) {
  const d = PHYSICS_DEFAULTS.geom, g = s.geom, out = {};
  if (g.density !== d.density) out.density = fmt(g.density);
  if (g.mass != null) out.mass = fmt(g.mass);
  if (g.friction.some((v, i) => v !== d.friction[i])) out.friction = fmt(g.friction);
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
