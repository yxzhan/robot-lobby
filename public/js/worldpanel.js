// The Physics panel: global physics options and default geom properties
// (physics.js). Options apply as soon as a field changes; default geom
// properties change masses, so they wait for Apply (a rebuild). How the
// floor and the lighting look is the Render panel's (render.js).

import { CONDIMS, CONES, INTEGRATORS, PHYSICS_DEFAULTS, SOLVERS, SPEEDS, STATE_RATES } from './physics.js';

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

// [key, label, unit, components, tooltip]
const OPTIONS = [
  ['gravity', 'Gravity', 'm/s²', 3, 'gravitational acceleration'],
  ['wind', 'Wind', 'm/s', 3, 'velocity of the medium (needs density or viscosity > 0)'],
  ['magnetic', 'Magnetic', 'T', 3, 'global magnetic flux (magnetometer sensors)'],
  ['density', 'Density', 'kg/m³', 1, 'density of the medium (air ≈ 1.2, water 1000): lift and drag'],
  ['viscosity', 'Viscosity', 'Pa·s', 1, 'viscosity of the medium (air ≈ 1.8e-5, water ≈ 0.0009)'],
  ['impratio', 'Impratio', '', 1, 'ratio of frictional to normal constraint impedance (> 1 makes friction harder, less slip)'],
  ['cone', 'Cone', '', CONES, 'friction cone: elliptic is more accurate (legged robots), pyramidal cheaper'],
  ['noslip_iterations', 'Noslip iter.', '', 1, 'noslip post-pass: takes the slip out of frictional contacts, so held things do not creep out of a grasp (0: off). Costs step time with many contacts'],
  ['noslip_tolerance', 'Noslip tol.', '', 1, 'noslip stops below this; it decides more than the iterations (MuJoCo\'s 1e-6 stops early and things still creep)'],
  ['integrator', 'Integrator', '', INTEGRATORS, 'implicitfast: stable with damping and stiff actuators, cheap (the menagerie models\'); implicit: also Coriolis terms; RK4: accurate, 4× the cost; Euler: MuJoCo\'s default'],
  ['solver', 'Solver', '', SOLVERS, 'constraint solver: Newton converges in few iterations (MuJoCo\'s default); CG cheaper per iteration; PGS the old one, harder friction'],
  ['iterations', 'Iterations', '', 1, 'most solver iterations per step: more if contacts jitter or do not converge'],
  ['tolerance', 'Tolerance', '', 1, 'the solver stops below this (0: always all the iterations)'],
  ['timestep', 'Timestep', 's', 1, 'physics step: larger costs less per simulated second but is less stable and accurate (walking policies were trained at 0.002-0.005 s)'],
  ['speed', 'Speed', '', SPEEDS, 'the most simulated time per second of wall time (1×: real time, max: as fast as the computer goes); a world that cannot keep up runs slower'],
  ['stateHz', 'Sync rate', 'Hz', STATE_RATES, 'shared world: states the server sends each page per second (more: smoother and fresher, more traffic for everyone)'],
  ['sleep', 'Sleep', '', 'bool', 'bodies at rest stop being simulated until something touches, pulls or moves them (robots with actuators never sleep); saves most of the physics time of large scenes. Changing it rebuilds the world'],
];
const GEOM = [
  ['density', 'Density', 'kg/m³', 1, 'for geoms that set neither density nor mass'],
  ['mass', 'Mass', 'kg', 1, 'per geom, for geoms that set neither (empty: from density)'],
  ['friction', 'Friction', '', 3, 'sliding, torsional, rolling, for geoms that set none'],
  ['condim', 'Condim', '', CONDIMS, 'contact dimensions: 1 frictionless, 3 sliding friction, 4 also torsional (a held thing does not spin in the fingers), 6 also rolling'],
  ['solref', 'Solref', '', 2, 'contact time constant (s) and damping ratio: a smaller time constant is stiffer, keep it above 2 timesteps'],
  ['solimp', 'Solimp', '', 3, 'dmin, dmax, width (m): impedance (0-1, nearer 1 harder) from no penetration to width deep'],
];

export class WorldPanel {
  // `app.physics` holds the settings, `app.setPhysics(next)` applies them.
  constructor(app, root) {
    this.app = app;
    this.body = root.querySelector('.body');
    this.draft = null;  // default geom properties being edited, until Apply
    this.render();
  }

  render() {
    const s = this.app.physics;
    this.draft ??= structuredClone(s.geom);
    this.body.replaceChildren();
    const num = (value, onchange, title, allowEmpty = false) => {
      const inp = el('input', { type: 'number', step: 'any', value: value ?? '', title, placeholder: allowEmpty ? 'auto' : '' });
      inp.onchange = () => {
        const v = inp.value.trim();
        if (v === '' && allowEmpty) onchange(null);
        else if (Number.isFinite(Number(v))) onchange(Number(v));
        else inp.value = value ?? '';
      };
      return inp;
    };
    const row = (label, unit, title, ...inputs) => el('div', { className: 'field', title },
      el('span', { textContent: label }), el('div', { className: 'ctl' }, ...inputs), el('span', { className: 'unit', textContent: unit }));

    for (const [k, label, unit, n, title] of OPTIONS) {
      if (k === 'stateHz' && !this.app.net) continue;  // the server's
      if (k === 'density') this.body.append(el('div', { className: 'sub minor', textContent: 'Medium (air, water…)' }));
      if (k === 'impratio') this.body.append(el('div', { className: 'sub minor', textContent: 'Contacts' }));
      if (k === 'integrator') this.body.append(el('div', { className: 'sub minor', textContent: 'Solver' }));
      if (k === 'timestep') this.body.append(el('div', { className: 'sub minor', textContent: 'Simulation' }));
      const set = (i) => (v) => {
        if (k === 'timestep' && !(v >= 1e-4 && v <= 0.05)) { this.render(); return; }  // MuJoCo needs > 0; 50 ms is no physics any more
        if (/iterations$/.test(k)) v = Math.max(k === 'iterations' ? 1 : 0, Math.round(v));
        if (/tolerance$/.test(k)) v = Math.max(0, v);
        const next = structuredClone(this.app.physics);
        if (!Array.isArray(s[k])) next[k] = v; else next[k][i] = v;
        this.app.setPhysics(next).then(() => this.render());
      };
      let inputs;
      if (n === 'bool') {
        const box = el('input', { type: 'checkbox', checked: !!s[k], title });
        box.onchange = () => set(0)(box.checked);
        inputs = [box];
      } else if (Array.isArray(n)) {
        const sel = el('select', { title });
        const label = (v) => (typeof v !== 'number' ? v : k === 'stateHz' ? String(v) : v === 0 ? 'max' : `${v}×`);
        for (const v of n) sel.append(new Option(label(v), v, false, v === s[k]));
        sel.onchange = () => { set(0)(typeof n[0] === 'number' ? Number(sel.value) : sel.value); sel.blur(); };
        inputs = [sel];
      } else inputs = n === 1 ? [num(s[k], set(0), title)] : s[k].map((v, i) => num(v, set(i), `${title} (${'xyz'[i]})`));
      this.body.append(row(label, unit, title, ...inputs));
    }
    this.body.append(el('div', { className: 'hint', textContent: 'Applied at once (Sleep rebuilds the world). Wind and fluid drag only act with a medium density or viscosity.' }));

    if (this.app.net) {  // network mode: the server's options (server/server.py)
      this.body.append(el('div', { className: 'hint', textContent: 'Shared world: these are the server\'s settings, for everyone. Default geom properties are not available in server mode yet.' }));
      const reset = el('button', { type: 'button', textContent: 'Defaults', title: 'Every option back to its default' });
      reset.onclick = () => this.app.setPhysics({ ...structuredClone(PHYSICS_DEFAULTS), geom: this.app.physics.geom });
      this.body.append(el('div', { className: 'bar' }, reset));
      return;
    }
    this.body.append(el('div', { className: 'sub', textContent: 'Default geom' }));
    const d = this.draft, dirty = JSON.stringify(d) !== JSON.stringify(s.geom);
    for (const [k, label, unit, n, title] of GEOM) {
      const set = (i) => (v) => {
        if (n === 1) d[k] = v; else d[k][i] = v ?? PHYSICS_DEFAULTS.geom[k][i];
        this.render();
      };
      let inputs;
      if (Array.isArray(n)) {
        const sel = el('select', { title });
        for (const v of n) sel.append(new Option(String(v), v, false, v === d[k]));
        sel.onchange = () => { set(0)(Number(sel.value)); sel.blur(); };
        inputs = [sel];
      } else inputs = n === 1 ? [num(d[k], set(0), title, k === 'mass')] : d[k].map((v, i) => num(v, set(i), title));
      this.body.append(row(label, unit, title, ...inputs));
    }
    this.body.append(el('div', { className: 'hint', textContent: 'For geoms that do not set these themselves or through their own default classes; the floor takes the friction. Apply rebuilds the world (masses change).' }));
    const status = el('span', { className: 'note' });
    const apply = el('button', { type: 'button', textContent: 'Apply', disabled: !dirty, className: dirty ? 'primary' : '' });
    apply.onclick = async () => {
      apply.disabled = true;
      const ok = await this.app.setPhysics({ ...structuredClone(this.app.physics), geom: structuredClone(d) });
      if (!ok) status.textContent = 'busy, try again';
      else this.render();
    };
    const reset = el('button', { type: 'button', textContent: 'Defaults', title: 'Every setting back to its default (MuJoCo\'s, but for sleep and noslip)' });
    reset.onclick = async () => {
      this.draft = structuredClone(PHYSICS_DEFAULTS.geom);
      await this.app.setPhysics(structuredClone(PHYSICS_DEFAULTS));
      this.render();
    };
    this.body.append(el('div', { className: 'bar' }, apply, reset, status));
    if (dirty) status.textContent = 'not applied';
  }
}
