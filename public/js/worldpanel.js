// The World panel: global physics options and default geom properties
// (physics.js), how the floor looks (floor.js) and the lighting
// (lighting.js). Options, the floor and the lighting
// apply as soon as a field changes; default geom properties change masses,
// so they wait for Apply (a rebuild).

import { CONES, PHYSICS_DEFAULTS, SPEEDS, STATE_RATES } from './physics.js';
import { FLOOR_STYLES, floorDefaults } from './floor.js';
import { LIGHTING_DEFAULTS } from './lighting.js';

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
  ['timestep', 'Timestep', 's', 1, 'physics step: larger costs less per simulated second but is less stable and accurate (walking policies were trained at 0.002-0.005 s)'],
  ['speed', 'Speed', '', SPEEDS, 'the most simulated time per second of wall time (1×: real time, max: as fast as the computer goes); a world that cannot keep up runs slower'],
  ['stateHz', 'Sync rate', 'Hz', STATE_RATES, 'shared world: states the server sends each page per second (more: smoother and fresher, more traffic for everyone)'],
  ['sleep', 'Sleep', '', 'bool', 'bodies at rest stop being simulated until something touches, pulls or moves them (robots with actuators never sleep); saves most of the physics time of large scenes. Changing it rebuilds the world'],
];
const GEOM = [
  ['density', 'Density', 'kg/m³', 1, 'for geoms that set neither density nor mass'],
  ['mass', 'Mass', 'kg', 1, 'per geom, for geoms that set neither (empty: from density)'],
  ['friction', 'Friction', '', 3, 'sliding, torsional, rolling, for geoms that set none'],
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
      if (k === 'timestep') this.body.append(el('div', { className: 'sub minor', textContent: 'Simulation' }));
      const set = (i) => (v) => {
        if (k === 'timestep' && !(v >= 1e-4 && v <= 0.05)) { this.render(); return; }  // MuJoCo needs > 0; 50 ms is no physics any more
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
    this.floor(row, num);
    this.lighting(row);

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
      const inputs = n === 1 ? [num(d[k], set(0), title, k === 'mass')] : d[k].map((v, i) => num(v, set(i), title));
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
    const reset = el('button', { type: 'button', textContent: 'MuJoCo defaults', title: 'Every setting back to MuJoCo\'s defaults' });
    reset.onclick = async () => {
      this.draft = structuredClone(PHYSICS_DEFAULTS.geom);
      await this.app.setPhysics(structuredClone(PHYSICS_DEFAULTS));
      this.render();
    };
    this.body.append(el('div', { className: 'bar' }, apply, reset, status));
    if (dirty) status.textContent = 'not applied';
  }

  // The Lighting section: sliders for the sun and the ambient light, live
  // while dragging (this page only).
  lighting(row) {
    const l = this.app.lighting;
    this.body.append(el('div', { className: 'sub', textContent: 'Lighting' }));
    const slider = (k, label, min, max, step, unit, title) => {
      const out = el('span', { className: 'unit', textContent: `${l[k]}${unit}` });
      const inp = el('input', { type: 'range', min, max, step, value: l[k], title });
      inp.oninput = () => {
        this.app.setLighting({ ...this.app.lighting, [k]: Number(inp.value) });
        out.textContent = `${inp.value}${unit}`;
      };
      const f = el('div', { className: 'field', title }, el('span', { textContent: label }), el('div', { className: 'ctl' }, inp), out);
      this.body.append(f);
    };
    slider('sun', 'Sun', 0, 300, 5, '%', 'brightness of the sunlight (100 %: the default)');
    slider('azimuth', 'Azimuth', 0, 360, 1, '°', 'where the sun stands around the world: degrees from +x toward +y');
    slider('elevation', 'Elevation', 5, 90, 1, '°', 'how high the sun stands above the horizon: low makes long shadows, 90 straight down');
    slider('ambient', 'Ambient', 0, 300, 5, '%', 'light from all around (the sky): fills the shadows (100 %: the default)');
    const title = 'the sunlight\'s color';
    const color = el('input', { type: 'color', value: l.color, title });
    color.oninput = () => this.app.setLighting({ ...this.app.lighting, color: color.value });
    const reset = el('button', { type: 'button', textContent: 'Default', title: 'every lighting setting back to its default' });
    reset.onclick = () => { this.app.setLighting({ ...LIGHTING_DEFAULTS }); this.render(); };
    this.body.append(row('Sun color', '', title, color, reset));
  }

  // The Floor section: style, cell size and color (this page only).
  floor(row, num) {
    const f = this.app.floorSettings, set = (change) => { this.app.setFloor({ ...this.app.floorSettings, ...change }); this.render(); };
    this.body.append(el('div', { className: 'sub', textContent: 'Floor' }));
    const sel = el('select', { title: 'how the floor looks (not its physics)' });
    for (const [k, s] of Object.entries(FLOOR_STYLES)) sel.append(new Option(s.label, k, false, k === f.style));
    sel.onchange = () => { set(floorDefaults(sel.value)); };  // each style starts from its own size and color
    this.body.append(row('Style', '', sel.title, sel));
    if (f.style === 'none') return;
    if (f.style !== 'plain') {
      const what = { checker: 'one checker cell', grid: 'one grid cell', tiles: 'one tile', wood: 'one plank\'s width', concrete: 'one repeat of the pattern' }[f.style];
      this.body.append(row('Size', 'm', what, num(f.size, (v) => { if (v > 0.01 && v <= 100) set({ size: v }); else this.render(); }, what)));
    }
    const title = f.style === 'checker' ? 'the light cells (the dark ones are darker)' : f.style === 'plain' ? 'the floor\'s color' : 'tints the pattern (white: as drawn)';
    const color = el('input', { type: 'color', value: f.color, title });
    color.oninput = () => this.app.setFloor({ ...this.app.floorSettings, color: color.value });
    color.onchange = () => this.render();
    const reset = el('button', { type: 'button', textContent: 'Default', title: 'this style\'s own size and color' });
    reset.onclick = () => set(floorDefaults(f.style));
    this.body.append(row('Color', '', title, color, reset));
    const tip = 'glossier toward polished: reflects the sky, and the robots with Reflections on (Render panel)';
    const shine = el('input', { type: 'range', min: 0, max: 100, step: 5, value: f.shine ?? 0, title: tip });
    const out = el('span', { className: 'unit', textContent: `${f.shine ?? 0}%` });
    shine.oninput = () => { this.app.setFloor({ ...this.app.floorSettings, shine: Number(shine.value) }); out.textContent = `${shine.value}%`; };
    this.body.append(el('div', { className: 'field', title: tip }, el('span', { textContent: 'Shine' }), el('div', { className: 'ctl' }, shine), out));
  }
}
