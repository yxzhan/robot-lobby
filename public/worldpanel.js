// The World panel: global physics options and default geom properties
// (physics.js). Options apply as soon as a field changes; default geom
// properties change masses, so they wait for Apply (a rebuild).

import { PHYSICS_DEFAULTS } from './physics.js';

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
  ['impratio', 'Impratio', '', 1, 'ratio of frictional to normal constraint impedance (elliptic cones: > 1 makes friction harder)'],
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
      if (k === 'density') this.body.append(el('div', { className: 'sub minor', textContent: 'Medium (air, water…)' }));
      const set = (i) => (v) => {
        const next = structuredClone(this.app.physics);
        if (n === 1) next[k] = v; else next[k][i] = v;
        this.app.setPhysics(next).then(() => this.render());
      };
      const inputs = n === 1 ? [num(s[k], set(0), title)] : s[k].map((v, i) => num(v, set(i), `${title} (${'xyz'[i]})`));
      this.body.append(row(label, unit, title, ...inputs));
    }
    this.body.append(el('div', { className: 'hint', textContent: 'Applied at once. Wind and fluid drag only act with a medium density or viscosity.' }));

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
}
