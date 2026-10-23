// The Render panel: drawing quality against GPU load, saved in this browser
// and applied at once (App.setQuality, main.js):
//
// - pixel ratio: canvas pixels per CSS pixel, at most the screen's own (a
//   2× screen draws 4× the pixels of a 1× one; 1.5 draws 56 % of 2×);
// - anti-aliasing: none, FXAA (a cheap post pass) or MSAA 4× (sharper edges,
//   4 samples per pixel of memory and fill);
// - ambient occlusion: Filament's SSAO, off, on (half resolution, low
//   quality) or high (full resolution, high quality);
// - shadows: the sun's shadow map, on or off (its size cannot be set: this
//   Filament build does not bind LightManager's shadow options);
// - reflections: Filament's screen-space reflections, on or off (the sky's
//   reflections, IBL, are always on: they cost nothing extra);
// - textures: the longest side image textures are scaled down to (main.js
//   decodeImage); a change decodes the scene's images again.
//
// The presets leave the resolution at the screen's: on an iPhone 15 it made
// little difference. A browser without saved settings starts on Medium, a
// phone or tablet (a finger for its main pointer) on Low.
//
// Below them: how the floor looks (floor.js), the lighting (lighting.js)
// and whether the camera follows the selection, each saved in this browser
// and applied as soon as a field changes.

import { FLOOR_STYLES, floorDefaults } from './floor.js';
import { LIGHTING_DEFAULTS } from './lighting.js';

const STORE = 'mujoco-filament/render';

export const PRESETS = {
  high: { pixelRatio: 'screen', antialiasing: 'msaa', ao: 'high', shadows: 'on', reflections: 'on', textures: 2048 },
  medium: { pixelRatio: 'screen', antialiasing: 'fxaa', ao: 'on', shadows: 'on', reflections: 'off', textures: 1024 },
  low: { pixelRatio: 'screen', antialiasing: 'none', ao: 'off', shadows: 'off', reflections: 'off', textures: 512 },
  ultralow: { pixelRatio: 'screen', antialiasing: 'none', ao: 'off', shadows: 'off', reflections: 'off', textures: 64 },  // Low, but textures
};
const PRESET_NAMES = { high: 'High', medium: 'Medium', low: 'Low', ultralow: 'Ultra low' };
const DEFAULT = matchMedia('(pointer: coarse)').matches ? 'low' : 'medium';

// [key, label, [[value, text]], tooltip]
const FIELDS = [
  ['pixelRatio', 'Resolution', [['screen', 'screen'], [2, '2×'], [1.5, '1.5×'], [1, '1×'], [0.75, '0.75×'], [0.5, '0.5×']],
    'canvas pixels per CSS pixel, at most the screen\'s own (now %): fewer pixels to shade, softer image'],
  ['antialiasing', 'Anti-alias', [['none', 'off'], ['fxaa', 'FXAA'], ['msaa', 'MSAA 4×']],
    'smooths jagged edges: FXAA is a cheap post pass, MSAA 4× is sharper but takes 4 samples per pixel'],
  ['ao', 'Occlusion', [['off', 'off'], ['on', 'on'], ['high', 'high']],
    'ambient occlusion (SSAO): contact darkening in corners and under objects; high is full resolution'],
  ['shadows', 'Shadows', [['off', 'off'], ['on', 'on']],
    'the sun\'s shadows: off saves drawing the whole scene a second time, into the shadow map'],
  ['reflections', 'Reflections', [['off', 'off'], ['on', 'on']],
    'screen-space reflections: shiny surfaces (a floor with Shine, below) mirror what is on screen; one more pass over the image'],
  ['textures', 'Textures', [[64, '64'], [512, '512'], [1024, '1024'], [2048, '2048']],
    'longest side of image textures in GPU memory: less memory and bandwidth, blurrier surfaces'],
];

export function loadQuality() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? 'null');
    if (saved) return { ...PRESETS[DEFAULT], ...saved };
  } catch { /* no storage */ }
  return { ...PRESETS[DEFAULT] };
}

function saveQuality(q) {
  try { localStorage.setItem(STORE, JSON.stringify(q)); } catch { /* no storage */ }
}

// The preset `q` matches, or 'custom'.
export const presetOf = (q) => Object.keys(PRESETS).find((p) => FIELDS.every(([k]) => PRESETS[p][k] === q[k])) ?? 'custom';

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

export class RenderPanel {
  // `app.quality` holds the settings, `app.setQuality(next)` applies them.
  constructor(app, root) {
    this.app = app;
    this.body = root.querySelector('.body');
    this.render();
  }

  set(next) {
    saveQuality(next);
    this.app.setQuality(next);
    this.render();
  }

  render() {
    const q = this.app.quality;
    this.body.replaceChildren();
    const row = (label, title, ctl) => el('div', { className: 'field', title }, el('span', { textContent: label }), el('div', { className: 'ctl' }, ctl));
    const select = (options, value, title, onchange) => {
      const sel = el('select', { title });
      for (const [v, text] of options) sel.append(new Option(text, String(v), false, v === value));
      sel.onchange = () => { onchange(options.find(([v]) => String(v) === sel.value)[0]); sel.blur(); };
      return sel;
    };
    const preset = presetOf(q);
    const presets = [...Object.entries(PRESET_NAMES), ...(preset === 'custom' ? [['custom', 'Custom']] : [])];
    this.body.append(row('Preset', 'all of the settings below at once', select(presets, preset, 'all of the settings below at once',
      (p) => p !== 'custom' && this.set({ ...PRESETS[p] }))));
    const screen = window.devicePixelRatio || 1;
    for (const [k, label, options, title] of FIELDS) {
      const tip = title.replace('%', `${+screen.toFixed(2)}×`);
      this.body.append(row(label, tip, select(options, q[k], tip, (v) => this.set({ ...this.app.quality, [k]: v }))));
    }
    const { width: w, height: h } = this.app.canvas;
    this.body.append(el('div', { className: 'hint', textContent: `Drawing ${w}×${h} px. Lower is lighter on the GPU.` }));
    // The rest: a label, its inputs and a unit per row.
    const num = (value, onchange, title) => {
      const inp = el('input', { type: 'number', step: 'any', value: value ?? '', title });
      inp.onchange = () => {
        if (Number.isFinite(Number(inp.value.trim())) && inp.value.trim() !== '') onchange(Number(inp.value));
        else inp.value = value ?? '';
      };
      return inp;
    };
    const field = (label, unit, title, ...inputs) => el('div', { className: 'field', title },
      el('span', { textContent: label }), el('div', { className: 'ctl' }, ...inputs), el('span', { className: 'unit', textContent: unit }));
    this.floor(field, num);
    this.lighting(field);
    this.camera(field);
  }

  // The Camera section, last: whether the camera follows the selection
  // (App.setFollow; saved in this browser, off by default).
  camera(row) {
    this.body.append(el('div', { className: 'sub', textContent: 'Camera' }));
    const title = 'The camera keeps following the body selected with a double-click; off: it only turns to it once';
    const box = el('input', { type: 'checkbox', checked: !!this.app.follow, title });
    box.onchange = () => this.app.setFollow(box.checked);
    this.body.append(row('Follow selection', '', title, box));
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

  // The Floor section: style, cell size, colors and shine (this page only).
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
    // Its colors side by side (the pattern's two, plain's one), then Default.
    const pickers = f.colors.map((value, i) => {
      const color = el('input', { type: 'color', value, title: f.colors.length > 1 ? `color ${i + 1} of the pattern` : 'the floor\'s color' });
      color.oninput = () => {
        const colors = [...this.app.floorSettings.colors];
        colors[i] = color.value;
        this.app.setFloor({ ...this.app.floorSettings, colors });
      };
      return color;
    });
    const reset = el('button', { type: 'button', textContent: 'Default', title: 'this style\'s own size, colors and shine' });
    reset.onclick = () => set(floorDefaults(f.style));
    this.body.append(row('Color', '', 'the floor\'s colors', ...pickers, reset));
    const tip = 'glossier toward polished: reflects the sky, and the robots with Reflections on (above)';
    const shine = el('input', { type: 'range', min: 0, max: 100, step: 5, value: f.shine ?? 0, title: tip });
    const out = el('span', { className: 'unit', textContent: `${f.shine ?? 0}%` });
    shine.oninput = () => { this.app.setFloor({ ...this.app.floorSettings, shine: Number(shine.value) }); out.textContent = `${shine.value}%`; };
    this.body.append(el('div', { className: 'field', title: tip }, el('span', { textContent: 'Shine' }), el('div', { className: 'ctl' }, shine), out));
  }
}
