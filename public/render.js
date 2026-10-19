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
// - textures: the longest side image textures are scaled down to (main.js
//   decodeImage); a change decodes the scene's images again.
//
// The presets leave the resolution at the screen's: on an iPhone 15 it made
// little difference. A browser without saved settings starts on Medium, a
// phone or tablet (a finger for its main pointer) on Low.

const STORE = 'mujoco-filament/render';

export const PRESETS = {
  high: { pixelRatio: 'screen', antialiasing: 'fxaa', ao: 'high', shadows: 'on', textures: 2048 },
  medium: { pixelRatio: 'screen', antialiasing: 'fxaa', ao: 'on', shadows: 'on', textures: 1024 },
  low: { pixelRatio: 'screen', antialiasing: 'fxaa', ao: 'off', shadows: 'off', textures: 512 },
  ultralow: { pixelRatio: 'screen', antialiasing: 'fxaa', ao: 'off', shadows: 'off', textures: 64 },  // Low, but textures
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
  }
}
