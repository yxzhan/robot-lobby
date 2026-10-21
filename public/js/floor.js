// The world floor's look (the World panel's Floor settings): a style, its
// cell size and a color. Checker is the procedural material (checker.mat);
// the other patterns are drawn here into a tileable texture, repeated over
// the floor and tinted by the color; plain is one color, none no floor.
// Only how the floor looks: the physics floor (world.js) stays the same.

const STORE = 'mujoco-filament/floor';

// style -> {label, size (m: one cell, tile, plank or texture repeat),
// color (hex: checker's light cells, the tint of a texture, plain's color),
// cells (cells one texture covers), roughness}
export const FLOOR_STYLES = {
  checker: { label: 'Checker', size: 1, color: '#737f8f' },
  grid: { label: 'Grid', size: 1, color: '#ffffff', cells: 1, roughness: 0.7 },
  tiles: { label: 'Tiles', size: 0.5, color: '#ffffff', cells: 4, roughness: 0.35 },
  wood: { label: 'Wood', size: 0.2, color: '#ffffff', cells: 4, roughness: 0.55 },
  concrete: { label: 'Concrete', size: 2, color: '#ffffff', cells: 1, roughness: 0.85 },
  plain: { label: 'Plain', size: 1, color: '#8c939c', roughness: 0.7 },
  none: { label: 'None' },
};

export const floorDefaults = (style = 'checker') => ({ style, size: FLOOR_STYLES[style].size ?? 1, color: FLOOR_STYLES[style].color ?? '#ffffff' });

export function loadFloor() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? 'null');
    if (saved && FLOOR_STYLES[saved.style]) return { ...floorDefaults(saved.style), ...saved };
  } catch { /* no storage */ }
  return floorDefaults();
}

export function saveFloor(f) {
  try { localStorage.setItem(STORE, JSON.stringify(f)); } catch { /* no storage */ }
}

// '#rrggbb' -> [r, g, b] in 0..1 (sRGB)
export const hexRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);

// ---------------------------------------------------------------------------
// Textures: RGBA8 pixels, tileable (they wrap at the edges)

const N = 512;

// Deterministic hash of a lattice point -> 0..1.
function hash(i, j, seed) {
  let h = (i * 374761393 + j * 668265263 + seed * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// Smooth value noise at (x, y) in lattice units, repeating every `p` cells
// (along y every `q`).
function noise(x, y, p, seed = 0, q = p) {
  const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  const at = (a, b) => hash(((a % p) + p) % p, ((b % q) + q) % q, seed);
  const a = at(i, j), b = at(i + 1, j), c = at(i, j + 1), d = at(i + 1, j + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// Octaves of noise over the texture (s, t in 0..1), base period `p`: 0..1.
function fbm(s, t, p, octaves, seed) {
  let sum = 0, amp = 0.5, norm = 0;
  for (let o = 0; o < octaves; o++, p *= 2, amp *= 0.5) {
    sum += amp * noise(s * p, t * p, p, seed + o);
    norm += amp;
  }
  return sum / norm;
}

function pixels(shade) {
  const rgba = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const [r, g, b] = shade(x, y), k = 4 * (y * N + x);
      rgba[k] = Math.max(0, Math.min(255, r * 255));
      rgba[k + 1] = Math.max(0, Math.min(255, g * 255));
      rgba[k + 2] = Math.max(0, Math.min(255, b * 255));
      rgba[k + 3] = 255;
    }
  }
  return rgba;
}

const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const scale = (c, k) => c.map((v) => v * k);

// Distance in pixels from x to the nearest multiple of `step`.
const edge = (x, step) => { const m = ((x % step) + step) % step; return Math.min(m, step - m); };

const SHADERS = {
  // One cell: a line at its border and four faint ones across it.
  grid(x, y) {
    const base = [0.62, 0.65, 0.69], line = [0.3, 0.33, 0.37];
    const major = Math.min(edge(x, N), edge(y, N));
    const minor = Math.min(edge(x, N / 5), edge(y, N / 5));
    const t = Math.max(major < 3 ? 1 : major < 4 ? 0.5 : 0, minor < 1 ? 0.35 : 0);
    return mix(base, line, t);
  },

  // 4 x 4 tiles, each a little different, in grout.
  tiles(x, y) {
    const step = N / 4, i = Math.floor(x / step), j = Math.floor(y / step);
    const grout = Math.min(edge(x, step), edge(y, step));
    if (grout < 3) return [0.42, 0.41, 0.39];
    const base = [0.86, 0.84, 0.8], k = 0.94 + 0.08 * hash(i, j, 7);
    const n = fbm(x / N, y / N, 16, 3, 11);
    const bevel = grout < 6 ? 0.9 + 0.1 * (grout - 3) / 3 : 1;
    return scale(base, k * (0.95 + 0.1 * n) * bevel);
  },

  // 4 planks along y, their ends staggered, with grain.
  wood(x, y) {
    const step = N / 4, i = Math.floor(x / step);
    const off = hash(i, 0, 3) * N, len = N / 2;  // two plank lengths per texture
    const yy = y + off, k = Math.floor(yy / len) % 2;  // wraps with the texture
    const seam = Math.min(edge(x, step), edge(yy, len));
    const tone = 0.85 + 0.25 * hash(i, k, 5);
    const light = [0.66, 0.47, 0.3], dark = [0.45, 0.3, 0.18];
    // Grain: lines along the plank, wavering slowly along it, finer ones between.
    const u = x / step, t = y / N;
    const warp = noise(u * 2, t * 3, 8, 21, 3) * 2.5 + noise(u * 8, t * 12, 32, 23, 12) * 0.4;
    const ring = (u * 7 + warp) % 1;
    const grain = Math.pow(Math.abs(ring * 2 - 1), 3) * 0.8 + 0.2 * noise(u * 40, t * 4, 160, 27, 4);
    let c = scale(mix(light, dark, 0.65 * grain * grain), tone);
    if (seam < 1.5) c = scale(c, 0.45);
    else if (seam < 3) c = scale(c, 0.8);
    return c;
  },

  // Concrete: blotches, fine grain and a few dark specks.
  concrete(x, y) {
    const s = x / N, t = y / N;
    const n = 0.6 * fbm(s, t, 4, 5, 41) + 0.4 * fbm(s, t, 64, 2, 51);
    const speck = hash(x, y, 61) > 0.996 ? 0.75 : 1;
    return scale([0.58, 0.57, 0.55], (0.8 + 0.4 * n) * speck);
  },
};

const cache = new Map();

// The texture of a pattern style: {w, h, rgba}, made once.
export function floorTexture(style) {
  if (!cache.has(style)) cache.set(style, { w: N, h: N, rgba: pixels(SHADERS[style]) });
  return cache.get(style);
}
