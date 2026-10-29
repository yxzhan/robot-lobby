// The world floor's look (the Render panel's Floor settings): a style, its
// cell size, its two colors and a shine. The patterns are drawn here into a
// tileable texture, repeated over the floor: each pattern is worked out once
// as a mix between its two colors and a shade per pixel (patternOf), so a
// new color only mixes the pixels again (floorTexture). Plain is one color,
// none no floor. Shine makes it glossier, toward polished. Only how the
// floor looks: the physics floor (world.js) stays the same.

const STORE = 'mujoco-filament/floor';

// style -> {label, size (m: one cell, tile, plank or texture repeat),
// colors (hex, sRGB: the pattern's two, or plain's one), shine (%, default
// 0), cells (cells one texture covers), roughness}
export const FLOOR_STYLES = {
  grid: { label: 'Grid', size: 1, colors: ['#9ea6b0', '#4d545e'], cells: 1, roughness: 0.7 },
  checker: { label: 'Checker', size: 1, colors: ['#737f8f', '#3f464f'], cells: 2, roughness: 0.6, shine: 100 },
  tiles: { label: 'Tiles', size: 0.5, colors: ['#dbd6cc', '#6b6863'], cells: 4, roughness: 0.35 },
  wood: { label: 'Wood', size: 0.2, colors: ['#a8784d', '#734d2e'], cells: 4, roughness: 0.55 },
  concrete: { label: 'Concrete', size: 2, colors: ['#b1aea8', '#77746e'], cells: 1, roughness: 0.85 },
  plain: { label: 'Plain', size: 1, colors: ['#8c939c'], roughness: 0.7 },
  none: { label: 'None' },
};
// Before there were two colors, one tinted the pattern; its default per
// style (white but for these), to turn a saved tint into two colors.
const OLD_TINT = { checker: '#737f8f', plain: '#8c939c' };

export const floorDefaults = (style = 'grid') => ({
  style, size: FLOOR_STYLES[style].size ?? 1, colors: [...(FLOOR_STYLES[style].colors ?? [])], shine: FLOOR_STYLES[style].shine ?? 0,
});

// hex * tint / base, per channel (sRGB, near enough).
const retint = (hex, tint, base = '#ffffff') => rgbHex(hexRgb(hex).map((v, i) => v * hexRgb(tint)[i] / (hexRgb(base)[i] || 1)));

export function loadFloor() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? 'null');
    if (saved && FLOOR_STYLES[saved.style]) {
      const f = { ...floorDefaults(saved.style), ...saved };
      if (saved.color && !saved.colors) f.colors = f.colors.map((c) => retint(c, saved.color, OLD_TINT[f.style]));  // an older tint
      delete f.color;
      return f;
    }
  } catch { /* no storage */ }
  return floorDefaults();
}

export function saveFloor(f) {
  try { localStorage.setItem(STORE, JSON.stringify(f)); } catch { /* no storage */ }
}

// '#rrggbb' -> [r, g, b] in 0..1 (sRGB), and back.
export const hexRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const rgbHex = (rgb) => '#' + rgb.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0')).join('');

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

// Every pixel of a pattern: t (0: the first color, 1: the second, mixed in
// between) and k (a shade multiplying the mixed color).
function pattern(shade) {
  const t = new Float32Array(N * N), k = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) [t[y * N + x], k[y * N + x]] = shade(x, y);
  }
  return { t, k };
}

// Distance in pixels from x to the nearest multiple of `step`.
const edge = (x, step) => { const m = ((x % step) + step) % step; return Math.min(m, step - m); };

// (x, y) -> [t, k], see pattern.
const SHADERS = {
  // 2 x 2 cells, light and dark.
  checker(x, y) {
    return [(Math.floor(x / (N / 2)) + Math.floor(y / (N / 2))) % 2 ? 0 : 1, 1];
  },

  // One cell: a line at its border and four faint ones across it.
  grid(x, y) {
    const major = Math.min(edge(x, N), edge(y, N));
    const minor = Math.min(edge(x, N / 5), edge(y, N / 5));
    return [Math.max(major < 3 ? 1 : major < 4 ? 0.5 : 0, minor < 1 ? 0.35 : 0), 1];
  },

  // 4 x 4 tiles, each a little different, in grout.
  tiles(x, y) {
    const step = N / 4, i = Math.floor(x / step), j = Math.floor(y / step);
    const grout = Math.min(edge(x, step), edge(y, step));
    if (grout < 3) return [1, 1];
    const k = 0.94 + 0.08 * hash(i, j, 7);
    const n = fbm(x / N, y / N, 16, 3, 11);
    const bevel = grout < 6 ? 0.9 + 0.1 * (grout - 3) / 3 : 1;
    return [0, k * (0.95 + 0.1 * n) * bevel];
  },

  // 4 planks along y, their ends staggered, with grain.
  wood(x, y) {
    const step = N / 4, i = Math.floor(x / step);
    const off = hash(i, 0, 3) * N, len = N / 2;  // two plank lengths per texture
    const yy = y + off, k = Math.floor(yy / len) % 2;  // wraps with the texture
    const seam = Math.min(edge(x, step), edge(yy, len));
    const tone = 0.85 + 0.25 * hash(i, k, 5);
    // Grain: lines along the plank, wavering slowly along it, finer ones between.
    const u = x / step, t = y / N;
    const warp = noise(u * 2, t * 3, 8, 21, 3) * 2.5 + noise(u * 8, t * 12, 32, 23, 12) * 0.4;
    const ring = (u * 7 + warp) % 1;
    const grain = Math.pow(Math.abs(ring * 2 - 1), 3) * 0.8 + 0.2 * noise(u * 40, t * 4, 160, 27, 4);
    return [0.65 * grain * grain, tone * (seam < 1.5 ? 0.45 : seam < 3 ? 0.8 : 1)];
  },

  // Concrete: blotches between light and dark, fine grain and a few dark specks.
  concrete(x, y) {
    const s = x / N, t = y / N;
    const n = 0.6 * fbm(s, t, 4, 5, 41) + 0.4 * fbm(s, t, 64, 2, 51);
    return [1 - n, hash(x, y, 61) > 0.996 ? 0.75 : 1];
  },
};

const patterns = new Map();
const patternOf = (style) => {
  if (!patterns.has(style)) patterns.set(style, pattern(SHADERS[style]));
  return patterns.get(style);
};

// The texture of a pattern style in two colors (hex): {w, h, rgba}. The
// pattern is worked out once; this only mixes the colors (a few ms).
export function floorTexture(style, [a, b]) {
  const { t, k } = patternOf(style), A = hexRgb(a), B = hexRgb(b);
  const rgba = new Uint8Array(N * N * 4);
  for (let p = 0; p < N * N; p++) {
    for (let c = 0; c < 3; c++) rgba[4 * p + c] = Math.max(0, Math.min(255, (A[c] + (B[c] - A[c]) * t[p]) * k[p] * 255));
    rgba[4 * p + 3] = 255;
  }
  return { w: N, h: N, rgba };
}
