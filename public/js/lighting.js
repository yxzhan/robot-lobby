// The scene's lighting (the Render panel's Lighting settings), saved in this
// browser: the sun's brightness, direction and color and the ambient light's
// brightness. Brightness is a percentage of the defaults main.js was tuned
// at; the direction is where the sun stands: azimuth (degrees from +x
// toward +y) and elevation above the horizon.

const STORE = 'mujoco-filament/lighting';

export const SUN_LUX = 90000, AMBIENT = 25000;  // at 100 %

export const LIGHTING_DEFAULTS = { sun: 100, azimuth: 217, elevation: 63, color: '#fffbf5', ambient: 100 };

export function loadLighting() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? 'null');
    if (saved) return { ...LIGHTING_DEFAULTS, ...saved };
  } catch { /* no storage */ }
  return { ...LIGHTING_DEFAULTS };
}

export function saveLighting(l) {
  try { localStorage.setItem(STORE, JSON.stringify(l)); } catch { /* no storage */ }
}

// The direction the sunlight travels (unit vector, toward the ground).
export function sunDirection({ azimuth, elevation }) {
  const a = azimuth * Math.PI / 180, e = elevation * Math.PI / 180;
  return [-Math.cos(e) * Math.cos(a), -Math.cos(e) * Math.sin(a), -Math.sin(e)];
}

// '#rrggbb' (sRGB) -> linear [r, g, b], as Filament's lights take it.
export const linearRgb = (hex) => [1, 3, 5].map((i) => Math.pow(parseInt(hex.slice(i, i + 2), 16) / 255, 2.2));
