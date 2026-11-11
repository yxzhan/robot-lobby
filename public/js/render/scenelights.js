// The model's own lights (MJCF <light>) as Filament lights, while the Render
// panel's Scene lights is on (lighting.js; off by default: the global sun and
// ambient light every scene alike). The global light is switched on its own
// (Global light): indoors, the scene's lights alone (or with a dim ambient
// light) show a room as lit, where the ambient light, the sky, washes it
// out, its glossy floor mirroring a blue sky through the roof. Spot, point
// and directional lights; image lights are left out. They follow their
// bodies (data.light_xpos, light_xdir).
//
// Shadows: a light with castshadow (MJCF's default, as in MuJoCo) casts
// them, as in MuJoCo 4's own Filament renderer, but only the brightest few
// (the Render panel's Light shadows, setShadows; 0 while the Shadows setting
// is off), each a shadow pass more of the scene: a lab twin has 43 ceiling
// panels. Their maps are mjrf's: filament.shadows.map_size of the scene's
// look (setShadowMap), else 2048², a directional light's in 4 cascades.
// PCF only: VSM / PCSS (soft shadows, a light's bulbradius) draw no shadows
// in Filament.js 1.80's WebGL backend.
//
// Brightness: a light with an intensity (MuJoCo 3.2+: candela, lux for a
// directional one) is that times SCENE_GAIN, which makes up for the
// camera's exposure (set for the sun: a twin's ceiling panels at their
// real candela leave a lab dim), tuned on the AICOR lab twin's 43 panels.
// One without (most MJCF, from before intensities), by its diffuse color's
// brightest channel: a directional one at MuJoCo's default 0.7 about as
// bright as the sun (LEGACY_LUX), a spot or point one LEGACY_CD. The
// panel's Scene % scales them all.
// Network mode: the server's world tables carry the lights (server.py
// world_tables), the page places them by their bodies (light_pos, light_dir
// in the body's frame, as mode "fixed"; a tracking light is placed so too).
// A server from before has none: no scene lights.

export const SCENE_GAIN = 6;
const LEGACY_CD = 30000, LEGACY_LUX = 130000;  // per unit of diffuse

// Light `id`'s world position and direction from its body's pose (data.xpos,
// xmat), where there is no data.light_xpos (network mode).
function placed(model, data, id) {
  const b = model.light_bodyid[id], R = data.xmat.subarray(9 * b, 9 * b + 9), t = data.xpos.subarray(3 * b, 3 * b + 3);
  const p = model.light_pos.subarray(3 * id, 3 * id + 3), d = model.light_dir.subarray(3 * id, 3 * id + 3);
  const turn = (v) => [0, 1, 2].map((r) => R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2]);
  const rp = turn(p);
  return [rp.map((v, r) => v + t[r]), turn(d)];
}

export class SceneLights {
  constructor(engine, scene) {
    this.engine = engine;
    this.scene = scene;
    this.lights = [];    // {entity, id (the model's light), type}
    this.key = null;     // the model and settings the lights were made for
    this.shadowCount = 0;   // how many lights may cast shadows (setShadows)
    this.mapSize = 2048;    // their shadow maps' size (setShadowMap)
  }

  // Their shadow maps `size` square (made again if it changes).
  setShadowMap(size) {
    if (size === this.mapSize) return;
    this.mapSize = size;
    this.key = null;
  }

  // At most `count` of the lights cast shadows.
  setShadows(count) {
    if (count === this.shadowCount) return;
    this.shadowCount = count;
    this.key = null;  // made again on the next sync
  }

  // Makes, updates or removes the lights for `model` / `data` (null: none)
  // and the lighting settings `l` (l.scene: on, l.sceneLevel: %).
  sync(model, data, l) {
    const on = !!(l.scene && model?.nlight && model.light_type && (data?.light_xpos || data?.xmat));
    if (!on || this.key?.model !== model || this.key.level !== l.sceneLevel) {
      this.clear();
      if (!on) return;
      this.build(model, l.sceneLevel / 100);
      this.key = { model, level: l.sceneLevel };
    }
    const lm = this.engine.getLightManager();
    for (const { entity, id, type } of this.lights) {
      const [pos, dir] = data.light_xpos
        ? [Array.from(data.light_xpos.subarray(3 * id, 3 * id + 3)), Array.from(data.light_xdir.subarray(3 * id, 3 * id + 3))]
        : placed(model, data, id);
      const inst = lm.getInstance(entity);
      if (type !== 1) lm.setPosition(inst, pos);
      if (type !== 2) lm.setDirection(inst, dir);
      inst.delete();
    }
  }

  build(model, level) {
    const F = Filament, T = F.LightManager$Type;
    // The lights to make, and how bright (to pick the shadow casters by).
    const list = [];
    for (let id = 0; id < model.nlight; id++) {
      const type = model.light_type[id];
      if (type > 2 || !model.light_active[id]) continue;  // an image light, or one switched off
      const diffuse = Array.from(model.light_diffuse.subarray(3 * id, 3 * id + 3));
      const peak = Math.max(...diffuse);
      if (peak <= 0) continue;
      const intensity = model.light_intensity?.[id] ?? 0;
      // Lux, or candela for a spot or point light; for ranking, a candela
      // counts as the lux it gives at 2 m.
      const amount = type === 1 ? (intensity > 0 ? intensity * SCENE_GAIN : peak * LEGACY_LUX) * level
        : (intensity > 0 ? intensity * SCENE_GAIN : peak * LEGACY_CD) * level;
      list.push({ id, type, diffuse, peak, amount, rank: type === 1 ? amount : amount / 4 });
    }
    // castshadow: MuJoCo's default is on (a model from before the field has none: on).
    const casters = new Set(list.filter((x) => (model.light_castshadow?.[x.id] ?? 1) !== 0)
      .sort((a, b) => b.rank - a.rank).slice(0, this.shadowCount).map((x) => x.id));
    for (const { id, type, diffuse, peak, amount } of list) {
      const color = diffuse.map((c) => c / peak);
      const entity = F.EntityManager.get().create();
      let b;
      if (type === 1) {
        b = F.LightManager.Builder(T.DIRECTIONAL).intensity(amount);
      } else {
        // Filament's point and (unfocused) spot lights take lumens: 4π per candela.
        b = F.LightManager.Builder(type === 2 ? T.POINT : T.SPOT).intensity(4 * Math.PI * amount)
          .falloff(model.light_range?.[id] || 10);
        if (type === 0) {
          // MuJoCo's cutoff is the cone's half angle in degrees; a larger
          // exponent gathers the light to the middle (a narrower full-bright core).
          const outer = Math.min(89, model.light_cutoff[id]) * Math.PI / 180;
          b = b.spotLightCone(outer / (1 + model.light_exponent[id] / 4), outer);
        }
      }
      b = b.color(color).castShadows(casters.has(id));
      if (casters.has(id)) {
        b = b.shadowOptions({ mapSize: this.mapSize, shadowCascades: type === 1 ? 4 : 1 });
      }
      b.build(this.engine, entity);
      this.scene.addEntity(entity);
      this.lights.push({ entity, id, type });
    }
  }

  clear() {
    for (const { entity } of this.lights) {
      this.scene.remove(entity);
      this.engine.destroyEntity(entity);
    }
    this.lights = [];
    this.key = null;
  }
}
