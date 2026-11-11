// How MuJoCo 4's own Filament renderer (mjrf, src/render/filament) draws a
// model, as far as Filament.js 1.53.4 can follow it:
//
// - materials (geomLook): a material without metallic / roughness (and
//   without PBR texture layers) is drawn with Filament's specular-glossiness
//   model, MuJoCo's classic look (mjrf's phong_*.mat): specular and
//   glossiness from the material's specular and shininess times the model's
//   filament.phong.* multipliers (0.2 and 0.1 by default), emission times
//   filament.phong.emissive_multiplier (0.3) as an emissive color. With
//   metallic or roughness, or a normal / ORM / occlusion / roughness /
//   metallic / emissive layer, PBR, the layers in lit_maps.mat. The Render
//   panel's Materials switches to "PBR" for the page's older look: PBR for
//   every geom, roughness estimated from the classic attributes (surface()).
// - the model's own render settings (renderSettings): <custom> <text> /
//   <numeric> elements named filament.*, the same names mjrf reads: color
//   grading (filament.cg.*), fog, bloom, vignette, ambient occlusion
//   (filament.ao.*). applySceneLook puts them on a view.
//
// What Filament.js 1.53.4 cannot do of it: the PBR neutral tone mapper (it
// has ACES, ACES legacy, filmic and linear: pbr_neutral leaves the default),
// shadow options (ShadowOptions is not bound: no filament.shadows.map_size,
// no cascades), shadow types other than PCF (no View.setShadowType, so no
// soft PCSS shadows and no light bulbradius) and image lights (mjLIGHT_IMAGE).

export const ROLE = { RGB: 1, OCCLUSION: 2, ROUGHNESS: 3, METALLIC: 4, NORMAL: 5, OPACITY: 6, EMISSIVE: 7, RGBA: 8, ORM: 9 };
export const NROLE = 10;

// The settings a model gives in <custom>: filament.* name -> string (text)
// or number array (numeric). A network-mode model (the server's render
// data) has them as `filament` already.
export function renderSettings(mujoco, model) {
  if (model.filament) return new Map(Object.entries(model.filament));
  const out = new Map();
  const T = mujoco?.mjtObj;
  if (!T?.mjOBJ_TEXT || !T.mjOBJ_NUMERIC) return out;  // (network mode's stand-in module has no such objects)
  const name = (type, i) => mujoco.mj_id2name(model, type, i) ?? '';
  for (let i = 0; i < (model.ntext ?? 0); i++) {
    const n = name(T.mjOBJ_TEXT.value, i);
    if (!n.startsWith('filament.')) continue;
    const a = model.text_adr[i], len = Math.max(0, model.text_size[i] - 1);
    out.set(n, new TextDecoder().decode(Uint8Array.from(model.text_data.subarray(a, a + len))).trim());
  }
  for (let i = 0; i < (model.nnumeric ?? 0); i++) {
    const n = name(T.mjOBJ_NUMERIC.value, i);
    if (!n.startsWith('filament.')) continue;
    const a = model.numeric_adr[i];
    out.set(n, Array.from(model.numeric_data.subarray(a, a + model.numeric_size[i])));
  }
  return out;
}

// A setting as a number (its first), a vector of `n`, a bool or a string;
// `dflt` when the model does not set it (or sets it as the wrong kind).
export function num(s, key, dflt) {
  const v = s.get(key);
  const x = Array.isArray(v) ? v[0] : typeof v === 'string' ? Number(v) : NaN;
  return Number.isFinite(x) ? x : dflt;
}
export function vec(s, key, n, dflt) {
  let v = s.get(key);
  if (typeof v === 'string') v = v.split(/[\s,]+/).filter(Boolean).map(Number);
  return Array.isArray(v) && v.length >= n && v.slice(0, n).every(Number.isFinite) ? v.slice(0, n) : dflt;
}
export function bool(s, key, dflt) {
  const v = s.get(key);
  if (v === undefined) return dflt;
  if (Array.isArray(v)) return v[0] !== 0;
  return !/^(false|0|off|no|disable|disabled)$/i.test(String(v));
}
export function str(s, key, dflt) {
  const v = s.get(key);
  return typeof v === 'string' && v ? v.toLowerCase() : dflt;
}

// The texture of `role` of material `matid`, -1 if none (2D textures only).
export function roleTex(model, matid, role) {
  if (matid < 0 || !model.nmat) return -1;
  const nrole = model.mat_texid.length / model.nmat;
  if (role >= nrole) return -1;
  const t = model.mat_texid[nrole * matid + role];
  return t >= 0 && model.tex_type[t] === 0 ? t : -1;
}

// How geom `g` is drawn: {
//   kind: 'phong' | 'pbr', maps (PBR with texture layers: lit_maps.mat),
//   rgb (texture id of the color layer, -1: none), normal, orm, occlusion,
//   roughnessTex, metallicTex, emissiveTex (texture ids, -1: none),
//   specular, glossiness (phong), roughness, metallic (pbr),
//   emissive: linear rgb, }
// `style` is the Render panel's Materials ('mujoco' or 'pbr'), `settings`
// the model's (renderSettings), `rgba` the geom's color as MuJoCo gives it
// (sRGB, like the rest of the page), `surface` main.js's estimated PBR
// surface of the material (surface()).
export function geomLook(model, matid, rgba, style, settings, surface) {
  const at = (field, dflt) => (matid >= 0 && model[field]?.length > matid ? model[field][matid] : dflt);
  const tex = (role) => roleTex(model, matid, role);
  let rgb = tex(ROLE.RGB);
  if (rgb < 0) rgb = tex(ROLE.RGBA);
  const look = {
    rgb, normal: tex(ROLE.NORMAL), orm: tex(ROLE.ORM), occlusion: tex(ROLE.OCCLUSION),
    roughnessTex: tex(ROLE.ROUGHNESS), metallicTex: tex(ROLE.METALLIC), emissiveTex: tex(ROLE.EMISSIVE),
  };
  const layers = look.normal >= 0 || look.orm >= 0 || look.occlusion >= 0 || look.roughnessTex >= 0 || look.metallicTex >= 0 || look.emissiveTex >= 0;
  const metallic = at('mat_metallic', -1), roughness = at('mat_roughness', -1);
  const pbr = style === 'pbr' || layers || metallic >= 0 || roughness >= 0;
  const emission = at('mat_emission', 0) * num(settings, 'filament.phong.emissive_multiplier', 0.3);
  // sRGB color -> linear, for the emissive (a linear color in Filament).
  const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  // With an emissive layer the layer is the color (times the factor), as mjrf's pbr.mat.
  look.emissive = look.emissiveTex >= 0 ? [emission, emission, emission] : [0, 1, 2].map((k) => lin(rgba[k]) * emission);
  if (pbr) {
    look.kind = 'pbr';
    look.maps = layers;
    // Unset ones as main.js estimates them (metallic 0, roughness from shininess, …);
    // with an ORM / roughness / metallic layer the factor multiplies it (1: the layer's own).
    look.roughness = roughness >= 0 ? Math.min(1, Math.max(0.03, roughness))
      : (look.orm >= 0 || look.roughnessTex >= 0) ? 1 : surface.roughness;
    look.metallic = metallic >= 0 ? Math.min(1, metallic) : (look.orm >= 0 || look.metallicTex >= 0) ? 1 : surface.metallic;
  } else {
    look.kind = 'phong';
    look.maps = false;
    // mjrf: a geom without a material takes specular 0.5, shininess 0.5.
    look.specular = at('mat_specular', 0.5) * num(settings, 'filament.phong.specular_multiplier', 0.2);
    look.glossiness = Math.min(1, at('mat_shininess', 0.5) * num(settings, 'filament.phong.shininess_multiplier', 0.1));
  }
  return look;
}

// The scene's color grading (filament.cg.*: tone mapping, exposure, …) as a
// Filament ColorGrading, shared by every view; null when the settings have none.
export function buildColorGrading(F, engine, s) {
  const keys = [...s.keys()];
  if (!keys.some((k) => k.startsWith('filament.cg.'))) return null;
  const TM = F.ColorGrading$ToneMapping;
  const tm = { aces: TM.ACES, aces_legacy: TM.ACES_LEGACY, filmic: TM.FILMIC, linear: TM.LINEAR }[str(s, 'filament.cg.tone_mapping', '')];
  let b = F.ColorGrading.Builder();
  if (tm !== undefined) b = b.toneMapping(tm);
  b = b.exposure(num(s, 'filament.cg.exposure', 0))
    .contrast(num(s, 'filament.cg.contrast', 1))
    .vibrance(num(s, 'filament.cg.vibrance', 1))
    .saturation(num(s, 'filament.cg.saturation', 1))
    .whiteBalance(num(s, 'filament.cg.temperature', 0), num(s, 'filament.cg.tint', 0))
    .luminanceScaling(bool(s, 'filament.cg.luminance_scaling', false))
    .gamutMapping(bool(s, 'filament.cg.gamut_mapping', false));
  const v3 = (k, d) => vec(s, k, 3, d), v4 = (k, d) => vec(s, k, 4, d);
  if (keys.some((k) => /^filament\.cg\.(slope|offset|power)$/.test(k))) {
    b = b.slopeOffsetPower(v3('filament.cg.slope', [1, 1, 1]), v3('filament.cg.offset', [0, 0, 0]), v3('filament.cg.power', [1, 1, 1]));
  }
  if (keys.some((k) => /^filament\.cg\.(shadows|midtones|highlights|tonal_ranges)$/.test(k))) {
    b = b.shadowsMidtonesHighlights(v4('filament.cg.shadows', [1, 1, 1, 0]), v4('filament.cg.midtones', [1, 1, 1, 0]),
      v4('filament.cg.highlights', [1, 1, 1, 0]), v4('filament.cg.tonal_ranges', [0, 0.333, 0.55, 1]));
  }
  if (keys.some((k) => /^filament\.cg\.(shadow_gamma|mid_point|highlight_scale)$/.test(k))) {
    b = b.curves(v3('filament.cg.shadow_gamma', [1, 1, 1]), v3('filament.cg.mid_point', [1, 1, 1]), v3('filament.cg.highlight_scale', [1, 1, 1]));
  }
  return b.build(engine);
}

// A view's look from the scene's settings `s` (renderSettings of the model
// that gives the scene's look: App.sceneLook), on top of what the Render
// panel sets (main.js viewQuality: anti-aliasing, AO on or off, SSR): the
// color grading `cg` (buildColorGrading; Filament's default when null),
// fog, bloom and vignette. `xr`: a WebXR eye, which takes the color grading
// only (fog and the screen-space effects come out different in each eye).
export function applySceneLook(F, view, s, cg, xr = false) {
  view.setColorGrading(cg);
  if (xr) return;
  const fog = bool(s, 'filament.fog.enabled', false);
  view.setFogOptions(fog ? {
    enabled: true,
    distance: num(s, 'filament.fog.distance', 0),
    cutOffDistance: num(s, 'filament.fog.cut_off_distance', 1e6),
    maximumOpacity: num(s, 'filament.fog.maximum_opacity', 1),
    height: num(s, 'filament.fog.height', 0),
    heightFalloff: num(s, 'filament.fog.height_falloff', 1),
    color: vec(s, 'filament.fog.color', 3, [1, 1, 1]),
    density: num(s, 'filament.fog.density', 0.1),
    inScatteringStart: num(s, 'filament.fog.in_scattering_start', 0),
    inScatteringSize: num(s, 'filament.fog.in_scattering_size', -1),
  } : { enabled: false });
  const Q = F.View$QualityLevel, quality = (k, d) => ({ low: Q.LOW, medium: Q.MEDIUM, high: Q.HIGH, ultra: Q.ULTRA }[str(s, k, '')] ?? d);
  view.setBloomOptions(bool(s, 'filament.bloom.enabled', false) ? {
    enabled: true,
    strength: num(s, 'filament.bloom.strength', 0.1),
    levels: Math.round(num(s, 'filament.bloom.levels', 6)),
    resolution: Math.round(num(s, 'filament.bloom.resolution', 384)),
    quality: quality('filament.bloom.quality', Q.LOW),
  } : { enabled: false });
  view.setVignetteOptions(bool(s, 'filament.vignette.enabled', false)
    ? { enabled: true, midPoint: num(s, 'filament.vignette.midpoint', 0.5) } : { enabled: false });
}

// The model's ambient occlusion options over the Render panel's (`ao`,
// main.js viewQuality): its finer settings, and off if it turns AO off.
export function sceneAo(F, s, ao) {
  if (!s.size) return ao;
  const Q = F.View$QualityLevel, quality = (k, d) => ({ low: Q.LOW, medium: Q.MEDIUM, high: Q.HIGH, ultra: Q.ULTRA }[str(s, k, '')] ?? d);
  return {
    ...ao,
    enabled: ao.enabled && bool(s, 'filament.ao.enabled', true),
    bilateralThreshold: num(s, 'filament.ao.bilateral_threshold', ao.bilateralThreshold),
    upsampling: quality('filament.ao.upsampling', ao.upsampling),
  };
}

// MuJoCo's separate occlusion / roughness / metallic layers packed into one
// ORM image (r, g, b), the size of the largest; a missing one is white (1).
// `imgs`: [occlusion, roughness, metallic], each {w, h, rgba} or null.
export function packOrm(imgs) {
  const w = Math.max(1, ...imgs.map((i) => i?.w ?? 1)), h = Math.max(1, ...imgs.map((i) => i?.h ?? 1));
  const out = new Uint8Array(4 * w * h).fill(255);
  imgs.forEach((img, ch) => {
    if (!img) return;
    for (let y = 0; y < h; y++) {
      const sy = Math.min(img.h - 1, Math.floor(y * img.h / h));
      for (let x = 0; x < w; x++) {
        const sx = Math.min(img.w - 1, Math.floor(x * img.w / w));
        out[4 * (y * w + x) + ch] = img.rgba[4 * (sy * img.w + sx)];  // the layer's red channel, as mjrf's pbr.mat reads it
      }
    }
  });
  return { w, h, rgba: out };
}
