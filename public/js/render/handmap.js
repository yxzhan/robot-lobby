// A tracked hand's angles (WebXR Hand Input joints), and a robot hand's
// actuators driven by them: VR hand tracking (xr.js fingers()) with a palm
// stuck to a chain's marker.
//
// A chain's gripper entry (json/ik_defaults.json; ik.js gripperOf) gives the
// actuator's ctrl with your hand open and with it a fist, and may name the
// angle of the hand that drives it in between (linear, clamped outside):
//
//   { "actuator": "hand_l_index_joint1", "open": 0, "closed": 1.48, "source": "index.mcp" }
//
// Where that angle is with your hand open and as a fist comes from your
// calibration (the VR menu's Calibrate hands: per hand, kept in this browser,
// CAL_STORE), else from a typical hand (TYPICAL); an entry's "range": [open,
// fist] (degrees) overrides both.
//
// Sources (degrees; curl and pinch 0..1), the same for either hand:
//   <finger>.mcp, .pip, .dip   index, middle, ring, pinky: the bend at the
//                              knuckle, the middle and the last joint (0
//                              straight)
//   <finger>.spread            sideways at the knuckle, in the palm's plane,
//                              + toward the thumb
//   thumb.cmc                  the thumb's metacarpal in the palm's plane:
//                              0 along the thumb side, 90 along the fingers
//                              (across the palm: more)
//   thumb.abduct               ... out of the palm's plane, + to the palm side
//   thumb.mcp, thumb.ip        the thumb's two bends
//   <finger>.curl, thumb.curl  0 straight .. 1 a fist (the whole finger)
//   pinch                      0 apart .. 1 thumb and index tips together
// An entry without a source: by its actuator's name, its finger's curl (xr.js),
// calibrated the same way. The VR menu's Hand angles shows them live.

export const FINGERS = ['index', 'middle', 'ring', 'pinky'];
export const CAL_STORE = 'robot-lobby/hand-calibration';

// A typical hand's values, open and as a fist (degrees; curl and pinch 0..1).
export const TYPICAL = (() => {
  const open = { 'thumb.cmc': 30, 'thumb.abduct': 0, 'thumb.mcp': 0, 'thumb.ip': 0, 'thumb.curl': 0, pinch: 0 };
  const fist = { 'thumb.cmc': 95, 'thumb.abduct': 30, 'thumb.mcp': 50, 'thumb.ip': 60, 'thumb.curl': 1, pinch: 0 };
  const spread = { index: 12, middle: 0, ring: -8, pinky: -15 };  // open: spread out (+ toward the thumb); a fist: together
  for (const f of FINGERS) {
    Object.assign(open, { [`${f}.mcp`]: 0, [`${f}.pip`]: 0, [`${f}.dip`]: 0, [`${f}.spread`]: spread[f], [`${f}.curl`]: 0 });
    Object.assign(fist, { [`${f}.mcp`]: 85, [`${f}.pip`]: 100, [`${f}.dip`]: 70, [`${f}.spread`]: 0, [`${f}.curl`]: 1 });
  }
  return { open, fist };
})();
const MIN_SPAN = { curl: 0.05, pinch: 0.05 };  // open and fist closer than this (else 3 degrees): no telling them apart

export function loadCalibration() {
  try { return JSON.parse(localStorage.getItem(CAL_STORE) ?? 'null') ?? {}; } catch { return {}; }
}
export function saveCalibration(cal) {
  try { localStorage.setItem(CAL_STORE, JSON.stringify(cal)); } catch { /* no storage */ }
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((v) => v / l); };
const DEG = 180 / Math.PI;
const between = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot(norm(a), norm(b))))) * DEG;

// joints: joint name -> {pos} (scene frame), side 'left' | 'right'. The palm's
// frame: X from the wrist to the middle knuckle, Y toward the thumb side (the
// pinky knuckle to the index one, made square to X), N out of the palm.
export function handAngles(joints, side) {
  const at = (n) => joints.get(n)?.pos;
  const w = at('wrist'), iK = at('index-finger-phalanx-proximal'), mK = at('middle-finger-phalanx-proximal'), pK = at('pinky-finger-phalanx-proximal');
  if (!w || !iK || !mK || !pK) return null;
  const X = norm(sub(mK, w)), y0 = sub(iK, pK), Y = norm(sub(y0, X.map((v) => v * dot(y0, X))));
  const c = cross(X, Y), N = side === 'right' ? c.map((v) => -v) : c;
  const out = {};
  const inPalm = (v) => Math.atan2(dot(v, Y), dot(v, X)) * DEG;  // the angle in the palm's plane, + toward the thumb
  for (const f of FINGERS) {
    const [M, P, I, D, T] = ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip'].map((j) => at(`${f}-finger-${j}`));
    if (!M || !P || !I || !D || !T) continue;
    const meta = sub(P, M), prox = sub(I, P), mid = sub(D, I), dist = sub(T, D);
    out[`${f}.mcp`] = between(meta, sub(prox, Y.map((v) => v * dot(prox, Y))));  // (its sideways part is the spread)
    out[`${f}.pip`] = between(prox, mid);
    out[`${f}.dip`] = between(mid, dist);
    out[`${f}.spread`] = inPalm(prox) - inPalm(meta);
  }
  const [tM, tP, tD, tT] = ['metacarpal', 'phalanx-proximal', 'phalanx-distal', 'tip'].map((j) => at(`thumb-${j}`));
  if (tM && tP && tD && tT) {
    const meta = sub(tP, tM);
    out['thumb.cmc'] = Math.atan2(dot(meta, X), dot(meta, Y)) * DEG;
    out['thumb.abduct'] = Math.asin(Math.max(-1, Math.min(1, dot(norm(meta), N)))) * DEG;
    out['thumb.mcp'] = between(meta, sub(tD, tP));
    out['thumb.ip'] = between(sub(tD, tP), sub(tT, tD));
  }
  return out;
}

// An entry's amount, 0 (open) .. 1 (closed), from the hand's values (angles,
// curls, pinch) and its calibration ({open, fist}: values; none: TYPICAL), or
// null: no source, or the hand has no such value. Open and fist too close
// together to tell apart: 0.
export function amountFor(entry, values, cal) {
  const src = entry.source;
  if (!src || values?.[src] == null) return null;
  const kind = src.includes('.') ? src.slice(src.indexOf('.') + 1) : src;
  const pick = (c) => (Number.isFinite(c?.open?.[src]) && Number.isFinite(c?.fist?.[src]) ? [c.open[src], c.fist[src]] : null);
  const [a, b] = entry.range ?? pick(cal) ?? pick(TYPICAL) ?? [0, 1];
  if (Math.abs(b - a) < (MIN_SPAN[kind] ?? 3)) return 0;
  return Math.max(0, Math.min(1, (values[src] - a) / (b - a)));
}

// A calibration step's samples (values per frame) averaged: source -> mean.
export function average(samples) {
  const sum = {}, n = {};
  for (const v of samples) for (const [k, x] of Object.entries(v)) if (Number.isFinite(x)) { sum[k] = (sum[k] ?? 0) + x; n[k] = (n[k] ?? 0) + 1; }
  return Object.fromEntries(Object.keys(sum).map((k) => [k, sum[k] / n[k]]));
}
