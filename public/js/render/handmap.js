// A tracked hand's angles (WebXR Hand Input joints), and a robot hand's
// actuators driven by them: VR hand tracking (xr.js fingers()) with a palm
// stuck to a chain's marker.
//
// A chain's gripper entry (json/ik_defaults.json; ik.js gripperOf) may name
// the angle that drives it and that angle's values for its open and closed
// ctrl (linear between, clamped outside):
//
//   { "actuator": "hand_l_index_joint1", "open": 0, "closed": 1.48,
//     "source": "index.mcp", "range": [0, 90] }
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
// An entry without a source: by its actuator's name, as before (xr.js).
// The VR menu's Hand angles shows them live, for setting the ranges.

export const FINGERS = ['index', 'middle', 'ring', 'pinky'];
export const DEFAULT_RANGE = { mcp: [0, 90], pip: [0, 90], dip: [0, 80], spread: [-15, 15], cmc: [30, 90], abduct: [0, 50],
  ip: [0, 80], curl: [0, 1], pinch: [0, 1] };

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
// curls, pinch), or null: no source, or the hand has no such value.
export function amountFor(entry, values) {
  const src = entry.source;
  if (!src || values?.[src] == null) return null;
  const kind = src.includes('.') ? src.slice(src.indexOf('.') + 1) : src;
  const [a, b] = entry.range ?? DEFAULT_RANGE[kind] ?? [0, 1];
  if (a === b) return null;
  return Math.max(0, Math.min(1, (values[src] - a) / (b - a)));
}
