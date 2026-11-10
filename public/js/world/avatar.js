// Avatars: you in the physics, the VR menu's Avatar mode (render/xr.js). Each
// avatar slot is a head and two hands, all mocap bodies: they move as set
// every session frame, nothing pushes them back, gravity does not act on
// them. A hand is a palm (a triangular prism: the wrist, the index and the
// little finger's knuckles, PALM) and a cylinder per finger bone (HAND_BONES,
// between the two joints it joins, an average hand's lengths); a controller
// is its palm only. The hands push, tip over and poke whatever they touch
// (robots too), but cannot hold anything (no grip forces); the head touches
// nothing (it only shows where its owner is). No avatar geom touches another
// (contype 0): no self collision.
//
// The world has its slots after every robot (world.js compileWorld; the
// server's server/avatar.py makes the same bodies), so the robots' body and
// mocap indices stay as they were: the page's own world LOCAL_SLOTS, a shared
// world the server's (its world tables' avatar_slots), one per client in
// Avatar mode. A slot nobody uses is parked out of the way (park(slot)).
// Their geoms are in GROUP, which rays skip (main.js castRay). xr.js
// avatarPose gives the parts' poses (handPose), render/avatars.js draws them.

export const SIDES = ['left', 'right'];
export const GROUP = 5;
export const LOCAL_SLOTS = 1;
export const HEAD_RADIUS = 0.1;  // m
export const HEAD_BACK = 0.08;   // m: the head's center behind the eyes

// The palm's frame: its origin the wrist, x toward the middle knuckle, y
// toward the thumb side (square to x), z = x × y. The prism: this triangle
// (m), PALM_HALF thick either side.
export const PALM = [[0, 0], [0.085, 0.03], [0.075, -0.035]];
export const PALM_HALF = 0.012;

// A finger bone: [part, from joint, to joint, radius, length] (m; an average
// adult hand's), a cylinder centered between the joints (WebXR Hand Input's).
const f = (finger, r, lengths) => [
  [`${finger}-proximal`, `${finger}-finger-phalanx-proximal`, `${finger}-finger-phalanx-intermediate`, r, lengths[0]],
  [`${finger}-intermediate`, `${finger}-finger-phalanx-intermediate`, `${finger}-finger-phalanx-distal`, r * 0.9, lengths[1]],
  [`${finger}-distal`, `${finger}-finger-phalanx-distal`, `${finger}-finger-tip`, r * 0.8, lengths[2]],
];
export const HAND_BONES = [
  ['thumb-metacarpal', 'thumb-metacarpal', 'thumb-phalanx-proximal', 0.011, 0.045],
  ['thumb-proximal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 0.01, 0.032],
  ['thumb-distal', 'thumb-phalanx-distal', 'thumb-tip', 0.009, 0.025],
  ...f('index', 0.0095, [0.045, 0.026, 0.022]),
  ...f('middle', 0.0095, [0.048, 0.03, 0.024]),
  ...f('ring', 0.009, [0.045, 0.028, 0.023]),
  ...f('pinky', 0.008, [0.035, 0.02, 0.02]),
];
export const HAND_PARTS = ['palm', ...HAND_BONES.map((b) => b[0])];

// A slot's parts, in body order: the head, then each hand's palm and bones.
export const PARTS = ['head', ...SIDES.flatMap((s) => HAND_PARTS.map((p) => `${s}_${p}`))];
export const SLOT_BODIES = PARTS.length;

// Every avatar name starts with PREFIX: the avatars' spec is attached under it
// (MuJoCo 4 warns on an empty prefix), its own names without it.
export const PREFIX = 'avatar';
export const bodyName = (slot, part) => `${PREFIX}${slot}_${part}`;

// Where a slot's unused bodies wait: 1 km up, the slots apart.
export const park = (slot) => [2 * slot, 0, 1000];
export const parked = (pos) => pos[2] > 500;

// The palm prism's six corners (x, y, z), PALM's triangle at -PALM_HALF and +PALM_HALF.
export const palmVertices = () => [-PALM_HALF, PALM_HALF].flatMap((z) => PALM.map(([x, y]) => [x, y, z]));

export function avatarXml(slots) {
  const geom = (g, touch = 1) => `<geom ${g} contype="0" conaffinity="${touch}" group="${GROUP}" mass="0.01"/>`;
  const bodies = [];
  for (let s = 0; s < slots; s++) {
    const at = park(s).join(' ');
    for (const p of PARTS) {
      const part = p.slice(p.indexOf('_') + 1), bone = HAND_BONES.find((b) => b[0] === part);
      const g = p === 'head' ? geom(`type="sphere" size="${HEAD_RADIUS}"`, 0)  // the head touches nothing
        : part === 'palm' ? geom('type="mesh" mesh="_palm"')
          : geom(`type="cylinder" size="${bone[3]} ${bone[4] / 2}"`);
      bodies.push(`<body name="${bodyName(s, p).slice(PREFIX.length)}" mocap="true" pos="${at}">${g}</body>`);
    }
  }
  return `<mujoco model="avatars"><asset><mesh name="_palm" vertex="${palmVertices().flat().join(' ')}"/></asset>`
    + `<worldbody>${bodies.join('')}</worldbody></mujoco>`;
}

// ---- poses ----

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((v) => v / l); };

// A rotation with these columns as a quaternion [w, x, y, z].
export function columnsQuat(x, y, z) {
  const r = [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]];  // row-major
  const t = r[0] + r[4] + r[8];
  if (t > 0) { const s = 2 * Math.sqrt(t + 1); return [s / 4, (r[7] - r[5]) / s, (r[2] - r[6]) / s, (r[3] - r[1]) / s]; }
  if (r[0] > r[4] && r[0] > r[8]) { const s = 2 * Math.sqrt(1 + r[0] - r[4] - r[8]); return [(r[7] - r[5]) / s, s / 4, (r[1] + r[3]) / s, (r[2] + r[6]) / s]; }
  if (r[4] > r[8]) { const s = 2 * Math.sqrt(1 + r[4] - r[0] - r[8]); return [(r[2] - r[6]) / s, (r[1] + r[3]) / s, s / 4, (r[5] + r[7]) / s]; }
  const s = 2 * Math.sqrt(1 + r[8] - r[0] - r[4]);
  return [(r[3] - r[1]) / s, (r[2] + r[6]) / s, (r[5] + r[7]) / s, s / 4];
}

// A tracked hand's parts' poses from its joints (joint -> {pos}, scene frame):
// part -> {pos, quat}; parts whose joints are not tracked are left out.
export function handPose(joints) {
  const at = (n) => joints.get(n)?.pos;
  const out = new Map();
  const w = at('wrist'), mK = at('middle-finger-phalanx-proximal'), iK = at('index-finger-phalanx-proximal'), pK = at('pinky-finger-phalanx-proximal');
  if (w && mK && iK && pK) {
    const x = norm(sub(mK, w)), y0 = sub(iK, pK), y = norm(sub(y0, x.map((v) => v * dot(y0, x))));
    out.set('palm', { pos: w, quat: columnsQuat(x, y, cross(x, y)) });
  }
  for (const [part, from, to] of HAND_BONES) {
    const a = at(from), b = at(to);
    if (!a || !b) continue;
    const z = norm(sub(b, a)), h = Math.abs(z[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    const x = norm(cross(h, z));
    out.set(part, { pos: a.map((v, i) => (v + b[i]) / 2), quat: columnsQuat(x, cross(z, x), z) });
  }
  return out;
}
