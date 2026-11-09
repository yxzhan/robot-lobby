// Avatars: you in the physics, the VR menu's Avatar mode (render/xr.js). Each
// avatar slot is a head, a torso, legs and two hands, a ball per joint of each
// hand (WebXR Hand Input's 25; a controller: its wrist only), all mocap
// bodies: they move as set every session frame, nothing pushes them back,
// gravity does not act on them. The torso and legs are shaped after MuJoCo's
// humanoid (its welcome scene) and come apart: the torso hangs below the head
// (NECK), turned the way it looks; the legs stand on the floor under it, so a
// crouch sinks the torso into them and a tall user stands above them. They push, tip over and poke whatever they touch (robots too), but
// cannot hold anything (no grip forces).
//
// The world has its slots after every robot (world.js compileWorld; the
// server's server/avatar.py makes the same bodies), so the robots' body and
// mocap indices stay as they were: the page's own world LOCAL_SLOTS, a shared
// world the server's (its world tables' avatar_slots), one per client in
// Avatar mode. A slot nobody uses is parked out of the way (park(slot)).
// Their geoms are in GROUP, which rays skip (main.js castRay), and collide
// with every ordinary geom (contype 0, conaffinity 1) but not with each
// other. render/avatars.js draws them.

export const SIDES = ['left', 'right'];
export const GROUP = 5;
export const LOCAL_SLOTS = 1;
export const HEAD_RADIUS = 0.09;  // m (the humanoid's)
export const HEAD_BACK = 0.08;   // m: the head's center behind the eyes
export const NECK = 0.19;        // m: the torso's origin below the head's center

// The humanoid's capsules {from, to, r} (m), in the torso's frame (x forward,
// y left, z up; its origin NECK below the head) and the legs' (on the floor
// under it).
const mirror = (c) => ({ from: [c.from[0], -c.from[1], c.from[2]], to: [c.to[0], -c.to[1], c.to[2]], r: c.r });
export const TORSO = [
  { from: [0, -0.07, 0], to: [0, 0.07, 0], r: 0.07 },              // chest
  { from: [-0.01, -0.06, -0.12], to: [-0.01, 0.06, -0.12], r: 0.06 },  // waist, upper
  { from: [-0.01, -0.06, -0.26], to: [-0.01, 0.06, -0.26], r: 0.06 },  // waist, lower
];
const RIGHT_LEG = [
  { from: [0, -0.1, 0.817], to: [0, -0.09, 0.477], r: 0.06 },      // thigh
  { from: [0, -0.09, 0.417], to: [0, -0.09, 0.117], r: 0.049 },    // shin
  { from: [-0.07, -0.1, 0.027], to: [0.14, -0.12, 0.027], r: 0.027 },  // foot
  { from: [-0.07, -0.08, 0.027], to: [0.14, -0.06, 0.027], r: 0.027 },
];
export const LEGS = [
  { from: [-0.02, -0.07, 0.857], to: [-0.02, 0.07, 0.857], r: 0.09 },  // pelvis
  ...RIGHT_LEG, ...RIGHT_LEG.map(mirror),
];

const FINGERS = [['thumb', ['metacarpal', 'phalanx-proximal', 'phalanx-distal', 'tip']],
  ...['index', 'middle', 'ring', 'pinky'].map((f) => [`${f}-finger`, ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip']])];
export const HAND_JOINTS = ['wrist', ...FINGERS.flatMap(([f, js]) => js.map((j) => `${f}-${j}`))];
// The bones between them (render/avatars.js): each finger a chain out of the wrist.
export const BONES = FINGERS.flatMap(([f, js]) => js.map((j, i) => [i ? `${f}-${js[i - 1]}` : 'wrist', `${f}-${j}`]));

// A slot's parts, in body order: the head, the torso, the legs, then each hand's joints.
export const PARTS = ['head', 'torso', 'legs', ...SIDES.flatMap((s) => HAND_JOINTS.map((j) => `${s}_${j}`))];
export const CAPSULES = { torso: TORSO, legs: LEGS };
export const SLOT_BODIES = PARTS.length;

export const bodyName = (slot, part) => `avatar${slot}_${part}`;

// m: a hand joint's ball (a little under the finger's half width; the wrist
// and metacarpals larger, to fill the palm).
export function radius(part) {
  if (part === 'head') return HEAD_RADIUS;
  if (part.endsWith('wrist')) return 0.022;
  if (part.endsWith('metacarpal')) return 0.014;
  if (part.endsWith('tip')) return 0.008;
  return 0.01;
}

// Where a slot's unused bodies wait: 1 km up, the slots apart.
export const park = (slot) => [2 * slot, 0, 1000];
export const parked = (pos) => pos[2] > 500;

export function avatarXml(slots) {
  const bodies = [];
  for (let s = 0; s < slots; s++) {
    const at = park(s).join(' ');
    for (const p of PARTS) {
      const geom = (g) => `<geom ${g} contype="0" conaffinity="1" group="${GROUP}" mass="0.01"/>`;
      const geoms = CAPSULES[p] ? CAPSULES[p].map((c) => geom(`type="capsule" fromto="${[...c.from, ...c.to].join(' ')}" size="${c.r}"`)).join('')
        : geom(`type="sphere" size="${radius(p)}"`);
      bodies.push(`<body name="${bodyName(s, p)}" mocap="true" pos="${at}">${geoms}</body>`);
    }
  }
  return `<mujoco model="avatars"><worldbody>${bodies.join('')}</worldbody></mujoco>`;
}
