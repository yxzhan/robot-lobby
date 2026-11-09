// Avatars: you in the physics, the VR menu's Avatar mode (render/xr.js). Each
// avatar slot is a head and two hands, a ball per joint of each hand (WebXR
// Hand Input's 25; a controller: its wrist only), all mocap bodies: they move
// as set every session frame, nothing pushes them back, gravity does not act
// on them. The hands' balls push, tip over and poke whatever they touch
// (robots too), but cannot hold anything (no grip forces); the head touches
// nothing (it only shows where its owner is).
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
export const HEAD_RADIUS = 0.1;  // m
export const HEAD_BACK = 0.08;   // m: the head's center behind the eyes

const FINGERS = [['thumb', ['metacarpal', 'phalanx-proximal', 'phalanx-distal', 'tip']],
  ...['index', 'middle', 'ring', 'pinky'].map((f) => [`${f}-finger`, ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip']])];
export const HAND_JOINTS = ['wrist', ...FINGERS.flatMap(([f, js]) => js.map((j) => `${f}-${j}`))];
// The bones between them (render/avatars.js): each finger a chain out of the wrist.
export const BONES = FINGERS.flatMap(([f, js]) => js.map((j, i) => [i ? `${f}-${js[i - 1]}` : 'wrist', `${f}-${j}`]));

// A slot's parts, in body order: the head, then each hand's joints.
export const PARTS = ['head', ...SIDES.flatMap((s) => HAND_JOINTS.map((j) => `${s}_${j}`))];
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
      const touch = p === 'head' ? 0 : 1;  // the head collides with nothing: only where it is (drawn by render/avatars.js)
      bodies.push(`<body name="${bodyName(s, p)}" mocap="true" pos="${at}"><geom type="sphere" size="${radius(p)}" `
        + `contype="0" conaffinity="${touch}" group="${GROUP}" mass="0.01"/></body>`);
    }
  }
  return `<mujoco model="avatars"><worldbody>${bodies.join('')}</worldbody></mujoco>`;
}
