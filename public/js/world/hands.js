// Your tracked hands in the physics (VR hand tracking, local physics only): a
// ball per joint of each hand (WebXR Hand Input's 25), each a mocap body
// that xr.js puts where the joint is every session frame. Mocap bodies move
// as set, nothing pushes them back: the hands push, tip over and pick at
// whatever they touch (robots too), but cannot hold (no grip forces).
//
// They are attached to the world after every robot (world.js compileWorld,
// prefix PREFIX), so the robots' body and mocap indices stay as they were;
// parked out of the way (PARK) while a hand is not tracked, holds a
// controller, or hand collision is off (the VR menu). Their geoms are in
// GROUP, which rays (main.js castRay) skip, and collide with every ordinary
// geom (contype 0, conaffinity 1) but not with each other.

export const PREFIX = 'xrhand_';
export const SIDES = ['left', 'right'];
export const GROUP = 5;
export const PARK = [0, 0, 1000];

const FINGERS = [['thumb', ['metacarpal', 'phalanx-proximal', 'phalanx-distal', 'tip']],
  ...['index', 'middle', 'ring', 'pinky'].map((f) => [`${f}-finger`, ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip']])];
export const HAND_JOINTS = ['wrist', ...FINGERS.flatMap(([f, js]) => js.map((j) => `${f}-${j}`))];

// m: a joint's ball (a little under the finger's half width; the wrist and
// metacarpals larger, to fill the palm).
function radius(joint) {
  if (joint === 'wrist') return 0.022;
  if (joint.endsWith('metacarpal')) return 0.014;
  if (joint.endsWith('tip')) return 0.008;
  return 0.01;
}

export const bodyName = (side, joint) => `${side}_${joint}`;  // in hands.xml (PREFIX added on attach)

export function handsXml() {
  const bodies = SIDES.flatMap((side) => HAND_JOINTS.map((j) => `
    <body name="${bodyName(side, j)}" mocap="true" pos="${PARK.join(' ')}">
      <geom type="sphere" size="${radius(j)}" contype="0" conaffinity="1" group="${GROUP}" mass="0.01"/>
    </body>`));
  return `<mujoco model="hands"><worldbody>${bodies.join('')}
  </worldbody></mujoco>`;
}

export const HAND_BODIES = SIDES.length * HAND_JOINTS.length;
