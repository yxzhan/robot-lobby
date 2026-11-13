// WebXR controllers (Quest Touch: the xr-standard gamepad mapping), in VR
// and AR alike. Each controller points a ray, which ends on the IK marker,
// the menu or the floor it meets.
//   grip (squeeze)  on an IK marker of the driven robot, reached by hand
//                   (within NEAR of it) or by the ray: the target follows the
//                   controller while held, moved as the hand moves and turned
//                   as it turns, about its own center; let go and grab again
//                   to go on (a clutch). Both hands can hold one chain each.
//                   Out of reach the arm stops where it gets to, the
//                   controller rumbling; let go, the marker goes back onto
//                   the tip (controllers/ik/leash.js).
//   A (right, held) on any body under the ray: pulls it with the selection
//                   spring (Perturb, like Shift+drag on the desktop), held at
//                   the end of the ray as on a stick. (Not on the grip: just
//                   holding the controllers squeezes it.)
//   Y (left, held)  opens or closes the menu (xrmenu.js; a short press also
//                   closes it): the trigger chooses the row under the ray.
//   trigger         while the same hand holds a marker: closes that chain's
//                   gripper as far as it is pressed (let go: open), once
//                   pressed at all during the hold; on a marker under the
//                   ray: opens or closes its gripper (like G); held on the
//                   goal marker: drags it over the floor like the mouse, the
//                   same hand's stick sideways turning it; on a robot:
//                   selects it and drives it (like a double-click), on
//                   anything else: releases it.
//   left stick      walks the driven robot: forward/back, turn (taking over
//                   from its goal, which it drops).
//   right stick     sideways: turns you by SNAP about where you stand;
//                   forward: aims at the floor (the spawn marker shows
//                   where), let go: you are there.
// Tracked hands (Quest hand tracking, WebXR Hand Input) stand in for the
// controllers (a hand holding a controller is that controller: its gestures
// are not read), their gestures read off the joints as the buttons above:
//   pinch           (thumb and index tips together) the trigger, as far as
//                   they are close.
//   right point     (index straight, middle, ring and little fingers
//                   curled) aims a teleport at the floor (the spawn marker
//                   shows where); the hand opened all the way: you are
//                   there; curled into a fist: not.
//   palm on a marker the palm's middle held on an IK marker (within ATTACH)
//                   for ATTACH_MS (a countdown by the hand meanwhile:
//                   countdown()): the marker sticks to the palm, following
//                   it as a held grip does, and the robot's fingers follow
//                   yours, each finger of its gripper (by actuator name:
//                   thumb, index, middle, ring, pinky / little) as far as
//                   yours is curled (a gripper with no named fingers: as far
//                   as you pinch). Losing track of the hand keeps it stuck
//                   (taken up again if the hand comes back within REGRAB of
//                   where it went and within REGRAB_MS, else let go). The
//                   hands let go when the menu opens or the session loses
//                   focus (the Meta button: the system's menu). While a hand
//                   is stuck, the menu is the only other gesture (a free
//                   hand can still stick to another marker).
//   left pinch, palm towards you, held: the menu (Y); it lets go of the
//                   markers the hands are stuck to.
// No pulling bodies (A) and no sticks otherwise: walking and turning stay on
// the controllers.
// You (the WebXR 'local-floor' space) stand in the scene through `rig`.

import { BALL } from '../app/marker.js';
import { XrMenu } from './xrmenu.js';
import { HEAD_BACK, handPose } from '../world/avatar.js';
import { FINGERS as HAND_FINGERS, HAND_DEFAULTS, amountFor, average, handAngles, loadCalibration, saveCalibration } from './handmap.js';
import { serverUrl } from '../app/endpoint.js';
import { snapToTip, strain } from '../controllers/ik/leash.js';

const RUMBLE_MS = 80;           // a rumble pulse while a held marker is at its leash's end
const NEAR = 0.08;             // m: hand-to-marker distance that grabs it without the ray
const ATTACH = 0.05;           // m: a tracked palm this near a marker's center ...
const ATTACH_MS = 3000;        // ... this long sticks the marker to it
const REGRAB = 0.08;            // m: a stuck hand back in sight this near where it went out of it goes on holding ...
const REGRAB_MS = 2000;        // ... and within this long; else the marker is let go (back onto the tip)
const CAL_STEP_MS = 4000;      // a calibration step (hands open, fists) ...
const CAL_SAMPLE_MS = 1500;    // ... its last this long averaged
const COUNTDOWN = { px: 128, size: 0.06, toward: 0.12, above: 0.05 };  // the attach countdown: canvas px, m wide; m toward the head and up from the palm (in front of the robot's hand it is on)
const STRAIGHT = 0.25;         // a finger's curl (0 straight … 1 a fist) under this: straight
const CURLED_IN = 0.55;        // ... over this: curled in
const FINGER_SEND_MS = 50;     // network mode: a stuck hand's fingers go to the server this often, at most
const SNAP = Math.PI / 6;      // rad per snap turn
const GOAL_TURN_RATE = Math.PI / 2;  // rad/s: a dragged goal turned with the stick pushed all the way
const DEAD = 0.15;             // stick dead zone
const RAY_MAX = 4;             // m: a ray that meets nothing
const RAY_RADIUS = 0.002;      // m
const HAND_RADIUS = 0.012;     // m
const HOT = [1, 0.85, 0.2];
const HANDS = { left: [0.35, 0.65, 1], right: [1, 0.55, 0.2], none: [0.8, 0.8, 0.8] };
// xr-standard buttons and axes (4 and 5: A/B on the right controller, X/Y on the left)
const TRIGGER = 0, SQUEEZE = 1, STICK_X = 2, STICK_Y = 3;
const PULL = { right: 4 };                   // A (none for a tracked hand)
const MENU = { left: 5, hand: { left: 5 } };  // Y (a hand's: the left palm-up pinch)
const MENU_HOLD = 400;         // ms the menu button is held to open the menu
// Hand gestures (m, and ratios of a fingertip's distance from the wrist to
// its knuckle's), with hysteresis: on below the first, off above the second.
const PINCH = [0.018, 0.026];  // thumb tip to index tip
const PINCH_SPAN = [0.012, 0.03];  // ... the trigger's value from 1 to 0
const FIST = [1.3, 1.5];       // middle, ring and little fingers, on average
const PALM_TO_HEAD = 0.6;      // cos: the left palm faces you
const PALM_UP = 0.6;           // cos: the palm faces up
const JOINT_RADIUS = 0.006;    // m: a joint's ball, at most (the runtime's radius: the finger's)
const BONE_RADIUS = 0.003;     // m
// A tracked hand's skeleton (WebXR Hand Input): its 25 joints, each finger a
// chain out of the wrist, and the bones joining them.
const FINGERS = [['thumb', ['metacarpal', 'phalanx-proximal', 'phalanx-distal', 'tip']],
  ...['index', 'middle', 'ring', 'pinky'].map((f) => [`${f}-finger`, ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip']])];
const JOINTS = ['wrist', ...FINGERS.flatMap(([f, js]) => js.map((j) => `${f}-${j}`))];
const BONES = FINGERS.flatMap(([f, js]) => js.map((j, i) => [i ? `${f}-${js[i - 1]}` : 'wrist', `${f}-${j}`]));
const CURLED = ['middle-finger', 'ring-finger', 'pinky-finger'];
// A finger's curl: the angles between its bones added up, over this (a fist).
const CURL_FULL = { thumb: 135 * Math.PI / 180, finger: 260 * Math.PI / 180 };
// A gripper actuator's finger, by its name.
const FINGER_NAMES = [[/thumb/i, 'thumb'], [/index/i, 'index'], [/middle/i, 'middle'], [/ring/i, 'ring'], [/pinky|little/i, 'pinky']];
const fingerOf = (name) => FINGER_NAMES.find(([re]) => re.test(name))?.[1] ?? null;
// The 'local-floor' space (Y up) in the scene (Z up), before `rig`: x -> x,
// y -> z, z -> -y, its origin 2.5 m back on -Y (the viewer looks along +Y at
// the scene origin). Column-major.
const XR_TO_SCENE = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, -2.5, 0, 1];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const angle = (a, b) => Math.acos(Math.min(1, Math.max(-1, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / ((Math.hypot(...a) * Math.hypot(...b)) || 1))));
const mulMV = (R, v) => [0, 1, 2].map((r) => R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2]);
const mulMM = (A, B) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]));
const mulMBt = (A, B) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[3 * r] * B[3 * c] + A[3 * r + 1] * B[3 * c + 1] + A[3 * r + 2] * B[3 * c + 2]));  // A Bᵀ
// column-major 4x4 a * b
function mul4(a, b) {
  const r = new Array(16);
  for (let c = 0; c < 4; c++) {
    for (let k = 0; k < 4; k++) r[4 * c + k] = a[k] * b[4 * c] + a[4 + k] * b[4 * c + 1] + a[8 + k] * b[4 * c + 2] + a[12 + k] * b[4 * c + 3];
  }
  return r;
}
// A column-major 4x4 as {pos, rot (row-major 3x3)}.
const pose = (m) => ({ pos: [m[12], m[13], m[14]], rot: [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]] });
// A rotation matrix (row-major 3x3) as a quaternion [w, x, y, z].
function rotQuat(r) {
  const t = r[0] + r[4] + r[8];
  if (t > 0) { const s = 2 * Math.sqrt(t + 1); return [s / 4, (r[7] - r[5]) / s, (r[2] - r[6]) / s, (r[3] - r[1]) / s]; }
  if (r[0] > r[4] && r[0] > r[8]) { const s = 2 * Math.sqrt(1 + r[0] - r[4] - r[8]); return [(r[7] - r[5]) / s, s / 4, (r[1] + r[3]) / s, (r[2] + r[6]) / s]; }
  if (r[4] > r[8]) { const s = 2 * Math.sqrt(1 + r[4] - r[0] - r[8]); return [(r[2] - r[6]) / s, (r[1] + r[3]) / s, s / 4, (r[5] + r[7]) / s]; }
  const s = 2 * Math.sqrt(1 + r[8] - r[0] - r[4]);
  return [(r[3] - r[1]) / s, (r[2] + r[6]) / s, (r[5] + r[7]) / s, s / 4];
}
// The transform putting a unit rod (radius 1, z from -0.5 to 0.5) from p to q, radius w.
function rodMatrix(p, q, w) {
  const d = sub(q, p), len = Math.hypot(...d);
  if (len < 1e-6) return null;
  const z = d.map((v) => v / len), a = Math.abs(z[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let x = [a[1] * z[2] - a[2] * z[1], a[2] * z[0] - a[0] * z[2], a[0] * z[1] - a[1] * z[0]];
  const xl = Math.hypot(...x);
  x = x.map((v) => v / xl);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return [...x.map((v) => v * w), 0, ...y.map((v) => v * w), 0, ...d, 0, ...p.map((v, i) => (v + q[i]) / 2), 1];
}
const HIDDEN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

export class XrControls {
  // `part(rod)`: a new hidden marker entity, {e, mi}: a unit ball, or a rod
  // (App.xrPart); `panel(...)`: the menu's panel (App.xrPanel).
  constructor(app, part, panel) {
    this.app = app;
    this.part = part;
    this.menu = new XrMenu(panel);
    this.makePanel = panel;  // (the attach countdowns: countdown())
    this.calibration = loadCalibration();  // side -> {open, fist}: your hands' values (handmap.js)
    this.cal = null;  // a calibration running (calibrate())
    this.calSaved = '';  // Save as defaults: how it went
    this.hands = new Map();  // handedness -> hand state
    this.rig = { x: 0, y: 0, yaw: 0 };
  }

  // Scene from 'local-floor' (column-major): rig * XR_TO_SCENE.
  sceneFromXr() {
    const { x, y, yaw } = this.rig, c = Math.cos(yaw), s = Math.sin(yaw);
    return mul4([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, x, y, 0, 1], XR_TO_SCENE);
  }

  start() {
    this.rig = { x: 0, y: 0, yaw: 0 };
    this.app.teleop.input.xr = true;
  }

  end() {
    this.app.setAvatar(null);
    for (const h of this.hands.values()) this.lost(h);
    this.hands.clear();
    this.menu.close();
    const t = this.app.teleop;
    t.input.xr = false;
    t.input.stick.fill(0);
    t.held.clear();
    this.app.setSpawnMarker(null);
  }

  hand(name) {
    let h = this.hands.get(name);
    if (!h) {
      h = { name, ball: this.part(false), ray: this.part(true), buttons: [], grab: null, rayColor: null };
      h.ball.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [...(HANDS[name] ?? HANDS.none), 1]);
      this.hands.set(name, h);
    }
    return h;
  }

  // A hand put down or lost: lets go and hides it.
  lost(h) {
    h.joints = null;  // (its avatar hand parked)
    h.dwell = null;
    this.countdown(h, null);
    this.cancelTeleport(h);
    if (h.grab?.kind === 'ik' && h.grab.hand) {  // still stuck, for a while (follow)
      this.lostSight(h.grab);
      if (performance.now() - h.grab.lost.at > REGRAB_MS) this.release(h);
    } else this.release(h);
    if (h.goal) this.dropGoal(h);
    this.put(h.ball, null);
    this.put(h.ray, null);
    this.showSkeleton(h, null);
  }

  put(part, m) {
    const tcm = this.app.engine.getTransformManager(), inst = tcm.getInstance(part.e);
    tcm.setTransform(inst, m ?? HIDDEN);
    inst.delete();
  }

  // Once per session frame, before it is drawn. `viewer`: the viewer pose.
  update(frame, space, viewer) {
    const S = this.sceneFromXr(), seen = new Set(), avatar = this.avatarMode();
    const calibrating = this.cal && this.cal.phase !== 'done';  // (a pinch in a fist must not click the menu)
    const stuck = [...this.hands.values()].some((h) => h.grab?.kind === 'ik' && h.grab.hand);  // a tracked hand on a marker
    // A controller and a tracked hand on the same side (Quest's concurrent
    // hands and controllers): the controller is that hand, its gestures are
    // not read.
    const sources = [...frame.session.inputSources];
    const held = new Set(sources.filter((src) => src.gamepad && !src.hand).map((src) => src.handedness));
    for (const src of sources) {
      if (!src.targetRaySpace || !(src.gamepad || src.hand)) continue;
      if (src.hand && held.has(src.handedness)) continue;
      const rayPose = frame.getPose(src.targetRaySpace, space);
      if (!rayPose) continue;
      const h = this.hand(src.handedness);
      seen.add(h);
      const r = pose(mul4(S, rayPose.transform.matrix));
      const dir = [-r.rot[2], -r.rot[5], -r.rot[8]];  // -z
      h.rayPose = r;
      h.pick = { eye: r.pos, dir, forward: dir };
      const gripPose = src.gripSpace && frame.getPose(src.gripSpace, space);
      h.grip = gripPose ? pose(mul4(S, gripPose.transform.matrix)) : r;
      const pad = src.hand ? this.gestures(h, src.hand, frame, space, viewer) : src.gamepad;
      h.haptic = src.hand ? null : src.gamepad?.hapticActuators?.[0] ?? null;  // (rumble)
      if (!src.hand) { this.showSkeleton(h, null); h.palm = h.curls = null; h.joints = null; }
      else if (h.palm) h.grip = h.palm;  // a tracked hand holds by the middle of its palm
      const pressed = pad.buttons.map((b) => b.pressed), was = h.buttons;
      const down = (i) => pressed[i] && !was[i], up = (i) => !pressed[i] && was[i];
      h.buttons = pressed;
      h.trigger = pad.buttons[TRIGGER]?.value ?? (pressed[TRIGGER] ? 1 : 0);
      h.menuHit = this.menu.hit(h.pick);
      const a = src.hand ? null : PULL[h.name], y = src.hand ? MENU.hand[h.name] : MENU[h.name];
      if (src.hand) {
        // Tracked hands: while one is stuck to a marker, only the menu (which
        // lets go); a free hand may stick to another. Avatar mode: the hand
        // is a hand (it pushes, in the physics), no sticking to markers.
        if (avatar || h.grab) h.dwell = null;
        else this.dwell(h);
        if (!stuck) this.handTeleport(h, viewer);
        else this.cancelTeleport(h);
        if (h.name === 'left') this.app.teleop.input.stick.fill(0);  // no stick on a hand
      }
      if (y != null && !calibrating) this.menuButton(h, down(y), pressed[y], up(y), viewer);  // (also while stuck: it lets go)
      if (!(src.hand && stuck)) {
        // Avatar mode: the menu, teleporting and turning only (no grabbing
        // markers or bodies, no trigger on them).
        if (!avatar && !src.hand && down(SQUEEZE) && !h.menuHit && !h.grab) this.grab(h);
        if (!src.hand && up(SQUEEZE) && h.grab?.kind === 'ik') this.release(h);
        if (!avatar && a != null && down(a) && !h.menuHit && !h.grab) this.pull(h);
        if (a != null && up(a) && (h.grab?.kind === 'body' || h.grab?.kind === 'pending')) this.release(h);
        if (down(TRIGGER) && !calibrating) {
          if (h.menuHit) this.menu.select(h.menuHit.row);
          else if (!avatar) this.trigger(h);
        }
        if (up(TRIGGER) && h.goal) this.dropGoal(h);
      }
      this.follow(h);
      this.countdown(h, viewer);
      if (avatar && h.name === 'left') this.app.teleop.input.stick.fill(0);  // (it walks the robot: teleop)
      else if (!src.hand) this.sticks(h, pad.axes, viewer, frame);
      this.draw(h);
    }
    for (const h of this.hands.values()) if (!seen.has(h)) this.lost(h);  // a controller put down or lost
    this.app.setAvatar(avatar ? this.avatarPose(S, viewer, seen) : null);
    if (calibrating) this.calibrate(seen);
    this.hover();
    const onMenu = [...this.hands.values()].find((h) => seen.has(h) && h.menuHit?.row >= 0);
    this.menu.hover(onMenu ? onMenu.menuHit.row : -1);
    this.menu.tick();  // live values (FPS)
  }

  // ---- modes (the menu's Mode) ----

  avatarMode() { return this.app.xrSettings.mode === 'avatar'; }

  // Teleop: the gestures and buttons drive the robots, no collision. Avatar:
  // a solid head and hands in the physics (world/avatar.js), seen by everyone
  // in a shared world; the hands only push, the menu and teleporting stay.
  setMode(mode) {
    this.app.setXrSettings({ mode });
    if (mode !== 'avatar') return;
    for (const h of this.hands.values()) {  // what a hand held in Teleop mode is let go
      h.dwell = null;
      if (h.grab) this.release(h);
      if (h.goal) this.dropGoal(h);
    }
    this.app.teleop.input.stick.fill(0);
  }

  // The avatar's pose (main.js setAvatar): the head behind the eyes (HEAD_BACK
  // along the view's +z), a tracked hand's palm and finger bones from its
  // joints (world/avatar.js handPose), a controller's grip as its hand's palm.
  avatarPose(S, viewer, seen) {
    const v = pose(mul4(S, viewer.transform.matrix)), z = [v.rot[2], v.rot[5], v.rot[8]];
    const head = { pos: v.pos.map((x, i) => x + HEAD_BACK * z[i]), quat: rotQuat(v.rot) };
    const hand = (name) => {
      const h = this.hands.get(name);
      if (!h || !seen.has(h)) return null;
      if (h.joints) return handPose(h.joints);
      return h.grip ? new Map([['palm', { pos: h.grip.pos, quat: rotQuat(h.grip.rot) }]]) : null;
    };
    return { head, left: hand('left'), right: hand('right') };
  }

  // A tracked hand's gestures as an xr-standard gamepad {buttons, axes}
  // (none while its joints are not tracked); its skeleton drawn.
  gestures(h, hand, frame, space, viewer) {
    const S = this.sceneFromXr(), joints = new Map();
    for (const name of JOINTS) {
      const p = hand.get(name) && frame.getJointPose(hand.get(name), space);
      if (p) joints.set(name, { ...pose(mul4(S, p.transform.matrix)), radius: p.radius });
    }
    const joint = (name) => joints.get(name);
    const wrist = joint('wrist'), thumb = joint('thumb-tip'), index = joint('index-finger-tip');
    const button = (pressed, value = pressed ? 1 : 0) => ({ pressed, value });
    h.joints = joints.size ? joints : null;  // (its avatar hand, Avatar mode)
    if (!wrist || !thumb || !index) {
      h.pinched = h.fisted = false;
      h.palm = h.curls = null;
      if (h.grab?.kind === 'ik') this.lostSight(h.grab);  // still stuck, for a while (follow)
      this.cancelTeleport(h);
      this.showSkeleton(h, null);
      return { buttons: [], axes: [] };
    }
    // The middle of the palm (between the wrist and the middle finger's
    // knuckle, in the wrist's frame), and how far each finger is curled.
    const knuckle = joint('middle-finger-phalanx-proximal');
    h.palm = { pos: knuckle ? wrist.pos.map((v, i) => (v + knuckle.pos[i]) / 2) : wrist.pos, rot: wrist.rot };
    h.curls = {};
    for (const [f, prefix, js] of [['thumb', 'thumb', ['metacarpal', 'phalanx-proximal', 'phalanx-distal', 'tip']],
      ...['index', 'middle', 'ring', 'pinky'].map((f) => [f, `${f}-finger`, ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip']])]) {
      const ps = js.map((j) => joint(`${prefix}-${j}`)?.pos);
      if (ps.some((p) => !p)) continue;
      let sum = 0;
      for (let i = 1; i < ps.length - 1; i++) sum += angle(sub(ps[i], ps[i - 1]), sub(ps[i + 1], ps[i]));
      h.curls[f] = Math.min(1, sum / CURL_FULL[f === 'thumb' ? 'thumb' : 'finger']);
    }
    const d = Math.hypot(...sub(thumb.pos, index.pos));
    h.pinched = h.pinched ? d < PINCH[1] : d < PINCH[0];
    const curl = CURLED.map((f) => {
      const tip = joint(`${f}-tip`), knuckle = joint(`${f}-phalanx-proximal`);
      return tip && knuckle ? Math.hypot(...sub(tip.pos, wrist.pos)) / (Math.hypot(...sub(knuckle.pos, wrist.pos)) || 1) : 2;
    }).reduce((s, v) => s + v, 0) / CURLED.length;
    h.fisted = h.fisted ? curl < FIST[1] : curl < FIST[0];
    // What a pinch is, chosen as it starts by the palm (the wrist's -y): the
    // left one towards the head, the menu; the right one up, a teleport
    // (aimed while held, as the right stick pushed forward); else the trigger.
    const palm = [-wrist.rot[1], -wrist.rot[4], -wrist.rot[7]];
    if (!h.pinched) h.pinchAs = null;
    else if (!h.pinchAs) {
      const head = pose(mul4(S, viewer.transform.matrix)).pos, toHead = sub(head, wrist.pos), l = Math.hypot(...toHead) || 1;
      if (h.name === 'left' && (palm[0] * toHead[0] + palm[1] * toHead[1] + palm[2] * toHead[2]) / l > PALM_TO_HEAD) h.pinchAs = 'menu';
      else h.pinchAs = 'trigger';
    }
    const pinch = Math.min(1, Math.max(0, (PINCH_SPAN[1] - d) / (PINCH_SPAN[1] - PINCH_SPAN[0])));
    h.pinch = pinch;
    // The hand's angles, curls and pinch: what a gripper entry's source names (handmap.js).
    h.values = { ...handAngles(joints, h.name), pinch };
    for (const [f, v] of Object.entries(h.curls)) h.values[`${f}.curl`] = v;
    const as = h.pinchAs, plain = !as || as === 'trigger';
    const buttons = [button(as === 'trigger', plain ? pinch : 0), button(h.fisted), button(false), button(false),
      button(h.fisted), button(as === 'menu')];
    this.showSkeleton(h, joints);
    // Hand shapes: pointing (index straight, the next three curled in),
    // open (all five straight), a full fist; the palm up.
    const c = h.curls, has = ['thumb', 'index', 'middle', 'ring', 'pinky'].every((f) => c[f] != null);
    const three = has && ['middle', 'ring', 'pinky'].every((f) => c[f] > CURLED_IN);
    h.pointing = three && c.index < STRAIGHT;
    h.fistAll = three && c.index > CURLED_IN;
    h.opened = has && Object.values(c).every((v) => v < STRAIGHT);
    h.palmUp = palm[2] > PALM_UP;
    return { buttons, axes: [0, 0, 0, 0] };
  }

  // A hand's skeleton from its joint poses (a Map by name), or hidden (null):
  // a ball on each joint, a rod along each bone.
  showSkeleton(h, joints) {
    if (!joints && !h.skeleton) return;
    h.skeleton ??= (() => {
      const color = [...(HANDS[h.name] ?? HANDS.none), 1], part = (rod) => {
        const p = this.part(rod);
        p.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, color);
        return p;
      };
      return { balls: JOINTS.map(() => part(false)), rods: BONES.map(() => part(true)) };
    })();
    JOINTS.forEach((name, i) => {
      const j = joints?.get(name), k = Math.min(j?.radius ?? JOINT_RADIUS, JOINT_RADIUS);
      this.put(h.skeleton.balls[i], j && [k, 0, 0, 0, 0, k, 0, 0, 0, 0, k, 0, ...j.pos, 1]);
    });
    BONES.forEach(([a, b], i) => {
      const p = joints?.get(a), q = joints?.get(b);
      this.put(h.skeleton.rods[i], p && q && rodMatrix(p.pos, q.pos, BONE_RADIUS));
    });
  }

  // The menu button: held MENU_HOLD ms opens or closes the menu, a short
  // press closes it.
  menuButton(h, down, held, up, viewer) {
    const now = performance.now();
    if (down) { h.menuAt = now; h.menuDone = false; }
    if (held && h.menuAt != null && !h.menuDone && now - h.menuAt > MENU_HOLD) {
      h.menuDone = true;
      if (this.menu.open) this.menu.close();
      else {
        this.letGoAll();  // the menu lets go of the markers the hands are stuck to
        this.menu.show(this.rootMenu(), pose(mul4(this.sceneFromXr(), viewer.transform.matrix)));
      }
    }
    if (up) {
      if (!h.menuDone && this.menu.open) this.menu.close();
      h.menuAt = null;
    }
  }

  // ---- the menu (xrmenu.js) ----

  rootMenu() {
    const app = this.app;
    return {
      title: 'Menu',
      items: [
        { label: 'Mode', value: () => (this.avatarMode() ? 'Avatar' : 'Teleop'), menu: () => ({
          title: 'Mode', items: [
            { label: 'Teleop: drive the robots', checked: () => !this.avatarMode(), action: () => { this.setMode('teleop'); } },
            { label: 'Avatar: you, solid', checked: () => this.avatarMode(), action: () => { this.setMode('avatar'); } },
          ] }) },
        { label: 'Reset scene', disabled: () => !app.model,
          action: () => {  // as the toolbar's Reset: every robot back to its spawn pose (a shared world: for everyone)
            for (const h of this.hands.values()) { h.dwell = null; if (h.grab) this.release(h); if (h.goal) this.dropGoal(h); }
            app.recorder?.live();
            app.reset();
            return true;
          } },
        ...this.realItems(),
        { label: 'Quality', menu: () => this.qualityMenu() },
        { label: 'Calibrate hands', menu: () => this.calibrateMenu() },
        { label: 'Hand angles', menu: () => ({ title: 'Hand angles', items: [
          { label: 'Left hand', menu: () => this.anglesMenu('left') },
          { label: 'Right hand', menu: () => this.anglesMenu('right') },
        ] }) },
        { label: 'FPS', info: true, value: () => (app.xr?.fps ? app.xr.fps.toFixed(0) : '…') },
        { label: 'Close', action: () => true },
      ],
    };
  }

  // The driven robot's real robot (ros/real.js), when its model can have one.
  realItems() {
    const r = this.app.teleop.active, rr = r && this.app.ros.robot(r);
    const real = rr?.real;
    if (!real?.available) return [];
    return [{
      label: 'Real robot', value: () => real.state === 'none' ? 'no bridge' : real.state,
      menu: () => ({
        title: 'Real robot',
        items: () => [
          { label: 'ROS', checked: () => rr.on, action: () => { rr.setOn(!rr.on); } },
          { label: 'State', info: true, value: () => real.text().slice(0, 34) },
          { label: 'Enable', disabled: () => !real.live || real.state === 'enabled', action: () => { real.enable(); } },
          { label: 'Disable', disabled: () => !real.live || real.state === 'disabled', action: () => { real.disable(); } },
          { label: 'Ghost', checked: () => real.ghostOn, action: () => { real.setGhost(!real.ghostOn); } },
        ],
      }),
    }];
  }

  // ---- calibrating the hands (handmap.js) ----

  // Both hands open wide for CAL_STEP_MS, then fists as long; the last
  // CAL_SAMPLE_MS of each averaged, per tracked hand: where each of its angles
  // is open and as a fist, what a robot hand's open and closed ctrl then go
  // with (kept in this browser).
  calibrateMenu() {
    const c = () => this.cal, left = () => Math.max(0, Math.ceil((c().until - performance.now()) / 1000));
    const step = { open: 'Open hands wide', fist: 'Make fists' };
    return {
      title: 'Calibrate hands',
      items: [
        { label: 'Steps: hands wide open, then fists', info: true },
        { label: '', info: true, value: () => {
          if (!c()) return 'press Start';
          if (c().phase === 'done') return c().done.length ? `done: ${c().done.join(', ')}` : 'no hand tracked: again';
          return `${step[c().phase]}  ${left()} s`;
        } },
        { label: 'Start', action: () => { this.startCalibration(); } },
        { label: 'In use', info: true, value: () => ['left', 'right'].map((s) => `${s}: ${this.calibration[s] ? 'yours' : HAND_DEFAULTS[s] ? 'default' : 'typical'}`).join(', ') },
        { label: 'Save as defaults', action: () => { this.saveDefaults(); } },
        { label: '', info: true, value: () => this.calSaved },
        { label: 'Back to the defaults', action: () => { this.calibration = {}; saveCalibration({}); this.cal = null; } },
      ],
    };
  }

  // Your calibrated hands written to json/hand_calibration.json on the server
  // (the other hand's default kept), the defaults for every browser from the
  // next page load on.
  async saveDefaults() {
    const sides = ['left', 'right'].filter((s) => this.calibration[s]);
    if (!sides.length) { this.calSaved = 'nothing calibrated here'; return; }
    this.calSaved = 'saving...';
    try {
      const res = await fetch(serverUrl('api/hand_calibration'), { method: 'POST', body: JSON.stringify(this.calibration) });
      if (!res.ok) throw new Error(await res.text() || res.statusText);
      for (const s of sides) HAND_DEFAULTS[s] = this.calibration[s];
      this.calSaved = `saved: ${sides.join(', ')}`;
    } catch (err) {
      this.calSaved = `not saved: ${err.message}`;
    }
  }

  // A hand's calibration: yours, else the default (handmap.js).
  calibrationOf(side) { return this.calibration[side] ?? HAND_DEFAULTS[side]; }

  startCalibration() {
    this.letGoAll();
    this.cal = { phase: 'open', until: performance.now() + CAL_STEP_MS, samples: { left: [], right: [] }, open: {}, fist: {}, done: [] };
  }

  // Every session frame while calibrating: the tracked hands' values sampled,
  // the next step when this one is over, the result kept after the fists.
  calibrate(seen) {
    const c = this.cal, now = performance.now();
    if (now > c.until - CAL_SAMPLE_MS) {
      for (const side of ['left', 'right']) {
        const h = this.hands.get(side);
        if (h && seen.has(h) && h.joints && h.values) c.samples[side].push({ ...h.values });
      }
    }
    if (now < c.until) return;
    for (const side of ['left', 'right']) {
      if (c.samples[side].length >= 10) c[c.phase][side] = average(c.samples[side]);
      c.samples[side] = [];
    }
    if (c.phase === 'open') {
      c.phase = 'fist';
      c.until = now + CAL_STEP_MS;
      return;
    }
    c.phase = 'done';
    for (const side of ['left', 'right']) {
      if (c.open[side] && c.fist[side]) {
        this.calibration[side] = { open: c.open[side], fist: c.fist[side] };
        c.done.push(side);
      }
    }
    saveCalibration(this.calibration);
  }

  // A tracked hand's angles live (handmap.js), degrees: for setting a robot
  // hand's gripper sources and ranges (json/ik_defaults.json).
  anglesMenu(side) {
    const v = () => this.hands.get(side)?.values ?? {};
    const show = (keys) => () => (keys.every((k) => v()[k] == null) ? 'not tracked' : keys.map((k) => (v()[k] == null ? '–' : Math.round(v()[k]))).join('  '));
    return {
      title: `${side === 'left' ? 'Left' : 'Right'} hand, degrees`,
      items: [
        { label: 'finger', info: true, value: () => 'mcp  pip  dip  spread' },
        ...HAND_FINGERS.map((f) => ({ label: f, info: true, value: show(['mcp', 'pip', 'dip', 'spread'].map((k) => `${f}.${k}`)) })),
        { label: 'thumb', info: true, value: () => 'cmc  abduct  mcp  ip' },
        { label: '', info: true, value: show(['thumb.cmc', 'thumb.abduct', 'thumb.mcp', 'thumb.ip']) },
        { label: 'pinch', info: true, value: () => (v().pinch == null ? '–' : v().pinch.toFixed(2)) },
      ],
    };
  }

  qualityMenu() {
    const app = this.app;
    const choose = (title, get, set, options) => ({
      title, items: options.map(([v, label]) => ({ label, checked: () => get() === v, action: () => { set(v); } })),
    });
    const msaa = { 1: 'off', 2: '2× MSAA', 4: '4× MSAA' };
    return {
      title: 'Quality',
      items: [
        { label: 'Resolution', value: () => `${app.xrSettings.scale}×`,
          menu: () => choose('Resolution', () => app.xrSettings.scale, (v) => app.setXrSettings({ scale: v }),
            [[0.6, '0.6×'], [0.8, '0.8×'], [1, '1× (headset default)'], [1.2, '1.2×'], [1.5, '1.5× (or native)']]) },
        { label: 'Anti-alias', value: () => msaa[app.xrSettings.msaa] ?? '',
          menu: () => choose('Anti-alias', () => app.xrSettings.msaa, (v) => app.setXrSettings({ msaa: v }),
            Object.entries(msaa).map(([v, l]) => [Number(v), l])) },
        { label: 'Shadows', value: () => (app.xrSettings.shadows ? 'on' : 'off'),
          menu: () => choose('Shadows', () => app.xrSettings.shadows, (v) => app.setXrSettings({ shadows: v }),
            [[false, 'off'], [true, 'on (the sun\'s)']]) },
        { label: 'Brightness', value: () => `${Math.round(100 * (app.xrSettings.brightness ?? 1))} %`,
          menu: () => choose('Brightness', () => app.xrSettings.brightness ?? 1, (v) => app.setXrSettings({ brightness: v }),
            [[1, '100 %'], [0.8, '80 %'], [0.65, '65 %'], [0.5, '50 %'], [0.35, '35 %'], [0.25, '25 %']]) },
        { label: 'Textures', value: () => `${app.quality.textures} px`,
          menu: () => choose('Textures', () => app.quality.textures, (v) => app.renderPanel.set({ ...app.quality, textures: v }),
            [[64, '64 px'], [512, '512 px'], [1024, '1024 px'], [2048, '2048 px']]) },
      ],
    };
  }

  // The driven robot's IK controller while it has markers, else null.
  ik() {
    const t = this.app.teleop, ik = t.active?.use?.ik && t.active.ik;
    return ik?.enabled && this.app.data ? ik : null;
  }

  // The marker index this hand reaches: by hand, else with its ray.
  reach(h) {
    if (!this.ik()) return null;
    const gizmos = this.app.teleop.gizmos();
    let best = null, bestD = NEAR;
    gizmos.forEach((m, i) => {
      const d = m && Math.hypot(...sub(m.pos, h.grip.pos));
      if (m && d < bestD) { best = i; bestD = d; }
    });
    return best ?? this.app.teleop.pick(h.pick)?.index ?? null;
  }

  // A tracked palm on a marker (the nearest within ATTACH, held by no other
  // hand) for ATTACH_MS: stuck to it.
  dwell(h) {
    const ik = this.ik(), now = performance.now();
    let index = null, best = ATTACH;
    if (ik && h.palm && !h.grab) {
      this.app.teleop.gizmos().forEach((m, i) => {
        const d = m && Math.hypot(...sub(m.pos, h.palm.pos));
        if (d < best && ![...this.hands.values()].some((o) => o.grab?.index === i)) { index = i; best = d; }
      });
    }
    if (index == null) h.dwell = null;
    else if (h.dwell?.index !== index) h.dwell = { index, since: now };
    else if (now - h.dwell.since >= ATTACH_MS) {
      h.dwell = null;
      this.grab(h, index);
    }
  }

  // The IK marker the hand reaches (or marker `index`), held.
  grab(h, index = this.reach(h)) {
    const { app } = this, ik = this.ik();
    const chain = index != null && ik?.chains[index];
    const target = chain && ![...this.hands.values()].some((o) => o.grab?.chain === chain) && ik.targetPose(chain, app.data);
    if (target) {
      ik.focus = index;
      app.teleop.held.set(index, BALL);
      h.grab = { kind: 'ik', ik, chain, index, c0: h.grip.pos, R0: h.grip.rot, p0: target.pos, T0: target.rot, hand: !!h.palm };
    }
  }

  // The body under the ray, pulled with the selection spring (one at a time).
  pull(h) {
    const { app } = this;
    const cast = app.castRay(h.pick), token = h.grab = { kind: 'pending' };
    Promise.resolve(cast).then((hit) => {
      if (h.grab !== token) return;  // let go meanwhile
      h.grab = null;
      if (!hit || hit.body <= 0 || !app.model) return;
      for (const o of this.hands.values()) if (o.grab?.kind === 'body') this.release(o);
      app.perturb.select(hit);
      if (!app.perturb.begin(h.pick)) return;
      const { rayPose: r } = h;
      h.grab = { kind: 'body', c0: r.pos, R0: r.rot, p0: hit.point };
    });
  }

  // Lets go of the goal marker: the goal is set where it is.
  dropGoal(h) {
    h.goal = false;
    if (this.app.teleop.dragging?.goal) this.app.teleop.release();
  }

  // The controller rumbles while the arm is behind the marker it holds
  // (out of reach: leash.js), the harder the further.
  rumble(h, k) {
    const now = performance.now();
    if (!h.haptic || k <= 0 || now - (h.rumbleAt ?? 0) < RUMBLE_MS) return;
    h.rumbleAt = now;
    try { h.haptic.pulse(0.15 + 0.5 * k, RUMBLE_MS); } catch { /* no haptics */ }
  }

  // Moves what the hand holds after it.
  follow(h) {
    const g = h.grab, { app } = this;
    if (h.goal) {
      if (app.teleop.dragging?.goal) app.teleop.dragTo(h.pick, null);
      else h.goal = false;  // dropped meanwhile (Reset)
    }
    if (g?.kind === 'ik') {
      if (g.ik !== this.ik() || !app.data) { this.release(h); return; }
      if (g.hand && !h.palm) {  // the stuck hand out of sight: the marker stays, but not for long
        if (g.lost && performance.now() - g.lost.at > REGRAB_MS) this.release(h);
        return;
      }
      if (g.lost) {  // ... and back: near where it went, taken up again from there; elsewhere (or late), let go
        const away = g.lost.pos ? Math.hypot(...sub(h.grip.pos, g.lost.pos)) : 0;
        if (away > REGRAB || performance.now() - g.lost.at > REGRAB_MS) { this.release(h); return; }
        const now = g.ik.targetPose(g.chain, app.data);
        if (!now) return;
        Object.assign(g, { lost: null, c0: h.grip.pos, R0: h.grip.rot, p0: now.pos, T0: now.rot });
      }
      g.seen = h.grip.pos;  // (where it was last: lostSight)
      const turn = mulMBt(h.grip.rot, g.R0);  // grip rotation since the grab
      app.teleop.held.set(g.index, BALL);
      const want = { pos: add(g.p0, sub(h.grip.pos, g.c0)), rot: mulMM(turn, g.T0) };
      g.ik.setTargetPose(g.chain, app.data, want);
      this.rumble(h, strain(g.ik.tipPose?.(g.chain, app.data), want));  // the arm behind (out of reach)
      if (g.hand) {
        this.fingers(h, g);
        return;
      }
      // The trigger closes the gripper part way, once it has been touched.
      if (h.trigger > 0.05) g.squeezed = true;
      if (g.squeezed && g.chain.gripper?.length && Math.abs(h.trigger - (g.amount ?? -1)) > 0.02) {
        g.amount = h.trigger;
        g.ik.setGripper(g.chain, app.data, h.trigger);
      }
    } else if (g?.kind === 'body') {
      if (!app.perturb.target) { h.grab = null; return; }
      const r = h.rayPose, turn = mulMBt(r.rot, g.R0);
      app.perturb.target = add(r.pos, mulMV(turn, sub(g.p0, g.c0)));
      app.perturb.sync();
    }
  }

  // A stuck hand's fingers on the gripper of the chain it holds: an actuator
  // whose entry names a source as that angle of the hand says (handmap.js:
  // per robot hand, json/ik_defaults.json), the others as far as their
  // finger is curled (by the actuator's name; one on no named finger: as the
  // hand is, on average); a gripper with no named fingers as far as you pinch.
  fingers(h, g) {
    const { app } = this, grip = g.chain.gripper ?? [];
    if (!grip.length || !h.curls) return;
    g.fingerOf ??= grip.map(({ a }) => fingerOf(g.ik.nameOf('mjOBJ_ACTUATOR', a)));
    // Each finger's curl as far between your calibrated open hand and fist (handmap.js).
    const cal = this.calibrationOf(h.name), curl = (f) => amountFor({ source: `${f}.curl` }, h.values, cal) ?? h.curls[f];
    const curls = Object.keys(h.curls).map(curl), all = curls.reduce((s, v) => s + v, 0) / (curls.length || 1);
    const named = g.fingerOf.some(Boolean);
    const amounts = grip.map((e, i) => amountFor(e, h.values, cal) ?? (!named ? h.pinch ?? 0 : (g.fingerOf[i] && h.curls[g.fingerOf[i]] != null ? curl(g.fingerOf[i]) : all)));
    const now = performance.now();
    if (g.amounts && !amounts.some((v, i) => Math.abs(v - g.amounts[i]) > 0.02)) return;
    if (app.net && now - (g.sentAt ?? -Infinity) < FINGER_SEND_MS) return;
    g.amounts = amounts;
    g.sentAt = now;
    g.ik.setGripper(g.chain, app.data, amounts);
  }

  release(h) {
    const g = h.grab;
    h.grab = null;
    if (g?.kind === 'ik') {
      this.app.teleop.held.delete(g.index);
      if (g.ik === this.ik()) snapToTip(g.ik, g.chain, this.app.data);  // the marker back onto the tip
    }
    if (g?.kind === 'body') this.app.perturb.end();
  }

  trigger(h) {
    const { app } = this, ik = this.ik();
    if (h.grab?.kind === 'ik') return;  // the gripper follows the trigger (follow)
    const index = ik && app.teleop.pick(h.pick)?.index;
    if (index != null && ik?.chains[index]) {
      ik.focus = index;
      ik.toggleGripper(ik.chains[index], app.data);
      return;
    }
    const t = app.teleop;
    if (!t.dragging && t.pickGoal(h.pick) && t.grab(h.pick, null)) {
      h.goal = true;
      return;
    }
    Promise.resolve(app.castRay(h.pick)).then((hit) => {
      if (!app.model) return;
      app.perturb.select(hit);
      const robot = (hit && app.robotOfBody(hit.body)) ?? null;
      app.teleop.drive(robot);
      app.camerasPanel.show(robot);
    });
  }

  sticks(h, axes, viewer, frame) {
    let x = Math.abs(axes[STICK_X] ?? 0) > DEAD ? axes[STICK_X] : 0;
    const y = Math.abs(axes[STICK_Y] ?? 0) > DEAD ? axes[STICK_Y] : 0;
    // Sideways on the hand dragging the goal: turns it.
    const now = frame.predictedDisplayTime ?? performance.now(), dt = Math.min(0.1, (now - (h.t ?? now)) / 1000);
    h.t = now;
    if (h.goal) {
      if (x) this.app.teleop.turnGoal(-x * GOAL_TURN_RATE * dt);
      x = 0;
    }
    if (h.name === 'left') {
      const stick = this.app.teleop.input.stick;
      stick[0] = -y;
      stick[1] = 0;
      stick[2] = -x;
      return;
    }
    if (h.name !== 'right') return;
    const head = pose(mul4(this.sceneFromXr(), viewer.transform.matrix)).pos;
    // Snap turns, one per push sideways.
    if (Math.abs(x) > 0.7 && !h.turned) {
      h.turned = true;
      const a = -Math.sign(x) * SNAP, c = Math.cos(a), s = Math.sin(a), rig = this.rig;
      const dx = rig.x - head[0], dy = rig.y - head[1];
      rig.x = head[0] + c * dx - s * dy;
      rig.y = head[1] + s * dx + c * dy;
      rig.yaw += a;
    } else if (Math.abs(x) < 0.3) {
      h.turned = false;
    }
    // Teleport: aim while pushed forward, go on letting go.
    if (y < -0.6) this.aimTeleport(h);
    else if (y > -0.3 && h.aim !== undefined) this.teleport(h, head);
  }

  // A teleport aimed with the hand's ray (h.aim: the floor spot, or null).
  aimTeleport(h) {
    h.aim = floorHit(h.pick);
    this.app.setSpawnMarker(h.aim);
  }

  // You go where the hand aimed (your head over it).
  teleport(h, head) {
    if (h.aim) {
      this.rig.x += h.aim[0] - head[0];
      this.rig.y += h.aim[1] - head[1];
    }
    this.cancelTeleport(h);
  }

  cancelTeleport(h) {
    if (h.aim === undefined) return;
    h.aim = undefined;
    this.app.setSpawnMarker(null);
  }

  // The right hand's teleport: aimed while pointing; the hand opened all the
  // way goes there, a full fist calls it off (in between, the aim holds).
  handTeleport(h, viewer) {
    if (h.name !== 'right' || !h.curls) return;
    if (h.pointing && !h.dwell) this.aimTeleport(h);
    else if (h.aim !== undefined && h.opened) this.teleport(h, pose(mul4(this.sceneFromXr(), viewer.transform.matrix)).pos);
    else if (h.fistAll) this.cancelTeleport(h);
  }

  // A stuck hand out of sight: when and where (the last palm seen), so that
  // coming back elsewhere or late lets go (follow: REGRAB, REGRAB_MS).
  lostSight(g) {
    g.lost ??= { at: performance.now(), pos: g.seen ?? null };
  }

  // The tracked hands stuck to markers let go of them (the menu opening,
  // the session losing focus).
  letGoAll() {
    for (const h of this.hands.values()) {
      h.dwell = null;
      if (h.grab?.kind === 'ik' && h.grab.hand) this.release(h);
    }
  }

  // The session's visibility (main.js): the Meta button's system menu (or
  // the headset taken off) blurs or hides it: the hands let go.
  visibility(state) {
    if (state !== 'visible') this.letGoAll();
  }

  // A palm resting on a marker (dwell): the seconds left till it sticks, and
  // a ring running out, on a small panel by the palm facing the head;
  // hidden otherwise (viewer null: hidden).
  countdown(h, viewer) {
    const d = viewer && h.dwell && h.palm ? h.dwell : null;
    if (!d) {
      if (h.countdown?.shown) { h.countdown.panel.put(null); h.countdown.shown = false; h.countdown.drawn = null; }
      return;
    }
    const left = Math.max(0, ATTACH_MS - (performance.now() - d.since)), step = Math.ceil(left / 100);
    let c = h.countdown;
    if (!c) {
      const canvas = Object.assign(document.createElement('canvas'), { width: COUNTDOWN.px, height: COUNTDOWN.px });
      c = h.countdown = { canvas, panel: this.makePanel(COUNTDOWN.px, COUNTDOWN.px, COUNTDOWN.size, COUNTDOWN.size), shown: false, drawn: null };
    }
    if (c.drawn !== step) {  // redrawn ten times a second
      c.drawn = step;
      const g = c.canvas.getContext('2d'), n = COUNTDOWN.px, r = n * 0.42;
      g.clearRect(0, 0, n, n);
      g.fillStyle = 'rgba(16, 20, 28, 0.78)';
      g.beginPath(); g.arc(n / 2, n / 2, r + 6, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = '#ffd34d'; g.lineWidth = 9; g.lineCap = 'round';
      g.beginPath(); g.arc(n / 2, n / 2, r, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * (left / ATTACH_MS)); g.stroke();
      g.fillStyle = '#ffffff'; g.font = `bold ${Math.round(n * 0.46)}px system-ui, sans-serif`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(String(Math.ceil(left / 1000)), n / 2, n / 2 + n * 0.03);
      c.panel.upload(c.canvas);
    }
    // Between the palm and the head, a little up, facing the head, upright.
    const head = pose(mul4(this.sceneFromXr(), viewer.transform.matrix)).pos, p = h.palm.pos;
    const f0 = [head[0] - p[0], head[1] - p[1], 0], fl = Math.hypot(f0[0], f0[1]) || 1;
    const normal = [f0[0] / fl, f0[1] / fl, 0], up = [0, 0, 1], right = [-normal[1], normal[0], 0];
    const pos = [p[0] + normal[0] * COUNTDOWN.toward, p[1] + normal[1] * COUNTDOWN.toward, p[2] + COUNTDOWN.above];
    c.panel.put([...right, 0, ...up, 0, ...normal, 0, ...pos, 1]);
    c.shown = true;
  }

  // The IK or goal marker under any ray is highlighted (TeleopPanel.hover).
  hover() {
    const t = this.app.teleop;
    if (t.dragging) return;
    t.hover = null;
    const dwelling = [...this.hands.values()].find((h) => h.dwell);
    if (dwelling) { t.hover = { index: dwelling.dwell.index, part: BALL }; return; }  // a palm on it, sticking
    for (const h of this.hands.values()) {
      if (h.grab || !h.pick || h.menuHit) continue;
      const hit = (this.ik() && t.pick(h.pick)) || (t.pickGoal(h.pick) && { goal: true });
      if (hit) { t.hover = hit; break; }
    }
  }

  draw(h) {
    const { pos, rot } = h.grip, k = HAND_RADIUS;
    this.put(h.ball, [rot[0] * k, rot[3] * k, rot[6] * k, 0, rot[1] * k, rot[4] * k, rot[7] * k, 0,
      rot[2] * k, rot[5] * k, rot[8] * k, 0, ...pos, 1]);
    // The ray ends where it meets the menu, a marker or the floor.
    const { eye, dir } = h.pick, t = this.app.teleop, menu = h.menuHit;
    const hit = !menu && !h.grab && this.ik() && t.pick(h.pick);
    const floor = floorHit(h.pick), onGoal = h.goal || (!menu && !h.grab && !hit && !!t.pickGoal(h.pick));
    let len = menu ? menu.depth : hit ? hit.depth : floor ? Math.min(RAY_MAX, Math.hypot(...sub(floor, eye))) : RAY_MAX;
    if (h.grab?.kind === 'body') len = Math.hypot(...sub(this.app.perturb.target ?? eye, eye)) || len;
    this.put(h.ray, rodMatrix(eye, eye.map((v, i) => v + dir[i] * len), RAY_RADIUS));
    const color = hit || h.grab || onGoal || menu?.row >= 0 ? HOT : [0.92, 0.92, 0.92];
    if (h.rayColor !== color) {
      h.ray.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [...color, 1]);
      h.rayColor = color;
    }
  }
}

// Where the ray {eye, dir} meets the floor (z = 0), or null.
function floorHit({ eye, dir }) {
  if (eye[2] <= 0 || dir[2] > -1e-6) return null;
  const s = -eye[2] / dir[2];
  return [eye[0] + s * dir[0], eye[1] + s * dir[1], 0];
}
