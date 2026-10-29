// WebXR controllers (Quest Touch: the xr-standard gamepad mapping), in VR
// and AR alike. Each controller points a ray, which ends on the IK marker,
// the menu or the floor it meets.
//   grip (squeeze)  on an IK marker of the driven robot, reached by hand
//                   (within NEAR of it) or by the ray: the target follows the
//                   controller while held, moved as the hand moves and turned
//                   as it turns, about its own center; let go and grab again
//                   to go on (a clutch). Both hands can hold one chain each.
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
//                   for ATTACH_MS: the marker sticks to the palm, following
//                   it as a held grip does, and the robot's fingers follow
//                   yours, each finger of its gripper (by actuator name:
//                   thumb, index, middle, ring, pinky / little) as far as
//                   yours is curled (a gripper with no named fingers: as far
//                   as you pinch). Losing track of the hand keeps it stuck
//                   (taken up again from where the hand comes back).
//   palm up, open   the hand stuck to a marker held still (within STILL)
//                   palm up with every finger straight for LET_GO_MS: lets
//                   go of it. While any hand is stuck, this is the only
//                   gesture (a free hand can still stick to another marker).
//   left pinch, palm towards you, held: the menu (Y).
// No pulling bodies (A) and no sticks otherwise: walking and turning stay on
// the controllers.
// You (the WebXR 'local-floor' space) stand in the scene through `rig`.

import { BALL } from '../app/marker.js';
import { XrMenu } from './xrmenu.js';

const NEAR = 0.08;             // m: hand-to-marker distance that grabs it without the ray
const ATTACH = 0.05;           // m: a tracked palm this near a marker's center ...
const ATTACH_MS = 3000;        // ... this long sticks the marker to it
const LET_GO_MS = 3000;        // a stuck palm up and open this long lets go
const STILL = 0.03;            // m: ... moving less than this meanwhile
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
    h.dwell = null;
    h.letGo = null;
    this.cancelTeleport(h);
    if (h.grab?.kind === 'ik' && h.grab.hand) h.grab.lost = true;  // still stuck (gestures: letGo)
    else this.release(h);
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
    const S = this.sceneFromXr(), seen = new Set();
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
      if (!src.hand) { this.showSkeleton(h, null); h.palm = h.curls = null; }
      else if (h.palm) h.grip = h.palm;  // a tracked hand holds by the middle of its palm
      const pressed = pad.buttons.map((b) => b.pressed), was = h.buttons;
      const down = (i) => pressed[i] && !was[i], up = (i) => !pressed[i] && was[i];
      h.buttons = pressed;
      h.trigger = pad.buttons[TRIGGER]?.value ?? (pressed[TRIGGER] ? 1 : 0);
      h.menuHit = this.menu.hit(h.pick);
      const a = src.hand ? null : PULL[h.name], y = src.hand ? MENU.hand[h.name] : MENU[h.name];
      if (src.hand) {
        // Tracked hands: while one is stuck to a marker, letting go (palm up,
        // open, still) is the only gesture; a free hand may stick to another.
        if (h.grab?.kind === 'ik') this.letGo(h);
        else this.dwell(h);
        if (!stuck) this.handTeleport(h, viewer);
        else this.cancelTeleport(h);
        if (h.name === 'left') this.app.teleop.input.stick.fill(0);  // no stick on a hand
      }
      if (!(src.hand && stuck)) {
        if (!src.hand && down(SQUEEZE) && !h.menuHit && !h.grab) this.grab(h);
        if (!src.hand && up(SQUEEZE) && h.grab?.kind === 'ik') this.release(h);
        if (a != null && down(a) && !h.menuHit && !h.grab) this.pull(h);
        if (a != null && up(a) && (h.grab?.kind === 'body' || h.grab?.kind === 'pending')) this.release(h);
        if (y != null) this.menuButton(h, down(y), pressed[y], up(y), viewer);
        if (down(TRIGGER)) {
          if (h.menuHit) this.menu.select(h.menuHit.row);
          else this.trigger(h);
        }
        if (up(TRIGGER) && h.goal) this.dropGoal(h);
      }
      this.follow(h);
      if (!src.hand) this.sticks(h, pad.axes, viewer, frame);
      this.draw(h);
    }
    for (const h of this.hands.values()) if (!seen.has(h)) this.lost(h);  // a controller put down or lost
    this.hover();
    const onMenu = [...this.hands.values()].find((h) => seen.has(h) && h.menuHit?.row >= 0);
    this.menu.hover(onMenu ? onMenu.menuHit.row : -1);
    this.menu.tick();  // live values (FPS)
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
    if (!wrist || !thumb || !index) {
      h.pinched = h.fisted = false;
      h.palm = h.curls = null;
      if (h.grab?.kind === 'ik') h.grab.lost = true;  // still stuck: taken up again where the hand comes back
      h.letGo = null;
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
      else this.menu.show(this.rootMenu(), pose(mul4(this.sceneFromXr(), viewer.transform.matrix)));
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
        { label: 'Reset TCP', disabled: () => !this.ik(),
          action: () => { this.ik()?.targets.clear(); return true; } },  // every IK target back on its tip
        { label: 'Quality', menu: () => this.qualityMenu() },
        { label: 'FPS', info: true, value: () => (app.xr?.fps ? app.xr.fps.toFixed(0) : '…') },
        { label: 'Close', action: () => true },
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

  // Moves what the hand holds after it.
  follow(h) {
    const g = h.grab, { app } = this;
    if (h.goal) {
      if (app.teleop.dragging?.goal) app.teleop.dragTo(h.pick, null);
      else h.goal = false;  // dropped meanwhile (Reset)
    }
    if (g?.kind === 'ik') {
      if (g.ik !== this.ik() || !app.data) { this.release(h); return; }
      if (g.hand && !h.palm) return;  // the stuck hand out of sight: the marker stays
      if (g.lost) {  // ... and back: taken up again from where the hand is now
        const now = g.ik.targetPose(g.chain, app.data);
        if (!now) return;
        Object.assign(g, { lost: false, c0: h.grip.pos, R0: h.grip.rot, p0: now.pos, T0: now.rot });
      }
      const turn = mulMBt(h.grip.rot, g.R0);  // grip rotation since the grab
      app.teleop.held.set(g.index, BALL);
      g.ik.setTargetPose(g.chain, app.data, { pos: add(g.p0, sub(h.grip.pos, g.c0)), rot: mulMM(turn, g.T0) });
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

  // A stuck hand's fingers on the gripper of the chain it holds: each
  // actuator as far as its finger is curled (one on no named finger: as the
  // hand is, on average); a gripper with no named fingers as far as you pinch.
  fingers(h, g) {
    const { app } = this, grip = g.chain.gripper ?? [];
    if (!grip.length || !h.curls) return;
    g.fingerOf ??= grip.map(({ a }) => fingerOf(g.ik.nameOf('mjOBJ_ACTUATOR', a)));
    const curls = Object.values(h.curls), all = curls.reduce((s, v) => s + v, 0) / (curls.length || 1);
    const named = g.fingerOf.some(Boolean);
    const amounts = grip.map((_, i) => (!named ? h.pinch ?? 0 : h.curls[g.fingerOf[i]] ?? all));
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
    if (g?.kind === 'ik') this.app.teleop.held.delete(g.index);
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

  // A hand stuck to a marker, held still palm up and open for LET_GO_MS:
  // lets go of it.
  letGo(h) {
    if (!h.palm || !(h.palmUp && h.opened)) { h.letGo = null; return; }
    const now = performance.now();
    if (!h.letGo || Math.hypot(...sub(h.palm.pos, h.letGo.pos)) > STILL) h.letGo = { since: now, pos: h.palm.pos };
    else if (now - h.letGo.since >= LET_GO_MS) {
      h.letGo = null;
      this.release(h);
    }
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
