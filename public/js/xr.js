// WebXR controllers (Quest Touch: the xr-standard gamepad mapping), in VR
// and AR alike. Each controller points a ray, which ends on the IK marker or
// the floor it meets.
//   grip (squeeze)  on an IK marker of the driven robot, reached by hand
//                   (within NEAR of it) or by the ray: the target follows the
//                   controller while held, moved as the hand moves and turned
//                   as it turns, about its own center; let go and grab again
//                   to go on (a clutch). Both hands can hold one chain each.
//                   On any other body: pulls it with the selection spring
//                   (Perturb, like Shift+drag), held at the end of the ray as
//                   on a stick.
//   trigger         on a held marker, or one under the ray: opens or closes
//                   that chain's gripper (like a right-click); on a robot:
//                   selects it and drives it (like a double-click), on
//                   anything else: releases it.
//   left stick      walks the driven robot: forward/back, turn.
//   right stick     sideways: turns you by SNAP about where you stand;
//                   forward: aims at the floor (the spawn marker shows
//                   where), let go: you are there.
// You (the WebXR 'local-floor' space) stand in the scene through `rig`.

import { BALL } from './marker.js';

const NEAR = 0.08;             // m: hand-to-marker distance that grabs it without the ray
const SNAP = Math.PI / 6;      // rad per snap turn
const DEAD = 0.15;             // stick dead zone
const RAY_MAX = 4;             // m: a ray that meets nothing
const RAY_RADIUS = 0.002;      // m
const HAND_RADIUS = 0.012;     // m
const HOT = [1, 0.85, 0.2];
const HANDS = { left: [0.35, 0.65, 1], right: [1, 0.55, 0.2], none: [0.8, 0.8, 0.8] };
// xr-standard buttons and axes
const TRIGGER = 0, SQUEEZE = 1, STICK_X = 2, STICK_Y = 3;
// The 'local-floor' space (Y up) in the scene (Z up), before `rig`: x -> x,
// y -> z, z -> -y, its origin 2.5 m back on -Y (the viewer looks along +Y at
// the scene origin). Column-major.
const XR_TO_SCENE = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, -2.5, 0, 1];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
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
  // (App.xrPart).
  constructor(app, part) {
    this.app = app;
    this.part = part;
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
    for (const h of this.hands.values()) {
      this.release(h);
      this.put(h.ball, null);
      this.put(h.ray, null);
    }
    this.hands.clear();
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

  put(part, m) {
    const tcm = this.app.engine.getTransformManager(), inst = tcm.getInstance(part.e);
    tcm.setTransform(inst, m ?? HIDDEN);
    inst.delete();
  }

  // Once per session frame, before it is drawn. `viewer`: the viewer pose.
  update(frame, space, viewer) {
    const S = this.sceneFromXr(), seen = new Set();
    for (const src of frame.session.inputSources) {
      if (!src.gamepad || !src.targetRaySpace) continue;  // tracked hands: not yet
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
      const pad = src.gamepad, pressed = pad.buttons.map((b) => b.pressed), was = h.buttons;
      const down = (i) => pressed[i] && !was[i], up = (i) => !pressed[i] && was[i];
      h.buttons = pressed;
      if (down(SQUEEZE)) this.grab(h);
      if (up(SQUEEZE)) this.release(h);
      if (down(TRIGGER)) this.trigger(h);
      this.follow(h);
      this.sticks(h, pad.axes, viewer);
      this.draw(h);
    }
    for (const h of this.hands.values()) {  // a controller put down or lost
      if (seen.has(h)) continue;
      this.release(h);
      this.put(h.ball, null);
      this.put(h.ray, null);
    }
    this.hover();
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

  grab(h) {
    this.release(h);
    const { app } = this, ik = this.ik(), index = this.reach(h);
    const chain = index != null && ik?.chains[index];
    const target = chain && ![...this.hands.values()].some((o) => o.grab?.chain === chain) && ik.targetPose(chain, app.data);
    if (target) {
      ik.focus = index;
      app.teleop.held.set(index, BALL);
      h.grab = { kind: 'ik', ik, chain, index, c0: h.grip.pos, R0: h.grip.rot, p0: target.pos, T0: target.rot };
      return;
    }
    // Anything else under the ray: pulled with the selection spring (one at a time).
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

  // Moves what the hand holds after it.
  follow(h) {
    const g = h.grab, { app } = this;
    if (g?.kind === 'ik') {
      if (g.ik !== this.ik() || !app.data) { this.release(h); return; }
      const turn = mulMBt(h.grip.rot, g.R0);  // grip rotation since the grab
      app.teleop.held.set(g.index, BALL);
      g.ik.setTargetPose(g.chain, app.data, { pos: add(g.p0, sub(h.grip.pos, g.c0)), rot: mulMM(turn, g.T0) });
    } else if (g?.kind === 'body') {
      if (!app.perturb.target) { h.grab = null; return; }
      const r = h.rayPose, turn = mulMBt(r.rot, g.R0);
      app.perturb.target = add(r.pos, mulMV(turn, sub(g.p0, g.c0)));
      app.perturb.sync();
    }
  }

  release(h) {
    const g = h.grab;
    h.grab = null;
    if (g?.kind === 'ik') this.app.teleop.held.delete(g.index);
    if (g?.kind === 'body') this.app.perturb.end();
  }

  trigger(h) {
    const { app } = this, ik = this.ik();
    const index = h.grab?.kind === 'ik' ? h.grab.index : ik && app.teleop.pick(h.pick)?.index;
    if (index != null && ik?.chains[index]) {
      ik.focus = index;
      ik.toggleGripper(ik.chains[index], app.data);
      return;
    }
    Promise.resolve(app.castRay(h.pick)).then((hit) => {
      if (!app.model) return;
      app.perturb.select(hit);
      app.teleop.drive((hit && app.robotOfBody(hit.body)) ?? null);
    });
  }

  sticks(h, axes, viewer) {
    const x = Math.abs(axes[STICK_X] ?? 0) > DEAD ? axes[STICK_X] : 0;
    const y = Math.abs(axes[STICK_Y] ?? 0) > DEAD ? axes[STICK_Y] : 0;
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
    if (y < -0.6) {
      h.aim = floorHit(h.pick);
      this.app.setSpawnMarker(h.aim);
    } else if (y > -0.3 && h.aim !== undefined) {
      if (h.aim) {
        this.rig.x += h.aim[0] - head[0];
        this.rig.y += h.aim[1] - head[1];
      }
      h.aim = undefined;
      this.app.setSpawnMarker(null);
    }
  }

  // The marker under any ray is highlighted (TeleopPanel.hover).
  hover() {
    const t = this.app.teleop;
    if (t.dragging || !this.ik()) return;
    t.hover = null;
    for (const h of this.hands.values()) {
      if (h.grab || !h.pick) continue;
      const hit = t.pick(h.pick);
      if (hit) { t.hover = hit; break; }
    }
  }

  draw(h) {
    const { pos, rot } = h.grip, k = HAND_RADIUS;
    this.put(h.ball, [rot[0] * k, rot[3] * k, rot[6] * k, 0, rot[1] * k, rot[4] * k, rot[7] * k, 0,
      rot[2] * k, rot[5] * k, rot[8] * k, 0, ...pos, 1]);
    // The ray ends where it meets a marker or the floor.
    const { eye, dir } = h.pick, t = this.app.teleop;
    const hit = !h.grab && this.ik() && t.pick(h.pick);
    const floor = floorHit(h.pick);
    let len = hit ? hit.depth : floor ? Math.min(RAY_MAX, Math.hypot(...sub(floor, eye))) : RAY_MAX;
    if (h.grab?.kind === 'body') len = Math.hypot(...sub(this.app.perturb.target ?? eye, eye)) || len;
    this.put(h.ray, rodMatrix(eye, eye.map((v, i) => v + dir[i] * len), RAY_RADIUS));
    const color = hit || h.grab ? HOT : [0.92, 0.92, 0.92];
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
