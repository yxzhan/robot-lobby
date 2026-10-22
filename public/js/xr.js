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
//                   ray: opens or closes its gripper (like a right-click); held on the
//                   goal marker: drags it over the floor like the mouse, the
//                   same hand's stick sideways turning it; on a robot:
//                   selects it and drives it (like a double-click), on
//                   anything else: releases it.
//   left stick      walks the driven robot: forward/back, turn (taking over
//                   from its goal, which it drops).
//   right stick     sideways: turns you by SNAP about where you stand;
//                   forward: aims at the floor (the spawn marker shows
//                   where), let go: you are there.
// You (the WebXR 'local-floor' space) stand in the scene through `rig`.

import { BALL } from './marker.js';
import { XrMenu } from './xrmenu.js';

const NEAR = 0.08;             // m: hand-to-marker distance that grabs it without the ray
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
const PULL = { right: 4 };     // A
const MENU = { left: 5 };      // Y
const MENU_HOLD = 400;         // ms the menu button is held to open the menu
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
    for (const h of this.hands.values()) {
      this.release(h);
      if (h.goal) this.dropGoal(h);
      this.put(h.ball, null);
      this.put(h.ray, null);
    }
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
      h.trigger = pad.buttons[TRIGGER]?.value ?? (pressed[TRIGGER] ? 1 : 0);
      h.menuHit = this.menu.hit(h.pick);
      const a = PULL[h.name], y = MENU[h.name];
      if (down(SQUEEZE) && !h.menuHit && !h.grab) this.grab(h);
      if (up(SQUEEZE) && h.grab?.kind === 'ik') this.release(h);
      if (a != null && down(a) && !h.menuHit && !h.grab) this.pull(h);
      if (a != null && up(a) && (h.grab?.kind === 'body' || h.grab?.kind === 'pending')) this.release(h);
      if (y != null) this.menuButton(h, down(y), pressed[y], up(y), viewer);
      if (down(TRIGGER)) {
        if (h.menuHit) this.menu.select(h.menuHit.row);
        else this.trigger(h);
      }
      if (up(TRIGGER) && h.goal) this.dropGoal(h);
      this.follow(h);
      this.sticks(h, pad.axes, viewer, frame);
      this.draw(h);
    }
    for (const h of this.hands.values()) {  // a controller put down or lost
      if (seen.has(h)) continue;
      this.release(h);
      if (h.goal) this.dropGoal(h);
      this.put(h.ball, null);
      this.put(h.ray, null);
    }
    this.hover();
    const onMenu = [...this.hands.values()].find((h) => seen.has(h) && h.menuHit?.row >= 0);
    this.menu.hover(onMenu ? onMenu.menuHit.row : -1);
    this.menu.tick();  // live values (FPS)
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

  // The IK marker the hand reaches, held.
  grab(h) {
    const { app } = this, ik = this.ik(), index = this.reach(h);
    const chain = index != null && ik?.chains[index];
    const target = chain && ![...this.hands.values()].some((o) => o.grab?.chain === chain) && ik.targetPose(chain, app.data);
    if (target) {
      ik.focus = index;
      app.teleop.held.set(index, BALL);
      h.grab = { kind: 'ik', ik, chain, index, c0: h.grip.pos, R0: h.grip.rot, p0: target.pos, T0: target.rot };
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
      const turn = mulMBt(h.grip.rot, g.R0);  // grip rotation since the grab
      app.teleop.held.set(g.index, BALL);
      g.ik.setTargetPose(g.chain, app.data, { pos: add(g.p0, sub(h.grip.pos, g.c0)), rot: mulMM(turn, g.T0) });
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

  // The IK or goal marker under any ray is highlighted (TeleopPanel.hover).
  hover() {
    const t = this.app.teleop;
    if (t.dragging) return;
    t.hover = null;
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
