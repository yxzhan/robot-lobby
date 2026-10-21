// Teleoperation: the Teleop panel picks which robot is driven, which of its
// controllers are on (any of them, or none: robot.use, switched by
// app.setUse) and the input; every other robot gets a zero command (it
// stands, or holds its base still). Controllers:
// - Policy (policy.js, robot.policy): a trained model picked from POLICIES,
//   e.g. GR00T locomotion, on by default for every G1 variant;
// - Base (base.js, robot.base): a mobile base, set up in the panel (kind,
//   wheels, radius, speeds, gains) or found in the model;
// - IK (ik.js, robot.ik): chains whose targets are RViz-style interactive
//   markers (marker.js) dragged with the mouse; the panel edits them.
// Locomotion policies and Base take a body twist command `cmd` = [vx, vy, yaw rate] and
// say how fast the keys drive them (`drive`, null while there is nothing to
// drive). Inputs (the Input select, where a ROS bridge or a gamepad would
// plug in too): the keyboard, or a goal marker on the floor (goal.js) that
// the robot drives to, dragged like a spawn marker (wheel or Q/E turn it).
// A robot keeps going to its goal while another one is driven, and lets it
// go on arriving (arrive). In network mode a robot's goal is the server's
// (server/goal.py), one for every browser: changes go there and come back
// to everyone (fromServer). Setups are saved per robot file.
// ROS (ros.js): any robot can also have a rosbridge connection of its own,
// switched on with its ROS chip: it publishes the robot's joint states and
// TF and takes joint commands and cmd_vel; a fresh cmd_vel drives the robot
// before the goal or the keys do.

import { dragPose, markerScale, pickMarker, startDrag } from './marker.js';
import { BaseController, KINDS } from './base.js';
import { POLICIES, POLICY_ROOT, defaultPolicy, isInstalled, policyById } from './policy.js';
import { bodyPose, goalCommand } from './goal.js';
import { RosLink, bridgeUrl } from './ros.js';

// Per held key: [command index (vx, vy, yaw rate), direction, drive speed].
const DRIVE = {
  w: [0, 1, 'forward'], s: [0, -1, 'back'], a: [1, 1, 'side'], d: [1, -1, 'side'],
  q: [2, 1, 'turn'], e: [2, -1, 'turn'],
};
const HEIGHT_RATE = 0.15;  // m/s while R/F is held
const HEIGHT_RANGE = [0.5, 1.0];
// Marker centers, by chain (at most this many markers are shown).
export const CHAIN_COLORS = [[0.2, 0.85, 1], [1, 0.35, 0.8], [1, 0.6, 0.15], [0.6, 0.4, 1],
  [0.3, 1, 0.75], [1, 1, 0.4], [0.55, 0.75, 1], [1, 0.55, 0.55]];

// Keys typed into a text field are not commands.
function isTyping(target) {
  if (!(target instanceof Element)) return false;
  if (target.closest('textarea, select, [contenteditable]:not([contenteditable=false])')) return true;
  return target.tagName === 'INPUT' && !['range', 'checkbox', 'radio', 'button', 'submit'].includes(target.type);
}

// Whether this is a touch screen (on-screen sticks, touch hints).
export const TOUCH = matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 0;

// WASD walk, QE turn, held R/F raise/lower the commanded base height; on a
// touch screen also two on-screen sticks (TouchSticks), left to walk and
// right to turn.
export class KeyboardInput {
  constructor() {
    this.held = new Set();
    this.stick = [0, 0, 0];  // [forward, left, turn left], each in [-1, 1]
    this.sticks = TOUCH ? new TouchSticks(this.stick) : null;
    const onKey = (e) => {
      const k = e.key.toLowerCase();
      if (!(k in DRIVE) && k !== 'r' && k !== 'f') return;
      if (e.type === 'keyup') { this.held.delete(k); return; }
      if (!this.enabled || e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return;
      e.preventDefault();
      this.held.add(k);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    window.addEventListener('blur', () => this.held.clear());
    this.enabled = false;  // only while a robot is being driven
  }

  // {cmd: [vx, vy, yaw rate], dh: height change} for the last `dt` seconds,
  // with the key speeds of `drive` ({forward, back, side, turn}).
  read(dt, drive) {
    const cmd = [0, 0, 0];
    for (const k of this.held) if (DRIVE[k]) cmd[DRIVE[k][0]] += DRIVE[k][1] * drive[DRIVE[k][2]];
    const [f, l, t] = this.stick;
    cmd[0] += f * (f > 0 ? drive.forward : drive.back);
    cmd[1] += l * drive.side;
    cmd[2] += t * drive.turn;
    return { cmd, dh: (this.held.has('r') - this.held.has('f')) * HEIGHT_RATE * dt };
  }
}

// Two on-screen sticks at the bottom of the screen: the left one walks
// (up forward, sideways strafes), the right one turns. They write `stick`
// ([forward, left, turn left] in [-1, 1]) and spring back when let go.
class TouchSticks {
  constructor(stick) {
    this.el = el('div', { id: 'sticks', hidden: true });
    const pad = (axes, label) => {
      const knob = el('div', { className: 'knob' });
      const p = el('div', { className: `stick ${label}` }, knob);
      let id = null, c = null;
      const set = (x, y) => {
        const r = p.clientWidth / 2;
        let dx = (x - c[0]) / r, dy = (y - c[1]) / r;
        const n = Math.hypot(dx, dy);
        if (n > 1) { dx /= n; dy /= n; }
        if (axes === 'move' && Math.abs(dx) < 0.15) dx = 0;  // a dead zone across: straight ahead stays straight
        knob.style.transform = `translate(${dx * r}px, ${dy * r}px)`;
        if (axes === 'move') { stick[0] = -dy; stick[1] = -dx; } else stick[2] = -dx;
      };
      p.addEventListener('pointerdown', (e) => {
        if (id !== null) return;
        e.preventDefault();
        id = e.pointerId;
        p.setPointerCapture(id);
        const b = p.getBoundingClientRect();
        c = [b.left + b.width / 2, b.top + b.height / 2];
        set(e.clientX, e.clientY);
      });
      p.addEventListener('pointermove', (e) => { if (e.pointerId === id) set(e.clientX, e.clientY); });
      const end = (e) => {
        if (e.pointerId !== id) return;
        id = null;
        knob.style.transform = '';
        if (axes === 'move') { stick[0] = 0; stick[1] = 0; } else stick[2] = 0;
      };
      p.addEventListener('pointerup', end);
      p.addEventListener('pointercancel', end);
      return p;
    };
    this.el.append(pad('move', 'left'), pad('turn', 'right'));
    document.body.append(this.el);
    this.stick = stick;
  }

  show(on) {
    if (this.el.hidden === !on) return;
    this.el.hidden = !on;
    if (!on) this.stick.fill(0);
  }
}

const FOLDED = 'mujoco-filament/teleop-folded';
const INPUT = 'mujoco-filament/teleop-input';
const INPUTS = [['keyboard', 'Keyboard'], ['goal', 'Goal marker']];
const GOAL_TURN = Math.PI / 12;  // per wheel notch or Q/E press while dragging a goal (like assets.js)
const GOAL_GRAB = 0.35;          // m: grab radius around the goal marker's center
const GOAL_GAP_MS = 50;          // network mode: a dragged goal goes to the server at most this often
const GOAL_ECHO_MS = 600;        // and the server's echoes of it are ignored for this long
const DEG = 180 / Math.PI;
const CONTROLLERS = [['policy', 'Policy', 'A trained policy (locomotion, hands, …)'],
  ['base', 'Base', 'Mobile base: wheels or planar joints'], ['ik', 'IK', 'IK chains with interactive markers']];

// The switched-on controllers of robot `r` that take twist commands.
export function locomotion(r) {
  return [r?.use?.policy && r.policy?.drive && r.policy, r?.use?.base && r.base?.drive && r.base].filter(Boolean);
}

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

export class TeleopPanel {
  constructor(app, root) {
    this.app = app;
    this.el = root;
    this.body = root.querySelector('.body');
    this.active = null;      // the driven robot
    this.known = new Set();  // controllable robots seen so far; a new one gets selected
    this.input = new KeyboardInput();
    this.ros = new Map();    // robot prefix -> RosLink (ros.js)
    this.rosView = null;     // the ROS section's live parts, while shown
    this.readout = null;
    this.errs = [];          // per chain: the span showing its tracking error
    this.grips = [];         // per chain with a gripper: {chain, slider, label} (its closure, kept up to date)
    this.closedChains = new Set();  // "<prefix><chain name>" folded in the panel
    // Folded sections ('base', 'ik'), kept across robots and page loads.
    try { this.folded = new Set(JSON.parse(localStorage.getItem(FOLDED) ?? '[]')); } catch { this.folded = new Set(); }
    this.dragging = null;    // marker.js drag state + {index, chain} while a marker is dragged, or {goal} for the goal marker
    this.hover = null;       // {index, part} of the marker under the pointer, or {goal: true}
    this.held = new Map();   // chain index -> marker part, held by a WebXR controller (xr.js)
    try { this.mode = localStorage.getItem(INPUT) === 'keyboard' ? 'keyboard' : 'goal'; } catch { this.mode = 'goal'; }
    this.goalReadout = null;
    // While the goal marker is dragged, the wheel and Q/E turn it and Esc puts
    // it back (capture phase: before the camera zoom and the drive keys).
    window.addEventListener('wheel', (e) => {
      if (!this.dragging?.goal) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      this.turnGoal(Math.sign(e.deltaY || e.deltaX) * GOAL_TURN);
    }, { capture: true, passive: false });
    window.addEventListener('keydown', (e) => {
      if (!this.dragging?.goal) return;
      const k = e.key.toLowerCase();
      if (k === 'q' || k === 'e') this.turnGoal(k === 'q' ? GOAL_TURN : -GOAL_TURN);
      else if (k === 'escape') { this.setGoal(this.dragging.robot, this.dragging.before); this.release(); }
      else return;
      e.preventDefault();
      e.stopImmediatePropagation();
    }, { capture: true });
    // G opens or closes the gripper of the chain dragged last.
    window.addEventListener('keydown', (e) => {
      if (e.key.toLowerCase() !== 'g' || e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return;
      const ik = this.active?.use?.ik && this.active.ik;
      const chain = ik?.enabled && ik.chains[ik.focus];
      if (chain) ik.toggleGripper(chain, this.app.data);
    });
    this.last = performance.now();
  }

  // Every frame, from App.frame (the window's animation loop, or the WebXR
  // session's: the window's stops in VR).
  update(now) {
    this.tick((now - this.last) / 1000);
    this.last = now;
  }

  controllable() {
    return (this.app.robots ?? []).filter((r) => r.use);
  }

  // Called for every new world (App.setWorld).
  build() {
    const robots = this.controllable();
    const added = robots.filter((r) => !this.known.has(r));
    this.known = new Set(robots);
    // ROS links follow their robot into the new world (by prefix: in
    // network mode every scene brings new robot objects); a removed robot's closes.
    for (const [prefix, link] of this.ros) {
      const r = robots.find((x) => x.prefix === prefix);
      if (r) link.robot = r;
      else { link.setOn(false); this.ros.delete(prefix); }
    }
    if (added.length) this.active = added.at(-1);
    else if (!robots.includes(this.active)) this.active = robots.at(-1) ?? null;
    this.dragging = this.hover = null;
    this.render();
  }

  // Drives `robot` (e.g. double-clicked) if it is controllable; anything
  // else (null, the floor, an object) releases control.
  drive(robot) {
    const next = robot?.use ? robot : null;
    if (next === this.active) { this.render(); return; }  // the selected body may matter (IK chain editor)
    this.active = next;
    this.render();
  }

  // ---- IK target markers (marker.js, drawn by App.syncMarkers) ----

  // The driven robot's IK targets, by chain: {pos, rot, scale, hot} (hot:
  // the marker part hovered or dragged), or null for a broken chain.
  gizmos() {
    const ik = this.active?.use?.ik && this.active.ik;
    if (!ik?.enabled || !this.app.data) return [];
    const g = this.dragging;
    return ik.chains.slice(0, CHAIN_COLORS.length).map((c, i) => {
      const pose = ik.targetPose(c, this.app.data);
      if (!pose) return null;
      const hot = this.held.has(i) ? this.held.get(i)
        : g ? (g.index === i ? g.part : -1) : (this.hover?.index === i ? this.hover.part : -1);
      return { ...pose, scale: markerScale(), hot };
    });
  }

  // The marker part under the pointer ray: {index, part, depth} or null.
  pick(ray) {
    let best = null;
    this.gizmos().forEach((m, index) => {
      const hit = m && pickMarker(ray, m, m.scale);
      if (hit && (!best || hit.depth < best.depth)) best = { index, ...hit };
    });
    return best;
  }

  // Right button on an arm's marker, or pressed while dragging one (the left
  // button held): opens or closes that chain's gripper, like G. False if no
  // marker is concerned.
  gripperClick(ray) {
    const ik = this.active?.use?.ik && this.active.ik;
    if (!ik?.enabled) return false;
    const index = this.dragging?.chain ? this.dragging.index : this.pick(ray)?.index;
    const chain = index != null && ik.chains[index];
    if (!chain) return false;
    ik.focus = index;
    ik.toggleGripper(chain, this.app.data);
    return true;
  }

  // Pointer moved with no button down: highlights the part under it.
  hoverAt(ray) {
    const hit = this.dragging ? null : this.pick(ray) ?? (this.pickGoal(ray) && { goal: true });
    this.hover = hit;
    return !!hit;
  }

  // ---- goal marker (goal.js, drawn by App.syncMarkers) ----

  // The driven robot's goal while the goal input is on: its own, else where
  // it stands (grabbing the marker there sets one).
  goalOf(r = this.active) {
    if (this.mode !== 'goal' || !r || !locomotion(r).length || !this.app.data) return null;
    if (r.goal) return r.goal;
    const p = bodyPose(this.app.data, r.bodyOffset);
    return { x: p.x, y: p.y, yaw: p.yaw };
  }

  // {pos, yaw, hot} for the goal marker, or null.
  goalMarker() {
    const g = this.goalOf();
    return g && { pos: [g.x, g.y, 0], yaw: g.yaw, hot: !!(this.dragging?.goal || this.hover?.goal) };
  }

  static floorHit({ eye, dir }) {
    if (dir[2] > -1e-6 || eye[2] <= 0) return null;
    const s = -eye[2] / dir[2];
    return [eye[0] + s * dir[0], eye[1] + s * dir[1]];
  }

  pickGoal(ray) {
    const g = this.goalOf(), p = g && TeleopPanel.floorHit(ray);
    return p && Math.hypot(p[0] - g.x, p[1] - g.y) < GOAL_GRAB ? p : null;
  }

  turnGoal(delta) {
    const r = this.dragging?.robot;
    if (!r?.goal) return;
    r.goal.yaw += delta;
    this.setGoal(r, r.goal);
    this.showGoal();
  }

  // Sets robot r's goal ({x, y, yaw} or null); in network mode it goes to
  // the server, at most every GOAL_GAP_MS while it is dragged.
  setGoal(r, goal) {
    r.goal = goal;
    r.goalState = {};
    r.arrived = false;
    const net = this.app.net;
    if (!net) return;
    clearTimeout(r.goalTimer);
    const send = () => {
      r.goalSentAt = performance.now();
      net.send('goal', { id: r.netId, goal: r.goal && { x: r.goal.x, y: r.goal.y, yaw: r.goal.yaw } });
    };
    const wait = (r.goalSentAt ?? -Infinity) + GOAL_GAP_MS - performance.now();
    if (wait <= 0) send();
    else r.goalTimer = setTimeout(send, wait);
  }

  // Network mode: robot r's goal as the server has it (set by any browser).
  // `arrived`: the goal the robot reached, which the server let go.
  fromServer(r, goal, arrived) {
    if (this.dragging?.goal && this.dragging.robot === r) return;
    if (arrived) {
      const g = r.goal, same = g && ['x', 'y', 'yaw'].every((k) => Math.abs(g[k] - arrived[k]) < 1e-6);
      if (same) this.arrive(r);  // (not a newer one of ours, on its way)
      return;
    }
    if (performance.now() - (r.goalSentAt ?? -Infinity) < GOAL_ECHO_MS) return;  // ours, or an older one
    r.goal = goal;
    r.goalState = {};
    r.arrived = false;
  }

  // At the goal: it is let go, so a robot pushed or carried off afterwards
  // stays where it is put (the marker goes back under it).
  arrive(r) {
    clearTimeout(r.goalTimer);
    r.goal = null;
    r.goalState = {};
    r.goalResult = null;
    r.arrived = true;
  }

  // Forgets the goals of `robots` (Reset): they stop where they are.
  clearGoals(robots) {
    for (const r of robots) {
      clearTimeout(r.goalTimer);
      r.goal = null;
      r.goalState = {};
      r.goalResult = null;
      r.arrived = false;
    }
    if (this.dragging?.goal && robots.includes(this.dragging.robot)) this.release();
  }

  // While the goal marker is dragged: a label by the pointer with the goal's
  // position and heading, like the one of a dragged asset (assets.js).
  showGoal(at) {
    const d = this.dragging;
    if (!d?.goal || this.app.xr) return;  // (not on the page in VR)
    if (!d.ghost) {
      d.ghost = el('div', { className: 'drag-ghost placing' },
        el('div', { textContent: `goal · ${d.robot.prefix.slice(0, -1)} · ${d.robot.label}` }), el('div', { className: 'readout' }));
      document.body.append(d.ghost);
    }
    if (at) d.ghost.style.transform = `translate(${at[0] + 14}px, ${at[1] + 10}px)`;
    const g = d.robot.goal, deg = ((g.yaw * DEG) % 360 + 540) % 360 - 180;  // [-180, 180)
    d.ghost.lastChild.textContent = `x ${g.x.toFixed(2)}  y ${g.y.toFixed(2)} m · yaw ${deg.toFixed(0)}°`;
  }

  // Starts dragging the marker part under the ray; false if there is none.
  // `at`: the pointer's client coordinates (for the goal's label).
  grab(ray, at) {
    const p = !this.pick(ray) && this.pickGoal(ray);
    if (p) {
      const r = this.active, g = this.goalOf();
      this.dragging = { goal: true, robot: r, before: r.goal, offset: [g.x - p[0], g.y - p[1]] };
      this.setGoal(r, { ...g });
      this.showGoal(at);
      return true;
    }
    const hit = this.pick(ray);
    if (!hit) return false;
    const ik = this.active.ik, chain = ik.chains[hit.index];
    const m = this.gizmos()[hit.index];
    const state = startDrag(hit.part, ray, m, m.scale, ray.forward);
    if (!state) return false;
    ik.focus = hit.index;
    this.dragging = { ...state, index: hit.index, chain };
    return true;
  }

  dragTo(ray, at) {
    const g = this.dragging;
    if (g?.goal) {
      const p = TeleopPanel.floorHit(ray);
      if (p) this.setGoal(g.robot, { ...g.robot.goal, x: p[0] + g.offset[0], y: p[1] + g.offset[1] });
      this.showGoal(at);
      return;
    }
    if (!g || !this.active?.ik) return;
    const pose = dragPose(g, ray);
    if (pose) this.active.ik.setTargetPose(g.chain, this.app.data, pose);
  }

  release() {
    const d = this.dragging;
    // The goal once more as let go: the server may have dropped it as reached meanwhile.
    if (d?.goal && d.robot.goal) this.setGoal(d.robot, d.robot.goal);
    this.dragging?.ghost?.remove();
    this.dragging = null;
  }

  // ---- panel ----

  render() {
    const robots = this.controllable();
    this.body.replaceChildren();
    this.readout = null;
    this.errs = [];
    this.grips = [];
    if (!robots.length) {
      this.body.innerHTML = '<div class="hint">Nothing to drive: spawn a robot, e.g. a Unitree G1 (Robots › Humanoids), which walks under the GR00T policy, or Garmi (mecanum base, two arms).</div>';
      return;
    }
    const field = (label, ...controls) => {
      const row = el('label', { className: 'field' }, el('span', { textContent: label }), ...controls);
      this.body.append(row);
      return controls[0];
    };
    field('Robot', this.select(
      [['', 'none'], ...robots.map((r) => [r.prefix, `${r.prefix.slice(0, -1)} · ${r.label}`])],
      this.active?.prefix ?? '',
      (v) => { this.active = robots.find((r) => r.prefix === v) ?? null; this.render(); },
    ));
    const r = this.active;
    if (!r) return;
    // Controllers: any of them on, or none (e.g. GR00T off to pose a G1 by hand).
    const chips = el('div', { className: 'chips' });
    for (const [kind, label, title] of CONTROLLERS) {
      const box = el('input', { type: 'checkbox', checked: !!r.use[kind] });
      box.onchange = () => {
        box.disabled = true;
        // Policy: the one made for this robot, else the last picked, else the first.
        const on = kind !== 'policy' ? box.checked
          : box.checked && (defaultPolicy(r.key) || r.lastPolicy || POLICIES.find(isInstalled)?.id || POLICIES[0]?.id || '');
        this.app.setUse(r, kind, on).finally(() => this.render());
      };
      chips.append(el('label', { className: 'chip' + (r.errors?.[kind] ? ' bad' : ''), title: r.errors?.[kind] ?? title }, box, el('span', { textContent: label })));
    }
    const link = this.ros.get(r.prefix);
    const rosBox = el('input', { type: 'checkbox', checked: !!link?.on });
    rosBox.onchange = () => {
      const l = this.ros.get(r.prefix) ?? new RosLink(this.app, r);
      this.ros.set(r.prefix, l);
      l.setOn(rosBox.checked);
      this.render();
    };
    chips.append(el('label', { className: 'chip' + (link?.status === 'error' ? ' bad' : ''), title: 'A rosbridge connection of this robot\'s own: joint states, TF, joint commands, cmd_vel' },
      rosBox, el('span', { textContent: 'ROS' })));
    this.body.append(el('div', { className: 'field' }, el('span', { textContent: 'Controllers' }), chips));
    const locos = locomotion(r);
    if (locos.length) {
      field('Input', this.select(INPUTS, this.mode, (v) => {
        this.mode = v;
        try { localStorage.setItem(INPUT, v); } catch { /* no storage */ }
        this.setGoal(r, null);  // a new goal starts where the robot is
        this.render();
      }));
      const keys = el('div', { className: 'hint' });
      if (this.mode === 'goal') {
        keys.innerHTML = TOUCH
          ? 'Drag the ring on the floor to where the robot should go; a second finger twisted around it turns it (the arrow is the heading).'
          : 'Drag the ring on the floor to where the robot should go; while dragging, the mouse wheel or '
            + '<kbd>Q</kbd><kbd>E</kbd> turn it (the arrow is the heading), <kbd>Esc</kbd> puts it back.';
        this.goalReadout = el('span', { className: 'note' });
        this.body.append(keys, el('div', { className: 'bar' },
          this.button('Stop', 'Forget the goal: the robot stops where it is', () => this.setGoal(r, null)), this.goalReadout));
      } else {
        this.goalReadout = null;
        keys.innerHTML = '<kbd>W</kbd><kbd>S</kbd> forward/back · ' + (locos.some((c) => c.drive.side) ? '<kbd>A</kbd><kbd>D</kbd> sideways · ' : '')
          + '<kbd>Q</kbd><kbd>E</kbd> turn' + (locos.some((c) => 'height' in c) ? ' · hold <kbd>R</kbd><kbd>F</kbd> height' : '')
          + (TOUCH ? ' · or the sticks at the bottom: left walks, right turns' : '');
        this.body.append(keys);
      }
      this.readout = el('div', { className: 'readout' });
      this.body.append(this.readout);
    }
    if (r.use.policy || r.errors?.policy) this.renderPolicy(r);
    if (r.use.base && r.base) this.renderBase(r.base);
    if (r.use.ik && r.ik) this.renderIk(r.ik);
    this.rosView = null;
    if (link?.on) this.renderRos(link);
  }

  // The robot's rosbridge connection (ros.js): address and state, the
  // topics, the publishing rate.
  renderRos(link) {
    const s = link.settings;
    const box = this.section('ros', 'ROS bridge', '');
    box.classList.add('ros');
    const changed = () => { link.save(); link.sync(); };
    const row = (label, title, ...controls) => box.append(el('div', { className: 'field', title }, el('span', { textContent: label }), el('div', { className: 'ctl' }, ...controls)));
    const url = el('input', { type: 'text', value: s.url, placeholder: 'localhost:9090', spellcheck: false });
    const apply = () => {
      if (url.value.trim() === s.url) return;
      s.url = url.value.trim() || 'localhost:9090';
      link.save();
      link.connect();
    };
    url.onchange = apply;
    url.onkeydown = (e) => { if (e.key === 'Enter') { apply(); url.blur(); } };
    row('Bridge', 'rosbridge_server\'s WebSocket (host:port, or a ws:// / wss:// URL)', url,
      this.button('↻', 'Connect again', () => link.connect()));
    const state = el('span', { className: 'ros-state' });
    box.append(el('div', { className: 'bar' }, state));
    const topic = (k, label, title, enabled = true) => {
      const on = el('input', { type: 'checkbox', checked: s[k].on, disabled: !enabled });
      const name = el('input', { type: 'text', value: s[k].topic, disabled: !enabled, spellcheck: false });
      on.onchange = () => { s[k].on = on.checked; changed(); };
      name.onchange = () => { s[k].topic = name.value.trim(); changed(); };
      name.onkeydown = (e) => { if (e.key === 'Enter') name.blur(); };
      row(label, title, on, name);
    };
    box.append(el('div', { className: 'sub', textContent: 'Publish' }));
    topic('jointStates', 'Joints', 'sensor_msgs/JointState: the robot\'s joint positions (and velocities in local mode)');
    topic('tf', 'TF', 'tf2_msgs/TFMessage: every body in its parent\'s frame, the root in "world"');
    {
      // Cameras: the namespace of their topics (the robot's id unless set), and JPEG or raw.
      const c = s.camera;
      const on = el('input', { type: 'checkbox', checked: c.on });
      const ns = el('input', { type: 'text', value: c.topic, placeholder: `${link.robot.prefix.slice(0, -1)} (robot id)`, title: 'Namespace of the camera topics: empty, the robot\'s id; / for none', spellcheck: false });
      on.onchange = () => { c.on = on.checked; changed(); };
      ns.onchange = () => { c.topic = ns.value.trim(); changed(); };
      ns.onkeydown = (e) => { if (e.key === 'Enter') ns.blur(); };
      const format = this.select([['jpeg', 'JPEG'], ['raw', 'raw']], c.format, (v) => { c.format = v; changed(); });
      row('Cameras', 'The cameras ticked in the Cameras panel, each frame: <namespace>/<camera>/image_raw/compressed (sensor_msgs/CompressedImage, JPEG) '
        + 'or …/image_raw (sensor_msgs/Image, rgb8), and …/camera_info; with TF, <camera>_optical_frame in its link\'s frame', on, ns, format);
    }
    box.append(el('div', { className: 'sub', textContent: 'Subscribe' }));
    topic('jointCommand', 'Joint cmd', 'sensor_msgs/JointState: position targets for the actuators on those joints');
    topic('cmdVel', 'cmd_vel', 'geometry_msgs/Twist: linear x, y and angular z drive the walking policy or the base');
    const hz = el('span', { className: 'unit', textContent: `${s.rate} Hz` });
    const rate = el('input', { type: 'range', min: 1, max: 60, step: 1, value: s.rate });
    rate.oninput = () => { s.rate = +rate.value; hz.textContent = `${s.rate} Hz`; };
    rate.onchange = () => { link.save(); link.restartTimer(); rate.blur(); };
    row('Rate', 'how often joint states and TF are published', rate, hz);
    const r = link.robot, busy = [r.use?.policy && 'the policy', r.use?.ik && 'IK'].filter(Boolean);
    if (s.jointCommand.on && busy.length) {
      box.append(el('div', { className: 'note warn', textContent: `${busy.join(' and ')} on: joint commands only move the joints ${busy.length > 1 ? 'they leave' : 'it leaves'} alone (untick ${busy.length > 1 ? 'them' : 'it'} above)` }));
    }
    const counts = el('div', { className: 'readout' });
    box.append(counts, el('div', { className: 'hint', textContent:
      'Joint commands set the actuators\' targets; a running policy or IK writes over the joints it drives. ' +
      'Cameras publish while ticked in the Cameras panel, at their own rate. ' +
      'Topics are per robot connection, so robots on one bridge need topics of their own.' }));
    this.rosView = { link, state, counts };
    link.onChange = () => this.showRos();
    this.showRos();
  }

  // The ROS section's state line and counters (each frame, and on changes).
  showRos() {
    const v = this.rosView;
    if (!v) return;
    const { link } = v;
    const text = { off: 'off', connecting: `connecting to ${bridgeUrl(link.settings.url)}…`, connected: `connected to ${bridgeUrl(link.settings.url)}`,
      error: `${link.error} · retrying` }[link.status];
    if (v.state.textContent !== text) v.state.textContent = text;
    v.state.className = `ros-state ${link.status}`;
    const tw = link.freshTwist();
    const counts = `sent ${link.sent} · received ${link.received}` + (link.dropped ? ` · camera frames dropped ${link.dropped}` : '') + (tw ? `\ncmd_vel vx ${tw[0].toFixed(2)} vy ${tw[1].toFixed(2)} ωz ${tw[2].toFixed(2)}` : '');
    if (v.counts.textContent !== counts) v.counts.textContent = counts;
  }

  // Which trained policy runs (POLICIES in policy.js), and its state.
  renderPolicy(r) {
    const p = policyById(r.use.policy);
    const box = this.section('policy', 'Policy', r.errors?.policy ? 'error' : p ? p.name.split(' · ')[0] : 'none');
    const models = this.select([['', 'none'], ...POLICIES.map((q) => [q.id, q.name + (isInstalled(q) ? '' : ' (not installed)')])],
      r.use.policy ?? '', (v) => {
        if (v) r.lastPolicy = v;
        this.app.setUse(r, 'policy', v).finally(() => this.render());
      });
    for (const o of models.options) {
      const q = policyById(o.value);
      if (q && !isInstalled(q)) { o.disabled = true; o.title = `put it in ${POLICY_ROOT}/${q.dir}/`; }
    }
    const row = el('div', { className: 'field' }, el('span', { textContent: 'Model' }), el('div', { className: 'ctl' }, models));
    box.append(row);
    // World settings the policy needs (its physics), if they differ.
    const ph = this.app.physics, want = p?.physics ?? {};
    const off = Object.keys(want).filter((k) => JSON.stringify(ph[k]) !== JSON.stringify(want[k]));
    if (r.use.policy && off.length) {
      const text = off.map((k) => `${k} ${want[k]}`).join(', ');
      box.append(el('div', { className: 'bar' },
        el('span', { className: 'note warn', textContent: `Needs ${text} (World panel)` }),
        this.button('Apply', `Set ${text} for the whole world`, () => {
          this.app.setPhysics({ ...structuredClone(ph), ...structuredClone(want) }).then(() => this.render());
        })));
    }
    const hint = el('div', { className: 'hint' });
    hint.textContent = (p ? `${p.about} ` : '') + 'Untick Policy to give the joints their own actuator gains back.';
    box.append(hint);
    if (r.errors?.policy) box.append(el('div', { className: 'error', textContent: r.errors.policy }));
  }

  // A foldable panel section: <details> with a summary of `title`, a short
  // `status` and optional leading controls; its fold is remembered by `key`.
  section(key, title, status, ...lead) {
    const det = el('details', { className: 'ik', open: !this.folded.has(key) });
    det.ontoggle = () => {
      if (det.open) this.folded.delete(key); else this.folded.add(key);
      try { localStorage.setItem(FOLDED, JSON.stringify([...this.folded])); } catch { /* no storage */ }
    };
    det.append(el('summary', { className: 'ik-head' }, ...lead, el('span', { textContent: title }),
      el('span', { className: 'count', textContent: status })));
    this.body.append(det);
    return det;
  }

  num(value, step, onchange, title = '') {
    const inp = el('input', { type: 'number', step, value: +(+value).toFixed(4), title });
    inp.onchange = () => onchange(Number(inp.value) || 0);
    return inp;
  }

  // The mobile-base setup of the driven robot (base.js).
  renderBase(c) {
    const cfg = c.config ?? {};
    const box = this.section('base', 'Base', c.error ? 'error' : cfg.kind ? (c.config.kind === 'planar' ? 'planar' : `${cfg.wheels?.length ?? 0} wheels`) : 'none');
    box.classList.add('base');
    const row = (label, ...controls) => box.append(el('div', { className: 'field' }, el('span', { textContent: label }), el('div', { className: 'ctl' }, ...controls)));
    const changed = () => { c.resolve(); c.reset(); this.render(); };
    const { model } = c;
    const T = (k) => this.app.mujoco.mjtObj[k].value;
    const robotActs = [];
    for (let a = 0; a < model.nu; a++) {
      const n = this.app.mujoco.mj_id2name(model, T('mjOBJ_ACTUATOR'), a) ?? '';
      if (n.startsWith(c.prefix)) robotActs.push([a, n.slice(c.prefix.length), BaseController.mode(model, a)]);
    }
    row('Kind', this.select(KINDS, cfg.kind ?? '', (v) => {
      const auto = c.autoConfig();
      c.config = { ...cfg, ...(auto.kind === v ? auto : {}), kind: v, auto: false };
      if (v === 'wheels') { c.config.wheels ??= []; c.config.radius ??= 0.05; }
      changed();
    }));
    if (!cfg.kind) {
      box.append(el('div', { className: 'hint', textContent: 'No mobile base found in this robot; pick a kind to set one up.' }));
    } else if (cfg.kind === 'planar') {
      const pos = robotActs.filter(([, , m]) => m === 'position').map(([, n]) => [n, n]);
      for (const k of ['x', 'y', 'yaw']) {
        row(k, this.select([['', '—'], ...pos], cfg.planar?.[k] ?? '', (v) => { cfg.planar = { ...cfg.planar, [k]: v }; changed(); }));
      }
    } else {
      const frees = [];
      for (let j = 0; j < model.njnt; j++) {
        if (model.jnt_type[j] === 0 && c.inRobot(model.jnt_bodyid[j])) frees.push(c.nameOf('mjOBJ_JOINT', j));
      }
      row('Base joint', this.select([['', 'root free joint'], ...frees.filter(Boolean).map((n) => [n, n])], cfg.base ?? '', (v) => { cfg.base = v; changed(); }));
      const spin = robotActs.filter(([, , m]) => m === 'velocity' || m === 'motor').map(([, n, m]) => [n, `${n} (${m})`]);
      // Hinge joints too, for a software servo (no actuator, or a broken one).
      for (let j = 0; j < model.njnt; j++) {
        if (model.jnt_type[j] === 3 && c.inRobot(model.jnt_bodyid[j])) spin.push([`joint:${c.nameOf('mjOBJ_JOINT', j)}`, `joint ${c.nameOf('mjOBJ_JOINT', j)} (software servo)`]);
      }
      const rollers = [['0', 'plain wheel'], ['1', 'rollers +45°'], ['-1', 'rollers −45°']];
      (cfg.wheels ?? []).forEach((w, i) => {
        row(`Wheel ${i + 1}`,
          this.select([['', '—'], ...spin], w.joint ? `joint:${w.joint}` : w.actuator, (v) => {
            if (v.startsWith('joint:')) { w.joint = v.slice(6); delete w.actuator; } else { w.actuator = v; delete w.joint; }
            changed();
          }),
          this.select(rollers, String(w.roller ?? 0), (v) => { w.roller = Number(v); changed(); }),
          this.button('✕', 'Remove this wheel', () => { cfg.wheels.splice(i, 1); changed(); }));
      });
      row('', this.button('+ Wheel', 'Add a driven wheel (a velocity servo or motor on its joint)', () => {
        const used = new Set(cfg.wheels.map((w) => (w.joint ? `joint:${w.joint}` : w.actuator)));
        const pick = spin.find(([n]) => !used.has(n))?.[0] ?? '';
        cfg.wheels.push(pick.startsWith('joint:') ? { joint: pick.slice(6), roller: 0 } : { actuator: pick, roller: 0 });
        changed();
      }));
      row('Radius', this.num(cfg.radius, 0.005, (v) => { cfg.radius = v; changed(); }, 'wheel radius (m)'));
    }
    if (cfg.kind) {
      const d = cfg.drive ?? {};
      row('Speeds', ...['forward', 'back', 'side', 'turn'].map((k) => this.num(d[k] ?? 0, 0.05,
        (v) => { cfg.drive = { ...d, [k]: v }; changed(); }, `${k} (${k === 'turn' ? 'rad/s' : 'm/s'}) for the keys`)));
      if (cfg.kind === 'planar') {
        row('Accel', ...[0, 1, 2].map((i) => this.num(cfg.accel?.[i] ?? 1, 0.1, (v) => { cfg.accel[i] = v; changed(); }, 'acceleration limit (m/s², rad/s²)')));
      }
      if (cfg.kind === 'wheels') {
        const grid = el('div', { className: 'gains' }, el('span'), ...['vx', 'vy', 'ωz'].map((t) => el('span', { textContent: t })));
        for (const [k, title] of [['ff', 'feedforward'], ['kp', 'P gain'], ['ki', 'I gain'], ['iclamp', 'integral limit'], ['accel', 'acceleration limit (m/s², rad/s²)']]) {
          grid.append(el('span', { textContent: k, title }), ...[0, 1, 2].map((i) => this.num(cfg[k]?.[i] ?? 0, 0.1, (v) => { cfg[k][i] = v; changed(); }, title)));
        }
        box.append(grid);
        if (c.wheels?.some((w) => w.mode === 'motor')) row('Motor gain', this.num(cfg.kw ?? 1, 0.1, (v) => { cfg.kw = v; changed(); }, 'torque per rad/s of wheel speed error'));
      }
    }
    if (c.error) box.append(el('div', { className: 'error', textContent: c.error }));
    const note = el('span', { className: 'note', textContent: c.saved ? 'saved' : cfg.auto ? 'automatic' : '' });
    box.append(el('div', { className: 'bar' },
      this.button('Save', `Use this base setup for every spawn of this robot (${this.app.net ? 'on the server, for everyone' : 'this browser'})`, () => { c.save(); this.render(); }),
      this.button(c.hasDefaults ? 'Defaults' : 'Auto', c.hasDefaults ? 'Back to the built-in base setup' : 'Back to the setup found in the model',
        () => { c.restoreDefaults(); c.reset(); this.render(); }),
      this.button('Export', 'Copy the setup as JSON (for base_defaults.json)', () => {
        const json = JSON.stringify({ [c.key]: c.config }, null, 2);
        navigator.clipboard?.writeText(json).catch(() => {});
        console.log(json);
        note.textContent = 'copied (also in the console)';
      }),
      note));
  }

  select(options, value, onchange) {
    const s = el('select');
    for (const [v, text] of options) s.append(new Option(text, v, false, v === value));
    s.onchange = () => { onchange?.(s.value); s.blur(); };  // blur: keys drive again
    s.disabled = options.length < 2;
    return s;
  }

  button(text, title, onclick) {
    return el('button', { type: 'button', textContent: text, title, onclick: (e) => { e.preventDefault(); onclick(); } });
  }

  // The selected (double-clicked) body if it belongs to the driven robot.
  selectedBody(ik) {
    const b = this.app.perturb.body;
    return b >= ik.bodies[0] && b < ik.bodies[1] ? b : -1;
  }

  renderIk(ik) {
    const box = this.section('ik', 'IK chains', `${ik.chains.length}`);
    const hint = el('div', { className: 'hint' });
    hint.innerHTML = ik.chains.length
      ? 'Drag a marker: arrows move along an axis, rings turn about it, the ball moves in the view plane · '
        + '<kbd>G</kbd> or a right-click (on a marker, or while dragging it) opens/closes its gripper. Pick bodies for tip/root by double-clicking them, then ◎.'
      : 'No IK chains found in this robot: double-click the body to move (e.g. a hand), then + Chain.';
    box.append(hint);

    const changed = (cfg, keepTarget = false) => {
      if (!keepTarget) ik.targets.delete(cfg.name);
      ik.resolve();
      this.render();
    };
    ik.chains.forEach((chain, i) => box.append(this.renderChain(ik, chain, i, changed)));

    const bar = el('div', { className: 'bar' });
    bar.append(
      this.button('+ Chain', 'New IK chain: tip = the selected body, root = the robot\'s base', () => {
        const names = new Set(ik.config.chains.map((c) => c.name));
        let n = ik.config.chains.length + 1;
        while (names.has(`chain${n}`)) n++;
        const sel = this.selectedBody(ik);
        const list = ik.bodyList();
        const tip = sel >= 0 ? sel : list.at(-1)[0];
        const path = ik.ancestors(tip);
        ik.config.chains.push({
          name: `chain${n}`, tip: ik.nameOf('mjOBJ_BODY', tip),
          root: ik.nameOf('mjOBJ_BODY', path[Math.min(path.length - 1, 6)]),  // a few joints up, not the whole tree
          site: '', offset: [0, 0, 0], locked: [], gripper: '',
        });
        ik.enabled = true;
        changed(ik.config.chains.at(-1));
      }),
      this.button('Save', `Use this IK setup for every spawn of this robot (${this.app.net ? 'on the server, for everyone' : 'this browser'})`, () => { ik.save(); this.render(); }),
      this.button(ik.hasDefaults ? 'Defaults' : 'Auto', ik.hasDefaults ? 'Back to the built-in IK setup' : 'Back to the chains found in the model', () => { ik.restoreDefaults(); this.render(); }),
      this.button('Export', 'Copy the setup as JSON (for ik_defaults.json)', () => {
        const json = JSON.stringify({ [ik.key]: ik.config }, null, 2);
        navigator.clipboard?.writeText(json).catch(() => {});
        console.log(json);
        exported.textContent = 'copied (also in the console)';
      }),
    );
    const exported = el('span', { className: 'note', textContent: ik.saved ? 'saved' : ik.config.auto ? 'automatic' : '' });
    bar.append(exported);
    box.append(bar);
  }

  renderChain(ik, chain, i, changed) {
    const { cfg } = chain;
    const key = this.active.prefix + cfg.name;
    const det = el('details', { className: 'chain', open: !this.closedChains.has(key) });
    det.ontoggle = () => (det.open ? this.closedChains.delete(key) : this.closedChains.add(key));
    const err = el('span', { className: 'err' });
    this.errs[i] = err;
    det.append(el('summary', {},
      el('span', { className: 'dot', style: `background:rgb(${(CHAIN_COLORS[i] ?? [0.5, 0.5, 0.5]).map((v) => v * 255).join(',')})` }),
      el('span', { className: 'title', textContent: (cfg.tip.startsWith(cfg.name) ? cfg.tip : `${cfg.name} · ${cfg.tip}`) + (cfg.site ? ` › ${cfg.site}` : '') }), err));
    const row = (label, ...controls) => det.append(el('div', { className: 'field' }, el('span', { textContent: label }), el('div', { className: 'ctl' }, ...controls)));

    const pick = (what) => this.button('◎', `${what} = the selected (double-clicked) body`, () => {
      const b = this.selectedBody(ik);
      if (b < 0) { alert('Double-click a body of this robot first.'); return; }
      cfg[what] = ik.nameOf('mjOBJ_BODY', b);
      if (what === 'tip') cfg.site = '';
      if (what === 'tip' && !ik.ancestors(b).some((a) => ik.nameOf('mjOBJ_BODY', a) === cfg.root)) cfg.root = cfg.tip;
      changed(cfg);
    });
    const bodies = ik.bodyList();
    row('Tip', this.select(bodies.map(([, n]) => [n, n]), cfg.tip, (v) => {
      cfg.tip = v;
      cfg.site = '';
      const anc = ik.ancestors(ik.id('mjOBJ_BODY', v)).map((b) => ik.nameOf('mjOBJ_BODY', b));
      if (!anc.includes(cfg.root)) cfg.root = v;
      changed(cfg);
    }), pick('tip'));
    const tipId = ik.id('mjOBJ_BODY', cfg.tip);
    const anc = tipId >= 0 ? ik.ancestors(tipId).map((b) => ik.nameOf('mjOBJ_BODY', b)) : [];
    row('Root', this.select(anc.map((n) => [n, n]), cfg.root, (v) => { cfg.root = v; changed(cfg); }), pick('root'));
    // The tip frame: a site fixed to the tip body (a TCP), or the body plus an offset.
    const sites = tipId >= 0 ? ik.tipSites(tipId) : [];
    row('TCP site', this.select([['', sites.length ? 'none: body + offset' : 'none on this body'], ...sites.map(([, n]) => [n, n])],
      cfg.site ?? '', (v) => { cfg.site = v; changed(cfg); }));
    if (!cfg.site) {
      const offset = cfg.offset.map((v, k) => {
        const inp = el('input', { type: 'number', step: 0.01, value: v, title: 'xyz'[k] + ' in the tip body frame (m)' });
        inp.onchange = () => { cfg.offset[k] = Number(inp.value) || 0; changed(cfg); };
        return inp;
      });
      row('Offset', ...offset);
    }
    // Gripper: actuators, each with the ctrl that opens and that closes it
    // (ik.js). Edited as a list; Auto finds them in the model again.
    const m = ik.model, mj = this.app.mujoco;
    const actName = (a) => (mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR.value, a) ?? '').slice(ik.prefix.length);
    const grip = () => (chain.gripper ?? []).map(({ a, open, closed }) => ({ actuator: actName(a), open, closed }));
    const setGrip = (list) => { cfg.gripper = list.length ? list : ''; changed(cfg, true); };
    const chainActs = new Set(chain.active?.map((d) => d.act)), have = new Set((chain.gripper ?? []).map((g) => g.a));
    const free = [['', '+ actuator…']];
    for (let a = 0; a < m.nu; a++) {
      const n = mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR.value, a) ?? '';
      if (n.startsWith(ik.prefix) && !chainActs.has(a) && !have.has(a)) free.push([String(a), n.slice(ik.prefix.length)]);
    }
    const add = this.select(free, '', (v) => {
      const a = Number(v), limited = m.actuator_ctrllimited[a];
      const [lo, hi] = limited ? [m.actuator_ctrlrange[2 * a], m.actuator_ctrlrange[2 * a + 1]] : [-1, 1];
      setGrip([...grip(), { actuator: actName(a), open: hi, closed: lo }]);
    });
    row('Gripper', add,
      this.button('Auto', 'Find the gripper actuators below the tip, and which way each closes', () => { cfg.gripper = 'auto'; changed(cfg, true); }));
    if (chain.gripper?.length) {
      // How far closed: 0 open … 1 closed (G or a right-click on the marker toggles it).
      const slider = el('input', { type: 'range', min: 0, max: 1, step: 0.01, title: 'How far the gripper is closed (G, or a right-click on the marker, toggles it)' });
      const label = el('span', { className: 'closure' }), grip = { chain, slider, label, until: 0 };
      slider.oninput = () => {
        grip.until = performance.now() + 600;  // (network mode: until the server's ctrl has caught up)
        ik.setGripper(chain, this.app.data, Number(slider.value));
        label.textContent = `${Math.round(slider.value * 100)}%`;
      };
      row('Closed', slider, label);
      this.grips.push(grip);
    }
    grip().forEach((g, k) => {
      const a = chain.gripper[k].a, span = m.actuator_ctrllimited[a] ? m.actuator_ctrlrange[2 * a + 1] - m.actuator_ctrlrange[2 * a] : 2;
      const step = +(span / 50).toPrecision(2);
      const edit = (field) => this.num(g[field], step, (v) => { const list = grip(); list[k][field] = v; setGrip(list); }, `ctrl that ${field === 'open' ? 'opens' : 'closes'} it`);
      det.append(el('div', { className: 'field grip' },
        el('span', { textContent: g.actuator, title: g.actuator }),
        el('div', { className: 'ctl' }, el('small', { textContent: 'open' }), edit('open'), el('small', { textContent: 'closed' }), edit('closed'),
          this.button('⇄', 'Swap open and closed', () => { const list = grip(); [list[k].open, list[k].closed] = [g.closed, g.open]; setGrip(list); }),
          this.button('✕', 'Not part of the gripper', () => setGrip(grip().filter((_, i) => i !== k))))));
    });

    if (chain.error) det.append(el('div', { className: 'error', textContent: chain.error }));
    const joints = el('div', { className: 'joints' });
    for (const jt of chain.joints) {
      const can = jt.active || !!jt.dof;  // movable, maybe locked
      const box = el('input', { type: 'checkbox', checked: jt.active, disabled: !can });
      box.onchange = () => {
        cfg.locked = box.checked ? cfg.locked.filter((n) => n !== jt.name) : [...cfg.locked, jt.name];
        changed(cfg, true);
      };
      joints.append(el('label', { className: can ? '' : 'off', title: jt.status },
        box, el('span', { textContent: jt.name }), el('span', { className: 'why', textContent: jt.status === 'ok' ? '' : jt.status })));
    }
    det.append(joints);
    det.append(el('div', { className: 'bar' },
      this.button('Reset target', 'Put the marker back on the tip', () => ik.targets.delete(cfg.name)),
      this.button('Rename', 'Rename this chain', () => {
        const n = prompt('Chain name', cfg.name)?.trim();
        if (!n || ik.config.chains.some((c) => c.name === n)) return;
        ik.targets.delete(cfg.name);
        cfg.name = n;
        changed(cfg);
      }),
      this.button('Remove', 'Delete this chain', () => {
        ik.config.chains.splice(i, 1);
        ik.targets.delete(cfg.name);
        ik.focus = 0;
        changed(cfg);
      })));
    return det;
  }

  // Every frame: the active robot gets the input's command, the others stand.
  tick(dt) {
    const robots = this.controllable();
    const locos = locomotion(this.active);
    // In WebXR the left stick drives while pushed (xr.js), and takes over
    // from the driven robot's goal, which is dropped.
    const xrStick = this.input.xr && this.input.stick.some((v) => v);
    if (xrStick && this.active?.goal && !this.dragging?.goal) this.setGoal(this.active, null);
    const keyboard = this.mode === 'keyboard' || xrStick;
    this.input.enabled = locos.length > 0 && keyboard;
    this.input.sticks?.show(this.input.enabled);
    for (const r of robots) {
      const mine = r === this.active ? locos : [];
      // A goal is followed until reached or cleared, also while another robot is
      // driven: here, or on the server (which sends goalResult) in network mode.
      const net = !!this.app.net;
      const goal = r.goal && !(r === this.active && keyboard) && this.app.data ? r.goal : null;
      const pose = goal && !net && bodyPose(this.app.data, r.bodyOffset);
      if (!net) r.goalResult = null;
      const twist = this.ros.get(r.prefix)?.freshTwist();
      for (const c of [r.policy, r.base]) {
        if (!c) continue;
        if (twist && locomotion(r).includes(c)) { c.cmd = [...twist]; continue; }  // ROS cmd_vel first
        if (goal && net && locomotion(r).includes(c)) continue;
        if (goal && locomotion(r).includes(c)) {
          r.goalResult = goalCommand(pose, goal, c.drive, r.goalState ??= {});
          c.cmd = r.goalResult.cmd;
          if (r.goalResult.phase === 'arrived' && !(this.dragging?.goal && this.dragging.robot === r)) this.arrive(r);
          continue;
        }
        if (!mine.includes(c) || !keyboard) { c.cmd = [0, 0, 0]; continue; }
        const { cmd, dh } = this.input.read(dt, c.drive);
        c.cmd = cmd;
        if ('height' in c) c.height = Math.min(HEIGHT_RANGE[1], Math.max(HEIGHT_RANGE[0], c.height + dh));
      }
    }
    if (this.goalReadout) {
      const g = this.active?.goalResult;
      const text = !this.active?.goal ? (this.active?.arrived ? 'arrived' : 'no goal') : !g ? '' : g.phase === 'arrived' ? 'arrived'
        : `${g.dist.toFixed(2)} m · ${(g.dyaw * DEG).toFixed(0)}° · ${g.phase === 'turn' ? 'turning' : 'driving'}`;
      if (this.goalReadout.textContent !== text) this.goalReadout.textContent = text;
    }
    const c = locos[0];
    if (this.readout && c) {
      const f = (v) => v.toFixed(2).padStart(5);
      let text = `cmd  vx ${f(c.cmd[0])}  vy ${f(c.cmd[1])}  ωz ${f(c.cmd[2])}`;
      if ('height' in c) text += `  height ${c.height.toFixed(2)} m`;
      if (c.measured) text += `\nmeas vx ${f(c.measured[0])}  vy ${f(c.measured[1])}  ωz ${f(c.measured[2])}`;
      if (this.readout.textContent !== text) this.readout.textContent = text;
    }
    this.showRos();
    const ik = this.active?.ik;
    this.errs.forEach((span, i) => {
      const e = ik?.enabled && ik.chains[i]?.err;
      const text = e ? `${(e[0] * 1000).toFixed(0)} mm · ${(e[1] * 57.3).toFixed(0)}°` : '';
      if (span.textContent !== text) span.textContent = text;
    });
    // Gripper sliders follow the gripper (G, VR, ROS), except one just moved.
    const data = this.app.data, now = performance.now();
    for (const { chain, slider, label, until } of this.grips) {
      const g = chain.gripper?.[0];
      if (!g || !data || now < until || g.closed === g.open) continue;
      const t = Math.min(1, Math.max(0, (data.ctrl[g.a] - g.open) / (g.closed - g.open)));
      if (Math.abs(Number(slider.value) - t) > 0.005) slider.value = t;
      const text = `${Math.round(t * 100)}%`;
      if (label.textContent !== text) label.textContent = text;
    }
  }
}
