// A real robot following this one (RealLink, rr.real): the lobby side of
// real_robot/ (python -m real_robot), on the robot's own bridge and prefix.
// The robot's joint states (robot.js) are the real robot's targets; from
// the bridge come, under the prefix:
//   real/joint_states  sensor_msgs/JointState: the real robot's joints, in
//                      the lobby's names, shown as a see-through ghost on
//                      the robot (App.makeGhost): blue while disabled, green
//                      while it follows, red in a fault;
//   real/status        std_msgs/String, JSON {robot, state (disabled |
//                      enabled | fault), reason, stale, error, clients};
// and to it goes real/enable (std_msgs/Bool).
//
// Enable first syncs: the robot (its joints, their actuators' targets, the
// IK's targets) is put where the real one is, so the real robot starts from
// where it stands; the bridge refuses a target too far from it. Sync needs
// the physics in this page (local mode, as in the Quest's browser).
//
// For the robot files with a `real` entry in their ROS profile
// (json/ros_profiles.json): {robot: the bridge's --robot name}.

import { every } from '../app/ticker.js';

const STALE_MS = 1500;       // no status this long: no bridge
const ENABLE_DELAY_MS = 150; // after a sync, before enabling: the synced joint states go out first
const COLORS = {
  disabled: [0.35, 0.65, 1, 0.3],
  enabled: [0.3, 1, 0.45, 0.35],
  fault: [1, 0.3, 0.25, 0.4],
  none: [0.7, 0.7, 0.7, 0.25],
};
const TYPES = { states: 'sensor_msgs/JointState', status: 'std_msgs/String', enable: 'std_msgs/Bool' };

export class RealLink {
  constructor(rr) {
    this.rr = rr;
    this.app = rr.app;
    this.topics = null;      // {states, status, enable} subscribed / advertised
    this.unsub = [];
    this.joints = null;      // Map: joint name -> position (the real robot's)
    this.realAt = 0;         // performance.now() of the last joint states
    this.status = null;      // the bridge's last status
    this.statusAt = 0;
    this.note = '';          // the last refusal here (Sync)
    this.ghostOn = true;
    this.ghost = null;
    this.timer = null;
  }

  get available() { return !!this.rr.profile.real; }

  // The bridge is there (its status comes).
  get live() { return !!this.status && performance.now() - this.statusAt < STALE_MS; }
  get fresh() { return !!this.joints && performance.now() - this.realAt < STALE_MS; }

  // none (no bridge) | disabled | enabled | fault
  get state() { return this.live ? this.status.state : 'none'; }

  // Subscribed and advertised as the robot's bridge and prefix are (RosRobot.sync).
  sync() {
    const rr = this.rr, b = rr.bridge;
    if (!this.available || !b) { this.detach(); return; }
    const name = (t) => rr.topicName(`real/${t}`, rr.synced);
    const want = { states: name('joint_states'), status: name('status'), enable: name('enable') };
    if (this.topics && this.bridge === b && Object.keys(want).every((k) => want[k] === this.topics[k])) return;
    this.detach();
    this.bridge = b;
    this.topics = want;
    this.unsub = [
      b.subscribe(want.states, TYPES.states, (m) => this.onStates(m)),
      b.subscribe(want.status, TYPES.status, (m) => this.onStatus(m)),
    ];
    b.advertise(want.enable, TYPES.enable, this);
    this.timer = every(500, () => this.draw());  // (hides a ghost whose states stopped)
  }

  detach() {
    for (const off of this.unsub) off();
    this.unsub = [];
    if (this.topics) this.bridge?.unadvertise(this.topics.enable, this);
    this.topics = null;
    this.bridge = null;
    this.timer?.();
    this.timer = null;
    this.status = null;
    this.joints = null;
    this.dropGhost();
  }

  onStates({ name = [], position = [] }) {
    const m = new Map();
    name.forEach((n, i) => { if (Number.isFinite(position[i])) m.set(n, position[i]); });
    this.joints = m;
    this.realAt = performance.now();
    this.draw();
  }

  onStatus({ data }) {
    try {
      const s = JSON.parse(data);
      const was = this.state;
      this.status = s;
      this.statusAt = performance.now();
      if (this.state !== was) this.rr.world.changed();
    } catch { /* not ours */ }
  }

  // ---- enable / disable ----

  send(on) {
    if (!this.topics || !this.bridge?.connected) return false;
    this.bridge.publish(this.topics.enable, { data: on });
    return true;
  }

  // Sync, then enable; '' or why not.
  enable() {
    const why = this.syncToReal();
    this.note = why;
    if (why) return why;
    setTimeout(() => this.send(true), ENABLE_DELAY_MS);
    return '';
  }

  disable() {
    this.note = '';
    return this.send(false);
  }

  // The robot where the real one is: its joints, the targets of their
  // actuators (and the profile's commands: the fingers' tendon), at rest;
  // the IK's targets start again on its tips. '' or why not.
  syncToReal() {
    const app = this.app, { mujoco, model, data } = app;
    if (app.net) return 'Sync needs the physics in this page (local mode)';
    if (!model || !data) return 'no world';
    if (!this.fresh) return 'no state from the real robot (is python -m real_robot running?)';
    if (this.state === 'enabled') return '';
    const t = this.rr.tables();
    for (const [name, v] of this.joints) {
      const jt = t.byName.get(name);
      if (!jt) continue;
      data.qpos[jt.q] = v;
      data.qvel[jt.v] = 0;
      const drives = t.commands.get(name) ?? (t.acts.get(jt.j) ?? []).map((a) => ({ a, scale: 1, offset: 0 }));
      for (const { a, scale, offset } of drives) {
        let u = v * scale + offset;
        if (model.actuator_ctrllimited[a]) u = Math.min(model.actuator_ctrlrange[2 * a + 1], Math.max(model.actuator_ctrlrange[2 * a], u));
        data.ctrl[a] = u;
      }
    }
    mujoco.mj_forward(model, data);
    this.rr.robot.ik?.reset();
    app.syncTransforms();
    return '';
  }

  // ---- the ghost ----

  setGhost(on) {
    this.ghostOn = on;
    this.draw();
  }

  draw() {
    const app = this.app, r = this.rr.robot;
    if (!this.ghostOn || !this.fresh || !r?.visual || !app.model) {
      if (this.ghost) app.poseGhost(this.ghost, null);
      return;
    }
    if (this.ghost && this.ghost.robot !== r) this.dropGhost();  // a new world: new robot objects
    this.ghost ??= app.makeGhost(r);
    const color = COLORS[this.state];
    if (this.ghost.color !== color) { app.ghostColor(this.ghost, color); this.ghost.color = color; }
    const t = this.rr.tables(), qpos = [];
    for (const [name, v] of this.joints) {
      const jt = t.byName.get(name);
      if (jt) qpos.push([jt.q, v]);
    }
    app.poseGhost(this.ghost, qpos);
  }

  dropGhost() {
    if (this.ghost) this.app.freeGhost(this.ghost);
    this.ghost = null;
  }

  // One line for the panels: the bridge's state and why.
  text() {
    if (!this.live) return this.rr.on ? 'no real_robot bridge on this address' : 'switch the robot\'s ROS on';
    const s = this.status;
    const parts = [`${s.robot}: ${s.state}`];
    if (s.state === 'enabled') parts.push(s.stale ? 'holding (no targets)' : `following, ${s.error?.toFixed(3) ?? '–'} rad behind`);
    if (s.reason && s.state !== 'enabled') parts.push(s.reason);
    return parts.join(' · ');
  }
}
