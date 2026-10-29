// The drive layer: who drives each robot (no DOM). Input sources push into
// it, and every frame (App.frame) it writes the body twist command `cmd`
// = [vx, vy, yaw rate] into each robot's switched-on locomotion
// controllers (a walking policy, a mobile base), first source first:
//   cmdVel  ROS cmd_vel (ros/robot.js), while fresher than TWIST_TIMEOUT;
//   keys    the keys, a gamepad, the touch or WebXR sticks of the Teleop
//           panel (teleop.js), pushed every frame they drive (they last
//           that frame): a reader of the controller's key speeds;
//   goal    the robot's goal marker (goal.js), followed until reached or
//           cleared, also while another robot is driven;
// and nothing: a zero command (it stands, or holds its base still).
// Goals are set here for every caller (the Teleop panel's marker,
// window.goals, behaviour trees, the Agent). In network mode a robot's goal
// is the server's (server/goal.py), one for every browser: changes go
// there and come back to everyone (fromServer); the server follows it, and
// the command written here goes to the server as `op cmd` (remote.js).
//
// Joints: per actuator, who writes it. A switched-on policy, base or IK
// keeps the joints it drives (owners); ROS joint commands (jointCommand)
// write the others. With a robot's takeover on, ROS commands win on those
// joints too while they come (fresher than JOINT_TIMEOUT): written over
// what the controllers wrote, before every physics step (override). In
// network mode the server has the same rules (server.py joint_command).

import { bodyPose, goalCommand } from './goal.js';
import { clampCommand } from '../controllers/commands.js';

const TWIST_TIMEOUT = 0.5;  // s: an older cmd_vel counts as none
const GOAL_GAP_MS = 50;     // network mode: a goal being dragged goes to the server at most this often
const GOAL_ECHO_MS = 600;   // and the server's echoes of it are ignored for this long
const JOINT_TIMEOUT = 0.5;  // s: a joint command taking over lasts this long
const OWNER_NAMES = { policy: 'policy', base: 'base', ik: 'IK', ros: 'ROS' };

// The switched-on controllers of robot `r` that take twist commands.
export function locomotion(r) {
  return [r?.use?.policy && r.policy?.drive && r.policy, r?.use?.base && r.base?.drive && r.base].filter(Boolean);
}

export class Drive {
  constructor(app) {
    this.app = app;
    this.sources = new Map();  // robot prefix -> {cmdVel: {twist, at}, keys: {read, dh}}
    this.held = new Set();     // robots whose goal is being dragged: not let go on arriving, no echoes
    this.onClear = [];         // (robots) => …: goals forgotten (Reset)
    this.frame = 0;            // ticks so far (the keys last the frame they are pushed in)
    this.overrides = new Map();  // robot prefix -> {model, ctrl: Map(actuator -> {u, at})}: joint commands taking over
    this.commanded = new Map();  // robot prefix -> {at, acts: Set}: the last joint commands (for the panels)
  }

  // Robot r's input from `source`: 'cmdVel' (a twist [vx, vy, wz]) or
  // 'keys' ({read(drive) -> {cmd, dh} or null for none: only the height,
  // dh: height change}), for this frame.
  push(r, source, value) {
    const s = this.sources.get(r.prefix) ?? {};
    this.sources.set(r.prefix, s);
    s[source] = source === 'cmdVel' ? { twist: value, at: performance.now() } : { ...value, frame: this.frame };
  }

  // The twist from ROS cmd_vel while it is fresh, else null.
  twist(r) {
    const t = this.sources.get(r?.prefix)?.cmdVel;
    return t && (performance.now() - t.at) / 1000 < TWIST_TIMEOUT ? t.twist : null;
  }

  // What drives robot r now: 'cmdVel', 'keys', 'goal' or null.
  source(r) {
    if (!locomotion(r).length) return null;
    if (this.twist(r)) return 'cmdVel';
    const keys = this.sources.get(r.prefix)?.keys;
    if (keys?.read && keys.frame >= this.frame - 1) return 'keys';
    return r.goal ? 'goal' : null;
  }

  // ---- goals ----

  goal(r) { return r?.goal ?? null; }

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

  // A goal being dragged (the Teleop panel's marker): kept while held.
  hold(r, on) {
    if (on) this.held.add(r);
    else this.held.delete(r);
  }

  // Network mode: robot r's goal as the server has it (set by any browser).
  // `arrived`: the goal the robot reached, which the server let go.
  fromServer(r, goal, arrived) {
    if (this.held.has(r)) return;
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
    for (const f of this.onClear) f(robots);
  }

  // ---- joints ----

  // {actuator: 'policy' | 'base' | 'ik'}: the actuators robot r's
  // switched-on controllers write (in network mode, the server's).
  owners(r) {
    const out = new Map(), m = this.app.model;
    if (this.app.net) {
      for (const [a, k] of Object.entries(r?.owners ?? {})) out.set(+a, k);
      return out;
    }
    if (!m || !r?.use) return out;
    const policy = r.use.policy && r.policy?.owns ? r.policy : null, base = r.use.base && r.base?.owns ? r.base : null;
    for (let a = 0; a < m.nu; a++) {
      if (m.actuator_trntype[a] !== 0) continue;  // joint transmissions
      const j = m.actuator_trnid[2 * a];
      if (policy?.owns(j)) out.set(a, 'policy');
      else if (base?.owns(j)) out.set(a, 'base');
    }
    if (r.use.ik && r.ik?.enabled) {
      for (const c of r.ik.chains ?? []) for (const d of c.active ?? []) if (d.act >= 0 && !out.has(d.act)) out.set(d.act, 'ik');
    }
    return out;
  }

  // ROS joint commands for robot r: [[actuator, ctrl], …]. `takeover`: on
  // the joints a controller drives too (else those are left to it).
  jointCommand(r, pairs, takeover = false) {
    const now = performance.now(), { model, data, net } = this.app;
    if (!model || !data || !pairs.length) return;
    const seen = this.commanded.get(r.prefix);
    if (seen?.model === model) { seen.at = now; for (const [a] of pairs) seen.acts.add(a); }
    else this.commanded.set(r.prefix, { model, at: now, acts: new Set(pairs.map(([a]) => a)) });
    if (net) { net.send('joint_cmd', { id: r.netId, ctrl: pairs, takeover: !!takeover }); return; }
    const owners = this.owners(r);
    let o = this.overrides.get(r.prefix);
    if (o?.model !== model) this.overrides.set(r.prefix, o = { model, ctrl: new Map() });
    for (const [a, u] of pairs) {
      if (!owners.has(a)) data.ctrl[a] = u;
      else if (takeover) o.ctrl.set(a, { u, at: now });
    }
  }

  // Local mode, before every physics step (App.advance), after the
  // controllers wrote: joint commands taking over, while fresh.
  override(data) {
    if (!this.overrides.size) return;
    const now = performance.now();
    for (const [prefix, o] of this.overrides) {
      if (o.model !== this.app.model) { this.overrides.delete(prefix); continue; }
      for (const [a, { u, at }] of o.ctrl) {
        if (now - at > JOINT_TIMEOUT * 1000) o.ctrl.delete(a);
        else data.ctrl[a] = u;
      }
    }
  }

  // Who writes robot r's joints: [{owner: 'policy' | 'base' | 'ik' | 'ros' | null, joints: [names]}],
  // in actuator order, neighbours with the same owner together. `ros`: ROS
  // joint commands are on (they write the joints no controller drives);
  // `takeover`: they win on the others too while they come.
  joints(r, ros = false, takeover = false) {
    const { model, mujoco } = this.app;
    if (!model || !r) return [];
    const owners = this.owners(r), name = (t, i) => mujoco.mj_id2name(model, mujoco.mjtObj[t].value, i) ?? '';
    const seen = this.commanded.get(r.prefix);
    const fresh = seen?.model === model && performance.now() - seen.at < JOINT_TIMEOUT * 1000;
    const out = [];
    for (let a = 0; a < model.nu; a++) {
      const act = name('mjOBJ_ACTUATOR', a);
      if (!act.startsWith(r.prefix)) continue;
      const joint = model.actuator_trntype[a] === 0 ? name('mjOBJ_JOINT', model.actuator_trnid[2 * a]) : act;
      let owner = owners.get(a) ?? (ros ? 'ros' : null);
      if (owner !== 'ros' && owner && ros && takeover && fresh && seen.acts.has(a)) owner = 'ros';
      const n = joint.slice(r.prefix.length);
      if (out.at(-1)?.owner === owner) out.at(-1).joints.push(n);
      else out.push({ owner, joints: [n] });
    }
    return out;
  }

  // joints() as a line: "fr3_joint1–7: IK · fr3_finger_joint1: ROS"; the
  // joints nothing writes left out.
  jointsText(r, ros, takeover) {
    return this.joints(r, ros, takeover).filter((g) => g.owner)
      .map((g) => `${shortList(g.joints)}: ${OWNER_NAMES[g.owner]}`).join(' · ');
  }

  // ---- every frame ----

  tick() {
    const net = !!this.app.net, data = this.app.data;
    for (const r of this.app.robots ?? []) {
      if (!r.use) continue;
      const src = this.sources.get(r.prefix), twist = this.twist(r);
      const keys = src?.keys?.frame === this.frame ? src.keys : null;
      const locos = locomotion(r);
      // A goal is followed until reached or cleared: here, or on the server
      // (which sends goalResult) in network mode.
      const goal = r.goal && !keys?.read && data ? r.goal : null;
      const pose = goal && !net && bodyPose(data, r.bodyOffset);
      if (!net) r.goalResult = null;
      for (const c of [r.policy, r.base]) {
        if (!c) continue;
        const loco = locos.includes(c);
        if (twist && loco) c.cmd = [...twist];  // ROS cmd_vel first
        else if (keys?.read && loco) c.cmd = keys.read(c.drive);
        else if (goal && loco && net) { /* the server follows it */ }
        else if (goal && loco) {
          r.goalResult = goalCommand(pose, goal, c.drive, r.goalState ??= {});
          c.cmd = r.goalResult.cmd;
          if (r.goalResult.phase === 'arrived' && !this.held.has(r)) this.arrive(r);
        } else c.cmd = [0, 0, 0];
        if (loco && keys?.dh && c.commands?.height) c.height = clampCommand(c.commands.height, c.height + keys.dh);  // R/F, the pad's height
      }
    }
    this.frame++;
  }
}

// Names with a running number in a row shortened (fr3_joint1 … fr3_joint7
// -> fr3_joint1–7), and a long list to its ends and count.
function shortList(names) {
  const out = [];
  for (const n of names) {
    const m = /^(.*?)(\d+)$/.exec(n), last = out.at(-1);
    if (m && last && last.stem === m[1] && +m[2] === last.to + 1) { last.to = +m[2]; continue; }
    out.push(m ? { stem: m[1], from: +m[2], to: +m[2] } : { name: n });
  }
  const list = out.map((x) => x.name ?? (x.from === x.to ? `${x.stem}${x.from}` : `${x.stem}${x.from}–${x.to}`));
  return list.length > 3 ? `${list[0]} … ${list.at(-1)} (${names.length})` : list.join(', ');
}
