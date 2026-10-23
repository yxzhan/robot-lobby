// Controllers in network mode: the server runs them (server/ik.py,
// server/base.py, server/policies.py: ports of ik.js, base.js, groot.js);
// these stand in for IkController, BaseController and a policy controller
// towards the Teleop panel and the markers, with the fields and methods
// those use.
//
// The server describes a robot's controllers (`desc`) whenever they change,
// and sends IK targets and errors and the base's measured twist a few times
// a second (status).
//
// IK: targets are kept in the frame of the chain's reference body, so a
// marker rides along with the robot as drawn here. Dragging a marker moves
// it here at once and sends the new target; the server's echo of that chain
// is ignored for a moment (ECHO_MS) so a status sent before the drag arrived
// does not pull it back.
//
// Base and policy: the panel writes `cmd` (and a policy's `height`) every
// frame; it goes to the server when it changes, and is repeated while not
// zero (REPEAT_MS) since the server stops a command that is not (a closed or
// frozen tab must not leave a robot walking). A policy's other commands
// (commands.js) are settings: one `command` request per change, everyone's
// values coming back with the status (also ignored for ECHO_MS after a
// change here).

const ECHO_MS = 600;
const REPEAT_MS = 250;
const MIN_GAP_MS = 50;  // a moving command changing every frame (the goal input) goes out at most this often

const mulMV = (R, v) => [0, 3, 6].map((r) => R[r] * v[0] + R[r + 1] * v[1] + R[r + 2] * v[2]);
const mulTV = (R, v) => [0, 1, 2].map((c) => R[c] * v[0] + R[3 + c] * v[1] + R[6 + c] * v[2]);
const mulMM = (A, B) => [0, 3, 6].flatMap((r) => [0, 1, 2].map((c) => A[r] * B[c] + A[r + 1] * B[3 + c] + A[r + 2] * B[6 + c]));
const mulTM = (A, B) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[r] * B[c] + A[3 + r] * B[3 + c] + A[6 + r] * B[6 + c]));

export class RemoteIk {
  constructor(app, robot, desc) {
    this.app = app;
    this.robot = robot;
    this.key = robot.key;
    this.prefix = robot.prefix;
    this.name = 'IK';
    this.focus = 0;
    this.remote = new Map();  // chain name -> {pos, rot} in its ref body frame
    this.setAt = new Map();   // chain name -> when it was last dragged here
    const send = (op, args = {}) => app.net.send(op, { id: robot.netId, ...args });
    this.send = send;
    // The panel deletes a target to put its marker back on the tip.
    this.targets = {
      delete: (name) => { this.remote.delete(name); this.setAt.delete(name); send('ik_reset_target', { chain: name }); },
      clear: () => { this.remote.clear(); this.setAt.clear(); send('ik_reset_target', { chain: null }); },
    };
    this.update(desc);
  }

  // The world's tables (net.js worldModel): actuator names for the panel.
  get model() { return this.app.model; }

  update(desc) {
    if (!desc) { this.chains = []; this.config = { chains: [] }; this.enabled = false; return; }
    this.config = desc.config;
    this.saved = desc.saved;
    this.hasDefaults = desc.hasDefaults;
    this.enabled = desc.enabled;
    this.bodies = desc.bodies;
    this.sites = desc.tipSites;
    const old = new Map((this.chains ?? []).map((c) => [c.cfg.name, c]));
    // Each chain's cfg is the entry of this.config it was resolved from (one
    // object, as in ik.js): the panel edits it, resolve() sends this.config.
    // (The server resolves config.chains in order, one chain each.)
    this.chains = desc.chains.map((c, i) => ({ ...c, cfg: this.config.chains[i] ?? c.cfg, err: old.get(c.cfg.name)?.err ?? null }));
    // A chain whose reference body changed (a joint locked or unlocked): its
    // target keeps its place in the world, like on the server.
    for (const c of this.chains) {
      const was = old.get(c.cfg.name), t = this.remote.get(c.cfg.name);
      if (!was || !t || was.ref === c.ref || was.ref < 0 || c.ref < 0 || !this.app.data) continue;
      const from = this.refFrame(was, this.app.data), to = this.refFrame(c, this.app.data);
      const pos = mulMV(from.rot, t.pos).map((v, i) => v + from.pos[i]);
      this.remote.set(c.cfg.name, { pos: mulTV(to.rot, pos.map((v, i) => v - to.pos[i])), rot: mulTM(to.rot, mulMM(from.rot, t.rot)) });
    }
    if (this.focus >= this.chains.length) this.focus = 0;
  }

  // From the server's status (5 Hz): targets and tracking errors.
  status({ targets, err }) {
    const now = performance.now();
    for (const [name, t] of Object.entries(targets)) {
      if (now - (this.setAt.get(name) ?? -Infinity) > ECHO_MS) this.remote.set(name, t);
    }
    for (const c of this.chains) c.err = err[c.cfg.name] ?? null;
  }

  refFrame(chain, data) {
    const b = chain.ref;
    return { pos: Array.from(data.xpos.subarray(3 * b, 3 * b + 3)), rot: Array.from(data.xmat.subarray(9 * b, 9 * b + 9)) };
  }

  // Target pose {pos, rot} in the world (for the markers), or null until the
  // server has one.
  targetPose(chain, data) {
    const t = !chain.error && chain.ref >= 0 && this.remote.get(chain.cfg.name);
    if (!t) return null;
    const ref = this.refFrame(chain, data);
    return { pos: mulMV(ref.rot, t.pos).map((v, i) => v + ref.pos[i]), rot: mulMM(ref.rot, t.rot) };
  }

  setTargetPose(chain, data, { pos, rot }) {
    const ref = this.refFrame(chain, data);
    const t = { pos: mulTV(ref.rot, pos.map((v, i) => v - ref.pos[i])), rot: mulTM(ref.rot, rot) };
    this.remote.set(chain.cfg.name, t);
    this.setAt.set(chain.cfg.name, performance.now());
    this.send('ik_target', { chain: chain.cfg.name, pos: t.pos, rot: t.rot });
  }

  toggleGripper(chain) { this.send('ik_gripper', { chain: chain.cfg.name }); }
  setGripper(chain, data, t) { this.send('ik_gripper', { chain: chain.cfg.name, amount: t }); }

  // After the panel edited this.config: the server resolves it and sends
  // the new chains (main.js then renders the panel again).
  resolve() { this.send('ik_config', { config: this.config }); }
  save() { this.send('ik_save'); }
  restoreDefaults() { this.remote.clear(); this.send('ik_defaults'); }
  release() {}
  reset() {}

  // ---- the robot's bodies and sites (for the chain editor) ----

  id(type, name) {
    const names = this.model.names[this.app.mujoco.mjtObj[type].value] ?? [];
    return names.indexOf(this.prefix + name);
  }

  nameOf(type, i) {
    const n = this.app.mujoco.mj_id2name(this.model, this.app.mujoco.mjtObj[type].value, i) ?? '';
    return n.startsWith(this.prefix) ? n.slice(this.prefix.length) : n;
  }

  bodyList() {
    const out = [];
    for (let b = this.bodies[0]; b < this.bodies[1]; b++) out.push([b, this.nameOf('mjOBJ_BODY', b)]);
    return out;
  }

  ancestors(tip) {
    const out = [];
    for (let b = tip; b >= this.bodies[0] && b < this.bodies[1]; b = this.model.body_parentid[b]) out.push(b);
    return out;
  }

  tipSites(tip) { return this.sites[tip] ?? []; }
}

// The drive command of a base or a policy, sent when it changes.
class RemoteDriver {
  constructor(app, robot, kind) {
    this.app = app;
    this.robot = robot;
    this.kind = kind;
    this.cmdNow = [0, 0, 0];
    this.sent = null;
    this.sentAt = 0;
    this.measured = null;
  }

  get cmd() { return this.cmdNow; }
  set cmd(v) { this.cmdNow = v; this.push(); }

  push() {
    const h = this.heightNow;
    const key = this.cmdNow.map((v) => v.toFixed(3)).join() + (h == null ? '' : `/${h.toFixed(3)}`);
    const moving = this.cmdNow.some((v) => Math.abs(v) > 1e-6);
    const now = performance.now();
    if (key === this.sent && !(moving && now - this.sentAt > REPEAT_MS)) return;
    if (moving && this.sent != null && now - this.sentAt < MIN_GAP_MS) return;  // a stop always goes at once
    if (this.sent == null && !moving && h == null) { this.sent = key; return; }  // nothing to say yet
    this.sent = key;
    this.sentAt = now;
    this.app.net.send('cmd', { id: this.robot.netId, kind: this.kind, cmd: this.cmdNow, height: h });
  }
}

// A trained policy (GrootController and the like) on the server.
export class RemotePolicy extends RemoteDriver {
  constructor(app, robot, desc, policyId) {
    super(app, robot, 'policy');
    this.policyId = policyId;
    this.name = desc.name;
    this.drive = desc.drive;
    this.commands = desc.commands ?? {};
    this.command = structuredClone(desc.command ?? {});
    this.changedAt = {};  // command name -> when it was set here
    if ('height' in this.commands) {  // a commanded base height: also the panel's R/F keys, sent with cmd
      Object.defineProperty(this, 'height', {
        get: () => this.heightNow,
        set: (v) => { this.command.height = v; this.changedAt.height = performance.now(); this.push(); },
        enumerable: true,
      });
    }
  }

  get heightNow() { return this.command.height; }

  update(desc) {
    this.name = desc.name;
    this.drive = desc.drive;
    this.status(desc.command);
  }

  // commands.js setCommand: the value is already a valid one.
  setCommand(name, value) {
    this.command[name] = value;
    this.changedAt[name] = performance.now();
    this.app.net.send('command', { id: this.robot.netId, name, value });
  }

  // The server's values (status, describe), but those just changed here.
  status(command) {
    const now = performance.now();
    for (const [k, v] of Object.entries(command ?? {})) {
      if (!(now - (this.changedAt[k] ?? -Infinity) < ECHO_MS)) this.command[k] = v;
    }
  }
}

// The mobile base (BaseController) on the server.
export class RemoteBase extends RemoteDriver {
  constructor(app, robot, desc) {
    super(app, robot, 'base');
    this.key = robot.key;
    this.prefix = robot.prefix;
    this.update(desc);
  }

  update(desc) {
    this.desc = desc;
    this.config = desc?.config ?? null;
    this.saved = !!desc?.saved;
    this.hasDefaults = !!desc?.hasDefaults;
    this.error = desc?.error ?? (desc ? null : 'not set up');
    this.wheels = desc?.wheels ?? [];
    this.name = desc?.name ?? 'Base';
    this.bodies = desc?.bodies ?? [0, 0];
  }

  // Keyboard speeds; null while the base is not set up.
  get drive() { return this.desc?.ready ? this.config.drive : null; }

  // The world's tables (net.js worldModel): actuators and joints for the panel.
  get model() { return this.app.model; }

  autoConfig() { return structuredClone(this.desc?.auto ?? { kind: '', auto: true }); }
  resolve() { this.app.net.send('base_config', { id: this.robot.netId, config: this.config }); }
  reset() {}
  save() { this.app.net.send('base_save', { id: this.robot.netId }); }
  restoreDefaults() { this.app.net.send('base_defaults', { id: this.robot.netId }); }
  release() {}
  inRobot(b) { return b >= this.bodies[0] && b < this.bodies[1]; }

  nameOf(type, i) {
    const n = this.app.mujoco.mj_id2name(this.model, this.app.mujoco.mjtObj[type].value, i) ?? '';
    return n.startsWith(this.prefix) ? n.slice(this.prefix.length) : n;
  }
}
