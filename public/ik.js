// Teleoperation by inverse kinematics: each chain moves a tip frame (a site,
// or a body plus an offset) to a target pose by damped-least-squares differential
// IK on the joints between a root body and the tip, writing the targets of
// their position actuators.
//
// A chain is just {root, tip}: its joints are found by walking body_parentid
// from the tip up to the root (the root's own joints included), so moving the
// root to the elbow or down to the lift changes which joints take part.
// Joints that cannot (no position actuator, owned by another controller,
// locked by the user) keep their place in the chain but are not moved. A
// joint that follows another through a joint equality (Garmi's two lift
// stages) adds its motion to the leader's Jacobian column.
//
// The target lives in the frame of the parent of the first body the chain
// can move (usually the root's parent), so it travels with the robot (and
// the lift, when the chain starts above it). IK runs on
// the *commanded* joint positions (forward kinematics in a scratch MjData),
// not the measured ones: sag under gravity does not wind the targets up.
//
// Configs, first found: one saved in localStorage from the Teleop panel,
// IK_DEFAULTS per robot file (keyed like base.js's BASE_DEFAULTS), or one
// made from the model (autoConfig): every serial run of at least MIN_CHAIN
// free position-servo joints that ends a branch of the tree (an arm, a leg,
// a neck with enough joints), gripper/finger joints left out and their
// actuator taken as the chain's gripper, and the tip put on the model's
// TCP site when it has one (TCP_SITES).

import { CONTROL_DT } from './groot.js';

export const IK_DEFAULTS = {
  'digital_twin/garmi/garmi.xml': {
    chains: [
      // Franka TCP: 0.1034 m past the hand flange, between the fingertips.
      { name: 'left', root: 'left_fr3_link1', tip: 'left_fr3_hand', offset: [0, 0, 0.1034], locked: [], gripper: 'left_gripper' },
      { name: 'right', root: 'right_fr3_link1', tip: 'right_fr3_hand', offset: [0, 0, 0.1034], locked: [], gripper: 'right_gripper' },
    ],
  },
  // panda.xml has no TCP site (only mjx_panda.xml's "gripper", 0.1 m out).
  'mujoco_menagerie/franka_emika_panda/panda.xml': {
    chains: [{ name: 'hand', root: 'link1', tip: 'hand', offset: [0, 0, 0.1034], locked: [], gripper: 'actuator8' }],
  },
};

const STORE = 'mujoco-filament/ik/';
const MIN_CHAIN = 3;  // fewest joints for an automatic chain (a 2-joint head is left alone)
// Joints of hands and grippers: not part of automatic chains.
const GRIPPER_JOINT = /finger|gripper|jaw|thumb|knuckle|claw|hand_|_hand|palm/i;
// Site names taken as a chain's TCP, best first (menagerie uses all of these:
// i2rt tcp_site, Kinova pinch_site, Robotiq/Trossen pinch, Shadow grasp_site,
// ALOHA gripper, SO-101 gripperframe, Lite6 end_effector, Talos palm, and the
// attachment_site flange of arms without a hand).
const TCP_SITES = [/tcp/i, /pinch/i, /grasp/i, /gripper|end_?effector|eef|(^|[_/])ee(_|$)/i, /hand_center|palm/i, /attachment/i];
const LAMBDA = 0.05;                   // DLS damping
const MAX_STEP = [0.03, 0.15];         // per tick: position error (m), rotation error (rad)
const MAX_SPEED = { 2: 0.15, 3: 2.0 }; // joint speed by type: slide (m/s), hinge (rad/s)
const NULL_GAIN = 0.5;                 // pull towards the home posture (1/s), in the null space
const ROT_WEIGHT = 0.3;                // m per rad: orientation error against position error

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const mulMV = (R, v) => [0, 3, 6].map((r) => R[r] * v[0] + R[r + 1] * v[1] + R[r + 2] * v[2]);
const mulTV = (R, v) => [0, 1, 2].map((c) => R[c] * v[0] + R[3 + c] * v[1] + R[6 + c] * v[2]);
const mulMM = (A, B) => [0, 3, 6].flatMap((r) => [0, 1, 2].map((c) => A[r] * B[c] + A[r + 1] * B[3 + c] + A[r + 2] * B[6 + c]));
const mulTM = (A, B) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[r] * B[c] + A[3 + r] * B[3 + c] + A[6 + r] * B[6 + c]));
const mulMT = (A, B) => [0, 3, 6].flatMap((r) => [0, 3, 6].map((c) => A[r] * B[c] + A[r + 1] * B[c + 1] + A[r + 2] * B[c + 2]));

// Rotation vector (axis * angle) of a row-major rotation matrix.
function rotVec(R) {
  const v = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
  const s = Math.hypot(...v) / 2, c = (R[0] + R[4] + R[8] - 1) / 2;
  const angle = Math.atan2(s, c);
  if (s < 1e-9) return [0, 0, 0];  // (near) identity; a half turn has no stable axis here
  return v.map((k) => (k / (2 * s)) * angle);
}

// Solves the small symmetric positive definite system A x = b in place.
function solve(A, b, n) {
  for (let i = 0; i < n; i++) {
    for (let k = i + 1; k < n; k++) {
      const f = A[k * n + i] / A[i * n + i];
      for (let c = i; c < n; c++) A[k * n + c] -= f * A[i * n + c];
      b[k] -= f * b[i];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let c = i + 1; c < n; c++) s -= A[i * n + c] * b[c];
    b[i] = s / A[i * n + i];
  }
  return b;
}

// The saved config for robot file `key`, else its IK_DEFAULTS entry, else
// null (bind then makes one from the model).
export function loadIkConfig(key) {
  try {
    const saved = localStorage.getItem(STORE + key);
    if (saved) return { config: JSON.parse(saved), saved: true };
  } catch { /* no storage */ }
  return { config: IK_DEFAULTS[key] ? structuredClone(IK_DEFAULTS[key]) : null, saved: false };
}

export class IkController {
  // `key` names the robot file (IK_DEFAULTS, saved configs); `prefix` is
  // the robot's name prefix in the world.
  constructor(key, prefix = '') {
    this.key = key;
    this.prefix = prefix;
    this.name = 'IK';
    this.focus = 0;      // chain the keys (gripper) act on: the last one dragged
    ({ config: this.config, saved: this.saved } = loadIkConfig(key));
    this.enabled = true; // holds every chain where it is until its marker moves
    this.chains = [];
    this.targets = new Map();  // chain name -> {pos, rot} in the root parent's frame
  }

  get hasDefaults() { return !!IK_DEFAULTS[this.key]; }

  save() {
    try { localStorage.setItem(STORE + this.key, JSON.stringify(this.config)); this.saved = true; } catch { /* no storage */ }
  }

  restoreDefaults() {
    try { localStorage.removeItem(STORE + this.key); } catch { /* no storage */ }
    this.saved = false;
    this.config = IK_DEFAULTS[this.key] ? structuredClone(IK_DEFAULTS[this.key]) : this.autoConfig();
    this.targets.clear();
    this.resolve();
  }

  // Next tick, every target restarts where its tip is (after a reset).
  reset() {
    this.targets.clear();
  }

  pose() {}

  // `owner(j)`: the name of another controller driving joint j, or null.
  // `bodies`: [first, end) world body ids of this robot.
  bind(mujoco, model, owner = () => null, bodies = [1, model.nbody]) {
    this.scratch?.delete();
    this.mujoco = mujoco;
    this.model = model;
    this.owner = owner;
    this.bodies = bodies;
    this.scratch = new mujoco.MjData(model);
    this.config ??= this.autoConfig();
    this.resolve();
  }

  // Position servos on a single joint (trntype 0, a -kp bias): joint -> actuator.
  servos() {
    const { model } = this, servo = new Map();
    for (let a = 0; a < model.nu; a++) {
      if (model.actuator_trntype[a] !== 0 || model.actuator_biasprm[10 * a + 1] >= 0) continue;
      servo.set(model.actuator_trnid[2 * a], a);
    }
    return servo;
  }

  // Chains found in the model (see the top of the file).
  autoConfig() {
    const { model } = this, [b0, b1] = this.bodies, servo = this.servos();
    const inRobot = (b) => b >= b0 && b < b1;
    const own = new Map(), children = new Map();
    for (let b = b0; b < b1; b++) {
      let n = 0;
      // The robot's own root body (its base, free or on planar joints) stays out.
      for (let j = model.body_jntadr[b], e = b === b0 ? j : j + model.body_jntnum[b]; j >= 0 && j < e; j++) {
        const type = model.jnt_type[j];
        if ((type === 2 || type === 3) && servo.has(j) && !this.owner(j) && !GRIPPER_JOINT.test(this.nameOf('mjOBJ_JOINT', j))) n++;
      }
      own.set(b, n);
      children.set(b, []);
      const p = model.body_parentid[b];
      if (inRobot(p)) children.get(p).push(b);
    }
    // Movable joints in each subtree (children have higher ids than their parent).
    const sub = new Map();
    const count = () => {
      for (let b = b1 - 1; b >= b0; b--) sub.set(b, own.get(b) + children.get(b).reduce((k, c) => k + sub.get(c), 0));
    };
    // The run of movable joints from a branch's last one up to where the
    // tree branches: {top (root body), n}.
    const run = (leaf) => {
      let top = leaf, n = 0;
      for (let b = leaf; inRobot(b); b = model.body_parentid[b]) {
        n += own.get(b);
        if (own.get(b)) top = b;
        const p = model.body_parentid[b];
        if (!inRobot(p) || children.get(p).filter((c) => sub.get(c) > 0).length > 1) break;
      }
      return { top, n };
    };
    const leaves = () => [...own.keys()].filter((b) => own.get(b) > 0 && sub.get(b) === own.get(b));
    // Short end runs (fingers, a pan-tilt head) are dropped, which may make
    // what they hung from an end run in turn.
    for (let changed = true; changed;) {
      changed = false;
      count();
      for (const leaf of leaves()) {
        const { top, n } = run(leaf);
        if (n >= MIN_CHAIN) continue;
        for (let b = leaf; ; b = model.body_parentid[b]) { own.set(b, 0); if (b === top) break; }
        changed = true;
      }
    }
    const used = new Set();
    const chains = leaves().map((leaf) => {
      // The tip goes on through fixed single children (the flange to a hand).
      let tip = leaf;
      for (let k = children.get(tip); k.length === 1 && model.body_jntnum[k[0]] === 0; k = children.get(tip)) tip = k[0];
      let name = this.nameOf('mjOBJ_BODY', tip).replace(/_(link|body)$/, '') || `chain${used.size + 1}`;
      while (used.has(name)) name += '_';
      used.add(name);
      const sites = this.tipSites(tip);
      const tcp = TCP_SITES.map((re) => sites.find(([, n]) => re.test(n))).find(Boolean);
      return {
        name, tip: this.nameOf('mjOBJ_BODY', tip), root: this.nameOf('mjOBJ_BODY', run(leaf).top),
        site: tcp?.[1] ?? '', offset: [0, 0, 0], locked: [], gripper: this.gripperFor(tip, servo),
      };
    });
    return { chains, auto: true };
  }

  // The one actuator driving joints below `tip` that no chain uses (a
  // gripper, a hand's grasp), by name; '' if none or several.
  gripperFor(tip, servo) {
    const { model } = this, below = (j) => this.ancestors(model.jnt_bodyid[j]).includes(tip);
    const found = [];
    for (let a = 0; a < model.nu; a++) {
      if (!this.nameOf('mjOBJ_ACTUATOR', a) || !this.mujoco.mj_id2name(model, this.mujoco.mjtObj.mjOBJ_ACTUATOR.value, a)?.startsWith(this.prefix)) continue;
      const id = model.actuator_trnid[2 * a];
      let joints = [];
      if (model.actuator_trntype[a] === 0) joints = [id];
      else if (model.actuator_trntype[a] === 3 && model.tendon_adr) {  // tendon: its joint wraps
        for (let w = model.tendon_adr[id], e = w + model.tendon_num[id]; w < e; w++) if (model.wrap_type[w] === 1) joints.push(model.wrap_objid[w]);
      }
      if (joints.length && joints.every(below) && !(joints.length === 1 && !GRIPPER_JOINT.test(this.nameOf('mjOBJ_JOINT', joints[0])) && servo.has(joints[0]))) {
        found.push(this.nameOf('mjOBJ_ACTUATOR', a));
      }
    }
    return found.length === 1 ? found[0] : '';
  }

  id(type, name) {
    return this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj[type].value, this.prefix + name);
  }

  nameOf(type, i) {
    const n = this.mujoco.mj_id2name(this.model, this.mujoco.mjtObj[type].value, i) ?? '';
    return n.startsWith(this.prefix) ? n.slice(this.prefix.length) : n;
  }

  // Sites rigidly attached to body `tip`, as [id, name]: on it, on bodies
  // below it with no joint in between, or on bodies above it up to the
  // first one with a joint (ALOHA's gripper site sits on the wrist link the
  // gripper base hangs from).
  tipSites(tip) {
    const { model } = this, inRobot = (b) => b >= this.bodies[0] && b < this.bodies[1];
    let top = tip;
    while (model.body_jntnum[top] === 0 && inRobot(model.body_parentid[top])) top = model.body_parentid[top];
    const rigid = (b) => {  // b is fixed to tip
      if (this.ancestors(tip).slice(0, this.ancestors(tip).indexOf(top) + 1).includes(b)) return true;
      for (; inRobot(b); b = model.body_parentid[b]) {
        if (b === tip) return true;
        if (model.body_jntnum[b] > 0) return false;
      }
      return false;
    };
    const out = [];
    for (let i = 0; i < model.nsite; i++) if (rigid(model.site_bodyid[i])) out.push([i, this.nameOf('mjOBJ_SITE', i)]);
    return out.filter(([, n]) => n);
  }

  // This robot's bodies as [id, name], in tree order.
  bodyList() {
    const out = [];
    for (let b = this.bodies[0]; b < this.bodies[1]; b++) out.push([b, this.nameOf('mjOBJ_BODY', b)]);
    return out;
  }

  // The tip's ancestors in this robot (the tip itself first, then up).
  ancestors(tip) {
    const out = [];
    for (let b = tip; b >= this.bodies[0] && b < this.bodies[1]; b = this.model.body_parentid[b]) out.push(b);
    return out;
  }

  // Rebuilds this.chains from this.config (after bind or an edit).
  resolve() {
    const { model } = this;
    if (!model) return;
    const servo = this.servos();
    // Joint equalities: follower -> [leader, coefficient] (q_f = c0 + c1 q_l + …).
    const follows = new Map();
    for (let e = 0; e < model.neq; e++) {
      if (model.eq_type[e] !== this.mujoco.mjtEq.mjEQ_JOINT.value || model.eq_obj2id[e] < 0) continue;
      follows.set(model.eq_obj1id[e], [model.eq_obj2id[e], model.eq_data[11 * e + 1]]);
    }
    this.chains = this.config.chains.map((cfg) => {
      const chain = { cfg, joints: [], active: [], followers: [], error: null };
      chain.tip = this.id('mjOBJ_BODY', cfg.tip);
      chain.root = this.id('mjOBJ_BODY', cfg.root);
      if (chain.tip < 0) { chain.error = `no body ${cfg.tip}`; return chain; }
      // A TCP site: the tip frame. Its body moves with the tip, so the joint
      // path is found from whichever of the two is lower in the tree.
      chain.site = cfg.site ? this.tipSites(chain.tip).find(([, n]) => n === cfg.site)?.[0] ?? -2 : -1;
      if (chain.site === -2) { chain.error = `no site ${cfg.site} fixed to ${cfg.tip}`; return chain; }
      if (chain.site >= 0 && this.ancestors(model.site_bodyid[chain.site]).includes(chain.tip)) chain.tip = model.site_bodyid[chain.site];
      const path = this.ancestors(chain.tip);
      const at = path.indexOf(chain.root);
      if (at < 0) { chain.error = `${cfg.root} is not ${cfg.tip} or above it`; return chain; }
      chain.gripper = cfg.gripper ? this.id('mjOBJ_ACTUATOR', cfg.gripper) : -1;
      for (const b of path.slice(0, at + 1).reverse()) {
        for (let j = model.body_jntadr[b], n = j + model.body_jntnum[b]; j >= 0 && j < n; j++) {
          const type = model.jnt_type[j], name = this.nameOf('mjOBJ_JOINT', j);
          const joint = { j, name, type, qadr: model.jnt_qposadr[j], status: 'ok' };
          const a = servo.get(j), own = this.owner(j);
          if (type !== 2 && type !== 3) joint.status = 'free or ball joint';
          else if (own) joint.status = `driven by ${own}`;
          else if (a === undefined) joint.status = follows.has(j) ? `follows ${this.nameOf('mjOBJ_JOINT', follows.get(j)[0])}` : 'no position actuator';
          else if (cfg.locked.includes(name)) joint.status = 'locked';
          if (joint.status === 'ok') {
            joint.act = a;
            joint.gear = model.actuator_gear[6 * a];
            joint.range = model.jnt_limited[j] ? [model.jnt_range[2 * j], model.jnt_range[2 * j + 1]] : [-Infinity, Infinity];
            if (model.actuator_ctrllimited[a]) {
              const [lo, hi] = [model.actuator_ctrlrange[2 * a], model.actuator_ctrlrange[2 * a + 1]].map((v) => v / joint.gear).sort((x, y) => x - y);
              joint.range = [Math.max(joint.range[0], lo), Math.min(joint.range[1], hi)];
            }
          }
          if (follows.has(j)) joint.follows = follows.get(j);
          chain.joints.push(joint);
        }
      }
      chain.active = chain.joints.filter((jt) => jt.status === 'ok');
      // Followers whose leader moves: [follower, leader index in active, coefficient].
      chain.followers = chain.joints.flatMap((jt) => {
        if (!jt.follows || jt.status === 'ok') return [];
        const k = chain.active.findIndex((l) => l.j === jt.follows[0]);
        return k < 0 ? [] : [[jt, k, jt.follows[1]]];
      });
      if (!chain.active.length) { chain.error = 'no joint it can move'; return chain; }
      // The target's frame: the parent of the first body the chain moves, so
      // a root on a free joint (G1's pelvis) or above joints owned elsewhere
      // still puts the target on the robot, not in the world.
      chain.ref = model.body_parentid[model.jnt_bodyid[chain.active[0].j]];
      return chain;
    });
  }

  // Forward kinematics of the commanded joint positions into this.scratch.
  commanded(data) {
    const s = this.scratch;
    s.qpos.set(data.qpos);
    for (const chain of this.chains) {
      for (const jt of chain.active) s.qpos[jt.qadr] = data.ctrl[jt.act] / jt.gear;
      for (const [f, k, c] of chain.followers) s.qpos[f.qadr] = c * s.qpos[chain.active[k].qadr];
    }
    this.mujoco.mj_kinematics(this.model, s);
    return s;
  }

  // World pose {pos, rot} of body b with a local offset, from MjData d.
  static frame(d, b, offset = [0, 0, 0]) {
    const rot = Array.from(d.xmat.subarray(9 * b, 9 * b + 9)), p = d.xpos.subarray(3 * b, 3 * b + 3);
    const o = mulMV(rot, offset);
    return { pos: [p[0] + o[0], p[1] + o[1], p[2] + o[2]], rot };
  }

  // The chain's tip frame {pos, rot} in MjData d: its site's, or its body's
  // moved by the offset.
  tipFrame(d, chain) {
    const i = chain.site;
    if (i < 0) return IkController.frame(d, chain.tip, chain.cfg.offset);
    return { pos: Array.from(d.site_xpos.subarray(3 * i, 3 * i + 3)), rot: Array.from(d.site_xmat.subarray(9 * i, 9 * i + 9)) };
  }

  // The chain's target, starting where its tip is (commanded) if it has none.
  target(chain, data, s = null) {
    let t = this.targets.get(chain.cfg.name);
    if (!t) {
      s ??= this.commanded(data);
      const tip = this.tipFrame(s, chain), ref = IkController.frame(s, chain.ref);
      t = { pos: mulTV(ref.rot, tip.pos.map((v, i) => v - ref.pos[i])), rot: mulTM(ref.rot, tip.rot) };
      t.home = chain.active.map((jt) => s.qpos[jt.qadr]);
      this.targets.set(chain.cfg.name, t);
    }
    return t;
  }

  // Target pose {pos, rot} in the world, or null (for the markers).
  targetPose(chain, data) {
    if (chain.error) return null;
    const t = this.target(chain, data), ref = IkController.frame(data, chain.ref);
    return { pos: mulMV(ref.rot, t.pos).map((v, i) => v + ref.pos[i]), rot: mulMM(ref.rot, t.rot) };
  }

  setTargetPose(chain, data, { pos, rot }) {
    const t = this.target(chain, data), ref = IkController.frame(data, chain.ref);
    t.pos = mulTV(ref.rot, pos.map((v, i) => v - ref.pos[i]));
    t.rot = mulTM(ref.rot, rot);
  }

  // Opens a closed gripper or closes an open one (the high end of its
  // ctrlrange taken as open, as for the Franka hand).
  toggleGripper(chain, data) {
    const a = chain.gripper, m = this.model;
    if (a < 0) return;
    const [lo, hi] = [m.actuator_ctrlrange[2 * a], m.actuator_ctrlrange[2 * a + 1]];
    data.ctrl[a] = data.ctrl[a] > (lo + hi) / 2 ? lo : hi;
  }

  // One control tick: every chain one DLS step towards its target.
  step(data, valid = () => true, dt = CONTROL_DT) {
    if (!this.enabled || !valid() || !this.chains.length) return;
    const s = this.commanded(data);
    for (const chain of this.chains) if (!chain.error) this.solveChain(chain, data, s, dt);
  }

  solveChain(chain, data, s, dt) {
    const { cfg, active } = chain, n = active.length;
    const t = this.target(chain, data, s);
    const tip = this.tipFrame(s, chain), ref = IkController.frame(data, chain.ref);
    const goal = mulMV(ref.rot, t.pos).map((v, i) => v + ref.pos[i]);
    let ep = goal.map((v, i) => v - tip.pos[i]);
    let er = rotVec(mulMT(mulMM(ref.rot, t.rot), tip.rot));  // R_goal R_tipᵀ
    chain.err = [Math.hypot(...ep), Math.hypot(...er)];
    const cap = (v, m) => { const l = Math.hypot(...v); return l > m ? v.map((k) => (k * m) / l) : v; };
    ep = cap(ep, MAX_STEP[0]);
    er = cap(er, MAX_STEP[1]);

    // Jacobian columns: d(tip position), d(tip rotation) per active joint.
    const col = (j, scale = 1) => {
      const a = s.xaxis.subarray(3 * j, 3 * j + 3);
      if (this.model.jnt_type[j] === 2) return [[...a].map((k) => k * scale), [0, 0, 0]];
      const r = cross(a, tip.pos.map((v, i) => v - s.xanchor[3 * j + i]));
      return [r.map((k) => k * scale), [...a].map((k) => k * scale)];
    };
    const cols = active.map((jt) => col(jt.j));
    for (const [f, k, c] of chain.followers) {
      const [p, r] = col(f.j, c);
      for (let i = 0; i < 3; i++) { cols[k][0][i] += p[i]; cols[k][1][i] += r[i]; }
    }
    // Position and orientation rows; a chain with fewer than 6 joints gets
    // the damped least-squares compromise between them.
    const rows = [], e = [], w = ROT_WEIGHT;
    for (let i = 0; i < 3; i++) { rows.push(cols.map((c) => c[0][i])); e.push(ep[i]); }
    for (let i = 0; i < 3; i++) { rows.push(cols.map((c) => c[1][i] * w)); e.push(er[i] * w); }
    const q = active.map((jt) => s.qpos[jt.qadr]);

    // dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e, plus the home pull projected into J's null
    // space. A joint the step would push past its range is taken out (its
    // column zeroed) and the step solved again, so the others do not chase
    // what it cannot do.
    const free = active.map(() => true);
    let dq;
    for (let pass = 0; pass < 4; pass++) {
      dq = this.dlsStep(rows.map((row) => row.map((v, c) => (free[c] ? v : 0))), e, n,
        q.map((v, c) => (free[c] ? NULL_GAIN * dt * (t.home[c] - v) : 0)));
      let again = false;
      active.forEach((jt, c) => {
        const next = q[c] + dq[c];
        if (free[c] && (next > jt.range[1] || next < jt.range[0])) { free[c] = false; again = true; }
      });
      if (!again) break;
    }

    // Joint speed limits, scaling the whole step so its direction is kept.
    const over = Math.max(1, ...active.map((jt, c) => Math.abs(dq[c]) / (MAX_SPEED[jt.type] * dt)));
    const scale = 1 / over;
    active.forEach((jt, c) => {
      const v = Math.max(jt.range[0], Math.min(jt.range[1], q[c] + dq[c] * scale));
      data.ctrl[jt.act] = v * jt.gear;
    });
  }

  // One damped least-squares step for Jacobian `rows` (m x n), error `e`,
  // plus `z` projected into the Jacobian's null space.
  dlsStep(rows, e, n, z) {
    const m = rows.length;
    const pinv = (v) => {
      const A = new Float64Array(m * m);
      for (let i = 0; i < m; i++) {
        for (let k = 0; k < m; k++) {
          let sum = i === k ? LAMBDA * LAMBDA : 0;
          for (let c = 0; c < n; c++) sum += rows[i][c] * rows[k][c];
          A[i * m + k] = sum;
        }
      }
      const y = solve(A, Float64Array.from(v), m);
      return Array.from({ length: n }, (_, c) => rows.reduce((sum, row, i) => sum + row[c] * y[i], 0));
    };
    const Jz = rows.map((row) => row.reduce((sum, v, c) => sum + v * z[c], 0));
    const back = pinv(Jz);
    return pinv(e).map((v, c) => v + z[c] - back[c]);
  }
}
