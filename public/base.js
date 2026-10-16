// Mobile-base teleoperation: a body twist command (vx, vy, yaw rate) drives
// the robot's base, with a PI loop on the measured twist (MuJoCo's ground
// truth as perfect odometry) against slip and drift.
//
// Kinds:
// - 'wheels': a floating base on driven wheels. Each wheel's position and
//   spin axis come from the model; its speed is the generic rolling
//   constraint ω = g·v / (r g·d), with v the wheel center's velocity, d the
//   rolling direction (axis × z) and g the direction the wheel pushes: d
//   for a plain wheel (differential and skid steer: no sideways motion),
//   d ± axis for a mecanum wheel whose rollers sit at ±45° (`roller`).
// - 'planar': x / y / yaw joints with position servos on the robot's root
//   body (TidyBot): the pose target is integrated from the command.
//
// Configs, first found: one saved from the Teleop panel, BASE_DEFAULTS per
// robot file, or one made from the model (autoConfig).

import { CONTROL_DT } from './groot.js';

export const BASE_DEFAULTS = {
  'digital_twin/garmi/garmi.xml': {
    kind: 'wheels',
    base: 'base_free',
    wheels: [
      { actuator: 'wheel_fl', roller: 1 }, { actuator: 'wheel_fr', roller: -1 },
      { actuator: 'wheel_rl', roller: -1 }, { actuator: 'wheel_rr', roller: 1 },
    ],
    radius: 0.0759,
    // Strafing slower than driving, like the real robot (teleop.py).
    drive: { forward: 0.7, back: 0.7, side: 0.42, turn: 1.2 },
    // Garmi's rollers give yaw a lever of only |x - y| = 4 cm, and the base
    // turns ~3x slower than that predicts: hence the yaw feedforward.
    ff: [1.0, 1.0, 3.0], kp: [1.6, 1.6, 4.0], ki: [0.0, 0.0, 4.0], iclamp: [0.5, 0.5, 2.0],
    accel: [2, 2, 4],
  },
};

export const KINDS = [['', 'none'], ['wheels', 'wheels (differential, skid, mecanum)'], ['planar', 'planar joints (x, y, yaw)']];
const STORE = 'mujoco-filament/base/';
// Per axis (vx, vy, yaw rate): feedforward, P, I, integral limit, and the
// command's acceleration limit (m/s², rad/s²): a step to full speed tips a
// tall robot over (TIAGo, RB-Y1).
const GAINS = { ff: [1, 1, 1], kp: [0.5, 0.5, 0.3], ki: [0.3, 0.3, 0.2], iclamp: [0.2, 0.2, 0.2], accel: [0.5, 0.5, 1.0] };

export function loadBaseConfig(key) {
  try {
    const saved = localStorage.getItem(STORE + key);
    if (saved) return { config: JSON.parse(saved), saved: true };
  } catch { /* no storage */ }
  return { config: BASE_DEFAULTS[key] ? structuredClone(BASE_DEFAULTS[key]) : null, saved: false };
}

export class BaseController {
  // `key` names the robot file (BASE_DEFAULTS, saved configs); `prefix` is
  // the robot's name prefix in the world.
  constructor(key, prefix = '') {
    this.key = key;
    this.prefix = prefix;
    this.cmd = [0, 0, 0];       // desired vx, vy (m/s), yaw rate (rad/s), base frame
    this.measured = null;       // last measured twist, for the readout
    ({ config: this.config, saved: this.saved } = loadBaseConfig(key));
    this.reset();
  }

  get name() {
    return this.config?.kind === 'planar' ? 'Planar base (pose servo)' : 'Wheeled base (twist PI)';
  }

  // Keyboard speeds (TeleopPanel); null while the base is not set up.
  get drive() {
    return this.ready ? this.config.drive : null;
  }

  get hasDefaults() { return !!BASE_DEFAULTS[this.key]; }

  reset() {
    this.integ = [0, 0, 0];
    this.ref = [0, 0, 0];  // the command after the acceleration limit
    this.goal = null;  // planar: the commanded [x, y, yaw]
  }

  pose() {}

  save() {
    try { localStorage.setItem(STORE + this.key, JSON.stringify(this.config)); this.saved = true; } catch { /* no storage */ }
  }

  restoreDefaults() {
    try { localStorage.removeItem(STORE + this.key); } catch { /* no storage */ }
    this.saved = false;
    this.config = BASE_DEFAULTS[this.key] ? structuredClone(BASE_DEFAULTS[this.key]) : this.autoConfig();
    this.resolve();
  }

  bind(mujoco, model, owner = () => null, bodies = [1, model.nbody]) {
    this.mujoco = mujoco;
    this.model = model;
    this.owner = owner;
    this.bodies = bodies;
    this.config ??= this.autoConfig();
    this.resolve();
  }

  id(type, name) {
    return this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj[type].value, this.prefix + name);
  }

  nameOf(type, i) {
    const n = this.mujoco.mj_id2name(this.model, this.mujoco.mjtObj[type].value, i) ?? '';
    return n.startsWith(this.prefix) ? n.slice(this.prefix.length) : n;
  }

  inRobot(b) { return b >= this.bodies[0] && b < this.bodies[1]; }

  // This controller's name if it drives joint `j` (IK leaves it alone).
  owns(j) {
    return this.ready && this.joints.has(j) ? 'base' : null;
  }

  // How actuator a drives its joint: 'velocity', 'position' or 'motor'
  // (torque); null if it is not on a single joint.
  static mode(model, a) {
    if (model.actuator_trntype[a] !== 0) return null;
    const b = model.actuator_biasprm;
    if (b[10 * a + 1] < 0) return 'position';
    if (b[10 * a + 2] < 0) return 'velocity';
    return 'motor';
  }

  // Forward kinematics of the model's default pose (qpos0), for geometry.
  rest() {
    const d = new this.mujoco.MjData(this.model);
    this.mujoco.mj_kinematics(this.model, d);
    return d;
  }

  // Joint j's anchor and axis in body `base`'s frame, from MjData d.
  inBase(d, base, j) {
    const R = d.xmat.subarray(9 * base, 9 * base + 9), p = d.xpos.subarray(3 * base, 3 * base + 3);
    const toBase = (v) => [0, 1, 2].map((c) => R[c] * v[0] + R[3 + c] * v[1] + R[6 + c] * v[2]);
    return {
      anchor: toBase([0, 1, 2].map((i) => d.xanchor[3 * j + i] - p[i])),
      axis: toBase(Array.from(d.xaxis.subarray(3 * j, 3 * j + 3))),
    };
  }

  // A config found in the model: a free root body on wheels (hinge joints
  // with a velocity servo or motor, horizontal axis), mecanum if a wheel
  // carries free rollers; or x / y / yaw position servos on the root body.
  autoConfig() {
    const { model } = this, root = this.bodies[0];
    const acts = new Map();  // joint -> actuator
    for (let a = 0; a < model.nu; a++) if (model.actuator_trntype[a] === 0) acts.set(model.actuator_trnid[2 * a], a);
    const rootJoints = [];
    for (let j = model.body_jntadr[root], e = j + model.body_jntnum[root]; j >= 0 && j < e; j++) rootJoints.push(j);
    const free = rootJoints.find((j) => model.jnt_type[j] === 0);
    const d = this.rest();
    try {
      if (free === undefined) {
        // Planar: two slides (x, y) and a hinge (yaw), all position servos.
        const pick = (type, k) => rootJoints.find((j) => model.jnt_type[j] === type && acts.has(j)
          && BaseController.mode(model, acts.get(j)) === 'position' && Math.abs(model.jnt_axis[3 * j + k]) > 0.9);
        const x = pick(2, 0), y = pick(2, 1), yaw = pick(3, 2);
        if (x === undefined || y === undefined || yaw === undefined) return { kind: '', auto: true };
        const n = (j) => this.nameOf('mjOBJ_ACTUATOR', acts.get(j));
        return { kind: 'planar', planar: { x: n(x), y: n(y), yaw: n(yaw) }, drive: { forward: 0.5, back: 0.3, side: 0.3, turn: 0.8 }, accel: [1, 1, 2], auto: true };
      }
      const wheels = [];
      for (const [j, a] of acts) {
        const b = model.jnt_bodyid[j];
        if (!this.inRobot(b) || model.jnt_type[j] !== 3 || this.owner(j)) continue;
        const mode = BaseController.mode(model, a);
        if (mode !== 'velocity' && mode !== 'motor') continue;
        const { axis } = this.inBase(d, root, j);
        if (Math.abs(axis[2]) > 0.3 || d.xanchor[3 * j + 2] > 0.5) continue;  // not a wheel: axis not level, or high up
        // Rollers: unactuated hinges on bodies hanging from the wheel. The
        // lowest one touches the floor; it pushes along its axis.
        let low = null;
        for (let k = 0; k < model.njnt; k++) {
          const kb = model.jnt_bodyid[k];
          if (model.body_parentid[kb] !== b || model.jnt_type[k] !== 3 || acts.has(k)) continue;
          if (!low || d.xpos[3 * kb + 2] < low.z) low = { z: d.xpos[3 * kb + 2], k };
        }
        let roller = 0;
        if (low) {
          const g = this.inBase(d, root, low.k).axis, dir = [axis[1], -axis[0]];
          roller = Math.sign((g[0] * axis[0] + g[1] * axis[1]) * (g[0] * dir[0] + g[1] * dir[1])) || 0;
        }
        wheels.push({ actuator: this.nameOf('mjOBJ_ACTUATOR', a), roller, z: d.xanchor[3 * j + 2], j, a });
      }
      if (wheels.length < 2) return { kind: '', auto: true };
      // Radius: the wheel axle's height in the default pose (the robot stands on z = 0).
      const radius = +Math.min(...wheels.map((w) => w.z)).toFixed(4);
      const mecanum = wheels.every((w) => w.roller);
      // Speeds within the slowest wheel's ctrlrange.
      const wmax = Math.min(...wheels.map(({ a, j }) => {
        const hi = model.actuator_ctrllimited[a] ? Math.abs(model.actuator_ctrlrange[2 * a + 1] / model.actuator_gear[6 * a]) : 10;
        return BaseController.mode(model, a) === 'velocity' ? hi : 10;
      }));
      const forward = +Math.min(0.6, 0.7 * wmax * radius).toFixed(2);
      // '' = the root body's free joint (it may have no name).
      const baseName = this.nameOf('mjOBJ_JOINT', free);
      return {
        kind: 'wheels', base: this.id('mjOBJ_JOINT', baseName) === free ? baseName : '',
        wheels: wheels.map(({ actuator, roller }) => ({ actuator, roller })),
        radius, drive: { forward, back: +(0.6 * forward).toFixed(2), side: mecanum ? +(0.6 * forward).toFixed(2) : 0, turn: 0.8 },
        ...structuredClone(GAINS), kw: 1, auto: true,
      };
    } finally {
      d.delete();
    }
  }

  // Rebuilds the resolved setup from this.config (after bind or an edit).
  resolve() {
    const { model, config: cfg } = this;
    this.ready = false;
    this.error = null;
    this.joints = new Set();
    if (!model || !cfg?.kind) return;
    for (const k of Object.keys(GAINS)) cfg[k] ??= [...GAINS[k]];
    cfg.drive ??= { forward: 0.5, back: 0.3, side: 0, turn: 0.8 };
    const actuator = (name) => {
      const a = this.id('mjOBJ_ACTUATOR', name);
      if (a < 0) throw new Error(`no actuator ${name}`);
      const mode = BaseController.mode(model, a);
      if (!mode) throw new Error(`${name} does not drive a joint`);
      const j = model.actuator_trnid[2 * a];
      this.joints.add(j);
      const range = model.actuator_ctrllimited[a] ? [model.actuator_ctrlrange[2 * a], model.actuator_ctrlrange[2 * a + 1]] : [-Infinity, Infinity];
      return { a, j, mode, gear: model.actuator_gear[6 * a], range, qadr: model.jnt_qposadr[j], dadr: model.jnt_dofadr[j] };
    };
    try {
      if (cfg.kind === 'planar') {
        this.axes = ['x', 'y', 'yaw'].map((k) => actuator(cfg.planar?.[k] ?? ''));
      } else {
        const root = this.bodies[0], rootFree = model.body_jntnum[root] > 0 && model.jnt_type[model.body_jntadr[root]] === 0 ? model.body_jntadr[root] : -1;
        const base = cfg.base ? this.id('mjOBJ_JOINT', cfg.base) : rootFree;
        if (base < 0 || model.jnt_type[base] !== 0) throw new Error(cfg.base ? `no free joint ${cfg.base}` : 'the robot has no free root joint');
        this.baseV = model.jnt_dofadr[base];
        this.baseBody = model.jnt_bodyid[base];
        if ((cfg.wheels ?? []).length < 2) throw new Error('needs two wheels or more');
        const d = this.rest();
        try {
          this.wheels = cfg.wheels.map((w) => {
            const info = actuator(w.actuator);
            if (info.mode === 'position') throw new Error(`${w.actuator} is a position servo`);
            const { anchor, axis } = this.inBase(d, this.baseBody, info.j);
            const dir = [axis[1], -axis[0]], l = Math.hypot(...dir) || 1;
            const dn = [dir[0] / l, dir[1] / l], an = [axis[0] / l, axis[1] / l];
            const s = w.roller || 0;  // push direction: d, or d ± axis for ±45° rollers
            const g = [dn[0] + s * an[0], dn[1] + s * an[1]];
            return { ...info, x: anchor[0], y: anchor[1], g, gd: g[0] * dn[0] + g[1] * dn[1] };
          });
        } finally {
          d.delete();
        }
      }
      this.ready = true;
    } catch (err) {
      this.error = err.message;
    }
  }

  // Base twist in the base frame: a free joint's qvel is the linear velocity
  // in the world frame, then the angular velocity in the body frame.
  measure(data) {
    if (this.config.kind === 'planar') {
      const [vx, vy, w] = this.axes.map((ax) => data.qvel[ax.dadr]), th = data.qpos[this.axes[2].qadr];
      return [Math.cos(th) * vx + Math.sin(th) * vy, -Math.sin(th) * vx + Math.cos(th) * vy, w];
    }
    const R = data.xmat.subarray(9 * this.baseBody, 9 * this.baseBody + 9);
    const v = data.qvel.subarray(this.baseV, this.baseV + 3);
    return [
      R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
      R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
      data.qvel[this.baseV + 5],
    ];
  }

  // Body twist -> wheel speeds (rad/s), in config.wheels order.
  wheelSpeeds([vx, vy, wz]) {
    const r = this.config.radius;
    return this.wheels.map((w) => {
      const v = [vx - wz * w.y, vy + wz * w.x];
      return (w.g[0] * v[0] + w.g[1] * v[1]) / (r * w.gd);
    });
  }

  // One control tick of CONTROL_DT seconds.
  step(data, valid = () => true, dt = CONTROL_DT) {
    if (!this.ready || !valid()) return;
    const cfg = this.config;
    const clamp = (v, [lo, hi]) => Math.max(lo, Math.min(hi, v));
    this.ref = this.ref.map((r, k) => r + clamp(this.cmd[k] - r, [-cfg.accel[k] * dt, cfg.accel[k] * dt]));
    if (cfg.kind === 'planar') {
      // Integrate the commanded pose, starting from where the base is.
      this.goal ??= this.axes.map((ax) => data.ctrl[ax.a] / ax.gear);
      const th = this.goal[2], [vx, vy, w] = this.ref;
      this.goal[0] += (Math.cos(th) * vx - Math.sin(th) * vy) * dt;
      this.goal[1] += (Math.sin(th) * vx + Math.cos(th) * vy) * dt;
      this.goal[2] += w * dt;
      this.axes.forEach((ax, k) => { data.ctrl[ax.a] = clamp(this.goal[k] * ax.gear, ax.range); });
      this.measured = this.measure(data);
      return;
    }
    const m = this.measured = this.measure(data);
    // No command: brake (wheels held still), like letting go of a joystick;
    // the loop would only chase caster and contact jitter.
    if (this.ref.every((r) => Math.abs(r) < 1e-3)) {
      this.integ = [0, 0, 0];
      for (const wh of this.wheels) data.ctrl[wh.a] = wh.mode === 'velocity' ? 0 : clamp(-(cfg.kw ?? 1) * data.qvel[wh.dadr], wh.range);
      return;
    }
    const twist = [0, 1, 2].map((k) => {
      const e = this.ref[k] - m[k];
      this.integ[k] = Math.max(-cfg.iclamp[k], Math.min(cfg.iclamp[k], this.integ[k] + cfg.ki[k] * e * dt));
      return cfg.ff[k] * this.ref[k] + cfg.kp[k] * e + this.integ[k];
    });
    this.wheelSpeeds(twist).forEach((w, i) => {
      const wh = this.wheels[i];
      // A velocity servo holds qdot = ctrl / gear; a motor gets a P loop on the wheel speed.
      const u = wh.mode === 'velocity' ? w * wh.gear : (cfg.kw ?? 1) * (w - data.qvel[wh.dadr]);
      data.ctrl[wh.a] = clamp(u, wh.range);
    });
  }
}
