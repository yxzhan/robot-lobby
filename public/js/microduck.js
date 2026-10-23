// The Microduck walking policy trained in microduck_rl (github.com/
// pollen-robotics/microduck_rl: mjlab, MuJoCo Warp, rsl_rl PPO) and exported
// by scripts/export_microduck.py: walk.onnx (velocity task), with the
// observation normalizer baked in, and policy.json, which has the rest:
//
// - joints / actuators, default_pose, action_scale: the 14 servos in action
//   order; target = default + action * scale (no clipping, as in training);
// - obs: mjlab's actor observation, 61 values, its parts in order: the root
//   body's angular velocity (base_ang_vel) and the gravity direction in its
//   frame (projected_gravity), q - default (joint_pos), qdot one control
//   tick old (joint_vel: training delays it by one step), the last action,
//   then the command block: the twist (command: vx, vy, yaw rate) and the
//   commands of policy.json's "commands" (cmd:head, the head joints' offsets
//   from the default pose; cmd:body, a body pose offset, which the walking
//   task barely follows), set in the Teleop panel (commands.js);
// - command_range, min_command: what the policy was trained to follow, and
//   its dead band (it stands for less);
// - bam: the XL330 servo model the policy was trained with (BAM M6,
//   github.com/Rhoban/bam, as bam.mjlab.BamActuator runs it). The robot's
//   position actuators become torque motors on bind (given back on unbind)
//   and before every physics step (substep):
//     duty   = clip(kp * error_gain * (target - q)), within the firmware's
//              current limit and ±max_pwm
//     torque = kt (vin duty - kt qdot) / R, vin sagging with the last torques
//   while the friction budget (Coulomb, Stribeck, and the directional and
//   quadratic load terms, from the gearbox load: bias and constraint forces
//   without the joint friction itself, and the last motor torque) goes into
//   dof_frictionloss / dof_damping for MuJoCo's solver to apply, with the
//   stiffened friction constraint of training. Targets reach the servos
//   `delay` seconds after inference, as the command delay of training.
// server/policies.py (MicroduckController, BamServos) is the same in Python.

import { runBatched } from './batch.js';
import { commandDefaults, commandVector } from './commands.js';

const FRICTION_DOF = 3;  // mjCNSTR_FRICTION_DOF

// {ort, session, spec}. `source(name)` returns a file's URL.
export async function loadMicroduckPolicy(ort, source) {
  const spec = await fetch(source('policy.json')).then((r) => {
    if (!r.ok) throw new Error(`policy.json: ${r.status} ${r.statusText}`);
    return r.json();
  });
  return { ort, session: await ort.InferenceSession.create(source('walk.onnx')), spec };
}

// Key speeds a bit inside the trained command ranges, and the dead band
// (minSpeed / minTurn: the goal marker's least command, goal.js).
function driveOf(r, min) {
  const top = (v) => +(0.8 * Math.abs(v)).toFixed(2);
  return {
    forward: top(r.lin_vel_x[1]), back: top(r.lin_vel_x[0]),
    side: Math.min(top(r.lin_vel_y[0]), top(r.lin_vel_y[1])),
    turn: Math.min(top(r.ang_vel_yaw[0]), top(r.ang_vel_yaw[1])),
    ...(min && { minSpeed: min.speed, minTurn: min.turn }),
  };
}

export class MicroduckController {
  constructor(policy, prefix = '', name = 'Microduck walking policy') {
    this.policy = policy;
    this.spec = policy.spec;
    this.session = policy.session;
    this.prefix = prefix;
    this.name = name;
    this.drive = driveOf(this.spec.command_range, this.spec.min_command);
    this.cmd = [0, 0, 0];
    this.commands = this.spec.commands ?? {};
    this.command = commandDefaults(this.commands);
    this.reset();
  }

  reset() {
    const n = this.spec.joints.length;
    this.action = new Float32Array(n);
    this.qvel = new Float64Array(n);  // the joint velocities of the last tick: the observation's
    this.pending = [];                // [[time, targets]]: commands reach the servos after the delay
    this.prevTorque = new Float64Array(n);  // the battery sag
  }

  bind(mujoco, model) {
    const T = mujoco.mjtObj, s = this.spec, b = s.bam;
    const id = (type, name) => {
      const i = mujoco.mj_name2id(model, type.value, this.prefix + name);
      if (i < 0) throw new Error(`${this.name}: no ${this.prefix}${name} in the model`);
      return i;
    };
    const joints = s.joints.map((n) => id(T.mjOBJ_JOINT, n));
    const acts = s.actuators.map((n) => id(T.mjOBJ_ACTUATOR, n));
    const root = id(T.mjOBJ_BODY, s.root);
    const free = model.body_jntadr[root];
    if (free < 0 || model.jnt_type[free] !== 0) throw new Error(`${this.name}: ${this.prefix}${s.root} has no free joint`);
    this.model = model;
    this.root = root;
    this.freeV = model.jnt_dofadr[free];
    this.qadr = Int32Array.from(joints, (j) => model.jnt_qposadr[j]);
    this.dofs = Int32Array.from(joints, (j) => model.jnt_dofadr[j]);
    this.act = Int32Array.from(acts);
    this.driven = new Set(joints);
    this.target = Float64Array.from(s.default_pose);
    const m = model, sl = (arr, i, k) => Array.from(arr.subarray(k * i, k * i + k));
    // Motors driven by BAM in place of the position servos, as bam.mjlab does in training.
    this.saved = {
      act: acts.map((a) => [sl(m.actuator_gainprm, a, 10), sl(m.actuator_biasprm, a, 10), m.actuator_ctrllimited[a],
        m.actuator_forcelimited[a], sl(m.actuator_forcerange, a, 2)]),
      dof: Array.from(this.dofs, (d) => [m.dof_damping[d], m.dof_armature[d], m.dof_frictionloss[d],
        sl(m.dof_solref, d, 2), sl(m.dof_solimp, d, 5)]),
    };
    for (const a of acts) {
      m.actuator_gainprm.fill(0, 10 * a, 10 * a + 10);
      m.actuator_gainprm[10 * a] = 1 / m.actuator_gear[6 * a];
      m.actuator_biasprm.fill(0, 10 * a, 10 * a + 10);
      m.actuator_ctrllimited[a] = 0;
      m.actuator_forcelimited[a] = 1;
      m.actuator_forcerange.set([-b.force_limit, b.force_limit], 2 * a);
    }
    for (const d of this.dofs) {
      m.dof_armature[d] = b.armature;
      m.dof_solref.set(b.solref_friction, 2 * d);
      m.dof_solimp.set(b.solimp_friction, 5 * d);
    }
  }

  unbind() {
    const m = this.model;
    if (!m || !this.saved) return;
    this.act.forEach((a, i) => {
      const [g, bias, cl, fl, fr] = this.saved.act[i];
      m.actuator_gainprm.set(g, 10 * a); m.actuator_biasprm.set(bias, 10 * a);
      m.actuator_ctrllimited[a] = cl; m.actuator_forcelimited[a] = fl; m.actuator_forcerange.set(fr, 2 * a);
    });
    this.dofs.forEach((d, i) => {
      const [damp, arm, fric, sr, si] = this.saved.dof[i];
      m.dof_damping[d] = damp; m.dof_armature[d] = arm; m.dof_frictionloss[d] = fric;
      m.dof_solref.set(sr, 2 * d); m.dof_solimp.set(si, 5 * d);
    });
    this.saved = null;
    this.driven = new Set();
  }

  owns(j) {
    return this.driven?.has(j) ? this.name : null;
  }

  pose(data) {
    this.spec.default_pose.forEach((v, i) => { data.qpos[this.qadr[i]] = v; });
    this.target = Float64Array.from(this.spec.default_pose);
  }

  // Before every physics step: the servos' torques and friction (see the top).
  substep(data) {
    while (this.pending.length && data.time >= this.pending[0][0] - 1e-9) this.target = this.pending.shift()[1];
    const p = this.spec.bam, m = this.model, n = this.dofs.length;
    const { qpos, qvel, qfrc_bias: bias, qfrc_constraint: cons, qfrc_actuator: actf } = data;
    let load = 0;
    for (let i = 0; i < n; i++) load += Math.abs(this.prevTorque[i]);
    const vin = Math.max(p.vin - p.vin_drop_gain * load, p.vin_min);
    // Our own joint friction, to take out of the constraint forces.
    const fric = new Map();
    const nefc = data.nefc, type = data.efc_type, efcId = data.efc_id, force = data.efc_force;
    for (let r = 0; r < nefc; r++) {
      if (type[r] === FRICTION_DOF) fric.set(efcId[r], (fric.get(efcId[r]) ?? 0) + force[r]);
    }
    for (let i = 0; i < n; i++) {
      const d = this.dofs[i], q = qpos[this.qadr[i]], dq = qvel[d];
      let duty = (this.target[i] - q) * p.kp * p.error_gain;
      if (p.max_current) {  // the firmware's current limit, a window on the duty cycle
        const c = p.kt * dq / vin, w = p.R * p.max_current / vin;
        duty = Math.min(c + w, Math.max(c - w, duty));
      }
      duty = Math.min(p.max_pwm, Math.max(-p.max_pwm, duty));
      const tau = p.kt * vin * duty / p.R - p.kt * p.kt * dq / p.R;
      data.ctrl[this.act[i]] = tau;
      this.prevTorque[i] = tau;
      const ext = -bias[d] + cons[d] - (fric.get(d) ?? 0), mot = actf[d];
      const st = Math.exp(-Math.pow(Math.abs(dq) / p.dtheta_stribeck, p.alpha));
      let loss = p.friction_base + st * p.friction_stribeck
        + Math.abs(ext * p.load_friction_external - mot * p.load_friction_motor)
        + st * Math.abs(ext * p.load_friction_external_stribeck - mot * p.load_friction_motor_stribeck);
      loss += st * (Math.abs(mot) > Math.abs(ext) ? p.load_friction_external_quad * ext * ext : p.load_friction_motor_quad * mot * mot);
      m.dof_frictionloss[d] = loss;
      m.dof_damping[d] = p.friction_viscous;
    }
  }

  // The policy input from `data` (synchronous); keeps qdot for the next tick.
  observe(data) {
    const { qpos, qvel, xmat } = data, s = this.spec;
    const R = xmat.subarray(9 * this.root, 9 * this.root + 9), v = this.freeV;
    const parts = {
      base_ang_vel: () => qvel.subarray(v + 3, v + 6),  // a free joint's angular velocity is in its body frame
      projected_gravity: () => [-R[6], -R[7], -R[8]],   // R^T (0, 0, -1)
      joint_pos: () => Array.from(this.qadr, (a, i) => qpos[a] - s.default_pose[i]),
      joint_vel: () => this.qvel,
      last_action: () => this.action,
      command: () => this.cmd,
    };
    const obs = new Float32Array(s.obs.reduce((t, [, k]) => t + k, 0));
    let o = 0;
    for (const [name, size] of s.obs) {
      const c = name.startsWith('cmd:') && this.commands[name.slice(4)];
      const part = c ? commandVector(c, this.command[name.slice(4)]) : parts[name]?.();
      if (!part || part.length !== size) throw new Error(`${this.name}: cannot build observation part ${name}`);
      obs.set(part, o);
      o += size;
    }
    this.qvel = Float64Array.from(this.dofs, (d) => qvel[d]);
    return obs;
  }

  async step(data, valid = () => true) {
    const obs = this.observe(data), { ort } = this.policy, s = this.spec;
    const action = await runBatched(ort, this.session, obs, s.input, s.output);
    if (!valid()) return;
    this.action = action;
    const target = Float64Array.from(this.action, (a, i) => s.default_pose[i] + a * s.action_scale);
    this.pending.push([data.time + s.bam.delay, target]);
  }
}
