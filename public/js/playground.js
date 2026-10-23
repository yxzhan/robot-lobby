// Joystick locomotion policies trained in MuJoCo Playground (training/,
// MJX + Brax PPO) and exported by training/export_onnx.py: policy.onnx (the
// observation normalization, the MLP and the tanh of the action mean) and
// policy.json, which says everything else, so one controller serves any of
// them:
//
// - joints / actuators: the policy's joints in action order, and the
//   actuators driving them;
// - default_pose, action_scale: target = default + action * scale, clipped to
//   ctrl_range;
// - actuator_gain / actuator_bias / dof_*: the PD servo (force = gain * target
//   + bias . [1, q, qdot], within force_range), joint damping, armature and
//   friction loss it was trained with, set on bind (and given back on
//   unbind) like GR00T's gains. A position servo (affine bias) gets the
//   gains; a torque motor (e.g. menagerie's go2.xml) gets the servo's
//   torque every physics step (substep), computed here;
// - obs: the observation's parts in order (training/export_onnx.py's
//   Observer is the reference; server/policies.py has the same in Python).
//   Sensor values are those of the forward pass at the start of the last
//   step (cvel, site_xpos, ...), in the frame of the `imu` site (the root
//   body's if there is none):
//     linvel       velocimeter: the imu site's linear velocity
//     gyro         its angular velocity
//     gravity      (0, 0, -1) in its frame
//     upvector     its z axis in the world frame
//     joint_vel    qdot
//     joint_pos    q - default pose
//     qpos_error   history_len frames of q - last targets, newest first
//     feet_pos     the foot geoms' centers relative to the root, root frame
//     last_action  the raw action before the last one (Playground updates
//                  it after building the observation)
//     command      vx, vy (m/s), yaw rate (rad/s)
//     cmd:<name>   one of its other commands (commands.js), declared in
//                  policy.json's "commands"
// - command_range: what it was trained to follow (the key speeds stay inside);
//   min_command ({speed, turn}, optional): its dead band, below which it
//   stands (drive.minSpeed / minTurn for the goal marker, goal.js).

import { runBatched } from './batch.js';
import { commandDefaults, commandVector } from './commands.js';

// {ort, session, spec}. `source(name)` returns a file's URL.
export async function loadPlaygroundPolicy(ort, source) {
  const spec = await fetch(source('policy.json')).then((r) => {
    if (!r.ok) throw new Error(`policy.json: ${r.status} ${r.statusText}`);
    return r.json();
  });
  const session = await ort.InferenceSession.create(source(spec.onnx));
  return { ort, session, spec };
}

// Key speeds within the trained command ranges (a bit inside them).
function driveOf(range, min) {
  const top = (r, side) => +(0.8 * Math.abs(r?.[side] ?? 0)).toFixed(2);
  return {
    forward: top(range.lin_vel_x, 1), back: top(range.lin_vel_x, 0),
    side: Math.min(top(range.lin_vel_y, 0), top(range.lin_vel_y, 1)),
    turn: Math.min(top(range.ang_vel_yaw, 0), top(range.ang_vel_yaw, 1)),
    ...(min && { minSpeed: min.speed, minTurn: min.turn }),
  };
}

export class PlaygroundController {
  constructor(policy, prefix = '', name = 'Playground policy') {
    this.policy = policy;
    this.spec = policy.spec;
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
    this.action = new Float32Array(n);      // applied last
    this.prevAction = new Float32Array(n);  // the one before: what the observation holds
    this.targets = new Float64Array(n);  // Playground starts the error history against zero targets
    this.error = new Float64Array(this.spec.history_len * n);
    this.servo = Float64Array.from(this.spec.default_pose);  // what torque motors hold until the first action
  }

  bind(mujoco, model) {
    const T = mujoco.mjtObj, s = this.spec;
    const id = (type, name) => {
      const i = mujoco.mj_name2id(model, type.value, this.prefix + name);
      if (i < 0) throw new Error(`${this.name}: no ${this.prefix}${name} in the model`);
      return i;
    };
    const joints = s.joints.map((n) => id(T.mjOBJ_JOINT, n));
    const acts = s.actuators.map((n) => id(T.mjOBJ_ACTUATOR, n));
    const feet = s.feet.map((n) => id(T.mjOBJ_GEOM, n));
    const root = id(T.mjOBJ_BODY, s.root);
    const free = model.body_jntadr[root];
    if (free < 0 || model.jnt_type[free] !== 0) throw new Error(`${this.name}: ${this.prefix}${s.root} has no free joint`);
    // Position servos (affine bias) or torque motors (fixed gain, no bias).
    const torque = acts.map((a) => {
      if (model.actuator_biastype[a] === 1) return false;
      if (model.actuator_gaintype[a] === 0 && model.actuator_biastype[a] === 0) return true;
      throw new Error(`${this.name}: actuator ${s.actuators[acts.indexOf(a)]} is neither a position servo nor a motor`);
    });
    this.model = model;
    this.root = root;
    this.tree = model.body_rootid[root];  // cvel is about its subtree's center of mass
    this.imu = s.imu ? id(T.mjOBJ_SITE, s.imu) : -1;
    this.torque = torque;
    this.qadr = Int32Array.from(joints, (j) => model.jnt_qposadr[j]);
    this.dofs = Int32Array.from(joints, (j) => model.jnt_dofadr[j]);
    this.act = Int32Array.from(acts);
    this.feet = Int32Array.from(feet);
    this.driven = new Set(joints);
    const gain = model.actuator_gainprm, bias = model.actuator_biasprm;
    const { dof_damping: damp, dof_armature: arm, dof_frictionloss: fric } = model;
    this.saved = {
      act: acts.map((a) => [gain[10 * a], bias[10 * a], bias[10 * a + 1], bias[10 * a + 2]]),
      dof: Array.from(this.dofs, (d) => [damp[d], arm[d], fric[d]]),
    };
    acts.forEach((a, i) => {
      if (torque[i]) return;
      gain[10 * a] = s.actuator_gain[i];
      for (let k = 0; k < 3; k++) bias[10 * a + k] = s.actuator_bias[i][k];
    });
    this.dofs.forEach((d, i) => {
      damp[d] = s.dof_damping[i]; arm[d] = s.dof_armature[i]; fric[d] = s.dof_frictionloss[i];
    });
  }

  unbind() {
    const m = this.model;
    if (!m || !this.saved) return;
    const gain = m.actuator_gainprm, bias = m.actuator_biasprm;
    this.act.forEach((a, i) => {
      if (this.torque[i]) return;
      const [g, b0, b1, b2] = this.saved.act[i];
      gain[10 * a] = g; bias[10 * a] = b0; bias[10 * a + 1] = b1; bias[10 * a + 2] = b2;
    });
    this.dofs.forEach((d, i) => {
      [m.dof_damping[d], m.dof_armature[d], m.dof_frictionloss[d]] = this.saved.dof[i];
    });
    this.saved = null;
    this.driven = new Set();
  }

  owns(j) {
    return this.driven?.has(j) ? this.name : null;
  }

  pose(data) {
    this.spec.default_pose.forEach((v, i) => {
      data.qpos[this.qadr[i]] = v;
      if (!this.torque[i]) data.ctrl[this.act[i]] = v;
    });
  }

  // Before every physics step: the trained servo's torque on torque motors.
  substep(data) {
    const s = this.spec, m = this.model;
    this.act.forEach((a, i) => {
      if (!this.torque[i]) return;
      const [b0, b1, b2] = s.actuator_bias[i], [flo, fhi] = s.force_range?.[i] ?? [-1e6, 1e6];
      const f = s.actuator_gain[i] * this.servo[i] + b0 + b1 * data.qpos[this.qadr[i]] + b2 * data.qvel[this.dofs[i]];
      let u = Math.min(fhi, Math.max(flo, f)) / (m.actuator_gainprm[10 * a] * m.actuator_gear[6 * a]);
      if (m.actuator_ctrllimited[a]) u = Math.min(m.actuator_ctrlrange[2 * a + 1], Math.max(m.actuator_ctrlrange[2 * a], u));
      data.ctrl[a] = u;
    });
  }

  // The policy input from `data` (synchronous); advances the error history.
  observe(data) {
    const s = this.spec, n = s.joints.length;
    const { qpos, qvel, xpos, xmat, cvel, geom_xpos: gx } = data;
    const R = xmat.subarray(9 * this.root, 9 * this.root + 9), p0 = xpos.subarray(3 * this.root, 3 * this.root + 3);
    // The sensor frame: the imu site's, else the root body's.
    const S = this.imu >= 0 ? data.site_xmat.subarray(9 * this.imu, 9 * this.imu + 9) : R;
    const ps = this.imu >= 0 ? data.site_xpos.subarray(3 * this.imu, 3 * this.imu + 3) : p0;
    const inS = (w) => [0, 1, 2].map((c) => S[c] * w[0] + S[3 + c] * w[1] + S[6 + c] * w[2]);  // S^T w
    const w = cvel.subarray(6 * this.root, 6 * this.root + 3);
    const q = Float64Array.from(this.qadr, (a) => qpos[a]);
    this.error.copyWithin(n, 0, this.error.length - n);
    for (let i = 0; i < n; i++) this.error[i] = q[i] - this.targets[i];
    const parts = {
      linvel: () => {  // cvel's linear part is at the tree's center of mass; move it to the site
        const v = cvel.subarray(6 * this.root + 3, 6 * this.root + 6), c = data.subtree_com.subarray(3 * this.tree, 3 * this.tree + 3);
        const r = [0, 1, 2].map((k) => ps[k] - c[k]);
        return inS([v[0] + w[1] * r[2] - w[2] * r[1], v[1] + w[2] * r[0] - w[0] * r[2], v[2] + w[0] * r[1] - w[1] * r[0]]);
      },
      gyro: () => inS(w),
      gravity: () => [-S[6], -S[7], -S[8]],
      upvector: () => [S[2], S[5], S[8]],
      joint_vel: () => Array.from(this.dofs, (d) => qvel[d]),
      joint_pos: () => q.map((v, i) => v - s.default_pose[i]),
      qpos_error: () => this.error,
      feet_pos: () => Array.from(this.feet).flatMap((g) => {
        const d = [0, 1, 2].map((k) => gx[3 * g + k] - p0[k]);
        return [0, 1, 2].map((c) => R[c] * d[0] + R[3 + c] * d[1] + R[6 + c] * d[2]);  // R^T d
      }),
      last_action: () => this.prevAction,
      command: () => this.cmd,
    };
    const obs = new Float32Array(s.obs.reduce((t, [, k]) => t + k, 0));
    let o = 0;
    for (const [name, size] of s.obs) {
      const c = name.startsWith('cmd:') && this.commands[name.slice(4)];
      const v = c ? commandVector(c, this.command[name.slice(4)]) : parts[name]?.();
      if (!v || v.length !== size) throw new Error(`${this.name}: cannot build observation part ${name}`);
      obs.set(v, o);
      o += size;
    }
    return obs;
  }

  async infer(obs) {
    const { ort, session } = this.policy;
    return runBatched(ort, session, obs, this.spec.input, this.spec.output);
  }

  apply(data, action) {
    const s = this.spec;
    this.prevAction = this.action;
    this.action = action;
    action.forEach((a, i) => {
      const [lo, hi] = s.ctrl_range[i];
      this.targets[i] = this.servo[i] = Math.min(hi, Math.max(lo, s.default_pose[i] + a * s.action_scale));
      if (!this.torque[i]) data.ctrl[this.act[i]] = this.targets[i];
    });
  }

  async step(data, valid = () => true) {
    const action = await this.infer(this.observe(data));
    if (valid()) this.apply(data, action);
  }
}
