// Joystick locomotion policies trained in MuJoCo Playground (training/,
// MJX + Brax PPO) and exported by training/export_onnx.py: policy.onnx (the
// observation normalization, the MLP and the tanh of the action mean) and
// policy.json, which says everything else, so one controller serves any of
// them:
//
// - joints / actuators: the policy's joints in action order, and the
//   position actuators driving them;
// - default_pose, action_scale: target = default + action * scale, clipped to
//   ctrl_range;
// - actuator_gain / actuator_bias / dof_*: the PD gains, joint damping,
//   armature and friction loss it was trained with, set on bind (and given
//   back on unbind) like GR00T's gains;
// - obs: the observation's parts in order (training/export_onnx.py's
//   Observer is the reference; server/policies.py has the same in Python):
//     gyro         root angular velocity in its own frame at the start of the
//                  last step (R^T cvel, as the gyro sensor computes it)
//     upvector     root z axis in the world frame
//     joint_pos    q - default pose
//     qpos_error   history_len frames of q - last targets, newest first
//     feet_pos     the foot geoms' centers relative to the root, root frame
//     last_action  the raw action before the last one (Playground updates
//                  it after building the observation)
//     command      vx, vy (m/s), yaw rate (rad/s)
// - command_range: what it was trained to follow (the key speeds stay inside).

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
function driveOf(range) {
  const top = (r, side) => +(0.8 * Math.abs(r?.[side] ?? 0)).toFixed(2);
  return {
    forward: top(range.lin_vel_x, 1), back: top(range.lin_vel_x, 0),
    side: Math.min(top(range.lin_vel_y, 0), top(range.lin_vel_y, 1)),
    turn: Math.min(top(range.ang_vel_yaw, 0), top(range.ang_vel_yaw, 1)),
  };
}

export class PlaygroundController {
  constructor(policy, prefix = '', name = 'Playground policy') {
    this.policy = policy;
    this.spec = policy.spec;
    this.prefix = prefix;
    this.name = name;
    this.drive = driveOf(this.spec.command_range);
    this.cmd = [0, 0, 0];
    this.reset();
  }

  reset() {
    const n = this.spec.joints.length;
    this.action = new Float32Array(n);      // applied last
    this.prevAction = new Float32Array(n);  // the one before: what the observation holds
    this.targets = new Float64Array(n);  // Playground starts the error history against zero targets
    this.error = new Float64Array(this.spec.history_len * n);
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
    this.model = model;
    this.root = root;
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
      data.ctrl[this.act[i]] = v;
    });
  }

  // The policy input from `data` (synchronous); advances the error history.
  observe(data) {
    const s = this.spec, n = s.joints.length;
    const { qpos, xpos, xmat, cvel, geom_xpos: gx } = data;
    const R = xmat.subarray(9 * this.root, 9 * this.root + 9), p0 = xpos.subarray(3 * this.root, 3 * this.root + 3);
    const q = Float64Array.from(this.qadr, (a) => qpos[a]);
    this.error.copyWithin(n, 0, this.error.length - n);
    for (let i = 0; i < n; i++) this.error[i] = q[i] - this.targets[i];
    const parts = {
      gyro: () => {
        const w = cvel.subarray(6 * this.root, 6 * this.root + 3);
        return [0, 1, 2].map((c) => R[c] * w[0] + R[3 + c] * w[1] + R[6 + c] * w[2]);  // R^T w
      },
      upvector: () => [R[2], R[5], R[8]],
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
      const v = parts[name]?.();
      if (!v || v.length !== size) throw new Error(`${this.name}: cannot build observation part ${name}`);
      obs.set(v, o);
      o += size;
    }
    return obs;
  }

  async infer(obs) {
    const { ort, session } = this.policy;
    const out = await session.run({ [this.spec.input]: new ort.Tensor('float32', obs, [1, obs.length]) });
    return Float32Array.from(out[this.spec.output].data);
  }

  apply(data, action) {
    const s = this.spec;
    this.prevAction = this.action;
    this.action = action;
    action.forEach((a, i) => {
      const [lo, hi] = s.ctrl_range[i];
      this.targets[i] = Math.min(hi, Math.max(lo, s.default_pose[i] + a * s.action_scale));
      data.ctrl[this.act[i]] = this.targets[i];
    });
  }

  async step(data, valid = () => true) {
    const action = await this.infer(this.observe(data));
    if (valid()) this.apply(data, action);
  }
}
