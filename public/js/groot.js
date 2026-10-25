// GR00T lower-body locomotion for the menagerie Unitree G1 (unitree_g1/g1.xml),
// running in the browser on onnxruntime-web.
//
// A port of lerobot's GrootLocomotionController
// (src/lerobot/robots/unitree_g1/controllers/gr00t_locomotion.py) plus the
// parts of its sim that the policy depends on (lerobot/unitree-g1-mujoco):
//
// - Two ONNX MLPs (516 -> 15), run at 50 Hz on 6 stacked 86-D observation
//   frames: the skills of the directory's policy.json (skills.js), Walk and
//   Balance, which stands in for it while |cmd| < 0.05, as lerobot switches.
// - The policy outputs targets for joints 0-14 (legs + waist) only. The arm
//   actuators keep whatever target they have (the default pose from pose(),
//   or what the Stage panel's control sliders set), PD-held with the
//   policy's gains, like g1_mujoco's arm holds. The policy observes the arms
//   where they actually are.
// - Joints are PD-controlled with the robot's per-joint kp/kd (lerobot's
//   config_unitree_g1). Menagerie's position actuators are exactly a PD loop
//   (force = kp*(ctrl - q) - kd*qdot), so their gains are overwritten in place
//   instead of computing torques by hand; MuJoCo then integrates the damping
//   implicitly. Torque limits come from the joints' actuatorfrcrange, which
//   matches the sim's motor_effort_limit_list.
// - The IMU is the floating base: orientation qpos[3:7] (world, wxyz) and
//   angular velocity qvel[3:6] (pelvis frame), as the sim's bridge reports it.

import { runBatched } from './batch.js';
import { commandDefaults } from './commands.js';
import { Skills, loadSessions, loadSpec } from './skills.js';

export const CONTROL_DT = 0.02;

// Menagerie joint (and actuator) names in G1_29_JointIndex order.
const JOINTS = [
  'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
  'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
  'waist_yaw', 'waist_roll', 'waist_pitch',
  'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
  'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
  'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow',
  'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
].map((n) => `${n}_joint`);
const N = JOINTS.length;  // 29
const NA = 15;            // policy-driven joints: legs + waist

const KP = [150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40, 250, 250, 250,
  50, 50, 80, 80, 40, 40, 40, 50, 50, 80, 80, 40, 40, 40];
const KD = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
  3, 3, 3, 3, 1.5, 1.5, 1.5, 3, 3, 3, 3, 1.5, 1.5, 1.5];

export const DEFAULT_ANGLES = new Float32Array(N);
DEFAULT_ANGLES[0] = DEFAULT_ANGLES[6] = -0.1;  // hip pitch
DEFAULT_ANGLES[3] = DEFAULT_ANGLES[9] = 0.3;   // knee
DEFAULT_ANGLES[4] = DEFAULT_ANGLES[10] = -0.2; // ankle pitch

const ACTION_SCALE = 0.25;
const ANG_VEL_SCALE = 0.25;
const DOF_VEL_SCALE = 0.05;
const CMD_SCALE = [2.0, 2.0, 0.25];
const FRAME = 86, HISTORY = 6;
// Its commands besides the twist (commands.js): the pelvis height it holds.
export const GROOT_COMMANDS = {
  height: { type: 'scalar', label: 'Height', unit: 'm', range: [0.5, 1.0], default: 0.74,
    about: 'the pelvis height it holds, standing and walking (also R/F)' },
};

// Robot files the policy is on for by default: every menagerie G1 variant
// (its POLICIES entry in policy.js). Any robot can switch it on in the
// Teleop panel; bind() fails if the G1's joints and actuators are not there.
export const G1_MODELS = /(^|\/)mujoco_menagerie\/unitree_g1\/[^/]+\.xml$/;

// {ort, sessions: {skill: session}, spec} from the directory's policy.json.
// `source(name)` returns a file's URL (or bytes).
export async function loadGrootPolicy(ort, source) {
  const spec = await loadSpec(source);
  return { ort, sessions: await loadSessions(ort, source, spec.skills), spec };
}

export class GrootController {
  // `policy` from loadGrootPolicy (shared by every robot); `prefix` is the
  // robot's name prefix in the world model ('' for a standalone model).
  constructor(policy, prefix = '') {
    this.policy = policy;
    this.prefix = prefix;
    this.name = 'GR00T locomotion policy';
    this.drive = { forward: 0.5, back: 0.3, side: 0.3, turn: 0.6 };  // keyboard speeds
    this.cmd = [0, 0, 0];  // vx, vy (m/s), yaw rate (rad/s)
    this.skills = new Skills(policy.spec.skills);
    this.commands = GROOT_COMMANDS;
    this.command = commandDefaults(GROOT_COMMANDS);
    this.reset();
  }

  // Skills (skills.js): what the panel shows and picks.
  get skillMenu() { return this.skills.menu; }
  get skill() { return this.skills.want; }
  setSkill(name) { this.skills.request(name); }
  get activeSkill() { return this.skills.shown; }

  // The commanded base height (R/F keys).
  get height() { return this.command.height; }
  set height(v) { this.command.height = v; }

  reset() {
    this.action = new Float32Array(NA);
    this.history = new Float32Array(FRAME * HISTORY);
    this.skills.reset();
  }

  // Finds the robot's joints and actuators in `model` and gives the actuators
  // the policy's PD gains. Call again whenever the model is recompiled.
  bind(mujoco, model) {
    const id = (type, name) => {
      const i = mujoco.mj_name2id(model, type.value, this.prefix + name);  // takes the enum as an int
      if (i < 0) throw new Error(`GR00T: no ${this.prefix}${name} in the model`);
      return i;
    };
    const T = mujoco.mjtObj;
    this.qadr = new Int32Array(N);
    this.vadr = new Int32Array(N);
    this.act = new Int32Array(N);
    this.driven = new Set();  // the policy's joints (legs + waist), off limits to arm IK
    const found = [];
    for (let i = 0; i < N; i++) { id(T.mjOBJ_JOINT, JOINTS[i]); id(T.mjOBJ_ACTUATOR, JOINTS[i]); }  // all there, or throw first
    this.model = model;
    this.savedGains = found;  // [actuator, gainprm[0], biasprm[0..2]] to give back in unbind()
    for (let i = 0; i < N; i++) {
      const j = id(T.mjOBJ_JOINT, JOINTS[i]);
      if (i < NA) this.driven.add(j);
      this.qadr[i] = model.jnt_qposadr[j];
      this.vadr[i] = model.jnt_dofadr[j];
      const a = id(T.mjOBJ_ACTUATOR, JOINTS[i]);
      this.act[i] = a;
      const gain = model.actuator_gainprm, bias = model.actuator_biasprm;  // mjNGAIN = mjNBIAS = 10
      found.push([a, gain[10 * a], bias[10 * a], bias[10 * a + 1], bias[10 * a + 2]]);
      gain[10 * a] = KP[i];
      bias[10 * a] = 0;
      bias[10 * a + 1] = -KP[i];
      bias[10 * a + 2] = -KD[i];
    }
    const base = id(T.mjOBJ_JOINT, 'floating_base_joint');
    this.baseQ = model.jnt_qposadr[base];
    this.baseV = model.jnt_dofadr[base];
  }

  // Gives the actuators their own gains back (the policy is switched off).
  unbind() {
    const gain = this.model?.actuator_gainprm, bias = this.model?.actuator_biasprm;
    for (const [a, g, b0, b1, b2] of this.savedGains ?? []) {
      gain[10 * a] = g;
      bias[10 * a] = b0; bias[10 * a + 1] = b1; bias[10 * a + 2] = b2;
    }
    this.savedGains = null;
    this.driven = new Set();
  }

  // This controller's name if it drives joint `j` (for ik.js), else null.
  owns(j) {
    return this.driven?.has(j) ? 'GR00T' : null;
  }

  // Puts the joints in the default pose and holds them there (for a fresh spawn).
  pose(data) {
    for (let i = 0; i < N; i++) {
      data.qpos[this.qadr[i]] = DEFAULT_ANGLES[i];
      data.ctrl[this.act[i]] = DEFAULT_ANGLES[i];
    }
  }

  // Appends the current observation to the history and returns the stacked
  // 516-D policy input. Reads `data` synchronously.
  observe(data) {
    const { qpos, qvel } = data;
    const [qw, qx, qy, qz] = qpos.subarray(this.baseQ + 3, this.baseQ + 7);
    const w = qvel.subarray(this.baseV + 3, this.baseV + 6);
    const f = new Float32Array(FRAME);
    for (let k = 0; k < 3; k++) f[k] = this.cmd[k] * CMD_SCALE[k];
    f[3] = this.height;
    // f[4..6]: orientation command, always zero.
    for (let k = 0; k < 3; k++) f[7 + k] = w[k] * ANG_VEL_SCALE;
    // Gravity in the pelvis frame (lerobot's get_gravity_orientation).
    f[10] = 2 * (-qz * qx + qw * qy);
    f[11] = -2 * (qz * qy + qw * qx);
    f[12] = 1 - 2 * (qw * qw + qz * qz);
    for (let i = 0; i < N; i++) {
      f[13 + i] = qpos[this.qadr[i]] - DEFAULT_ANGLES[i];
      f[42 + i] = qvel[this.vadr[i]] * DOF_VEL_SCALE;
    }
    f.set(this.action, 71);
    this.history.copyWithin(0, FRAME);  // oldest frame first
    this.history.set(f, FRAME * (HISTORY - 1));
    return this.history.slice();
  }

  // Runs the skill for `time` on an observed input; resolves to the new action.
  async infer(obs, time) {
    const session = this.policy.sessions[this.skills.pick(this, time)];
    this.action = await runBatched(this.policy.ort, session, obs);
    return this.action;
  }

  // Writes the joint targets for `action` (legs + waist); the arm targets
  // are left to the user.
  apply(data, action = this.action) {
    for (let i = 0; i < NA; i++) data.ctrl[this.act[i]] = DEFAULT_ANGLES[i] + action[i] * ACTION_SCALE;
  }

  // One control tick: observe, infer, apply. `valid()` is checked after the
  // inference so a world swapped in meanwhile is not written to.
  async step(data, valid = () => true) {
    const action = await this.infer(this.observe(data), data.time);
    if (valid()) this.apply(data, action);
  }
}
