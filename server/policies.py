"""Trained policies on the server: the ones of public/js/policy.js's POLICIES
that have a port here (same ids, so the Teleop panel's choice carries
over), running on onnxruntime. A policy controller has the interface
policy.js describes (bind, unbind, owns, pose, reset, step; locomotion ones
also cmd, drive; any of them commands and command, see
public/js/commands.js, of which command_default, clamp_command and
command_vector here are ports), with a synchronous step.

GrootController is a port of public/js/groot.js (see there for the
observation, the two MLPs and the PD gains); PlaygroundController of
public/js/playground.js (policies from training/: policy.onnx + policy.json);
MicroduckController and BamServos of public/js/microduck.js (microduck_rl's
policies on the BAM servo model: scripts/export_microduck.py).
"""

import copy
import json
import math
import pathlib
import re

import mujoco
import numpy as np

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
POLICY_ROOT = PUBLIC / 'pretrained_policies'
OBJ = mujoco.mjtObj
# As a plain int: numpy compares an array with a MuJoCo enum one element at a time, in Python.
FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)

# ---------------------------------------------------------------------------
# Commands besides the twist (public/js/commands.js)

COMMAND_TYPES = ('scalar', 'vector', 'enum', 'bool')


def _clamp(v, r=None):
  return min(r[1], max(r[0], v)) if r else v


def command_default(d):
  if 'default' in d:
    return copy.deepcopy(d['default'])
  t = d.get('type')
  if t == 'scalar':
    return _clamp(0.0, d.get('range'))
  if t == 'vector':
    rng = d.get('range') or [None] * len(d.get('dims', []))
    return [_clamp(0.0, r) for r in rng]
  if t == 'enum':
    return (d.get('values') or [''])[0]
  if t == 'bool':
    return False
  return None


def command_defaults(defs):
  return {k: command_default(d) for k, d in (defs or {}).items()}


def _number(x):
  n = float(x)
  if not math.isfinite(n):
    raise ValueError(f'not a number: {x}')
  return n


def clamp_command(d, v):
  """`v` made a valid value of the command; ValueError if it cannot be one."""
  t = d.get('type')
  if t == 'scalar':
    return _clamp(_number(v), d.get('range'))
  if t == 'vector':
    n = len(command_default(d))
    if not isinstance(v, list) or len(v) != n:
      raise ValueError(f'expected {n} numbers')
    rng = d.get('range') or [None] * n
    return [_clamp(_number(x), rng[i]) for i, x in enumerate(v)]
  if t == 'enum':
    if v not in d['values']:
      raise ValueError(f"not one of {', '.join(map(str, d['values']))}")
    return v
  if t == 'bool':
    return bool(v)
  return v


def command_vector(d, v):
  """The command's part of an observation (cmd:<name>)."""
  t = d.get('type')
  if t == 'scalar':
    return np.array([v], float)
  if t == 'vector':
    return np.asarray(v, float)
  if t == 'enum':
    return np.array([float(x == v) for x in d['values']])
  if t == 'bool':
    return np.array([float(bool(v))])
  raise ValueError(f'a {t} command cannot be observed')


def observation(spec, parts, commands, command):
  """The observation spec['obs'] lists, from `parts` ({name: () -> values})
  and the commands (cmd:<name>)."""
  out = []
  for name, size in spec['obs']:
    key = name[4:] if name.startswith('cmd:') else None
    v = command_vector(commands[key], command[key]) if key in commands else np.asarray(parts[name](), float)
    if v.size != size:
      raise ValueError(f'observation part {name}: {v.size} values, not {size}')
    out.append(v.ravel())
  return np.concatenate(out).astype(np.float32)[None]


# ---------------------------------------------------------------------------
# GR00T locomotion, Unitree G1

JOINTS = [f'{n}_joint' for n in (
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
    'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow',
    'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw')]
N, NA = len(JOINTS), 15
KP = np.array([150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40, 250, 250, 250,
               50, 50, 80, 80, 40, 40, 40, 50, 50, 80, 80, 40, 40, 40], float)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
               3, 3, 3, 3, 1.5, 1.5, 1.5, 3, 3, 3, 3, 1.5, 1.5, 1.5], float)
DEFAULT_ANGLES = np.zeros(N, np.float32)
DEFAULT_ANGLES[[0, 6]] = -0.1
DEFAULT_ANGLES[[3, 9]] = 0.3
DEFAULT_ANGLES[[4, 10]] = -0.2
ACTION_SCALE, ANG_VEL_SCALE, DOF_VEL_SCALE = 0.25, 0.25, 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], np.float32)
FRAME, HISTORY = 86, 6
GROOT_COMMANDS = {  # as groot.js
    'height': {'type': 'scalar', 'label': 'Height', 'unit': 'm', 'range': [0.5, 1.0], 'default': 0.74,
               'about': 'the pelvis height it holds, standing and walking (also R/F)'},
}


class GrootController:
  def __init__(self, sessions, prefix):
    self.balance, self.walk = sessions
    self.prefix = prefix
    self.name = 'GR00T locomotion policy'
    self.drive = {'forward': 0.5, 'back': 0.3, 'side': 0.3, 'turn': 0.6}
    self.cmd = [0.0, 0.0, 0.0]
    self.commands = GROOT_COMMANDS
    self.command = command_defaults(GROOT_COMMANDS)
    self.model = None
    self.saved_gains = None
    self.driven = set()
    self.reset()

  @property
  def height(self):
    return self.command['height']

  @height.setter
  def height(self, v):
    self.command['height'] = v

  def reset(self):
    self.action = np.zeros(NA, np.float32)
    self.history = np.zeros(FRAME * HISTORY, np.float32)

  def bind(self, model):
    def find(obj, name):
      i = mujoco.mj_name2id(model, obj, self.prefix + name)
      if i < 0:
        raise ValueError(f'GR00T: no {self.prefix}{name} in the model')
      return i
    for n in JOINTS:  # all there, or fail before changing anything
      find(OBJ.mjOBJ_JOINT, n)
      find(OBJ.mjOBJ_ACTUATOR, n)
    self.model = model
    self.qadr, self.vadr, self.act = np.zeros(N, int), np.zeros(N, int), np.zeros(N, int)
    self.driven = set()
    self.saved_gains = []
    for i, n in enumerate(JOINTS):
      j, a = find(OBJ.mjOBJ_JOINT, n), find(OBJ.mjOBJ_ACTUATOR, n)
      if i < NA:
        self.driven.add(j)
      self.qadr[i], self.vadr[i], self.act[i] = model.jnt_qposadr[j], model.jnt_dofadr[j], a
      self.saved_gains.append((a, model.actuator_gainprm[a, 0], model.actuator_biasprm[a, :3].copy()))
      model.actuator_gainprm[a, 0] = KP[i]
      model.actuator_biasprm[a, :3] = (0, -KP[i], -KD[i])
    base = find(OBJ.mjOBJ_JOINT, 'floating_base_joint')
    self.base_q, self.base_v = model.jnt_qposadr[base], model.jnt_dofadr[base]

  def unbind(self):
    for a, g, b in self.saved_gains or []:
      self.model.actuator_gainprm[a, 0] = g
      self.model.actuator_biasprm[a, :3] = b
    self.saved_gains = None
    self.driven = set()

  def owns(self, j):
    return 'GR00T' if j in self.driven else None

  def pose(self, data):
    data.qpos[self.qadr] = DEFAULT_ANGLES
    data.ctrl[self.act] = DEFAULT_ANGLES

  def observe(self, data):
    qw, qx, qy, qz = data.qpos[self.base_q + 3:self.base_q + 7]
    f = np.zeros(FRAME, np.float32)
    f[0:3] = np.array(self.cmd, np.float32) * CMD_SCALE
    f[3] = self.height
    f[7:10] = data.qvel[self.base_v + 3:self.base_v + 6] * ANG_VEL_SCALE
    f[10] = 2 * (-qz * qx + qw * qy)
    f[11] = -2 * (qz * qy + qw * qx)
    f[12] = 1 - 2 * (qw * qw + qz * qz)
    f[13:42] = data.qpos[self.qadr] - DEFAULT_ANGLES
    f[42:71] = data.qvel[self.vadr] * DOF_VEL_SCALE
    f[71:86] = self.action
    self.history[:-FRAME] = self.history[FRAME:]
    self.history[-FRAME:] = f
    return self.history[None, :].copy()

  def step(self, data):
    session = self.balance if math.hypot(*self.cmd) < 0.05 else self.walk
    out = session.run(None, {session.get_inputs()[0].name: self.observe(data)})
    self.action = np.asarray(out[0], np.float32).reshape(-1)[:NA]
    data.ctrl[self.act[:NA]] = DEFAULT_ANGLES[:NA] + self.action * ACTION_SCALE

  def describe(self):
    return {'name': self.name, 'drive': self.drive, 'commands': self.commands, 'command': self.command}


def load_groot(directory):
  import onnxruntime as ort
  opts = ort.SessionOptions()
  opts.intra_op_num_threads = 1  # tiny MLPs, many robots: no thread pools
  open_ = lambda n: ort.InferenceSession(str(directory / n), opts, providers=['CPUExecutionProvider'])
  return open_('GR00T-WholeBodyControl-Balance.onnx'), open_('GR00T-WholeBodyControl-Walk.onnx')


# ---------------------------------------------------------------------------
# MuJoCo Playground joystick policies (training/export_onnx.py)


def drive_of(r, low=None):
  """Key speeds a bit inside the trained command ranges, and the dead band
  (minSpeed / minTurn: the goal marker's least command, public/js/goal.js)."""
  top = lambda v: round(0.8 * abs(v), 2)
  d = {'forward': top(r['lin_vel_x'][1]), 'back': top(r['lin_vel_x'][0]),
       'side': min(top(r['lin_vel_y'][0]), top(r['lin_vel_y'][1])),
       'turn': min(top(r['ang_vel_yaw'][0]), top(r['ang_vel_yaw'][1]))}
  if low:
    d.update(minSpeed=low['speed'], minTurn=low['turn'])
  return d


class PlaygroundController:
  """See public/js/playground.js: position servos get the trained gains, torque
  motors the trained servo's torque every physics step (substep); sensor
  values are the forward pass's at the start of the last step, in the imu
  site's frame (the root body's without one)."""

  def __init__(self, policy, prefix, name='Playground policy'):
    self.session, self.spec = policy
    self.prefix = prefix
    self.name = name
    self.drive = drive_of(self.spec['command_range'], self.spec.get('min_command'))
    self.cmd = [0.0, 0.0, 0.0]
    self.commands = self.spec.get('commands', {})
    self.command = command_defaults(self.commands)
    self.model = None
    self.saved = None
    self.driven = set()
    self.default = np.array(self.spec['default_pose'])
    self.reset()

  def reset(self):
    n = len(self.spec['joints'])
    self.action = np.zeros(n, np.float32)  # applied last
    self.prev_action = np.zeros(n, np.float32)  # the one before: what the observation holds
    self.targets = np.zeros(n)  # Playground starts the error history against zero targets
    self.error = np.zeros(self.spec['history_len'] * n)
    self.servo = self.default.copy()  # what torque motors hold until the first action

  def bind(self, model):
    s = self.spec

    def find(obj, name):
      i = mujoco.mj_name2id(model, obj, self.prefix + name)
      if i < 0:
        raise ValueError(f'{self.name}: no {self.prefix}{name} in the model')
      return i
    joints = [find(OBJ.mjOBJ_JOINT, n) for n in s['joints']]
    acts = np.array([find(OBJ.mjOBJ_ACTUATOR, n) for n in s['actuators']])
    feet = np.array([find(OBJ.mjOBJ_GEOM, n) for n in s['feet']], int)
    root = find(OBJ.mjOBJ_BODY, s['root'])
    free = model.body_jntadr[root]
    if free < 0 or model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError(f"{self.name}: {self.prefix}{s['root']} has no free joint")
    AFFINE, FIXED, NONE = mujoco.mjtBias.mjBIAS_AFFINE, mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
    torque = np.zeros(len(acts), bool)
    for i, a in enumerate(acts):
      if model.actuator_biastype[a] == AFFINE:
        continue
      if model.actuator_gaintype[a] == FIXED and model.actuator_biastype[a] == NONE:
        torque[i] = True
      else:
        raise ValueError(f"{self.name}: actuator {s['actuators'][i]} is neither a position servo nor a motor")
    self.model, self.root, self.act, self.feet, self.torque = model, root, acts, feet, torque
    self.tree = model.body_rootid[root]  # cvel is about its subtree's center of mass
    self.imu = find(OBJ.mjOBJ_SITE, s['imu']) if s.get('imu') else None
    self.qadr = model.jnt_qposadr[joints]
    self.dofs = model.jnt_dofadr[joints]
    self.driven = set(joints)
    servos = acts[~torque]
    self.saved = (servos, model.actuator_gainprm[servos, 0].copy(), model.actuator_biasprm[servos, :3].copy(),
                  model.dof_damping[self.dofs].copy(), model.dof_armature[self.dofs].copy(),
                  model.dof_frictionloss[self.dofs].copy())
    model.actuator_gainprm[servos, 0] = np.array(s['actuator_gain'])[~torque]
    model.actuator_biasprm[servos, :3] = np.array(s['actuator_bias'])[~torque]
    model.dof_damping[self.dofs] = s['dof_damping']
    model.dof_armature[self.dofs] = s['dof_armature']
    model.dof_frictionloss[self.dofs] = s['dof_frictionloss']
    self.lo, self.hi = np.array(s['ctrl_range']).T
    self.gain = np.array(s['actuator_gain'])
    self.bias = np.array(s['actuator_bias'])
    self.flo, self.fhi = np.array(s.get('force_range', [[-1e6, 1e6]] * len(acts))).T
    # torque -> ctrl of the motors
    self.per_ctrl = model.actuator_gainprm[acts, 0] * model.actuator_gear[acts, 0]

  def unbind(self):
    if self.saved is None:
      return
    m = self.model
    servos, *vals = self.saved
    (m.actuator_gainprm[servos, 0], m.actuator_biasprm[servos, :3], m.dof_damping[self.dofs],
     m.dof_armature[self.dofs], m.dof_frictionloss[self.dofs]) = vals
    self.saved = None
    self.driven = set()

  def owns(self, j):
    return self.name if j in self.driven else None

  def pose(self, data):
    data.qpos[self.qadr] = self.default
    data.ctrl[self.act[~self.torque]] = self.default[~self.torque]

  def substep(self, data):
    """Before every physics step: the trained servo's torque on torque motors."""
    if not self.torque.any():
      return
    t = self.torque
    f = self.gain * self.servo + self.bias[:, 0] + self.bias[:, 1] * data.qpos[self.qadr] \
        + self.bias[:, 2] * data.qvel[self.dofs]
    u = np.clip(f, self.flo, self.fhi) / self.per_ctrl
    m, a = self.model, self.act[t]
    lim = m.actuator_ctrllimited[a].astype(bool)
    u = u[t]
    u[lim] = np.clip(u[lim], m.actuator_ctrlrange[a[lim], 0], m.actuator_ctrlrange[a[lim], 1])
    data.ctrl[a] = u

  def observe(self, data):
    n = len(self.default)
    R = data.xmat[self.root].reshape(3, 3)
    S, p = ((data.site_xmat[self.imu].reshape(3, 3), data.site_xpos[self.imu]) if self.imu is not None
            else (R, data.xpos[self.root]))
    w, v0 = data.cvel[self.root, :3], data.cvel[self.root, 3:]
    q = data.qpos[self.qadr]
    self.error = np.roll(self.error, n)
    self.error[:n] = q - self.targets
    parts = {
        'linvel': lambda: (v0 + np.cross(w, p - data.subtree_com[self.tree])) @ S,
        'gyro': lambda: w @ S,
        'gravity': lambda: -S[2],
        'upvector': lambda: S[:, 2],
        'joint_pos': lambda: q - self.default,
        'joint_vel': lambda: data.qvel[self.dofs],
        'qpos_error': lambda: self.error,
        'feet_pos': lambda: ((data.geom_xpos[self.feet] - data.xpos[self.root]) @ R).ravel(),
        'last_action': lambda: self.prev_action,
        'command': lambda: np.asarray(self.cmd, float),
    }
    return observation(self.spec, parts, self.commands, self.command)

  def step(self, data):
    s = self.spec
    self.prev_action = self.action
    self.action = np.asarray(self.session.run(None, {s['input']: self.observe(data)})[0], np.float32).reshape(-1)
    self.targets = np.clip(self.default + self.action * s['action_scale'], self.lo, self.hi)
    self.servo = self.targets.copy()
    data.ctrl[self.act[~self.torque]] = self.targets[~self.torque]

  def describe(self):
    return {'name': self.name, 'drive': self.drive, 'commands': self.commands, 'command': self.command}


def load_playground(directory):
  import onnxruntime as ort
  spec = json.loads((directory / 'policy.json').read_text())
  opts = ort.SessionOptions()
  opts.intra_op_num_threads = 1
  return ort.InferenceSession(str(directory / spec['onnx']), opts, providers=['CPUExecutionProvider']), spec


# ---------------------------------------------------------------------------
# Microduck (microduck_rl, mjlab): scripts/export_microduck.py


class BamServos:
  """XL330 servos as BAM models them (see public/js/microduck.js): the firmware's
  P loop gives a duty cycle, the DC motor a torque, and the M6 friction budget
  goes into the joints' friction loss and damping, where MuJoCo's solver
  applies it. update() runs before every physics step, as one function
  compiled with numba (_bam_update): in numpy, the calls' own overhead
  (some 0.5 µs each, whatever the size) cost more than the physics step of
  a microduck. join() makes one bank of the servos of several robots."""

  def __init__(self, p, model, acts, dofs, qadr, sizes=None):
    self.p, self.model, self.acts, self.dofs, self.qadr = p, model, acts, dofs, qadr
    self.sizes = sizes or [len(acts)]
    self.starts = np.cumsum([0, *self.sizes[:-1]])  # each robot's first servo
    self.robot = np.repeat(np.arange(len(self.sizes)), self.sizes)
    self.target = np.zeros(len(acts))
    self.prev = np.zeros(len(acts))  # last motor torques: the battery sag (per robot)
    self.consts = np.array([{**p, 'gain': p['kp'] * p['error_gain'], 'max_current': p.get('max_current') or 0}[k]
                            for k in BAM_CONSTS], float)
    self.data = None  # the data the views below are of (bind)

  @staticmethod
  def join(banks):
    cat = lambda f: np.concatenate([getattr(b, f) for b in banks])
    out = BamServos(banks[0].p, banks[0].model, cat('acts'), cat('dofs'), cat('qadr'),
                    [n for b in banks for n in b.sizes])
    out.target, out.prev = cat('target'), cat('prev')
    return out

  def split(self, banks):
    """Hands the state back to the banks join() made this one of."""
    at = 0
    for b in banks:
      n = len(b.acts)
      b.target, b.prev = self.target[at:at + n].copy(), self.prev[at:at + n].copy()
      at += n

  def bind(self, data):
    """The views update() works on, of `data` and the model. The damping is
    a constant: set here, once."""
    m, d = self.model, self.dofs
    self.data = data
    self.v = (data.qpos, data.qvel, data.ctrl, data.qfrc_bias, data.qfrc_constraint, data.qfrc_actuator,
              m.dof_frictionloss)
    m.dof_damping[d] = self.p['friction_viscous']

  def update(self, data):
    if data is not self.data:
      self.bind(data)
    self.prev = _bam_update(self.consts, *self.v, self.qadr, self.dofs, self.acts, self.target, self.prev,
                            self.starts, self.robot, data.efc_type, data.efc_id, data.efc_force, FRICTION_DOF)


# BamServos.consts, in this order.
BAM_CONSTS = ['kt', 'R', 'vin', 'vin_drop_gain', 'vin_min', 'gain',  # gain: kp error_gain
              'max_pwm', 'max_current',
              'dtheta_stribeck', 'alpha', 'friction_base', 'friction_stribeck',
              'load_friction_external', 'load_friction_motor', 'load_friction_external_stribeck',
              'load_friction_motor_stribeck', 'load_friction_external_quad', 'load_friction_motor_quad']


def _bam_kernel(c, qpos, qvel, ctrl, bias, cons, act, loss_out, qadr, dofs, acts, target, prev, starts, robot,
                efc_type, efc_id, efc_force, friction_dof):
  """BamServos.update, compiled by numba (without it, as Python: slower, but
  the same): returns the new motor torques."""
  (kt, R, vin0, drop, vin_min, gain, max_pwm, max_current, dth, alpha,
   fb, fs, lfe, lfm, lfes, lfms, lfeq, lfmq) = (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
                                               c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17])
  n = len(acts)
  sag = np.zeros(len(starts))
  for i in range(n):
    sag[robot[i]] += abs(prev[i])
  # Our own joint friction, by dof.
  fric = np.zeros(len(qvel))
  for r in range(len(efc_type)):
    if efc_type[r] == friction_dof:
      fric[efc_id[r]] += efc_force[r]
  tau = np.empty(n)
  for i in range(n):
    d = dofs[i]
    q, dq = qpos[qadr[i]], qvel[d]
    vin = max(vin0 - drop * sag[robot[i]], vin_min)
    duty = (target[i] - q) * gain
    if max_current:
      center, span = kt * dq / vin, R * max_current / vin
      duty = min(max(duty, center - span), center + span)
    duty = min(max(duty, -max_pwm), max_pwm)
    t = (kt / R) * (vin * duty - kt * dq)
    ctrl[acts[i]] = t
    tau[i] = t
    ext = cons[d] - bias[d] - fric[d]
    mot = act[d]
    s = math.exp(-(abs(dq) / dth) ** alpha)
    loss = (fb + s * fs + abs(ext * lfe - mot * lfm) + s * abs(ext * lfes - mot * lfms))
    loss += s * (lfeq * ext * ext if abs(mot) > abs(ext) else lfmq * mot * mot)
    loss_out[d] = loss
  return tau


try:
  import numba
  _bam_update = numba.njit(cache=True)(_bam_kernel)  # compiled on first use (~1 s), then cached in __pycache__
except ImportError:
  print('numba not installed: the Microduck servos run as Python, slower')
  _bam_update = _bam_kernel


class Substeps:
  """The controllers' substep() before every physics step of one
  World.advance: the Microducks' servos of one spec as one bank. done() hands
  the banks' state back to the controllers."""

  def __init__(self, controllers):
    self.others = [c for c in controllers if hasattr(c, 'substep') and not isinstance(c, MicroduckController)]
    self.ducks = [c for c in controllers if isinstance(c, MicroduckController)]
    groups = {}
    for c in self.ducks:
      groups.setdefault(id(c.spec['bam']), []).append(c)
    self.banks, self.slots = [], []  # slots: (controller, its bank, its servos there)
    for g in groups.values():
      bank = BamServos.join([c.servos for c in g])
      self.banks.append((bank, g))
      at = 0
      for c in g:
        n = len(c.servos.acts)
        self.slots.append((c, bank, slice(at, at + n)))
        at += n

  def step(self, data):
    for c in self.others:
      c.substep(data)  # software servos
    for c, bank, at in self.slots:
      t = c.due(data)
      if t is not None:
        bank.target[at] = t
    for bank, _ in self.banks:
      bank.update(data)

  def done(self):
    for bank, g in self.banks:
      bank.split([c.servos for c in g])


def skill_command(skills):
  """The command that switches skills, if there is more than one (see
  public/js/microduck.js)."""
  names = list(skills)
  if len(names) < 2:
    return None
  about = ' · '.join(f"{k}: {skills[k].get('about') or skills[k].get('label') or k}" for k in names[1:])
  return {'type': 'enum', 'label': 'Skill', 'values': names, 'default': names[0],
          'about': f'{about} (a timed one goes back to {names[0]}; K: {names[1]})'}


class MicroduckController:
  """See public/js/microduck.js: the microduck_rl policies on the BAM servo
  model, walking and the skills it switches to for a while."""

  def __init__(self, policy, prefix, name='Microduck walking policy'):
    self.sessions, self.spec = policy
    self.prefix = prefix
    self.name = name
    self.drive = drive_of(self.spec['command_range'], self.spec.get('min_command'))
    self.cmd = [0.0, 0.0, 0.0]
    self.skills = self.spec['skills']
    skill = skill_command(self.skills)
    self.commands = {**self.spec.get('commands', {}), **({'skill': skill} if skill else {})}
    self.command = command_defaults(self.commands)
    self.model = None
    self.saved = None
    self.driven = set()
    self.default = np.array(self.spec['default_pose'])
    self.reset()

  def reset(self):
    n = len(self.default)
    self.action = np.zeros(n, np.float32)
    self.qvel = np.zeros(n)  # the joint velocities of the last tick: the observation's
    self.pending = []        # [(time, targets)]: commands reach the servos after the delay
    self.running = None      # the skill of the last tick, and since when: (name, data.time)
    if 'skill' in self.commands:
      self.command['skill'] = self.commands['skill']['default']

  def bind(self, model):
    s, b = self.spec, self.spec['bam']

    def find(obj, name):
      i = mujoco.mj_name2id(model, obj, self.prefix + name)
      if i < 0:
        raise ValueError(f'{self.name}: no {self.prefix}{name} in the model')
      return i
    joints = [find(OBJ.mjOBJ_JOINT, n) for n in s['joints']]
    acts = np.array([find(OBJ.mjOBJ_ACTUATOR, n) for n in s['actuators']])
    root = find(OBJ.mjOBJ_BODY, s['root'])
    free = model.body_jntadr[root]
    if free < 0 or model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError(f"{self.name}: {self.prefix}{s['root']} has no free joint")
    self.model, self.root, self.act = model, root, acts
    self.free_v = model.jnt_dofadr[free]
    self.qadr = model.jnt_qposadr[joints]
    self.dofs = model.jnt_dofadr[joints]
    self.driven = set(joints)
    d = self.dofs
    # Motors driven by BAM in place of the position servos, as bam.mjlab does in training.
    self.saved = (model.actuator_gainprm[acts].copy(), model.actuator_biasprm[acts].copy(),
                  model.actuator_ctrllimited[acts].copy(), model.actuator_forcelimited[acts].copy(),
                  model.actuator_forcerange[acts].copy(), model.dof_damping[d].copy(),
                  model.dof_armature[d].copy(), model.dof_frictionloss[d].copy(),
                  model.dof_solref[d].copy(), model.dof_solimp[d].copy())
    model.actuator_gainprm[acts] = 0
    model.actuator_gainprm[acts, 0] = 1 / model.actuator_gear[acts, 0]
    model.actuator_biasprm[acts] = 0
    model.actuator_ctrllimited[acts] = 0
    model.actuator_forcelimited[acts] = 1
    model.actuator_forcerange[acts] = (-b['force_limit'], b['force_limit'])
    model.dof_armature[d] = b['armature']
    model.dof_solref[d] = b['solref_friction']
    model.dof_solimp[d] = b['solimp_friction']
    self.servos = BamServos(b, model, acts, d, self.qadr)
    self.servos.target = self.default.copy()

  def unbind(self):
    if self.saved is None:
      return
    m, a, d = self.model, self.act, self.dofs
    (m.actuator_gainprm[a], m.actuator_biasprm[a], m.actuator_ctrllimited[a], m.actuator_forcelimited[a],
     m.actuator_forcerange[a], m.dof_damping[d], m.dof_armature[d], m.dof_frictionloss[d],
     m.dof_solref[d], m.dof_solimp[d]) = self.saved
    self.saved = None
    self.driven = set()

  def owns(self, j):
    return self.name if j in self.driven else None

  def pose(self, data):
    data.qpos[self.qadr] = self.default
    self.servos.target = self.default.copy()

  def due(self, data):
    """The servo targets whose delay is up by now, if any."""
    t = None
    while self.pending and data.time >= self.pending[0][0] - 1e-9:
      t = self.pending.pop(0)[1]
    return t

  def substep(self, data):  # one robot alone (Substeps does them all)
    t = self.due(data)
    if t is not None:
      self.servos.target = t
    self.servos.update(data)

  def skill(self, data):
    """The skill to run at data.time: the one asked for, back to the first
    once it has run its duration."""
    first = next(iter(self.skills))
    want = self.command.get('skill', first)
    if self.running is None or want != self.running[0]:
      self.running = (want, data.time)
    d = self.skills[want].get('duration')
    if d and data.time - self.running[1] >= d - 1e-9:
      self.command['skill'] = first
      self.running = (first, data.time)
    return self.running[0]

  def observe(self, data, skill=None):
    R = data.xmat[self.root].reshape(3, 3)
    v = self.free_v
    parts = {
        'base_ang_vel': lambda: data.qvel[v + 3:v + 6],
        'projected_gravity': lambda: -R[2],
        'joint_pos': lambda: data.qpos[self.qadr] - self.default,
        'joint_vel': lambda: self.qvel,
        'last_action': lambda: self.action,
        'command': lambda: self.cmd,
    }
    blind = bool(skill) and self.skills[skill].get('commands') is False  # trained with zeros there
    if blind:
      for name, size in self.spec['obs']:
        if name == 'command' or name.startswith('cmd:'):
          parts[name] = lambda size=size: np.zeros(size)
    obs = observation(self.spec, parts, {} if blind else self.commands, self.command)
    self.qvel = data.qvel[self.dofs].copy()
    return obs

  def step(self, data):
    s = self.spec
    skill = self.skill(data)
    obs = self.observe(data, skill)
    self.action = np.asarray(self.sessions[skill].run(None, {s['input']: obs})[0], np.float32).reshape(-1)
    self.pending.append((data.time + s['bam']['delay'], self.default + self.action * s['action_scale']))

  def describe(self):
    return {'name': self.name, 'drive': self.drive, 'commands': self.commands, 'command': self.command}


def load_microduck(directory):
  import onnxruntime as ort
  spec = json.loads((directory / 'policy.json').read_text())
  spec.setdefault('skills', {'walk': {'onnx': 'walk.onnx', 'label': 'Walk'}})
  opts = ort.SessionOptions()
  opts.intra_op_num_threads = 1
  sessions = {k: ort.InferenceSession(str(directory / sk['onnx']), opts, providers=['CPUExecutionProvider'])
              for k, sk in spec['skills'].items()}
  return sessions, spec


# ---------------------------------------------------------------------------
# Registry (ids as in policy.js)

POLICIES = [
    {'id': 'groot-g1', 'name': 'GR00T locomotion · Unitree G1', 'dir': 'GR00T-WholeBodyControl_g1',
     'defaultFor': re.compile(r'(^|/)mujoco_menagerie/unitree_g1/[^/]+\.xml$'),
     'load': load_groot, 'create': GrootController},
    {'id': 'playground-spot', 'name': 'Walking · Boston Dynamics Spot', 'dir': 'playground_spot',
     'defaultFor': re.compile(r'(^|/)mujoco_menagerie/boston_dynamics_spot/[^/]+\.xml$'),
     'load': load_playground,
     'create': lambda policy, prefix: PlaygroundController(policy, prefix, 'Spot joystick policy')},
    {'id': 'playground-go2', 'name': 'Walking · Unitree Go2', 'dir': 'playground_go2',
     'defaultFor': re.compile(r'(^|/)mujoco_menagerie/unitree_go2/go2\.xml$'),
     'load': load_playground,
     'create': lambda policy, prefix: PlaygroundController(policy, prefix, 'Go2 joystick policy')},
    {'id': 'microduck-walk', 'name': 'Walking · Microduck', 'dir': 'microduck_rl',
     'defaultFor': re.compile(r'(^|/)custom/microduck/microduck\.xml$'),
     'load': load_microduck,
     'create': MicroduckController},
]

_loaded = {}


def by_id(pid):
  return next((p for p in POLICIES if p['id'] == pid), None)


def installed(p):
  return (POLICY_ROOT / p['dir']).is_dir()


def default_policy(key):
  return next((p['id'] for p in POLICIES if p.get('defaultFor') and p['defaultFor'].search(key) and installed(p)), '')


def create(pid, prefix):
  """A controller of policy `pid` for the robot with name prefix `prefix`;
  its weights are loaded once and shared."""
  p = by_id(pid)
  if p is None:
    raise ValueError(f'no policy {pid} on the server')
  if not installed(p):
    raise ValueError(f"{p['name']}: pretrained_policies/{p['dir']}/ is not installed")
  if pid not in _loaded:
    _loaded[pid] = p['load'](POLICY_ROOT / p['dir'])
  return p['create'](_loaded[pid], prefix)
