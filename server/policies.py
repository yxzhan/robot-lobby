"""Trained policies on the server: the ones of public/policy.js's POLICIES
that have a port here (same ids, so the Teleop panel's choice carries
over), running on onnxruntime. A policy controller has the interface
policy.js describes (bind, unbind, owns, pose, reset, step; locomotion ones
also cmd, drive, height), with a synchronous step.

GrootController is a port of public/groot.js (see there for the
observation, the two MLPs and the PD gains).
"""

import math
import pathlib
import re

import mujoco
import numpy as np

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
POLICY_ROOT = PUBLIC / 'pretrained_policies'
OBJ = mujoco.mjtObj

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


class GrootController:
  def __init__(self, sessions, prefix):
    self.balance, self.walk = sessions
    self.prefix = prefix
    self.name = 'GR00T locomotion policy'
    self.drive = {'forward': 0.5, 'back': 0.3, 'side': 0.3, 'turn': 0.6}
    self.cmd = [0.0, 0.0, 0.0]
    self.height = 0.74
    self.model = None
    self.saved_gains = None
    self.driven = set()
    self.reset()

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
    return {'name': self.name, 'drive': self.drive, 'height': self.height}


def load_groot(directory):
  import onnxruntime as ort
  opts = ort.SessionOptions()
  opts.intra_op_num_threads = 1  # tiny MLPs, many robots: no thread pools
  open_ = lambda n: ort.InferenceSession(str(directory / n), opts, providers=['CPUExecutionProvider'])
  return open_('GR00T-WholeBodyControl-Balance.onnx'), open_('GR00T-WholeBodyControl-Walk.onnx')


# ---------------------------------------------------------------------------
# Registry (ids as in policy.js)

POLICIES = [
    {'id': 'groot-g1', 'name': 'GR00T locomotion · Unitree G1', 'dir': 'GR00T-WholeBodyControl_g1',
     'defaultFor': re.compile(r'(^|/)mujoco_menagerie/unitree_g1/[^/]+\.xml$'),
     'load': load_groot, 'create': GrootController},
]

_loaded = {}


def by_id(pid):
  return next((p for p in POLICIES if p['id'] == pid), None)


def installed(p):
  return (POLICY_ROOT / p['dir']).is_dir()


def default_policy(key):
  return next((p['id'] for p in POLICIES if p['defaultFor'].search(key) and installed(p)), '')


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
