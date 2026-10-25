"""Shared-world server: MuJoCo (Python) steps one physics world, every
connected browser renders it.

    python3 server/server.py [port]   # default 4007, then open http://localhost:4007/

It serves public/ (like scripts/serve.py, Cache-Control: no-store), the
render data of each model file at /render?rel=… (render_asset.py) and a
WebSocket at /ws, and a small HTTP API for scripts (GET /api/robots, POST
/api/goal: goal markers, see api_goal). The page sees /server.json and switches to network mode
(public/js/net.js): it loads no MuJoCo at all. It draws each model from its
render data and places its bodies from the qpos / mocap state broadcast
here (its own forward kinematics); picking and everything else that needs
the model are requests to this server.

Protocol (client -> server, JSON {op, ...}; `rid` asks for a reply):
  spawn {rel, label, pos, yaw}   remove {ids}         move {id, pos, yaw}
  place {id, pos, quat}          reset {}             reset_robot {id}
  pause {paused}                 physics {physics}    qpos {adr, dof, v}
  ctrl {a, v}                    perturb {body, local, target} (body -1: let go)
  ray {eye, dir} -> {body, point} or {body: -1}        ping {t} -> pong {t}
  Teleop (robot `id`): use {kind: policy|base|ik, on (policy: its id or '')}
  cmd {kind: policy|base, cmd: [vx, vy, yaw rate], height?} (repeated while
  not zero; stops after CMD_TIMEOUT without one; a moving one drops the goal)
  command {name, value}   one of the policy's other commands (its describe()'s
  `commands`, public/js/commands.js), kept until changed
  skill {name, duration?}   the policy's manual skill to run (its
  describe()'s `skills`, public/js/skills.js): the model, not an input of
  it; a timed one for `duration` s instead of its own
  goal {goal: {x, y, yaw} or null} (the robot drives there: goal.py)
  ik_config {config}   ik_target {chain, pos, rot} (in the chain's ref body
  frame)   ik_reset_target {chain or null}   ik_gripper {chain, amount?}
  (amount: 0 open … 1 closed, or a list of them by gripper actuator; none:
  toggle)   ik_save   ik_defaults
  base_config {config}   base_save   base_defaults
  stage_load {id} or {bundle}   a published stage, or one the page has,
                    replaces the world (for everyone)
Server -> client:
  {type: 'scene', version, items, physics, paused, world, stage?}   the world
                              (`world`: names, joint and actuator tables for
                              the panels; `stage`: the id of the stage just
                              loaded)
  {type: 'meta', items, physics, paused}             same world, new settings
  {type: 'robot', item}                     a robot's controllers changed
                              (item.use, item.errors, item.ik / base / policy:
                              their describe())
  {type: 'status', robots: {id: {ik: {targets, err}, base: {measured},
                              goal: {phase, dist, dyaw}, policy: {command,
                              skill, active}}}} (skill: the manual one
                              picked; active: the model running)
                              STATUS_HZ times a second
  {type: 'goal', id, goal, arrived?}  a robot's goal changed (null: none;
                              arrived: the goal it reached, let go)
  {type: 'reset', id}            after reset (id null) / reset_robot {id}
  {type: 'reply', rid, ...} / {type: 'error', op, msg, rid?}
  {type: 'pong', t}
  binary state, physics['stateHz'] times a second (30 by default): int32 [version, epoch, nq, nmocap,
  nu], 4 bytes padding, float64 time, then float32 [qpos, mocap_pos,
  mocap_quat, ctrl]; ctrl only about CTRL_HZ times a second (nu 0 in the
  other states). `epoch` changes whenever the state jumps (reset, edits), so
  clients do not interpolate across it.

Robots are attached in order (like world.js), so bodies, joints, actuators
and mocap bodies are numbered the same in the world here and in the page,
which lays the world out from each robot's own model.
The Teleop panel's controllers run here, ticked every IK.CONTROL_DT of sim
time with their software servos before every step, like App.advance does in
the browser: trained policies (policies.py, GR00T on onnxruntime), mobile
bases (base.py), IK (ik.py) and the goal marker (goal.py), ports of
groot.js, base.js, ik.js and goal.js; a goal is one per robot, for every
browser. Setups
saved from the panel go to server/saved/{ik,base}.json. Not yet: the Physics
panel's default geom properties.

Stages (public/js/stage.js): a page saves its own in its browser; the
server's are the published ones, the community's, folders under
mjcf_assets/saved_stage/ named by a random id: stage.json, thumbnail.jpg,
ik.json, base.json, cameras.json. GET /api/stages lists them, GET
/api/stages/<id> reads one as a bundle, POST /api/stages/capture returns the
world as a new bundle without keeping it (the page sends its name,
thumbnail, view and cameras; it saves it in the browser), POST
/api/stages/publish {bundle, author: {name, email}} writes a bundle and
commits it in the library's git repository (git_commit_stage), DELETE
/api/stages/<id> deletes one; the stage_load request loads one of them, or a
bundle the page sends.
"""

import asyncio
import base64
import copy
import gzip
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
import uuid
import warnings

import mujoco
import numpy as np
from aiohttp import WSMsgType, web, web_fileresponse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import base as BASE  # noqa: E402
import goal as GOAL  # noqa: E402
import ik as IK  # noqa: E402
import policies as POL  # noqa: E402
import render_asset  # noqa: E402

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
STATUS_HZ = 5         # controller status (IK targets and errors) broadcasts per second
CTRL_HZ = 5           # ctrl goes with a state this often: only the panels show it
STEP_BUDGET = 0.02     # s of wall time stepping per loop iteration at most


def cpus():
  """The CPUs this process may use: its affinity, capped by a container's
  CPU quota (a Binder pod sees the host's cores but may use one or two)."""
  n = len(os.sched_getaffinity(0))
  try:
    quota, period = pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if quota != 'max':
      n = min(n, max(1, math.ceil(int(quota) / int(period))))
  except (OSError, ValueError):
    pass
  return n


# MuJoCo's own thread pool for each world: it solves the constraint islands
# (robots apart from each other) side by side, the same result as one
# thread; eight G1s step in 0.78 ms instead of 1.3. PHYSICS_THREADS in the
# environment overrides it (1: none).
PHYSICS_THREADS = int(os.environ.get('PHYSICS_THREADS') or min(8, cpus()))

# As in the browser (physics.js): MuJoCo's pyramidal cones and impratio 1
# (elliptic and 10 made friction too strong), noslip on (soft friction
# creeps: held things slide out of a grasp without it).
PHYSICS_DEFAULTS = {
    'gravity': [0, 0, -9.81], 'wind': [0, 0, 0], 'magnetic': [0, -0.5, 0],
    'density': 0, 'viscosity': 0, 'impratio': 1, 'cone': 'pyramidal',
    'noslip_iterations': 100, 'noslip_tolerance': 1e-10,
    'integrator': 'implicitfast', 'solver': 'Newton', 'iterations': 100, 'tolerance': 1e-8,
    'timestep': 0.005,  # s, live
    'speed': 1,         # the most sim seconds per wall second; 0: as fast as it goes
    'stateHz': 30,      # state broadcasts per second (not physics, but the server's too)
    'sleep': True,  # island sleeping (physics.js): set at compile time
}
CONES = ['pyramidal', 'elliptic']  # mjtCone order
INTEGRATORS = ['Euler', 'RK4', 'implicit', 'implicitfast']  # mjtIntegrator order
SOLVERS = ['PGS', 'CG', 'Newton']  # mjtSolver order
OPTION_KEYS = [k for k in PHYSICS_DEFAULTS if k not in ('sleep', 'speed', 'stateHz')]

# A model's own <option> loses against the world's (like in the browser):
# no need to hear about it. Other MuJoCo warnings go to the console, not to
# MUJOCO_LOG.TXT in the working directory.
warnings.filterwarnings('ignore', message='Attach conflict')
# IK setups saved from the Teleop panel (for everyone on this server).
IK_SAVED = IK.SavedConfigs(pathlib.Path(__file__).resolve().parent / 'saved' / 'ik.json')
BASE_SAVED = IK.SavedConfigs(pathlib.Path(__file__).resolve().parent / 'saved' / 'base.json')
CMD_TIMEOUT = 1.0  # s: a drive command not repeated for this long stops (a frozen or vanished browser)
mujoco.set_mju_user_warning(lambda msg: msg.startswith('Attach conflict') or print(f'MuJoCo: {msg.strip()}'))


def fmt(v):
  return ' '.join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)


def base_xml(physics):
  opts = ' '.join(f'{k}="{fmt(physics[k])}"' for k in OPTION_KEYS)
  return f"""
<mujoco model="world">
  <option {opts}>{'<flag sleep="enable"/>' if physics.get('sleep') else ''}</option>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
  </worldbody>
</mujoco>"""


def apply_options(model, physics):
  o = model.opt
  o.timestep = physics['timestep']
  for k in ('gravity', 'wind', 'magnetic'):
    o.__setattr__(k, physics[k])
  o.density, o.viscosity, o.impratio = physics['density'], physics['viscosity'], physics['impratio']
  o.cone = max(0, CONES.index(physics['cone']) if physics['cone'] in CONES else 0)
  o.integrator = INTEGRATORS.index(physics['integrator'])
  o.solver = SOLVERS.index(physics['solver'])
  o.iterations, o.tolerance = physics['iterations'], physics['tolerance']
  o.noslip_iterations, o.noslip_tolerance = physics['noslip_iterations'], physics['noslip_tolerance']


def checked_physics(physics, old):
  """Physics panel settings from a page or a stage: known keys only (the rest
  as they were), in range."""
  new = {k: physics.get(k, v) for k, v in old.items()}
  new['timestep'] = min(0.05, max(1e-4, float(new['timestep'])))
  new['speed'] = max(0.0, float(new['speed']))
  new['stateHz'] = min(120, max(1, int(new['stateHz'])))
  new['iterations'] = max(1, int(new['iterations']))
  new['noslip_iterations'] = max(0, int(new['noslip_iterations']))
  new['tolerance'], new['noslip_tolerance'] = max(0.0, float(new['tolerance'])), max(0.0, float(new['noslip_tolerance']))
  for k, names in (('cone', CONES), ('integrator', INTEGRATORS), ('solver', SOLVERS)):
    if new[k] not in names:
      new[k] = old[k] if old[k] in names else PHYSICS_DEFAULTS[k]
  return new


def asset_path(rel):
  """The model file of an asset (relative to public/, inside mjcf_assets/)."""
  p = (PUBLIC / rel).resolve()
  root = (PUBLIC / 'mjcf_assets').resolve()
  if root not in p.parents or p.suffix != '.xml' or not p.is_file():
    raise ValueError(f'no model {rel}')
  return p


# ---------------------------------------------------------------------------
# Per model file: sizes and first keyframe (compiled on its own, cached).

_infos = {}


def model_info(rel):
  if rel not in _infos:
    spec = mujoco.MjSpec.from_file(str(asset_path(rel)))
    m = spec.compile()
    key = None
    if len(spec.keys):
      k = spec.keys[0]
      key = {'qpos': np.array(k.qpos) if len(k.qpos) == m.nq else None,
             'ctrl': np.array(k.ctrl) if len(k.ctrl) == m.nu else None}
    raw = render_asset.build(spec, m, rel, PUBLIC)
    _infos[rel] = {'dims': {'nq': m.nq, 'nv': m.nv, 'na': m.na, 'nu': m.nu, 'nmocap': m.nmocap},
                   'nbody': m.nbody, 'key': key, 'render': gzip.compress(raw, 5), 'render_size': len(raw),
                   'etag': '"' + hashlib.sha1(raw).hexdigest() + '"'}
  return _infos[rel]


class Robot:
  def __init__(self, rid, rel, label, pos, yaw):
    self.id, self.rel, self.label = rid, rel, label
    self.prefix = f'r{rid}/'
    self.pos, self.yaw = [float(v) for v in pos], float(yaw)
    self.info = model_info(rel)
    self.initial = None  # {qpos, ctrl} after spawning: what Reset goes back to
    self.body_offset = 0
    # Robots (not environments or objects) get the Teleop panel's
    # controllers, keyed by their file under mjcf_assets/robots/ like in the
    # browser (main.js makeRobot, loadUse): a trained policy (policies.py),
    # a mobile base (base.py) and IK chains (ik.py), each switched on or off
    # in `use`.
    m = re.search(r'(?:^|/)mjcf_assets/robots/(.+)$', rel)
    self.key = m.group(1) if m else None
    self.errors = {}
    self.use = self.policy = self.base = self.ik = None
    self.policy_id = ''
    self.cmd_at = {}  # 'policy' / 'base' -> (when its drive command last came, from which browser)
    self.goal, self.goal_state, self.goal_result = None, {}, None  # the goal marker's pose (goal.py)
    self.arrived = False
    if self.key:
      self.use = {'policy': POL.default_policy(self.key), 'base': None, 'ik': True}
      self.base = BASE.BaseController(self.key, self.prefix, BASE_SAVED)
      self.ik = IK.IkController(self.key, self.prefix, IK_SAVED)
      self.make_policy()

  @property
  def dims(self):
    return self.info['dims']

  def bodies(self):
    return (self.body_offset, self.body_offset + self.info['nbody'] - 1)

  def item(self):
    out = {'id': self.id, 'rel': self.rel, 'label': self.label, 'pos': self.pos, 'yaw': self.yaw}
    if self.key:
      out.update(key=self.key, use=self.use, errors=self.errors, goal=self.goal,
                 ik=self.ik.describe() if self.ik.model is not None else None,
                 base=self.base.describe() if self.base.model is not None else None,
                 policy=self.policy.describe() if self.policy and self.use['policy'] else None)
    return out

  def make_policy(self):
    """self.policy for the policy use['policy'] names (main.js makePolicy)."""
    pid = self.use['policy']
    if self.policy is not None and self.policy_id != pid:
      self.policy.unbind()
      self.policy = None
    if not pid:
      return
    try:
      if self.policy is None:
        self.policy, self.policy_id = POL.create(pid, self.prefix), pid
    except Exception as err:
      self.errors['policy'] = f'could not load: {err}'
      self.use['policy'] = ''

  # What the policy, else the base, drives is off limits to the others.
  def by_policy(self, j):
    return (self.use['policy'] and self.policy and self.policy.owns(j)) or None

  def owner(self, j):
    return self.by_policy(j) or (self.use['base'] and self.base.owns(j)) or None

  def bind_controller(self, kind, model):
    """Binds one controller to `model`; False (errors[kind] set) if it cannot
    drive this robot (main.js bindController)."""
    try:
      if kind == 'policy':
        self.policy.bind(model)
      elif kind == 'base':
        self.base.bind(model, self.by_policy, self.bodies())
      else:
        self.ik.bind(model, self.owner, self.bodies())
      self.errors.pop(kind, None)
      return True
    except Exception as err:
      if kind != 'policy':
        getattr(self, kind).model = None
      self.errors[kind] = str(err)
      return False

  def bind(self, model):
    """Finds the controllers' joints and actuators in a new world."""
    if not self.key:
      return
    if self.use['policy'] and self.policy and not self.bind_controller('policy', model):
      self.use['policy'] = ''
    self.bind_controller('base', model)
    if self.use['base'] is None:
      self.use['base'] = bool(self.base.config and self.base.config.get('kind'))
    if not self.bind_controller('ik', model):
      self.use['ik'] = False
    self.ik.enabled = bool(self.use['ik'])

  def controllers(self):
    """The switched-on controllers, in tick order."""
    if not self.key:
      return []
    out = []
    if self.use['policy'] and self.policy is not None and self.policy.model is not None:
      out.append(self.policy)
    if self.use['base'] and self.base.model is not None:
      out.append(self.base)
    if self.use['ik'] and self.ik.model is not None:
      out.append(self.ik)
    return out

  def locomotion(self):
    """The switched-on controllers that take a drive command (teleop.js locomotion)."""
    return [c for c in self.controllers() if c is not self.ik and getattr(c, 'drive', None)]

  def set_goal(self, goal, arrived=False):
    self.goal = goal and {k: float(goal[k]) for k in ('x', 'y', 'yaw')}
    self.goal_state, self.goal_result = {}, None
    self.arrived = arrived  # the last goal was reached (rather than cleared or replaced)
    if goal is None:
      for c in self.locomotion():
        c.cmd = [0.0, 0.0, 0.0]

  def follow_goal(self, data):
    """Sets the drive command towards the goal, if there is one. Returns the
    goal if the robot just reached it: it is let go then (pushed off
    afterwards, the robot stays where it is put), as in teleop.js."""
    locos = self.locomotion() if self.goal else []
    self.goal_result = None
    if not locos:
      return None
    pose = GOAL.body_pose(data, self.body_offset)
    for c in locos:
      self.goal_result = GOAL.goal_command(pose, self.goal, c.drive, self.goal_state)
      c.cmd = list(self.goal_result['cmd'])
    if self.goal_result['phase'] != 'arrived':
      return None
    reached = self.goal
    self.set_goal(None, arrived=True)
    return reached

  def status(self):
    out = {}
    if self.goal_result:
      out['goal'] = {k: self.goal_result[k] for k in ('phase', 'dist', 'dyaw')}
    if self.use['ik'] and self.ik.model is not None:
      out['ik'] = self.ik.status()
    if self.use['base'] and self.base.ready:
      out['base'] = {'measured': self.base.measured}
    if self.use['policy'] and self.policy is not None:
      policy = {}
      if getattr(self.policy, 'commands', None):
        policy['command'] = self.policy.command
      skills = getattr(self.policy, 'skills', None)
      if skills is not None and skills.shown:  # policies.Skills: the one picked, the one running
        policy['skill'], policy['active'] = skills.want, skills.shown
      if policy:
        out['policy'] = policy
    return out


# Each robot's qpos/qvel/act/ctrl/mocap entries are a contiguous range of
# each array (main.js stateLayout).
STATE_FIELDS = [('qpos', 1, 'nq'), ('qvel', 1, 'nv'), ('act', 1, 'na'), ('ctrl', 1, 'nu'),
                ('mocap_pos', 3, 'nmocap'), ('mocap_quat', 4, 'nmocap')]


def state_layout(robots):
  layout, at = {}, {'nq': 0, 'nv': 0, 'na': 0, 'nu': 0, 'nmocap': 0}
  for r in robots:
    layout[r.id] = {f: n * at[d] for f, n, d in STATE_FIELDS}
    for k in at:
      at[k] += r.dims[k]
  return layout


def flat(data, field):
  return getattr(data, field).reshape(-1)


def share_assets(spec, robots):
  """Copies of one model file use the meshes of its first copy: attaching
  gives each copy its own (prefixed) ones, and compiling ten G1s then builds
  ten sets of mesh data, ~750 MB instead of ~75 MB. (Textures stay per copy:
  a material's texture names cannot be redirected like a geom's mesh.)"""
  first = {}
  for r in robots:
    first.setdefault(r.rel, r.prefix)
  assets = list(spec.meshes)
  names = {a.name for a in assets}
  remap = {}  # a copy's asset -> the first copy's
  for r in robots:
    p0 = first[r.rel]
    if p0 == r.prefix:
      continue
    for a in assets:
      if a.name.startswith(r.prefix) and p0 + a.name[len(r.prefix):] in names:
        remap[a.name] = p0 + a.name[len(r.prefix):]
  if not remap:
    return
  for g in spec.geoms:
    if g.meshname in remap:
      g.meshname = remap[g.meshname]
  for a in assets:
    if a.name in remap:
      spec.delete(a)


def compile_world(robots, physics):
  spec = mujoco.MjSpec.from_string(base_xml(physics))
  for r in robots:
    child = mujoco.MjSpec.from_file(str(asset_path(r.rel)))
    frame = spec.worldbody.add_frame(pos=r.pos, quat=[math.cos(r.yaw / 2), 0, 0, math.sin(r.yaw / 2)])
    spec.attach(child, frame=frame, prefix=r.prefix)
  share_assets(spec, robots)
  model = spec.compile()
  offset = 1
  for r in robots:
    r.body_offset = offset
    offset += r.info['nbody'] - 1
  if offset != model.nbody:
    raise RuntimeError(f'world has {model.nbody} bodies, expected {offset}')
  return model


# ---------------------------------------------------------------------------
# Stages (public/js/stage.js): folders under mjcf_assets/saved_stage/.

STAGE_VERSION = 1
STAGE_FILES = ('ik.json', 'base.json', 'cameras.json')
STAGE_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
QPOS_SIZE = {0: 7, 1: 4}  # free, ball joints; slide and hinge have one
THUMB_PREFIX = 'data:image/jpeg;base64,'


def stages_dir():
  return (PUBLIC / 'mjcf_assets').resolve() / 'saved_stage'


def stage_path(sid):
  if not isinstance(sid, str) or not STAGE_ID.match(sid):
    raise ValueError(f'bad stage id {sid!r}')
  return stages_dir() / sid


def robot_tables(m, r, at):
  """A robot's joints [(name, qposadr, size)], actuators [(name, id)] and
  mocap bodies [(name, mocap id)], names without its prefix ('#k': its k-th
  one, unnamed), as stage.js robotTables."""
  def local(obj, i, k):
    n = mujoco.mj_id2name(m, obj, i) or ''
    return n[len(r.prefix):] if n.startswith(r.prefix) else n or f'#{k}'
  q0, nq = at['qpos'], r.dims['nq']
  js = [j for j in range(m.njnt) if q0 <= m.jnt_qposadr[j] < q0 + nq]
  joints = [(local(mujoco.mjtObj.mjOBJ_JOINT, j, j - js[0]), int(m.jnt_qposadr[j]), QPOS_SIZE.get(int(m.jnt_type[j]), 1))
            for j in js]
  u0 = at['ctrl']
  ctrls = [(local(mujoco.mjtObj.mjOBJ_ACTUATOR, a, a - u0), a) for a in range(u0, u0 + r.dims['nu'])]
  b0 = r.body_offset
  mocaps = [(local(mujoco.mjtObj.mjOBJ_BODY, b, b - b0), int(m.body_mocapid[b]))
            for b in range(b0, b0 + r.info['nbody'] - 1) if m.body_mocapid[b] >= 0]
  return joints, ctrls, mocaps


def capture_state(m, d, r, at):
  joints, ctrls, mocaps = robot_tables(m, r, at)
  rnd = lambda v: round(float(v), 6)
  return {'qpos': {n: rnd(d.qpos[adr]) if k == 1 else [rnd(v) for v in d.qpos[adr:adr + k]] for n, adr, k in joints},
          'ctrl': {n: rnd(d.ctrl[a]) for n, a in ctrls},
          'mocap': {n: {'pos': [rnd(v) for v in d.mocap_pos[i]], 'quat': [rnd(v) for v in d.mocap_quat[i]]} for n, i in mocaps}}


def apply_state(m, d, r, at, state):
  """An item's saved state into `d` (unknown names skipped), at rest."""
  joints, ctrls, mocaps = robot_tables(m, r, at)
  J = {n: (adr, k) for n, adr, k in joints}
  for n, v in (state.get('qpos') or {}).items():
    vals = v if isinstance(v, list) else [v]
    if n in J and len(vals) == J[n][1]:
      d.qpos[J[n][0]:J[n][0] + J[n][1]] = vals
  C = dict(ctrls)
  for n, v in (state.get('ctrl') or {}).items():
    if n in C:
      d.ctrl[C[n]] = v
  M = dict(mocaps)
  for n, v in (state.get('mocap') or {}).items():
    if n in M:
      d.mocap_pos[M[n]], d.mocap_quat[M[n]] = v['pos'], v['quat']
  d.qvel[at['qvel']:at['qvel'] + r.dims['nv']] = 0
  d.act[at['act']:at['act'] + r.dims['na']] = 0


def stage_summary(stage, folder):
  return {'id': folder.name, 'name': stage.get('name', ''), 'saved': stage.get('saved'),
          'count': len(stage.get('items', [])), 'thumb': (folder / 'thumbnail.jpg').is_file(),
          'author': stage.get('author') or ''}


def check_bundle(bundle):
  """A bundle's stage, if it is one (items with rel and pos); ValueError if not."""
  stage = bundle.get('stage') if isinstance(bundle, dict) else None
  items = stage.get('items') if isinstance(stage, dict) else None
  if not isinstance(items, list) or any(not isinstance(it, dict) or not isinstance(it.get('rel'), str)
                                        or not isinstance(it.get('pos'), list) for it in items):
    raise ValueError('not a stage: items need rel and pos')
  return stage


def list_stages():
  out = []
  root = stages_dir()
  for folder in sorted(root.iterdir()) if root.is_dir() else []:
    try:
      out.append(stage_summary(json.loads((folder / 'stage.json').read_text()), folder))
    except (OSError, ValueError):
      continue
  return sorted(out, key=lambda s: s['saved'] or '', reverse=True)


def read_stage(sid):
  folder = stage_path(sid)
  stage = json.loads((folder / 'stage.json').read_text())
  stage['id'] = sid
  files = {}
  for f in STAGE_FILES:
    try:
      files[f] = json.loads((folder / f).read_text())
    except (OSError, ValueError):
      files[f] = {}
  thumb = folder / 'thumbnail.jpg'
  return {'stage': stage, 'files': files,
          'thumbnail': THUMB_PREFIX + base64.b64encode(thumb.read_bytes()).decode() if thumb.is_file() else None}


def write_stage(bundle):
  """Writes a bundle as a stage folder (its id kept if it is a good one, else
  a new one); returns its summary."""
  stage = dict(check_bundle(bundle))
  if not (isinstance(stage.get('id'), str) and STAGE_ID.match(stage['id'])):
    stage['id'] = str(uuid.uuid4())
  folder = stage_path(stage['id'])
  folder.mkdir(parents=True, exist_ok=True)
  (folder / 'stage.json').write_text(json.dumps(stage, indent=1))
  files = bundle.get('files') or {}
  for f in STAGE_FILES:
    (folder / f).write_text(json.dumps(files.get(f) or {}, indent=1))
  thumb = bundle.get('thumbnail')
  if isinstance(thumb, str) and thumb.startswith(THUMB_PREFIX):
    (folder / 'thumbnail.jpg').write_bytes(base64.b64decode(thumb[len(THUMB_PREFIX):]))
  return stage_summary(stage, folder)


AUTHOR_NAME = re.compile(r'[<>\n\r\t]')
AUTHOR_EMAIL = re.compile(r'^[^\s<>@]+@[^\s<>@]+$')
COMMITTER = {'GIT_COMMITTER_NAME': 'Robot Lobby', 'GIT_COMMITTER_EMAIL': 'robot-lobby@localhost'}


def publish_stage(bundle, author):
  """Writes a bundle as a new published stage (a new id if one of that id
  is there: a stage is never written over another's) with its author's
  name, and commits it (git_commit_stage); returns its summary with
  `commit` (the commit's id, or null and `commit_error`)."""
  stage = dict(check_bundle(bundle))
  name = AUTHOR_NAME.sub(' ', str((author or {}).get('name') or '')).strip()[:100]
  email = str((author or {}).get('email') or '').strip()[:200]
  if email and not AUTHOR_EMAIL.match(email):
    raise ValueError(f'not an email address: {email!r}')
  if not (isinstance(stage.get('id'), str) and STAGE_ID.match(stage['id'])) or stage_path(stage['id']).exists():
    stage['id'] = str(uuid.uuid4())
  stage['author'] = name
  stage['published'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
  out = write_stage({**bundle, 'stage': stage})
  try:
    out['commit'] = git_commit_stage(stage, name, email)
  except (OSError, RuntimeError, subprocess.SubprocessError) as err:
    print(f'stage {stage["id"]}: written, not committed: {err}')
    out['commit'], out['commit_error'] = None, str(err)
  return out


def git_commit_stage(stage, name, email):
  """Commits the stage's folder alone in the library's git repository,
  authored by `name` <`email`> (Anonymous if none), committed by the
  server; returns the commit's id. Another server on the same library may
  be committing too: a held index lock is waited for a little."""
  lib = stages_dir().parent
  rel = f'saved_stage/{stage["id"]}'
  env = {**os.environ, **COMMITTER}

  def git(*args):
    for _ in range(10):
      p = subprocess.run(['git', '-C', str(lib), *args], capture_output=True, text=True, timeout=60, env=env)
      if p.returncode == 0:
        return p.stdout.strip()
      if 'index.lock' not in p.stderr:
        raise RuntimeError(f'git {args[0]}: {p.stderr.strip() or p.stdout.strip()}')
      time.sleep(0.5)
    raise RuntimeError(f'git {args[0]}: the index stayed locked')
  if git('rev-parse', '--is-inside-work-tree') != 'true':
    raise RuntimeError(f'{lib} is not a git repository')
  git('add', '--', rel)
  git('commit', '--quiet', '-m', f'Stage: {stage.get("name") or stage["id"]}', f'--author={name or "Anonymous"} <{email}>', '--', rel)
  return git('rev-parse', 'HEAD')


def delete_stage(sid):
  folder = stage_path(sid)
  if not (folder / 'stage.json').is_file():
    raise ValueError(f'no stage {sid}')
  for f in folder.iterdir():
    f.unlink()
  folder.rmdir()


# ---------------------------------------------------------------------------

class World:
  def __init__(self):
    self.physics = dict(PHYSICS_DEFAULTS)
    self.robots = []
    self.next_id = 1
    self.version = 0
    self.epoch = 0
    self.paused = False
    self.model = self.data = None
    self.sim_start = time.monotonic()
    self.perturbs = {}  # client -> {body, local, target, prev}
    self.perturbed = False  # xfrc_applied holds perturbation forces
    self.next_control = 0   # sim time of the next controller tick (IK.CONTROL_DT apart)
    self.clients = set()
    self.lock = asyncio.Lock()  # one scene change at a time
    # The model and data belong to the stepping thread while it holds this
    # (run); requests that change or read them take it in between.
    self.stepping = asyncio.Lock()
    self.dirty = True

  # -- scene changes (compiled off the event loop, stepping goes on) --------

  # -- stages ----------------------------------------------------------------

  def stage_bundle(self, extra):
    """The world as a new stage bundle (stage.js), with what only the page
    knows from `extra`: name, thumbnail, view, cameras (setups), on (robot id
    -> its switched-on cameras)."""
    layout, ik, base, items = state_layout(self.robots), {}, {}, []
    on = extra.get('on') or {}
    for r in self.robots:
      it = {'rel': r.rel, 'label': r.label, 'pos': [round(v, 4) for v in r.pos], 'yaw': round(r.yaw, 4)}
      if r.key:
        it['use'] = dict(r.use)
        for kind, out in (('ik', ik), ('base', base)):
          cfg = getattr(r, kind).config
          if cfg and not cfg.get('auto'):
            out[r.key] = cfg
      if on.get(str(r.id)):
        it['cameras'] = list(on[str(r.id)])
      it['state'] = capture_state(self.model, self.data, r, layout[r.id])
      items.append(it)
    stage = {'version': STAGE_VERSION, 'id': str(uuid.uuid4()), 'name': str(extra.get('name') or 'Stage'),
             'saved': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'items': items,
             'physics': dict(self.physics), 'view': extra.get('view')}
    return {'stage': stage, 'files': {'ik.json': ik, 'base.json': base, 'cameras.json': extra.get('cameras') or {}},
            'thumbnail': extra.get('thumbnail')}

  async def load_stage(self, bundle):
    """A stage replaces the world: its items spawned anew (numbered on from
    the last id; its controllers switched on and set up as saved) in a new
    MjData from time 0, each in its saved state, which Reset goes back to.
    Items whose model is gone are left out. Every page is told (scene with
    `stage`, then a reset)."""
    stage, files = bundle['stage'], bundle.get('files') or {}
    async with self.lock:
      first = self.next_id
      physics = checked_physics(stage.get('physics') or {}, self.physics)

      def make():
        robots, kept = [], []
        for it in stage.get('items', []):
          try:
            model_info(it['rel'])
          except Exception as err:
            print(f'stage {stage.get("name")}: skipping {it.get("rel")}: {err}')
            continue
          r = Robot(first + len(robots), it['rel'], it.get('label') or it['rel'], it['pos'], it.get('yaw', 0))
          if r.key:
            if it.get('use'):
              r.use.update({k: v for k, v in it['use'].items() if k in ('policy', 'base', 'ik')})
              r.make_policy()
            for kind in ('ik', 'base'):
              cfg = (files.get(f'{kind}.json') or {}).get(r.key)
              if cfg:
                c = getattr(r, kind)
                c.config, c.saved = copy.deepcopy(cfg), False
          robots.append(r)
          kept.append(it)
        return robots, kept, compile_world(robots, physics)

      robots, kept, model = await asyncio.to_thread(make)
      async with self.stepping:
        for r in self.robots:
          if r.policy is not None:
            r.policy.unbind()
        self.physics = physics
        self.next_id = first + len(robots)
        self.data, self.robots = None, []  # nothing carries over
        self.swap(model, robots, robots)
        layout, d = state_layout(robots), self.data
        for r, it in zip(robots, kept):
          at = layout[r.id]
          apply_state(model, d, r, at, it.get('state') or {})
          r.initial = {'qpos': d.qpos[at['qpos']:at['qpos'] + r.dims['nq']].copy(),
                       'ctrl': d.ctrl[at['ctrl']:at['ctrl'] + r.dims['nu']].copy()}
          for c in (r.policy, r.base, r.ik):
            if c:
              c.reset()
        mujoco.mj_forward(model, d)
        self.sim_start = time.monotonic()
        await self.broadcast_scene(stage=stage.get('id'))
        await self.send_all(json.dumps({'type': 'reset', 'id': None}))

  async def set_scene(self, robots, fresh=(), after=None):
    async with self.lock:
      model = await asyncio.to_thread(compile_world, robots, self.physics)
      async with self.stepping:
        self.swap(model, robots, fresh)
        if after:
          after()
        await self.broadcast_scene()

  async def restart(self):
    """Full restart (the toolbar's button): the model files read again
    (their cached sizes and keyframes dropped), every robot made anew at its
    spawn point (numbered from 1 again; fresh controllers, the policy loaded
    again, the same ones switched on), in a new MjData from time 0."""
    async with self.lock:
      old, first = list(self.robots), 1

      def make():
        _infos.clear()
        robots = []
        for k, r in enumerate(old):
          n = Robot(first + k, r.rel, r.label, r.pos, r.yaw)
          if n.key and r.use:
            n.use.update(r.use)
            n.make_policy()
          robots.append(n)
        return robots, compile_world(robots, self.physics)

      robots, model = await asyncio.to_thread(make)
      async with self.stepping:
        for r in old:
          if r.policy is not None:
            r.policy.unbind()
        self.next_id = first + len(robots)
        self.data, self.robots = None, []  # nothing carries over
        self.swap(model, robots, robots)
        await self.broadcast_scene()
        await self.send_all(json.dumps({'type': 'reset', 'id': None}))  # pages drop goals and the like

  def swap(self, model, robots, fresh):
    data = mujoco.MjData(model)
    if PHYSICS_THREADS > 1:
      mujoco.mju_threadpool(data, PHYSICS_THREADS)
    old, old_layout = self.data, state_layout(self.robots)
    layout = state_layout(robots)
    old_ids = {r.id for r in self.robots}
    if old is not None:
      for r in robots:
        if r.id not in old_ids:
          continue
        a, b = old_layout[r.id], layout[r.id]
        for f, n, d in STATE_FIELDS:
          k = n * r.dims[d]
          flat(data, f)[b[f]:b[f] + k] = flat(old, f)[a[f]:a[f] + k]
      data.time = old.time
    for r in robots:
      r.bind(model)
    for r in fresh:
      self.start_pose(model, data, r, layout[r.id])
      if r.key and r.use['policy'] and r.policy:
        r.policy.pose(data)
    mujoco.mj_forward(model, data)
    for r in fresh:
      at, d = layout[r.id], r.dims
      r.initial = {'qpos': data.qpos[at['qpos']:at['qpos'] + d['nq']].copy(),
                   'ctrl': data.ctrl[at['ctrl']:at['ctrl'] + d['nu']].copy()}
    self.model, self.data, self.robots = model, data, robots
    self.next_control = data.time
    self.perturbs.clear()
    self.version += 1
    self.epoch += 1
    self.resync()
    self.dirty = True

  def start_pose(self, model, data, r, at):
    """The robot's own first keyframe, moved to its spawn point and heading."""
    key = r.info['key']
    q0, nq, nu = at['qpos'], r.dims['nq'], r.dims['nu']
    if key and key['qpos'] is not None:
      q = data.qpos
      q[q0:q0 + nq] = key['qpos']
      c, s = math.cos(r.yaw), math.sin(r.yaw)
      hw, hz = math.cos(r.yaw / 2), math.sin(r.yaw / 2)
      for j in range(model.njnt):
        adr = model.jnt_qposadr[j]
        if not q0 <= adr < q0 + nq or model.jnt_type[j] != 0:
          continue
        x, y = q[adr], q[adr + 1]
        q[adr] = r.pos[0] + c * x - s * y
        q[adr + 1] = r.pos[1] + s * x + c * y
        q[adr + 2] += r.pos[2]
        w, qx, qy, qz = q[adr + 3:adr + 7]
        q[adr + 3:adr + 7] = [hw * w - hz * qz, hw * qx - hz * qy, hw * qy + hz * qx, hw * qz + hz * w]
      self.lift_above_floor(model, data, r, q0, nq)
    if key and key['ctrl'] is not None:
      data.ctrl[at['ctrl']:at['ctrl'] + nu] = key['ctrl']

  @staticmethod
  def lift_above_floor(model, data, r, q0, nq):
    """A keyframe made for another floor height: raise free joints so the
    collision geoms clear ours (main.js liftAboveFloor)."""
    mujoco.mj_kinematics(model, data)
    first, end = r.body_offset, r.body_offset + r.info['nbody'] - 1
    low = math.inf
    for g in range(model.ngeom):
      b = model.geom_bodyid[g]
      if first <= b < end and (model.geom_contype[g] or model.geom_conaffinity[g]):
        low = min(low, data.geom_xpos[g, 2] - model.geom_rbound[g])
    lift = r.pos[2] - low
    if not lift > 0.1:
      return
    for j in range(model.njnt):
      adr = model.jnt_qposadr[j]
      if q0 <= adr < q0 + nq and model.jnt_type[j] == 0:
        data.qpos[adr + 2] += lift

  # -- requests -------------------------------------------------------------

  async def handle(self, ws, msg):
    # These may rebuild the world: they take the stepping lock themselves,
    # around everything but compiling, so stepping goes on meanwhile.
    if msg['op'] in ('spawn', 'remove', 'physics', 'move', 'restart', 'stage_load'):
      return await self.handle_op(ws, msg)
    async with self.stepping:
      return await self.handle_op(ws, msg)

  async def handle_op(self, ws, msg):
    op = msg.get('op')
    m, d = self.model, self.data
    if op == 'spawn':
      await asyncio.to_thread(model_info, msg['rel'])  # compile it off the event loop (cached)
      r = Robot(self.next_id, msg['rel'], msg.get('label') or msg['rel'], msg['pos'], msg.get('yaw', 0))
      self.next_id += 1
      await self.set_scene([*self.robots, r], [r])
    elif op == 'restart':
      await self.restart()
    elif op == 'stage_load':  # a published stage, or a bundle the page has (its own)
      if msg.get('bundle') is not None:
        check_bundle(msg['bundle'])
        await self.load_stage(msg['bundle'])
      else:
        await self.load_stage(await asyncio.to_thread(read_stage, msg['id']))
    elif op == 'remove':
      ids = set(msg['ids'])
      await self.set_scene([r for r in self.robots if r.id not in ids])
    elif op == 'move':
      await self.move_fixed(self.robot(msg['id']), msg['pos'], msg['yaw'])
    elif op == 'place':
      self.place_free(self.robot(msg['id']), msg['pos'], msg['quat'])
      await self.broadcast_meta()
    elif op == 'reset':
      mujoco.mj_resetData(m, d)
      layout = state_layout(self.robots)
      for r in self.robots:
        self.write_initial(r, layout[r.id])
      mujoco.mj_forward(m, d)
      for r in self.robots:
        r.set_goal(None)
        for c in r.controllers():
          c.reset()
      self.sim_start = time.monotonic()
      self.next_control = 0
      self.jumped()
      await self.send_all(json.dumps({'type': 'reset', 'id': None}))
    elif op == 'reset_robot':
      r = self.robot(msg['id'])
      at = state_layout(self.robots)[r.id]
      self.write_initial(r, at)
      d.qvel[at['qvel']:at['qvel'] + r.dims['nv']] = 0
      d.act[at['act']:at['act'] + r.dims['na']] = 0
      mujoco.mj_forward(m, d)
      r.set_goal(None)
      for c in (r.policy, r.base, r.ik):
        if c:
          c.reset()
      self.jumped()
      await self.send_all(json.dumps({'type': 'reset', 'id': r.id}))
    elif op == 'pause':
      self.paused = bool(msg['paused'])
      self.resync()
      await self.broadcast_meta()
    elif op == 'physics':
      old, self.physics = self.physics, checked_physics(msg['physics'], PHYSICS_DEFAULTS)
      if old['speed'] != self.physics['speed'] and self.data is not None:
        self.resync()
      if bool(old.get('sleep')) != bool(self.physics['sleep']):
        await self.set_scene(list(self.robots))  # compiled in (the state carries over)
      else:
        async with self.stepping:
          if self.model is not None:
            apply_options(self.model, self.physics)
          await self.broadcast_meta()
    elif op == 'qpos':
      d.qpos[int(msg['adr'])] = float(msg['v'])
      d.qvel[int(msg['dof'])] = 0
      mujoco.mj_forward(m, d)
      self.jumped()
    elif op == 'ctrl':
      d.ctrl[int(msg['a'])] = float(msg['v'])
      self.dirty = True
    elif op == 'perturb':
      self.set_perturb(ws, msg)
    elif op == 'ray':
      return self.ray(msg['eye'], msg['dir'])
    elif op in ('use', 'cmd', 'command', 'skill', 'goal', 'ik_config', 'ik_target', 'ik_reset_target', 'ik_gripper', 'ik_save', 'ik_defaults',
                'base_config', 'base_save', 'base_defaults'):
      await self.controller_op(ws, op, self.robot(msg['id']), msg)
    else:
      raise ValueError(f'unknown op {op!r}')

  async def controller_op(self, ws, op, r, msg):
    """Teleop panel requests for robot `r` (public/js/remote.js)."""
    if not r.key:
      raise ValueError(f'{r.label} has no controllers')
    ik, base, d = r.ik, r.base, self.data
    chain = lambda: next((c for c in ik.chains if c['cfg']['name'] == msg['chain']), None)
    if op == 'use':
      self.set_use(r, msg['kind'], msg['on'])
    elif op == 'cmd':  # held keys: repeated while not zero, no echo
      c = r.policy if msg['kind'] == 'policy' else base
      if c is not None:
        c.cmd = [max(-5.0, min(5.0, float(v))) for v in msg['cmd'][:3]]
        if msg.get('height') is not None and 'height' in getattr(c, 'commands', {}):
          c.height = POL.clamp_command(c.commands['height'], msg['height'])
        r.cmd_at[msg['kind']] = (time.monotonic(), ws)
        if r.goal and any(c.cmd):  # driven with the keys: the goal is dropped
          r.set_goal(None)
          c.cmd = [max(-5.0, min(5.0, float(v))) for v in msg['cmd'][:3]]
          await self.send_all(json.dumps({'type': 'goal', 'id': r.id, 'goal': None}))
      return
    elif op == 'command':  # a policy command (the panel's sliders): everyone sees it in the status
      c = r.policy if r.use['policy'] else None
      d = c is not None and getattr(c, 'commands', {}).get(msg['name'])
      if not d:
        raise ValueError(f"{r.label}: no policy command {msg['name']!r}")
      c.command[msg['name']] = POL.clamp_command(d, msg['value'])
      return
    elif op == 'skill':  # the policy's manual skill (the panel's skill controls): everyone sees it in the status
      c = r.policy if r.use['policy'] else None
      if c is None or not hasattr(c, 'set_skill'):
        raise ValueError(f'{r.label}: its policy has no skills')
      c.set_skill(msg['name'], msg.get('duration'))
      return
    elif op == 'goal':
      r.set_goal(msg.get('goal'))
      for kind in list(r.cmd_at):  # a goal replaces the keys' last command
        del r.cmd_at[kind]
      await self.send_all(json.dumps({'type': 'goal', 'id': r.id, 'goal': r.goal}))
      return
    elif (ik.model is None and op.startswith('ik_')) or (base.model is None and op.startswith('base_')):
      raise ValueError(f'{r.label}: {op.split("_")[0]} is not set up')
    elif op == 'ik_config':
      ik.config = msg['config']
      ik.resolve()
    elif op == 'ik_target':  # while a marker is dragged: no echo to everyone
      c = chain()
      if c and not c['error']:
        ik.set_target(c, d, msg['pos'], msg['rot'])
      return
    elif op == 'ik_reset_target':
      if msg.get('chain') is None:
        ik.targets.clear()
      else:
        ik.targets.pop(msg['chain'], None)
      return
    elif op == 'ik_gripper':
      c = chain()
      if c:
        if msg.get('amount') is None:
          ik.toggle_gripper(c, d)
        else:
          a = msg['amount']
          clamp = lambda v: min(1.0, max(0.0, float(v)))
          ik.set_gripper(c, d, [clamp(v) for v in a] if isinstance(a, list) else clamp(a))
        self.dirty = True
      return
    elif op in ('ik_save', 'base_save'):
      (ik if op == 'ik_save' else base).save()
      for o in self.robots:  # other copies of the robot file now use it too
        if o is not r and o.key == r.key:
          (o.ik if op == 'ik_save' else o.base).saved = True
    elif op == 'ik_defaults':
      ik.restore_defaults()
    elif op == 'base_config':  # the panel edited the setup (Teleop renderBase)
      base.release(d)
      base.config = msg['config']
      base.resolve()
      base.reset()
      self.rebind_ik(r)
    elif op == 'base_defaults':
      base.release(d)
      base.restore_defaults()
      base.reset()
      self.rebind_ik(r)
    await self.broadcast_robot(r)

  def rebind_ik(self, r):
    """What the base drives changed: automatic IK chains are found again."""
    if r.ik.config and r.ik.config.get('auto'):
      r.ik.config = None
      r.ik.targets.clear()
    r.bind_controller('ik', self.model)
    r.ik.enabled = bool(r.use['ik'])

  def set_use(self, r, kind, on):
    """Switches a controller on or off (main.js app.setUse); for 'policy',
    `on` is the policy id ('' for none)."""
    m, d = self.model, self.data
    if kind not in ('policy', 'base', 'ik'):
      raise ValueError(f'no controller {kind}')
    # An automatic base setup leaves the policy's joints out: find it again.
    rebase = kind == 'policy' and r.base.config and r.base.config.get('auto')
    if rebase:
      r.base.release(d)
    if kind == 'policy':
      if r.policy:
        r.policy.unbind()
      r.use['policy'] = on or ''
      r.errors.pop('policy', None)
      r.make_policy()
      if r.use['policy']:
        r.policy.reset()
        if not r.bind_controller('policy', m):
          r.use['policy'] = ''
    else:
      r.use[kind] = bool(on)
    if rebase:
      r.base.config = None
      r.bind_controller('base', m)
      r.use['base'] = bool(r.base.config and r.base.config.get('kind')) and r.use['base'] is not False
    if kind == 'base':
      r.base.reset()
      if on:
        r.base.resolve()
      else:
        r.base.release(d)
    if kind == 'ik' and not on:
      r.ik.release(d)
    if kind != 'ik' and r.ik.config and r.ik.config.get('auto'):
      r.ik.config = None
    r.ik.targets.clear()
    r.bind_controller('ik', m)
    r.ik.enabled = bool(r.use['ik'])

  def expire_commands(self, now, gone=None):
    """Drive commands stop when their browser stops repeating them, or leaves."""
    for r in self.robots:
      for kind, (at, ws) in list(r.cmd_at.items()):
        if now - at > CMD_TIMEOUT or ws is gone:
          c = r.policy if kind == 'policy' else r.base
          if c is not None:
            c.cmd = [0.0, 0.0, 0.0]
          del r.cmd_at[kind]

  async def controller_failed(self, c, err):
    traceback.print_exception(err)
    r = next(r for r in self.robots if c in (r.policy, r.base, r.ik))
    kind = 'policy' if c is r.policy else 'base' if c is r.base else 'ik'
    if hasattr(c, 'release'):
      c.release(self.data)
    r.use[kind] = '' if kind == 'policy' else False
    if kind == 'ik':
      c.enabled = False
    r.errors[kind] = f'stopped: {err}'
    await self.broadcast_robot(r)

  async def broadcast_robot(self, r):
    await self.send_all(json.dumps({'type': 'robot', 'item': r.item()}))

  def ray(self, eye, direction):
    """The first drawn geom (groups 0-2) along a ray: its body and the hit point."""
    geom = np.zeros(1, np.int32)
    eye, direction = np.array(eye, float), np.array(direction, float)
    dist = mujoco.mj_ray(self.model, self.data, eye, direction, np.array([1, 1, 1, 0, 0, 0], np.uint8), 1, -1, geom)
    if dist < 0 or geom[0] < 0:
      return {'body': -1}
    return {'body': int(self.model.geom_bodyid[geom[0]]), 'point': (eye + dist * direction).tolist()}

  def robot(self, rid):
    r = next((r for r in self.robots if r.id == rid), None)
    if r is None:
      raise ValueError(f'no robot {rid}')
    return r

  def jumped(self):
    self.epoch += 1
    self.dirty = True

  def write_initial(self, r, at):
    if r.initial is None:
      return
    self.data.qpos[at['qpos']:at['qpos'] + r.dims['nq']] = r.initial['qpos']
    self.data.ctrl[at['ctrl']:at['ctrl'] + r.dims['nu']] = r.initial['ctrl']

  def root_free_joint(self, r):
    m, b = self.model, r.body_offset
    if m.body_jntnum[b] <= 0:
      return -1
    j = m.body_jntadr[b]
    return j if m.jnt_type[j] == 0 else -1

  def place_free(self, r, pos, quat):
    """Moves a robot on a free joint, at rest (App.placeFree)."""
    j = self.root_free_joint(r)
    if j < 0:
      raise ValueError(f'{r.label} has a fixed base')
    m, d = self.model, self.data
    adr, dof = m.jnt_qposadr[j], m.jnt_dofadr[j]
    d.qpos[adr:adr + 3] = pos
    d.qpos[adr + 3:adr + 7] = quat
    d.qvel[dof:dof + 6] = 0
    at = state_layout(self.robots)[r.id]
    if r.initial is not None:
      k = adr - at['qpos']
      r.initial['qpos'][k:k + 7] = d.qpos[adr:adr + 7]
    w, x, y, z = quat
    r.pos = [float(v) for v in pos]
    r.yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    mujoco.mj_forward(m, d)
    self.jumped()

  async def move_fixed(self, r, pos, yaw):
    """A fixed-base robot to a new attach frame: rebuilds, carrying its mocap
    bodies along (main.js moveFixed)."""
    p0, y0 = r.pos, r.yaw
    r.pos, r.yaw = [float(v) for v in pos], float(yaw)

    def carry():
      m, d, dy = self.model, self.data, r.yaw - y0
      c0, s0, c1, s1 = math.cos(-y0), math.sin(-y0), math.cos(r.yaw), math.sin(r.yaw)
      for b in range(r.body_offset, r.body_offset + r.info['nbody'] - 1):
        k = m.body_mocapid[b]
        if k < 0:
          continue
        mp, mq = d.mocap_pos[k], d.mocap_quat[k]
        lx = c0 * (mp[0] - p0[0]) - s0 * (mp[1] - p0[1])
        ly = s0 * (mp[0] - p0[0]) + c0 * (mp[1] - p0[1])
        mp[:] = [r.pos[0] + c1 * lx - s1 * ly, r.pos[1] + s1 * lx + c1 * ly, mp[2] - p0[2] + r.pos[2]]
        hw, hz = math.cos(dy / 2), math.sin(dy / 2)
        w, x, y, z = mq
        mq[:] = [hw * w - hz * z, hw * x - hz * y, hw * y + hz * x, hw * z + hz * w]
      mujoco.mj_forward(m, d)

    try:
      await self.set_scene(list(self.robots), after=carry)
    except Exception:
      r.pos, r.yaw = p0, y0
      raise

  # -- Shift+drag (inspect.js Perturb), one per client ----------------------

  OMEGA, MAX_ACC = 15, 100

  def set_perturb(self, ws, msg):
    b = int(msg.get('body', -1))
    if b <= 0 or self.model is None or b >= self.model.nbody or msg.get('target') is None:
      self.perturbs.pop(ws, None)
    else:
      prev = self.perturbs.get(ws)
      self.perturbs[ws] = {'body': b, 'local': np.array(msg['local'], float), 'target': np.array(msg['target'], float),
                           'prev': prev['prev'] if prev and prev['body'] == b else None}

  def apply_perturbs(self):
    m, d = self.model, self.data
    d.xfrc_applied[:] = 0
    for p in self.perturbs.values():
      b = p['body']
      pt = d.xpos[b] + d.xmat[b].reshape(3, 3) @ p['local']
      v = (pt - p['prev'][0]) / (d.time - p['prev'][1]) if p['prev'] and d.time > p['prev'][1] else np.zeros(3)
      p['prev'] = (pt, d.time)
      mass, w = m.body_subtreemass[b], self.OMEGA
      f = mass * (w * w * (p['target'] - pt) - 2 * w * v)
      n, cap = np.linalg.norm(f), mass * self.MAX_ACC
      if n > cap:
        f *= cap / n
      d.xfrc_applied[b, :3] += f
      d.xfrc_applied[b, 3:] += np.cross(pt - d.xipos[b], f)

  # -- stepping and broadcasting -------------------------------------------

  def advance(self, target):
    """Steps the world towards sim time `target`, for STEP_BUDGET of wall
    time at most, ticking the controllers every IK.CONTROL_DT. Runs in a
    thread of its own (run) while the event loop goes on with the network
    (mj_step lets go of the GIL). The controllers tick one after the other:
    they are mostly Python, which threads would only take turns at.
    Returns the goals reached [(robot, goal)] and the controllers that
    failed [(controller, error)], for run to tell the clients."""
    m, d = self.model, self.data
    t0 = time.perf_counter()
    reached, failed = [], []
    controllers = [c for r in self.robots for c in r.controllers()]
    substeps = POL.Substeps(controllers)
    while d.time < target and time.perf_counter() - t0 < STEP_BUDGET:
      if controllers and d.time >= self.next_control - 1e-9:
        for r in self.robots:
          goal = r.follow_goal(d)
          if goal:
            reached.append((r, goal))
        for c in list(controllers):
          try:
            c.step(d)
          except Exception as err:  # a broken controller stops alone
            failed.append((c, err))
            controllers.remove(c)
        if failed:
          substeps.done()
          substeps = POL.Substeps(controllers)
        self.next_control += IK.CONTROL_DT
      if self.perturbs or self.perturbed:  # once more after the last one let go: clears its force
        self.apply_perturbs()
        self.perturbed = bool(self.perturbs)
      substeps.step(d)  # software servos
      mujoco.mj_step(m, d)
    substeps.done()
    return reached, failed

  async def run(self):
    next_broadcast, next_status, sent = 0, 0, 0
    while True:
      now = time.monotonic()
      if self.model is not None and not self.paused:
        async with self.stepping:
          speed = self.physics['speed']
          target = (now - self.sim_start) * speed if speed else math.inf
          reached, failed = await asyncio.to_thread(self.advance, target)
          if self.data.time < target:  # slower than asked: let it slip
            self.resync()
          self.dirty = True
          for r, goal in reached:
            await self.send_all(json.dumps({'type': 'goal', 'id': r.id, 'goal': None, 'arrived': goal}))
          for c, err in failed:
            await self.controller_failed(c, err)
      if self.dirty and now >= next_broadcast and self.model is not None:
        self.dirty = False
        hz = self.physics['stateHz']
        next_broadcast = max(next_broadcast + 1 / hz, now)
        sent += 1
        await self.send_all(self.state_bytes(ctrl=sent % max(1, round(hz / CTRL_HZ)) == 0))
      if now >= next_status and self.model is not None:
        next_status = now + 1 / STATUS_HZ
        self.expire_commands(now)
        status = {r.id: r.status() for r in self.robots if r.controllers()}
        if status:
          await self.send_all(json.dumps({'type': 'status', 'robots': status}))
      # Stepping runs on a thread, so the network goes on meanwhile: at the
      # "max" speed there is no need to wait between rounds.
      await asyncio.sleep(0 if self.physics['speed'] == 0 and not self.paused else 0.004)

  def resync(self):
    """Sim time follows wall time at the physics speed from now on:
    target = (now - sim_start) * speed."""
    self.sim_start = time.monotonic() - (self.data.time if self.data is not None else 0) / (self.physics['speed'] or 1)

  def state_bytes(self, ctrl=True):
    m, d = self.model, self.data
    head = np.array([self.version, self.epoch, m.nq, m.nmocap, m.nu if ctrl else 0, 0], np.int32)
    parts = [d.qpos, d.mocap_pos.reshape(-1), d.mocap_quat.reshape(-1)] + ([d.ctrl] if ctrl else [])
    return head.tobytes() + np.float64(d.time).tobytes() + np.concatenate(parts).astype(np.float32).tobytes()

  def world_tables(self):
    """What the page's panels read from the model (inspect.js, teleop.js)."""
    m = self.model
    names = lambda obj, n: [mujoco.mj_id2name(m, obj, i) or '' for i in range(n)]
    out = {k: getattr(m, k).reshape(-1).tolist() for k in (
        'jnt_type', 'jnt_bodyid', 'jnt_qposadr', 'jnt_dofadr', 'jnt_limited', 'jnt_range',
        'body_jntnum', 'body_jntadr', 'body_mocapid', 'body_parentid', 'body_mass', 'body_subtreemass',
        'actuator_trntype', 'actuator_trnid', 'actuator_ctrllimited', 'actuator_ctrlrange', 'actuator_biasprm')}
    out.update({k: int(getattr(m, k)) for k in ('nbody', 'njnt', 'nu', 'nq', 'nv', 'na', 'nmocap')})
    out['names'] = {'body': names(mujoco.mjtObj.mjOBJ_BODY, m.nbody), 'joint': names(mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
                    'actuator': names(mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)}
    return out

  def scene_msg(self, stage=None):
    msg = {'type': 'scene', 'version': self.version, 'items': [r.item() for r in self.robots],
           'physics': self.physics, 'paused': self.paused, 'world': self.world_tables()}
    if stage:
      msg['stage'] = stage
    return msg

  # -- the HTTP API (scripts) -----------------------------------------------

  def find_robots(self, which):
    """Robots by id (3), prefix ('r3') or label, a list of those, or 'all':
    every robot with a switched-on walking policy or base."""
    if which in (None, 'all'):
      return [r for r in self.robots if r.locomotion()]
    out = []
    for w in which if isinstance(which, list) else [which]:
      r = next((r for r in self.robots if w in (r.id, f'r{r.id}', r.label, str(r.id))), None)
      if r is None:
        raise ValueError(f'no robot {w!r}')
      out.append(r)
    return out

  def robot_info(self, r):
    pose = GOAL.body_pose(self.data, r.body_offset) if self.data is not None else None
    res = r.goal_result and {k: r.goal_result[k] for k in ('phase', 'dist', 'dyaw')}
    return {'id': r.id, 'name': f'r{r.id}', 'label': r.label, 'rel': r.rel, 'drives': bool(r.locomotion()),
            'pose': pose, 'goal': r.goal, 'progress': res, 'arrived': r.arrived}

  async def set_goals(self, robots, goal=None, forward=0.0, left=0.0, turn=0.0):
    """Each robot's goal: `goal` ({x, y, yaw}, world frame) if given, else
    the robot's pose moved `forward` / `left` (m, in its own frame) and
    turned by `turn` (rad); goal None and no move clears it. Like the
    Teleop panel's goal marker: every browser sees it."""
    for r in robots:
      if goal is None and not (forward or left or turn):
        g = None
      elif goal is not None:
        g = goal
      else:
        p = GOAL.body_pose(self.data, r.body_offset)
        c, s = math.cos(p['yaw']), math.sin(p['yaw'])
        g = {'x': p['x'] + c * forward - s * left, 'y': p['y'] + s * forward + c * left, 'yaw': p['yaw'] + turn}
      r.set_goal(g)
      for kind in list(r.cmd_at):  # a goal replaces the keys' last command
        del r.cmd_at[kind]
      await self.send_all(json.dumps({'type': 'goal', 'id': r.id, 'goal': r.goal}))

  async def broadcast_scene(self, stage=None):
    await self.send_all(json.dumps(self.scene_msg(stage)))
    await self.send_all(self.state_bytes())

  async def broadcast_meta(self):
    await self.send_all(json.dumps({'type': 'meta', 'items': [r.item() for r in self.robots],
                                    'physics': self.physics, 'paused': self.paused}))
    self.dirty = True

  async def send_all(self, payload):
    for ws in list(self.clients):
      try:
        if isinstance(payload, str):
          await ws.send_str(payload)
        else:
          await ws.send_bytes(payload)
      except (ConnectionError, RuntimeError):
        self.clients.discard(ws)


# ---------------------------------------------------------------------------

world = World()


async def ws_handler(request):
  ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
  await ws.prepare(request)
  world.clients.add(ws)
  print(f'client joined ({len(world.clients)} connected)')
  await ws.send_str(json.dumps(world.scene_msg()))
  if world.model is not None:
    await ws.send_bytes(world.state_bytes())
  try:
    async for msg in ws:
      if msg.type != WSMsgType.TEXT:
        continue
      req = json.loads(msg.data)
      if req.get('op') == 'ping':
        await ws.send_str(json.dumps({'type': 'pong', 't': req.get('t')}))
        continue
      try:
        reply = await world.handle(ws, req)
        if 'rid' in req:
          await ws.send_str(json.dumps({'type': 'reply', 'rid': req['rid'], **(reply or {})}))
      except Exception as err:  # a bad request fails alone
        print(f'{req.get("op")} failed: {err}')
        await ws.send_str(json.dumps({'type': 'error', 'op': req.get('op'), 'msg': str(err), 'rid': req.get('rid')}))
  finally:
    world.clients.discard(ws)
    async with world.stepping:
      world.perturbs.pop(ws, None)
      world.expire_commands(time.monotonic(), gone=ws)
    print(f'client left ({len(world.clients)} connected)')
  return ws


@web.middleware
async def no_store(request, handler):
  resp = await handler(request)
  resp.headers.setdefault('Cache-Control', 'no-store')
  return resp


@web.middleware
async def cors(request, handler):
  """Lets a page from anywhere (the static site on GitHub Pages) fetch from
  here: the asset files, /render, /server.json, the API. Behind Binder's
  jupyter-server-proxy the token in the URL is what guards the server; on
  its own the server was never guarded (WebSockets ignore CORS anyway)."""
  origin = request.headers.get('Origin')
  if request.method == 'OPTIONS' and origin:  # preflight (a POST with JSON)
    resp = web.Response(status=204, headers={
      'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': request.headers.get('Access-Control-Request-Headers', 'Content-Type'),
      'Access-Control-Max-Age': '600'})
  else:
    try:
      resp = await handler(request)
    except web.HTTPException as err:  # the page reads the error's text too
      if origin:
        err.headers['Access-Control-Allow-Origin'] = origin
      raise
  if origin:
    resp.headers['Access-Control-Allow-Origin'] = origin
    resp.headers['Vary'] = 'Origin'
  return resp


async def index(_request):
  return web.FileResponse(PUBLIC / 'index.html')


async def render(request):
  rel = request.query.get('rel', '')
  try:
    info = await asyncio.to_thread(model_info, rel)
  except Exception as err:
    raise web.HTTPNotFound(text=f'{rel}: {err}')
  # Cached by the browser, checked again on every use (ETag).
  headers = {'Cache-Control': 'no-cache', 'ETag': info['etag']}
  if request.headers.get('If-None-Match') == info['etag']:
    return web.Response(status=304, headers=headers)
  return web.Response(body=info['render'], content_type='application/octet-stream',
                      headers={**headers, 'Content-Encoding': 'gzip', 'X-Raw-Size': str(info['render_size'])})


async def api_robots(_request):
  """[{id, name, label, rel, drives, pose {x, y, yaw}, goal, progress, arrived}]"""
  return web.json_response([world.robot_info(r) for r in world.robots])


async def api_goal(request):
  """Goal markers from a script. JSON body:
    robots   'all' (default: every robot that walks or drives), an id, 'r3',
             a label, or a list of those
    forward, left (m), turn (rad)   relative to each robot's current pose, or
    goal     {x, y, yaw} in the world frame, the same for all, or
    neither (or goal null)          clears their goals: they stop
  Replies with the robots as GET /api/robots. A robot's `arrived` turns true
  when it reaches its goal (which is then let go)."""
  try:
    msg = await request.json()
    robots = world.find_robots(msg.get('robots', 'all'))
    goal = msg.get('goal')
    if goal is not None:
      goal = {k: float(goal[k]) for k in ('x', 'y', 'yaw')}
    async with world.stepping:
      await world.set_goals(robots, goal, *(float(msg.get(k, 0)) for k in ('forward', 'left', 'turn')))
  except (ValueError, KeyError, TypeError) as err:
    raise web.HTTPBadRequest(text=str(err))
  return web.json_response([world.robot_info(r) for r in robots])


async def api_stages(_request):
  """[{id, name, saved, count, thumb}], newest first."""
  return web.json_response(await asyncio.to_thread(list_stages))


async def api_stage(request):
  """A stage as a bundle (GET), or deleted (DELETE)."""
  sid = request.match_info['id']
  try:
    if request.method == 'DELETE':
      await asyncio.to_thread(delete_stage, sid)
      return web.json_response({'id': sid})
    return web.json_response(await asyncio.to_thread(read_stage, sid))
  except (ValueError, OSError) as err:
    raise web.HTTPNotFound(text=str(err))


async def api_capture_stage(request):
  """The world as a new stage bundle, not kept (the page saves it in its
  browser); the JSON body has what only the page knows: {name, thumbnail (a
  JPEG data URL), view, cameras, on}."""
  try:
    extra = await request.json()
    async with world.stepping:
      return web.json_response(world.stage_bundle(extra))
  except (ValueError, KeyError, TypeError) as err:
    raise web.HTTPBadRequest(text=str(err))


async def api_publish_stage(request):
  """Publishes a stage bundle: {bundle, author: {name, email}} (both
  optional); returns its summary, with `commit` (publish_stage)."""
  try:
    body = await request.json()
    return web.json_response(await asyncio.to_thread(publish_stage, body['bundle'], body.get('author')))
  except (ValueError, KeyError, TypeError) as err:
    raise web.HTTPBadRequest(text=str(err))


async def server_json(_request):
  return web.json_response({'server': 'mujoco-python', 'mujoco': mujoco.__version__})


async def main():
  sys.stdout.reconfigure(line_buffering=True)
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 4007
  await world.set_scene([])
  app = web.Application(middlewares=[cors, no_store], client_max_size=32 << 20)  # stage bundles
  app.router.add_get('/ws', ws_handler)
  app.router.add_get('/server.json', server_json)
  app.router.add_get('/api/robots', api_robots)
  app.router.add_post('/api/goal', api_goal)
  app.router.add_get('/api/stages', api_stages)
  app.router.add_post('/api/stages/capture', api_capture_stage)
  app.router.add_post('/api/stages/publish', api_publish_stage)
  app.router.add_get('/api/stages/{id}', api_stage)
  app.router.add_delete('/api/stages/{id}', api_stage)
  app.router.add_get('/render', render)
  app.router.add_get('/', index)
  # aiohttp guesses static types from Python's built-in table, which before
  # 3.13 has no .webp (the panel thumbnails): served as octet-stream,
  # Safari does not show them.
  web_fileresponse.CONTENT_TYPES.add_type('image/webp', '.webp')
  app.router.add_static('/', PUBLIC, follow_symlinks=True)
  runner = web.AppRunner(app)
  await runner.setup()
  await web.TCPSite(runner, port=port).start()
  print(f'shared world on http://localhost:{port}/ (MuJoCo {mujoco.__version__}, serving {PUBLIC})')
  await world.run()


if __name__ == '__main__':
  asyncio.run(main())
