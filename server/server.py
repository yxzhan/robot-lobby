"""Shared-world server: MuJoCo (Python) steps one physics world, every
connected browser renders it.

    python3 server/server.py [port]   # default 4007, then open http://localhost:4007/

It serves public/ (like scripts/serve.py, Cache-Control: no-store), the
render data of each model file at /render?rel=… (render_asset.py) and a
WebSocket at /ws. The page sees /server.json and switches to network mode
(public/net.js): it loads no MuJoCo at all. It draws each model from its
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
  goal {goal: {x, y, yaw} or null} (the robot drives there: goal.py)
  ik_config {config}   ik_target {chain, pos, rot} (in the chain's ref body
  frame)   ik_reset_target {chain or null}   ik_gripper {chain}   ik_save
  ik_defaults   base_config {config}   base_save   base_defaults
Server -> client:
  {type: 'scene', version, items, physics, paused, world}   the world (`world`:
                              names, joint and actuator tables for the panels)
  {type: 'meta', items, physics, paused}             same world, new settings
  {type: 'robot', item}                     a robot's controllers changed
                              (item.use, item.errors, item.ik / base / policy:
                              their describe())
  {type: 'status', robots: {id: {ik: {targets, err}, base: {measured},
                              goal: {phase, dist, dyaw}}}}  STATUS_HZ times a second
  {type: 'goal', id, goal, arrived?}  a robot's goal changed (null: none;
                              arrived: the goal it reached, let go)
  {type: 'reset', id}            after reset (id null) / reset_robot {id}
  {type: 'reply', rid, ...} / {type: 'error', op, msg, rid?}
  {type: 'pong', t}
  binary state, STATE_HZ times a second: int32 [version, epoch, nq, nmocap,
  nu], 4 bytes padding, float64 time, then float32 [qpos, mocap_pos,
  mocap_quat, ctrl]; ctrl only in every CTRL_EVERY-th state (nu 0 in the
  others). `epoch` changes whenever the state jumps (reset, edits), so
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
saved from the panel go to server/saved/{ik,base}.json. Not yet: the World
panel's default geom properties.
"""

import asyncio
import gzip
import hashlib
import json
import math
import pathlib
import re
import sys
import time
import warnings

import mujoco
import numpy as np
from aiohttp import WSMsgType, web

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import base as BASE  # noqa: E402
import goal as GOAL  # noqa: E402
import ik as IK  # noqa: E402
import policies as POL  # noqa: E402
import render_asset  # noqa: E402

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
STATE_HZ = 30         # state broadcasts per second
STATUS_HZ = 5         # controller status (IK targets and errors) broadcasts per second
CTRL_EVERY = 6        # ctrl goes with every 6th state (5 Hz): only the panels show it
STEP_BUDGET = 0.02     # s of wall time stepping per loop iteration at most

PHYSICS_DEFAULTS = {
    'gravity': [0, 0, -9.81], 'wind': [0, 0, 0], 'magnetic': [0, -0.5, 0],
    'density': 0, 'viscosity': 0, 'impratio': 1, 'cone': 'pyramidal',
    'sleep': True,  # island sleeping (physics.js): set at compile time
}
CONES = ['pyramidal', 'elliptic']
OPTION_KEYS = [k for k in PHYSICS_DEFAULTS if k != 'sleep']

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
  <option timestep="0.002" integrator="implicitfast" {opts}>{'<flag sleep="enable"/>' if physics.get('sleep') else ''}</option>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
  </worldbody>
</mujoco>"""


def apply_options(model, physics):
  o = model.opt
  for k in ('gravity', 'wind', 'magnetic'):
    o.__setattr__(k, physics[k])
  o.density, o.viscosity, o.impratio = physics['density'], physics['viscosity'], physics['impratio']
  o.cone = max(0, CONES.index(physics['cone']) if physics['cone'] in CONES else 0)


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

  def set_goal(self, goal):
    self.goal = goal and {k: float(goal[k]) for k in ('x', 'y', 'yaw')}
    self.goal_state, self.goal_result = {}, None
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
    self.set_goal(None)
    return reached

  def status(self):
    out = {}
    if self.goal_result:
      out['goal'] = {k: self.goal_result[k] for k in ('phase', 'dist', 'dyaw')}
    if self.use['ik'] and self.ik.model is not None:
      out['ik'] = self.ik.status()
    if self.use['base'] and self.base.ready:
      out['base'] = {'measured': self.base.measured}
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
    self.dirty = True

  # -- scene changes (compiled off the event loop, stepping goes on) --------

  async def set_scene(self, robots, fresh=(), after=None):
    async with self.lock:
      model = await asyncio.to_thread(compile_world, robots, self.physics)
      self.swap(model, robots, fresh)
      if after:
        after()
      await self.broadcast_scene()

  def swap(self, model, robots, fresh):
    data = mujoco.MjData(model)
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
    self.sim_start = time.monotonic() - data.time
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
    op = msg.get('op')
    m, d = self.model, self.data
    if op == 'spawn':
      await asyncio.to_thread(model_info, msg['rel'])  # compile it off the event loop (cached)
      r = Robot(self.next_id, msg['rel'], msg.get('label') or msg['rel'], msg['pos'], msg.get('yaw', 0))
      self.next_id += 1
      await self.set_scene([*self.robots, r], [r])
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
      self.sim_start = time.monotonic() - d.time
      await self.broadcast_meta()
    elif op == 'physics':
      old, self.physics = self.physics, {k: msg['physics'].get(k, v) for k, v in PHYSICS_DEFAULTS.items()}
      apply_options(m, self.physics)
      if bool(old.get('sleep')) != bool(self.physics['sleep']):
        await self.set_scene(list(self.robots))  # compiled in (the state carries over)
      else:
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
    elif op in ('use', 'cmd', 'goal', 'ik_config', 'ik_target', 'ik_reset_target', 'ik_gripper', 'ik_save', 'ik_defaults',
                'base_config', 'base_save', 'base_defaults'):
      await self.controller_op(ws, op, self.robot(msg['id']), msg)
    else:
      raise ValueError(f'unknown op {op!r}')

  async def controller_op(self, ws, op, r, msg):
    """Teleop panel requests for robot `r` (public/remote.js)."""
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
        if msg.get('height') is not None and hasattr(c, 'height'):
          c.height = max(0.3, min(1.2, float(msg['height'])))
        r.cmd_at[msg['kind']] = (time.monotonic(), ws)
        if r.goal and any(c.cmd):  # driven with the keys: the goal is dropped
          r.set_goal(None)
          c.cmd = [max(-5.0, min(5.0, float(v))) for v in msg['cmd'][:3]]
          await self.send_all(json.dumps({'type': 'goal', 'id': r.id, 'goal': None}))
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
        ik.toggle_gripper(c, d)
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
    import traceback
    traceback.print_exc()
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

  async def run(self):
    next_broadcast, next_status, sent = 0, 0, 0
    while True:
      now = time.monotonic()
      if self.model is not None and not self.paused:
        m, d = self.model, self.data
        target, t0 = now - self.sim_start, time.perf_counter()
        controllers = [c for r in self.robots for c in r.controllers()]
        while d.time < target and time.perf_counter() - t0 < STEP_BUDGET:
          if controllers and d.time >= self.next_control - 1e-9:
            for r in self.robots:
              reached = r.follow_goal(d)
              if reached:
                await self.send_all(json.dumps({'type': 'goal', 'id': r.id, 'goal': None, 'arrived': reached}))
            for c in controllers:
              try:
                c.step(d)
              except Exception as err:  # a broken controller stops alone
                await self.controller_failed(c, err)
                controllers = [c for r in self.robots for c in r.controllers()]
            self.next_control += IK.CONTROL_DT
          if self.perturbs or self.perturbed:  # once more after the last one let go: clears its force
            self.apply_perturbs()
            self.perturbed = bool(self.perturbs)
          for c in controllers:
            if hasattr(c, 'substep'):
              c.substep(d)  # software servos
          mujoco.mj_step(m, d)
        if d.time < target:  # slower than real time: let it slip
          self.sim_start = now - d.time
        self.dirty = True
      if self.dirty and now >= next_broadcast and self.model is not None:
        self.dirty = False
        next_broadcast = max(next_broadcast + 1 / STATE_HZ, now)
        sent += 1
        await self.send_all(self.state_bytes(ctrl=sent % CTRL_EVERY == 0))
      if now >= next_status and self.model is not None:
        next_status = now + 1 / STATUS_HZ
        self.expire_commands(now)
        status = {r.id: r.status() for r in self.robots if r.controllers()}
        if status:
          await self.send_all(json.dumps({'type': 'status', 'robots': status}))
      await asyncio.sleep(0.004)

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
        'body_jntnum', 'body_jntadr', 'body_mocapid', 'body_parentid',
        'actuator_trntype', 'actuator_trnid', 'actuator_ctrllimited', 'actuator_ctrlrange', 'actuator_biasprm')}
    out.update({k: int(getattr(m, k)) for k in ('nbody', 'njnt', 'nu', 'nq', 'nv', 'na', 'nmocap')})
    out['names'] = {'body': names(mujoco.mjtObj.mjOBJ_BODY, m.nbody), 'joint': names(mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
                    'actuator': names(mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)}
    return out

  def scene_msg(self):
    return {'type': 'scene', 'version': self.version, 'items': [r.item() for r in self.robots],
            'physics': self.physics, 'paused': self.paused, 'world': self.world_tables()}

  async def broadcast_scene(self):
    await self.send_all(json.dumps(self.scene_msg()))
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
    world.perturbs.pop(ws, None)
    world.expire_commands(time.monotonic(), gone=ws)
    print(f'client left ({len(world.clients)} connected)')
  return ws


@web.middleware
async def no_store(request, handler):
  resp = await handler(request)
  resp.headers.setdefault('Cache-Control', 'no-store')
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


async def server_json(_request):
  return web.json_response({'server': 'mujoco-python', 'mujoco': mujoco.__version__})


async def main():
  sys.stdout.reconfigure(line_buffering=True)
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 4007
  await world.set_scene([])
  app = web.Application(middlewares=[no_store])
  app.router.add_get('/ws', ws_handler)
  app.router.add_get('/server.json', server_json)
  app.router.add_get('/render', render)
  app.router.add_get('/', index)
  app.router.add_static('/', PUBLIC, follow_symlinks=True)
  runner = web.AppRunner(app)
  await runner.setup()
  await web.TCPSite(runner, port=port).start()
  print(f'shared world on http://localhost:{port}/ (MuJoCo {mujoco.__version__}, serving {PUBLIC})')
  await world.run()


if __name__ == '__main__':
  asyncio.run(main())
