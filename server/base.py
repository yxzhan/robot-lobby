"""Mobile-base teleoperation on the server: a port of public/js/controllers/base/base.js (see
there for the wheel model, mecanum rollers, the twist PI loop, software
servo wheels and planar bases). The two must stay in step; the browser runs
base.js in local mode and talks to this one in network mode
(public/js/net/remote.js). Configs saved from the Teleop panel live on the server
(server/saved/base.json).
"""

import copy
import math

import mujoco
import numpy as np

import shipped
from ik import CONTROL_DT, SavedConfigs  # noqa: F401  (SavedConfigs: for server.py)

BASE_DEFAULTS = shipped.load(shipped.BASE_DEFAULTS)

WHEEL_RATE = 50
GAINS = {'ff': [1, 1, 1], 'kp': [0.5, 0.5, 0.3], 'ki': [0.3, 0.3, 0.2], 'iclamp': [0.2, 0.2, 0.2], 'accel': [0.5, 0.5, 1.0]}
OBJ = mujoco.mjtObj


def mode(model, a):
  """How actuator a drives its joint: 'velocity', 'position', 'motor', or None."""
  if model.actuator_trntype[a] != 0:
    return None
  b = model.actuator_biasprm[a]
  if b[1] < 0:
    return 'position'
  if b[2] < 0:
    return 'velocity'
  return 'motor'


class BaseController:
  def __init__(self, key, prefix, saved):
    self.key, self.prefix, self.store = key, prefix, saved
    self.cmd = [0.0, 0.0, 0.0]
    self.measured = None
    cfg = saved.get(key)
    self.saved = cfg is not None
    self.config = copy.deepcopy(cfg if cfg is not None else BASE_DEFAULTS.get(key))
    self.model = None
    self.muted = None
    self.ready = False
    self.error = None
    self.joints = set()
    self.auto = None  # the setup found in the model (the panel's Kind select starts from it)
    self.reset()

  @property
  def name(self):
    return 'Planar base (pose servo)' if (self.config or {}).get('kind') == 'planar' else 'Wheeled base (twist PI)'

  @property
  def drive(self):
    return self.config['drive'] if self.ready else None

  @property
  def has_defaults(self):
    return self.key in BASE_DEFAULTS

  def reset(self):
    self.integ = [0.0, 0.0, 0.0]
    self.ref = [0.0, 0.0, 0.0]
    self.wheel_targets = None
    self.goal = None

  def save(self):
    self.store.put(self.key, self.config)
    self.saved = True

  def restore_defaults(self):
    self.store.put(self.key, None)
    self.saved = False
    self.config = copy.deepcopy(BASE_DEFAULTS[self.key]) if self.has_defaults else self.auto_config()
    self.resolve()

  def bind(self, model, owner=lambda j: None, bodies=None):
    self.muted = None  # a fresh model: nothing muted in it
    self.model = model
    self.owner = owner
    self.bodies = bodies or (1, model.nbody)
    self.auto = self.auto_config()
    if self.config is None:
      self.config = copy.deepcopy(self.auto)
    self.resolve()

  def id(self, obj, name):
    return mujoco.mj_name2id(self.model, getattr(OBJ, obj), self.prefix + name)

  def name_of(self, obj, i):
    n = mujoco.mj_id2name(self.model, getattr(OBJ, obj), i) or ''
    return n[len(self.prefix):] if n.startswith(self.prefix) else n

  def in_robot(self, b):
    return self.bodies[0] <= b < self.bodies[1]

  def owns(self, j):
    return 'base' if self.ready and j in self.joints else None

  def rest(self):
    d = mujoco.MjData(self.model)
    mujoco.mj_kinematics(self.model, d)
    return d

  @staticmethod
  def in_base(d, base, j):
    R, p = d.xmat[base].reshape(3, 3), d.xpos[base]
    return R.T @ (d.xanchor[j] - p), R.T @ d.xaxis[j]

  def auto_config(self):
    m, root = self.model, self.bodies[0]
    acts = {}
    for a in range(m.nu):
      if m.actuator_trntype[a] == 0:
        acts[int(m.actuator_trnid[a, 0])] = a
    j0 = m.body_jntadr[root]
    root_joints = list(range(j0, j0 + m.body_jntnum[root])) if j0 >= 0 else []
    free = next((j for j in root_joints if m.jnt_type[j] == 0), None)
    d = self.rest()
    if free is None:
      def pick(jtype, k):
        return next((j for j in root_joints if m.jnt_type[j] == jtype and j in acts and mode(m, acts[j]) == 'position'
                     and abs(m.jnt_axis[j, k]) > 0.9), None)
      x, y, yaw = pick(2, 0), pick(2, 1), pick(3, 2)
      if x is None or y is None or yaw is None:
        return {'kind': '', 'auto': True}
      n = lambda j: self.name_of('mjOBJ_ACTUATOR', acts[j])
      return {'kind': 'planar', 'planar': {'x': n(x), 'y': n(y), 'yaw': n(yaw)},
              'drive': {'forward': 0.5, 'back': 0.3, 'side': 0.3, 'turn': 0.8}, 'accel': [1, 1, 2], 'auto': True}
    wheels = []
    for j, a in acts.items():
      b = m.jnt_bodyid[j]
      if not self.in_robot(b) or m.jnt_type[j] != 3 or self.owner(j):
        continue
      md = mode(m, a)
      if md not in ('velocity', 'motor'):
        continue
      _, axis = self.in_base(d, root, j)
      if abs(axis[2]) > 0.3 or d.xanchor[j, 2] > 0.5:
        continue
      low = None
      for k in range(m.njnt):
        kb = m.jnt_bodyid[k]
        if m.body_parentid[kb] != b or m.jnt_type[k] != 3 or k in acts:
          continue
        if low is None or d.xpos[kb, 2] < low[0]:
          low = (d.xpos[kb, 2], k)
      roller = 0
      if low:
        _, g = self.in_base(d, root, low[1])
        dr = (axis[1], -axis[0])
        roller = int(np.sign((g[0] * axis[0] + g[1] * axis[1]) * (g[0] * dr[0] + g[1] * dr[1])))
      wheels.append({'actuator': self.name_of('mjOBJ_ACTUATOR', a), 'roller': roller, 'z': float(d.xanchor[j, 2]), 'j': j, 'a': a})
    if len(wheels) < 2:
      wheels = []
      for j in range(m.njnt):
        b = m.jnt_bodyid[j]
        if m.jnt_type[j] != 3 or j in acts or m.jnt_limited[j] or m.body_parentid[b] != root or self.owner(j):
          continue
        _, axis = self.in_base(d, root, j)
        if abs(axis[2]) > 0.3 or d.xanchor[j, 2] > 0.5:
          continue
        wheels.append({'joint': self.name_of('mjOBJ_JOINT', j), 'roller': 0, 'z': float(d.xanchor[j, 2]), 'j': j, 'a': -1})
    if len(wheels) < 2:
      return {'kind': '', 'auto': True}
    radius = round(min(w['z'] for w in wheels), 4)
    mecanum = all(w['roller'] for w in wheels)

    def wmax(a):
      if a < 0:
        return 10
      hi = abs(m.actuator_ctrlrange[a, 1] / m.actuator_gear[a, 0]) if m.actuator_ctrllimited[a] else 10
      return hi if mode(m, a) == 'velocity' else 10

    forward = round(min(0.6, 0.7 * min(wmax(w['a']) for w in wheels) * radius), 2)
    base_name = self.name_of('mjOBJ_JOINT', free)
    return {
        'kind': 'wheels', 'base': base_name if self.id('mjOBJ_JOINT', base_name) == free else '',
        'wheels': [{'joint': w['joint'], 'roller': w['roller']} if 'joint' in w else {'actuator': w['actuator'], 'roller': w['roller']}
                   for w in wheels],
        'radius': radius,
        'drive': {'forward': forward, 'back': round(0.6 * forward, 2), 'side': round(0.6 * forward, 2) if mecanum else 0, 'turn': 0.8},
        **copy.deepcopy(GAINS), 'kw': 1, 'auto': True,
    }

  def resolve(self):
    m, cfg = self.model, self.config
    self.unmute()
    self.ready = False
    self.error = None
    self.joints = set()
    if m is None or not cfg or not cfg.get('kind'):
      return
    for k, v in GAINS.items():
      cfg.setdefault(k, list(v))
    cfg.setdefault('drive', {'forward': 0.5, 'back': 0.3, 'side': 0, 'turn': 0.8})

    def actuator(name):
      a = self.id('mjOBJ_ACTUATOR', name)
      if a < 0:
        raise ValueError(f'no actuator {name}')
      md = mode(m, a)
      if not md:
        raise ValueError(f'{name} does not drive a joint')
      j = int(m.actuator_trnid[a, 0])
      self.joints.add(j)
      rng = [float(m.actuator_ctrlrange[a, 0]), float(m.actuator_ctrlrange[a, 1])] if m.actuator_ctrllimited[a] else [-math.inf, math.inf]
      return {'a': a, 'j': j, 'mode': md, 'gear': float(m.actuator_gear[a, 0]), 'range': rng,
              'qadr': int(m.jnt_qposadr[j]), 'dadr': int(m.jnt_dofadr[j])}

    try:
      if cfg['kind'] == 'planar':
        self.axes = [actuator((cfg.get('planar') or {}).get(k, '')) for k in ('x', 'y', 'yaw')]
      else:
        root = self.bodies[0]
        root_free = int(m.body_jntadr[root]) if m.body_jntnum[root] > 0 and m.jnt_type[m.body_jntadr[root]] == 0 else -1
        base = self.id('mjOBJ_JOINT', cfg['base']) if cfg.get('base') else root_free
        if base < 0 or m.jnt_type[base] != 0:
          raise ValueError(f"no free joint {cfg['base']}" if cfg.get('base') else 'the robot has no free root joint')
        self.base_v = int(m.jnt_dofadr[base])
        self.base_body = int(m.jnt_bodyid[base])
        if len(cfg.get('wheels') or []) < 2:
          raise ValueError('needs two wheels or more')
        d = self.rest()
        self.wheels = []
        for w in cfg['wheels']:
          if w.get('joint'):
            j = self.id('mjOBJ_JOINT', w['joint'])
            if j < 0 or m.jnt_type[j] != 3:
              raise ValueError(f"no hinge joint {w['joint']}")
            self.joints.add(j)
            info = {'j': j, 'mode': 'force', 'qadr': int(m.jnt_qposadr[j]), 'dadr': int(m.jnt_dofadr[j])}
            self.mute(j)
          else:
            info = actuator(w.get('actuator', ''))
            if info['mode'] == 'position':
              raise ValueError(f"{w['actuator']} is a position servo")
          anchor, axis = self.in_base(d, self.base_body, info['j'])
          dr = (axis[1], -axis[0])
          ln = math.hypot(*dr) or 1
          dn, an = (dr[0] / ln, dr[1] / ln), (axis[0] / ln, axis[1] / ln)
          s = w.get('roller') or 0
          g = (dn[0] + s * an[0], dn[1] + s * an[1])
          self.wheels.append({**info, 'x': float(anchor[0]), 'y': float(anchor[1]), 'g': g, 'gd': g[0] * dn[0] + g[1] * dn[1]})
      self.ready = True
    except ValueError as err:
      self.error = str(err)

  def measure(self, data):
    if self.config['kind'] == 'planar':
      vx, vy, w = (data.qvel[ax['dadr']] for ax in self.axes)
      th = data.qpos[self.axes[2]['qadr']]
      return [math.cos(th) * vx + math.sin(th) * vy, -math.sin(th) * vx + math.cos(th) * vy, float(w)]
    R = data.xmat[self.base_body].reshape(3, 3)
    v = R.T @ data.qvel[self.base_v:self.base_v + 3]
    return [float(v[0]), float(v[1]), float(data.qvel[self.base_v + 5])]

  def wheel_speeds(self, twist):
    vx, vy, wz = twist
    r = self.config['radius']
    return [(w['g'][0] * (vx - wz * w['y']) + w['g'][1] * (vy + wz * w['x'])) / (r * w['gd']) for w in self.wheels]

  def step(self, data, dt=CONTROL_DT):
    if not self.ready:
      return
    cfg = self.config
    clamp = lambda v, lo, hi: max(lo, min(hi, v))
    self.ref = [r + clamp(self.cmd[k] - r, -cfg['accel'][k] * dt, cfg['accel'][k] * dt) for k, r in enumerate(self.ref)]
    if cfg['kind'] == 'planar':
      if self.goal is None:
        self.goal = [data.ctrl[ax['a']] / ax['gear'] for ax in self.axes]
      th, (vx, vy, w) = self.goal[2], self.ref
      self.goal[0] += (math.cos(th) * vx - math.sin(th) * vy) * dt
      self.goal[1] += (math.sin(th) * vx + math.cos(th) * vy) * dt
      self.goal[2] += w * dt
      for k, ax in enumerate(self.axes):
        data.ctrl[ax['a']] = clamp(self.goal[k] * ax['gear'], *ax['range'])
      self.measured = self.measure(data)
      return
    m = self.measured = self.measure(data)
    if all(abs(r) < 1e-3 for r in self.ref):
      self.integ = [0.0, 0.0, 0.0]
      self.wheel_targets = [0.0] * len(self.wheels)
      for wh in self.wheels:
        if wh['mode'] == 'force':
          continue
        data.ctrl[wh['a']] = 0 if wh['mode'] == 'velocity' else clamp(-cfg.get('kw', 1) * data.qvel[wh['dadr']], *wh['range'])
      return
    twist = []
    for k in range(3):
      e = self.ref[k] - m[k]
      self.integ[k] = max(-cfg['iclamp'][k], min(cfg['iclamp'][k], self.integ[k] + cfg['ki'][k] * e * dt))
      twist.append(cfg['ff'][k] * self.ref[k] + cfg['kp'][k] * e + self.integ[k])
    self.wheel_targets = self.wheel_speeds(twist)
    for w, wh in zip(self.wheel_targets, self.wheels):
      if wh['mode'] == 'force':
        continue
      u = w * wh['gear'] if wh['mode'] == 'velocity' else cfg.get('kw', 1) * (w - data.qvel[wh['dadr']])
      data.ctrl[wh['a']] = clamp(u, *wh['range'])

  def substep(self, data):
    if not self.ready or self.config['kind'] != 'wheels' or self.wheel_targets is None:
      return
    m = self.model
    dt = m.opt.timestep
    for w, wh in zip(self.wheel_targets, self.wheels):
      if wh['mode'] != 'force':
        continue
      dd = wh['dadr']
      inertia = m.dof_M0[dd]
      kv = min(max(inertia * WHEEL_RATE, m.dof_damping[dd]), 1.5 * inertia / dt)
      data.qfrc_applied[dd] = kv * (w - data.qvel[dd]) + m.dof_damping[dd] * w + m.dof_frictionloss[dd] * np.sign(w) + data.qfrc_bias[dd]

  def release(self, data):
    for wh in getattr(self, 'wheels', None) or []:
      if wh['mode'] == 'force':
        data.qfrc_applied[wh['dadr']] = 0
    self.wheel_targets = None
    self.unmute()

  def mute(self, j):
    m = self.model
    if self.muted is None:
      self.muted = []
    for a in range(m.nu):
      if m.actuator_trntype[a] != 0 or m.actuator_trnid[a, 0] != j:
        continue
      self.muted.append((a, m.actuator_gainprm[a, :3].copy(), m.actuator_biasprm[a, :3].copy()))
      m.actuator_gainprm[a, :3] = 0
      m.actuator_biasprm[a, :3] = 0

  def unmute(self):
    if not self.muted or self.model is None:
      return
    for a, g, b in self.muted:
      self.model.actuator_gainprm[a, :3] = g
      self.model.actuator_biasprm[a, :3] = b
    self.muted = None

  # -- what the browser shows (public/js/net/remote.js) ---------------------------

  def describe(self):
    return {'config': self.config, 'saved': self.saved, 'hasDefaults': self.has_defaults, 'error': self.error,
            'ready': self.ready, 'name': self.name, 'auto': self.auto, 'bodies': list(self.bodies),
            'wheels': [{'mode': w['mode']} for w in self.wheels] if self.ready and self.config.get('kind') == 'wheels' else []}
