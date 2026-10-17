"""Teleoperation by inverse kinematics on the server: a port of public/ik.js
(see there for how chains, dofs, followers, tendons and software servos
work). The two must stay in step; the browser runs ik.js in local mode and
talks to this one in network mode (public/remote.js).

Differences: configs are saved on the server (for everyone, SavedConfigs)
instead of in the browser, and targets can be set directly in the frame of
the chain's reference body (what the browser sends while a marker is
dragged).
"""

import json
import math
import pathlib
import re

import mujoco
import numpy as np

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
IK_DEFAULTS = json.loads((PUBLIC / 'ik_defaults.json').read_text())

CONTROL_DT = 0.02  # s, like groot.js
MIN_CHAIN = 3
GRIPPER_JOINT = re.compile(r'finger|gripper|jaw|thumb|knuckle|claw|hand_|_hand|palm', re.I)
TCP_SITES = [re.compile(p, re.I) for p in (
    r'tcp', r'pinch', r'grasp', r'gripper|end_?effector|eef|(^|[_/])ee(_|$)', r'hand_center|palm', r'attachment')]
LAMBDA = 0.05
MAX_STEP = (0.03, 0.15)
MAX_SPEED = {2: 0.15, 3: 2.0}
NULL_GAIN = 0.5
ROT_WEIGHT = 0.3
SERVO_OMEGA = 20

OBJ = mujoco.mjtObj
TRN_JOINT, TRN_TENDON = int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_TENDON)
EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)
WRAP_JOINT = int(mujoco.mjtWrap.mjWRAP_JOINT)


class SavedConfigs:
  """IK configs saved from the Teleop panel, per robot file, in a JSON file."""

  def __init__(self, path):
    self.path = pathlib.Path(path)
    try:
      self.data = json.loads(self.path.read_text())
    except (OSError, ValueError):
      self.data = {}

  def get(self, key):
    return self.data.get(key)

  def put(self, key, config):
    if config is None:
      self.data.pop(key, None)
    else:
      self.data[key] = config
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.path.write_text(json.dumps(self.data, indent=2))


def rot_vec(R):
  """Rotation vector of a row-major 3x3 rotation (flat)."""
  v = np.array([R[7] - R[5], R[2] - R[6], R[3] - R[1]])
  s, c = np.linalg.norm(v) / 2, (R[0] + R[4] + R[8] - 1) / 2
  if s < 1e-9:
    return np.zeros(3)
  return v / (2 * s) * math.atan2(s, c)


class IkController:
  def __init__(self, key, prefix, saved):
    self.key, self.prefix, self.store = key, prefix, saved
    self.name = 'IK'
    cfg = saved.get(key)
    self.saved = cfg is not None
    self.config = json.loads(json.dumps(cfg if cfg is not None else IK_DEFAULTS.get(key))) if (cfg or key in IK_DEFAULTS) else None
    self.enabled = True
    self.chains = []
    self.targets = {}        # chain name -> {pos, rot (flat 3x3), home} in the chain's ref body frame
    self.servo_targets = {}  # joint -> software servo target
    self.applied = set()
    self.model = None

  @property
  def has_defaults(self):
    return self.key in IK_DEFAULTS

  def save(self):
    self.store.put(self.key, self.config)
    self.saved = True

  def restore_defaults(self):
    self.store.put(self.key, None)
    self.saved = False
    self.config = json.loads(json.dumps(IK_DEFAULTS[self.key])) if self.has_defaults else self.auto_config()
    self.targets.clear()
    self.resolve()

  def reset(self):
    self.targets.clear()
    self.servo_targets.clear()

  def bind(self, model, owner=lambda j: None, bodies=None):
    self.model = model
    self.owner = owner
    self.bodies = bodies or (1, model.nbody)
    self.scratch = mujoco.MjData(model)
    if self.config is None:
      self.config = self.auto_config()
    self.resolve()

  # -- names ----------------------------------------------------------------

  def id(self, obj, name):
    return mujoco.mj_name2id(self.model, getattr(OBJ, obj), self.prefix + name)

  def name_of(self, obj, i):
    n = mujoco.mj_id2name(self.model, getattr(OBJ, obj), i) or ''
    return n[len(self.prefix):] if n.startswith(self.prefix) else n

  def in_robot(self, b):
    return self.bodies[0] <= b < self.bodies[1]

  def ancestors(self, tip):
    out, b = [], tip
    while self.in_robot(b):
      out.append(b)
      b = self.model.body_parentid[b]
    return out

  def tip_sites(self, tip):
    m = self.model
    top = tip
    while m.body_jntnum[top] == 0 and self.in_robot(m.body_parentid[top]):
      top = m.body_parentid[top]
    anc = self.ancestors(tip)
    above = anc[:anc.index(top) + 1] if top in anc else []

    def rigid(b):
      if b in above:
        return True
      while self.in_robot(b):
        if b == tip:
          return True
        if m.body_jntnum[b] > 0:
          return False
        b = m.body_parentid[b]
      return False

    out = [(i, self.name_of('mjOBJ_SITE', i)) for i in range(m.nsite) if rigid(m.site_bodyid[i])]
    return [(i, n) for i, n in out if n]

  # -- what moves the joints ------------------------------------------------

  def servos(self):
    m, servo = self.model, {}
    for a in range(m.nu):
      if m.actuator_trntype[a] != TRN_JOINT or m.actuator_biasprm[a, 1] >= 0:
        continue
      servo[int(m.actuator_trnid[a, 0])] = a
    return servo

  def drives(self):
    m, servo, tendon = self.model, self.servos(), {}
    for a in range(m.nu):
      if m.actuator_trntype[a] != TRN_TENDON or m.actuator_biasprm[a, 1] >= 0:
        continue
      t, joints, fixed = int(m.actuator_trnid[a, 0]), [], True
      for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
        if m.wrap_type[w] != WRAP_JOINT:
          fixed = False
          break
        joints.append((int(m.wrap_objid[w]), float(m.wrap_prm[w])))
      if not fixed:
        continue
      ten = {'a': a, 't': t, 'gear': float(m.actuator_gear[a, 0]), 'joints': joints}
      for j, _ in joints:
        if j not in servo and j not in tendon:
          tendon[j] = ten
    return servo, tendon

  # -- automatic chains -----------------------------------------------------

  def auto_config(self):
    m, (b0, b1) = self.model, self.bodies
    servo, tendon = self.drives()
    robot_servos = any(self.in_robot(m.jnt_bodyid[j]) for j in [*servo, *tendon])

    def movable(j):
      return j in servo or j in tendon or (not robot_servos and bool(m.jnt_limited[j]))

    own, children = {}, {}
    for b in range(b0, b1):
      n = 0
      j0 = m.body_jntadr[b]
      end = j0 if b == b0 else j0 + m.body_jntnum[b]
      for j in range(j0, end) if j0 >= 0 else ():
        if m.jnt_type[j] in (2, 3) and movable(j) and not self.owner(j) and not GRIPPER_JOINT.search(self.name_of('mjOBJ_JOINT', j)):
          n += 1
      own[b] = n
      children[b] = []
      p = m.body_parentid[b]
      if self.in_robot(p):
        children[p].append(b)
    sub = {}

    def count():
      for b in range(b1 - 1, b0 - 1, -1):
        sub[b] = own[b] + sum(sub[c] for c in children[b])

    def run(leaf):
      top, n, b = leaf, 0, leaf
      while self.in_robot(b):
        n += own[b]
        if own[b]:
          top = b
        p = m.body_parentid[b]
        if not self.in_robot(p) or len([c for c in children[p] if sub[c] > 0]) > 1:
          break
        b = p
      return top, n

    def leaves():
      return [b for b in own if own[b] > 0 and sub[b] == own[b]]

    changed = True
    while changed:
      changed = False
      count()
      for leaf in leaves():
        top, n = run(leaf)
        if n >= MIN_CHAIN:
          continue
        b = leaf
        while True:
          own[b] = 0
          if b == top:
            break
          b = m.body_parentid[b]
        changed = True
    used, chains = set(), []
    for leaf in leaves():
      tip = leaf
      while len(children[tip]) == 1 and m.body_jntnum[children[tip][0]] == 0:
        tip = children[tip][0]
      name = re.sub(r'_(link|body)$', '', self.name_of('mjOBJ_BODY', tip)) or f'chain{len(used) + 1}'
      while name in used:
        name += '_'
      used.add(name)
      sites = self.tip_sites(tip)
      tcp = next((s for s in (next((x for x in sites if re.search(r, x[1])), None) for r in TCP_SITES) if s), None)
      chains.append({'name': name, 'tip': self.name_of('mjOBJ_BODY', tip), 'root': self.name_of('mjOBJ_BODY', run(leaf)[0]),
                     'site': tcp[1] if tcp else '', 'offset': [0, 0, 0], 'locked': [], 'gripper': self.gripper_for(tip, servo)})
    return {'chains': chains, 'auto': True}

  def gripper_for(self, tip, servo):
    m = self.model

    def below(j):
      return tip in self.ancestors(m.jnt_bodyid[j])

    found = []
    for a in range(m.nu):
      full = mujoco.mj_id2name(m, OBJ.mjOBJ_ACTUATOR, a) or ''
      if not full or not full.startswith(self.prefix):
        continue
      i, joints = int(m.actuator_trnid[a, 0]), []
      if m.actuator_trntype[a] == TRN_JOINT:
        joints = [i]
      elif m.actuator_trntype[a] == TRN_TENDON and m.ntendon:
        joints = [int(m.wrap_objid[w]) for w in range(m.tendon_adr[i], m.tendon_adr[i] + m.tendon_num[i]) if m.wrap_type[w] == WRAP_JOINT]
      if joints and all(below(j) for j in joints) and not (
          len(joints) == 1 and not GRIPPER_JOINT.search(self.name_of('mjOBJ_JOINT', joints[0])) and joints[0] in servo):
        found.append(self.name_of('mjOBJ_ACTUATOR', a))
    return found[0] if len(found) == 1 else ''

  # -- chains from the config -----------------------------------------------

  def resolve(self):
    m = self.model
    if m is None:
      return
    servo, tendon = self.drives()
    follows = {}
    for e in range(m.neq):
      if m.eq_type[e] != EQ_JOINT or m.eq_obj2id[e] < 0:
        continue
      follows[int(m.eq_obj1id[e])] = (int(m.eq_obj2id[e]), float(m.eq_data[e, 1]))

    def ctrl_range(a, gear, rng):
      if not m.actuator_ctrllimited[a]:
        return rng
      lo, hi = sorted(v / gear for v in m.actuator_ctrlrange[a])
      return [max(rng[0], lo), min(rng[1], hi)]

    def joint_range(j):
      return [float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])] if m.jnt_limited[j] else [-math.inf, math.inf]

    def info(j, c=1.0):
      return {'j': j, 'c': c, 'qadr': int(m.jnt_qposadr[j]), 'dadr': int(m.jnt_dofadr[j])}

    self.chains = []
    for cfg in self.config['chains']:
      chain = {'cfg': cfg, 'joints': [], 'active': [], 'followers': [], 'error': None, 'gripper': -1, 'ref': -1, 'err': None}
      self.chains.append(chain)
      chain['tip'] = self.id('mjOBJ_BODY', cfg['tip'])
      chain['root'] = self.id('mjOBJ_BODY', cfg['root'])
      if chain['tip'] < 0:
        chain['error'] = f"no body {cfg['tip']}"
        continue
      site = -1
      if cfg.get('site'):
        site = next((i for i, n in self.tip_sites(chain['tip']) if n == cfg['site']), -2)
      chain['site'] = site
      if site == -2:
        chain['error'] = f"no site {cfg['site']} fixed to {cfg['tip']}"
        continue
      if site >= 0 and chain['tip'] in self.ancestors(m.site_bodyid[site]):
        chain['tip'] = int(m.site_bodyid[site])
      path = self.ancestors(chain['tip'])
      if chain['root'] not in path:
        chain['error'] = f"{cfg['root']} is not {cfg['tip']} or above it"
        continue
      at = path.index(chain['root'])
      chain['gripper'] = self.id('mjOBJ_ACTUATOR', cfg['gripper']) if cfg.get('gripper') else -1
      by_tendon = {}
      for b in reversed(path[:at + 1]):
        j0 = m.body_jntadr[b]
        for j in range(j0, j0 + m.body_jntnum[b]) if j0 >= 0 else ():
          jtype, name = int(m.jnt_type[j]), self.name_of('mjOBJ_JOINT', j)
          joint = {'j': j, 'name': name, 'type': jtype, 'status': '', 'active': False, 'dof': None, 'locked': False}
          own, ten, dof = self.owner(j), tendon.get(j), None
          if jtype not in (2, 3):
            joint['status'] = 'free or ball joint'
          elif own:
            joint['status'] = f'driven by {own}'
          elif j in servo:
            a = servo[j]
            gear = float(m.actuator_gear[a, 0])
            dof = {'kind': 'joint', 'name': name, 'joints': [info(j)], 'k': 1.0, 'act': a, 'gear': gear, 'type': jtype,
                   'range': ctrl_range(a, gear, joint_range(j))}
          elif ten:
            joint['status'] = f"via {self.name_of('mjOBJ_ACTUATOR', ten['a'])} (tendon)"
            if ten['a'] not in by_tendon:
              rng = ctrl_range(ten['a'], ten['gear'], [-math.inf, math.inf])
              if m.tendon_limited[ten['t']]:
                rng = [max(rng[0], m.tendon_range[ten['t'], 0]), min(rng[1], m.tendon_range[ten['t'], 1])]
              dof = {'kind': 'tendon', 'name': self.name_of('mjOBJ_ACTUATOR', ten['a']), 'joints': [info(jj, c) for jj, c in ten['joints']],
                     'k': sum(c * c for _, c in ten['joints']), 'act': ten['a'], 'gear': ten['gear'], 'type': jtype, 'range': rng}
              by_tendon[ten['a']] = dof
              joint['dof'] = dof
              chain['active'].append(dof)
            else:
              joint['dof'] = by_tendon[ten['a']]
            dof = None  # already added (or shared)
          elif j in follows:
            joint['status'] = f"follows {self.name_of('mjOBJ_JOINT', follows[j][0])}"
          else:
            joint['status'] = 'software servo'
            dof = {'kind': 'servo', 'name': name, 'joints': [info(j)], 'k': 1.0, 'type': jtype, 'range': joint_range(j)}
          if dof:
            joint['dof'] = dof
            chain['active'].append(dof)
          if joint['dof'] is not None and name in cfg.get('locked', []):
            joint['locked'] = True
          if j in follows:
            joint['follows'] = follows[j]
          chain['joints'].append(joint)
      locked = [jt['dof'] for jt in chain['joints'] if jt['locked']]
      chain['active'] = [d for d in chain['active'] if not any(d is x for x in locked)]
      for jt in chain['joints']:
        jt['active'] = jt['dof'] is not None and not any(jt['dof'] is x for x in locked)
        if jt['locked']:
          jt['status'] = 'locked'
        elif jt['dof'] is not None and not jt['active']:
          jt['status'] = f"locked with {jt['dof']['name']}"
        elif jt['active'] and not jt['status']:
          jt['status'] = 'ok'
      chain['followers'] = []
      for jt in chain['joints']:
        if 'follows' not in jt or jt['dof'] is not None:
          continue
        k = next((i for i, d in enumerate(chain['active']) if len(d['joints']) == 1 and d['joints'][0]['j'] == jt['follows'][0]), -1)
        if k >= 0:
          chain['followers'].append((jt, k, jt['follows'][1]))
      if not chain['active']:
        chain['error'] = 'no joint it can move'
        continue
      chain['ref'] = int(m.body_parentid[m.jnt_bodyid[chain['active'][0]['joints'][0]['j']]])

  # -- commanded positions and software servos ------------------------------

  def value(self, dof, data):
    if dof['kind'] != 'servo':
      return data.ctrl[dof['act']] / dof['gear']
    j, qadr = dof['joints'][0]['j'], dof['joints'][0]['qadr']
    if j not in self.servo_targets:
      self.servo_targets[j] = float(data.qpos[qadr])
    return self.servo_targets[j]

  def set_value(self, dof, data, v):
    if dof['kind'] == 'servo':
      self.servo_targets[dof['joints'][0]['j']] = v
    else:
      data.ctrl[dof['act']] = v * dof['gear']

  def commanded(self, data):
    s = self.scratch
    s.qpos[:] = data.qpos
    for chain in self.chains:
      for dof in chain['active']:
        v = self.value(dof, data)
        for jj in dof['joints']:
          s.qpos[jj['qadr']] = jj['c'] * v / dof['k']
      for f, k, c in chain['followers']:
        s.qpos[m_qadr(self.model, f['j'])] = c * s.qpos[chain['active'][k]['joints'][0]['qadr']]
    mujoco.mj_kinematics(self.model, s)
    return s

  def substep(self, data):
    want = set()
    if self.enabled:
      for chain in self.chains:
        if chain['error']:
          continue
        for dof in chain['active']:
          if dof['kind'] != 'servo' or dof['joints'][0]['j'] not in self.servo_targets:
            continue
          jj, w = dof['joints'][0], SERVO_OMEGA
          mass = self.model.dof_M0[jj['dadr']]
          data.qfrc_applied[jj['dadr']] = mass * (w * w * (self.servo_targets[jj['j']] - data.qpos[jj['qadr']]) - 2 * w * data.qvel[jj['dadr']]) + data.qfrc_bias[jj['dadr']]
          want.add(jj['dadr'])
    for d in self.applied - want:
      data.qfrc_applied[d] = 0
    self.applied = want

  def release(self, data):
    for d in self.applied:
      data.qfrc_applied[d] = 0
    self.applied = set()

  # -- frames and targets ---------------------------------------------------

  @staticmethod
  def frame(d, b, offset=(0, 0, 0)):
    R = d.xmat[b].copy()
    return d.xpos[b] + R.reshape(3, 3) @ np.asarray(offset, float), R

  def tip_frame(self, d, chain):
    i = chain['site']
    if i < 0:
      return self.frame(d, chain['tip'], chain['cfg'].get('offset', (0, 0, 0)))
    return d.site_xpos[i].copy(), d.site_xmat[i].copy()

  def target(self, chain, data, s=None):
    name = chain['cfg']['name']
    t = self.targets.get(name)
    if t is not None and t['ref'] != chain['ref']:
      # Its reference body changed (a joint locked or unlocked): the same
      # pose in the world, in the new body's frame.
      fp, fr = self.frame(data, t['ref'])
      tp, tr = self.frame(data, chain['ref'])
      F, T = fr.reshape(3, 3), tr.reshape(3, 3)
      t['pos'] = T.T @ (F @ t['pos'] + fp - tp)
      t['rot'] = (T.T @ F @ np.asarray(t['rot']).reshape(3, 3)).reshape(-1)
      t['ref'] = chain['ref']
    if t is None:
      s = s if s is not None else self.commanded(data)
      tp, tr = self.tip_frame(s, chain)
      rp, rr = self.frame(s, chain['ref'])
      R = rr.reshape(3, 3)
      t = {'pos': R.T @ (tp - rp), 'rot': (R.T @ tr.reshape(3, 3)).reshape(-1), 'ref': chain['ref'],
           'home': {d['name']: self.value(d, data) for d in chain['active']}}  # by dof: a lock keeps the target
      self.targets[name] = t
    return t

  def set_target(self, chain, data, pos, rot):
    """A target in the chain's ref body frame (as the browser sends it)."""
    t = self.target(chain, data)
    t['pos'], t['rot'] = np.asarray(pos, float), np.asarray(rot, float)

  def toggle_gripper(self, chain, data):
    a, m = chain['gripper'], self.model
    if a < 0:
      return
    lo, hi = m.actuator_ctrlrange[a]
    data.ctrl[a] = lo if data.ctrl[a] > (lo + hi) / 2 else hi

  # -- solving --------------------------------------------------------------

  def step(self, data, dt=CONTROL_DT):
    if not self.enabled or not self.chains:
      return
    s = self.commanded(data)
    for chain in self.chains:
      if not chain['error']:
        self.solve_chain(chain, data, s, dt)

  def solve_chain(self, chain, data, s, dt):
    active, n = chain['active'], len(chain['active'])
    t = self.target(chain, data, s)
    tp, tr = self.tip_frame(s, chain)
    rp, rr = self.frame(data, chain['ref'])
    R = rr.reshape(3, 3)
    goal = R @ t['pos'] + rp
    ep = goal - tp
    er = rot_vec(((R @ t['rot'].reshape(3, 3)) @ tr.reshape(3, 3).T).reshape(-1))
    chain['err'] = [float(np.linalg.norm(ep)), float(np.linalg.norm(er))]

    def cap(v, mx):
      norm = np.linalg.norm(v)
      return v * mx / norm if norm > mx else v

    ep, er = cap(ep, MAX_STEP[0]), cap(er, MAX_STEP[1])

    def col(j, scale=1.0):
      a = s.xaxis[j]
      if self.model.jnt_type[j] == 2:
        return a * scale, np.zeros(3)
      return np.cross(a, tp - s.xanchor[j]) * scale, a * scale

    J = np.zeros((6, n))
    for c, dof in enumerate(active):
      for jj in dof['joints']:
        p, r = col(jj['j'], jj['c'] / dof['k'])
        J[:3, c] += p
        J[3:, c] += r
    for f, k, coef in chain['followers']:
      p, r = col(f['j'], coef)
      J[:3, k] += p
      J[3:, k] += r
    J[3:] *= ROT_WEIGHT
    e = np.concatenate([ep, er * ROT_WEIGHT])
    q = np.array([self.value(d, data) for d in active])
    home = np.array([t['home'].get(d['name'], q[c]) for c, d in enumerate(active)])
    free = np.ones(n, bool)
    for _ in range(4):
      z = np.where(free, NULL_GAIN * dt * (home - q), 0.0)
      dq = dls_step(J * free, e, z)
      again = False
      for c, dof in enumerate(active):
        nxt = q[c] + dq[c]
        if free[c] and (nxt > dof['range'][1] or nxt < dof['range'][0]):
          free[c] = False
          again = True
      if not again:
        break
    over = max([1.0] + [abs(dq[c]) / (MAX_SPEED[d['type']] * dt) for c, d in enumerate(active)])
    for c, dof in enumerate(active):
      self.set_value(dof, data, min(dof['range'][1], max(dof['range'][0], q[c] + dq[c] / over)))

  # -- what the browser shows (public/remote.js) -------------------------

  def describe(self):
    m = self.model
    chains = []
    for c in self.chains:
      chains.append({
          'cfg': c['cfg'], 'error': c['error'], 'gripper': c['gripper'], 'ref': c['ref'],
          'active': [{'act': d.get('act', -1), 'name': d['name']} for d in c['active']],
          'joints': [{'name': jt['name'], 'status': jt['status'], 'active': jt['active'], 'dof': jt['dof'] is not None,
                      'locked': jt['locked']} for jt in c['joints']],
      })
    sites = {}
    if m is not None:
      for b in range(*self.bodies):
        found = self.tip_sites(b)
        if found:
          sites[b] = found
    return {'config': self.config, 'saved': self.saved, 'hasDefaults': self.has_defaults, 'enabled': self.enabled,
            'bodies': list(self.bodies), 'chains': chains, 'tipSites': sites}

  def status(self):
    out = {'targets': {}, 'err': {}}
    for c in self.chains:
      name = c['cfg']['name']
      t = self.targets.get(name)
      if t is not None:
        out['targets'][name] = {'pos': np.asarray(t['pos']).tolist(), 'rot': np.asarray(t['rot']).tolist()}
      if c['err'] is not None:
        out['err'][name] = c['err']
    return out


def m_qadr(model, j):
  return int(model.jnt_qposadr[j])


def dls_step(J, e, z):
  """dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e, plus z projected into J's null space."""
  A = J @ J.T + LAMBDA * LAMBDA * np.eye(J.shape[0])
  pinv = lambda v: J.T @ np.linalg.solve(A, v)
  return pinv(e) + z - pinv(J @ z)
