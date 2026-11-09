"""Avatars in the shared world: the server's side of public/js/world/avatar.js
(the same bodies, names, order and shapes; keep them in step). SLOTS avatars
after every robot (compile_world), each a head, a torso and legs (MuJoCo's
humanoid's capsules, apart: avatar.js) and two hands of 25 joint balls,
mocap bodies parked out of the way while nobody uses the slot. A client in
the VR menu's Avatar mode sends its pose (op avatar) and gets a slot; it is
parked again when the client leaves Avatar mode or the world.
"""

SLOTS = 4
SIDES = ('left', 'right')
GROUP = 5
HEAD_RADIUS = 0.09

# The humanoid's capsules (from, to, r), as avatar.js TORSO and LEGS.
TORSO = [((0, -0.07, 0), (0, 0.07, 0), 0.07), ((-0.01, -0.06, -0.12), (-0.01, 0.06, -0.12), 0.06),
         ((-0.01, -0.06, -0.26), (-0.01, 0.06, -0.26), 0.06)]
_RIGHT_LEG = [((0, -0.1, 0.817), (0, -0.09, 0.477), 0.06), ((0, -0.09, 0.417), (0, -0.09, 0.117), 0.049),
              ((-0.07, -0.1, 0.027), (0.14, -0.12, 0.027), 0.027), ((-0.07, -0.08, 0.027), (0.14, -0.06, 0.027), 0.027)]
LEGS = [((-0.02, -0.07, 0.857), (-0.02, 0.07, 0.857), 0.09)] + _RIGHT_LEG + [
    ((f[0], -f[1], f[2]), (t[0], -t[1], t[2]), r) for f, t, r in _RIGHT_LEG]
CAPSULES = {'torso': TORSO, 'legs': LEGS}

_FINGERS = [('thumb', ('metacarpal', 'phalanx-proximal', 'phalanx-distal', 'tip'))] + [
    (f'{f}-finger', ('metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip'))
    for f in ('index', 'middle', 'ring', 'pinky')]
HAND_JOINTS = ['wrist'] + [f'{f}-{j}' for f, js in _FINGERS for j in js]
PARTS = ['head', 'torso', 'legs'] + [f'{s}_{j}' for s in SIDES for j in HAND_JOINTS]
HANDS_AT = 3  # the first hand joint's index in PARTS
SLOT_BODIES = len(PARTS)


def body_name(slot, part):
  return f'avatar{slot}_{part}'


def radius(part):
  if part == 'head':
    return HEAD_RADIUS
  if part.endswith('wrist'):
    return 0.022
  if part.endswith('metacarpal'):
    return 0.014
  if part.endswith('tip'):
    return 0.008
  return 0.01


def park(slot):
  return [2.0 * slot, 0.0, 1000.0]


def avatar_xml(slots=SLOTS):
  bodies = []
  for s in range(slots):
    at = ' '.join(f'{v:g}' for v in park(s))
    for p in PARTS:
      tail = f'contype="0" conaffinity="1" group="{GROUP}" mass="0.01"/>'
      if p in CAPSULES:
        geoms = ''.join(f'<geom type="capsule" fromto="{" ".join(f"{v:g}" for v in (*a, *b))}" size="{r:g}" {tail}'
                        for a, b, r in CAPSULES[p])
      else:
        geoms = f'<geom type="sphere" size="{radius(p)}" {tail}'
      bodies.append(f'<body name="{body_name(s, p)}" mocap="true" pos="{at}">{geoms}</body>')
  return f'<mujoco model="avatars"><worldbody>{"".join(bodies)}</worldbody></mujoco>'


class Avatars:
  """The slots of a compiled world: client id -> slot, the mocap ids of each
  slot's parts, the pose an op avatar sets."""

  def __init__(self):
    self.owners = {}   # slot -> client id
    self.mocap = []    # slot -> [mocap id per part, PARTS' order]

  def bind(self, model, mujoco):
    self.mocap = []
    for s in range(SLOTS):
      ids = []
      for p in PARTS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(s, p))
        ids.append(int(model.body_mocapid[b]) if b >= 0 else -1)
      self.mocap.append(ids)

  def slot_of(self, cid, take=True):
    for s, c in self.owners.items():
      if c == cid:
        return s
    if not take:
      return None
    free = [s for s in range(len(self.mocap)) if s not in self.owners]
    if not free:
      return None
    self.owners[free[0]] = cid
    return free[0]

  def set(self, data, slot, msg):
    """Op avatar's pose: head, torso, legs [x, y, z] with head_quat, torso_quat,
    legs_quat [w, x, y, z]; left / right [[x, y, z] or None per hand joint]
    (missing: parked)."""
    ids = self.mocap[slot]
    home = park(slot)
    for i, part in enumerate(('head', 'torso', 'legs')):
      p = msg.get(part)
      data.mocap_pos[ids[i]] = p if p else home
      if msg.get(f'{part}_quat'):
        data.mocap_quat[ids[i]] = msg[f'{part}_quat']
    for k, side in enumerate(SIDES):
      joints = msg.get(side) or []
      for j in range(len(HAND_JOINTS)):
        p = joints[j] if j < len(joints) else None
        data.mocap_pos[ids[HANDS_AT + k * len(HAND_JOINTS) + j]] = p if p else home

  def release(self, data, cid):
    """The client's slot parked and free again; True if it had one."""
    s = self.slot_of(cid, take=False)
    if s is None:
      return False
    if data is not None and s < len(self.mocap):
      for i in self.mocap[s]:
        if i >= 0:
          data.mocap_pos[i] = park(s)
    del self.owners[s]
    return True
