"""Avatars in the shared world: the server's side of public/js/world/avatar.js
(the same bodies, names, order and shapes; keep them in step). SLOTS avatars
after every robot (compile_world), each a head (touching nothing) and two
hands: a palm (a triangular prism) and a cylinder per finger bone, mocap
bodies parked out of the way while nobody uses the slot. A client in the VR
menu's Avatar mode sends its parts' poses (op avatar) and gets a slot; it is
parked again when the client leaves Avatar mode or the world.
"""

SLOTS = 4
SIDES = ('left', 'right')
GROUP = 5
HEAD_RADIUS = 0.1
PALM = [(0, 0), (0.085, 0.03), (0.075, -0.035)]
PALM_HALF = 0.012


def _finger(f, r, lengths):
  return [(f'{f}-proximal', r, lengths[0]), (f'{f}-intermediate', r * 0.9, lengths[1]), (f'{f}-distal', r * 0.8, lengths[2])]


# (part, radius, length), as avatar.js HAND_BONES.
HAND_BONES = ([('thumb-metacarpal', 0.011, 0.045), ('thumb-proximal', 0.01, 0.032), ('thumb-distal', 0.009, 0.025)]
              + _finger('index', 0.0095, [0.045, 0.026, 0.022]) + _finger('middle', 0.0095, [0.048, 0.03, 0.024])
              + _finger('ring', 0.009, [0.045, 0.028, 0.023]) + _finger('pinky', 0.008, [0.035, 0.02, 0.02]))
HAND_PARTS = ['palm'] + [b[0] for b in HAND_BONES]
PARTS = ['head'] + [f'{s}_{p}' for s in SIDES for p in HAND_PARTS]
SLOT_BODIES = len(PARTS)


# Every avatar name starts with PREFIX: the avatars' spec is attached under it
# (MuJoCo 4 warns on an empty prefix), its own names without it.
PREFIX = 'avatar'


def body_name(slot, part):
  return f'{PREFIX}{slot}_{part}'


def park(slot):
  return [2.0 * slot, 0.0, 1000.0]


def avatar_xml(slots=SLOTS):
  def geom(g, touch=1):
    return f'<geom {g} contype="0" conaffinity="{touch}" group="{GROUP}" mass="0.01"/>'
  bones = {b[0]: b for b in HAND_BONES}
  bodies = []
  for s in range(slots):
    at = ' '.join(f'{v:g}' for v in park(s))
    for p in PARTS:
      part = p.split('_', 1)[1] if '_' in p else p
      if p == 'head':
        g = geom(f'type="sphere" size="{HEAD_RADIUS:g}"', 0)  # the head touches nothing
      elif part == 'palm':
        g = geom('type="mesh" mesh="_palm"')
      else:
        _, r, length = bones[part]
        g = geom(f'type="cylinder" size="{r:g} {length / 2:g}"')
      bodies.append(f'<body name="{body_name(s, p)[len(PREFIX):]}" mocap="true" pos="{at}">{g}</body>')
  verts = ' '.join(f'{v:g}' for z in (-PALM_HALF, PALM_HALF) for x, y in PALM for v in (x, y, z))
  return (f'<mujoco model="avatars"><asset><mesh name="_palm" vertex="{verts}"/></asset>'
          f'<worldbody>{"".join(bodies)}</worldbody></mujoco>')


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
    """Op avatar's pose: head [x, y, z], head_quat [w, x, y, z]; left / right
    [[x, y, z, qw, qx, qy, qz] or None per HAND_PARTS] (missing: parked)."""
    ids = self.mocap[slot]
    home = park(slot)
    head = msg.get('head')
    data.mocap_pos[ids[0]] = head if head else home
    if msg.get('head_quat'):
      data.mocap_quat[ids[0]] = msg['head_quat']
    for k, side in enumerate(SIDES):
      parts = msg.get(side) or []
      for j in range(len(HAND_PARTS)):
        p = parts[j] if j < len(parts) else None
        i = ids[1 + k * len(HAND_PARTS) + j]
        if p and len(p) == 7:
          data.mocap_pos[i] = p[:3]
          data.mocap_quat[i] = p[3:]
        else:
          data.mocap_pos[i] = home

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
