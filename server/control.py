"""Who drives which robot in the shared world (docs/control.md).

Every connection is a client: a page introduces itself with `hello {id,
name}` (its id kept by its tab, so a reload comes back as the same client),
and gets a color. A robot with controllers (Robot.key) is free or held by one
client. Its robot ops (ROBOT_OPS, and ctrl / qpos on its actuators and
joints) need the hold; the first one from a client on a free robot takes it.
A hold ends when its holder lets go (`hold {id, on: false}`), has sent the
robot nothing for IDLE, has been gone for GRACE, or the robot goes; another
client may `take` it once the holder has been idle on it for IDLE_TAKE.
Everything else (touching the world, spawning, resetting, physics, stages)
stays open to everyone.
Each client's device ("iPhone · Safari", "Windows · Chrome") comes from its
User-Agent (device()), for the people list; one that has not named itself
is called after its kind of device and a number: "PC 1", "Mobile 2", "VR 1"
("Script 1" for a script).
"""

import re
import time
import uuid

IDLE = 5.0        # s without a robot op from its holder: the hold lets go
IDLE_TAKE = 5.0   # s: someone else may take it over (no sooner than it lets go by itself)
GRACE = 30.0      # s a client that dropped keeps its holds (a reload, a phone waking up)
COLORS = ['#4fc3f7', '#ff8a65', '#aed581', '#ba68c8', '#ffd54f', '#4db6ac', '#f06292', '#9575cd',
          '#e57373', '#81c784', '#64b5f6', '#ffb74d']
NAME = re.compile(r'[<>\n\r\t]')

# Ops on one robot (msg['id']): need its hold, and take a free one.
ROBOT_OPS = {'use', 'cmd', 'command', 'skill', 'goal', 'ik_config', 'ik_target', 'ik_reset_target', 'ik_gripper',
             'ik_save', 'ik_defaults', 'base_config', 'base_save', 'base_defaults', 'joint_cmd', 'reset_robot',
             'move', 'place'}


def device(ua, touch=False):
  """A short name for the device and browser of a User-Agent: 'iPhone ·
  Safari', 'Quest · Browser', 'Windows · Edge'. `touch`: the page has a
  touch screen (iPadOS Safari says it is a Mac)."""
  ua = ua or ''
  if re.search(r'OculusBrowser|Quest', ua):
    dev = 'Quest'
  elif 'iPhone' in ua or 'iPod' in ua:
    dev = 'iPhone'
  elif 'iPad' in ua or ('Macintosh' in ua and touch):
    dev = 'iPad'
  elif 'Android' in ua:
    dev = 'Android phone' if 'Mobile' in ua else 'Android tablet'
  elif 'CrOS' in ua:
    dev = 'Chromebook'
  elif 'Windows' in ua:
    dev = 'Windows'
  elif 'Macintosh' in ua or 'Mac OS X' in ua:
    dev = 'Mac'
  elif 'Linux' in ua:
    dev = 'Linux'
  elif re.match(r'(Python|aiohttp|curl|node|Deno)', ua, re.I) or not ua:
    return 'script'
  else:
    dev = 'unknown device'
  for pat, name in ((r'OculusBrowser', 'Browser'), (r'HeadlessChrome', 'Headless Chrome'), (r'Edg[A-Z]?/', 'Edge'),
                    (r'OPR/|Opera', 'Opera'), (r'SamsungBrowser', 'Samsung Internet'), (r'Firefox/|FxiOS', 'Firefox'),
                    (r'Chrome/|CriOS', 'Chrome'), (r'Safari/', 'Safari')):
    if re.search(pat, ua):
      return f'{dev} · {name}'
  return dev


def kind(dev):
  """PC, Mobile, VR or Script, for a default name, from device()'s."""
  if dev == 'script':
    return 'Script'
  if dev.startswith(('Quest', 'Vision')):
    return 'VR'
  if dev.startswith(('iPhone', 'iPad', 'Android')):
    return 'Mobile'
  return 'PC'


class Held(Exception):
  """A robot op from a client that does not hold the robot."""

  def __init__(self, r, holder):
    super().__init__(f'{r.label} (r{r.id}) is held by {holder.name}')
    self.info = {'code': 'held', 'id': r.id, 'holder': holder.id}


class Client:
  def __init__(self, cid, name, color):
    self.id, self.name, self.color = cid, name, color
    self.ws = None
    self.gone_at = None
    self.ua, self.device = '', ''
    self.named = False  # a name of its own (else 'PC 1' and the like)

  def public(self):
    return {'id': self.id, 'name': self.name, 'color': self.color, 'here': self.ws is not None, 'device': self.device,
            'kind': kind(self.device)}


class Control:
  def __init__(self):
    self.clients = {}  # id -> Client, also those gone for less than GRACE
    self.by_ws = {}    # ws -> Client
    self.holds = {}    # robot id -> [client id, when it last sent the robot an op (monotonic)]
    self.changed = self.moved = False  # clients / holds to send again

  def color(self):
    used = {c.color for c in self.clients.values()}
    return next((c for c in COLORS if c not in used), COLORS[len(self.clients) % len(COLORS)])

  def default_name(self, c):
    """'<kind> <n>': the least n no one else has with that kind."""
    k = kind(c.device)
    if c.name.startswith(f'{k} '):
      return c.name
    taken = {x.name for x in self.clients.values() if x is not c}
    return next(f'{k} {n}' for n in range(1, len(taken) + 2) if f'{k} {n}' not in taken)

  def join(self, ws, ua=''):
    c = Client(f'anon-{uuid.uuid4().hex[:8]}', '', self.color())
    c.ws, c.ua, c.device = ws, ua, device(ua)
    c.name = self.default_name(c)
    self.clients[c.id] = c
    self.by_ws[ws] = c
    self.changed = True
    return c

  def hello(self, ws, cid, name, touch=False):
    """The page's own id (kept by its tab) and name: a client back within
    GRACE gets its holds again."""
    c = self.by_ws[ws]
    ua = c.ua
    cid = str(cid or '')[:64]
    back = self.clients.get(cid)
    if cid and cid != c.id and (back is None or back.ws is None):
      del self.clients[c.id]
      if back is None:
        c.id = cid
        self.clients[cid] = c
      else:  # back within GRACE: its holds, color and name
        back.ws, back.gone_at = ws, None
        self.by_ws[ws] = c = back
    c.ua, c.device = ua, device(ua, bool(touch))
    self.rename(c, name)
    if not c.named:  # (an iPad is a Mobile only now that it said it has a touch screen)
      c.name = self.default_name(c)
    self.changed = self.moved = True
    return c

  def rename(self, c, name):
    name = NAME.sub('', str(name or '')).strip()[:40]
    if name:
      c.name, c.named = name, True
    self.changed = True

  def leave(self, ws):
    c = self.by_ws.pop(ws, None)
    if c is not None:
      c.ws, c.gone_at = None, time.monotonic()
      self.changed = True

  def holder(self, rid):
    h = self.holds.get(rid)
    return self.clients.get(h[0]) if h else None

  def check(self, c, op, msg, robots):
    """Whether client c may send `op` to `robots` (those with controllers it
    acts on): raises Held, or False to drop it quietly (a stop to a robot
    someone else drives). Takes free robots for c."""
    now = time.monotonic()
    for r in robots:
      h = self.holder(r.id)
      if h is not None and h is not c:
        if op == 'cmd' and not any(msg.get('cmd') or []) and msg.get('height') is None:
          return False
        raise Held(r, h)
    if op == 'remove' or (op == 'cmd' and not any(msg.get('cmd') or []) and msg.get('height') is None):
      return True  # neither takes a robot
    for r in robots:
      if r.id not in self.holds:
        self.moved = True
      self.holds[r.id] = [c.id, now]
    return True

  def claim(self, c, r):
    """c holds r (a robot it just spawned)."""
    self.holds[r.id] = [c.id, time.monotonic()]
    self.moved = True

  def take(self, c, r):
    h = self.holder(r.id)
    if h is not None and h is not c:
      idle = time.monotonic() - self.holds[r.id][1]
      if h.ws is not None and idle < IDLE_TAKE:
        raise ValueError(f'{h.name} is driving {r.label}: it can be taken over after {IDLE_TAKE:.0f} s idle')
    self.holds[r.id] = [c.id, time.monotonic()]
    self.moved = True

  def release(self, c, r):
    if self.holder(r.id) is c:
      del self.holds[r.id]
      self.moved = True

  def clear(self):
    if self.holds:
      self.holds.clear()
      self.moved = True

  def expire(self, robots):
    """Holds of robots gone, of holders idle for IDLE or gone for GRACE; a
    robot walking to its goal counts as busy."""
    now = time.monotonic()
    by_id = {r.id: r for r in robots}
    for rid, h in list(self.holds.items()):
      c, r = self.clients.get(h[0]), by_id.get(rid)
      if r is not None and r.goal and c is not None and c.ws is not None:
        h[1] = now
      if r is None or c is None or now - h[1] > IDLE or (c.ws is None and now - c.gone_at > GRACE):
        del self.holds[rid]
        self.moved = True
    for cid, c in list(self.clients.items()):
      if c.ws is None and now - c.gone_at > GRACE and not any(h[0] == cid for h in self.holds.values()):
        del self.clients[cid]
        self.changed = True

  def clients_msg(self, c):
    return {'type': 'clients', 'you': c.id, 'clients': [x.public() for x in self.clients.values()]}

  def holds_msg(self):
    now = time.monotonic()
    return {'type': 'holds', 'take_after': IDLE_TAKE,
            'holds': {str(rid): {'client': h[0], 'idle': round(now - h[1], 1)} for rid, h in self.holds.items()}}
