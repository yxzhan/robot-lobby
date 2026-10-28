"""The lobby's robots for py_trees trees run in the Behavior panel (Pyodide,
public/js/pybt.js): behaviours that drive robots by their goal markers, the
same as the text form's leaves (public/js/bt.js runs them), and what the
world looks like, for conditions of one's own.

    import py_trees
    from lobby import GoTo, Move, Wait

    root = py_trees.composites.Sequence("Patrol", memory=True, children=[
        GoTo("r1", x=1, y=0),
        Wait(2),
        Move("all", forward=0.5),
    ])

The script defines `root` (a behaviour) or `tree` (a
py_trees.trees.BehaviourTree); the panel ticks it every 0.1 s of simulated
time while the simulation runs (alike at any speed of the Physics panel).

Behaviours (each also takes name=...):
  GoTo(robot, x, y, yaw=None)            to a pose on the floor (m, rad)
  Move(robot, forward=0, left=0, turn=0) relative to where each robot is
  Approach(robot, target, distance=0.4)  to `distance` from any spawned thing
  Follow(robot, target, distance=0.4)    the same, and keeps following (RUNNING)
  Stop(robot)                            clears the goals
  Wait(seconds)                          simulated time (py_trees.timers.Timer
                                         counts wall time: not paused, not sped up)
robot: a name ('r1'), a label (every robot of it), 'all', or a list. A move
succeeds when its goals are reached, fails when one is taken from it, and
clears its goals when interrupted. Angles in rad, or a string '90deg'.

The world: robots() lists {name, label, drives, pose: {x, y, yaw}, goal,
arrived} of every spawned thing, pose(name) one's pose, distance(a, b)
between two, body(name) any body of the world ('r1/ball', or 'ball': the
first so named) as {x, y, z, yaw, up}, upright(robot) the z of its trunk's
up axis (1 standing, 0 lying on its side), time() the simulated time (s).
Acting outside the behaviours: set_goal(robot, x, y, yaw) and stop(robot)
(its goal marker, as GoTo / Stop do), use_skill(robot, skill, duration=None)
(a policy skill, e.g. the Microduck's 'kick_right'), reset_world() (the
toolbar's Reset: everything back to where it was spawned, time back to 0),
place(robot, x, y, yaw) (a robot on a free joint moved there at once, at
rest; Reset still puts it where it was spawned, unless spawn=True).
"""

import math

import py_trees
from py_trees.common import Status

import lobby_js  # registered by pybt.js: check, leaf, robots, pose, time, body, upright, skill, set_goal, stop, reset, place

__all__ = ['GoTo', 'Move', 'Approach', 'Follow', 'Stop', 'Wait', 'robots', 'pose', 'time', 'distance',
           'body', 'upright', 'set_goal', 'stop', 'use_skill', 'reset_world', 'place']

_STATUS = {'running': Status.RUNNING, 'success': Status.SUCCESS, 'failure': Status.FAILURE}


def _robots_arg(robot):
  return ','.join(robot) if isinstance(robot, (list, tuple, set)) else str(robot)


class _Leaf(py_trees.behaviour.Behaviour):
  """Runs bt.js's leaf TYPE with these parameters."""
  TYPE = None

  def __init__(self, name, **params):
    super().__init__(name)
    self._params = {k: v for k, v in params.items() if v is not None}
    lobby_js.check(self.TYPE, self._params)  # a wrong parameter fails here, at the line making it
    self._leaf = None

  def initialise(self):
    if self._leaf is not None:
      self._leaf.halt()
    self._leaf = lobby_js.leaf(self.TYPE, self._params)

  def update(self):
    status = _STATUS[self._leaf.tick()]
    self.feedback_message = self._leaf.info
    return status

  def terminate(self, new_status):
    if new_status == Status.INVALID and self._leaf is not None:  # interrupted: its goals go
      self._leaf.halt()
    if new_status == Status.INVALID:
      self._leaf = None


def _label(*parts):
  return ' '.join(str(p) for p in parts if p)  # (the parts not given: None, False, 0)


class GoTo(_Leaf):
  TYPE = 'GoTo'

  def __init__(self, robot, x, y, yaw=None, name=None):
    super().__init__(name or _label('GoTo', _robots_arg(robot), f'x={x}', f'y={y}', yaw is not None and f'yaw={yaw}'),
                     robot=_robots_arg(robot), x=x, y=y, yaw=yaw)


class Move(_Leaf):
  TYPE = 'Move'

  def __init__(self, robot, forward=0, left=0, turn=0, name=None):
    super().__init__(name or _label('Move', _robots_arg(robot), forward and f'forward={forward}', left and f'left={left}',
                                    turn and f'turn={turn}'),
                     robot=_robots_arg(robot), forward=forward, left=left, turn=turn)


class Approach(_Leaf):
  TYPE = 'Approach'

  def __init__(self, robot, target, distance=0.4, name=None):
    super().__init__(name or _label('Approach', _robots_arg(robot), target, f'distance={distance}'),
                     robot=_robots_arg(robot), target=str(target), distance=distance)


class Follow(_Leaf):
  TYPE = 'Follow'

  def __init__(self, robot, target, distance=0.4, name=None):
    super().__init__(name or _label('Follow', _robots_arg(robot), target, f'distance={distance}'),
                     robot=_robots_arg(robot), target=str(target), distance=distance)


class Stop(_Leaf):
  TYPE = 'Stop'

  def __init__(self, robot, name=None):
    super().__init__(name or _label('Stop', _robots_arg(robot)), robot=_robots_arg(robot))


class Wait(_Leaf):
  TYPE = 'Wait'

  def __init__(self, seconds, name=None):
    super().__init__(name or f'Wait {seconds}', seconds=seconds)


def robots():
  return lobby_js.robots().to_py()


def pose(name):
  """{x, y, yaw} of a spawned thing: a name, a label or a part of one ('ball')."""
  p = lobby_js.pose(str(name))
  if p is None:
    raise KeyError(f'no {name}')
  return p.to_py()


def distance(a, b):
  """Between two spawned things on the floor (m)."""
  pa, pb = pose(a), pose(b)
  return math.hypot(pa['x'] - pb['x'], pa['y'] - pb['y'])


def time():
  return lobby_js.time()


def body(name):
  """{x, y, z, yaw, up} of a body by its name ('r1/ball', or 'ball'), or None."""
  b = lobby_js.body(str(name))
  return None if b is None else b.to_py()


def upright(robot):
  """The z of `robot`'s trunk up axis: 1 standing, 0 lying on its side (None: no such robot)."""
  return lobby_js.upright(str(robot))


def set_goal(robot, x, y, yaw=None):
  """Sends `robot` (a name or a label) to a pose, as its goal marker; yaw: facing the way there if None."""
  if yaw is None:
    p = pose(robot)
    yaw = math.atan2(y - p['y'], x - p['x'])
  lobby_js.set_goal(str(robot), float(x), float(y), float(yaw))


def stop(robot):
  """Clears `robot`'s goal: it stops where it is."""
  lobby_js.stop(str(robot))


def use_skill(robot, skill, duration=None):
  """Runs a skill of `robot`'s policy (e.g. 'kick_right'), for `duration` s if given."""
  lobby_js.skill(str(robot), str(skill), None if duration is None else float(duration))


def reset_world():
  """The toolbar's Reset: every spawned thing back where it was spawned, time back to 0."""
  lobby_js.reset()


def place(robot, x, y, yaw, spawn=False):
  """Moves `robot` (on a free joint: a walking robot, a ball) to x, y facing
  yaw, at once and at rest, keeping its height and tilt. Reset still puts it
  where it was spawned; with spawn=True this is its spawn pose from then on
  (as the Stage panel's Pose sets it)."""
  lobby_js.place(str(robot), float(x), float(y), float(yaw), bool(spawn))


# ---- for pybt.js ----

def _rows(root):
  """The tree view's rows: (depth, class, name, status, feedback) depth first."""
  out = []

  def walk(b, depth):
    out.append((depth, type(b).__name__, b.name, b.status.value.lower(), b.feedback_message or ''))
    for c in b.children:
      walk(c, depth + 1)
  walk(root, 0)
  return out


def _check(source):
  """A syntax error of `source`: (line, column, message), or None."""
  try:
    compile(source, '<tree>', 'exec')
  except SyntaxError as err:
    return (err.lineno or 1, err.offset or 1, err.msg)
  return None
