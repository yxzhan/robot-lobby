"""Which of Pollen's published Microduck policies (pretrained_policies/
microduck_official: the Hub's pollen-robotics/microduck-policies) run as
they are in the lobby: each on the lobby's Microduck (mjcf_assets, the
groundcontact model) with server/policies.py's MicroduckController (the BAM
servo model, as the page has it) in the viewer's default physics, through
the test its manifest entry asks for; prints what happened.

    .venv/bin/python scripts/microduck_official_test.py [--only velstand,alpha_walking]
"""

import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'server'))
import policies as POL  # noqa: E402

OFFICIAL = ROOT / 'pretrained_policies' / 'microduck_official'
SPEC = OFFICIAL / 'policy.json'  # the obs contract, default pose, BAM
DUCK = ROOT / 'mjcf_assets' / 'robots' / 'custom' / 'microduck' / 'microduck.xml'
BALL = ROOT / 'mjcf_assets' / 'objects' / 'custom' / 'microduck_ball' / 'microduck_ball.xml'
P = 'r/'


class Sim:
  def __init__(self, ball=None):
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1])
    duck = mujoco.MjSpec.from_file(str(DUCK))
    spec.worldbody.add_frame().attach_body(duck.worldbody.first_body(), P, '')
    if ball is not None:
      b = mujoco.MjSpec.from_file(str(BALL))
      spec.worldbody.add_frame(pos=ball).attach_body(b.worldbody.first_body(), 'ball/', '')
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    nq = mujoco.MjModel.from_xml_path(str(DUCK)).nq
    self.d.qpos[:nq] = mujoco.MjModel.from_xml_path(str(DUCK)).key_qpos[0]
    base = json.loads(SPEC.read_text())
    base['skills'] = {'walk': {'onnx': 'x'}}
    self.sessions = {}
    self.c = POL.MicroduckController(({'walk': None}, base), P)
    self.c.bind(self.m)
    self.c.pose(self.d)
    mujoco.mj_forward(self.m, self.d)
    self.root = self.m.body(P + 'trunk_base').id
    self.next = 0.0

  def use(self, name, fresh=False):
    if name not in self.sessions:
      self.sessions[name] = ort.InferenceSession(str(OFFICIAL / f'{name}.onnx'), providers=['CPUExecutionProvider'])
    self.c.sessions = {'walk': self.sessions[name]}
    if fresh:
      self.c.action[:] = 0

  def run(self, seconds, cmd=None, each=None):
    end = self.d.time + seconds
    while self.d.time < end - 1e-9:
      if self.d.time >= self.next - 1e-9:
        if cmd is not None:
          self.c.cmd = list(cmd(self.d.time)) if callable(cmd) else list(cmd)
        self.c.step(self.d)
        self.next += 0.02
        if each:
          each(self)
      self.c.substep(self.d)
      mujoco.mj_step(self.m, self.d)

  def lay(self, pitch):  # face-down (+90°) or face-up (-90°) on the floor, at HOME
    v = self.m.body_jntadr[self.root]
    a = self.m.jnt_qposadr[v]
    self.d.qpos[a:a + 3] = [0, 0, 0.08]
    self.d.qpos[a + 3:a + 7] = [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]
    self.d.qvel[:] = 0
    mujoco.mj_forward(self.m, self.d)

  @property
  def up(self):
    return float(self.d.xmat[self.root][8])

  @property
  def z(self):
    return float(self.d.xpos[self.root][2])

  @property
  def xy(self):
    return self.d.xpos[self.root][:2].copy()

  @property
  def yaw(self):
    R = self.d.xmat[self.root].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])

  def state(self):
    return f'up {self.up:.2f}, trunk z {self.z:.3f} m'


def standing(s):
  return s.up > 0.9 and s.z > 0.09


def test_perpetual(name):
  out = []
  s = Sim()
  s.use(name)
  s.run(1.0, [0, 0, 0])
  p0 = s.xy
  s.run(8.0, [0, 0, 0])
  out.append(f'stand 8 s: {"ok" if standing(s) else "FELL"} ({s.state()}, drift {np.linalg.norm(s.xy - p0) * 100:.0f} cm)')
  if standing(s):
    y0, p0 = s.yaw, s.xy
    s.run(6.0, [0.3, 0, 0])
    d = s.xy - p0
    fwd = d[0] * math.cos(y0) + d[1] * math.sin(y0)
    out.append(f'walk vx 0.3 for 6 s: {fwd:.2f} m forward ({fwd / 6:.2f} m/s), {"ok" if standing(s) else "FELL"}')
    s.run(1.0, [0, 0, 0])
    for wz in (0.8, -0.8):
      turned, last = [0.0], [s.yaw]

      def each(s):
        turned[0] += (s.yaw - last[0] + math.pi) % (2 * math.pi) - math.pi  # unwrapped, tick by tick
        last[0] = s.yaw
      s.run(4.0, [0, 0, wz], each)
      out.append(f'turn {wz:+.1f} rad/s for 4 s: {math.degrees(turned[0]):+.0f} deg ({turned[0] / 4:+.2f} rad/s; wanted {math.degrees(wz * 4):+.0f}), '
                 f'{"ok" if standing(s) else "FELL"}')
      s.run(1.0, [0, 0, 0])
  for label, pitch in (('face-down', math.pi / 2), ('face-up', -math.pi / 2)):
    s = Sim()
    s.use(name)
    s.lay(pitch)
    t_up = []
    s.run(10.0, [0, 0, 0], each=lambda s: (not t_up and standing(s)) and t_up.append(s.d.time))
    out.append(f'get up from {label}: {"UP at %.1f s" % t_up[0] if t_up else "no"}; after 10 s {s.state()}')
  return out


def test_sitstand():
  s = Sim()
  s.use('alpha_sitstand')
  s.run(2.0, [0, 0, 0])
  a = s.state()
  s.run(4.0, [1, 0, 0])
  b = s.state()
  s.run(4.0, [0, 0, 0])
  return [f'stand {a} -> sit flag 4 s: {b} -> stand flag 4 s: {s.state()}']


def test_ground_pick():
  s = Sim()
  s.use('alpha_stand')
  s.run(2.0, [0, 0, 0])
  s.use('alpha_ground_pick', fresh=True)
  head = {b for b in range(s.m.nbody) if any(k in (s.m.body(b).name or '') for k in ('jaw', 'head'))}
  geoms = [g for g in range(s.m.ngeom) if s.m.geom_bodyid[g] in head]
  lowest = lambda s: min(s.d.geom_xpos[g][2] - s.m.geom_rbound[g] for g in geoms)  # the head's lowest point (bounding spheres)
  standing_low = lowest(s)
  low = [1.0]

  def each(s):
    low[0] = min(low[0], lowest(s))
  t0 = s.d.time
  s.run(2.8, lambda t: [math.cos(2 * math.pi * (t - t0) / 4.0), math.sin(2 * math.pi * (t - t0) / 4.0), 0], each)
  s.use('velstand')
  s.run(3.0, [0, 0, 0])
  return [f'pick (2.8 s): the head\'s lowest point {low[0] * 100:.1f} cm (standing: {standing_low * 100:.1f} cm, bounding spheres); then velstand 3 s: {s.state()}']


def test_kick(side):
  y = -0.042 if side == 'right' else 0.042
  s = Sim(ball=[0.095, y, 0.05])
  s.use('alpha_stand')
  s.run(1.0, [0, 0, 0])
  bid = s.m.body('ball/ball').id if mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_BODY, 'ball/ball') >= 0 else None
  if bid is None:
    bid = next(b for b in range(s.m.nbody) if (s.m.body(b).name or '').startswith('ball/'))
  b0 = s.d.xpos[bid].copy()
  s.use(f'ball_kick_{side}', fresh=True)
  s.run(0.5, [0, 0, 0])
  s.use('velstand')
  s.run(2.0, [0, 0, 0])
  d = s.d.xpos[bid] - b0
  return [f'kick: ball moved {np.linalg.norm(d[:2]):.2f} m, heading {math.degrees(math.atan2(d[1], d[0])):.0f} deg; duck {s.state()}']


def test_roulade():
  s = Sim()
  s.use('alpha_stand')
  s.run(1.0, [0, 0, 0])
  turn = [0.0]
  last = [s.d.xmat[s.root].reshape(3, 3).copy()]

  def each(s):
    R = s.d.xmat[s.root].reshape(3, 3)
    turn[0] += math.acos(max(-1, min(1, (np.trace(last[0].T @ R) - 1) / 2)))
    last[0] = R.copy()
  s.use('roulade', fresh=True)
  s.run(1.0, [0, 0, 0], each)
  mid = s.state()
  s.use('velstand')
  s.run(3.0, [0, 0, 0], each)
  return [f'roll (1 s): turned {math.degrees(turn[0]):.0f} deg in all; after it {mid}; velstand 3 s later {s.state()}']


TESTS = {
  'velstand': lambda: test_perpetual('velstand'),
  'alpha_walking': lambda: test_perpetual('alpha_walking'),
  'alpha_stand': lambda: test_perpetual('alpha_stand'),
  'alpha_sitstand': test_sitstand,
  'alpha_ground_pick': test_ground_pick,
  'ball_kick_right': lambda: test_kick('right'),
  'ball_kick_left': lambda: test_kick('left'),
  'roulade': test_roulade,
}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--only', default=','.join(TESTS))
  args = ap.parse_args()
  for name in args.only.split(','):
    print(f'{name}:')
    for line in TESTS[name]():
      print(f'  {line}')
    sys.stdout.flush()
  print('roller, roller_crouch: not run (they need the roller Microduck, with wheels, which the library lacks)')


if __name__ == '__main__':
  main()
