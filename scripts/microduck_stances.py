"""States the Microduck is in under its walking policy, for training its
skills to start from them: run in the viewer's own physics (server/policies.py:
MicroduckController on the BAM servo model, the robot file of the library),
each sample walks with random commands, kicks once or more (the walking
policy taking over after each kick), with random head commands in some,
and is recorded in one of three ways:

  settled    standing under the walking policy with a zero command until
             it has settled (1.5-3 s), feet ~84 mm apart
  after_kick 0-2 s after a kick, while the feet slide back together from
             as much as ~150 mm apart, still moving (velocities recorded)
  noslip     settled, but with the solver's noslip on (100 iterations): the
             feet cannot slide back and stay ~95 mm apart

microduck_rl's BallKick task starts episodes from them. Its kick first only
kicked from HOME (after one kick, the walking policy's stand, neck 0.15 rad
down and the legs a few hundredths off, made it stand still); trained on
settled stances, it kicked only with the feet at most ~85 mm apart (after a
kick they are wider for 2 s, and for good with noslip).

    python3 scripts/microduck_stances.py --out kick_stances.npz [--samples 3000 --workers 16]

The .npz holds, per sample: joint_pos and joint_vel (14, the policy's joint
order, absolute), root_z (trunk height), root_tilt (the trunk's quaternion
w x y z with its yaw taken out), root_lin_vel (the trunk's velocity in its
yaw frame), root_ang_vel (its angular velocity in its own frame, as a free
joint's qvel has it), right_foot / left_foot (the feet sites relative to the
trunk in its yaw frame, x y z), head_cmd (the head command it stood with),
mode (0 settled, 1 after_kick, 2 noslip).
"""

import argparse
import multiprocessing as mp
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'server'))
ROBOT = ROOT / 'mjcf_assets' / 'robots' / 'custom' / 'microduck' / 'microduck.xml'
PREFIX = 'r/'
SETTLE = (1.5, 3.0)        # s standing before the record
AFTER_KICK = (0.0, 2.0)    # s after a kick's end, for after_kick
HEAD_RANGE = 0.5           # of the head command's trained range, when one is drawn
MODES = {'settled': 0.4, 'after_kick': 0.4, 'noslip': 0.2}  # how often each
KICK = 0.3                 # s, the kick's duration in the viewer (policy.json): 70% of the kicks, the rest up to 1 s


def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def sample(seed):
  """One settled stance, or None if the duck fell."""
  import policies as POL

  rng = np.random.default_rng(seed)
  mode = rng.choice(list(MODES), p=np.array(list(MODES.values())) / sum(MODES.values()))
  spec = mujoco.MjSpec()
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # the viewer's defaults
  if mode == 'noslip':
    spec.option.noslip_iterations, spec.option.noslip_tolerance = 100, 1e-10
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1])
  robot = mujoco.MjSpec.from_file(str(ROBOT))
  spec.worldbody.add_frame().attach_body(robot.worldbody.first_body(), PREFIX, '')
  m = spec.compile()
  d = mujoco.MjData(m)
  d.qpos[:mujoco.MjModel.from_xml_path(str(ROBOT)).nq] = mujoco.MjModel.from_xml_path(str(ROBOT)).key_qpos[0]
  c = POL.create('microduck-walk', PREFIX)
  c.bind(m)
  c.pose(d)
  mujoco.mj_forward(m, d)
  nxt = [0.0]

  def run(t):
    end = d.time + t
    while d.time < end:
      if d.time >= nxt[0] - 1e-9:
        c.step(d)
        nxt[0] += 0.02
      c.substep(d)
      mujoco.mj_step(m, d)

  head = c.commands['head']
  if rng.random() < 0.3:
    c.command['head'] = [float(rng.uniform(lo, hi) * HEAD_RANGE) for lo, hi in head['range']]
  def kick():
    c.cmd = [0.0, 0.0, 0.0]
    run(rng.uniform(0.3, 1.5))
    t = KICK if rng.random() < 0.7 else float(rng.uniform(KICK, 1.0))
    c.set_skill('kick_right', t)
    run(t)

  run(0.5)
  for _ in range(rng.integers(1, 4)):  # a few things in a row
    if rng.random() < 0.6:  # walk
      cmd = [rng.uniform(-0.3, 0.35), rng.uniform(-0.2, 0.2), rng.uniform(-0.8, 0.8)]
      c.cmd = [v if rng.random() > 0.3 else 0.0 for v in cmd]
      run(rng.uniform(0.8, 3.0))
      c.cmd = [0.0, 0.0, 0.0]
      run(rng.uniform(0.0, 1.5))
    else:  # kick (the walking policy takes over after it)
      kick()
      run(1.2)
  c.cmd = [0.0, 0.0, 0.0]
  if mode != 'settled':  # the last thing a kick: its wide stance
    kick()
  run(rng.uniform(*(AFTER_KICK if mode == 'after_kick' else SETTLE)))
  root = m.body(PREFIX + 'trunk_base').id
  R, p = d.xmat[root].reshape(3, 3), d.xpos[root]
  if R[2, 2] < 0.9 or p[2] < 0.09:
    return None
  q = d.xquat[root]
  yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
  tilt = quat_mul(np.array([np.cos(yaw / 2), 0, 0, -np.sin(yaw / 2)]), q)
  cy, sy = np.cos(yaw), np.sin(yaw)

  def foot(name):
    r = d.site(PREFIX + name).xpos - p
    return np.array([cy * r[0] + sy * r[1], -sy * r[0] + cy * r[1], r[2]])
  free = m.body_jntadr[root]
  v = d.qvel[m.jnt_dofadr[free]:m.jnt_dofadr[free] + 6]
  return {
    'joint_pos': d.qpos[c.qadr].copy(), 'joint_vel': d.qvel[c.dofs].copy(),
    'root_z': float(p[2]), 'root_tilt': tilt / np.linalg.norm(tilt),
    'root_lin_vel': np.array([cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1], v[2]]), 'root_ang_vel': v[3:6].copy(),
    'right_foot': foot('right_foot'), 'left_foot': foot('left_foot'), 'head_cmd': np.array(c.command['head']),
    'mode': list(MODES).index(mode),
  }


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--out', required=True, type=pathlib.Path)
  ap.add_argument('--samples', type=int, default=3000)
  ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  with mp.Pool(args.workers) as pool:
    got = [s for s in pool.imap_unordered(sample, range(args.seed, args.seed + args.samples), chunksize=4) if s]
  out = {k: np.stack([s[k] for s in got]) for k in got[0]}
  np.savez(args.out, **out)
  dev = out['joint_pos'] - np.array(__import__('json').loads(
      (ROOT / 'pretrained_policies' / 'microduck_official' / 'policy.json').read_text())['default_pose'])
  print(f'{args.out}: {len(got)} of {args.samples} stances (the others fell); |joint - HOME| p50 / p95 / max by joint:')
  for name, col in zip(('L_yaw', 'L_roll', 'L_pitch', 'L_knee', 'L_ankle', 'neck_p', 'head_p', 'head_y', 'head_r',
                        'R_yaw', 'R_roll', 'R_pitch', 'R_knee', 'R_ankle'), np.abs(dev).T):
    print(f'  {name:8s} {np.percentile(col, 50):.3f} {np.percentile(col, 95):.3f} {col.max():.3f}')
  print(f"  root z {out['root_z'].min():.3f}-{out['root_z'].max():.3f}; right foot x {out['right_foot'][:, 0].min()*1000:.0f}"
        f"..{out['right_foot'][:, 0].max()*1000:.0f} mm, y {out['right_foot'][:, 1].min()*1000:.0f}..{out['right_foot'][:, 1].max()*1000:.0f} mm")
  width = (out['left_foot'][:, 1] - out['right_foot'][:, 1]) * 1000
  for k, name in enumerate(MODES):
    w = width[out['mode'] == k]
    if len(w):
      print(f'  {name:10s} {len(w):5d}: feet {w.min():.0f}-{w.max():.0f} mm apart (p5 {np.percentile(w, 5):.0f}, p50 '
            f'{np.percentile(w, 50):.0f}, p95 {np.percentile(w, 95):.0f}), max |joint vel| p50 '
            f'{np.percentile(np.abs(out["joint_vel"][out["mode"] == k]).max(1), 50):.2f} rad/s')


if __name__ == '__main__':
  main()
