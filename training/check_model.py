"""Runs an exported policy on a robot file as the viewer does, on the CPU.

  uv run check_model.py ../pretrained_policies/playground_spot_arm \\
      ../mjcf_assets/robots/mujoco_menagerie/boston_dynamics_spot/spot_arm.xml

The training model is not the viewer's: this loads the robot file itself
(mesh collision, its own actuators for joints the policy does not drive,
e.g. an arm, held at the home key), on a floor, with the viewer's physics
defaults (public/js/world/physics.js: 2 ms, elliptic cone, impratio 10), sets the
trained gains and joint parameters as the viewer's controllers do
(playground.js / PlaygroundController) and builds the observation with
export_onnx.Observer. Scenarios: walking forward; commands changing every
second and every 0.2 s (several seeds); and, with --arm, the same with the
arm actuators holding other poses. Prints falls and how well the command was
tracked.
"""

import argparse
import json
import os
import pathlib

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
import numpy as np
import onnxruntime as ort

from export_onnx import Observer

PHYSICS = {'timestep': 0.002, 'cone': mujoco.mjtCone.mjCONE_ELLIPTIC, 'impratio': 10}
# Arm poses for --arm (joint name -> angle), besides the home key's.
ARM_POSES = {
    'reach forward': {'arm_sh0': 0, 'arm_sh1': -0.6, 'arm_el0': 0.6, 'arm_el1': 0, 'arm_wr0': 0, 'arm_wr1': 0},
    'reach left, up': {'arm_sh0': 1.2, 'arm_sh1': -1.6, 'arm_el0': 1.2, 'arm_el1': 0, 'arm_wr0': 0.5, 'arm_wr1': 0},
    'reach down front': {'arm_sh0': 0, 'arm_sh1': -0.2, 'arm_el0': 1.8, 'arm_el1': 0, 'arm_wr0': 1.2, 'arm_wr1': 0},
}


def load_world(robot):
  spec = mujoco.MjSpec.from_file(str(robot))
  spec.worldbody.add_geom(name='check_floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
  m = spec.compile()
  m.opt.timestep = PHYSICS['timestep']
  m.opt.cone = PHYSICS['cone']
  m.opt.impratio = PHYSICS['impratio']
  return m


def bind(m, spec):
  """The trained gains and joint parameters, as the viewer's controllers set them."""
  acts = np.array([m.actuator(n).id for n in spec['actuators']])
  dofs = np.array([m.joint(n).dofadr[0] for n in spec['joints']])
  m.actuator_gainprm[acts, 0] = spec['actuator_gain']
  m.actuator_biasprm[acts, :3] = spec['actuator_bias']
  m.dof_damping[dofs] = spec['dof_damping']
  m.dof_armature[dofs] = spec['dof_armature']
  m.dof_frictionloss[dofs] = spec['dof_frictionloss']


def run(m, spec, sess, commands, seconds, hold=None):
  """Commands: a function of time -> [vx, vy, wz]. `hold`: joint -> angle for
  the actuators the policy does not drive. Returns (fell at, mean speed error)."""
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  own = {m.actuator(n).id for n in spec['actuators']}
  for a in range(m.nu):
    if a in own or m.actuator_trntype[a] != 0:
      continue
    j = m.actuator_trnid[a, 0]
    name = m.joint(j).name
    q = (hold or {}).get(name, d.qpos[m.jnt_qposadr[j]])
    d.ctrl[a] = q
    d.qpos[m.jnt_qposadr[j]] = q
  ob = Observer(m, spec)
  for n, q in zip(spec['joints'], spec['default_pose']):
    d.qpos[m.joint(n).qposadr[0]] = q
  mujoco.mj_forward(m, d)
  substeps = round(spec['ctrl_dt'] / m.opt.timestep)
  root = ob.root
  err, fell = [], None
  for k in range(round(seconds / spec['ctrl_dt'])):
    t = k * spec['ctrl_dt']
    cmd = commands(t)
    ob.act_on(d, sess.run(None, {'obs': ob.observe(d, cmd)})[0][0])
    for _ in range(substeps):
      mujoco.mj_step(m, d)
    R = d.xmat[root].reshape(3, 3)
    if R[2, 2] < 0.5:
      fell = t
      break
    if t > 1:  # tracking, after settling: body-frame velocity against the command
      v = R.T @ d.qvel[ob.vadr:ob.vadr + 3]
      w = d.qvel[ob.vadr + 5]
      err.append(np.hypot(v[0] - cmd[0], v[1] - cmd[1]) + 0.3 * abs(w - cmd[2]))
  return fell, float(np.mean(err)) if err else float('nan')


def random_commands(spec, every, seed):
  rng = np.random.default_rng(seed)
  r = spec['command_range']
  table = {}

  def at(t):
    k = int(t / every)
    if k not in table:
      table[k] = [rng.uniform(*r['lin_vel_x']), rng.uniform(*r['lin_vel_y']), rng.uniform(*r['ang_vel_yaw'])]
      if rng.random() < 0.2:
        table[k] = [0.0, 0.0, 0.0]
    return table[k]
  return at


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('policy', type=pathlib.Path)
  ap.add_argument('robot', type=pathlib.Path)
  ap.add_argument('--seeds', type=int, default=10)
  ap.add_argument('--seconds', type=float, default=20)
  ap.add_argument('--arm', action='store_true', help='also with the arm held in other poses (ARM_POSES)')
  args = ap.parse_args()
  spec = json.loads((args.policy / 'policy.json').read_text())
  sess = ort.InferenceSession(str(args.policy / spec['onnx']), providers=['CPUExecutionProvider'])
  m = load_world(args.robot)
  bind(m, spec)

  poses = {'home': None}
  if args.arm:
    poses.update(ARM_POSES)
  for pose, hold in poses.items():
    fell, e = run(m, spec, sess, lambda t: [0.5, 0, 0], 10, hold)
    print(f'[{pose}] forward 0.5 m/s, 10 s: ' + (f'fell at {fell:.1f} s' if fell is not None else 'up') + f', error {e:.2f}')
    for every in (1.0, 0.2):
      falls, errs = 0, []
      for seed in range(args.seeds):
        fell, e = run(m, spec, sess, random_commands(spec, every, seed), args.seconds, hold)
        falls += fell is not None
        errs.append(e)
      print(f'[{pose}] commands every {every} s, {args.seeds} x {args.seconds:.0f} s: {falls} fell, '
            f'mean error {np.nanmean(errs):.2f}')


if __name__ == '__main__':
  main()
