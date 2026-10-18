"""Exports a policy trained by train.py to ONNX for the viewer.

  uv run export_onnx.py runs/<name>                 # -> ../pretrained_policies/playground_spot/
  uv run export_onnx.py runs/<name> --out DIR

Writes DIR/policy.onnx (observation normalization + MLP + tanh of the
action mean: the deterministic action, float32 [1, obs] -> [1, act]) and
DIR/policy.json (what a controller needs besides the weights: joint and
actuator names, default pose, action scale, the actuator / joint
parameters the policy was trained with, the observation layout and command
ranges). Then checks it:

1. ONNX vs. Brax's own inference function on random observations.
2. A closed loop in plain MuJoCo (CPU, the task's own model) with the ONNX
   policy and the observation built as in policy.json (as the viewer's
   controllers do), walking forward: how far it got and whether it fell.
"""

import argparse
import json
import os
import pathlib
import pickle

os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # small; leaves the GPU to a training run
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import onnx
import onnxruntime as ort
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from onnx import TensorProto, helper, numpy_helper

import envs  # registers our own tasks

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# Per task: where the policy goes and how to build its observation outside
# Playground. `obs` lists the observation's parts in order (see the task's
# _get_obs); sensors read at the `imu` site are in its frame (the root body's
# without one). `command_range`: what the task samples commands from.
TASKS = {
    'SpotFlatTerrainJoystick': {
        'dir': 'playground_spot',
        'obs': [
            ['gyro', 3],             # body angular velocity, body frame, at the last step's start (R^T cvel)
            ['upvector', 3],         # body z axis in the world frame
            ['joint_pos', 12],       # q - default pose
            ['qpos_error', 36],      # history_len frames of q - last motor targets, newest first
            ['feet_pos', 12],        # FL, FR, HL, HR foot centers relative to the body, body frame
            ['last_action', 12],     # the raw action before the last one (updated after the observation)
            ['command', 3],          # vx, vy (m/s), yaw rate (rad/s)
        ],
        'feet': ['FL', 'FR', 'HL', 'HR'],  # foot geoms (sites in Playground's model)
        'root': 'body',
        'imu': None,  # at the body origin
        'command_range': {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-0.8, 0.8], 'ang_vel_yaw': [-1.0, 1.0]},
    },
    # Go1's joystick task (mujoco_playground/_src/locomotion/go1/joystick.py) on a Go2 (envs/go2.py).
    'Go2JoystickFlatTerrain': {
        'dir': 'playground_go2',
        'obs': [
            ['linvel', 3],           # imu velocimeter: its linear velocity in its frame, at the last step's start
            ['gyro', 3],             # imu angular velocity in its frame, at the last step's start
            ['gravity', 3],          # gravity direction (0, 0, -1) in the imu frame
            ['joint_pos', 12],       # q - default pose
            ['joint_vel', 12],       # qdot
            ['last_action', 12],     # the raw action before the last one
            ['command', 3],          # vx, vy (m/s), yaw rate (rad/s)
        ],
        'feet': [],
        'root': 'base',
        'imu': 'imu',
        # command_config a: amplitudes, sampled in [-a, a] (b: chance of not zeroing an axis).
        'command_range': {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-0.8, 0.8], 'ang_vel_yaw': [-1.2, 1.2]},
        # Measured on go2.xml: it stands for commands below about these (the
        # task's tracking reward is lenient and its stand_still cost is not).
        'min_command': {'speed': 0.3, 'turn': 0.3},
    },
}


def load_run(run):
  meta = json.loads((run / 'run.json').read_text())
  with open(run / 'params.pkl', 'rb') as f:
    params = pickle.load(f)
  return meta, params


def mlp_layers(policy_params):
  p = policy_params['params']
  names = sorted(p, key=lambda n: int(n.split('_')[1]))
  return [(np.asarray(p[n]['kernel'], np.float32), np.asarray(p[n]['bias'], np.float32)) for n in names]


def build_onnx(mean, std, layers, act_size):
  """obs -> (obs - mean) / std -> Dense+swish ... -> Dense -> first act_size -> tanh."""
  inits = [numpy_helper.from_array(mean.astype(np.float32), 'obs_mean'),
           numpy_helper.from_array(std.astype(np.float32), 'obs_std')]
  nodes = [helper.make_node('Sub', ['obs', 'obs_mean'], ['obs_c']),
           helper.make_node('Div', ['obs_c', 'obs_std'], ['h0'])]
  x = 'h0'
  for i, (w, b) in enumerate(layers):
    inits += [numpy_helper.from_array(w, f'w{i}'), numpy_helper.from_array(b, f'b{i}')]
    y = f'z{i}'
    nodes.append(helper.make_node('Gemm', [x, f'w{i}', f'b{i}'], [y]))
    if i < len(layers) - 1:  # swish
      nodes += [helper.make_node('Sigmoid', [y], [f's{i}']), helper.make_node('Mul', [y, f's{i}'], [f'h{i + 1}'])]
      x = f'h{i + 1}'
    else:
      x = y
  inits += [numpy_helper.from_array(np.array([0], np.int64), 'start'),
            numpy_helper.from_array(np.array([act_size], np.int64), 'end'),
            numpy_helper.from_array(np.array([1], np.int64), 'axis')]
  nodes += [helper.make_node('Slice', [x, 'start', 'end', 'axis'], ['loc']),
            helper.make_node('Tanh', ['loc'], ['action'])]
  graph = helper.make_graph(
      nodes, 'policy',
      [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, len(mean)])],
      [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, act_size])],
      inits)
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], producer_name='mujoco-filament training')
  model.ir_version = 8  # onnxruntime-web reads it
  onnx.checker.check_model(model)
  return model


def describe(env, env_cfg, task):
  """policy.json minus the observation statistics."""
  m = env.mj_model
  acts = [m.actuator(i).name for i in range(m.nu)]
  joints = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
  dofs = [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
  return {
      'env': env.__class__.__module__ + '.' + env.__class__.__name__,
      'ctrl_dt': env_cfg.ctrl_dt,
      'sim_dt': env_cfg.sim_dt,
      'root': task['root'],
      'imu': task['imu'],
      'joints': joints,
      'actuators': acts,
      'feet': task['feet'],
      'default_pose': [float(v) for v in env._default_pose],
      'action_scale': env_cfg.action_scale,
      # What the policy's actuators and joints were trained with: a
      # controller sets these on bind (and gives them back on unbind).
      'actuator_gain': [float(m.actuator_gainprm[i, 0]) for i in range(m.nu)],
      'actuator_bias': [[float(v) for v in m.actuator_biasprm[i, :3]] for i in range(m.nu)],
      'ctrl_range': [[float(v) for v in (m.actuator_ctrlrange[i] if m.actuator_ctrllimited[i] else (-1e3, 1e3))]
                     for i in range(m.nu)],
      # Torque limits: a controller that closes the PD loop itself (on torque
      # motors) clips to these.
      'force_range': [[float(v) for v in (m.actuator_forcerange[i] if m.actuator_forcelimited[i] else (-1e6, 1e6))]
                      for i in range(m.nu)],
      'dof_damping': [float(m.dof_damping[d]) for d in dofs],
      'dof_armature': [float(m.dof_armature[d]) for d in dofs],
      'dof_frictionloss': [float(m.dof_frictionloss[d]) for d in dofs],
      'obs': task['obs'],
      'history_len': env_cfg.history_len,
      'command_range': task['command_range'],
      'min_command': task.get('min_command'),
  }


class Observer:
  """The policy's observation from MuJoCo data, as policy.json describes it
  (the reference for the viewer's controllers; no noise)."""

  def __init__(self, m, spec):
    self.spec = spec
    j = [m.joint(n) for n in spec['joints']]
    self.qadr = np.array([x.qposadr[0] for x in j])
    self.dofs = np.array([x.dofadr[0] for x in j])
    self.root = m.body(spec['root']).id
    self.tree = m.body_rootid[self.root]  # cvel is about its subtree center of mass
    self.imu = m.site(spec['imu']).id if spec.get('imu') else None
    free = m.body_jntadr[self.root]
    self.vadr = m.jnt_dofadr[free]
    self.feet = np.array([m.geom(n).id for n in spec['feet']], int)
    self.act = np.array([m.actuator(n).id for n in spec['actuators']])
    self.default = np.array(spec['default_pose'])
    self.lo, self.hi = np.array(spec['ctrl_range']).T
    self.reset()

  def reset(self):
    n = len(self.default)
    self.last_action = np.zeros(n)  # the action applied last
    self.prev_action = np.zeros(n)  # the one before: what the observation holds
    self.targets = np.zeros(n)  # Playground starts its error history against zero targets
    self.error = np.zeros(self.spec['history_len'] * n)

  def observe(self, d, command):
    n = len(self.default)
    R = d.xmat[self.root].reshape(3, 3)
    q = d.qpos[self.qadr]
    self.error = np.roll(self.error, n)
    self.error[:n] = q - self.targets
    feet = (d.geom_xpos[self.feet] - d.xpos[self.root]) @ R  # rows: R^T (p - p0)
    # Sensors are computed in the forward pass at the start of the last step,
    # as cvel, xpos, xmat and subtree_com are: cvel is the root's angular and
    # linear velocity (world frame) at its tree's center of mass.
    S, p = (d.site_xmat[self.imu].reshape(3, 3), d.site_xpos[self.imu]) if self.imu is not None else (R, d.xpos[self.root])
    w, v0 = d.cvel[self.root, :3], d.cvel[self.root, 3:]
    v = v0 + np.cross(w, p - d.subtree_com[self.tree])
    parts = {
        'linvel': v @ S,
        'gyro': w @ S,
        'gravity': -S[2],  # S^T (0, 0, -1)
        'upvector': S[:, 2],
        'joint_vel': d.qvel[self.dofs],
        'joint_pos': q - self.default,
        'qpos_error': self.error,
        'feet_pos': feet.ravel(),
        'last_action': self.prev_action,
        'command': np.asarray(command, float),
    }
    return np.concatenate([parts[k] for k, _ in self.spec['obs']]).astype(np.float32)[None]

  def act_on(self, d, action):
    self.prev_action = self.last_action
    self.last_action = np.asarray(action, float)
    self.targets = np.clip(self.default + self.last_action * self.spec['action_scale'], self.lo, self.hi)
    d.ctrl[self.act] = self.targets


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('run', type=pathlib.Path)
  ap.add_argument('--out', type=pathlib.Path)
  ap.add_argument('--seconds', type=float, default=10.0, help='closed-loop check length')
  args = ap.parse_args()

  meta, (norm, policy_params, _) = load_run(args.run)
  task = TASKS[meta['env']]
  out = args.out or ROOT / 'pretrained_policies' / task['dir']
  out.mkdir(parents=True, exist_ok=True)
  env_cfg = registry.get_default_config(meta['env'])
  env_cfg.impl = 'jax'  # only its MuJoCo model and config are used, on the CPU
  env = registry.load(meta['env'], config=env_cfg)
  act_size = env.action_size

  key = meta['network']['policy_obs_key']
  mean, std = np.asarray(norm.mean[key]), np.asarray(norm.std[key])
  layers = mlp_layers(policy_params)
  model = build_onnx(mean, std, layers, act_size)
  onnx.save(model, out / 'policy.onnx')
  spec = describe(env, env_cfg, task)
  assert sum(n for _, n in spec['obs']) == len(mean), 'observation layout does not match the policy input'
  spec.update({
      'onnx': 'policy.onnx', 'input': 'obs', 'output': 'action',
      'trained': {'run': args.run.name, 'steps': meta['ppo_config']['num_timesteps'],
                  'seconds': round(meta['seconds']), 'final_reward': (meta['progress'] or [{}])[-1].get('reward')},
  })
  (out / 'policy.json').write_text(json.dumps(spec, indent=1))
  print(f'wrote {out}/policy.onnx ({(out / "policy.onnx").stat().st_size // 1024} KB) and policy.json')

  # 1. ONNX vs. Brax.
  nets = ppo_networks.make_ppo_networks(
      {k: v.shape for k, v in norm.mean.items()}, act_size,
      preprocess_observations_fn=running_statistics.normalize,
      **meta['network'])
  infer = jax.jit(ppo_networks.make_inference_fn(nets)((norm, policy_params), deterministic=True))
  sess = ort.InferenceSession(str(out / 'policy.onnx'), providers=['CPUExecutionProvider'])
  rng = np.random.default_rng(0)
  worst = 0.0
  for _ in range(100):
    obs = {k: (np.asarray(norm.mean[k]) + np.asarray(norm.std[k]) * rng.standard_normal(norm.mean[k].shape)).astype(np.float32)
           for k in norm.mean}
    a_jax = np.asarray(infer({k: jp.asarray(v) for k, v in obs.items()}, jax.random.PRNGKey(0))[0])
    a_onnx = sess.run(None, {'obs': obs[key][None]})[0][0]
    worst = max(worst, float(np.abs(a_jax - a_onnx).max()))
  print(f'check 1: ONNX vs. Brax, 100 random observations: max |difference| {worst:.2e}')
  assert worst < 1e-4, 'ONNX export does not match Brax'

  # 2. Closed loop in plain MuJoCo with the ONNX policy.
  m = env.mj_model
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.keyframe('home').id)
  mujoco.mj_forward(m, d)
  ob = Observer(m, spec)
  substeps = round(spec['ctrl_dt'] / m.opt.timestep)
  cmd = [0.5, 0.0, 0.0]
  root = ob.root
  start = d.xpos[root].copy()
  fell = None
  for k in range(round(args.seconds / spec['ctrl_dt'])):
    ob.act_on(d, sess.run(None, {'obs': ob.observe(d, cmd)})[0][0])
    for _ in range(substeps):
      mujoco.mj_step(m, d)
    if d.xmat[root][8] < 0.85 and fell is None:
      fell = k * spec['ctrl_dt']
  moved = d.xpos[root] - start
  print(f'check 2: MuJoCo (CPU) closed loop, command vx {cmd[0]} m/s for {args.seconds:.0f} s: '
        f'moved x {moved[0]:+.2f} y {moved[1]:+.2f} m, height {d.xpos[root][2]:.2f} m, '
        + (f'fell at {fell:.1f} s' if fell is not None else 'stayed up'))


if __name__ == '__main__':
  main()
