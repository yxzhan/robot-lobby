"""Trains a MuJoCo Playground locomotion task with Brax PPO (MJX on the GPU).

  uv run train.py                                   # SpotFlatTerrainJoystick, its tuned config
  uv run train.py --steps 20_000_000 --name quick   # a short run (pipeline test)

Writes runs/<name>/params.pkl (normalizer statistics, policy and value
parameters) and runs/<name>/run.json (environment, its config, the PPO
config): what export_onnx.py turns into a policy for the viewer.
"""

import argparse
import functools
import json
import os
import pathlib
import pickle
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # XLA's autotuning chatter
import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

HERE = pathlib.Path(__file__).resolve().parent


def quiet(env):
  """Turns off MuJoCo Warp's overflow prints: Playground's tasks run the
  solver for 1 iteration on purpose, and every step would print that the
  iteration limit was reached."""
  mx = getattr(env, '_mjx_model', None)
  impl = getattr(getattr(mx, 'opt', None), '_impl', None)
  if impl is not None and hasattr(impl, 'warn_overflow'):
    env._mjx_model = mx.replace(opt=mx.opt.replace(_impl=impl.replace(warn_overflow=0)))
  return env


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--env', default='SpotFlatTerrainJoystick')
  ap.add_argument('--steps', type=int, help='environment steps (default: the tuned config, 100M for Spot)')
  ap.add_argument('--envs', type=int, help='parallel environments (default: the tuned config, 8192)')
  ap.add_argument('--evals', type=int, help='evaluations (progress lines) during training')
  ap.add_argument('--njmax', type=int, default=64,
                  help='constraint rows per environment (MuJoCo Warp drops the ones beyond it, silently; '
                       "Spot's task config has 28, which its friction loss and contacts already fill)")
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--name', help='run directory under runs/ (default: <env>-<time>)')
  args = ap.parse_args()

  env_cfg = registry.get_default_config(args.env)
  ppo_cfg = locomotion_params.brax_ppo_config(args.env)
  if args.steps:
    ppo_cfg.num_timesteps = args.steps
  if args.envs:
    ppo_cfg.num_envs = args.envs
    env_cfg.naconmax = 4 * args.envs  # contact buffer, sized per environment
  if args.evals:
    ppo_cfg.num_evals = args.evals
  if 'njmax' in env_cfg and args.njmax:
    env_cfg.njmax = max(env_cfg.njmax, args.njmax)
  out = HERE / 'runs' / (args.name or f"{args.env}-{time.strftime('%Y%m%d-%H%M%S')}")
  out.mkdir(parents=True, exist_ok=True)

  env, eval_env = (quiet(registry.load(args.env, config=env_cfg)) for _ in range(2))
  randomizer = registry.get_domain_randomizer(args.env)
  net_cfg = dict(ppo_cfg.network_factory)
  train_cfg = {k: v for k, v in ppo_cfg.to_dict().items() if k != 'network_factory'}
  print(f'{args.env}: {ppo_cfg.num_timesteps:,} steps, {ppo_cfg.num_envs} envs, devices {jax.devices()}')
  print(f'policy {net_cfg}')

  t0 = time.monotonic()
  log = []

  def progress(step, metrics):
    secs = time.monotonic() - t0
    r = metrics.get('eval/episode_reward', float('nan'))
    sd = metrics.get('eval/episode_reward_std', float('nan'))
    print(f'{step:>12,} steps  reward {r:8.2f} ± {sd:6.2f}  {secs:7.0f} s', flush=True)
    log.append({'step': int(step), 'reward': float(r), 'reward_std': float(sd), 'seconds': secs})

  _, params, _ = ppo.train(
      environment=env,
      eval_env=eval_env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      randomization_fn=randomizer,
      network_factory=functools.partial(ppo_networks.make_ppo_networks, **net_cfg),
      progress_fn=progress,
      seed=args.seed,
      **train_cfg,
  )
  secs = time.monotonic() - t0

  with open(out / 'params.pkl', 'wb') as f:
    pickle.dump(jax.device_get(params), f)
  (out / 'run.json').write_text(json.dumps({
      'env': args.env, 'env_config': env_cfg.to_dict(), 'ppo_config': train_cfg, 'network': net_cfg,
      'seconds': secs, 'progress': log,
  }, indent=2, default=str))
  print(f'done in {secs:.0f} s: {out}')


if __name__ == '__main__':
  main()
