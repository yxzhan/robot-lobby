"""Checks a task's imagination against its env on real transitions (no world
model): the rebuilt actor observation against the env's, and every reward
term against the env's reward manager, while a policy drives the robot.

    python scripts/check_task.py --task duck-walk --onnx <policy.onnx> [--envs 256 --steps 300]

The observation differs by the env's noise (and the IMU's 0-1 step lag, so
both are tried); reward terms are compared as the mean over envs and steps
and as the correlation per (env, step).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smlab.tasks import make_task  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="duck-walk")
  ap.add_argument("--onnx", required=True, help="a policy of the task's observation and actions")
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--noise", type=float, default=0.1, help="action noise, for varied motion")
  args = ap.parse_args()

  import onnxruntime as ort

  sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
  task = make_task(args.task)
  # Without pushes: they change the velocities after the env's rewards are computed,
  # so the state read after a step would not be the one the env rewarded.
  env = task.make_env(args.envs, seed=1, edit=lambda c: c.events.pop("push_robot", None))
  obs, _ = env.reset()
  task.bind(env)
  obs = obs["actor"]
  rm = env.reward_manager
  names = list(rm._term_names)
  a_prev = torch.zeros(args.envs, task.act_dim, device=env.device)
  ours, theirs, obs_err = {n: [] for n in names}, {n: [] for n in names}, []

  def act(o):
    import numpy as np
    x = o.cpu().numpy().astype("float32")
    name = sess.get_inputs()[0].name
    out = np.concatenate([sess.run(None, {name: x[i : i + 1]})[0] for i in range(x.shape[0])])  # batch-1 ONNX
    return torch.as_tensor(out, device=env.device)

  # The first third settles the episodes (air times, EMAs), then compare.
  for t in range(args.steps):
    s0 = task.state(env)
    rec0 = {k: v.clone() for k, v in task.read_record(env).items()}
    a = act(obs) + args.noise * torch.randn(args.envs, task.act_dim, device=env.device)
    obs1, _, term, trunc, _ = env.step(a)
    done = term | trunc
    s1 = task.state(env)
    if t > args.steps // 3:
      keep = ~done & (env.episode_length_buf > 2)
      k = task.rollout_consts(torch.arange(args.envs, device=env.device))
      vals, _ = task.terms(s0, s1, a, a_prev, rec0, k)
      for i, n in enumerate(names):
        theirs[n].append(rm._step_reward[keep, i] / (rm._term_cfgs[i].weight or 1.0))
        ours[n].append(vals[n][keep] if n in vals else torch.zeros_like(theirs[n][-1]))
      errs = []
      for late in (0.0, 1.0):
        k["imu_late"] = torch.full((args.envs, 1), late, device=env.device)
        mine = observe_clean(task, s0, s1, a, rec0["cmd"], k)
        errs.append((mine - obs1["actor"]).abs())
      obs_err.append(torch.minimum(*errs)[keep])
    a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)
    obs = obs1["actor"]

  e = torch.cat(obs_err)
  parts = [("ang_vel", 0, 3), ("gravity", 3, 6), ("joint_pos", 6, 20), ("joint_vel", 20, 34), ("action", 34, 48), ("commands", 48, 61)]
  print("\nobservation |ours - env| (env noise half-widths: ang_vel 0.03, gravity 0.01, joint_pos 0.001, joint_vel 0.25)")
  for name, i, j in parts:
    print(f"  {name:10s} mean {e[:, i:j].mean():.4f}  p99 {e[:, i:j].quantile(0.99):.4f}  max {e[:, i:j].max():.4f}")
  print("\nreward term (unweighted)   env mean    ours mean   corr")
  for n in names:
    x, y = torch.cat(theirs[n]), torch.cat(ours[n])
    corr = torch.corrcoef(torch.stack([x, y]))[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else float("nan")
    print(f"  {n:24s} {x.mean():10.4f} {y.mean():10.4f}   {float(corr):.3f}")


def observe_clean(task, s0, s1, a, cmd, k):
  """task.observe without its noise."""
  from smlab.tasks import duck_walk
  saved = dict(duck_walk.OBS_NOISE)
  duck_walk.OBS_NOISE.update({n: 0.0 for n in saved})
  try:
    return task.observe(s0, s1, a, cmd, k)
  finally:
    duck_walk.OBS_NOISE.update(saved)


if __name__ == "__main__":
  main()
