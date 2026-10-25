"""Evaluates a policy deterministically in the task's env: falls per minute,
each reward term per second, velocity tracking error; optionally a video.

    python scripts/eval.py --task duck-walk --checkpoint logs/duck-walk/<run>/model_10000.pt
    python scripts/eval.py --task duck-walk --onnx ../../pretrained_policies/microduck_rl/walk.onnx
    python scripts/eval.py ... --envs 1 --seconds 20 --video walk.mp4

A checkpoint is one of ours (its actor); --onnx any policy with the task's
observation and actions (e.g. a PPO one exported by rsl_rl).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smlab.core.nets import Actor  # noqa: E402
from smlab.tasks import make_task  # noqa: E402


def load_policy(args, obs_dim, act_dim, device):
  if args.zero:
    return lambda o: torch.zeros(o.shape[0], act_dim, device=device)
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden = tuple(ckpt["cfg"]["actor_hidden"])
    actor = Actor(obs_dim, hidden, act_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return lambda o: actor(o)
  import numpy as np
  import onnxruntime as ort

  sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
  name, fixed = sess.get_inputs()[0].name, sess.get_inputs()[0].shape[0] == 1

  def act(o):
    x = o.cpu().numpy().astype("float32")
    y = (np.concatenate([sess.run(None, {name: x[i : i + 1]})[0] for i in range(len(x))])
         if fixed else sess.run(None, {name: x})[0])
    return torch.as_tensor(y, device=device)
  return act


@torch.no_grad()
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="duck-walk")
  ap.add_argument("--checkpoint", default="")
  ap.add_argument("--onnx", default="")
  ap.add_argument("--zero", action="store_true", help="zero actions: the default pose")
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--video", default="")
  args = ap.parse_args()
  assert bool(args.checkpoint) + bool(args.onnx) + args.zero == 1, "give --checkpoint, --onnx or --zero"

  task = make_task(args.task)

  def edit(c):
    if args.video:
      c.viewer.width, c.viewer.height = 960, 540
      c.viewer.distance, c.viewer.elevation, c.viewer.azimuth = 1.2, -20.0, 135.0
      c.viewer.max_extra_envs = 0

  env = task.make_env(args.envs, args.seed, edit=edit, render=bool(args.video))
  obs, _ = env.reset()
  task.bind(env)
  policy = load_policy(args, task.obs_dim, task.act_dim, env.device)
  rm = env.reward_manager
  names = list(rm._term_names)
  sums = torch.zeros(len(names), device=env.device)
  falls, steps, frames = 0, int(args.seconds / env.step_dt), []
  vel_err = 0.0
  sensor = env.scene.sensors.get("feet_ground_contact")
  feet_down = torch.zeros(3, device=env.device)  # steps with 0, 1, 2 feet down
  lifts, lift_air = 0, 0.0
  for _ in range(steps):
    cmd = env.command_manager.get_command("twist").clone()
    obs, _, term, trunc, _ = env.step(policy(obs["actor"]))
    sums += (rm._step_reward * torch.tensor([c.weight != 0 for c in rm._term_cfgs], device=env.device)).sum(0)
    falls += int(term.sum())  # terminations other than time-outs: falls (or a NaN state)
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_err += float((v - cmd[:, :2]).norm(dim=-1).mean())
    if sensor is not None:
      contact = sensor.data  # a fresh snapshot each step
      down = (contact.found > 0).sum(-1)
      feet_down += torch.bincount(down, minlength=3)[:3].float()
      landed = sensor.compute_first_contact(dt=env.step_dt)
      lifts += int(landed.sum())
      lift_air += float((contact.last_air_time * landed).sum())
    if args.video:
      frames.append(env.render())
  minutes = args.envs * args.seconds / 60
  print(f"{args.checkpoint or args.onnx or 'zero actions'}: {args.envs} envs x {args.seconds:.0f} s")
  print(f"  falls per robot-minute   {falls / minutes:.3f}")
  print(f"  mean |v_xy - cmd|        {vel_err / steps:.3f} m/s")
  if sensor is not None:
    f = (feet_down / feet_down.sum()).tolist()
    print(f"  feet down 2 / 1 / 0      {f[2]:.2f} / {f[1]:.2f} / {f[0]:.2f} of the time")
    print(f"  landings per foot-second {lifts / (2 * args.envs * args.seconds):.2f}, "
          f"mean air time before one {lift_air / max(lifts, 1):.3f} s")
  print(f"  total reward per second  {float(sums.sum()) / (steps * args.envs):.3f}")
  for n, v in zip(names, sums.tolist()):
    if v:
      print(f"    {n:24s} {v / (steps * args.envs):8.3f}")
  if args.video:
    import imageio

    imageio.mimsave(args.video, frames, fps=round(1 / env.step_dt), codec="libx264", quality=7)
    print(f"  video: {args.video}")


if __name__ == "__main__":
  main()
