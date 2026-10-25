"""How well a checkpoint's world model predicts real transitions: per state
part, the error of its 1-step and k-step (autoregressive) predictions against
the error of predicting no change, on fresh rollouts of the checkpoint's actor.

    python scripts/wm_check.py --checkpoint logs/duck-walk/<run>/model_1000.pt [--k 8]

A ratio well under 1 means the model explains the motion; near or above 1,
it does no better than "nothing moves".
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smlab.core.nets import Actor  # noqa: E402
from smlab.core.world_model import WorldModel  # noqa: E402
from smlab.tasks import make_task  # noqa: E402
from smlab.tasks.base import pose_delta  # noqa: E402


@torch.no_grad()
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="duck-walk")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=200)
  ap.add_argument("--k", type=int, default=8)
  ap.add_argument("--noise", type=float, default=0.05)
  args = ap.parse_args()

  task = make_task(args.task)
  env = task.make_env(args.envs, seed=3)
  obs, _ = env.reset()
  task.bind(env)
  ckpt = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
  cfg = ckpt["cfg"]
  actor = Actor(task.obs_dim, tuple(cfg["actor_hidden"]), task.act_dim).to(env.device)
  actor.load_state_dict(ckpt["actor"])
  wm = WorldModel(task.state_dim, task.act_dim, task.ctx_dim, 3 if task.planar else 0,
                  tuple(cfg["wm_hidden"])).to(env.device)
  wm.load_state_dict(ckpt["world_model"])

  S, P, A, X, D = [], [], [], [], []
  a_prev = torch.zeros(args.envs, task.act_dim, device=env.device)
  AP = []
  for _ in range(args.steps):
    S.append(task.state(env)); P.append(task.pose(env)); X.append(task.context(env)); AP.append(a_prev)
    a = actor(obs["actor"]) + args.noise * torch.randn(args.envs, task.act_dim, device=env.device)
    obs, _, term, trunc, _ = env.step(a)
    done = term | trunc
    A.append(a); D.append(done)
    a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)
  S, P, A, AP, X, D = (torch.stack(v, 1) for v in (S, P, A, AP, X, D))  # (N, T, ...)

  parts = {n: getattr(sys.modules[type(task).__module__], n) for n in
           ("Z", "GRAV", "V_B", "W_B", "Q", "QD", "FOOT_P", "FOOT_V", "FOOT_H", "ANGMOM", "CONTACT")
           if hasattr(sys.modules[type(task).__module__], n)}
  for k in (1, args.k):
    err = {n: [] for n in parts} | {"pose": []}
    base = {n: [] for n in parts} | {"pose": []}
    for t0 in range(0, args.steps - k, 4):
      ok = ~D[:, t0 : t0 + k].any(1)
      if not ok.any():
        continue
      s, pose = S[ok, t0], P[ok, t0]
      for j in range(k):
        s1, dp = wm(s, A[ok, t0 + j], AP[ok, t0 + j], X[ok, t0 + j])
        s = s1
      true = S[ok, t0 + k]
      for n, sl in parts.items():
        err[n].append((s[:, sl] - true[:, sl]).square().mean(-1))
        base[n].append((S[ok, t0][:, sl] - true[:, sl]).square().mean(-1))
    print(f"\n{k}-step prediction: RMS error, RMS of 'no change', ratio")
    for n in parts:
      e, b = torch.cat(err[n]).mean().sqrt(), torch.cat(base[n]).mean().sqrt()
      print(f"  {n:8s} {float(e):9.4f} {float(b):9.4f}   {float(e / b.clamp(min=1e-9)):.2f}")


if __name__ == "__main__":
  main()
