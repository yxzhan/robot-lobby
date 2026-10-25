"""Imagined rollouts through the world model (SuperMimic Eq. 2 and 3).

The task computes the reward, the soft fall probability and the next actor
observation from the imagined state (Task.imagine_step), so the H-step return
is differentiable in the actor's parameters; a fall becomes a survival weight
c_t that discounts the later rewards and the bootstrap value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from smlab.tasks.base import apply_pose_delta


@dataclass(kw_only=True)
class ImagineCfg:
  horizon: int = 32
  gamma: float = 0.99
  lam: float = 0.95
  compile: bool = True
  """torch.compile one imagined step (fuses the many small elementwise kernels)."""


class Imagination:
  def __init__(self, cfg: ImagineCfg, task, device: str):
    self.cfg, self.task, self.device = cfg, task, device
    self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

  def _step(self, s, pose, a_prev, obs, rec, xi, k, w, actor, wm):
    """One imagined step. Pure function of its inputs (compiled when cfg.compile)."""
    crit_in = self.task.critic_input(s, rec, xi)
    a = actor(obs)
    s_next, dpose = wm(s, a, a_prev, xi)
    pose_next = apply_pose_delta(pose, dpose) if self.task.planar else pose
    r, d, obs, rec = self.task.imagine_step(s, s_next, pose_next, a, a_prev, rec, xi, k, w)
    return s_next, pose_next, a, obs, rec, r, d, crit_in

  def rollout(self, wm, actor, critic, seed: dict, k: dict, w: dict):
    """Imagine H steps from replay seeds. Returns the differentiable objective
    and the data the critic is fit on."""
    cfg = self.cfg
    s, pose, a_prev, obs, rec, xi = (seed[n] for n in ("s", "pose", "a_prev", "obs", "rec", "xi"))
    B = s.shape[0]
    surv = torch.ones(B, device=self.device)
    ret = torch.zeros(B, device=self.device)
    disc = 1.0
    rewards, survs, crit_in = [], [], []
    for _ in range(cfg.horizon):
      survs.append(surv)
      s, pose, a_prev, obs, rec, r, d, x = self._step_fn(s, pose, a_prev, obs, rec, xi, k, w, actor, wm)
      crit_in.append(x)
      ret = ret + disc * surv * r
      rewards.append(r)
      surv = surv * (1.0 - d)
      disc *= cfg.gamma
    x_H = self.task.critic_input(s, rec, xi)
    v_H = critic(x_H).squeeze(-1)
    objective = ret + disc * surv * v_H
    return objective, {
      "rewards": torch.stack(rewards, 1),  # (B, H)
      "surv": torch.stack(survs + [surv], 1),  # (B, H+1)
      "critic_in": torch.stack(crit_in + [x_H], 1),  # (B, H+1, D)
    }

  @torch.no_grad()
  def gae_targets(self, critic, data) -> torch.Tensor:
    """TD(lambda) returns over the imagined steps (SuperMimic Eq. 3)."""
    cfg = self.cfg
    r, surv = data["rewards"].detach(), data["surv"].detach()
    v = critic(data["critic_in"].detach()).squeeze(-1)  # (B, H+1)
    H = r.shape[1]
    targets = torch.zeros_like(r)
    nxt = v[:, H]
    for t in reversed(range(H)):
      cont = surv[:, t + 1] / surv[:, t].clamp(min=1e-6)  # 1 - d_{t+1}
      nxt = r[:, t] + cfg.gamma * cont * ((1 - cfg.lam) * v[:, t + 1] + cfg.lam * nxt)
      targets[:, t] = nxt
    return targets
