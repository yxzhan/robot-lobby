"""Deterministic world model f(s, a, a_prev, xi) -> (s', planar pose delta).

The task defines the state s (what the rewards, the actor observation and the
termination need, plus whatever makes the dynamics predictable, e.g. foot
positions), the context xi (domain-randomized parameters the model may
condition on) and whether the robot moves on the floor plane (then the model
also predicts the planar motion (dx, dy, dyaw) in the previous yaw frame, so
the dynamics stay translation/yaw invariant and the pose is integrated
outside it).

Fit on k-step autoregressive windows from the replay buffer (SuperMimic
Eq. 1, state term only; reward and termination are analytic in the task).
Inputs and outputs are normalized with running statistics of collected data.
"""

from __future__ import annotations

import torch
from torch import nn

from smlab.core.nets import RunningNorm, mlp


class WorldModel(nn.Module):
  def __init__(self, state_dim: int, act_dim: int, ctx_dim: int, pose_dim: int = 3,
               hidden: tuple[int, ...] = (1024, 1024, 1024)):
    super().__init__()
    self.state_dim, self.pose_dim = state_dim, pose_dim
    in_dim = state_dim + 2 * act_dim + ctx_dim
    out_dim = state_dim + pose_dim
    self.net = mlp(in_dim, hidden, out_dim)
    self.in_norm = RunningNorm(in_dim)
    self.out_norm = RunningNorm(out_dim)  # the per-step change of s, and the pose delta

  @staticmethod
  def _inputs(s, a, a_prev, xi):
    return torch.cat([s, a, a_prev, xi], dim=-1)

  def update_stats(self, s, a, a_prev, xi, s_next, pose_delta) -> None:
    self.in_norm.update(self._inputs(s, a, a_prev, xi))
    self.out_norm.update(torch.cat([s_next - s, pose_delta], dim=-1))

  def forward(self, s, a, a_prev, xi):
    """Returns (s_next, pose_delta)."""
    y = self.out_norm.unnormalize(self.net(self.in_norm(self._inputs(s, a, a_prev, xi))))
    return s + y[:, : self.state_dim], y[:, self.state_dim :]

  def loss(self, s0, actions, actions_prev, xi, s_targets, pose_targets, valid):
    """k-step autoregressive loss in normalized output units.

    s0 (B, S); actions, actions_prev (B, k, A); xi (B, k, C); s_targets
    (B, k, S); pose_targets (B, k, P); valid (B, k) masks steps after a reset
    inside the window.
    """
    k = actions.shape[1]
    scale = self.out_norm.var.sqrt() + 1e-6
    s = s0
    total, denom = 0.0, valid.sum().clamp(min=1.0)
    for j in range(k):
      s_pred, pose_pred = self(s, actions[:, j], actions_prev[:, j], xi[:, j])
      err = torch.cat([s_pred - s_targets[:, j], pose_pred - pose_targets[:, j]], dim=-1)
      total = total + ((err / scale) ** 2).mean(-1).mul(valid[:, j]).sum()
      s = s_pred
    return total / denom
