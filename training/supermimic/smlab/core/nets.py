"""Small networks shared by the world model, actor and critic."""

from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
  """Running mean/std over batches (Welford-style merge), frozen when not updated."""

  def __init__(self, dim: int, eps: float = 1e-4, clip: float = 10.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(eps))
    self.clip = clip

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1]).float()
    b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = b_mean - self.mean
    tot = self.count + b_n
    self.mean += delta * b_n / tot
    self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
    self.count = tot

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)

  def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
    return y * (self.var.sqrt() + 1e-6) + self.mean


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU) -> nn.Sequential:
  layers, d = [], in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)


class Actor(nn.Module):
  """Deterministic policy: normalized observation -> action (the env's action units)."""

  def __init__(self, obs_dim: int, hidden: tuple[int, ...], act_dim: int):
    super().__init__()
    self.norm = RunningNorm(obs_dim)
    self.net = mlp(obs_dim, hidden, act_dim, act=nn.ELU)
    # Near-zero actions to start: the robot holds its default pose rather than
    # being thrown over by a random network.
    with torch.no_grad():
      self.net[-1].weight.mul_(0.01)
      self.net[-1].bias.zero_()

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(obs))


class Critic(nn.Module):
  def __init__(self, in_dim: int, hidden: tuple[int, ...]):
    super().__init__()
    self.norm = RunningNorm(in_dim)
    self.net = mlp(in_dim, hidden, 1, act=nn.ELU)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(x))


def clip_grads(params, global_norm: float, elem: float) -> torch.Tensor:
  params = [p for p in params if p.grad is not None]
  norm = torch.nn.utils.clip_grad_norm_(params, global_norm)
  for p in params:
    p.grad.clamp_(-elem, elem)
  return norm
