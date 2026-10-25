"""Fixed-size per-env replay buffer of simulator transitions (oldest replaced first),
and an archive of rare world-model windows.

Each slot stores the state *before* the action, the executed (noisy) action and
whatever the task records (its commands and reward memory, e.g. air times).
The next state is the following slot of the same env unless `done` is set, in
which case the following slot starts a new episode.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
  def __init__(self, num_envs: int, length: int, fields: dict[str, tuple[int, torch.dtype]],
               device: str):
    """fields: name -> (per-env dim, dtype), besides `done`."""
    self.n, self.L, self.device = num_envs, length, device
    self.data = {k: torch.zeros(num_envs, length, d, device=device, dtype=t) for k, (d, t) in fields.items()}
    self.done = torch.zeros(num_envs, length, device=device)
    self.ptr = 0
    self.size = 0

  def __getitem__(self, key: str) -> torch.Tensor:
    return self.data[key]

  def add(self, done: torch.Tensor, **values: torch.Tensor) -> None:
    i = self.ptr
    for k, v in values.items():
      self.data[k][:, i] = v.to(self.data[k].dtype)
    self.done[:, i] = done.float()
    self.ptr = (self.ptr + 1) % self.L
    self.size = min(self.size + 1, self.L)

  def _phys(self, logical: torch.Tensor) -> torch.Tensor:
    start = (self.ptr - self.size) % self.L
    return (start + logical) % self.L

  def sample_windows(self, batch: int, k: int):
    """Random k-step windows: (env_ids, slot indices (B, k+1), valid (B, k))."""
    assert self.size > k + 1, "buffer too small for windows"
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t0 = torch.randint(0, self.size - k, (batch,), device=self.device)
    idx = self._phys(t0.unsqueeze(-1) + torch.arange(k + 1, device=self.device))
    done = self.done[env.unsqueeze(-1), idx[:, :k]]
    # Transition j (slot j -> j+1) is valid if no slot 0..j ended an episode.
    valid = torch.cumprod(1.0 - done, dim=1)
    return env, idx, valid

  def sample_states(self, batch: int):
    """Random (env_ids, slot) pairs to seed imagined rollouts."""
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t = self._phys(torch.randint(0, self.size, (batch,), device=self.device))
    return env, t


class WindowArchive:
  """Ring store of k-step world-model windows with rare events (falls, flight,
  hard impacts). Unlike the replay buffer it is not overwritten every few
  seconds of sim time, so the world model keeps seeing them."""

  def __init__(self, capacity: int, shapes: dict[str, tuple[int, ...]], device: str):
    self.cap, self.device = capacity, device
    self.data = {k: torch.zeros(capacity, *shape, device=device) for k, shape in shapes.items()}
    self.ptr = 0
    self.size = 0

  def add(self, windows: dict[str, torch.Tensor]) -> None:
    n = next(iter(windows.values())).shape[0]
    if n == 0:
      return
    n = min(n, self.cap)
    idx = (self.ptr + torch.arange(n, device=self.device)) % self.cap
    for k, v in self.data.items():
      v[idx] = windows[k][:n].to(v.dtype)
    self.ptr = (self.ptr + n) % self.cap
    self.size = min(self.size + n, self.cap)

  def sample(self, batch: int) -> dict[str, torch.Tensor]:
    idx = torch.randint(0, self.size, (batch,), device=self.device)
    return {k: v[idx] for k, v in self.data.items()}
