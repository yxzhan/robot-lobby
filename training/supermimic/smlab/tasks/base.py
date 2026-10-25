"""What a task gives the SuperMimic core (smlab/core).

The core collects transitions in the task's mjlab env, fits a world model of
the task's state and improves the actor through imagined rollouts in it. The
task supplies everything robot- and task-specific:

  make_env(num_envs, seed, play, edit, render)
                             the mjlab env (training config: commands,
                             domain randomization, curricula, its rewards)
  bind(env)                  read constants (default pose, limits, encoder
                             bias, ...) once the env exists
  state_dim, ctx_dim, act_dim, obs_dim, critic_dim
  planar                     the robot moves on the floor: the world model
                             also predicts (dx, dy, dyaw) in its yaw frame
  state(env)                 (N, S) world-model state
  pose(env)                  (N, 3) planar pose (x, y, yaw)
  context(env)               (N, C) domain-randomized parameters (per step:
                             some are drawn again at every reset)
  record                     {name: dim} of what imagination needs besides the
                             state and is read from the env each step (the
                             commands, reward memory such as air times)
  read_record(env)           {name: (N, dim)}
  weights(env)               {reward term: weight} now (curricula change them)
  rollout_consts(env_ids)    per-rollout constants for imagine_step, indexed
                             by the seeds' envs
  imagine_step(s, s1, pose1, a, a_prev, rec, xi, k, w)
                             one imagined step after the world model gave s1:
                             (reward (B,) already times dt, fall probability
                             (B,), the actor observation at s1, rec at s1);
                             differentiable, a pure function (torch.compile)
  critic_input(s, rec, xi)   (B, critic_dim)
  rarity(s, valid)           per window (B, k+1, S): how much the world model
                             should see it (contact changes, falls)
  archive_worthy(s, valid)   per window: rare enough to keep for long
  calibrate(buf, unpack)     optional: fit task constants on the first
                             collected data
  log_keys                   env log keys printed with every log line
"""

from __future__ import annotations

import torch


class Task:
  name = "task"
  planar = True
  log_keys: tuple[str, ...] = ()
  record: dict[str, int] = {}

  def make_env(self, num_envs: int, seed: int, play: bool = False, edit=None, render: bool = False):
    """edit(env_cfg), if given, changes the config before the env is made;
    render: env.render() returns frames."""
    raise NotImplementedError

  def bind(self, env) -> None:
    pass

  def calibrate(self, buf, unpack) -> None:
    """After the first steps are in the replay buffer `buf` (its records split
    by unpack(packed)), before training starts."""

  def saved_consts(self) -> dict:
    """Task constants saved with a checkpoint."""
    return {}


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate v (..., 3) by unit quaternions q (..., 4) (w, x, y, z)."""
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v + w * t + torch.cross(xyz, t, dim=-1)


def yaw_rotate_inverse(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """World vector(s) v (N, 3) into the yaw frame."""
  c, s = yaw.cos(), yaw.sin()
  return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)


def pose_delta(pose0: torch.Tensor, pose1: torch.Tensor) -> torch.Tensor:
  """Planar motion from pose0 to pose1, expressed in pose0's yaw frame."""
  c, s = pose0[:, 2].cos(), pose0[:, 2].sin()
  dx, dy = pose1[:, 0] - pose0[:, 0], pose1[:, 1] - pose0[:, 1]
  dyaw = torch.atan2((pose1[:, 2] - pose0[:, 2]).sin(), (pose1[:, 2] - pose0[:, 2]).cos())
  return torch.stack([c * dx + s * dy, -s * dx + c * dy, dyaw], dim=-1)


def apply_pose_delta(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
  c, s = pose[:, 2].cos(), pose[:, 2].sin()
  x = pose[:, 0] + c * delta[:, 0] - s * delta[:, 1]
  y = pose[:, 1] + s * delta[:, 0] + c * delta[:, 1]
  return torch.stack([x, y, pose[:, 2] + delta[:, 2]], dim=-1)
