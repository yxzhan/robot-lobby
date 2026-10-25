"""SuperMimic training loop, independent of the robot and the task.

One loop, three phases per iteration (SuperMimic Alg. A.1):
  1. Collect: the deterministic actor plus action noise steps the mjlab env;
     transitions go into a fixed-size replay buffer.
  2. Fit world model: k-step autoregressive windows from the buffer.
  3. Improve policy: BPTT of the imagined H-step return through the frozen
     world model, bootstrapped by a critic fit to TD(lambda) targets.

Adapted from go2-supermimic (github.com/Zhengtianjin/go2-supermimic, MIT), a
re-implementation of SuperMimic (Liao et al. 2026, arXiv:2610.28480), with
everything robot- and task-specific behind smlab.tasks.base.Task.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from smlab.core.buffer import ReplayBuffer, WindowArchive
from smlab.core.imagine import ImagineCfg, Imagination
from smlab.core.nets import Actor, Critic, clip_grads
from smlab.core.world_model import WorldModel
from smlab.tasks.base import apply_pose_delta, pose_delta


@dataclass(kw_only=True)
class TrainCfg:
  task: str = "duck-walk"
  num_envs: int = 4096
  buffer_len: int = 400
  iterations: int = 20_000
  warmup_iterations: int = 300
  """World-model-only iterations before policy updates start."""
  calibrate_steps: int = 100
  """Env steps collected before the task calibrates (Task.calibrate)."""
  action_noise: float = 0.05
  """Std of the Gaussian collection noise, in the env's action units (for the
  Microduck radians: go2-supermimic's 0.1 x 0.25 rad/unit, doubled)."""
  explore_frac: float = 0.25
  """Fraction of envs collecting with larger, temporally correlated (OU) action
  noise, so the world model sees more varied motion than the current policy."""
  explore_noise: float = 0.1
  explore_theta: float = 2.0
  """Mean reversion of the exploration OU noise [1/s]."""
  wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
  wm_window: int = 8
  wm_batch: int = 2048
  wm_updates: int = 2
  wm_prioritized_frac: float = 0.25
  """Fraction of each world-model batch drawn from current windows the task
  rates rare (contact changes, falls), where the model is least accurate."""
  wm_archive_frac: float = 0.25
  archive_size: int = 100_000
  archive_keep_prob: float = 0.02
  lr: float = 2e-4
  actor_lr_scale: float = 1.0
  lr_final_frac: float = 0.03
  lr_decay_iterations: int = 20_000
  horizon: int = 32
  imag_batch: int = 4096
  actor_hidden: tuple[int, ...] = (512, 256, 128)
  critic_hidden: tuple[int, ...] = (512, 256, 128)
  critic_epochs: int = 5
  critic_minibatches: int = 4
  critic_tau: float = 0.01
  """The bootstrap values (the rollout's last value and the TD(lambda) targets)
  come from a copy of the critic that follows it by this fraction per update:
  bootstrapping from the critic being fit, the actor drives it into states it
  overrates and the values run away."""
  grad_clip: float = 1.0
  grad_clip_elem: float = 0.1
  compile: bool = True
  curriculum_steps_per_iteration: int = 24
  """The env's curricula count env steps; PPO takes this many per iteration
  (rsl_rl num_steps_per_env), this loop one. The env's step counter is set to
  iteration x this, so curricula follow the iteration count as with PPO."""
  actor_ramp_iterations: int = 300
  """Whenever the actor starts (or restarts after a rollback) moving, its learning
  rate ramps up linearly over this many iterations, so the world model sees the
  new behaviour before the actor can exploit its errors."""
  resume: str = ""
  continue_run: bool = False
  collapse_ratio: float = 3.0
  collapse_floor: float = 0.05
  """Collapse guard: if falls per robot-second exceed both collapse_ratio times
  the lowest rate since policy updates started and collapse_floor, twice in a
  row, reload the last checkpoint and halve the learning rate."""
  log_interval: int = 50
  save_interval: int = 1000
  seed: int = 0
  run_name: str = "supermimic"
  log_root: str = "logs"


def lr_at(cfg: TrainCfg, it: int) -> float:
  t = min(1.0, it / cfg.lr_decay_iterations)
  frac = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * t))
  return cfg.lr * frac


class Trainer:
  def __init__(self, cfg: TrainCfg, task):
    from torch.utils.tensorboard import SummaryWriter

    self.cfg, self.task = cfg, task
    torch.manual_seed(cfg.seed)
    self.env = task.make_env(cfg.num_envs, cfg.seed)
    self.dev = self.env.device
    obs, _ = self.env.reset()
    task.bind(self.env)
    self.obs = obs["actor"]
    obs_dim = self.obs.shape[-1]
    assert obs_dim == task.obs_dim, (obs_dim, task.obs_dim)
    S, A, C = task.state_dim, task.act_dim, task.ctx_dim
    self.rec_dims = dict(task.record)
    R = sum(self.rec_dims.values())

    self.wm = WorldModel(S, A, C, 3 if task.planar else 0, cfg.wm_hidden).to(self.dev)
    self.actor = Actor(obs_dim, cfg.actor_hidden, A).to(self.dev)
    self.critic = Critic(task.critic_dim, cfg.critic_hidden).to(self.dev)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
    self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
    self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
    f32, f16 = torch.float32, torch.float16
    self.buf = ReplayBuffer(cfg.num_envs, cfg.buffer_len, {
      "s": (S, f32), "pose": (3, f32), "a": (A, f32), "a_prev": (A, f32), "xi": (C, f32),
      "rec": (R, f32), "obs": (obs_dim, f16),  # observations dominate memory: half precision
    }, self.dev)
    k = cfg.wm_window
    self.archive = WindowArchive(cfg.archive_size, {
      "s": (k + 1, S), "pose": (k + 1, 3), "a": (k, A), "a_prev": (k, A), "xi": (k, C), "valid": (k,),
    }, self.dev)
    self.imag = Imagination(ImagineCfg(horizon=cfg.horizon, compile=cfg.compile), task, self.dev)
    self.explore = torch.arange(cfg.num_envs, device=self.dev) < int(cfg.explore_frac * cfg.num_envs)
    self.ou = torch.zeros(cfg.num_envs, A, device=self.dev)
    self.a_prev = torch.zeros(cfg.num_envs, A, device=self.dev)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    self.log_dir = Path(cfg.log_root) / task.name / f"{stamp}_{cfg.run_name}"
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(str(self.log_dir))
    self.ep_stats: dict[str, list[float]] = defaultdict(list)
    self.env_steps = 0
    self.it = 0
    self.policy_start = cfg.warmup_iterations
    self.lr_scale = 1.0
    self.ramp_start = 0
    self.last_ckpt: Path | None = None
    self.collapse_strikes = 0
    self.best_falls = float("inf")
    self.nonfinite_steps = 0
    self.start_it = 0
    if cfg.resume:
      self.load(cfg.resume)

  # Record (task memory) packed into one buffer tensor, in task.record order.

  def pack(self, rec: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([rec[n].float().reshape(rec[n].shape[0], -1) for n in self.rec_dims], -1)

  def unpack(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
    out, i = {}, 0
    for n, d in self.rec_dims.items():
      out[n] = packed[:, i : i + d]
      i += d
    return out

  def load(self, path: str) -> None:
    ckpt = torch.load(path, map_location=self.dev, weights_only=False)
    self.wm.load_state_dict(ckpt["world_model"])
    self.actor.load_state_dict(ckpt["actor"])
    self.critic.load_state_dict(ckpt["critic"])
    self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
    for name in ("opt_wm", "opt_actor", "opt_critic"):
      if name in ckpt:
        getattr(self, name).load_state_dict(ckpt[name])
    if self.cfg.continue_run:
      self.start_it = int(ckpt.get("iteration", 0))
    print(f"resumed from {path} (iteration {ckpt.get('iteration')})", flush=True)

  # Phase 1.

  @torch.no_grad()
  def collect(self) -> None:
    env, cfg, task = self.env, self.cfg, self.task
    s, pose, xi = task.state(env), task.pose(env), task.context(env)
    rec = self.pack(task.read_record(env))
    a = self.actor(self.obs)
    dt = env.step_dt
    sigma_ou = cfg.explore_noise * (2 * cfg.explore_theta) ** 0.5
    self.ou = self.ou * (1 - cfg.explore_theta * dt) + sigma_ou * dt**0.5 * torch.randn_like(a)
    a = a + torch.where(self.explore.unsqueeze(-1), self.ou, cfg.action_noise * torch.randn_like(a))
    obs, _, term, trunc, extras = env.step(a)
    done = term | trunc
    self.ou = torch.where(done.unsqueeze(-1), torch.zeros_like(self.ou), self.ou)
    self.buf.add(done, s=s, pose=pose, a=a, a_prev=self.a_prev, xi=xi, rec=rec, obs=self.obs)
    keep = ~done
    if keep.any():
      s1 = task.state(env)
      dpose = pose_delta(pose, task.pose(env))
      self.wm.update_stats(s[keep], a[keep], self.a_prev[keep], xi[keep], s1[keep], dpose[keep])
    self.actor.norm.update(self.obs)
    self.a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)
    self.obs = obs["actor"]
    self.env_steps += env.num_envs
    for key, v in extras.get("log", {}).items():
      if key.startswith(("Episode_Reward", "Episode_Termination", "Metrics")):
        self.ep_stats[key].append(float(v))

  # Phase 2.

  def _gather(self, env, idx, valid) -> dict[str, torch.Tensor]:
    b, k = self.buf, self.cfg.wm_window
    e = env.unsqueeze(-1)
    return {"s": b["s"][e, idx], "pose": b["pose"][e, idx], "a": b["a"][e, idx[:, :k]],
            "a_prev": b["a_prev"][e, idx[:, :k]], "xi": b["xi"][e, idx[:, :k]], "valid": valid}

  def _wm_batch(self) -> dict[str, torch.Tensor]:
    """Uniform windows, rare windows from the buffer, and archived rare windows."""
    cfg, b = self.cfg, self.buf
    n_prio = int(cfg.wm_batch * cfg.wm_prioritized_frac)
    n_arch = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.size > 0 else 0
    n_uni = cfg.wm_batch - n_prio - n_arch
    parts = [self._gather(*b.sample_windows(n_uni, cfg.wm_window))]
    cand = self._gather(*b.sample_windows(4 * max(n_prio, 1), cfg.wm_window))
    score = self.task.rarity(cand["s"], cand["valid"])
    keep = self.task.archive_worthy(cand["s"], cand["valid"]) & (torch.rand_like(score) < cfg.archive_keep_prob)
    self.archive.add({k: v[keep] for k, v in cand.items()})
    top = (score + 1e-3 * torch.rand_like(score)).topk(n_prio).indices
    parts.append({k: v[top] for k, v in cand.items()})
    if n_arch:
      parts.append(self.archive.sample(n_arch))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

  def fit_world_model(self) -> float:
    cfg, k = self.cfg, self.cfg.wm_window
    losses = []
    for _ in range(cfg.wm_updates):
      w = self._wm_batch()
      pose_tgt = torch.stack([pose_delta(w["pose"][:, j], w["pose"][:, j + 1]) for j in range(k)], 1)
      if not self.task.planar:
        pose_tgt = pose_tgt[..., :0]
      loss = self.wm.loss(w["s"][:, 0], w["a"], w["a_prev"], w["xi"], w["s"][:, 1:], pose_tgt, w["valid"])
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      gn = clip_grads(self.wm.parameters(), cfg.grad_clip, float("inf"))
      if bool(torch.isfinite(gn)):
        self.opt_wm.step()
      else:
        self.nonfinite_steps += 1
      losses.append(loss.item())
    return sum(losses) / len(losses)

  # Phase 3.

  def improve_policy(self) -> dict[str, float]:
    cfg, b = self.cfg, self.buf
    env, t = b.sample_states(cfg.imag_batch)
    seed = {"s": b["s"][env, t], "pose": b["pose"][env, t], "a_prev": b["a_prev"][env, t],
            "obs": b["obs"][env, t].float(), "rec": self.unpack(b["rec"][env, t]), "xi": b["xi"][env, t]}
    k = self.task.rollout_consts(env)
    # Weights as tensors: changing them (curricula) does not recompile the step.
    w = {n: torch.tensor(float(v), device=self.dev) for n, v in self.task.weights(self.env).items()}
    for p in self.wm.parameters():
      p.requires_grad_(False)
    objective, data = self.imag.rollout(self.wm, self.actor, self.critic_target, seed, k, w)
    loss = -objective.mean()
    self.opt_actor.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = clip_grads(self.actor.parameters(), cfg.grad_clip, cfg.grad_clip_elem)
    finite = bool(torch.isfinite(gnorm)) and bool(torch.isfinite(loss))
    if finite:
      self.opt_actor.step()
    else:
      self.nonfinite_steps += 1
    for p in self.wm.parameters():
      p.requires_grad_(True)

    # Critic on the imagined transitions.
    self.critic.norm.update(data["critic_in"][:, :-1].detach().reshape(-1, data["critic_in"].shape[-1]))
    targets = self.imag.gae_targets(self.critic_target, data)
    x = data["critic_in"][:, :-1].detach().reshape(-1, data["critic_in"].shape[-1])
    y = targets.reshape(-1)
    n = x.shape[0]
    mb = n // cfg.critic_minibatches
    c_losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.dev)
      for i in range(cfg.critic_minibatches):
        sel = perm[i * mb : (i + 1) * mb]
        c_loss = (self.critic(x[sel]).squeeze(-1) - y[sel]).square().mean()
        self.opt_critic.zero_grad(set_to_none=True)
        c_loss.backward()
        gn = clip_grads(self.critic.parameters(), cfg.grad_clip, float("inf"))
        if bool(torch.isfinite(gn)):
          self.opt_critic.step()
        c_losses.append(c_loss.item())
    with torch.no_grad():
      for p, q in zip(self.critic.parameters(), self.critic_target.parameters()):
        q.lerp_(p, cfg.critic_tau)
      for n in ("mean", "var", "count"):
        getattr(self.critic_target.norm, n).copy_(getattr(self.critic.norm, n))
    return {
      "imagined_return": objective.mean().item(),
      "imagined_reward": data["rewards"].mean().item(),
      "imagined_survival": data["surv"][:, -1].mean().item(),
      "actor_grad_norm": float(gnorm),
      "critic_loss": sum(c_losses) / len(c_losses),
      "value": float(targets.mean()),
    }

  @torch.no_grad()
  def imagined_terms(self, batch: int = 1024) -> dict[str, float]:
    """Each reward term's weighted value per second along imagined rollouts
    (survival-weighted mean), to set beside the env's Episode_Reward ones."""
    if not hasattr(self.task, "terms"):
      return {}
    b = self.buf
    env, t = b.sample_states(batch)
    s, pose, a_prev = b["s"][env, t], b["pose"][env, t], b["a_prev"][env, t]
    obs, rec, xi = b["obs"][env, t].float(), self.unpack(b["rec"][env, t]), b["xi"][env, t]
    k = self.task.rollout_consts(env)
    w = self.task.weights(self.env)
    sums, surv, total = {}, torch.ones(batch, device=self.dev), 0.0
    for _ in range(self.cfg.horizon):
      a = self.actor(obs)
      s1, dpose = self.wm(s, a, a_prev, xi)
      pose1 = apply_pose_delta(pose, dpose) if self.task.planar else pose
      values, rec1 = self.task.terms(s, s1, a, a_prev, rec, k)
      for n, v in values.items():
        sums[n] = sums.get(n, 0.0) + float((surv * w.get(n, 0.0) * v).sum())
      total += float(surv.sum())
      _, d, obs, rec = self.task.imagine_step(s, s1, pose1, a, a_prev, rec, xi, k,
                                              {n: torch.tensor(float(x), device=self.dev) for n, x in w.items()})
      surv = surv * (1 - d)
      s, pose, a_prev = s1, pose1, a
    return {n: v / max(total, 1e-6) for n, v in sums.items()}

  def save(self, it: int) -> Path:
    path = self.log_dir / f"model_{it}.pt"
    torch.save({
      "supermimic": True, "task": self.task.name, "iteration": it, "cfg": asdict(self.cfg),
      "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
      "critic_target": self.critic_target.state_dict(),
      "world_model": self.wm.state_dict(), "opt_wm": self.opt_wm.state_dict(),
      "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
      "obs_dim": self.obs.shape[-1], "task_consts": self.task.saved_consts(),
    }, path)
    return path

  def run(self) -> None:
    cfg = self.cfg
    t_start = time.time()
    while self.buf.size < max(cfg.wm_window + 2, cfg.calibrate_steps):
      self.collect()
    self.task.calibrate(self.buf, self.unpack)
    for it in range(self.start_it + 1, cfg.iterations + 1):
      self.it = it
      self.env.common_step_counter = it * cfg.curriculum_steps_per_iteration
      lr = lr_at(cfg, it) * self.lr_scale
      start = max(self.ramp_start, self.policy_start)
      ramp = min(1.0, max(0.0, it - start) / max(1, cfg.actor_ramp_iterations))
      for opt, scale in ((self.opt_wm, 1.0), (self.opt_actor, ramp * cfg.actor_lr_scale), (self.opt_critic, 1.0)):
        for g in opt.param_groups:
          g["lr"] = lr * scale
      self.collect()
      stats = {"wm_loss": self.fit_world_model()}
      if it > self.policy_start:
        stats.update(self.improve_policy())
      if it % cfg.log_interval == 0:
        self.log(it, stats, time.time() - t_start)
      if it % cfg.save_interval == 0 or it == cfg.iterations:
        if self.collapse_strikes == 0:
          self.last_ckpt = self.save(it)
          print(f"saved {self.last_ckpt}", flush=True)

  def log(self, it: int, stats: dict[str, float], elapsed: float) -> None:
    for k, v in stats.items():
      self.writer.add_scalar(f"train/{k}", v, it)
    line = [f"it {it}/{self.cfg.iterations}", f"{elapsed / 60:.1f} min", f"env_steps {self.env_steps / 1e6:.1f}M"]
    line += [f"{k} {v:.4g}" for k, v in stats.items()]
    if self.nonfinite_steps:
      line.append(f"nonfinite_skipped {self.nonfinite_steps}")
    # The env logs the envs that fell in a step: per robot-second here.
    falls = self.ep_stats.get("Episode_Termination/fell_over", [])
    falls_rate = (sum(falls) / len(falls) if falls else 0.0) / (self.env.num_envs * self.env.step_dt)
    line.append(f"falls/robot-min {falls_rate * 60:.2f}")
    self.writer.add_scalar("Metrics/falls_per_robot_minute", falls_rate * 60, it)
    for k, vals in self.ep_stats.items():
      mean = sum(vals) / len(vals)
      self.writer.add_scalar(k, mean, it)
      if k in self.task.log_keys and k != "Episode_Termination/fell_over":
        line.append(f"{k.split('/')[-1]} {mean:.3g}")
    self.ep_stats.clear()
    self.writer.add_scalar("time/minutes", elapsed / 60, it)
    print(" | ".join(line), flush=True)
    if it > self.policy_start and it % (self.cfg.log_interval * 10) == 0:
      terms = self.imagined_terms()
      for n, v in terms.items():
        self.writer.add_scalar(f"Imagined_Reward/{n}", v, it)
      print("  imagined reward/s: " + " | ".join(f"{n} {v:.3g}" for n, v in terms.items() if v), flush=True)
    self._collapse_guard(falls_rate)

  def _collapse_guard(self, falls: float) -> None:
    """Model exploitation shows up as many real falls: roll back and slow down."""
    if self.it <= self.policy_start:
      return
    self.best_falls = min(self.best_falls, falls)
    if falls <= max(self.cfg.collapse_floor, self.cfg.collapse_ratio * self.best_falls):
      self.collapse_strikes = 0
      return
    self.collapse_strikes += 1
    if self.collapse_strikes >= 2 and self.last_ckpt is not None:
      # The world model is kept: it has just learned from the real falls the actor
      # exploited, and reloading it would let the same exploit happen again.
      ckpt = torch.load(self.last_ckpt, map_location=self.dev, weights_only=False)
      self.actor.load_state_dict(ckpt["actor"])
      self.critic.load_state_dict(ckpt["critic"])
      self.critic_target.load_state_dict(ckpt["critic_target"])
      self.lr_scale *= 0.5
      self.collapse_strikes = 0
      self.ramp_start = self.it
      self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr, betas=(0.9, 0.95))
      self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
      print(f"collapse guard: {falls * 60:.2f} falls/robot-min, rolled back to {self.last_ckpt}, "
            f"lr scale {self.lr_scale:.3g}", flush=True)
