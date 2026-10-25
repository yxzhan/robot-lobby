"""Microduck walking (microduck_rl's Mjlab-Velocity-Flat-MicroDuck) for SuperMimic.

The env is microduck_rl's own training env, unchanged: BAM servo model,
domain randomization, pushes, velocity / head / body pose commands, curricula.
Imagination mirrors its 61-D actor observation and its reward terms:

World-model state s (57):
  base height (1), gravity in the trunk frame (3), trunk linear velocity in
  the trunk frame (3) and angular velocity (3), q - default (14), qdot (14),
  the feet sites relative to the trunk in the yaw frame (2 x 3) and their
  velocity in the yaw frame (2 x 3), the foot height sensor's heights (2),
  the root angular momentum (3), the contact sensor's flags per foot (2).
The model predicts the contact flags too (as numbers, used clamped to
[0, 1]): from the joint angles it knows heel from toe, which the foot height
alone does not (a height threshold misjudges ~6% of foot samples).

Record (what imagination needs besides s, read from the env each step):
  cmd (13): twist (vx, vy, yaw rate), head pose (4), body pose (6)
  air (2): the contact sensor's current air time per foot
  peak (2): feet_swing_height's peak height since lift-off
  ema (4): head_pose_bias's EMA of the head error

Observation (as the env builds it): IMU angular velocity and gravity (a fixed
per-env IMU misalignment, 0-1 steps late), joint positions with the per-env
encoder bias, joint velocities one step late, the last action, the commands;
with the env's uniform noise.

Rewards: every term of the env with its live weight, as smooth functions of
the imagined state (the predicted contact, a smooth air-time window), so air
time, landings and slip have gradients. The air-time window starts wide (see
AIR_SOFTNESS): with the env's hard window, standing feet get no gradient to
step and the policy shuffles instead. A gait hint of imagination's own
(GAIT_HINT, annealed to zero) rewards the feet taking turns early on. self_collisions (rare, unmodelled) is
left out; body_pose_tracking has weight 0 in this task.
"""

from __future__ import annotations

import math

import torch

from smlab.tasks.base import Task, quat_apply, yaw_rotate_inverse

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
NJ = 14
Z = slice(0, 1)
GRAV = slice(1, 4)
V_B = slice(4, 7)
W_B = slice(7, 10)
Q = slice(10, 24)
QD = slice(24, 38)
FOOT_P = slice(38, 44)
FOOT_V = slice(44, 50)
FOOT_H = slice(50, 52)
ANGMOM = slice(52, 55)
CONTACT = slice(55, 57)
STATE_DIM = 57
FEET = ("left_foot", "right_foot")
NECK = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
OBS_NOISE = {"ang_vel": 0.03, "gravity": 0.01, "joint_pos": 0.001, "joint_vel": 0.25}
# Softness of the air-time window [s]: wide at first, so that a foot on the
# ground (air time 0) has a gradient towards lifting, narrowing to nearly the
# env's hard window as training goes on (iterations: the env's step counter
# over the steps per PPO iteration it stands for).
AIR_SOFTNESS = (0.08, 0.01)
AIR_SOFTNESS_ITERATIONS = 5000
# Gait hint (imagination only, like go2-supermimic's dodge hint): reward per
# second for one foot down and the other up, while the raised one has been up
# less than the air-time window's end (so standing on one leg earns nothing:
# the feet have to take turns). Without it the policy keeps both feet down 96%
# of the time and lifts them for 0.05 s, out of reach of the 0.125 s window.
GAIT_HINT = (3.0, 0.0)
GAIT_HINT_ITERATIONS = 5000
STEPS_PER_ITERATION = 24
FALL_SOFTNESS = math.radians(5.0)


class DuckWalk(Task):
  name = "duck-walk"
  planar = True
  state_dim = STATE_DIM
  act_dim = NJ
  obs_dim = 61
  record = {"cmd": 13, "air": 2, "peak": 2, "ema": 4}
  log_keys = ("Episode_Reward/track_linear_velocity", "Episode_Reward/track_angular_velocity",
              "Episode_Reward/air_time", "Episode_Reward/upright", "Episode_Termination/fell_over")

  # Env side.

  def make_env(self, num_envs: int, seed: int, play: bool = False, edit=None, render: bool = False):
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(TASK_ID, play=play)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if edit:
      edit(cfg)
    self.env_cfg = cfg
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)

  def bind(self, env) -> None:
    self.env = env
    robot = env.scene["robot"]
    self.robot = robot
    names = list(robot.joint_names)
    assert len(names) == NJ, names
    term = env.action_manager.get_term("joint_pos")
    assert term._target_ids.tolist() == list(range(NJ)) and float(term._scale) == 1.0
    self.feet, _ = robot.find_sites(FEET, preserve_order=True)
    self.neck = torch.tensor([names.index(n) for n in NECK], device=env.device)
    d = robot.data
    self.default_q = d.default_joint_pos.clone()
    self.soft_limits = d.soft_joint_pos_limits.clone()
    rm = env.reward_manager
    self.reward_terms = list(rm._term_names)
    self.term_cfg = dict(zip(rm._term_names, rm._term_cfgs))
    p = lambda n: self.term_cfg[n].params  # noqa: E731
    self.std_lin = p("track_linear_velocity")["std"]
    self.std_ang = p("track_angular_velocity")["std"]
    self.std_up = p("upright")["std"]
    pose = self.term_cfg["pose"]
    self.pose_ids = torch.tensor(pose.params["asset_cfg"].joint_ids, device=env.device)
    self.pose_std = torch.stack([pose.func.std_standing, pose.func.std_walking, pose.func.std_running])
    self.walk_threshold = pose.params["walking_threshold"]
    self.run_threshold = pose.params["running_threshold"]
    self.air_min, self.air_max = p("air_time")["threshold_min"], p("air_time")["threshold_max"]
    self.clearance_target = p("foot_clearance")["target_height"]
    self.swing_target = p("foot_swing_height")["target_height"]
    self.cmd_threshold = p("air_time")["command_threshold"]
    self.head_std = p("head_pose_tracking")["std"]
    self.head_tau = p("head_pose_bias")["tau_s"]
    self.fall_angle = self.env_cfg.terminations["fell_over"].params["limit_angle"]
    self.dt = env.step_dt
    self._bind_context(env)

  def _bind_context(self, env) -> None:
    """Foot friction, trunk and head COM offsets, armature scale: drawn at
    startup or every reset by the env's events."""
    robot, ix = self.robot, self.robot.indexing
    geoms, _ = robot.find_geoms(("left_foot_collision",))
    self.foot_geom = ix.geom_ids[geoms[0]]
    head = self.env_cfg.events["randomize_head_com"].params["asset_cfg"].body_names
    bodies, _ = robot.find_bodies(("trunk_base", *head), preserve_order=True)
    self.com_bodies = ix.body_ids[bodies]
    self.dofs = ix.joint_v_adr
    self.ipos0 = self._per_env(env.sim.model.body_ipos)[:, self.com_bodies].clone()
    self.arm0 = self._per_env(env.sim.model.dof_armature)[:, self.dofs].clone()
    self.ctx_dim = 1 + 3 * len(self.com_bodies) + 1

  def _per_env(self, x: torch.Tensor) -> torch.Tensor:
    """A model field per env (the domain-randomized ones are already)."""
    n = self.env.num_envs
    return x if x.ndim >= 2 and x.shape[0] == n else x.expand(n, *x.shape)

  @property
  def critic_dim(self) -> int:
    return STATE_DIM + sum(self.record.values()) + self.ctx_dim

  def state(self, env) -> torch.Tensor:
    d = self.robot.data
    heading = d.heading_w
    root = d.root_link_pos_w
    feet_p = d.site_pos_w[:, self.feet] - root.unsqueeze(1)
    feet_v = d.site_lin_vel_w[:, self.feet]
    rel = torch.cat([yaw_rotate_inverse(feet_p[:, i], heading) for i in range(2)], -1)
    vel = torch.cat([yaw_rotate_inverse(feet_v[:, i], heading) for i in range(2)], -1)
    return torch.cat([
      root[:, 2:3] - env.scene.terrain.env_origins[:, 2:3],
      d.projected_gravity_b, d.root_link_lin_vel_b, d.root_link_ang_vel_b,
      d.joint_pos - self.default_q, d.joint_vel, rel, vel,
      env.scene["foot_height_scan"].data.heights,
      env.scene["robot/root_angmom"].data,
      (env.scene["feet_ground_contact"].data.found > 0).float(),
    ], -1).nan_to_num()

  def pose(self, env) -> torch.Tensor:
    d = self.robot.data
    return torch.cat([d.root_link_pos_w[:, :2], d.heading_w.unsqueeze(-1)], -1)

  def context(self, env) -> torch.Tensor:
    m = env.sim.model
    fr = self._per_env(m.geom_friction)[:, self.foot_geom, 0:1]
    ipos = self._per_env(m.body_ipos)[:, self.com_bodies] - self.ipos0
    arm = (self._per_env(m.dof_armature)[:, self.dofs] / self.arm0.clamp(min=1e-9)).mean(-1, keepdim=True)
    return torch.cat([fr, ipos.reshape(env.num_envs, -1), arm], -1)

  def read_record(self, env) -> dict[str, torch.Tensor]:
    cm = env.command_manager
    contact = env.scene["feet_ground_contact"].data
    swing = self.term_cfg["foot_swing_height"].func
    ema = getattr(env, "_head_bias_ema", None)
    return {
      "cmd": torch.cat([cm.get_command("twist"), cm.get_command("head_pose"), cm.get_command("body_pose")], -1),
      "air": contact.current_air_time,
      "peak": swing.peak_heights,
      "ema": ema if ema is not None else torch.zeros(env.num_envs, 4, device=env.device),
    }

  def weights(self, env) -> dict[str, float]:
    return {n: c.weight for n, c in self.term_cfg.items()}

  def saved_consts(self) -> dict:
    return {}

  # Imagination side.

  def rollout_consts(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    B = env_ids.shape[0]
    return {
      "default_q": self.default_q[env_ids],
      "soft_limits": self.soft_limits[env_ids],
      "enc_bias": self.robot.data.encoder_bias[env_ids],
      "imu_q": self.env._imu_misalign_quat[env_ids],
      "imu_late": (torch.rand(B, 1, device=env_ids.device) < 0.5).float(),
      "air_softness": torch.tensor(self.anneal(AIR_SOFTNESS, AIR_SOFTNESS_ITERATIONS), device=env_ids.device),
      "gait_hint": torch.tensor(self.anneal(GAIT_HINT, GAIT_HINT_ITERATIONS), device=env_ids.device),
    }

  def anneal(self, ends: tuple[float, float], iterations: int) -> float:
    """From ends[0] to ends[1] linearly over the first `iterations`."""
    f = min(1.0, self.env.common_step_counter / STEPS_PER_ITERATION / iterations)
    return ends[0] + f * (ends[1] - ends[0])

  def observe(self, s0, s1, a, cmd, k) -> torch.Tensor:
    late = k["imu_late"]
    w = late * s0[:, W_B] + (1 - late) * s1[:, W_B]
    g = late * s0[:, GRAV] + (1 - late) * s1[:, GRAV]
    noise = lambda x, n: x + (torch.rand_like(x) * 2 - 1) * OBS_NOISE[n]  # noqa: E731
    return torch.cat([
      noise(quat_apply(k["imu_q"], w), "ang_vel"),
      noise(quat_apply(k["imu_q"], g), "gravity"),
      noise(s1[:, Q] + k["enc_bias"], "joint_pos"),
      noise(s0[:, QD], "joint_vel"),  # one step late
      a, cmd,
    ], -1)

  def terms(self, s0, s1, a, a_prev, rec, k):
    """Each reward term's value at s1 (unweighted, per step) and the record at s1."""
    cmd = rec["cmd"]
    speed = cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()
    active = (speed > self.cmd_threshold).float()
    v, w, g = s1[:, V_B], s1[:, W_B], s1[:, GRAV]
    q = s1[:, Q] + k["default_q"]
    lim = k["soft_limits"]
    regime = torch.stack([speed < self.walk_threshold,
                          (speed >= self.walk_threshold) & (speed < self.run_threshold),
                          speed >= self.run_threshold], -1).float()
    std = regime @ self.pose_std  # (B, J)
    h, fv = s1[:, FOOT_H], s1[:, FOOT_V].reshape(-1, 2, 3)
    fv_xy = fv[..., :2].square().sum(-1)
    c0, c1 = s0[:, CONTACT].clamp(0, 1), s1[:, CONTACT].clamp(0, 1)
    first = c1 * (1 - c0)
    air = (1 - c1) * (rec["air"] + self.dt)
    soft = k["air_softness"]
    in_range = torch.sigmoid((air - self.air_min) / soft) * torch.sigmoid((self.air_max - air) / soft)
    peak = torch.where(c1 < 0.5, torch.maximum(rec["peak"], h), rec["peak"])
    swing = ((peak / self.swing_target - 1).square() * first).sum(-1)
    peak = peak * (1 - first)
    head_err = s1[:, Q][:, self.neck] - cmd[:, 3:7]
    alpha = min(1.0, self.dt / max(self.head_tau, 1e-6))
    ema = (1 - alpha) * rec["ema"] + alpha * head_err
    up = -g
    values = {
      "track_linear_velocity": torch.exp(-((cmd[:, :2] - v[:, :2]).square().sum(-1) + v[:, 2].square()) / self.std_lin**2),
      "track_angular_velocity": torch.exp(-((cmd[:, 2] - w[:, 2]).square() + w[:, :2].square().sum(-1)) / self.std_ang**2),
      "upright": torch.exp(-g[:, :2].square().sum(-1) / self.std_up**2),
      "pose": torch.exp(-(s1[:, Q][:, self.pose_ids].square() / std.square()).mean(-1)),
      # World-frame xy angular velocity: what is left after the vertical part.
      "body_ang_vel": (w.square().sum(-1) - (w * up).sum(-1).square()).clamp(min=0),
      "angular_momentum": s1[:, ANGMOM].square().sum(-1),
      "dof_pos_limits": ((lim[..., 0] - q).clamp(min=0) + (q - lim[..., 1]).clamp(min=0)).sum(-1),
      "action_rate_l2": (a - a_prev).square().sum(-1),
      "air_time": in_range.sum(-1) * active,
      "foot_clearance": ((h - self.clearance_target).abs() * (fv_xy + 1e-8).sqrt()).sum(-1) * active,
      "foot_swing_height": swing * active,
      "foot_slip": (fv_xy * c1).sum(-1) * active,
      "head_pose_tracking": torch.exp(-(head_err / self.head_std).square()).mean(-1),
      "head_pose_bias": -ema.abs().mean(-1),
      "gait_hint": ((1 - c1) * c1.flip(-1) * torch.sigmoid((self.air_max - air) / soft)).sum(-1) * active,
    }
    rec1 = {"cmd": cmd, "air": air, "peak": peak, "ema": ema}
    return values, rec1

  def fall_prob(self, s: torch.Tensor) -> torch.Tensor:
    """Soft bad_orientation: tilt from the gravity z-component."""
    tilt = torch.acos((-s[:, GRAV][:, 2]).clamp(-1.0 + 1e-4, 1.0 - 1e-4))  # acos' is infinite at +-1
    return torch.sigmoid((tilt - self.fall_angle) / FALL_SOFTNESS)

  def imagine_step(self, s0, s1, pose1, a, a_prev, rec, xi, k, w):
    values, rec1 = self.terms(s0, s1, a, a_prev, rec, k)
    r = (sum(w[n] * v for n, v in values.items() if n in w) + k["gait_hint"] * values["gait_hint"]) * self.dt
    obs = self.observe(s0, s1, a, rec["cmd"], k)
    return r, self.fall_prob(s1), obs, rec1

  def critic_input(self, s, rec, xi) -> torch.Tensor:
    return torch.cat([s] + [rec[n] for n in self.record] + [xi], -1)

  def rarity(self, s: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Contact changes and an early end (a fall) per window."""
    contact = (s[..., CONTACT] > 0.5).float()  # (B, k+1, 2)
    changes = (contact[:, 1:] - contact[:, :-1]).abs().sum((1, 2))
    ended = (valid[:, -1] < 0.5).float() * 4.0
    return changes + ended

  def archive_worthy(self, s: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """A fall, a hard tilt rate, or both feet off the ground for two steps."""
    off = (s[..., CONTACT] < 0.5).all(-1).float().sum(-1) >= 2
    tilt_rate = s[..., W_B][..., :2].norm(dim=-1).amax(-1) > 3.0
    return (valid[:, -1] < 0.5) | tilt_rate | off
