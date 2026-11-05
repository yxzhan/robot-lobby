"""Between the lobby's targets and the robot: what may reach it, when.

  disabled  nothing is commanded, the robot holds. Enabling needs a fresh
            target within spec.start_tolerance of the real robot on every
            joint (the lobby's Sync puts its robot where the real one is
            first), and clears a fault the robot reports (driver.recover).
  enabled   each control tick: the latest target, kept spec.limit_margin
            inside the joint limits, approached at most spec.max_vel (times
            `speed`) from the last command, so a jump in the targets (a
            marker grabbed far away, a lost packet) becomes a bounded
            motion. A target older than `timeout` (the Quest asleep, Wi-Fi
            gone) holds the robot where it is until targets come again.
  fault     the robot reported an error, or fell more than spec.max_lead
            behind its command (blocked, a collision): it holds until
            enabled again.
"""

from __future__ import annotations

import dataclasses
import math

DISABLED, ENABLED, FAULT = 'disabled', 'enabled', 'fault'


@dataclasses.dataclass
class Step:
  q: list[float] | None   # the command to send, or None: hold
  stale: bool = False


class Supervisor:
  def __init__(self, spec, timeout=0.3, speed=0.5):
    self.spec = spec
    self.timeout = timeout    # s without a target: hold
    self.speed = speed        # of spec.max_vel
    self.state = DISABLED
    self.reason = ''          # why it is disabled or in fault
    self.cmd = None           # the last command
    self.error = 0.0          # rad: the largest |command - real| of the last tick

  def offsets(self, target, real):
    return [abs(t - r) for t, r in zip(target, real)]

  def enable(self, target, target_age, real):
    """Tries to enable; returns '' or why not."""
    if target is None or target_age > self.timeout:
      return self.refuse('no targets from the lobby (is its ROS on, and this bridge its address?)')
    if real is None:
      return self.refuse('no state from the robot')
    off = self.offsets(self.spec.clamp(target), real)
    worst = max(range(len(off)), key=off.__getitem__)
    if off[worst] > self.spec.start_tolerance:
      return self.refuse(f'the lobby is {off[worst]:.2f} rad from the real robot at {self.spec.joints[worst]} '
                         f'(at most {self.spec.start_tolerance}): Sync it to the real robot first')
    self.state, self.reason, self.cmd = ENABLED, '', list(real)
    return ''

  def refuse(self, why):
    if self.state != FAULT:
      self.reason = why
    return why

  def disable(self, why=''):
    if self.state == ENABLED:
      self.state, self.reason = DISABLED, why
    elif self.state == FAULT and not why:  # disabling clears a fault (enabling again needs a Sync)
      self.state = DISABLED

  def fault(self, why):
    self.state, self.reason, self.cmd = FAULT, why, None

  def step(self, target, target_age, real, dt, robot_fault=''):
    """One control tick: the command, or None to hold."""
    if robot_fault and self.state == ENABLED:
      self.fault(f'the robot: {robot_fault}')
    if self.state != ENABLED:
      return Step(None)
    if target is None or target_age > self.timeout:
      self.cmd = list(real)  # going on from where it holds
      return Step(None, stale=True)
    goal = self.spec.clamp(target)
    cmd = []
    for c, g, v in zip(self.cmd, goal, self.spec.max_vel):
      d = v * self.speed * dt
      cmd.append(c + max(-d, min(d, g - c)))
    off = self.offsets(cmd, real)
    self.error = max(off) if off else 0.0
    if self.error > self.spec.max_lead:
      j = self.spec.joints[off.index(self.error)]
      self.fault(f'{j} is {self.error:.2f} rad behind its command (blocked?)')
      return Step(None)
    if not all(math.isfinite(v) for v in cmd):
      self.fault('a target that is not a number')
      return Step(None)
    self.cmd = cmd
    return Step(cmd)
