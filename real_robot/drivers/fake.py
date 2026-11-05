"""A stand-in for a real robot: follows its commands at its spec's joint
speed limits, its gripper too, so the whole chain (lobby, bridge, safety
layer, ghost) runs without hardware:

    python -m real_robot --robot panda --driver fake

`lag` (s) delays what it reports, like a slow network or controller."""

from __future__ import annotations

import collections
import time

from .base import RobotDriver, RobotState


class FakeDriver(RobotDriver):
  def __init__(self, spec, start=None, lag=0.0, **options):
    super().__init__(spec, **options)
    self.q = list(start) if start else [(lo + hi) / 2 for lo, hi in zip(spec.lower, spec.upper)]
    self.target = list(self.q)
    self.dq = [0.0] * len(self.q)
    g = spec.gripper
    self.grip = g.open if g else None
    self.grip_target = self.grip
    self.lag = lag
    self.history = collections.deque()
    self.t = None
    self.fault = ''

  def connect(self):
    self.t = time.monotonic()

  def step(self):
    now = time.monotonic()
    dt, self.t = now - self.t, now
    for i, (q, t, v) in enumerate(zip(self.q, self.target, self.spec.max_vel)):
      dq = max(-v * dt, min(v * dt, t - q))
      self.q[i] = q + dq
      self.dq[i] = dq / dt if dt > 0 else 0.0
    g = self.spec.gripper
    if g:
      rate = abs(g.open - g.closed) * dt / 0.5  # open to closed in 0.5 s
      self.grip += max(-rate, min(rate, self.grip_target - self.grip))

  def read(self):
    self.step()
    now = time.monotonic()
    self.history.append((now, RobotState(list(self.q), list(self.dq), None, self.grip, self.fault)))
    while len(self.history) > 1 and self.history[1][0] <= now - self.lag:
      self.history.popleft()
    return self.history[0][1]

  def command(self, q):
    self.target = list(q)

  def command_gripper(self, fraction):
    g = self.spec.gripper
    if g:
      self.grip_target = g.value(round(fraction) if g.binary else fraction)

  def hold(self):
    self.target = list(self.q)

  def recover(self):
    self.fault = ''
