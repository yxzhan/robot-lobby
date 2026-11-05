"""What every real robot looks like to the bridge: a RobotSpec (its joints as
the lobby names them, their limits, its gripper) and a RobotDriver (connect,
read, command, stop). A new robot is one module in this package with both,
registered in drivers/__init__.py; the bridge, the safety layer and the
lobby side stay as they are.

Units are the lobby's: joint positions in rad (m for a slide), the gripper
as the lobby's finger joint reports it (spec.gripper: which joint, its open
and closed values). A driver turns them into its robot's own (the Panda's
gripper width, the SO-101's servo ticks).
"""

from __future__ import annotations

import abc
import dataclasses
import math


@dataclasses.dataclass
class GripperSpec:
  joint: str              # the lobby's joint that stands for it (in joint states and commands)
  open: float             # that joint's value when open ...
  closed: float           # ... and closed
  binary: bool = False    # the real gripper only opens or closes (the Panda hand: move vs grasp)
  mirror: list[str] = dataclasses.field(default_factory=list)  # lobby joints reported with the same value (the Panda's second finger)

  def fraction(self, value):
    """Joint value -> 0 (open) .. 1 (closed)."""
    span = self.closed - self.open
    return min(1.0, max(0.0, (value - self.open) / span)) if span else 0.0

  def value(self, fraction):
    """0 (open) .. 1 (closed) -> joint value."""
    return self.open + (self.closed - self.open) * min(1.0, max(0.0, fraction))


@dataclasses.dataclass
class RobotSpec:
  name: str
  lobby_key: str          # the lobby's robot file (public/json/ros_profiles.json's key)
  joints: list[str]       # the arm's joints, in the driver's order, named as the lobby names them
  lower: list[float]      # joint limits (rad), the real robot's
  upper: list[float]
  max_vel: list[float]    # rad/s the bridge never commands beyond
  gripper: GripperSpec | None = None
  limit_margin: float = 0.02      # rad kept clear of each limit
  start_tolerance: float = 0.15   # rad: the lobby's target this near the real robot to enable
  max_lead: float = 0.5           # rad: the command never more than this ahead of the real robot (else a fault)

  def __post_init__(self):
    n = len(self.joints)
    for f in ('lower', 'upper', 'max_vel'):
      if len(getattr(self, f)) != n:
        raise ValueError(f'{self.name}: {f} has {len(getattr(self, f))} values for {n} joints')

  def clamp(self, q):
    m = self.limit_margin
    return [min(hi - m, max(lo + m, v)) for v, lo, hi in zip(q, self.lower, self.upper)]


@dataclasses.dataclass
class RobotState:
  q: list[float]                     # arm joint positions, spec.joints' order
  dq: list[float] | None = None
  tau: list[float] | None = None     # measured joint torques, if the robot has them
  gripper: float | None = None       # the lobby's finger joint value (spec.gripper)
  fault: str = ''                    # the robot reports an error (reflex, communication...)

  def finite(self):
    return len(self.q) > 0 and all(math.isfinite(v) for v in self.q)


class RobotDriver(abc.ABC):
  """One real robot. The bridge calls these from its control thread only:
  a driver needs no locking of its own. command() is called at the bridge's
  rate with targets already filtered by the safety layer (in limits, near
  the robot, velocity limited); it should not block for long."""

  spec: RobotSpec

  def __init__(self, spec: RobotSpec, **options):
    self.spec = spec
    self.options = options

  @abc.abstractmethod
  def connect(self):
    """Opens the connection; raises if the robot cannot be reached."""

  @abc.abstractmethod
  def read(self) -> RobotState:
    """The robot's state now."""

  @abc.abstractmethod
  def command(self, q):
    """Moves the arm towards joint positions q (spec.joints' order)."""

  def command_gripper(self, fraction):
    """0 open .. 1 closed (spec.gripper.binary: only past halfway counts)."""

  @abc.abstractmethod
  def hold(self):
    """Stops where the arm is (disabled, stale targets, a fault)."""

  def recover(self):
    """Clears a fault the robot reports, before enabling again."""

  def close(self):
    """Lets go of the robot (the bridge exits)."""
