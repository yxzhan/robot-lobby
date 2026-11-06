"""Franka Emika Panda (FER) through franky (github.com/TimSchneider42/franky),
on libfranka 0.9.x: what Panda System 4.2.x speaks (libfranka >= 0.10 is
for the FR3 only). franky's wheels are built per libfranka version: take the
0.9.2 ones from its GitHub releases (see real_robot/README.md).

Each command is an asynchronous JointMotion to the target: franky preempts
the running one and plans the new one with Ruckig within the robot's limits
scaled by `dynamics` (relative_dynamics_factor), so targets coming at
30-50 Hz over Wi-Fi with jitter still give a smooth, jerk-limited motion.

The hand runs on a thread of its own (franka::Gripper's calls block, a
grasp for seconds): the latest command wins, the width is read about 10
times a second. Closing is a grasp (it holds whatever is in between, with
`grasp_force`), opening a move to the full width.
"""

from __future__ import annotations

import threading
import time

from .base import GripperSpec, RobotDriver, RobotSpec, RobotState

PANDA = RobotSpec(
    name='panda',
    lobby_key='mujoco_menagerie/franka_emika_panda/panda.xml',
    joints=[f'joint{i}' for i in range(1, 8)],
    # Panda datasheet (FER): joint limits and maximum joint speeds.
    lower=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    upper=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    max_vel=[2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
    # The lobby's finger_joint1: one finger's opening, 0 .. 0.04 m.
    gripper=GripperSpec(joint='finger_joint1', open=0.04, closed=0.0, binary=True, mirror=['finger_joint2']),
)

MAX_WIDTH = 0.08  # m, both fingers


class PandaDriver(RobotDriver):
  def __init__(self, spec=PANDA, ip='172.16.0.2', dynamics=0.15, grasp_force=20.0, gripper_speed=0.1, **options):
    super().__init__(spec, **options)
    self.ip = ip
    self.dynamics = dynamics
    self.grasp_force = grasp_force
    self.gripper_speed = gripper_speed
    self.robot = None
    self.gripper = None
    self.width = None            # m, the hand's last reading
    self.grip_want = None        # 0 open / 1 closed: what the hand thread should do next
    self.grip_done = None        # what it did last
    self.grip_error = ''
    self.stop_hand = threading.Event()
    self.hand_thread = None
    self.fault = ''

  def connect(self):
    import franky  # here: the fake driver and the tests need no franky
    self.franky = franky
    self.robot = franky.Robot(self.ip)
    self.robot.recover_from_errors()
    self.robot.relative_dynamics_factor = self.dynamics
    if self.spec.gripper:
      try:
        self.gripper = franky.Gripper(self.ip)
        self.width = self.gripper.width
      except Exception as e:  # the arm still works without its hand
        self.gripper, self.grip_error = None, f'gripper: {e}'
      if self.gripper:
        self.hand_thread = threading.Thread(target=self.hand_loop, name='panda-hand', daemon=True)
        self.hand_thread.start()

  def read(self):
    s = self.robot.state
    finger = self.width / 2 if self.width is not None else None
    fault = self.fault or self.grip_error
    return RobotState(q=list(s.q), dq=list(s.dq), tau=list(s.tau_J), gripper=finger, fault=fault)

  def move(self, q):
    try:
      f = self.franky  # franky 2 takes a JointState only, 1.x a list too
      target = f.JointState(list(q)) if hasattr(f, 'JointState') else list(q)
      self.robot.move(f.JointMotion(target), asynchronous=True)
    except Exception as e:  # a reflex (collision, limits) ends the running motion with an error
      self.fault = str(e)

  def command(self, q):
    if not self.fault:
      self.move(q)

  def hold(self):
    if not self.fault:
      self.move(self.robot.state.q)

  def recover(self):
    self.robot.recover_from_errors()
    self.fault = ''

  def command_gripper(self, fraction):
    self.grip_want = 1 if fraction > 0.5 else 0

  def hand_loop(self):
    last_read = 0.0
    while not self.stop_hand.is_set():
      want = self.grip_want
      try:
        if want is not None and want != self.grip_done:
          self.grip_done = want
          if want:
            self.gripper.grasp(0.0, self.gripper_speed, self.grasp_force, epsilon_outer=MAX_WIDTH)
          else:
            self.gripper.move(MAX_WIDTH, self.gripper_speed)
        if time.monotonic() - last_read > 0.1:
          self.width = self.gripper.width
          last_read = time.monotonic()
        self.grip_error = ''
      except Exception as e:  # a failed grasp (nothing between the fingers is not one: epsilon_outer)
        self.grip_error = f'gripper: {e}'
      self.stop_hand.wait(0.02)

  def close(self):
    self.stop_hand.set()
    if self.robot:
      try:
        self.hold()
      except Exception:
        pass
