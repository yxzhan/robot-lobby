"""python3 -m real_robot.ros2 --robot panda [--driver real|fake|lobby] ...

The ROS 2 front end: a robot as the hardware of ros2_control's
topic_based_ros2_control, its arm velocity commanded (forward_velocity_
controller; giskardpy's universal_robot_velocity drives it), its gripper an
action. The same drivers and safety layer as the lobby's WebSocket front end
(__main__.py), with one more driver here: `lobby`, the lobby's robot over
rosbridge (a digital twin to try a program on first).

Topics, under --ns (/<robot>/hw):

  <ns>/joint_commands   in   sensor_msgs/JointState, velocity: the arm's
                             (topic_based_ros2_control's joint_commands_topic)
  <ns>/joint_states     out  the arm and the gripper's joint as ros2_control
                             reads them (its joint_states_topic): the arm's
                             positions the commanded ones (giskard closes its
                             loop on what it commanded, not through the robot's
                             tracking lag), the gripper measured
  <ns>/measured_joint_states out  the robot's measured joints
  <ns>/status           out  std_msgs/String, JSON (safety.py's state, reason)
  <ns>/enable           srv  std_srvs/SetBool: enable / disable (--enable: at start)
  <ns>/gripper_cmd      action control_msgs/ParallelGripperCommand: the gripper's
                             joint (spec.gripper: the Panda's finger_joint1,
                             each finger's opening 0 .. 0.04 m)

Each control tick (--rate) the velocities are integrated from the last
command into the next, which the safety layer bounds (limits, speed, the
robot not falling behind); commands older than --timeout hold the robot.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import rclpy
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

from . import drivers
from .drivers.base import RobotDriver, RobotState
from .safety import ENABLED, Supervisor

GRIP_TOLERANCE = 0.002   # spec.gripper units (m): reached
GRIP_STALL_S = 0.5       # not moving this long: stalled (holding something) or done
GRIP_TIMEOUT_S = 10.0


class LobbyDriver(RobotDriver):
  """The lobby's robot through ROS (rosbridge_server): its joint states in,
  position targets out on its joint command topic (in the lobby: its IK off,
  its ROS on; the ROS profile's prefix names the topics, /panda/...)."""

  def __init__(self, spec, node=None, states='', commands='', **options):
    super().__init__(spec, **options)
    self.node = node
    self.states = states or f'/{spec.name}/joint_states'
    self.commands = commands or f'/{spec.name}/joint_command'
    self.q = None
    self.grip = None
    self.grip_cmd = None
    self.lock = threading.Lock()

  def connect(self):
    self.pub = self.node.create_publisher(JointState, self.commands, 10)
    self.node.create_subscription(JointState, self.states, self.on_state, 10)

  def on_state(self, msg):
    by = dict(zip(msg.name, msg.position))
    if all(j in by for j in self.spec.joints):
      g = self.spec.gripper
      with self.lock:
        self.q = [by[j] for j in self.spec.joints]
        if g and g.joint in by:
          self.grip = by[g.joint]

  def read(self):
    with self.lock:
      return RobotState(q=list(self.q or []), gripper=self.grip)

  def send(self, q):
    names, pos = list(self.spec.joints), [float(v) for v in q]
    if self.grip_cmd is not None:
      names.append(self.spec.gripper.joint)
      pos.append(self.grip_cmd)
    self.pub.publish(JointState(name=names, position=pos))

  def command(self, q):
    self.send(q)

  def command_gripper(self, fraction):
    self.grip_cmd = self.spec.gripper.value(fraction)

  def hold(self):
    with self.lock:
      q = self.q
    if q:
      self.send(q)


class Ros2Front(Node):
  def __init__(self, a):
    super().__init__(f'{a.robot}_hw')
    self.a = a
    self.ns = (a.ns or f'/{a.robot}/hw').rstrip('/')
    opts = {'fake': {'lag': a.lag},
            'lobby': {'node': self, 'states': a.lobby_states, 'commands': a.lobby_commands},
            'panda': {'ip': a.ip, 'dynamics': a.dynamics, 'grasp_force': a.grasp_force},
            'so101': {'port': a.serial, 'calibration': a.calibration}}
    if a.driver == 'lobby':
      self.driver = LobbyDriver(drivers.spec(a.robot), **opts['lobby'])
    else:
      self.driver = drivers.driver(a.robot, a.driver, **(opts['fake'] if a.driver == 'fake' else opts.get(a.robot, {})))
    self.spec = self.driver.spec
    self.safety = Supervisor(self.spec, timeout=a.timeout, speed=a.speed)
    self.lock = threading.Lock()
    self.vel = None            # the arm's commanded velocities, spec.joints' order
    self.vel_t = 0.0
    self.requests = [True] if a.enable else []
    self.grip_goal = None      # 0 open .. 1 closed, for the driver
    self.real = None
    self.status = None
    self.status_t = 0.0
    self.holding = True

    control = MutuallyExclusiveCallbackGroup()
    other = ReentrantCallbackGroup()
    self.pub_states = self.create_publisher(JointState, f'{self.ns}/joint_states', 10)
    self.pub_measured = self.create_publisher(JointState, f'{self.ns}/measured_joint_states', 10)
    self.pub_status = self.create_publisher(String, f'{self.ns}/status', 10)
    self.create_subscription(JointState, f'{self.ns}/joint_commands', self.on_velocity, 10, callback_group=other)
    self.create_service(SetBool, f'{self.ns}/enable', self.on_enable, callback_group=other)
    if self.spec.gripper:
      ActionServer(self, ParallelGripperCommand, f'{self.ns}/gripper_cmd', self.on_grip, callback_group=other)
    self.driver.connect()
    self.get_logger().info(f'{self.spec.name} ({a.driver}) on {self.ns}/…')
    self.create_timer(1.0 / a.rate, self.tick, callback_group=control)

  # ---- in ----

  def on_velocity(self, msg):
    by = dict(zip(msg.name, msg.velocity))
    if not all(j in by for j in self.spec.joints):
      return
    with self.lock:
      self.vel = [float(by[j]) for j in self.spec.joints]
      self.vel_t = time.monotonic()

  def on_enable(self, req, res):
    with self.lock:
      self.requests.append(bool(req.data))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0 and self.requests:  # handled by the next tick
      time.sleep(0.01)
    res.success = (self.safety.state == ENABLED) == bool(req.data)
    res.message = self.safety.reason or self.safety.state
    return res

  def on_grip(self, handle):
    g = self.spec.gripper
    cmd = handle.request.command
    pos = dict(zip(cmd.name, cmd.position)).get(g.joint, cmd.position[0] if cmd.position else g.open)
    self.grip_goal = g.fraction(pos)
    want = g.value(round(self.grip_goal) if g.binary else self.grip_goal)
    t0 = last_move = time.monotonic()
    last = None
    result = ParallelGripperCommand.Result()
    while True:
      time.sleep(0.05)
      now, at = time.monotonic(), self.real.gripper if self.real else None
      if at is None:
        if now - t0 > GRIP_TIMEOUT_S:
          break
        continue
      if last is None or abs(at - last) > GRIP_TOLERANCE / 4:
        last, last_move = at, now
      if abs(at - want) <= GRIP_TOLERANCE:
        result.reached_goal = True
        break
      if now - last_move > GRIP_STALL_S and now - t0 > GRIP_STALL_S * 2:
        result.stalled = True  # (holding something, as a grasp should)
        break
      if now - t0 > GRIP_TIMEOUT_S:
        break
    at = self.real.gripper if self.real and self.real.gripper is not None else float('nan')
    result.state = JointState(name=[g.joint], position=[at])
    handle.succeed()
    return result

  # ---- the control tick ----

  def tick(self):
    now, dt = time.monotonic(), 1.0 / self.a.rate
    real = self.real = self.driver.read()
    if not real.finite():
      return  # (the lobby's first states not in yet)
    with self.lock:
      vel, age = self.vel, now - self.vel_t
      requests, self.requests = self.requests, []
    for want in requests:
      if want:
        if real.fault:
          self.driver.recover()
        self.safety.disable()
        why = self.safety.enable(real.q, 0.0, real.q)
        self.get_logger().info(f'enable: {why or "enabled"}')
      else:
        self.safety.disable('disabled (service)')
        self.get_logger().info('disabled')
    # The next command: the last one moved by the commanded velocities.
    base = self.safety.cmd if self.safety.cmd is not None else real.q
    target = [c + v * dt for c, v in zip(base, vel)] if vel is not None and age <= self.a.timeout else None
    step = self.safety.step(target, 0.0 if target else age, real.q, dt, real.fault)
    if step.q is not None:
      self.driver.command(step.q)
      self.holding = False
    elif not self.holding:
      self.driver.hold()
      self.holding = True
    if self.grip_goal is not None and self.spec.gripper and self.safety.state == ENABLED:
      self.driver.command_gripper(self.grip_goal)
    self.publish(real, step.q if step.q is not None else (self.safety.cmd or real.q))
    status = {'robot': self.spec.name, 'driver': self.a.driver, 'state': self.safety.state, 'reason': self.safety.reason,
              'stale': step.stale, 'fault': real.fault}
    if status != self.status or now - self.status_t > 0.5:
      self.pub_status.publish(String(data=json.dumps(status)))
      self.status, self.status_t = status, now

  def publish(self, real, commanded):
    stamp = self.get_clock().now().to_msg()
    g = self.spec.gripper
    extra = [g.joint] if g and real.gripper is not None else []
    grip = [real.gripper] if extra else []
    vel = list(self.vel) if self.vel is not None and self.safety.state == ENABLED and not self.holding else [0.0] * len(self.spec.joints)
    s = JointState(name=list(self.spec.joints) + extra, position=[float(v) for v in commanded] + grip, velocity=vel + [0.0] * len(extra))
    s.header.stamp = stamp
    self.pub_states.publish(s)
    m = JointState(name=list(self.spec.joints) + extra, position=[float(v) for v in real.q] + grip)
    m.header.stamp = stamp
    self.pub_measured.publish(m)


def main():
  p = argparse.ArgumentParser(prog='python3 -m real_robot.ros2', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--robot', default='panda', choices=sorted(drivers.ROBOTS))
  p.add_argument('--driver', default='real', choices=['real', 'fake', 'lobby'])
  p.add_argument('--ns', help='topic namespace (default /<robot>/hw)')
  p.add_argument('--enable', action='store_true', help='enable at start (else the enable service)')
  p.add_argument('--rate', type=float, default=100.0, help='control loop, Hz')
  p.add_argument('--speed', type=float, default=0.5, help='fraction of the joints\' max speeds commanded at most')
  p.add_argument('--timeout', type=float, default=0.3, help='s without velocity commands before the robot holds')
  p.add_argument('--ip', default='172.16.0.2', help='Panda: the robot\'s address (FCI)')
  p.add_argument('--dynamics', type=float, default=0.15, help='Panda: franky\'s relative_dynamics_factor')
  p.add_argument('--grasp-force', type=float, default=20.0, help='Panda: N')
  p.add_argument('--serial', default='/dev/ttyACM0', help='SO-101: the servo bus')
  p.add_argument('--calibration', help='SO-101: calibration file')
  p.add_argument('--lag', type=float, default=0.0, help='fake: s its states lag behind')
  p.add_argument('--lobby-states', default='', help='lobby: its joint states (default /<robot>/joint_states)')
  p.add_argument('--lobby-commands', default='', help='lobby: its joint commands (default /<robot>/joint_command)')
  a = p.parse_args(rclpy.utilities.remove_ros_args()[1:])
  rclpy.init()
  node = Ros2Front(a)
  ex = MultiThreadedExecutor(num_threads=4)
  ex.add_node(node)
  try:
    ex.spin()
  except (KeyboardInterrupt, ExternalShutdownException):
    pass
  finally:
    try:
      node.driver.hold()
      node.driver.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()
