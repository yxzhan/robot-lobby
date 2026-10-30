#!/usr/bin/env python3
"""The real GARMI's ROS interface on top of the lobby's ROS bridge, so that
giskardpy's GarmiVelocityInterface (garmi_velocity.py, closed loop) and
coraplex's RealGarmiPlanExecutor drive a GARMI in the lobby as they drive the
robot: scripts/garmi_lobby_demo.py runs the apartment demo against it.

    lobby (rosbridge, prefix r1)        this node                      giskard / coraplex
    r1/joint_states        ──►  rollers left out ──►  /garmi/joint_states
    /tf  world → r1/base_link ──►  planar, from the start ──►  /r100_0603/platform/odom/filtered
                                    map → odom (static: where the plan believes GARMI starts)
    r1/joint_command       ◄──  velocities integrated ◄──  /garmi/arms/{left,right}_arm_joint_velocity_controller/reference
                                 finger openings     ◄──  /garmi/arms/{left,right}/franka_gripper/{move,grasp,homing}
    r1/cmd_vel             ◄──  Twist             ◄──  /r100_0603/joy_teleop/cmd_vel (TwistStamped)

The arms' velocities become position targets (the lobby's arms are position
servos): integrated from where the arm is when a command stream starts, never
more than MAX_LEAD ahead of it, held when the commands stop. The hands are
franka_gripper's actions (franka_msgs, from franka_ros2), done when the
fingers stop. Lift and head are not commanded, as on the real robot.

In the lobby: spawn GARMI alone, switch its IK off in the Teleop panel (its
arms are the IK's by default; Take over would hand them back to the IK
whenever the commands pause) and its ROS on in the ROS panel (the bridge's
address); the profile (public/json/ros_profiles.json) maps the finger joints
onto the hands' actuators. Then, with ROS Jazzy sourced (and franka_msgs on
the path):

    ros2 run rosbridge_server rosbridge_websocket          # port 9090
    python3 scripts/garmi_lobby_bridge.py [--lobby r1] [--start -4.8 1.5 3.1416]
    python -m giskardpy.middleware.ros2.scripts.other_robots.garmi.garmi_velocity   # CRAM venv
    python scripts/garmi_lobby_demo.py                                               # CRAM venv
"""

import argparse
import math
import re
import time

import rclpy
from control_msgs.msg import MultiDOFCommand
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster

try:
  from franka_msgs.action import Grasp, Homing, Move
except ImportError:  # no hands then, the arms and the base still work
  Grasp = Homing = Move = None

SIDES = ('left', 'right')
ARM = {s: [f'{s}_fr3_joint{i}' for i in range(1, 8)] for s in SIDES}  # GarmiJoint.left_arm() order
FINGER = {s: f'{s}_fr3_finger_joint1' for s in SIDES}
ROLLER = re.compile(r'_r\d+_j$')  # the lobby's mecanum rollers: not in GARMI's URDF

JOINT_STATES = '/garmi/joint_states'
ARM_COMMAND = '/garmi/arms/{}_arm_joint_velocity_controller/reference'
GRIPPER = '/garmi/arms/{}/franka_gripper'
ODOM = '/r100_0603/platform/odom/filtered'
CMD_VEL = '/r100_0603/joy_teleop/cmd_vel'

RATE = 50          # Hz: joint commands to the lobby
STALE = 0.2        # s without a velocity command: the arm holds
MAX_LEAD = 0.3     # rad: a target at most this far from where the joint is
FINGER_MAX = 0.04  # m: one finger's stroke
BASE_HEIGHT = 0.0259  # m: base_link over the floor, in map -> odom (as cram_vrb_lab's localization stand-in)


class GarmiLobbyBridge(Node):
  def __init__(self, lobby, start):
    super().__init__('garmi_lobby_bridge')
    self.base_frame = f'{lobby}/base_link'
    self.q = {}            # measured joint positions
    self.target = {}       # position targets sent to the lobby
    self.vel = {s: None for s in SIDES}  # (values, time) of the last arm command
    self.odom_start = None
    self.states_at = 0.0   # when the lobby's joint states last came
    group = ReentrantCallbackGroup()

    self.states_out = self.create_publisher(JointState, JOINT_STATES, 10)
    self.odom_out = self.create_publisher(Odometry, ODOM, 10)
    self.command_out = self.create_publisher(JointState, f'{lobby}/joint_command', 10)
    self.twist_out = self.create_publisher(Twist, f'{lobby}/cmd_vel', 10)
    self.create_subscription(JointState, f'{lobby}/joint_states', self.on_states, 10, callback_group=group)
    self.create_subscription(TFMessage, '/tf', self.on_tf, 10, callback_group=group)
    self.create_subscription(TwistStamped, CMD_VEL, lambda m: self.twist_out.publish(m.twist), 10, callback_group=group)
    for s in SIDES:
      self.create_subscription(MultiDOFCommand, ARM_COMMAND.format(s), lambda m, s=s: self.on_arm(s, m), 10, callback_group=group)
      if Move is not None:
        ns = GRIPPER.format(s)
        ActionServer(self, Move, f'{ns}/move', lambda g, s=s: self.move(s, g), callback_group=group)
        ActionServer(self, Grasp, f'{ns}/grasp', lambda g, s=s: self.grasp(s, g), callback_group=group)
        ActionServer(self, Homing, f'{ns}/homing', lambda g, s=s: self.homing(s, g), callback_group=group)
    if Move is None:
      self.get_logger().warn('no franka_msgs: the hands have no franka_gripper actions')

    # The localization: map -> odom where the plan's world has GARMI start.
    x, y, yaw = start
    tf = TransformStamped()
    tf.header.stamp = self.get_clock().now().to_msg()
    tf.header.frame_id, tf.child_frame_id = 'map', 'odom'
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, BASE_HEIGHT
    tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    self.static_tf = StaticTransformBroadcaster(self)
    self.static_tf.sendTransform(tf)

    self.last_tick = time.monotonic()
    self.create_timer(1 / RATE, self.tick, callback_group=group)
    self.get_logger().info(f'lobby robot {lobby}; GARMI starts at map ({x:.2f}, {y:.2f}, yaw {yaw:.2f})')

  # -- the robot's state ---------------------------------------------------

  def on_states(self, msg):
    keep = [i for i, n in enumerate(msg.name) if not ROLLER.search(n)]
    pick = lambda a: [a[i] for i in keep] if len(a) == len(msg.name) else []
    out = JointState(header=msg.header, name=[msg.name[i] for i in keep],
                     position=pick(msg.position), velocity=pick(msg.velocity), effort=pick(msg.effort))
    self.states_out.publish(out)
    now = time.monotonic()
    first = not self.q or now - self.states_at > 1.0  # also after a gap: the page may hold a new robot
    self.states_at = now
    self.q.update(zip(msg.name, msg.position))
    if first:
      self.odom_start = None
      self.target ={n: self.q[n] for s in SIDES for n in ARM[s] + [FINGER[s]] if n in self.q}
      self.get_logger().info(f'joint states from the lobby: {len(out.name)} joints')

  def on_tf(self, msg):
    for t in msg.transforms:
      if t.child_frame_id != self.base_frame:
        continue
      p, r = t.transform.translation, t.transform.rotation
      yaw = math.atan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y * r.y + r.z * r.z))
      if self.odom_start is None:
        self.odom_start = (p.x, p.y, yaw)
      x0, y0, a0 = self.odom_start
      c, s = math.cos(a0), math.sin(a0)
      dx, dy, da = p.x - x0, p.y - y0, yaw - a0
      odom = Odometry()
      odom.header.stamp = t.header.stamp
      odom.header.frame_id, odom.child_frame_id = 'odom', 'base_link'
      odom.pose.pose.position.x, odom.pose.pose.position.y = c * dx + s * dy, -s * dx + c * dy
      odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(da / 2), math.cos(da / 2)
      self.odom_out.publish(odom)

  # -- the arms: velocities into position targets ----------------------------

  def on_arm(self, side, msg):
    names = list(msg.dof_names) or ARM[side]
    fresh = self.vel[side] is not None and time.monotonic() - self.vel[side][1] < STALE
    if not fresh:  # a new stream starts from where the arm is
      self.target.update({n: self.q[n] for n in ARM[side] if n in self.q})
    self.vel[side] = (dict(zip(names, msg.values)), time.monotonic())

  def tick(self):
    now = time.monotonic()
    dt, self.last_tick = now - self.last_tick, now
    if not self.target:
      return
    for s in SIDES:
      if self.vel[s] is None or now - self.vel[s][1] > STALE:
        continue
      for n, v in self.vel[s][0].items():
        if n in self.target and n in self.q:
          t = self.target[n] + v * dt
          self.target[n] = min(max(t, self.q[n] - MAX_LEAD), self.q[n] + MAX_LEAD)
    msg = JointState(name=list(self.target), position=list(self.target.values()))
    msg.header.stamp = self.get_clock().now().to_msg()
    self.command_out.publish(msg)

  # -- the hands: franka_gripper's actions -----------------------------------

  def width(self, side):
    return 2 * self.q.get(FINGER[side], 0.0)

  def close_to(self, side, width, timeout=5.0):
    """The fingers sent to `width` (m between them); their width once they stopped."""
    self.target[FINGER[side]] = min(max(width / 2, 0.0), FINGER_MAX)
    t0, last, still = time.monotonic(), None, None
    while time.monotonic() - t0 < timeout:
      time.sleep(0.05)
      w, now = self.width(side), time.monotonic()
      if last is not None and abs(w - last) < 2e-4:
        still = still or now
        if now - still > 0.3 and now - t0 > 0.5:
          break
      else:
        still = None
      last = w
    return self.width(side)

  def finish(self, handle, result, ok, error=''):
    result.success, result.error = ok, error
    handle.succeed() if ok else handle.abort()
    return result

  def move(self, side, handle):
    goal = handle.request
    w = self.close_to(side, goal.width)
    ok = abs(w - goal.width) < 0.005
    self.get_logger().info(f'{side} hand move to {goal.width:.3f} m: {w:.3f} m')
    return self.finish(handle, Move.Result(), ok, '' if ok else f'stopped at {w:.3f} m')

  def grasp(self, side, handle):
    goal = handle.request
    w = self.close_to(side, goal.width)
    ok = goal.width - goal.epsilon.inner <= w <= goal.width + goal.epsilon.outer
    self.get_logger().info(f'{side} hand grasp at {goal.width:.3f} m: {w:.3f} m')
    return self.finish(handle, Grasp.Result(), ok, '' if ok else f'stopped at {w:.3f} m')

  def homing(self, side, handle):
    self.close_to(side, 2 * FINGER_MAX)
    return self.finish(handle, Homing.Result(), True)


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--lobby', default='r1', help="the lobby robot's ROS prefix (default r1)")
  parser.add_argument('--start', type=float, nargs=3, default=(-4.8, 1.5, math.pi), metavar=('X', 'Y', 'YAW'),
                      help="where GARMI starts in the plan's map (default the apartment demo's GARMI_START)")
  args, _ = parser.parse_known_args()
  rclpy.init()
  node = GarmiLobbyBridge(args.lobby, args.start)
  executor = MultiThreadedExecutor()
  executor.add_node(node)
  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
  main()
