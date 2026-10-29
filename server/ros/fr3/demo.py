"""Drives the lobby's FR3 through the controllers fr3_lobby.launch.py runs,
as a program for the real robot would (no lobby in it): the arm along a
FollowJointTrajectory goal (home, a reach forward and down, home again), the
hand closed and opened with ParallelGripperCommand; prints what the
controllers report and the joint states at the end of each.

    source /opt/ros/jazzy/setup.bash
    python3 server/ros/fr3/demo.py
"""

import sys

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM = [f'fr3_joint{i}' for i in range(1, 8)]
HOME = [0, 0, 0, -1.5708, 0, 1.5708, -0.7854]
REACH = [0.4, 0.5, 0, -1.8, 0, 2.3, 0.4]


class Demo(Node):
  def __init__(self):
    super().__init__('lobby_fr3_demo')
    self.arm = ActionClient(self, FollowJointTrajectory, '/fr3_arm_controller/follow_joint_trajectory')
    self.hand = ActionClient(self, ParallelGripperCommand, '/fr3_gripper/gripper_cmd')
    self.state = {}
    self.create_subscription(JointState, '/joint_states', lambda m: self.state.update(zip(m.name, m.position)), 10)

  def wait(self, future, timeout=30.0):
    rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
    if not future.done():
      sys.exit('no answer (is the lobby\'s FR3 connected to the bridge?)')
    return future.result()

  def send(self, client, goal, what):
    if not client.wait_for_server(timeout_sec=30.0):
      sys.exit(f'no {client._action_name} (is fr3_lobby.launch.py running?)')
    handle = self.wait(client.send_goal_async(goal))
    if not handle.accepted:
      sys.exit(f'{what}: rejected')
    result = self.wait(handle.get_result_async(), 60.0)
    rclpy.spin_once(self, timeout_sec=0.2)
    return result

  def move(self, positions, seconds, what):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ARM
    goal.trajectory.points = [JointTrajectoryPoint(positions=[float(p) for p in positions],
                                                   time_from_start=Duration(sec=int(seconds), nanosec=int(seconds % 1 * 1e9)))]
    r = self.send(self.arm, goal, what)
    err = max(abs(self.state.get(j, float('nan')) - p) for j, p in zip(ARM, positions))
    print(f'{what}: error code {r.result.error_code} ({r.result.error_string or "ok"}), joints off by {err:.4f} rad at most')

  def grip(self, width, what):
    goal = ParallelGripperCommand.Goal()
    goal.command.name = ['fr3_finger_joint1']
    goal.command.position = [float(width)]
    r = self.send(self.hand, goal, what)
    print(f'{what}: reached {r.result.reached_goal}, stalled {r.result.stalled}, '
          f'finger at {r.result.state.position[0] if r.result.state.position else float("nan"):.4f} m')


def main():
  rclpy.init()
  d = Demo()
  d.move(HOME, 3, 'home')
  d.move(REACH, 4, 'reach forward and down')
  d.grip(0.0, 'close the hand')
  d.grip(0.04, 'open the hand')
  d.move(HOME, 4, 'home again')
  d.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
