"""The lobby's FR3 with its hand under ros2_control, as franka_ros2 runs the
real one: robot_state_publisher (TF from fr3.urdf), the controller manager
with topic_based_ros2_control as the hardware (the lobby's ROS bridge on its
topics), the joint state broadcaster, a joint trajectory controller for the
arm (FollowJointTrajectory on /fr3_arm_controller/follow_joint_trajectory)
and a parallel gripper controller for the hand (ParallelGripperCommand on
/fr3_gripper/gripper_cmd), and rosbridge for the browser.

    source /opt/ros/jazzy/setup.bash
    ros2 launch server/ros/fr3/fr3_lobby.launch.py      # rosbridge on :9090

Then in the lobby: spawn Franka FR3 with hand, untick its IK, tick ROS
(its profile sets the topics; Bridge localhost:9090). server/ros/fr3/demo.py
moves it.
"""

import pathlib

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = pathlib.Path(__file__).resolve().parent


def generate_launch_description():
  urdf = (HERE / 'fr3.urdf').read_text()
  description = {'robot_description': urdf}
  sim = {'use_sim_time': LaunchConfiguration('use_sim_time')}
  spawner = lambda name: Node(package='controller_manager', executable='spawner', arguments=[name, '--controller-manager-timeout', '60'],
                              parameters=[sim], output='screen')
  return LaunchDescription([
      DeclareLaunchArgument('use_sim_time', default_value='false',
                            description='true: the lobby\'s /clock (tick Clock and Stamps: simulated time in its ROS section)'),
      DeclareLaunchArgument('port', default_value='9090', description='rosbridge\'s WebSocket port'),
      Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[description, sim], output='screen'),
      Node(package='controller_manager', executable='ros2_control_node', parameters=[description, str(HERE / 'controllers.yaml'), sim],
           output='screen'),
      spawner('joint_state_broadcaster'),
      spawner('fr3_arm_controller'),
      spawner('fr3_gripper'),
      Node(package='rosbridge_server', executable='rosbridge_websocket', parameters=[{'port': LaunchConfiguration('port')}, sim],
           output='screen'),
  ])
