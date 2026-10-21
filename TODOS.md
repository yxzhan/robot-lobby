# TODOs

## ROS: drive robots like the real hardware

Goal: high-level programs (MoveIt, trajectory/gripper action clients) work unchanged
between the real robot (e.g. Franka FR3 on franka_ros2) and the lobby.

Plan: keep ros2_control on the ROS side and swap only the hardware plugin for
[topic_based_ros2_control](https://github.com/PickNikRobotics/topic_based_ros2_control)
(as Isaac Sim does), with the lobby as the simulator behind its topics.

- Per-robot ROS profile (like `ik_defaults.json`): joint and frame names mapped to the
  real robot's URDF (`joint1` → `fr3_joint1`), topic names.
- Option to stop publishing body TF (robot_state_publisher computes it from the URDF).
- `joint_states` with velocity (server mode too) and effort (`actuator_force`).
- Optional `/clock` from simulation time (`use_sim_time`).
- Gripper as a position joint behind `parallel_gripper_action_controller`
  (`control_msgs/GripperCommand`); franka_msgs Grasp/Move need a small shim if used.
- Camera topic and frame names like the real drivers (e.g. RealSense).
- Example launch + controller yaml for FR3 in `server/`.
- Later: server mode talks to ROS directly (rclpy in `server/server.py`) instead of
  browser → rosbridge, for higher rate and lower latency.

Limits: rosbridge JSON gives tens to ~200 Hz; no 1 kHz torque control.
Alternative without the lobby: [mujoco_ros2_control](https://control.ros.org/jazzy/doc/mujoco_ros2_control/doc/index.html)
(MuJoCo inside the ROS process).
