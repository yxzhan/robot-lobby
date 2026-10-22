# ROS bridge

Anything spawned (robots, and environments and objects too: the Teleop
panel lists them with no controllers) can have a rosbridge connection of its own
(`public/js/ros.js`, rosbridge_suite's WebSocket protocol, no roslibjs),
switched on with the **ROS** chip in the [Teleop panel](teleop.md) and made from
this browser, in local and network mode alike. Its section in the panel:
the bridge's address (`localhost:9090` by default, or a `ws://` / `wss://`
URL) with the connection's state (retried every 2 s while it is down);
what it publishes, each with its topic: the joint states
(`sensor_msgs/JointState` on `joint_states`: the robot's hinge and slide
joints, names without the `r1/` prefix; velocities only in local mode, the
server sends positions) and TF (`tf2_msgs/TFMessage` on `/tf`: every body
in its parent's frame, the root bodies in `world`); what it takes: joint
commands (`sensor_msgs/JointState` on `joint_command`: position targets
for the actuators on those joints; a running policy or IK keeps writing
the joints it drives, the panel says so) and `cmd_vel`
(`geometry_msgs/Twist` on `cmd_vel`: linear x / y and angular z drive the
walking policy or the base like the keys do, before a goal; one older than
0.5 s counts as zero); the publishing rate (1–60 Hz, 30 by default); and
the cameras. Each camera of the robot ticked in the Cameras panel is
published every frame it renders (its own rate): `<camera>/image_raw/compressed`
(`sensor_msgs/CompressedImage`, JPEG, the default) or
`<camera>/image_raw` (`sensor_msgs/Image`, `rgb8`), and
`<camera>/camera_info` (`sensor_msgs/CameraInfo`: a pinhole of
its vertical field of view, fx = fy = (h/2)/tan(fovy/2), centered, no
distortion). With TF on, every camera
of the robot also has `<camera>_optical_frame` (z forward, x right, y down)
in its link's frame. Frames are dropped while the WebSocket is more than
8 MB behind (counted in the panel). One **prefix** per robot, the robot's
id by default (`r1`, `r2`, or typed in; `/` for none), goes before every
name, so robots of one model on one bridge do not overwrite each other:
the topics not starting with `/` (`r1/joint_states`, `r1/joint_command`,
`r1/cmd_vel`, `r1/wrist_cam/image_raw/compressed`; `/tf` stays shared) and
the TF frames (`r1/base_link`, `r1/wrist_cam_optical_frame`; the roots
still in `world`). Joint names in the messages stay plain. For one robot
with robot_state_publisher, set the prefix to `/` or use its
`frame_prefix`.
Stamps are wall time in ROS 2's form (`sec`, `nanosec`). Settings are kept
per robot file in this browser (the prefix per robot). A page on HTTPS (GitHub Pages, Binder) may
still connect to `ws://localhost`; browsers treat localhost as secure.
