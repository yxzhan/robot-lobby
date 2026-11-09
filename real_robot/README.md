# Real robots following the lobby

The lobby's robot is the target, and a real robot follows it. You teleoperate
in the lobby (mouse, gamepad, or a Meta Quest 3 in WebXR: grab the IK marker,
use the trigger for the gripper). The real robot shows up as a see-through
**ghost** on the lobby robot: blue while disabled, green while it follows,
red in a fault.

```
Quest 3 / desktop browser                     the robot's computer
lobby (physics + IK in the page)   Wi-Fi     python -m real_robot
  panda/joint_states  ────────────────────►  safety (limits, speed, timeouts)
  panda/real/enable   ────────────────────►      │
  panda/real/joint_states ◄───────────────       ▼
  panda/real/status       ◄───────────────  driver ──► Panda (franky / libfranka 0.9.x)
                                                    └─► SO-101 (Feetech bus, to do)
```

`python -m real_robot` is a small rosbridge-compatible WebSocket server
(`rosbridge.py`), so no ROS is needed. The lobby connects to it the way it
connects to `rosbridge_server`.

## Run

On the robot's computer, from the repo root:

```bash
python -m real_robot --robot panda --ip 172.16.0.2        # the Panda's FCI address
python -m real_robot --robot panda --driver fake          # no hardware: a simulated stand-in
```

In the lobby (the Quest's browser opening `http://<lobby host>:<port>/?local`
works, because the physics runs in the page):

1. Spawn the Franka Emika Panda. Its ROS profile gives it the prefix `panda`
   (topics `panda/…`), which the bridge uses by default for `--robot panda`,
   whatever order things are spawned in. For a second Panda, change its
   prefix in the ROS panel and pass the same value with `--prefix`.
2. ROS panel: set **Bridge** to `<robot computer>:9090` and tick the robot.
3. Under **Real robot**, press **Enable**. This first syncs the lobby robot
   to where the real one is, then the real robot follows. **Disable**
   makes it hold where it is.
   In VR: hold Y to open the menu, then go to **Real robot**, which offers ROS,
   Enable, Disable and Ghost.

The bridge refuses to enable when the lobby robot is more than
`start_tolerance` from the real one; **Sync** fixes that. When the targets
stop (the headset sleeps, Wi-Fi drops) for `--timeout` (0.3 s), the robot
holds. It goes to **fault** and holds when it falls `max_lead` (0.5 rad)
behind its command (blocked, a collision) or when the robot reports an
error. **Enable** recovers from a fault. When the last lobby disconnects,
the robot is disabled.

### In a shared world (server mode): VR drives, a PC follows

With the physics on the lobby server (`npm run server`, every browser a client of one world),
the robot and its real one are two jobs for two clients:

- **The PC** (next to the robot, a browser on `http://<server>:<port>/`): in the ROS panel,
  **Bridge** `<robot computer>:9090` and the robot ticked. Its ROS is this browser's then
  (one browser publishes a robot: the first to switch it on), so its joint states, the real
  robot's targets, go out from here. **Enable** syncs the server's robot to the real one
  (op `real_sync`: its joints, actuator targets, IK targets on the tips) and enables once the
  synced pose has come back (within 2 cm / rad).
- **The Quest** (the same address, no `?local`): drives the robot (IK markers, hands) as any
  client does; it needs no bridge, and does not care whether a real robot follows.

Sync is allowed from the browser publishing the robot's ROS or the robot's holder, so the PC
syncs while the Quest holds it.

**Keep the physical E-stop at hand.** The software limits do not replace it.

Step by step on the real Panda (both the lobby teleop and the giskard demo, Docker): [TESTING_zh.md](TESTING_zh.md).

## Options

| option | default | |
|---|---|---|
| `--speed` | 0.5 | fraction of each joint's max speed the bridge commands at most |
| `--rate` | 50 | control loop, Hz |
| `--timeout` | 0.3 | s without targets before holding |
| `--dynamics` | 0.15 | Panda: franky's `relative_dynamics_factor` (Ruckig's vel/acc/jerk scale) |
| `--grasp-force` | 20 | Panda: N |
| `--port` / `--host` | 9090 / 0.0.0.0 | the WebSocket |

## Panda setup (System 4.2.x → libfranka 0.9.x)

Panda System 4.2.x speaks libfranka **0.9.x**. libfranka ≥ 0.10 and
franka_ros2 are for the FR3 only. franky publishes wheels per libfranka
version (PyPI's are for newer libfranka, the FR3): install a libfranka 0.9.2 one from
<https://github.com/TimSchneider42/franky/releases>, e.g. v2.0.0's
`franky_control-2.0.0+libfranka.0.9.2-cp312-cp312-manylinux_2_34_x86_64.whl` (your
python's `cpXY`; `manylinux_2_28` for glibc < 2.34). The driver works with franky 1.1.x and 2.x.
Use a PREEMPT_RT kernel, as for any FCI program. In Desk, unlock the brakes
and activate FCI.

The bridge needs `websockets` (the repo's `.venv` has it). Only the Panda
driver needs franky.

## As a package

`real_robot/` is a pip package of its own (`pyproject.toml`; `lobby-real-robot`, the
module `real_robot`), for a computer or an image without the rest of the lobby:

```bash
pip install "git+https://github.com/yxzhan/robot-lobby@real-robot-bridge#subdirectory=real_robot"
pip install -e ./real_robot        # from this checkout, editable
```

It brings `websockets`; not rclpy (`real_robot.ros2`: the ROS python, `/usr/bin/python3
-m pip install --user --break-system-packages ...`) nor franky (above). ijcai_2026_workshop's
Panda demo uses it that way. From this repository's root, `python -m real_robot` works
without installing.

## ROS 2 (ros2_control, giskardpy)

```bash
python3 -m real_robot.ros2 --robot panda --driver real --ip 172.16.0.2   # with ROS 2 sourced
```

The same drivers and safety layer behind ros2_control's `topic_based_ros2_control`: the arm
velocity commanded (a `forward_velocity_controller`, e.g. giskardpy's
`universal_robot_velocity`), the velocities integrated into position commands each tick,
the commanded positions reported back (`<ns>/joint_states`, ns `/panda/hw`), the measured ones
on `<ns>/measured_joint_states`, the gripper an action (`<ns>/gripper_cmd`,
`control_msgs/ParallelGripperCommand`), enabled by `<ns>/enable` (`std_srvs/SetBool`) or
`--enable`. `--driver lobby` drives the lobby's robot over rosbridge instead (its
`/panda/joint_states`, `/panda/joint_command`): a digital twin to try a program on first.
See `ros2.py`; ijcai_2026_workshop's `demo/panda_giskard/` runs giskardpy on it.

Run it with the ROS python (`/usr/bin/python3`, rclpy); for `--driver real`, franky installed
there.

## Adding a robot (SO-101 next)

1. `drivers/<robot>.py`: a `RobotSpec` (the arm's joints as the lobby
   model names them, limits, max speeds, the gripper's lobby joint and its
   open/closed values) and a `RobotDriver` subclass (`connect`, `read`,
   `command`, `hold`, optionally `command_gripper`, `recover`, `close`).
   See `drivers/base.py`.
2. One line in `drivers/__init__.py`'s `ROBOTS`.
3. The lobby: a `"real": {"robot": "<name>"}` entry in
   `public/json/ros_profiles.json` under the model's file, which shows the
   Real robot section and the ghost.

`drivers/so101.py` has the spec (matching `robotstudio_so101/so101.xml`),
the steps (Feetech STS3215 sync read/write, calibration) and a driver
stub. Its lobby profile is already in place. The lobby has no IK config for
the SO-101 yet (`public/json/ik_defaults.json`), so add one before using it
in VR.

## Tests

```bash
.venv/bin/python -m real_robot.tests.test_bridge
```

The test runs the safety layer and a WebSocket client in the lobby's place
against the fake driver: it enables, follows, closes the gripper, holds when
the targets stop, and disables.
