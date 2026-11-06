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

**Keep the physical E-stop at hand.** The software limits do not replace it.

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
version: install the libfranka 0.9.2 ones from
<https://github.com/TimSchneider42/franky/releases> (`libfranka_0-9-2_wheels.zip`),
for example `pip install franky_control-*-cp3XX-*.whl`.
Use a PREEMPT_RT kernel, as for any FCI program. In Desk, unlock the brakes
and activate FCI.

The bridge needs `websockets` (the repo's `.venv` has it). Only the Panda
driver needs franky.

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
