# Policies

Pretrained policies live in `pretrained_policies/<dir>/`, one checkout per
model (not tracked by git). `npm install` (or `node scripts/vendor.mjs`)
links the folder as `public/pretrained_policies` and lists what is there in
`public/json/policies.json`; a registered policy whose directory is missing is
shown as "not installed" and never switched on by default. Policies are
registered in `POLICIES` (`public/js/policy.js`): each entry has an id, a
name, its `dir`, the robot files it starts on for (`defaultFor`),
`load(file)` for its weights (`file(name)` is the URL of a file in its
directory; shared, loaded on first use; onnxruntime-web is there as
`onnxRuntime()`) and `create(weights, prefix)` for a controller with
`bind` / `unbind` / `owns` / `pose` / `reset` / async `step`, plus `cmd` /
`drive` for a locomotion policy driven by the keys, and an optional
`substep(data)` run before every physics step (actuator models).
An entry's optional `physics` names Physics settings it needs (e.g.
`{cone: 'elliptic', impratio: 100}`): the Policy section shows them when
the world differs, with a button that applies them. The panel's Policy
section picks the model; a hand or quadruped policy is one more entry.
## Commands

What a policy takes besides the robot's state and the drive twist (a head
pose, a base height, a gait, …) it declares as **commands**
(`public/js/commands.js`): `commands` on the controller, `{name: definition}`,
with the current values in `command`. A policy that reads `policy.json`
(Playground, Microduck) takes them from its `"commands"`, and its `obs` list
puts one in the observation as the part `cmd:<name>`; GR00T declares its
`height` in code. A definition has a `type`, `label`, `about` and `default`,
and per type:

| type | value | in the panel | in the observation |
|---|---|---|---|
| `scalar` | a number within `range` [lo, hi] | a slider | the number |
| `vector` | numbers, one per `dims` name, each within its `range` | a slider per dimension | the numbers |
| `enum` | one of `values` | a select | one-hot over `values` |
| `bool` | true or false | a checkbox | 0 or 1 |
| anything else | as given | a JSON text field | (cannot be observed) |

`unit` is shown after a value (a vector can have one per dimension). The
Teleop panel's Policy section makes the controls (double-click a slider for
its default; **Defaults** resets them all), so a new policy needs no panel
code. Values are clamped to their ranges wherever they are set. A command
named `height` also follows **R**/**F** (and the gamepad's height actions).
In network mode a change is a `command {id, name, value}` request; the
server keeps the values (they outlast the keys' drive command) and sends
them with the status, so every browser shows them.

Microduck's are `head` (the four head joints' offsets from the default
pose, up to each joint's reach: the walking task trains it to hold them)
and `body` (a body pose offset trained only within ±5 mm and ±0.05 rad,
which the walking task barely follows); GR00T's is `height` (0.5-1 m).

## Skills: policies of more than one model

A policy can run one of several models of the same observation and actions,
its **skills**, listed in its directory's `policy.json` (GR00T,
Microduck). A skill is the model, a level above the commands, which are
inputs of the model running. One shared implementation picks the model every
control tick, `public/js/skills.js` in the browser and `Skills` in
`server/policies.py`:

```json
"skills": {
  "walk":       {"onnx": "walk.onnx", "label": "Walk"},
  "balance":    {"onnx": "balance.onnx", "when": {"command_below": 0.05}},
  "kick_right": {"onnx": "kick_right.onnx", "label": "Kick", "duration": 0.3, "key": "k", "commands": false}
}
```

- The first is the default.
- One with `when` is **automatic**: it runs instead of the default while its
  condition holds (the first that does). Conditions: `command_below: v`,
  the drive twist's norm |(vx, vy, yaw rate)| under v.
- The others (the default too) are **manual**: with two or more, the
  controller has a `skillMenu` (each skill's `label`, `about`, `duration`,
  `key`), the one picked (`skill`) and `setSkill(name)`; in network mode a
  `skill {id, name}` request, and the status brings the pick and the model
  running back to every browser. The Policy section's Skills, above its
  Commands, has a button for each one with a `duration` (lit while it runs;
  pressed again, the time starts over) and beside it how long it runs (empty:
  its `duration`; 0.1-60 s, kept in the browser per policy and sent with the
  pick, `skill {id, name, duration}`), a Skill select of the others, and a
  Keys row that binds a key to each,
  `key` (policy.json) to start with: click a skill's key, press another
  (<kbd>Esc</kbd>: none; not the drive keys W A S D Q E R F G); the keys set
  are kept in the browser per policy and work while the robot is driven.
  One with a `duration`
  runs that long in simulation time (as many physics steps at any speed of
  the Physics panel), then the default again and the select with it; one without
  until another is picked. Automatic skills only stand in for the default.
- `"commands": false`: trained with zeros for the twist and the commands,
  and observed so.
- `"reset_action": true`: trained from episodes whose action history starts
  at zero; the controller's last action is zeroed when the skill starts
  (otherwise it reads the previous model's as its own).

The Teleop panel's readout shows the model running after the command (in
network mode the server sends it with the status). Every skill keeps the
controller's state as it is (the observation history, the last action):
a switch is a hard swap, as on the robots.

`public/js/torchscript.js` reads the tensors of a TorchScript `.pt` (a zip
of raw storages) in the browser, so small exported MLPs/RNNs need no
conversion.

## Walking G1 (GR00T policy)

If the GR00T locomotion policies are checked out into `pretrained_policies/`
(`git clone https://huggingface.co/nepyope/GR00T-WholeBodyControl_g1
pretrained_policies/GR00T-WholeBodyControl_g1`), they load the first time a
robot runs the `groot-g1` policy. `public/js/groot.js` is a port of lerobot's
`GrootLocomotionController` running on onnxruntime-web (WASM, one thread,
~0.2 ms per inference): two MLPs, the skills of its `policy.json` (Walk,
and Balance, automatic for commands under 0.05, as lerobot switches them)
at 50 Hz on 6 stacked 86-D observations, driving legs and waist,
while the arms hold whatever target their actuators have (the default pose
at spawn; set them with IK chains or the Stage panel's control sliders, as
in g1_mujoco). The policy's PD gains are written
into the menagerie position actuators (`gainprm`/`biasprm`), and the IMU is
the floating base (`qpos[3:7]`, `qvel[3:6]`), as in lerobot's G1 sim.
Inference is async, so `App.advance()` awaits it every 0.02 s of sim time.
`g1.xml` and `g1_with_hands.xml` both walk.

## Walking Spot and Go2 (trained in MuJoCo Playground)

`training/` trains locomotion policies on the GPU with MuJoCo Playground
(MJX / MuJoCo Warp + Brax PPO) and exports them for the viewer; see
[`training/README.md`](../training/README.md). `uv run train.py` then `uv run export_onnx.py
runs/<name>` put `policy.onnx` and `policy.json` into
`pretrained_policies/playground_spot/` (or `playground_go2/`), and every
menagerie Spot (`boston_dynamics_spot/*.xml`) and the Go2
(`unitree_go2/go2.xml`) start with the `playground-spot` /
`playground-go2` policy on. The Spot policy is trained on the whole
`spot_arm.xml` (`training/envs/spot_arm.py`: Playground's Spot task, the
arm's joints fixed in a different pose in every training environment, with
domain randomization, pushes and commands every 1-4 s), so it walks with
the arm stowed, reaching out or moved by IK, and on `spot.xml` too.
Playground has no Go2 task: `training/envs/go2.py` runs its Go1 task on a
Go2 built from `go2.xml`. Both walk without falling under the keys or the
goal marker; `training/check_model.py` measures that on the viewer's own
model files. `public/js/playground.js` (and `PlaygroundController` in
`server/policies.py`) runs any such policy from its `policy.json`: it writes
the trained PD gains and joint damping / armature / friction loss into the
robot, builds the observation (body gyro and up vector, joint positions,
a history of tracking errors, foot positions, last action, command, or
for Go2 body velocity, gravity and joint velocities) and sets the targets
`default + action * scale` every 0.02 s; on `go2.xml`'s torque motors it
computes the trained PD torque itself before every physics step.

## Walking Microduck (Pollen's published policies)

Pollen Robotics' [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
(mjlab, MuJoCo Warp, rsl_rl PPO) trains its 14-servo bipedal duck, and
Pollen publishes the set the robot ships with on the Hugging Face Hub
([pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies),
Apache-2.0). Its ONNX files are in `pretrained_policies/microduck_official/`
(revision `665ae4e`, 2026-10-12, with the Hub's `manifest.json`), beside a
`policy.json` for them: microduck_rl's observation contract, default pose and
servo model, which every policy of the set shares. With a microduck_rl clone
next to `package.json`, the library repository's `scripts/import_microduck.py`
copies the robot (`custom/microduck/microduck.xml`, the ground-contact
collision model, so it can lie on the floor) and two of its scenes into the
library. The duck starts with `microduck-official` (Walk + get up, tricks ·
Microduck (Pollen)) on, and so do its three other colors
(`custom/microduck_sky`, `_graphite`, `_lavender`: the same model, recolored
by `import_microduck.py`).

Our own Microduck policies (a walk and a right kick trained here, and a
SuperMimic walk, `training/supermimic`) are at git tag
`before-official-microduck`; a saved stage or a remembered choice naming
them (`microduck-walk`, `microduck-supermimic`) gets the official set
(`RENAMED` in `policy.js` and `policies.py`). `scripts/export_microduck.py`
still writes a `policy.json` and ONNX files for checkpoints of our own.

### Skills

Its `policy.json` has velstand as the default and the others as skills (see
Skills above), as microduck_rl's daemon swaps them on the robot:

| skill | ONNX | key | what it does (tested in the viewer's physics) |
|---|---|---|---|
| Walk + get up (default) | `velstand.onnx` (v8) | | walks (slowly: ~0.08 m/s for 0.3), stands at zero command, turns (0.8 → ~1 rad/s), and gets back up by itself after a fall (face-down 0.4 s, face-up 0.6 s); body roll/pitch command while standing |
| Kick (right foot), (left foot) | `ball_kick_right.onnx`, `_left` | K, J | 0.5 s: a ball at the toe (9.5 cm ahead, 4.2 cm to that side) goes 1.6-3 m, about 7-9° off straight, to the kicking foot's side |
| Forward roll | `roulade.onnx` | O | 1 s: a forward roll; velstand stands it up after |
| Touch the ground | `alpha_ground_pick.onnx` | P | 2.8 s: bows and touches the floor with the mouth tip; its twist slot is a phase clock (`phase: {period: 4}`: cos, sin of 2π t/4) |
| Sit | `alpha_sitstand.onnx` | C | sits (trunk 0.06 m) and stays seated: its twist slot is the posture flag (`twist: [1, 0, 0]`); pick Walk to stand up |
| Stand | `alpha_stand.onnx` | | holds a stand, gets up after a fall, does not walk |
| Walk (alpha) | `alpha_walking.onnx` | | the older walk: a little faster (0.14 m/s), barely turns, no get-up |

The episodic ones start from a zero action history (`reset_action`) and see
zeros for the commands (`commands: false`). The kicks do not see the ball
(the robot has no way to): aim the duck first. `roller.onnx` and
`roller_crouch.onnx` are in the directory but not offered: they need the
Microduck on wheels, which the library lacks. `scripts/microduck_official_test.py`
runs each in the viewer's physics; `public/behaviors/microduck_skills.py` (a
Behavior-panel example) tries them all on a spawned duck and a ball.

They were trained on the BAM model of the XL330 servos (firmware P loop,
DC motor with back-EMF and battery sag, load-dependent friction), not on
plain position actuators, and do not stand without it. `public/js/microduck.js`
(and `MicroduckController` / `BamServos` in `server/policies.py`) turns the
robot's position actuators into motors on bind, computes the servo torques
and writes the friction budget into the joints before every physics step,
delays the targets by the training's command delay, and builds mjlab's 61-D
observation (trunk angular velocity and gravity, joint positions and
velocities, last action, the drive command, and the head and body pose
commands, see Commands above).
