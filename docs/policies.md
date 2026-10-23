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
An entry's optional `physics` names World settings it needs (e.g.
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

`public/js/torchscript.js` reads the tensors of a TorchScript `.pt` (a zip
of raw storages) in the browser, so small exported MLPs/RNNs need no
conversion.

## Walking G1 (GR00T policy)

If the GR00T locomotion policies are checked out into `pretrained_policies/`
(`git clone https://huggingface.co/nepyope/GR00T-WholeBodyControl_g1
pretrained_policies/GR00T-WholeBodyControl_g1`), they load the first time a
robot runs the `groot-g1` policy. `public/js/groot.js` is a port of lerobot's
`GrootLocomotionController` running on onnxruntime-web (WASM, one thread,
~0.2 ms per inference): two MLPs (Balance when the command is ~0, Walk
otherwise) at 50 Hz on 6 stacked 86-D observations, driving legs and waist,
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

## Walking Microduck (trained in microduck_rl)

Pollen Robotics' [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
(mjlab, MuJoCo Warp, rsl_rl PPO) trains its 14-servo bipedal duck; with a
clone next to `package.json`, `scripts/import_microduck.py` copies the
robot (`custom/microduck/microduck.xml`) and two of its scenes into
the library, and after training `scripts/export_microduck.py` (run in the
clone's environment, see its docstring) exports the walking policy to
`pretrained_policies/microduck_rl/` (`walk.onnx`, `policy.json`). The duck
starts with `microduck-walk` on (keys and goal marker).

It was trained on the BAM model of the XL330 servos (firmware P loop,
DC motor with back-EMF and battery sag, load-dependent friction), not on
plain position actuators, and does not stand without it. `public/js/microduck.js`
(and `MicroduckController` / `BamServos` in `server/policies.py`) turns the
robot's position actuators into motors on bind, computes the servo torques
and writes the friction budget into the joints before every physics step,
delays the targets by the training's command delay, and builds mjlab's 61-D
observation (trunk angular velocity and gravity, joint positions and
velocities, last action, the drive command, and the head and body pose
commands, see Commands above). In the viewer's world it walks without
falling at ~2/3 of the commanded speed forward and back, turns at the
commanded rate, but sidesteps slowly (0.2 m/s → 0.05); standing still it
does not start below ~0.2 m/s or ~0.6 rad/s (`min_command`, so the goal
marker commands at least that) and often does not turn right in place, as
in microduck_rl's own CPU rehearsal (`scripts/infer_policy.py`).
