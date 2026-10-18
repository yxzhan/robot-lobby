# Training locomotion policies (MuJoCo Playground)

Train a policy on the GPU with [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(MJX / MuJoCo Warp physics, Brax PPO), export it to ONNX and run it in the
viewer, locally (onnxruntime-web) or on the server (onnxruntime). No Isaac
Lab: training and the viewer both run MuJoCo.

```sh
cd training
uv sync                                   # .venv: playground, jax[cuda12], onnx, onnxruntime
uv run train.py --name spot               # SpotFlatTerrainJoystick, 100M steps (~5 min on an RTX 3080)
uv run export_onnx.py runs/spot           # -> ../pretrained_policies/playground_spot/
uv run train.py --env Go2JoystickFlatTerrain --name go2   # 200M steps (~9 min)
uv run export_onnx.py runs/go2            # -> ../pretrained_policies/playground_go2/
cd .. && node scripts/vendor.mjs          # re-index public/policies.json
```

Then spawn **boston_dynamics_spot** or **unitree_go2** (Robots ›
Quadrupeds): it starts with its policy on (`playground-spot` /
`playground-go2`, Teleop panel › Policy) and walks with the keys or the goal
marker.

| Task | Policy in its training env | In the viewer |
|---|---|---|
| `SpotFlatTerrainJoystick` (Playground's) | 89% fall within 20 s (eval reward peaks at 34 near 23M steps, then drops) | walks straight; can tip over when commands change quickly |
| `Go2JoystickFlatTerrain` (`envs/go2.py`) | 0 of 256 fall in 20 s | no falls with commands changing every 0.2 s; stands for commands below ~0.3 m/s / 0.3 rad/s |

## Files

- `pyproject.toml`, `uv.lock`: the environment. `jax` is held below 0.10
  because Brax 0.14 still calls `jax.device_put_replicated`.
- `envs/`: tasks of our own, registered with Playground on import.
  `go2.py`: Playground has no Go2 task, so this is its Go1 joystick task
  (rewards, observation, PPO config, domain randomization) on a Go2 model
  built with MjSpec from the viewer's `go2.xml` (same bodies, inertias,
  joints) the way Playground models its Go1 for training: feet-only
  collision, position servos (Kp 35) instead of the torque motors, Euler at
  4 ms with one solver iteration (`envs/build/`, generated).
- `train.py`: Playground's tuned PPO config for the task (`--steps`,
  `--envs`, `--evals` override it; `--njmax` raises the constraint rows per
  environment, which MuJoCo Warp drops silently beyond the limit). Writes
  `runs/<name>/params.pkl` and `run.json` (configs, reward per evaluation,
  wall time).
- `export_onnx.py`: writes `policy.onnx` (observation normalization, the
  MLP, tanh of the action mean, built directly with `onnx`, no torch / tf)
  and `policy.json` (joints, actuators, default pose, action scale, the
  actuator gains and joint damping / armature / friction loss the policy was
  trained with, observation layout, command ranges). Then checks it: ONNX
  against Brax's own inference on random inputs, and a closed loop in plain
  MuJoCo on the CPU (walking forward for 10 s).

## In the viewer

`public/playground.js` (browser) and `PlaygroundController` in
`server/policies.py` read `policy.json`, so a newly exported policy needs no
code change, only a `POLICIES` entry (`public/policy.js` and
`server/policies.py`) if it is for another robot or directory. On bind they
write the trained PD gains and joint parameters into the robot's actuators
and joints (and give them back on unbind), build the observation each 0.02 s
as `Observer` in `export_onnx.py` does (matching Playground's to 1e-6: the
sensors are those of the forward pass at the start of the last step, and the
observed last action is the one before the last), and set the position
targets `default + action * action_scale`. On torque motors (`go2.xml`) they
close the trained servo loop themselves every physics step (`substep`),
within the trained torque limits. `min_command` in `policy.json` is the
policy's dead band; the goal marker never commands less.

The viewer runs the menagerie model (`boston_dynamics_spot/spot.xml`), not
Playground's training copy: the same bodies, inertias and joints, but mesh
collision for the body and legs and softer foot contacts. A policy trained
only on Playground's model may track less well or fall in the viewer under
quickly changing commands; train longer or with domain randomization if it
does.
