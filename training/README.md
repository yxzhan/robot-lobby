# Training locomotion policies (MuJoCo Playground)

(`supermimic/`: model-based training, SuperMimic, on mjlab tasks such as
microduck_rl's; its own README.)

Train a policy on the GPU with [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(MJX / MuJoCo Warp physics, Brax PPO), export it to ONNX and run it in the
viewer, locally (onnxruntime-web) or on the server (onnxruntime). No Isaac
Lab: training and the viewer both run MuJoCo.

```sh
cd training
uv sync                                   # .venv: playground, jax[cuda12], onnx, onnxruntime
uv run train.py --env SpotArmJoystickFlatTerrain --steps 300_000_000 \
    --set "command_config.lin_vel_y=[-0.5, 0.5]" --name spot-arm   # envs/spot_arm.py (~14 min on an RTX 3080)
uv run export_onnx.py runs/spot-arm --out ../pretrained_policies/playground_spot
uv run check_model.py ../pretrained_policies/playground_spot \
    ../mjcf_assets/robots/mujoco_menagerie/boston_dynamics_spot/spot_arm.xml --arm   # on the viewer's model
uv run train.py --env Go2JoystickFlatTerrain --name go2   # 200M steps (~9 min)
uv run export_onnx.py runs/go2            # -> ../pretrained_policies/playground_go2/
cd .. && node scripts/vendor.mjs          # re-index public/policies.json
```

Then spawn a **boston_dynamics_spot** model or **unitree_go2** (Robots ›
Quadrupeds): it starts with its policy on (`playground-spot` /
`playground-go2`, Teleop panel › Policy) and walks with the keys or the goal
marker.

| Task | Policy in its training env | In the viewer |
|---|---|---|
| `SpotFlatTerrainJoystick` (Playground's, the earlier Spot policy) | 89% fall within 20 s (eval reward peaks at 34 near 23M steps, then drops) | walks straight; on `spot_arm.xml` 36 of 40 `check_model.py` runs fell, on `spot.xml` 7 of 10 |
| `SpotArmJoystickFlatTerrain` (`envs/spot_arm.py`, 300M steps, sideways ±0.5 m/s) | eval reward 38 | `check_model.py --arm` on `spot_arm.xml`: 5 of 80 runs fell, all but one with random commands every 0.2 s; never walking straight or with the arm in any of its poses under commands every 1 s but twice; `spot.xml`: about as well |
| `Go2JoystickFlatTerrain` (`envs/go2.py`) | 0 of 256 fall in 20 s | no falls with commands changing every 0.2 s; stands for commands below ~0.3 m/s / 0.3 rad/s |

## Files

- `pyproject.toml`, `uv.lock`: the environment. `jax` is held below 0.10
  because Brax 0.14 still calls `jax.device_put_replicated`.
- `envs/`: tasks of our own, registered with Playground on import.
  `spot_arm.py`: Playground's Spot joystick task (rewards, observation,
  gains, PPO config) on a model built from the viewer's `spot_arm.xml`:
  feet-only collision, Euler at 4 ms, the arm's seven joints fixed (bodies,
  masses and inertias kept, so the policy drives the 12 leg joints and the
  task is unchanged). Every training environment turns the fixed arm into a
  pose of its own (30% stowed, the rest anywhere in `ARM_RANGE`), on top of
  Go1's domain randomization (Playground's Spot task has none); pushes on;
  a new command every 1-4 s instead of every 4 s. Go1's task, tried first
  on this robot, only learnt to stand.
  `go2.py`: Playground has no Go2 task, so this is its Go1 joystick task
  (rewards, observation, PPO config, domain randomization) on a Go2 model
  built with MjSpec from the viewer's `go2.xml` (same bodies, inertias,
  joints) the way Playground models its Go1 for training: feet-only
  collision, position servos (Kp 35) instead of the torque motors, Euler at
  4 ms with one solver iteration (`envs/build/`, generated).
- `train.py`: Playground's tuned PPO config for the task (`--steps`,
  `--envs`, `--evals` override it; `--njmax` raises the constraint rows per
  environment, which MuJoCo Warp drops silently beyond the limit; `--set
  key=value` changes the task config, e.g. `--set Kp=500` or `--set
  "command_config.lin_vel_y=[-0.5, 0.5]"`, and export_onnx.py follows it). Writes
  `runs/<name>/params.pkl` and `run.json` (configs, reward per evaluation,
  wall time).
- `export_onnx.py`: writes `policy.onnx` (observation normalization, the
  MLP, tanh of the action mean, built directly with `onnx`, no torch / tf)
  and `policy.json` (joints, actuators, default pose, action scale, the
  actuator gains and joint damping / armature / friction loss the policy was
  trained with, observation layout, command ranges). Then checks it: ONNX
  against Brax's own inference on random inputs, and a closed loop in plain
  MuJoCo on the CPU (walking forward for 10 s).
- `check_model.py`: the exported policy on a robot file as the viewer runs
  it (the file's own collision and actuators, the viewer's physics
  defaults, the trained gains set as its controllers do): walking forward,
  then random commands every 1 s and every 0.2 s over several seeds, and
  with `--arm` again with the arm held in other poses. Prints the falls and
  the tracking error.

## In the viewer

`public/js/playground.js` (browser) and `PlaygroundController` in
`server/policies.py` read `policy.json`, so a newly exported policy needs no
code change, only a `POLICIES` entry (`public/js/policy.js` and
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

The viewer runs the menagerie model files, not the training copies: the
same bodies, inertias and joints, but mesh collision for the body and legs,
softer foot contacts and the viewer's physics (2 ms, elliptic cone).
`check_model.py` measures the difference: for the Spot policy it is small
(about as many falls in its training model as on `spot_arm.xml`), what made
it fall was quickly changing commands, which the 1-4 s commands and the
randomization address.
