# Training locomotion policies (MuJoCo Playground)

Train a policy on the GPU with [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(MJX / MuJoCo Warp physics, Brax PPO), export it to ONNX and run it in the
viewer, locally (onnxruntime-web) or on the server (onnxruntime). No Isaac
Lab: training and the viewer both run MuJoCo.

```sh
cd training
uv sync                                   # .venv: playground, jax[cuda12], onnx, onnxruntime
uv run train.py --name spot               # SpotFlatTerrainJoystick, 100M steps
uv run export_onnx.py runs/spot           # -> ../pretrained_policies/playground_spot/
cd .. && node scripts/vendor.mjs          # re-index public/policies.json
```

Then spawn **boston_dynamics_spot** (Robots › Quadrupeds): it starts with
the `playground-spot` policy on (Teleop panel › Policy) and walks with the
keys or the goal marker.

## Files

- `pyproject.toml`, `uv.lock`: the environment. `jax` is held below 0.10
  because Brax 0.14 still calls `jax.device_put_replicated`.
- `train.py`: Playground's tuned PPO config for the task (`--steps`,
  `--envs`, `--evals` override it). Writes `runs/<name>/params.pkl` and
  `run.json` (configs, reward per evaluation, wall time).
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
as `Observer` in `export_onnx.py` does, and set the position targets
`default + action * action_scale`.

The viewer runs the menagerie model (`boston_dynamics_spot/spot.xml`), not
Playground's training copy: the same bodies, inertias and joints, but mesh
collision for the body and legs and softer foot contacts. A policy trained
only on Playground's model may track less well or fall in the viewer under
quickly changing commands; train longer or with domain randomization if it
does.
