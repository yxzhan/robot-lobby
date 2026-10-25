# SuperMimic training on mjlab tasks (smlab)

Model-based RL as in [SuperMimic](https://arxiv.org/abs/2610.28480) (a learned
world model plus back-propagation through imagined rollouts), with the loop
split from the robot: `smlab/core` is task-independent, a task in
`smlab/tasks` supplies the mjlab env and everything imagination must mirror.
The core is adapted from
[go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) (MIT).

It runs in the environment of a [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
clone (mjlab 1.3, its tasks and BAM servo model; `uv sync` there), not in
`training/`'s Playground one:

```sh
cd training/supermimic
PY="env -u PYTHONPATH <microduck_rl>/.venv/bin/python"   # ROS's PYTHONPATH breaks some imports
$PY scripts/check_task.py --onnx ../../pretrained_policies/microduck_rl/walk.onnx   # imagination vs env
$PY scripts/train.py --task duck-walk --run-name walk --iterations 6000 --lr-decay-iterations 6000
$PY scripts/eval.py --checkpoint logs/duck-walk/<run>/model_6000.pt               # falls, rewards
$PY scripts/eval.py --onnx ../../pretrained_policies/microduck_rl/walk.onnx        # PPO, same metrics
$PY scripts/eval.py --checkpoint ... --envs 1 --seconds 15 --video walk.mp4        # MUJOCO_GL=egl headless
$PY scripts/wm_check.py --checkpoint logs/duck-walk/<run>/model_6000.pt           # world model error
$PY scripts/export_onnx.py logs/duck-walk/<run>/model_6000.pt walk.onnx
```

Every field of `TrainCfg` (`smlab/core/trainer.py`) is a flag. The defaults
(4096 envs, buffer 400, imagination batch 4096) take ~9 GB on an RTX 3080 at
~0.29 s per iteration; `--num-envs 2048 --imag-batch 2048` for less.

## Layout

    smlab/core/   trainer.py (collect / fit world model / improve policy, logging,
                  checkpoints, collapse guard), world_model.py, imagine.py
                  (rollouts, TD(lambda)), buffer.py, nets.py
    smlab/tasks/  base.py (the Task interface: what a task must give), duck_walk.py
    scripts/      train.py, eval.py, check_task.py, wm_check.py, export_onnx.py

Changes from go2-supermimic's loop besides the split: bootstrap values from an
EMA copy of the critic (bootstrapping from the critic being fit, its values
ran away within a few hundred iterations); the actor's last layer starts near
zero; the collapse guard counts falls per robot-second against the lowest
rate so far; collection noise in the env's action units, set for the
Microduck (1 rad per unit, go2's is 0.25).

## Adding a task

Subclass `smlab.tasks.base.Task` (its docstring lists what to supply) and
register it in `smlab/tasks/__init__.py`. The work is in three places:

- **State**: what the rewards, the observation and the termination need, plus
  whatever makes the dynamics predictable (foot positions, contact flags).
- **Observation**: the env's actor observation rebuilt from imagined states,
  with its delays, biases and noise.
- **Rewards**: each env term as a smooth function of the imagined state, and
  the termination as a probability. A term whose gradient vanishes where
  the policy starts (a hard window, an event) teaches nothing by
  back-propagation; give it a wide smooth version that narrows over training.

`scripts/check_task.py` compares the last two with the env on real transitions
before any training: the observation should differ only by the env's noise,
each reward term should match in mean and correlate per sample.

## duck-walk

microduck_rl's `Mjlab-Velocity-Flat-MicroDuck`, unchanged (BAM servos, domain
randomization, pushes, velocity / head / body commands, curricula). The
env's curricula count env steps; the trainer sets the step counter to
iteration x 24, as rsl_rl's PPO would. The world model predicts the contact
sensor's flags (a foot height threshold misjudged 6% of samples); the
air-time window starts 0.06 s soft and narrows to 0.01 s over 4000 iterations.

Check against the env (PPO walking policy driving, no pushes): observation
within the env's noise; reward terms with per-sample correlation 0.91-1.00,
except foot_swing_height (0.6) and foot_slip (0.9, half the mean), which read
landing velocities the env takes one physics substep earlier.

Deterministic evaluation, 128 envs x 20 s with pushes (RTX 3080):

| Policy | Falls / robot-min | track_linear | track_angular | air_time | Reward / s |
|---|---|---|---|---|---|
| zero actions (default pose) | 46.5 | 1.01 | 0.67 | - | 5.36 |
| SuperMimic, 6000 it (29 min), hard air-time window | 3.19 | 1.12 | 1.35 | 0.003 | 7.23 |
| SuperMimic, 6000 it (29 min), annealed window | 0.023 | 1.13 | 1.31 | 0.005 | 7.30 |
| PPO (rsl_rl, 5000 it, 3.3 h; `walk.onnx`) | 0.12-0.21 | 1.28 | 1.25 | 1.35 | 8.37 |

It stands and follows the velocity commands (PPO's training run was at a
similar point after the same 29 minutes) but shuffles: its feet hardly
leave the ground, which is nearly all of the gap in reward to PPO. Not yet
tried: a gait shaping term, a longer run, a warm start from PPO.
