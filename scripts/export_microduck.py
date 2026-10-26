"""Exports the Microduck policies trained in a microduck_rl clone
(github.com/pollen-robotics/microduck_rl) for the viewer:

  pretrained_policies/microduck_rl/walk.onnx        Mjlab-Velocity-Flat-MicroDuck
  pretrained_policies/microduck_rl/kick_right.onnx  Mjlab-BallKick-Flat-MicroDuck
  pretrained_policies/microduck_rl/policy.json      what public/js/microduck.js and
                                                    server/policies.py need
  pretrained_policies/microduck_supermimic/policy.json
      the same for the walking policy trained with SuperMimic
      (training/supermimic: its export_onnx.py writes walk.onnx there), if
      it is there: a policy of its own, run by the same controller

Walking is the policy the duck runs; the others are skills it switches to
for a while and comes back from (policy.json's "skills"), sharing its
observation layout, as microduck_rl's daemon swaps them on the robot.

The ONNX file comes from the clone's own scripts/export.py (observation
normalizer baked in). policy.json has the joints, default pose and 61-D
observation layout the policy was trained with, and the BAM model of the
XL330 servos (github.com/Rhoban/bam, M6 parameters as bundled with it, and
microduck_rl's firmware gain, battery and command delay), which the viewer
runs in place of the model's position actuators.

    cd microduck_rl
    uv run python ../scripts/export_microduck.py --walk logs/rsl_rl/velocity/<run>/model_4999.pt \
        --kick-right logs/rsl_rl/ball_kick_right/<run>/model_9999.pt
    cd .. && node scripts/vendor.mjs   # re-index public/json/policies.json

A policy left out keeps its ONNX; policy.json is rewritten either way.
"""

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'pretrained_policies' / 'microduck_rl'
TASKS = {'walk': 'Mjlab-Velocity-Flat-MicroDuck', 'kick_right': 'Mjlab-BallKick-Flat-MicroDuck'}
SUPERMIMIC = ROOT / 'pretrained_policies' / 'microduck_supermimic'
# The skills besides walking: run for `duration` seconds, then back to
# walking (microduck_rl's "episodic" policies). The kick task starts from a
# stand with the ball at the right toe (it does not see the ball), kicks at
# once and settles; its command slots are zero in training (commands: false).
# 0.3 s: the kick itself (the ball is off at ~0.2 s), walking takes over the
# settling, as tried in the viewer (2 s, the daemon's, stands a while).
# reset_action: its episodes start with a zero action history; reading the
# walking policy's last action as its own, it does not kick.
SKILLS = {
    'kick_right': {'label': 'Kick (right foot)', 'duration': 0.3, 'commands': False, 'key': 'k', 'reset_action': True,
                   'about': 'kicks a ball at the right toe forward from a stand, then walking again'},
}

# microduck_rl's src/mjlab_microduck/robot/microduck_constants.py (HOME_FRAME,
# _BAM_ACTUATOR_KWARGS) and scripts/infer_policy.py (its deployment rehearsal).
JOINTS = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
          'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
          'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
DEFAULT_POSE = [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                0.0, 0.0873, 0.4579, 0.0049, -0.4530]
KP_FW = 200.0                 # firmware P gain
VIN_RANGE = (6.5, 8.2)        # battery voltage, sampled per env in training
VIN = 7.4                     # infer_policy.py's nominal 2S LiPo
VIN_DROP_GAIN = 0.1           # V/Nm: sag vin - gain * sum|tau| (training samples 0-0.2)
VIN_MIN = 6.0
DELAY_STEPS = (3, 6)          # command delay in 5 ms physics steps
# The velocity task's head_pose and body_pose commands (microduck_velocity_env_cfg.py):
# the head's are offsets of its joints from the default pose, sampled up to
# each joint's reach by the end of the head_pose_range curriculum (step
# 2000 * 24); the body pose stays within a few mm and degrees there, at a
# reward weight of 0 (the standup task is the one that follows it).
COMMANDS = {
    'head': {'type': 'vector', 'label': 'Head', 'dims': ['neck pitch', 'head pitch', 'head yaw', 'head roll'],
             'unit': 'rad', 'range': [[-1.1, 1.1], [-1.1, 1.1], [-1.4, 1.4], [-0.31, 0.31]],
             'about': 'head joint offsets from the default pose, held while walking'},
    'body': {'type': 'vector', 'label': 'Body', 'dims': ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
             'unit': ['m', 'm', 'm', 'rad', 'rad', 'rad'],
             'range': [[-0.005, 0.005]] * 3 + [[-0.05, 0.05]] * 3,
             'about': 'body pose offset: trained only within these few mm and degrees, and barely followed'},
}
# bam.mjlab.BamActuator's stiffened joint friction (stiff_frictionloss=True).
SOLREF_FRICTION = [-5.0e4, -2.0e2]
SOLIMP_FRICTION = [0.99, 0.9999, 0.001, 0.5, 2.0]


def export(task, checkpoint, onnx):
  subprocess.run(['uv', 'run', 'scripts/export.py', task, '--checkpoint-file', str(checkpoint),
                  '--onnx-file', str(onnx), '--device', 'cpu'], check=True)


def spec(sources):
  import bam
  from bam.model import load_model
  m6 = json.loads((pathlib.Path(bam.__file__).parent / 'params' / 'xl330' / 'm6.json').read_text())
  params = {k: v for k, v in m6.items() if isinstance(v, float) and k != 'q_offset'}
  # The firmware current limit training runs with: bam.mjlab.BamActuator keeps
  # XL330Actuator's default (microduck_rl leaves max_current unset).
  xl330 = load_model(motor_name='xl330', model='m6').actuator
  return {
      'source': 'https://github.com/pollen-robotics/microduck_rl',
      'checkpoints': sources,
      'ctrl_dt': 0.02,
      'root': 'trunk_base',
      'joints': JOINTS,
      'actuators': JOINTS,
      'default_pose': DEFAULT_POSE,
      'action_scale': 1.0,
      'input': 'obs',
      'output': 'actions',
      # mjlab's actor observation terms, in order: root angular velocity and
      # gravity in the trunk frame, q - default, qdot (one control tick old,
      # as in training), the last action, then the 13-D command block
      # [twist (vx, vy, yaw rate), head pose (4), body pose (6)], the poses
      # being the commands below (public/js/commands.js).
      'obs': [['base_ang_vel', 3], ['projected_gravity', 3], ['joint_pos', 14], ['joint_vel', 14],
              ['last_action', 14], ['command', 3], ['cmd:head', 4], ['cmd:body', 6]],
      'commands': COMMANDS,
      # Policies by name, the first the one it runs; any other a skill of
      # SKILLS, offered when its ONNX is there.
      'skills': {'walk': {'onnx': 'walk.onnx', 'label': 'Walk'},
                 **{k: {'onnx': f'{k}.onnx', **v} for k, v in SKILLS.items() if (OUT / f'{k}.onnx').exists()}},
      'joint_vel_lag': 1,
      'command_range': {'lin_vel_x': [-0.4, 0.4], 'lin_vel_y': [-0.3, 0.3], 'ang_vel_yaw': [-1.0, 1.0]},
      # The policy's dead band: standing still, it does not start for
      # less (measured in the viewer's world and in microduck_rl's own CPU
      # rehearsal, scripts/infer_policy.py, alike). Turning right in place it
      # often does not start at all; walking, it turns either way.
      'min_command': {'speed': 0.2, 'turn': 0.6},
      'bam': {
          'model': 'xl330 m6', **params,
          'kp': KP_FW, 'error_gain': xl330.error_gain, 'max_pwm': xl330.max_pwm, 'max_current': xl330.max_current,
          'vin': VIN, 'vin_drop_gain': VIN_DROP_GAIN, 'vin_min': VIN_MIN,
          'force_limit': max(VIN_RANGE) * params['kt'] / params['R'],
          'delay': sum(DELAY_STEPS) / 2 * 0.005,
          'solref_friction': SOLREF_FRICTION, 'solimp_friction': SOLIMP_FRICTION,
      },
  }


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  for name, task in TASKS.items():
    ap.add_argument(f"--{name.replace('_', '-')}", type=pathlib.Path, help=f'checkpoint of {task}')
  args = ap.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  index = OUT / 'policy.json'
  sources = json.loads(index.read_text()).get('checkpoints', {}) if index.exists() else {}
  sources = {k: v for k, v in sources.items() if k in TASKS}
  for name, task in TASKS.items():
    ckpt = getattr(args, name)
    if ckpt:
      export(task, ckpt, OUT / f'{name}.onnx')
      sources[name] = {'task': task, 'checkpoint': str(ckpt)}
  index.write_text(json.dumps(spec(sources), indent=1) + '\n')
  print(f'{index.relative_to(ROOT)}: {", ".join(sorted(sources)) or "no checkpoints"}')
  if (SUPERMIMIC / 'walk.onnx').exists():
    # The same task, robot and servo model; walking alone.
    index = SUPERMIMIC / 'policy.json'
    old = json.loads(index.read_text()) if index.exists() else {}
    sm = spec(old.get('checkpoints', {}))
    sm['source'] = 'training/supermimic (SuperMimic: world model + back-propagation through imagined rollouts)'
    sm['skills'] = {'walk': {'onnx': 'walk.onnx', 'label': 'Walk'}}
    index.write_text(json.dumps(sm, indent=1) + '\n')
    print(f'{index.relative_to(ROOT)}')


if __name__ == '__main__':
  main()
