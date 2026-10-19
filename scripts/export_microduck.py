"""Exports the Microduck walking policy trained in a microduck_rl clone
(github.com/pollen-robotics/microduck_rl) for the viewer:

  pretrained_policies/microduck_rl/walk.onnx    Mjlab-Velocity-Flat-MicroDuck
  pretrained_policies/microduck_rl/policy.json  what public/microduck.js and
                                                server/policies.py need

The ONNX file comes from the clone's own scripts/export.py (observation
normalizer baked in). policy.json has the joints, default pose and 61-D
observation layout the policy was trained with, and the BAM model of the
XL330 servos (github.com/Rhoban/bam, M6 parameters as bundled with it, and
microduck_rl's firmware gain, battery and command delay), which the viewer
runs in place of the model's position actuators.

    cd microduck_rl
    uv run python ../scripts/export_microduck.py --walk logs/rsl_rl/velocity/<run>/model_4999.pt
    cd .. && node scripts/vendor.mjs   # re-index public/policies.json

Without --walk the ONNX is kept and only policy.json is rewritten.
"""

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'pretrained_policies' / 'microduck_rl'
TASKS = {'walk': 'Mjlab-Velocity-Flat-MicroDuck'}

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
      # [twist (vx, vy, yaw rate), head pose (4), body pose (6)].
      'obs': [['base_ang_vel', 3], ['projected_gravity', 3], ['joint_pos', 14], ['joint_vel', 14],
              ['last_action', 14], ['command', 3], ['head_command', 4], ['body_command', 6]],
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
  ap.add_argument('--walk', type=pathlib.Path, help='checkpoint of Mjlab-Velocity-Flat-MicroDuck')
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


if __name__ == '__main__':
  main()
