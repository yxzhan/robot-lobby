"""Tasks of our own, registered with Playground's locomotion registry on
import (train.py and export_onnx.py import this package).

PPO_LIKE: whose tuned PPO config a task uses; RANDOMIZERS: its domain
randomization (Playground's registry only knows its own tasks')."""

import functools

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize

from . import go2, spot_arm

locomotion.register_environment('Go2JoystickFlatTerrain', functools.partial(go2.Joystick, task='flat_terrain'),
                                go2.default_config)
locomotion.register_environment('SpotArmJoystickFlatTerrain', functools.partial(spot_arm.Joystick, task='flat_terrain'),
                                spot_arm.default_config)

PPO_LIKE = {'Go2JoystickFlatTerrain': 'Go1JoystickFlatTerrain', 'SpotArmJoystickFlatTerrain': 'SpotFlatTerrainJoystick'}
RANDOMIZERS = {'Go2JoystickFlatTerrain': go1_randomize.domain_randomize}


def randomizer(env_name, env_cfg):
  """A task of ours' domain randomization, or None (Playground's own then)."""
  if env_name == 'SpotArmJoystickFlatTerrain':
    return spot_arm.make_randomizer(env_cfg)
  return RANDOMIZERS.get(env_name)
