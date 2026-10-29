"""The page's shipped defaults that the server reads too: where they live
under public/js (each beside the module that uses it), in one place."""

import json
import pathlib

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
IK_DEFAULTS = PUBLIC / 'js' / 'controllers' / 'ik' / 'defaults.json'      # ik.js, ik.py
BASE_DEFAULTS = PUBLIC / 'js' / 'controllers' / 'base' / 'defaults.json'  # base.js, base.py
ROS_PROFILES = PUBLIC / 'js' / 'ros' / 'profiles.json'                    # ros/robot.js


def load(path):
  return json.loads(path.read_text())
