"""The page's shipped defaults that the server reads too: where they live
under public/json, in one place."""

import json
import pathlib

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
IK_DEFAULTS = PUBLIC / 'json' / 'ik_defaults.json'      # ik.js, ik.py
BASE_DEFAULTS = PUBLIC / 'json' / 'base_defaults.json'  # base.js, base.py
ROS_PROFILES = PUBLIC / 'json' / 'ros_profiles.json'    # ros/robot.js


def load(path):
  return json.loads(path.read_text())
