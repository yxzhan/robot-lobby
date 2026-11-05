"""The robots the bridge knows: per robot its spec and its real driver (a
module and class, imported only when used: franky or a serial SDK need not
be installed to run another robot or the fake one).

A new robot: a module here with a RobotSpec and a RobotDriver subclass
(drivers/base.py), one line below, and the lobby's profile for its model
(public/json/ros_profiles.json, keyed by spec.lobby_key)."""

from __future__ import annotations

import importlib

from .base import GripperSpec, RobotDriver, RobotSpec, RobotState  # noqa: F401

ROBOTS = {
    # name: (module, spec, driver class)
    'panda': ('panda', 'PANDA', 'PandaDriver'),
    'so101': ('so101', 'SO101', 'SO101Driver'),
}


def spec(name) -> RobotSpec:
  mod, sp, _ = ROBOTS[name]
  return getattr(importlib.import_module(f'.{mod}', __name__), sp)


def driver(name, kind='real', **options) -> RobotDriver:
  """Robot `name`'s driver: its real one, or kind 'fake' (drivers/fake.py)."""
  if kind == 'fake':
    from .fake import FakeDriver
    return FakeDriver(spec(name), **options)
  mod, sp, cls = ROBOTS[name]
  m = importlib.import_module(f'.{mod}', __name__)
  return getattr(m, cls)(getattr(m, sp), **options)
