"""Tasks by name (scripts' --task)."""

from smlab.tasks.duck_walk import DuckWalk

TASKS = {DuckWalk.name: DuckWalk}


def make_task(name: str):
  if name not in TASKS:
    raise SystemExit(f"no task {name}: one of {', '.join(TASKS)}")
  return TASKS[name]()
