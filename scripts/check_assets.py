"""Compiles every model of the library index (public/json/assets.json, written
by scripts/vendor.mjs) with MuJoCo, to catch missing files before the GitHub
Pages build publishes them (.github/workflows/pages.yml).

  python3 scripts/check_assets.py
"""
import json
import sys
from pathlib import Path

import mujoco

PUBLIC = Path(__file__).resolve().parent.parent / 'public'


def main():
  bad = []
  models = json.loads((PUBLIC / 'json' / 'assets.json').read_text())
  for m in models:
    rel = f"{m['dir']}/{m['main']}"
    try:
      mujoco.MjModel.from_xml_path(str(PUBLIC / rel))
      print(f'ok   {rel}')
    except Exception as err:  # noqa: BLE001 - report them all
      bad.append(rel)
      print(f'FAIL {rel}: {err}')
  if bad:
    sys.exit(f'{len(bad)} of {len(models)} models do not compile')
  print(f'{len(models)} models compile')


if __name__ == '__main__':
  main()
