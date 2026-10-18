"""Copies Pollen Robotics' Microduck (github.com/pollen-robotics/microduck_rl,
a clone next to package.json by default) into the model library as digital
twins:

  mjcf_assets/robots/digital_twin/microduck/microduck.xml
      robot_groundcontact.xml, the model of its scene.xml and most of its
      tasks (collision geoms for the parts that touch the floor, group 3),
      with scene.xml's STAND keyframe as its own first keyframe (the viewer
      spawns a robot in it), and the meshes it uses in assets/;
  mjcf_assets/environments/digital_twin/microduck_apartment/microduck_apartment.xml
      apartment.xml: six rooms in 7 x 6 m with a staircase going down from
      the corridor (the viewer's floor at z = 0 covers its hole);
  mjcf_assets/environments/digital_twin/microduck_vslam_room/microduck_vslam_room.xml
      vslam_room.xml: an 11 x 11 m room with textured walls and pillars.

Each gets an asset.json (where it came from) and a README.md with the mesh
license (CC BY-NC-SA 4.0, Pollen Robotics). Screenshots (<name>.png) come
from scripts/render_previews.mjs.

    python3 scripts/import_microduck.py [/path/to/microduck_rl]
    node scripts/vendor.mjs   # re-index
"""

import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBRARY = ROOT / 'mjcf_assets'
UPSTREAM = 'https://github.com/pollen-robotics/microduck_rl'
ROBOT = 'robot_groundcontact.xml'
KEY = 'STAND'
ENVIRONMENTS = {'microduck_apartment': 'apartment.xml', 'microduck_vslam_room': 'vslam_room.xml'}

LICENSE = f"""
## License

From [microduck_rl]({UPSTREAM}) by Pollen Robotics, copied by
`scripts/import_microduck.py`{{changes}}. The MJCF is Apache 2.0; the meshes
(`assets/*.stl`) are © Pollen Robotics under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(non-commercial; adaptations under the same license). See the upstream
`LICENSE` and `LICENSE-MODELS.md`.
"""


def keyframe(scene, name):
  """The <key name=...> element of `scene` as text, whitespace collapsed."""
  scene = re.sub(r'<!--.*?-->', '', scene, flags=re.S)  # an older STAND is commented out
  m = re.search(rf'<key\s+name="{name}".*?/>', scene, re.S)
  if not m:
    raise SystemExit(f'no keyframe {name} in scene.xml')
  return re.sub(r'\s+', ' ', m.group(0)).replace('" ', '"\n         ', 1)


def write_meta(out, rel, readme):
  (out / 'asset.json').write_text(json.dumps({'source': UPSTREAM, 'file': rel}, indent=2) + '\n')
  (out / 'README.md').write_text(readme)


def robot(src):
  out = LIBRARY / 'robots' / 'digital_twin' / 'microduck'
  shutil.rmtree(out, ignore_errors=True)
  (out / 'assets').mkdir(parents=True)
  xml = (src / ROBOT).read_text()
  if '<keyframe' in xml:
    raise SystemExit(f'{ROBOT} has a keyframe of its own now: check which one comes first')
  key = keyframe((src / 'scene.xml').read_text(), KEY)
  xml = xml.replace('<mujoco model="microduck">', '<mujoco model="microduck">\n  <!-- Copied by scripts/import_microduck.py: '
                    f'{ROBOT} with scene.xml\'s {KEY} keyframe. -->', 1)
  head, tail = xml.rsplit('</mujoco>', 1)
  xml = f'{head}  <keyframe>\n    {key}\n  </keyframe>\n</mujoco>{tail}'
  meshes = sorted(set(re.findall(r'<mesh\s[^>]*file="([^"]+)"', xml)))
  for f in meshes:
    shutil.copy2(src / 'assets' / f, out / 'assets' / f)
  (out / 'microduck.xml').write_text(xml)
  write_meta(out, f'src/mjlab_microduck/robot/microduck/{ROBOT}', (
      '# Microduck\n\nA 14-joint bipedal duck robot (5 per leg, 4 in the neck and head) driven by\n'
      'Dynamixel XL330 position servos, ~20 cm tall. The model is `robot_groundcontact.xml`\n'
      f'(collision geoms for the parts that touch the floor), standing in its `{KEY}` keyframe.\n'
      + LICENSE.format(changes=', with a keyframe added')))
  return out, len(meshes)


def environment(src, name, file):
  out = LIBRARY / 'environments' / 'digital_twin' / name
  shutil.rmtree(out, ignore_errors=True)
  out.mkdir(parents=True)
  xml = (src / file).read_text()
  if re.search(r'\sfile="', xml):
    raise SystemExit(f'{file} references files now: copy them too')
  (out / f'{name}.xml').write_text(xml.replace('<mujoco model=', f'<!-- Copied by scripts/import_microduck.py: {file}. -->\n<mujoco model=', 1))
  title = name.replace('microduck_', '').replace('_', ' ')
  write_meta(out, f'src/mjlab_microduck/robot/microduck/{file}',
             f'# Microduck {title}\n\nThe Microduck project\'s `{file}`, unchanged.\n' + LICENSE.format(changes=''))
  return out


def main():
  repo = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'microduck_rl'
  src = repo / 'src' / 'mjlab_microduck' / 'robot' / 'microduck'
  if not (src / ROBOT).exists():
    raise SystemExit(f'no {src / ROBOT}: pass the path of a microduck_rl clone')
  out, n = robot(src)
  print(f'{out.relative_to(ROOT)}: {n} meshes')
  for name, file in ENVIRONMENTS.items():
    print(environment(src, name, file).relative_to(ROOT))


if __name__ == '__main__':
  main()
