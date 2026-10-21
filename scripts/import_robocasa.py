"""Makes RoboCasa's kitchen objects spawnable as free-floating assets.

mjcf_assets/objects/robocasa/ is an untouched copy of RoboCasa's objects
(robocasa/models/assets/objects/objaverse/): <category>/<instance>/model.xml
with visual/ and collision/ next to it. Those MJCFs have no joint (RoboCasa
adds one in Python), are centered on their bounding box, and keep their
collision geoms in group 0, so scripts/vendor.mjs does not index them.
Instead, for every instance this writes
mjcf_assets/objects/robocasa_spawn/<instance>/<instance>.xml (the naming
convention of scripts/vendor.mjs), a copy of model.xml with:
  - <compiler meshdir/texturedir> pointing back at the instance directory,
    so the meshes and textures stay where they are,
  - a free joint on the outer body, raised so its bounding box (the reg_bbox
    geom) sits on z = 0,
  - collision geoms moved to group 3, which the viewer does not draw,
  - mass="0" on the region class (reg_bbox and co., the invisible boxes
    RoboCasa places objects by): objaverse objects leave it at the default
    density, 1000 kg/m^3, which made a banana 0.71 kg instead of 0.15 and a
    basket 32 kg instead of 2.6,
and an asset.json grouping it under its category in the asset panel.
Nothing under robocasa/ is written. Screenshots
(robocasa_spawn/<instance>/<instance>.png) come from scripts/render_previews.mjs.

    python3 scripts/import_robocasa.py [category…]   # default: all
    node scripts/vendor.mjs   # re-index
"""

import json
import os
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / 'mjcf_assets' / 'objects' / 'robocasa'
OUT = ROOT / 'mjcf_assets' / 'objects' / 'robocasa_spawn'


def convert(model_xml, name, dst):
  tree = ET.parse(model_xml)
  root = tree.getroot()
  root.set('model', name)
  rel = os.path.relpath(model_xml.parent, dst).replace(os.sep, '/')
  root.insert(0, ET.Element('compiler', meshdir=rel, texturedir=rel))
  for d in root.iter('default'):
    if d.get('class') == 'collision':
      d.find('geom').set('group', '3')
    if d.get('class') == 'region':
      d.find('geom').set('mass', '0')
  for g in root.iter('geom'):  # collision geoms that set their group themselves
    if g.get('class') == 'collision' or (g.get('contype') not in ('0', None) and g.get('group') == '0'):
      g.set('group', '3')
  bbox = root.find('.//geom[@name="reg_bbox"]')  # in the object body or the outer one
  z = float((bbox.get('pos') or '0 0 0').split()[2])
  half = float(bbox.get('size').split()[2])
  outer = root.find('worldbody/body')
  outer.set('name', 'root')
  outer.set('pos', f'0 0 {half - z + 0.002:.5f}')
  outer.insert(0, ET.Element('freejoint', name='root'))
  ET.indent(tree)
  return tree


def main():
  if not SRC.is_dir():
    raise SystemExit(f'no {SRC}: copy RoboCasa\'s objects/objaverse/ there')
  cats = sys.argv[1:] or sorted(p.name for p in SRC.iterdir() if p.is_dir())
  OUT.mkdir(parents=True, exist_ok=True)
  done = 0
  for cat in cats:
    for inst in sorted((SRC / cat).iterdir()):
      model = inst / 'model.xml'
      if not model.exists():
        print(f'skip {inst}: no model.xml')
        continue
      name = inst.name  # <category>_<n>, unique across categories
      dst = OUT / name
      if dst.exists():  # keep a rendered screenshot
        for f in dst.iterdir():
          if f.suffix != '.png':
            shutil.rmtree(f) if f.is_dir() else f.unlink()
      dst.mkdir(exist_ok=True)
      try:
        tree = convert(model, name, dst)
      except (AttributeError, ET.ParseError) as err:
        print(f'skip {model}: {err}')
        shutil.rmtree(dst)
        continue
      tree.write(dst / f'{name}.xml', encoding='unicode')
      (dst / 'asset.json').write_text(json.dumps({
          'group': cat,
          'source': f'robocasa objaverse/{cat}/{name}',
      }, indent=1) + '\n')
      done += 1
  print(f'{done} objects of {len(cats)} categories -> {OUT}')


if __name__ == '__main__':
  main()
