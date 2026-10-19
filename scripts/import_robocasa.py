"""Copies RoboCasa kitchen objects into mjcf_assets/objects/robocasa/ as
free-floating assets.

RoboCasa's object MJCFs (robocasa/models/assets/objects/{lightwheel,objaverse}/
<category>/<instance>/model.xml) have no joint (RoboCasa adds one in Python),
are centered on their bounding box, and keep their collision geoms in group
0. For each picked category this copies one instance into <name>/ and writes
<name>.xml (the naming convention of scripts/vendor.mjs), a copy of
model.xml with:
  - a free joint on the outer body, raised so its bounding box (the reg_bbox
    geom) sits on z = 0,
  - collision geoms moved to group 3, which the viewer does not draw,
  - mass="0" on the region class (reg_bbox, the invisible bounding box
    RoboCasa places objects by): its objaverse objects leave it at the
    default density, 1000 kg/m^3, which made a banana 0.71 kg instead of
    0.15 and a basket 32 kg instead of 2.6 (the lightwheel ones set it),
and an asset.json recording where it came from. Screenshots (<name>.png)
come from scripts/render_previews.mjs.

    python3 scripts/import_robocasa.py [/path/to/robocasa]
    node scripts/vendor.mjs   # re-index
"""

import json
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'mjcf_assets' / 'objects' / 'robocasa'
MAX_BYTES = 15 * 2**20  # skip instances bigger than this, try the next one

# (source, category); the asset is named after the category, suffixed with
# the source where both have one.
PICKS = [
    ('objaverse', c) for c in [
        'apple', 'banana', 'orange', 'lemon', 'avocado', 'carrot', 'tomato', 'potato',
        'broccoli', 'bell_pepper', 'egg', 'bread', 'croissant', 'donut', 'cupcake',
        'bowl', 'plate', 'mug', 'cup', 'coffee_cup', 'fork', 'knife', 'spoon', 'spatula',
        'ladle', 'pan', 'kettle', 'teapot', 'cutting_board', 'rolling_pin', 'sponge',
        'can', 'bottled_water', 'milk', 'wine', 'ketchup', 'cereal', 'boxed_drink',
    ]
] + [
    ('lightwheel', c) for c in [
        'pot', 'saucepan', 'kettle', 'pitcher', 'glass_cup', 'jar', 'colander',
        'basket', 'fruit_bowl', 'tray', 'whisk', 'wooden_spoon', 'knife_block',
        'dish_rack', 'flower_vase', 'plant', 'stool',
    ]
]


def size(path):
  return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def convert(model_xml, name):
  tree = ET.parse(model_xml)
  root = tree.getroot()
  root.set('model', name)
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
  objects = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/home/jovyan/robocasa') / 'robocasa/models/assets/objects'
  cats = {}
  for src, cat in PICKS:
    cats.setdefault(cat, []).append(src)
  OUT.mkdir(parents=True, exist_ok=True)
  done = []
  for src, cat in PICKS:
    name = cat if len(cats[cat]) == 1 or src == 'objaverse' else f'{cat}_{src}'
    insts = sorted(p for p in (objects / src / cat).iterdir() if (p / 'model.xml').exists())
    inst = next((p for p in insts if size(p) <= MAX_BYTES), None)
    if inst is None:
      print(f'skip {src}/{cat}: no instance under {MAX_BYTES >> 20} MB')
      continue
    dst = OUT / name
    if dst.exists():
      shutil.rmtree(dst)
    shutil.copytree(inst, dst)
    convert(dst / 'model.xml', name).write(dst / f'{name}.xml', encoding='unicode')
    (dst / 'model.xml').unlink()
    (dst / 'asset.json').write_text(json.dumps({
        'source': f'robocasa {src}/{cat}/{inst.name}',
    }, indent=1) + '\n')
    done.append((name, size(dst)))
  total = sum(s for _, s in done)
  print(f'{len(done)} objects, {total / 2**20:.0f} MB -> {OUT}')


if __name__ == '__main__':
  main()
