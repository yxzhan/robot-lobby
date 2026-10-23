"""Copies the picked RoboCasa kitchen objects into mjcf_assets/objects/robocasa/
as free-floating assets.

RoboCasa's objaverse objects (robocasa/models/assets/objects/objaverse/
<category>/<instance>/model.xml) have no joint (RoboCasa adds one in Python),
are centered on their bounding box, and keep their collision geoms in group
0. Each instance in PICKS (named <category>_<n>, as RoboCasa's directory) is
copied to robocasa/<name>/ (visual/, collision/) with <name>.xml (the naming
convention of scripts/vendor.mjs), a copy of model.xml with:
  - a free joint on the outer body, raised so its bounding box (the reg_bbox
    geom) sits on z = 0,
  - collision geoms moved to group 3, which the viewer does not draw,
  - mass="0" on the region class (reg_bbox and co., the invisible boxes
    RoboCasa places objects by): objaverse objects leave it at the default
    density, 1000 kg/m^3, which made a banana 0.71 kg instead of 0.15 and a
    basket 32 kg instead of 2.6,
and an asset.json recording where it came from.
A screenshot (<name>.jpg) already in robocasa/<name>/ is kept; missing ones
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

# The instances kept (picked in the viewer from all 914 objaverse ones).
PICKS = [
    'apple_0', 'avocado_4', 'bagel_5', 'bagel_9', 'bagged_food_3', 'baguette_6', 'banana_22',
    'bar_0', 'bar_2', 'beer_10', 'bottled_drink_1', 'bottled_water_3', 'boxed_drink_3',
    'boxed_food_2', 'can_17', 'can_4', 'canned_food_0', 'canned_food_19', 'carrot_6',
    'cereal_12', 'cereal_8', 'cheese_0', 'coffee_cup_11', 'coffee_cup_12', 'coffee_cup_20',
    'cup_5', 'cutting_board_13', 'donut_2', 'egg_0', 'egg_13', 'fork_0', 'fork_9', 'hot_dog_5',
    'jam_0', 'jam_5', 'jug_1', 'ketchup_8', 'kettle_0', 'kettle_11', 'knife_17', 'knife_20',
    'ladle_2', 'ladle_4', 'lemon_3', 'milk_0', 'mug_1', 'mug_14', 'pan_0', 'pan_24', 'pear_18',
    'plate_0', 'plate_12', 'plate_19', 'plate_4', 'potato_3', 'rolling_pin_0', 'scissors_4',
    'soap_dispenser_2', 'spatula_0', 'sponge_4', 'spoon_3', 'spoon_4', 'spray_0', 'steak_3',
    'steak_6', 'teapot_11', 'teapot_5', 'teapot_7', 'tray_0', 'tray_7', 'water_bottle_0',
    'water_bottle_10', 'water_bottle_17', 'water_bottle_20', 'wine_4', 'yogurt_1',
]


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
  objects = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/home/jovyan/robocasa') / 'robocasa/models/assets/objects/objaverse'
  OUT.mkdir(parents=True, exist_ok=True)
  for name in PICKS:
    cat = name.rsplit('_', 1)[0]
    inst = objects / cat / name
    dst = OUT / name
    shot = dst / f'{name}.jpg'
    jpg = shot.read_bytes() if shot.exists() else None
    if dst.exists():
      shutil.rmtree(dst)
    shutil.copytree(inst, dst)
    convert(dst / 'model.xml', name).write(dst / f'{name}.xml', encoding='unicode')
    (dst / 'model.xml').unlink()
    if jpg:
      shot.write_bytes(jpg)
    (dst / 'asset.json').write_text(json.dumps({
        'source': f'robocasa objaverse/{cat}/{name}',
    }, indent=1) + '\n')
  print(f'{len(PICKS)} objects -> {OUT}')


if __name__ == '__main__':
  main()
