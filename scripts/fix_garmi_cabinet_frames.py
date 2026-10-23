"""Gives the Garmi apartment cabinet's links frames of their own (cram-vrb-lab
assets/garmi-apartment): the cabinet asset's geometry is in the apartment
map's coordinates, and each link (Static, door_N, drawer_N) is translated by
minus the cabinet's place in the scene, so every link frame lies near the
map's origin (in TF too) instead of on its door or drawer.

Each moving link's frame goes to its joint's anchor (a door's hinge, a
drawer's slide point: physics:localPos1), Static's to the cabinet's own
frame (Actor_0000); everything under a link (its meshes, handles and the
collision shapes world.usda authors on it) and the joints' local positions
are moved back by as much, so nothing moves in the world. The links have no
rotation or scale, so this is a translation throughout.

The values are changed where they are authored (cabinet_…_articulated.usda,
world.usda), line by line in the text: saving the layers through USD would
drop their comments. Then the files are read again and checked: every prim
under the cabinet and every joint anchor keeps its world pose (1e-6), else
the files are put back as they were.

    python3 scripts/fix_garmi_cabinet_frames.py ~/cram-vrb-lab/assets/garmi-apartment/world.usda
    python3 scripts/usd2mjcf.py ~/cram-vrb-lab/assets/garmi-apartment/world.usda \\
        mjcf_assets/environments/aicor/garmi_apartment_usd
"""

import argparse
import re

from pxr import Gf, Usd, UsdGeom, UsdPhysics

CABINET = '/Root/Meshes/Assets/cabinet/Actor_0000'
FRAME_OP = 'xformOp:translate:frame'  # put in front where a child has no translate first


def strongest(attr):
  """The strongest spec of an attribute (where its value is authored), or None."""
  stack = attr.GetPropertyStack(Usd.TimeCode.Default()) if attr else []
  return stack[0] if stack else None


def vec(v):
  return '(' + ', '.join(repr(float(x)) for x in v) + ')'


class Edits:
  """Changes to make in the layers' text: per layer file, per prim path (in
  that layer), attribute name -> (declaration, value text)."""

  def __init__(self):
    self.files = {}

  def set(self, layer, prim_path, name, decl, value):
    self.files.setdefault(layer.realPath, {}).setdefault(str(prim_path), {})[name] = (decl, value)

  def apply(self):
    """Rewrites the files; returns their old texts."""
    old = {}
    for path, prims in self.files.items():
      with open(path) as f:
        old[path] = f.read()
      lines = patch(old[path].split('\n'), prims)
      with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return old


PRIM = re.compile(r'^\s*(def|over|class)\b[^"]*"([^"]+)"')
ATTR = re.compile(r'^(\s*)(?:uniform\s+)?[\w\[\]]+\s+([\w:]+)\s*=')


def patch(lines, prims):
  """Sets attributes in a .usda text: `prims` maps a prim path to {name:
  (declaration, value)}; an attribute the prim's block has is replaced, the
  others are added at the top of the block. Prim blocks are found by their
  `def` / `over` lines and the braces on lines of their own (a dictionary's
  closing brace has another indentation)."""
  stack, pending, todo = [], None, {p: dict(a) for p, a in prims.items()}
  out = []
  for line in lines:
    m = PRIM.match(line)
    if m:
      pending = m.group(2)
    s = line.strip()
    if s == '{' and pending is not None:
      parent = stack[-1][0] if stack else ''
      path = f'{parent}/{pending}'
      indent = len(line) - len(line.lstrip())
      stack.append((path, indent))
      pending = None
      out.append(line)
      continue
    if s == '}' and stack and len(line) - len(line.lstrip()) == stack[-1][1]:
      path, indent = stack.pop()
      for name, (decl, value) in todo.pop(path, {}).items():  # not in the block: added
        out.insert(block_start(out, indent), f'{" " * (indent + 4)}{decl} {name} = {value}')
      out.append(line)
      continue
    a = ATTR.match(line)
    if a and stack and a.group(2) in todo.get(stack[-1][0], {}):
      decl, value = todo[stack[-1][0]].pop(a.group(2))
      out.append(f'{a.group(1)}{decl} {a.group(2)} = {value}')
      continue
    out.append(line)
  left = {p: a for p, a in todo.items() if a}
  if left:
    raise RuntimeError(f'not found in the text: {left}')
  return out


def block_start(out, indent):
  """Where in `out` the innermost open block (its '{' at `indent`) begins: after that line."""
  for i in range(len(out) - 1, -1, -1):
    if out[i].strip() == '{' and len(out[i]) - len(out[i].lstrip()) == indent:
      return i + 1
  raise RuntimeError('block start not found')


def shift(prim, d, edits):
  """Moves a prim by d in its parent's frame: its first op if that is a
  translate, else a translate op in front of the others."""
  order = list(prim.GetAttribute('xformOpOrder').Get() or []) if prim.GetAttribute('xformOpOrder') else []
  if order and order[0].startswith('xformOp:translate'):
    spec = strongest(prim.GetAttribute(order[0]))
    value = spec.default
    edits.set(spec.layer, spec.path.GetPrimPath(), order[0], str(spec.typeName), vec(value[i] + d[i] for i in range(3)))
    return
  where = strongest(prim.GetAttribute('xformOpOrder')) or prim.GetPrimStack()[0]
  layer, path = where.layer, where.path.GetPrimPath()
  edits.set(layer, path, FRAME_OP, 'double3', vec(d))
  edits.set(layer, path, 'xformOpOrder', 'uniform token[]', '[' + ', '.join(f'"{o}"' for o in [FRAME_OP, *order]) + ']')


def snapshot(stage, root, joints):
  """World matrices of everything under the cabinet but its links, and the
  world positions of both ends of every joint."""
  cache = UsdGeom.XformCache()
  out = {}
  for p in Usd.PrimRange(root):
    if p.IsA(UsdGeom.Xformable) and p.GetParent() != root:
      out[str(p.GetPath())] = cache.GetLocalToWorldTransform(p)
  for j in joints:
    for k in (0, 1):
      body = stage.GetPrimAtPath(j.GetRelationship(f'physics:body{k}').GetTargets()[0])
      local = Gf.Vec3d(j.GetAttribute(f'physics:localPos{k}').Get())
      out[f'{j.GetPath()}:{k}'] = cache.GetLocalToWorldTransform(body).Transform(local)
  return out


def frames(stage, root):
  return {c.GetName(): Gf.Vec3d(c.GetAttribute('xformOp:translate').Get()) for c in root.GetChildren()
          if c.GetAttribute('xformOp:translate')}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('world', help='the apartment scene (world.usda)')
  args = ap.parse_args()
  stage = Usd.Stage.Open(args.world)
  root = stage.GetPrimAtPath(CABINET)
  joints = [p for p in root.GetChild('joints').GetChildren() if p.IsA(UsdPhysics.Joint)]
  before = snapshot(stage, root, joints)

  # The new frame of each link, in its old frame: its joint's anchor, or for
  # Static the cabinet's frame (its translate undone).
  offsets = {}
  for j in joints:
    link = stage.GetPrimAtPath(j.GetRelationship('physics:body1').GetTargets()[0])
    offsets[link.GetPath()] = Gf.Vec3d(j.GetAttribute('physics:localPos1').Get())
  static = root.GetChild('Static')
  offsets[static.GetPath()] = -Gf.Vec3d(static.GetAttribute('xformOp:translate').Get())

  edits = Edits()
  for path, p in offsets.items():
    link = stage.GetPrimAtPath(path)
    if list(link.GetAttribute('xformOpOrder').Get()) != ['xformOp:translate']:
      raise RuntimeError(f'{path}: expected a translate only')
    if p.GetLength() < 1e-9:
      continue  # already where it belongs (run before)
    shift(link, p, edits)
    for child in link.GetChildren():
      if child.IsA(UsdGeom.Xformable):
        shift(child, -p, edits)
      elif any(q.IsA(UsdGeom.Xformable) for q in Usd.PrimRange(child)):
        raise RuntimeError(f'{child.GetPath()}: geometry under a prim that is not xformable')
    for j in joints:
      for k in (0, 1):
        if j.GetRelationship(f'physics:body{k}').GetTargets()[0] == path:
          spec = strongest(j.GetAttribute(f'physics:localPos{k}'))
          edits.set(spec.layer, spec.path.GetPrimPath(), f'physics:localPos{k}', str(spec.typeName),
                    vec(spec.default[i] - p[i] for i in range(3)))
    print(f'{link.GetName()}: frame moved by {tuple(round(v, 4) for v in p)}')
  if not edits.files:
    print('nothing to do: the links have their frames already')
    return

  old = edits.apply()
  for layer in stage.GetUsedLayers():
    layer.Reload(force=True)
  after = snapshot(stage, root, joints)
  worst = 0.0
  for k, m in before.items():
    n = after[k]
    worst = max(worst, max(abs(m[i][j] - n[i][j]) for i in range(4) for j in range(4)) if isinstance(m, Gf.Matrix4d)
                else (m - n).GetLength())
  if worst > 1e-6 or len(after) != len(before):
    for path, text in old.items():
      with open(path, 'w') as f:
        f.write(text)
    raise RuntimeError(f'something moved in the world ({worst}): the files are put back')
  print(f'checked {len(before)} prims and joint ends: none moved (max {worst:.1e})')
  for name, t in frames(stage, root).items():
    print(f'  {name}: translate {tuple(round(v, 4) for v in t)}')
  for path in old:
    print('wrote', path)


if __name__ == '__main__':
  main()
