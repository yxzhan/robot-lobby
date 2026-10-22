"""Converts a ROS robot description (URDF / xacro) into the model library as
MJCF, set up for the viewer. What is particular to a robot is in a TOML file
next to this script (scripts/urdf2mjcf/<robot>.toml, documented in
pr2.toml); the rest is generic:

  - xacro is run with a throwaway ament index of the ROS packages found
    under the directories given (and the installed ones), so $(find ...) and
    package:// URIs resolve to them (ROS 2's xacro: source
    /opt/ros/<distro>/setup.bash first);
  - every URDF link becomes a body of the same name (fixed links too, so the
    TF frames match the real robot's); `drop` leaves links out with
    everything below them (gazebo-only mechanism models), `weld` makes
    joints fixed (casters, springs);
  - Collada visuals become OBJ, one per material, with the color map as a
    PNG (a single-color map becomes a plain rgba); STL is copied; MuJoCo
    takes the convex hulls of collision meshes;
  - inertias come from the URDF (rotated into the link frame); a moving body
    without one gets its mass from its collision geoms;
  - URDF mimic joints become joint equalities; when the joint they follow is
    left out, the first follower leads instead;
  - position servos by `servo` rules (gains, armature), a planar base (x /
    y / yaw servos on the root, as on TidyBot) when `base.kind = "planar"`
    (a free joint by default, none when "fixed"),
    gravity compensation from the `gravcomp` bodies down;
  - MJCF cameras where the Gazebo camera / depth sensors are, looking along
    the reference link's +x like Gazebo, with their field of view, and
    where `camera` puts them;
  - TCP sites for the viewer's IK (`tcp`);
  - contact excludes for neighbouring bodies (at most 3 joints apart) whose
    collision hulls overlap at home or in many random poses, as the
    overlapping hulls of an arm's links do;
  - a `home` keyframe (the viewer spawns a robot in its first keyframe).

Needs pycollada, numpy, Pillow and mujoco.

    python3 scripts/urdf2mjcf.py scripts/urdf2mjcf/pr2.toml /path/to/pr2_common [more package dirs…]
    node scripts/vendor.mjs   # re-index
    node scripts/render_previews.mjs robots/digital_twin/pr2   # its screenshot
"""

import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET

import collada
import mujoco
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBRARY = ROOT / 'mjcf_assets'
EXCLUDE_HOPS = 3        # neighbours: bodies at most this many joints apart
EXCLUDE_SHARE = 0.1     # excluded when their hulls touch in this share of random poses
SAMPLES = 300
MAX_FACES = 50000       # meshes with more triangles are simplified (max_faces)
KEYS = {   # what a config may hold, by table ('' the top level)
  '': ['name', 'out', 'source', 'package', 'xacro', 'root', 'root_pos', 'root_rpy', 'xacro_args', 'title', 'readme', 'license',
       'gravcomp', 'armature', 'exclude', 'collide', 'links', 'joints', 'base', 'servo', 'cameras', 'camera', 'tcp',
       'colors', 'max_faces',
       'home'],
  'links': ['drop', 'no_collision'],
  'joints': ['weld'],
  'base': ['kind', 'kp_xy', 'kp_yaw', 'damping'],
  'cameras': ['max_width', 'names'],
}


def fmt(v):
  return ' '.join(f'{x:.6g}' if abs(x) > 1e-15 else '0' for x in v)


def rpy_matrix(r, p, y):
  cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
  return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]])


def quat(m):
  w = math.sqrt(max(0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
  x = math.sqrt(max(0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
  y = math.sqrt(max(0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
  z = math.sqrt(max(0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
  x, y, z = math.copysign(x, m[2, 1] - m[1, 2]), math.copysign(y, m[0, 2] - m[2, 0]), math.copysign(z, m[1, 0] - m[0, 1])
  return [w, x, y, z]


def origin(el):
  """pos, rotation matrix of an element's <origin>."""
  o = el.find('origin') if el is not None else None
  get = lambda k: [float(v) for v in (o.get(k, '0 0 0') if o is not None else '0 0 0').split()]
  return np.array(get('xyz')), rpy_matrix(*get('rpy'))


def frame(pos, rot):
  attrs = f' pos="{fmt(pos)}"' if np.abs(pos).max() > 1e-12 else ''
  if np.abs(rot - np.eye(3)).max() > 1e-9:
    attrs += f' quat="{fmt(quat(rot))}"'
  return attrs


def read_stl(path):
  """A binary STL's triangles (n, 3, 3), None for an ASCII one."""
  data = path.read_bytes()
  n = int.from_bytes(data[80:84], 'little') if len(data) >= 84 else -1
  if len(data) != 84 + 50 * n:
    return None
  rec = np.frombuffer(data, np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), n, 84)
  return rec['v'].astype(np.float64)


def write_stl(path, tris):
  rec = np.zeros(len(tris), np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
  rec['v'] = tris
  n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
  rec['n'] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
  with open(path, 'wb') as f:
    f.write(b'scripts/urdf2mjcf.py'.ljust(80, b' '))
    f.write(len(tris).to_bytes(4, 'little'))
    f.write(rec.tobytes())


def matches(patterns, name):
  return any(re.search(p, name) for p in patterns)


def find_packages(roots):
  """ROS packages {name: dir}: those under `roots` (the shallowest of a name
  wins), then the installed ones (AMENT_PREFIX_PATH's share/)."""
  found = {}
  for root in roots:
    for xml in sorted(pathlib.Path(root).rglob('package.xml'), key=lambda p: len(p.parts)):
      if any(part.startswith('.') or part == 'node_modules' for part in xml.parts):
        continue
      name = ET.parse(xml).getroot().findtext('name')
      if name:
        found.setdefault(name.strip(), xml.parent.resolve())
  for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep):
    share = pathlib.Path(prefix, 'share')
    if prefix and share.is_dir():
      for d in share.iterdir():
        if (d / 'package.xml').exists():
          found.setdefault(d.name, d)
  return found


def xacro(packages, cfg):
  """The expanded URDF, with a throwaway ament index for $(find <package>)."""
  with tempfile.TemporaryDirectory() as tmp:
    index = pathlib.Path(tmp, 'share', 'ament_index', 'resource_index', 'packages')
    index.mkdir(parents=True)
    for name, d in packages.items():
      (index / name).touch()
      os.symlink(d, pathlib.Path(tmp, 'share', name))
    env = dict(os.environ, AMENT_PREFIX_PATH=os.pathsep.join([tmp, os.environ.get('AMENT_PREFIX_PATH', '')]),
               ROS_PACKAGE_PATH=os.pathsep.join([str(d.parent) for d in packages.values()]))
    find = lambda v: re.sub(r'\$\(find (\w+)\)', lambda m: str(packages[m.group(1)]), v)
    args = [f'{k}:={find(v)}' for k, v in cfg.get('xacro_args', {}).items()]
    try:
      run = subprocess.run(['xacro', str(packages[cfg['package']] / cfg['xacro']), *args], env=env,
                           capture_output=True, text=True)
    except FileNotFoundError:
      raise SystemExit('no xacro: source a ROS setup.bash first')
    if run.returncode:
      raise SystemExit(f'xacro failed:\n{run.stderr}')
    return run.stdout


class Converter:
  def __init__(self, cfg, urdf, packages, out):
    self.cfg, self.packages, self.out = cfg, packages, out
    self.robot = ET.fromstring(urdf)
    self.links = {l.get('name'): l for l in self.robot.findall('link')}
    self.children = {}
    for j in self.robot.findall('joint'):
      self.children.setdefault(j.find('parent').get('link'), []).append(j)
    self.urdf_colors = {m.get('name'): m.find('color').get('rgba') for m in self.robot.findall('material')
                        if m.find('color') is not None}
    self.meshes = {}       # mesh name -> file
    self.visuals = {}      # (mesh file, scale) -> [(mesh name, material or None)]
    self.materials = {}    # material name -> ('texture', png) or ('rgba', rgba)
    self.textures = {}     # image path -> material name
    self.joints = []       # (name, kind, range), in body order
    self.mimic = {}        # follower -> (leader, multiplier, offset)
    self.kept = set()      # links that are bodies
    self.cameras = self.gazebo_cameras()
    self.stems = {}        # (stem, extension) -> the mesh files with it
    for el in self.robot.iter('mesh'):
      path = self.mesh_path(el.get('filename'))
      self.stems.setdefault((path.stem.lower(), path.suffix.lower()), set()).add(path)

  # --- meshes and materials -------------------------------------------------

  def mesh_path(self, uri):
    m = re.match(r'package://([^/]+)/(.*)', uri)
    if m:
      if m.group(1) not in self.packages:
        raise SystemExit(f'mesh {uri}: no package {m.group(1)}')
      return self.packages[m.group(1)] / m.group(2)
    if uri.startswith('file://'):
      return pathlib.Path(uri[len('file://'):])
    raise SystemExit(f'mesh {uri}: not a package:// or file:// URI')

  def asset_name(self, path):
    """A unique name for a mesh file: its stem, with its directory if needed."""
    stem = path.stem.lower()
    return stem if len(self.stems[stem, path.suffix.lower()]) == 1 else f'{path.parent.name.lower()}_{stem}'

  def material_for_image(self, image):
    if image in self.textures:
      return self.textures[image]
    px = np.asarray(Image.open(image).convert('RGB'))
    if np.ptp(px.reshape(-1, 3), axis=0).max() <= 2:   # one color: no texture needed
      rgba = fmt(list(px.reshape(-1, 3).mean(0) / 255) + [1])
      name = 'rgba_' + ''.join(f'{int(round(float(c) * 255)):02x}' for c in rgba.split()[:3])
      self.materials[name] = ('rgba', rgba)
    else:
      name = f'{image.parent.name}_{image.stem}'.lower()
      Image.open(image).convert('RGB').save(self.out / 'assets' / f'{name}.png')
      self.materials[name] = ('texture', f'{name}.png')
    self.textures[image] = name
    return name

  def dae(self, path, visual):
    """OBJ files of a Collada mesh in its link frame, one per material: [(file, material or None)]."""
    dae = collada.Collada(str(path), ignore=[collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError])
    scale = dae.assetInfo.unitmeter or 1.0
    groups = {}   # material key -> (image or rgba, [triangle sets])
    for geom in dae.scene.objects('geometry'):
      for prim in geom.primitives():
        if isinstance(prim, collada.lineset.BoundLineSet):
          continue
        tris = prim if isinstance(prim, collada.triangleset.BoundTriangleSet) else prim.triangleset()
        mat, key, look = tris.material, None, None
        if visual and mat is not None:
          d = mat.effect.diffuse
          if hasattr(d, 'sampler'):
            look = ('image', path.parent / pathlib.Path(d.sampler.surface.image.path).name)
          elif d is not None:
            look = ('rgba', fmt(list(d[:3]) + [1]))
          key = look
        groups.setdefault(key, (look, []))[1].append(tris)
    out = []
    base = self.asset_name(path) + ('' if visual else '_collision')
    for k, (look, sets) in enumerate(groups.values()):
      name = base if len(groups) == 1 else f'{base}_{k}'
      textured = look is not None and look[0] == 'image'
      if not textured:
        # Untextured: binary STL (MuJoCo works out the normals), simplified if big.
        tris = np.concatenate([t.vertex[t.vertex_index] * scale for t in sets])
        file = f'{name}.stl'
        write_stl(self.out / 'assets' / file, self.simplify(tris, path.name))
      else:
        file = f'{name}.obj'
        self.write_obj(path, file, sets, scale)
      material = None
      if look is not None:
        if textured:
          material = self.material_for_image(look[1])
        else:
          material = 'rgba_' + ''.join(f'{int(round(float(c) * 255)):02x}' for c in look[1].split()[:3])
          self.materials[material] = ('rgba', look[1])
      out.append((file, material))
    return out

  def simplify(self, tris, what):
    """Triangles (n, 3, 3), cut down to max_faces."""
    limit = self.cfg.get('max_faces', MAX_FACES)
    if len(tris) <= limit:
      return tris
    import fast_simplification
    points, inverse = np.unique(tris.reshape(-1, 3).round(7), axis=0, return_inverse=True)
    points, faces = fast_simplification.simplify(points.astype(np.float32), inverse.reshape(-1, 3).astype(np.int32),
                                                 target_reduction=1 - limit / len(tris))
    print(f'{what}: {len(tris)} triangles simplified to {len(faces)}')
    return points[faces]

  def write_obj(self, path, file, sets, scale):
    """Textured triangle sets as OBJ, with their texcoords and normals (only
    the vertices they use: a Collada source can be shared by many)."""
    v, vt, vn = [], [], []
    for tris in sets:
      v.append(tris.vertex[tris.vertex_index].reshape(-1, 3) * scale)
      vt.append(tris.texcoordset[0][tris.texcoord_indexset[0]].reshape(-1, 2) if len(tris.texcoordset)
                else np.zeros((3 * len(tris.vertex_index), 2)))
      vn.append(tris.normal[tris.normal_index].reshape(-1, 3) if tris.normal is not None
                else np.repeat(np.cross(*(np.diff(tris.vertex[tris.vertex_index], axis=1).transpose(1, 0, 2))), 3, axis=0))
    corners = np.hstack([np.concatenate(v), np.concatenate(vt), np.concatenate(vn)]).round(6)
    unique, index = np.unique(corners, axis=0, return_inverse=True)   # corners shared by triangles
    with open(self.out / 'assets' / file, 'w') as f:
      f.write(f'# {path.name}, converted by scripts/urdf2mjcf.py\n')
      f.writelines(f'v {fmt(c[:3])}\n' for c in unique)
      f.writelines(f'vt {fmt(c[3:5])}\n' for c in unique)
      f.writelines(f'vn {fmt(c[5:])}\n' for c in unique)
      f.writelines('f ' + ' '.join(f'{i}/{i}/{i}' for i in t) + '\n' for t in index.reshape(-1, 3) + 1)
    if len(corners) // 3 > self.cfg.get('max_faces', MAX_FACES):
      print(f'warning: {path.name}: {len(corners) // 3} textured triangles, not simplified')

  def mesh(self, uri, scale, visual):
    """[(MJCF mesh name, material or None)] for a URDF mesh."""
    path = self.mesh_path(uri)
    key = (path, scale, visual)
    if key not in self.visuals:
      scaled = '' if scale == '1 1 1' else '_' + re.sub(r'[^0-9]+', '_', scale).strip('_')
      suffix = ('' if visual else '_collision') + scaled
      if path.suffix.lower() == '.dae':
        files = self.dae(path, visual)
      else:
        file = f'{self.asset_name(path)}{path.suffix.lower()}'
        tris = read_stl(path) if path.suffix.lower() == '.stl' else None
        if tris is not None and len(tris) > self.cfg.get('max_faces', MAX_FACES):
          write_stl(self.out / 'assets' / file, self.simplify(tris, path.name))
        else:
          shutil.copy2(path, self.out / 'assets' / file)
        files = [(file, None)]
      parts = []
      for file, material in files:
        # A Collada mesh's files are named for visual / collision already.
        name = pathlib.Path(file).stem + (scaled if path.suffix.lower() == '.dae' else suffix)
        self.meshes[name] = (file, scale, visual)
        parts.append((name, material))
      self.visuals[key] = parts
    return self.visuals[key]

  def urdf_material(self, el):
    m = el.find('material')
    if m is None:
      return None
    name = m.get('name', '')
    color = m.find('color')
    rgba = color.get('rgba') if color is not None else self.urdf_colors.get(name)
    rgba = self.cfg.get('colors', {}).get(name, rgba) or '0.5 0.5 0.5 1'
    mat = 'urdf_' + re.sub(r'\W', '_', name or rgba)
    self.materials[mat] = ('rgba', fmt([float(c) for c in rgba.split()]))
    return mat

  def geoms(self, link, ind):
    out = []
    name = link.get('name')
    for kind in ('visual', 'collision'):
      if kind == 'collision' and matches(self.cfg['links'].get('no_collision', []), name):
        continue
      for el in link.findall(kind):
        g = el.find('geometry/*')
        pos, rot = origin(el)
        base = f'class="{kind}"{frame(pos, rot)}'
        if g.tag == 'mesh':
          for mesh, mat in self.mesh(g.get('filename'), g.get('scale', '1 1 1'), kind == 'visual'):
            attrs = f'{base} type="mesh" mesh="{mesh}"'
            if kind == 'visual':
              mat = mat or self.urdf_material(el) or self.default_material()
              attrs += f' material="{mat}"'
            out.append(f'{ind}<geom {attrs}/>')
          continue
        if g.tag == 'box':
          base += f' type="box" size="{fmt([float(s) / 2 for s in g.get("size").split()])}"'
        elif g.tag == 'cylinder':
          base += f' type="cylinder" size="{g.get("radius")} {float(g.get("length")) / 2:.6g}"'
        elif g.tag == 'sphere':
          base += f' type="sphere" size="{g.get("radius")}"'
        else:
          raise SystemExit(f'{name}: {g.tag} geometry')
        if kind == 'visual':
          base += f' material="{self.urdf_material(el) or self.default_material()}"'
        out.append(f'{ind}<geom {base}/>')
    return out

  def default_material(self):
    self.materials['urdf_default'] = ('rgba', '0.5 0.5 0.5 1')
    return 'urdf_default'

  # --- bodies ----------------------------------------------------------------

  def inertial(self, link, ind, moving):
    el = link.find('inertial')
    if el is None:
      has_collision = link.find('collision') is not None and not matches(self.cfg['links'].get('no_collision', []), link.get('name'))
      # A moving body needs mass: from its collision geoms, else a token one.
      return [] if not moving or has_collision else [f'{ind}<inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>']
    mass = float(el.find('mass').get('value'))
    pos, rot = origin(el)
    i = {k: float(v) for k, v in el.find('inertia').attrib.items()}
    tensor = np.array([[i['ixx'], i['ixy'], i['ixz']], [i['ixy'], i['iyy'], i['iyz']], [i['ixz'], i['iyz'], i['izz']]])
    t = rot @ tensor @ rot.T
    w, v = np.linalg.eigh(t)
    if w.min() <= 0:   # not a physical inertia: its principal moments made positive
      print(f'warning: {link.get("name")}: inertia with principal moments {fmt(w)}, made positive')
      w = np.maximum(w, max(w.max() * 1e-3, 1e-8))
      t = v @ np.diag(w) @ v.T
    full = [t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]]
    return [f'{ind}<inertial pos="{fmt(pos)}" mass="{mass:.6g}" fullinertia="{fmt(full)}"/>']

  def servo(self, joint):
    for rule in self.cfg.get('servo', []):
      if re.search(rule['match'], joint):
        return rule
    return None

  def body(self, link, joint, ind, gravcomp=False):
    name = link.get('name')
    self.kept.add(name)
    gravcomp = gravcomp or name in self.cfg.get('gravcomp', [])
    lines, moving = [], False
    if joint is None:   # the root
      pos, rpy = self.cfg.get('root_pos', [0, 0, 0]), self.cfg.get('root_rpy', [0, 0, 0])
      lines.append(f'{ind}<body name="{name}" childclass="robot"{frame(np.array(pos, float), rpy_matrix(*rpy))}>')
      if self.cfg.get('base', {}).get('kind') == 'planar':
        d = self.cfg['base'].get('damping', [100, 50])
        lines.append(f'{ind}  <joint name="base_x" type="slide" axis="1 0 0" damping="{d[0]}"/>')
        lines.append(f'{ind}  <joint name="base_y" type="slide" axis="0 1 0" damping="{d[0]}"/>')
        lines.append(f'{ind}  <joint name="base_yaw" type="hinge" axis="0 0 1" damping="{d[1]}"/>')
        self.joints += [(j, 'base', None) for j in ('base_x', 'base_y', 'base_yaw')]
        moving = True
      elif self.cfg.get('base', {}).get('kind', 'free') == 'free':
        lines.append(f'{ind}  <freejoint name="root"/>')
        moving = True
    else:
      pos, rot = origin(joint)
      grav = ' gravcomp="1"' if gravcomp else ''
      lines.append(f'{ind}<body name="{name}"{frame(pos, rot)}{grav}>')
      kind, jname = joint.get('type'), joint.get('name')
      if kind not in ('fixed', 'floating', 'planar') and not matches(self.cfg['joints'].get('weld', []), jname):
        moving = True
        axis = [float(v) for v in (joint.find('axis').get('xyz') if joint.find('axis') is not None else '1 0 0').split()]
        lim = joint.find('limit')
        rng = (float(lim.get('lower')), float(lim.get('upper'))) if kind in ('revolute', 'prismatic') else None
        dyn = joint.find('dynamics')
        damping = float(dyn.get('damping', 0)) if dyn is not None else 0
        frictionloss = float(dyn.get('friction', 0)) if dyn is not None else 0
        mim = joint.find('mimic')
        if mim is not None:
          self.mimic[jname] = (mim.get('joint'), float(mim.get('multiplier', 1)), float(mim.get('offset', 0)))
        rule = self.servo(jname)
        armature = rule.get('armature', 0) if rule else self.cfg.get('armature', 0)
        attrs = f'name="{jname}" type="{"slide" if kind == "prismatic" else "hinge"}" axis="{fmt(axis)}"'
        if rng:
          attrs += f' range="{fmt(rng)}"'
        if damping:
          attrs += f' damping="{damping:.6g}"'
        if frictionloss:
          attrs += f' frictionloss="{frictionloss:.6g}"'
        if armature:
          attrs += f' armature="{armature:.6g}"'
        lines.append(f'{ind}  <joint {attrs}/>')
        self.joints.append((jname, kind, rng))
    lines += self.inertial(link, ind + '  ', moving)
    lines += self.geoms(link, ind + '  ')
    for cam in self.cameras:
      if cam['link'] == name:
        # Looking along the link's +x with +z up, as a Gazebo camera, or
        # optical frames (ROS: z forward, y down) look along +z.
        axes = '1 0 0 0 -1 0' if cam.get('optical') else '0 -1 0 0 0 1'
        lines.append(f'{ind}  <camera name="{cam["name"]}" xyaxes="{axes}" fovy="{cam["fovy"]:.4g}" '
                     f'resolution="{cam["width"]} {cam["height"]}"/>')
    tcp = self.cfg.get('tcp', {})
    for j in self.children.get(name, []):
      child = j.find('child').get('link')
      if child in tcp:
        pos, rot = origin(j)
        lines.append(f'{ind}  <site name="{tcp[child]}"{frame(pos, rot)}/>')
    for j in self.children.get(name, []):
      child = j.find('child').get('link')
      if not matches(self.cfg['links'].get('drop', []), child):
        lines += self.body(self.links[child], j, ind + '  ', gravcomp)
    lines.append(f'{ind}</body>')
    return lines

  def gazebo_cameras(self):
    """Cameras of the Gazebo camera / depth sensors: [{name, link, fovy, width, height}]."""
    keep = self.cfg.get('cameras', {})
    names, max_width = keep.get('names'), keep.get('max_width', 640)
    out = []
    for g in self.robot.findall('gazebo'):
      for s in g.findall('sensor'):
        if s.get('type') not in ('camera', 'depth'):
          continue
        for c in s.iter('camera'):
          sensor = s.get('name')
          if names is not None and sensor not in names:
            continue
          hfov = float(c.findtext('horizontal_fov'))
          w, h = int(c.findtext('image/width')), int(c.findtext('image/height'))
          k = min(1, max_width / w)
          out.append({'name': names[sensor] if names else sensor, 'link': g.get('reference'),
                      'fovy': math.degrees(2 * math.atan(math.tan(hfov / 2) * h / w)),
                      'width': round(w * k), 'height': round(h * k)})
    for c in self.cfg.get('camera', []):   # given by hand
      out.append({'name': c['name'], 'link': c['link'], 'fovy': c.get('fovy', 45), 'optical': c.get('optical', False),
                  'width': c.get('width', 640), 'height': c.get('height', 480)})
    return out

  # --- the model -------------------------------------------------------------

  def leaders(self):
    """Mimic joints as (follower, leader, multiplier, offset), followers of a
    left-out joint re-expressed in terms of the first of them."""
    kept = {j for j, *_ in self.joints}
    out = {}
    for f, (l, m, o) in self.mimic.items():
      if f not in kept:
        continue
      if l not in kept:
        group = [g for g, (gl, *_) in self.mimic.items() if gl == l and g in kept]
        new = group[0]
        if new == f:
          continue   # the new leader
        _, ml, ol = self.mimic[new]
        # f = m q + o, new = ml q + ol  =>  f = m/ml new + o - m ol / ml
        l, m, o = new, m / ml, o - m * ol / ml
      out[f] = (l, m, o)
    return out

  def actuators(self, followers):
    home = self.cfg.get('home', {})
    base = self.cfg.get('base', {})
    acts = []   # (name, joint, attrs, ctrl at home)
    for j, kind, rng in self.joints:
      if j in followers:
        continue
      if kind == 'base':
        kp = base.get('kp_yaw', 20000) if j == 'base_yaw' else base.get('kp_xy', 100000)
        acts.append((j, j, f'kp="{kp}" dampratio="1"', 0))
        continue
      rule = self.servo(j)
      if not rule:
        continue
      name = re.sub(rule['match'], rule['name'], j) if 'name' in rule else j.removesuffix('_joint')
      attrs = f'kp="{rule["kp"]}" dampratio="{rule.get("dampratio", 1)}"'
      if rng:
        attrs += f' ctrlrange="{fmt(rng)}"'
      if 'forcerange' in rule:
        attrs += f' forcerange="-{rule["forcerange"]} {rule["forcerange"]}"'
      ctrl = min(max(home.get(j, 0), rng[0]), rng[1]) if rng else home.get(j, 0)
      acts.append((name, j, attrs, ctrl))
    return acts

  def mjcf(self, exclude=()):
    roots = set(self.links) - {j.find('child').get('link') for j in self.robot.findall('joint')}
    (root,) = [self.cfg['root']] if 'root' in self.cfg else roots
    self.joints, self.mimic = [], {}
    body = self.body(self.links[root], None, '    ')
    followers = self.leaders()
    acts = self.actuators(followers)
    home = self.cfg.get('home', {})
    ranges = {j: rng for j, _, rng in self.joints}
    clamp = lambda j, v: min(max(v, ranges[j][0]), ranges[j][1]) if ranges.get(j) else v
    value = {j: clamp(j, home.get(j, 0)) for j, *_ in self.joints if j not in followers}
    for f, (l, m, o) in followers.items():
      value[f] = value[l] * m + o
    qpos = [value[j] for j, *_ in self.joints]
    assets = []
    for name, (kind, v) in sorted(self.materials.items()):
      if kind == 'texture':
        assets.append(f'    <texture name="{name}" type="2d" file="{v}"/>')
        assets.append(f'    <material name="{name}" texture="{name}" specular="0.3" shininess="0.4"/>')
      else:
        assets.append(f'    <material name="{name}" rgba="{v}" specular="0.3" shininess="0.4"/>')
    for name, (file, scale, visual) in sorted(self.meshes.items()):
      sc = f' scale="{scale}"' if scale != '1 1 1' else ''
      # Visual meshes weigh nothing: shell, which a flat mesh has too.
      inertia = ' inertia="shell"' if visual else ''
      assets.append(f'    <mesh name="{name}" file="{file}"{sc}{inertia}/>')
    eq = [f'    <joint joint1="{f}" joint2="{l}" polycoef="{o:.6g} {m:.6g} 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>'
          for f, (l, m, o) in followers.items()]
    cfg = self.cfg
    return '\n'.join([
      f'<!-- Converted by scripts/urdf2mjcf.py (scripts/urdf2mjcf/{cfg["name"]}.toml) from',
      f'     {cfg["package"]}/{cfg["xacro"]} ({cfg["source"]}). Body names are the URDF links. -->',
      f'<mujoco model="{cfg["name"]}">',
      '  <compiler angle="radian" meshdir="assets" texturedir="assets" autolimits="true" inertiafromgeom="auto"'
      ' balanceinertia="true"/>',
      '',
      '  <option integrator="implicitfast"/>',
      '',
      '  <default>',
      '    <mesh inertia="convex"/>',  # no volume errors for open meshes (only links without an inertial use it)
      '    <default class="robot">',
      '      <default class="visual">',
      '        <geom group="2" contype="0" conaffinity="0" density="0"/>',
      '      </default>',
      '      <default class="collision">',
      '        <geom group="3"/>',
      '      </default>',
      '    </default>',
      '  </default>',
      '',
      '  <asset>',
      *assets,
      '  </asset>',
      '',
      '  <worldbody>',
      *body,
      '  </worldbody>',
      '',
      *(['  <contact>', *[f'    <exclude body1="{a}" body2="{b}"/>' for a, b in exclude], '  </contact>', '']
        if exclude else []),
      *(['  <equality>', *eq, '  </equality>', ''] if eq else []),
      '  <actuator>',
      *[f'    <position name="{n}" joint="{j}" {a}/>' for n, j, a, _ in acts],
      '  </actuator>',
      '',
      '  <keyframe>',
      f'    <key name="home" qpos="{fmt(qpos)}"',
      f'         ctrl="{fmt([c for *_, c in acts])}"/>',
      '  </keyframe>',
      '</mujoco>',
      '',
    ])


def find_excludes(path, extra, keep):
  """Body pairs to exclude from contact: neighbours whose hulls overlap at home
  or in many random poses (see EXCLUDE_*) but not those in `keep` (real
  collisions), plus `extra`. Prints the pairs that still touch at home."""
  m = mujoco.MjModel.from_xml_path(str(path))
  d = mujoco.MjData(m)
  weld = m.body_weldid
  def hops(a, b):
    """Joints between two weld groups along the tree (None if far apart)."""
    up = lambda x: weld[m.body_parentid[x]]
    path_a, x = [weld[a]], weld[a]
    while x:
      x = up(x); path_a.append(x)
    x, n = weld[b], 0
    while x not in path_a:
      x = up(x); n += 1
    return n + path_a.index(x)
  def touching():
    mujoco.mj_forward(m, d)
    pairs = set()
    for c in d.contact[:d.ncon]:
      a, b = sorted((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]))
      pairs.add((a, b))
    return pairs
  mujoco.mj_resetDataKeyframe(m, d, 0)
  at_home = touching()
  counts = {}
  rng = np.random.default_rng(0)
  for _ in range(SAMPLES):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for j in range(m.njnt):
      if m.jnt_type[j] in (2, 3) and m.jnt_bodyid[j] != 1:
        lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi)
        d.qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
    for p in touching():
      counts[p] = counts.get(p, 0) + 1
  name = lambda b: m.body(b).name
  out = set()
  for p in at_home | {p for p, n in counts.items() if n >= EXCLUDE_SHARE * SAMPLES}:
    pair = tuple(name(b) for b in p)
    if hops(*p) <= EXCLUDE_HOPS and pair not in keep and pair[::-1] not in keep:
      out.add(pair)
  out |= {tuple(p) for p in extra}
  for p in sorted(at_home):
    if tuple(name(b) for b in p) not in out:
      print(f'warning: {name(p[0])} and {name(p[1])} touch at home')
  return sorted(out)


def main():
  if len(sys.argv) < 3:
    raise SystemExit(__doc__)
  cfg = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
  for table, keys in KEYS.items():
    unknown = set(cfg.get(table, {}) if table else cfg) - set(keys)
    if unknown:
      raise SystemExit(f'{sys.argv[1]}: unknown keys {sorted(unknown)} in {f"[{table}]" if table else "the top level"}'
                       ' (top-level keys go before the first [table])')
  cfg.setdefault('links', {}); cfg.setdefault('joints', {})
  packages = find_packages(sys.argv[2:])
  pkg_dir = packages.get(cfg['package'])
  if not pkg_dir or not (pkg_dir / cfg['xacro']).exists():
    raise SystemExit(f'no {cfg["package"]}/{cfg["xacro"]} under {sys.argv[2:]}: pass the path of a {cfg["source"]} clone')
  urdf = xacro(packages, cfg)
  out = LIBRARY / cfg['out']
  shutil.rmtree(out / 'assets', ignore_errors=True)
  (out / 'assets').mkdir(parents=True)
  conv = Converter(cfg, urdf, packages, out)
  xml = out / f'{cfg["name"]}.xml'
  xml.write_text(conv.mjcf())
  xml.write_text(conv.mjcf(find_excludes(xml, cfg.get('exclude', []), {tuple(p) for p in cfg.get('collide', [])})))
  sha = subprocess.run(['git', '-C', str(pkg_dir), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
  (out / 'asset.json').write_text(json.dumps(
    {'source': cfg['source'], 'file': f'{cfg["package"]}/{cfg["xacro"]}', 'commit': sha}, indent=2) + '\n')
  (out / 'README.md').write_text(f'# {cfg["title"]}\n\n{cfg["readme"].strip()}\n\n## License\n\n{cfg["license"].strip()}\n'
                                 f'Converted by `scripts/urdf2mjcf.py` with `scripts/urdf2mjcf/{cfg["name"]}.toml`.\n')
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  mujoco.mj_forward(m, d)
  def bottom(g):   # of a collision geom at home
    R = d.geom_xmat[g].reshape(3, 3)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      i = m.geom_dataid[g]
      return (d.geom_xpos[g] + m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]] @ R.T)[:, 2].min()
    return d.geom_xpos[g][2] - np.abs(R[2]) @ m.geom_size[g] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX \
      else d.geom_xpos[g][2] - m.geom_rbound[g]
  low = min((bottom(g) for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]), default=0)
  if cfg.get('base', {}).get('kind', 'free') != 'fixed' and low < -0.005:
    print(f'warning: collision geoms reach {-low:.3f} m below the floor at home (root_pos, no_collision)')
  print(f'{out.relative_to(ROOT)}: {len(conv.meshes)} meshes, {m.njnt} joints, {m.nu} actuators, '
        f'{m.ncam} cameras, {m.neq} equalities, {m.body_subtreemass[0]:.1f} kg')


if __name__ == '__main__':
  main()
