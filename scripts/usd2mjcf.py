"""Converts a USD scene (as exported from Unreal through Omniverse: meshes with
UsdPreviewSurface materials, UCX_* collision meshes, UsdPhysics rigid bodies
and joints) into the model library as an MJCF environment:

  - every child of the default prim is a body of its name (a static one,
    welded to the world), and so is every rigid body; a rigid body that is
    body1 of a revolute / prismatic joint hangs from body0's body on a hinge /
    slide at the joint's frame (its range from the joint's limits), one that
    is in no joint gets a free joint (furniture, a cup);
  - visible meshes become mesh geoms in their body's frame (the transforms,
    scale included, baked into the vertices), split by GeomSubset material:
    OBJ with the texture coordinates when the material's diffuse color is a
    texture, else binary STL; a texture of one color becomes a plain rgba;
  - collision: a UCX box (8 points) becomes a box geom, other collision
    meshes their convex hull (boundingCube: the box around it;
    convexDecomposition / meshSimplification, concave: boxes over 5 cm
    voxels of its surface, or its plane if thin and wide); a visible
    mesh with no collision geom of its own around it gets boxes: a thin, wide
    one (a wall, a floor) the boxes that cover it in its plane (5 cm cells,
    so door and window openings stay open), anything else its bounding box
    in its body's frame (boxes over 5 cm voxels of its surface instead when
    that box would hold something that moves: an armchair in a partition's);
  - a rigid body's physics:mass is spread over its collision geoms by volume;
  - contacts of a jointed body with the bodies it touches at rest (a drawer
    in its cabinet) are excluded; doors on horizontal hinges are gravity
    compensated (their springs), and joints get some friction.

    python3 scripts/usd2mjcf.py scene.usda mjcf_assets/environments/<source>/<name> [--offset 0 0 0.07]
    node scripts/vendor.mjs   # re-index

Needs pxr (usd-core), numpy, Pillow, mujoco; fast_simplification for meshes
over --max-faces.
"""

import argparse
import math
import pathlib
import re
import shutil
import sys

import mujoco
import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from urdf2mjcf import fmt, quat, write_stl  # noqa: E402

CELL = 0.05          # m, the grid thin meshes are covered with boxes on
FINE = 0.01          # m, the voxels of a mesh whose 5 cm ones would hold something that moves
THIN = 0.25          # m: thinner than this, and
WIDE = 1.0           # wider than this in both other directions: a wall or floor
MIN_HALF = 0.01      # m, the thinnest half size of a generated box
DENSITY = 300        # kg/m³ of a rigid body without physics:mass


def name_of(path):
  return re.sub(r'\W', '_', str(path).rsplit('/', 1)[-1])


def srgb(c):
  """A linear color component in sRGB."""
  c = min(max(float(c), 0.0), 1.0)
  return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def moving(prim):
  """A rigid body that moves (not kinematic, not disabled)."""
  if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
    return False
  rb = UsdPhysics.RigidBodyAPI(prim)
  return rb.GetRigidBodyEnabledAttr().Get() is not False and not rb.GetKinematicEnabledAttr().Get()


def collides(prim):
  return prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False


def matrix(prim, cache):
  """A prim's local-to-world transform as a 4x4 column-vector matrix."""
  return np.array(cache.GetLocalToWorldTransform(prim)).T


def pose(m):
  """(rotation, translation) of a transform, its scale left out."""
  r = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
  u, _, vt = np.linalg.svd(r)
  return u @ vt, m[:3, 3].copy()


def box_of(points):
  """A box (center, half sizes, rotation) through 8 points that form one, else None."""
  if len(points) != 8:
    return None
  c = points.mean(0)
  d = points - c
  # The box's axes: from a corner to the nearest corners, each square to the others.
  axes = []
  for v in sorted(points[1:] - points[0], key=np.linalg.norm):
    v = v / max(np.linalg.norm(v), 1e-12)
    if all(abs(v @ x) < 1e-3 for x in axes):
      axes.append(v)
  if len(axes) != 3:
    return None
  r = np.array(axes).T
  if abs(abs(np.linalg.det(r)) - 1) > 1e-3:
    return None
  if np.linalg.det(r) < 0:
    r[:, 2] *= -1
  local = d @ r
  half = np.abs(local).max(0)
  if np.abs(np.abs(local) - half).max() > 1e-4 + 1e-3 * half.max():
    return None
  return c, half, r


class Converter:
  def __init__(self, stage, out, offset, max_faces):
    self.stage, self.out, self.offset, self.max_faces = stage, out, np.array(offset), max_faces
    self.cache = UsdGeom.XformCache()
    self.root = stage.GetDefaultPrim()
    self.materials = {}    # key -> (name, kind, value, alpha)
    self.textures = {}     # image path -> (texture name, uniform rgb or None)
    self.meshes = []       # (name, file)
    self.names = set()
    self.voxelize = set()  # meshes whose bounding box would hold something that moves: boxes over voxels
    self.fine = set()      # meshes whose 5 cm voxel boxes would: 1 cm ones
    self.no_collision = None  # regex: bodies whose meshes get no generated collision geoms

  def unique(self, name):
    base, k = name, 1
    while name in self.names:
      name = f'{base}_{k}'
      k += 1
    self.names.add(name)
    return name

  # --- the body tree ---------------------------------------------------------

  def owner(self, prim):
    """The body a prim belongs to: its nearest moving rigid body, else its
    topmost transformable ancestor (under scopes)."""
    p, top = prim, None
    while p and p != self.root:
      if moving(p):
        return p
      if p.IsA(UsdGeom.Xformable):
        top = p
      p = p.GetParent()
    return top

  def tops(self):
    """The topmost transformable prims under the default prim (through scopes)."""
    out = []
    it = iter(Usd.PrimRange(self.root))
    for p in it:
      if p != self.root and p.IsA(UsdGeom.Xformable):
        if any(q.IsA(UsdGeom.Gprim) for q in Usd.PrimRange(p)):   # not a light or a camera
          out.append(p)
        it.PruneChildren()
    return out

  def label(self, prim):
    """A body's name: the transformable prims on its path, the topmost one
    named for the scope it is in when it is an Actor_NNNN (chair/Actor_0001
    /Geom/body: chair_0001_body)."""
    names, p = [], prim
    while p and p != self.root:
      if p.IsA(UsdGeom.Xformable):
        name = p.GetName()
        if re.match(r'Actor_\d+$', name) and p.GetParent() != self.root:
          name = p.GetParent().GetName() + name[len('Actor'):]
        names.append(name)
      p = p.GetParent()
    return '_'.join(re.sub(r'\W', '_', n) for n in reversed(names))

  def tree(self):
    bodies = self.tops()
    bodies += [p for p in Usd.PrimRange(self.root) if moving(p) and p not in bodies]
    joints = {}
    for p in Usd.PrimRange(self.root):
      if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
        j = UsdPhysics.Joint(p)
        if not j.GetJointEnabledAttr().Get():
          continue
        b1 = j.GetBody1Rel().GetTargets()
        b0 = j.GetBody0Rel().GetTargets()
        child = self.stage.GetPrimAtPath(b1[0]) if b1 else None
        parent = self.owner(self.stage.GetPrimAtPath(b0[0])) if b0 and b0[0] != self.root.GetPath() else None
        joints.setdefault(child, []).append((p, parent))
    parent = {}
    for b in bodies:
      if b in joints:
        parents = {pp for _, pp in joints[b]}
        if len(parents) > 1:
          print(f'warning: {b.GetPath()}: joints to {len(parents)} bodies, the first used')
        parent[b] = joints[b][0][1]
      elif moving(b):
        parent[b] = None   # free: at the top, as MuJoCo wants free joints
      else:
        parent[b] = self.owner(b.GetParent()) if b.GetParent() != self.root else None
    return bodies, joints, parent

  # --- materials ---------------------------------------------------------------

  def material(self, prim):
    """The MJCF material of a mesh or subset: (name, textured, uv transform
    (scale, translate) or None). UsdPreviewSurface, or Omniverse's OmniPBR /
    OmniGlass MDL: the diffuse color or texture, opacity, metallic,
    roughness."""
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    color, image, alpha, metallic, roughness, uvx = (0.7, 0.7, 0.7), None, 1.0, None, None, None
    label = 'default'
    if mat:
      label = mat.GetPrim().GetName()
      shader = mat.ComputeSurfaceSource()[0] or mat.ComputeSurfaceSource('mdl')[0]
      get = lambda n: shader.GetInput(n).Get() if shader and shader.GetInput(n) else None
      mdl = str(shader.GetSourceAsset('mdl')) if shader and shader.GetImplementationSource() == 'sourceAsset' else ''
      if 'OmniGlass' in mdl:
        color, alpha = tuple(get('glass_color') or (1, 1, 1)), 0.25
        roughness, metallic = get('frosting_roughness') or 0.0, 0.0
      elif 'OmniPBR' in mdl:
        color = tuple(get('diffuse_color_constant') or color)
        tint = get('diffuse_tint')
        f = get('diffuse_texture')
        if f is not None and f.resolvedPath:
          image = pathlib.Path(f.resolvedPath)
          if tint is not None:
            color = tuple(tint)
        if get('enable_opacity') and get('opacity_constant') is not None:
          alpha = float(get('opacity_constant'))
        metallic, roughness = get('metallic_constant'), get('reflection_roughness_constant')
        sc, tr = get('texture_scale'), get('texture_translate')
        if (sc is not None and tuple(sc) != (1, 1)) or (tr is not None and tuple(tr) != (0, 0)):
          uvx = (np.array(sc if sc is not None else (1, 1)), np.array(tr if tr is not None else (0, 0)))
      elif shader:
        d = shader.GetInput('diffuseColor')
        if d:
          src = d.GetConnectedSources()[0]
          if src:
            tex = UsdShade.Shader(src[0].source.GetPrim())
            f = tex.GetInput('file').Get() if tex.GetInput('file') else None
            if f is not None and f.resolvedPath:
              image = pathlib.Path(f.resolvedPath)
          elif d.Get() is not None:
            color = tuple(d.Get())
        o = shader.GetInput('opacity')
        if o and o.Get() is not None and not o.GetConnectedSources()[0]:
          alpha = float(o.Get())
        metallic, roughness = get('metallic'), get('roughness')
    # USD's colors (UsdPreviewSurface, MDL) are linear; MJCF's rgba is sRGB.
    color = tuple(srgb(c) for c in color)
    tname = None
    if image is not None:
      tname, rgb = self.texture(image)
      if rgb is not None:   # one color: no texture needed
        image, color = None, rgb
    pbr = '' if metallic is None else f' metallic="{float(metallic):.3g}" roughness="{float(roughness if roughness is not None else 0.5):.3g}"'
    key = (str(image), tuple(round(c, 4) for c in color), round(alpha, 3), pbr)
    if key not in self.materials:
      name = self.unique(re.sub(r'\W', '_', label))
      rgba = fmt(list(color) + [alpha]) if not image else f'1 1 1 {alpha:.3g}'
      self.materials[key] = (name, tname, rgba, pbr)
    name, tex, _, _ = self.materials[key]
    return name, tex is not None, uvx if tex is not None else None

  def texture(self, image):
    if image not in self.textures:
      px = np.asarray(Image.open(image).convert('RGB')).reshape(-1, 3)
      if px.std(0).max() <= 4:   # (nearly) one color: brushed metal, chrome, black fabric
        self.textures[image] = (None, tuple(px.mean(0) / 255))
      else:
        name = self.unique('T_' + re.sub(r'\W', '_', image.stem))
        Image.open(image).convert('RGB').save(self.out / 'assets' / f'{name}.png')
        self.textures[image] = (name, None)
    return self.textures[image]

  # --- meshes -------------------------------------------------------------------

  def triangles(self, prim, to_body):
    """Triangles of a mesh in its body's frame: points (n, 3, 3), uv (n, 3, 2) or
    None, normals (n, 3, 3) or None, and each triangle's face index."""
    m = UsdGeom.Mesh(prim)
    pts = np.array(m.GetPointsAttr().Get(), float)
    counts = np.array(m.GetFaceVertexCountsAttr().Get())
    idx = np.array(m.GetFaceVertexIndicesAttr().Get())
    left = m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    corner, face = [], []
    for f, (s, n) in enumerate(zip(starts, counts)):
      for k in range(1, n - 1):
        tri = [s, s + k, s + k + 1]
        corner.append(tri[::-1] if left else tri)
        face.append(f)
    corner = np.array(corner).reshape(-1, 3)
    face = np.array(face)
    w = to_body @ matrix(prim, self.cache)
    p = pts[idx[corner]] @ w[:3, :3].T + w[:3, 3]
    if np.linalg.det(w[:3, :3]) < 0:   # mirrored: keep the faces outward
      p = p[:, ::-1]
      corner = corner[:, ::-1]

    def corners(pv):
      if not pv or not pv.HasValue():
        return None
      v = np.array(pv.ComputeFlattened(), float)
      interp = pv.GetInterpolation()
      if interp == UsdGeom.Tokens.faceVarying and len(v) == len(idx):
        return v[corner]
      if interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(v) == len(pts):
        return v[idx[corner]]
      if interp == UsdGeom.Tokens.uniform and len(v) == len(counts):
        return np.repeat(v[face][:, None], 3, axis=1)
      return None

    pv = UsdGeom.PrimvarsAPI(prim)
    uv = corners(pv.GetPrimvar('st'))
    normals = corners(pv.GetPrimvar('normals')) if pv.HasPrimvar('normals') else None
    if normals is None and m.GetNormalsAttr().HasValue():
      n = np.array(m.GetNormalsAttr().Get(), float)
      interp = m.GetNormalsInterpolation()
      normals = (n[corner] if interp == UsdGeom.Tokens.faceVarying and len(n) == len(idx)
                 else n[idx[corner]] if len(n) == len(pts) else None)
    if normals is not None:
      nm = np.linalg.inv(w[:3, :3]).T
      normals = normals @ nm.T
      normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-12)
    return p, uv, normals, face

  def write_mesh(self, name, tris, uv, normals):
    if uv is None:
      if len(tris) > self.max_faces:
        tris = self.simplify(tris, name)
      file = f'{name}.stl'
      write_stl(self.out / 'assets' / file, tris)
    else:
      file = f'{name}.obj'
      if len(tris) > self.max_faces:
        tris, uv, normals = self.simplify_textured(tris, uv, name)
      n = normals if normals is not None else np.repeat(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, None], 3, axis=1)
      corners = np.hstack([tris.reshape(-1, 3), uv.reshape(-1, 2), n.reshape(-1, 3)]).round(6)
      unique, index = np.unique(corners, axis=0, return_inverse=True)
      with open(self.out / 'assets' / file, 'w') as f:
        f.write('# converted by scripts/usd2mjcf.py\n')
        f.writelines(f'v {fmt(c[:3])}\n' for c in unique)
        f.writelines(f'vt {fmt(c[3:5])}\n' for c in unique)
        f.writelines(f'vn {fmt(c[5:])}\n' for c in unique)
        f.writelines('f ' + ' '.join(f'{i}/{i}/{i}' for i in t) + '\n' for t in index.reshape(-1, 3) + 1)
    self.meshes.append((name, file))
    return name

  def simplify_textured(self, tris, uv, what):
    """Like simplify, keeping texture coordinates: the mesh's corners (position
    and uv) are its vertices, so seams stay; each vertex left takes the uv of
    one it merged."""
    import fast_simplification
    corners = np.hstack([tris.reshape(-1, 3), uv.reshape(-1, 2)]).round(7)
    unique, inverse = np.unique(corners, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    _, _, collapses = fast_simplification.simplify(unique[:, :3].astype(np.float32), faces,
                                                   target_reduction=1 - self.max_faces / len(tris), return_collapses=True)
    points, faces, mapping = fast_simplification.replay_simplification(unique[:, :3].astype(np.float32), faces, collapses)
    st = np.zeros((len(points), 2))
    st[mapping] = unique[:, 3:5]
    print(f'{what}: {len(tris)} textured triangles simplified to {len(faces)}')
    return points[faces].astype(float), st[faces], None

  def simplify(self, tris, what):
    import fast_simplification
    points, inverse = np.unique(tris.reshape(-1, 3).round(7), axis=0, return_inverse=True)
    points, faces = fast_simplification.simplify(points.astype(np.float32), inverse.reshape(-1, 3).astype(np.int32),
                                                 target_reduction=1 - self.max_faces / len(tris))
    print(f'{what}: {len(tris)} triangles simplified to {len(faces)}')
    return points[faces]

  # --- collision -------------------------------------------------------------------

  def boxes_covering(self, tris):
    """Boxes (center, half) covering a thin, wide mesh in its plane, by 5 cm cells."""
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    t = int(np.argmin(ext))
    a, b = [k for k in range(3) if k != t]
    nx, ny = max(1, int(math.ceil(ext[a] / CELL))), max(1, int(math.ceil(ext[b] / CELL)))
    grid = np.zeros((nx, ny), bool)
    # Mark the cells whose centers lie in a triangle (projected), and the
    # cells the vertices are in (slivers).
    cx = lo[a] + (np.arange(nx) + 0.5) * ext[a] / nx
    cy = lo[b] + (np.arange(ny) + 0.5) * ext[b] / ny
    for tri in tris:
      p = tri[:, [a, b]]
      x0, y0 = p.min(0)
      x1, y1 = p.max(0)
      ix = np.nonzero((cx >= x0) & (cx <= x1))[0]
      iy = np.nonzero((cy >= y0) & (cy <= y1))[0]
      if len(ix) and len(iy):
        X, Y = np.meshgrid(cx[ix], cy[iy], indexing='ij')
        d = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        if abs(d) > 1e-12:
          l1 = ((p[1, 1] - p[2, 1]) * (X - p[2, 0]) + (p[2, 0] - p[1, 0]) * (Y - p[2, 1])) / d
          l2 = ((p[2, 1] - p[0, 1]) * (X - p[2, 0]) + (p[0, 0] - p[2, 0]) * (Y - p[2, 1])) / d
          inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (1 - l1 - l2 >= -1e-6)
          grid[np.ix_(ix, iy)] |= inside
    out = []
    # Greedy rectangles: grow each run along a, then along b while the rows match.
    used = np.zeros_like(grid)
    for i in range(nx):
      for j in range(ny):
        if not grid[i, j] or used[i, j]:
          continue
        j1 = j
        while j1 + 1 < ny and grid[i, j1 + 1] and not used[i, j1 + 1]:
          j1 += 1
        i1 = i
        while i1 + 1 < nx and grid[i1 + 1, j:j1 + 1].all() and not used[i1 + 1, j:j1 + 1].any():
          i1 += 1
        used[i:i1 + 1, j:j1 + 1] = True
        c = np.zeros(3)
        h = np.zeros(3)
        c[a] = lo[a] + (i + i1 + 1) / 2 * ext[a] / nx
        c[b] = lo[b] + (j + j1 + 1) / 2 * ext[b] / ny
        h[a] = (i1 - i + 1) / 2 * ext[a] / nx
        h[b] = (j1 - j + 1) / 2 * ext[b] / ny
        c[t] = (lo[t] + hi[t]) / 2
        h[t] = max(ext[t] / 2, MIN_HALF)
        if t == 2 and ext[t] < 2 * MIN_HALF:   # a floor plane: the box under it, its top the floor
          c[t] = hi[t] - MIN_HALF
        out.append((c, h))
    return out

  def voxel_boxes(self, tris, cell=CELL):
    """Boxes covering a mesh's surface by `cell` voxels, merged greedily."""
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    n = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    size = (hi - lo) / n
    grid = np.zeros(n, bool)
    # Each voxel's box around the surface points in it: merged boxes are
    # fitted to these, not to whole cells (a 2 cm shelf board gets a 2 cm box).
    vlo = np.full(tuple(n) + (3,), np.inf)
    vhi = np.full(tuple(n) + (3,), -np.inf)
    # Points spread over each triangle, finer than a cell.
    for tri in tris:
      e = max(np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[0]), np.linalg.norm(tri[2] - tri[1]))
      k = max(1, int(math.ceil(e / (cell / 2))))
      u, v = np.meshgrid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1))
      keep = u + v <= 1
      p = tri[0] + u[keep, None] * (tri[1] - tri[0]) + v[keep, None] * (tri[2] - tri[0])
      p = np.vstack([p, tri])
      i = np.clip(((p - lo) / np.maximum(size, 1e-9)).astype(int), 0, n - 1)
      grid[i[:, 0], i[:, 1], i[:, 2]] = True
      np.minimum.at(vlo, (i[:, 0], i[:, 1], i[:, 2]), p)
      np.maximum.at(vhi, (i[:, 0], i[:, 1], i[:, 2]), p)
    out, used = [], np.zeros_like(grid)
    for i, j, k in zip(*np.nonzero(grid)):
      if used[i, j, k]:
        continue
      i1, j1, k1 = i, j, k
      free = lambda a, b, c, d, e, f: grid[a:b, c:d, e:f].all() and not used[a:b, c:d, e:f].any()
      while i1 + 1 < n[0] and free(i1 + 1, i1 + 2, j, j1 + 1, k, k1 + 1):
        i1 += 1
      while j1 + 1 < n[1] and free(i, i1 + 1, j1 + 1, j1 + 2, k, k1 + 1):
        j1 += 1
      while k1 + 1 < n[2] and free(i, i1 + 1, j, j1 + 1, k1 + 1, k1 + 2):
        k1 += 1
      used[i:i1 + 1, j:j1 + 1, k:k1 + 1] = True
      a = vlo[i:i1 + 1, j:j1 + 1, k:k1 + 1].reshape(-1, 3).min(0)
      b = vhi[i:i1 + 1, j:j1 + 1, k:k1 + 1].reshape(-1, 3).max(0)
      out.append(((a + b) / 2, np.maximum((b - a) / 2, MIN_HALF)))
    return out

  def collision(self, prim, to_body):
    """Collision geoms of a collision prim: [(kind, center, half / mesh, rotation)]."""
    w = to_body @ matrix(prim, self.cache)
    if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Sphere):
      rot, pos = pose(w)
      scale = np.linalg.norm(w[:3, :3], axis=0)
      if prim.IsA(UsdGeom.Sphere):
        return [('sphere', pos, np.array([prim.GetAttribute('radius').Get() * scale.max()]), rot)]
      k = 'XYZ'.index(prim.GetAttribute('axis').Get() or 'Z')
      # MuJoCo's cylinders and capsules run along z: turn the axis onto it.
      turn = np.eye(3)[:, [(k + 1) % 3, (k + 2) % 3, k]]
      r = prim.GetAttribute('radius').Get() * max(scale[(k + 1) % 3], scale[(k + 2) % 3])
      h = prim.GetAttribute('height').Get() * scale[k] / 2
      return [('cylinder' if prim.IsA(UsdGeom.Cylinder) else 'capsule', pos, np.array([r, h]), rot @ turn)]
    if prim.IsA(UsdGeom.Cube):
      size = UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0
      corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size / 2
      pts = corners @ w[:3, :3].T + w[:3, 3]
    else:
      pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float) @ w[:3, :3].T + w[:3, 3]
    approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
    box = box_of(pts)
    if box is not None:
      return [('box', box[0], box[1], box[2])]
    if approx == 'boundingCube':
      lo, hi = pts.min(0), pts.max(0)
      return [('box', (lo + hi) / 2, np.maximum((hi - lo) / 2, MIN_HALF), np.eye(3))]
    if not len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []):
      return []
    tris, _, _, _ = self.triangles(prim, to_body)
    if approx is None and not moving(self.owner(prim)):
      approx = 'none'   # PhysX's default for a static mesh: the triangles as they are
    if approx in ('convexDecomposition', 'meshSimplification', 'none', 'sdf'):
      # Concave (a table, a wall with openings): boxes.
      lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
      ext = np.sort(hi - lo)
      if ext[0] < THIN and ext[1] > WIDE:
        return [('box', c, h, np.eye(3)) for c, h in self.boxes_covering(tris)]
      label = f'{self.label(self.owner(prim))}_{name_of(prim.GetPath())}'
      boxes = self.voxel_boxes(tris, FINE if label in self.fine else CELL)
      return [('box', c, h, np.eye(3), f'{label}_vox') for c, h in boxes]
    name = self.unique(f'{name_of(prim.GetPath())}_hull')
    from scipy.spatial import ConvexHull   # MuJoCo takes the hull anyway: only its faces
    pts = np.unique(tris.reshape(-1, 3).round(7), axis=0)
    hull = ConvexHull(pts)
    self.write_mesh(name, pts[hull.simplices], None, None)
    return [('mesh', None, name, None)]

  # --- the model ---------------------------------------------------------------------

  def body_xml(self, body, parent_pose, children, joints, ind):
    rot, pos = pose(matrix(body, self.cache))
    world_rot, world_pos = rot, pos
    if parent_pose is None:
      pos = pos + self.offset
    else:
      prot, ppos = parent_pose
      rot, pos = prot.T @ rot, prot.T @ (pos - ppos)
    to_body = np.eye(4)
    to_body[:3, :3], to_body[:3, 3] = world_rot.T, -world_rot.T @ world_pos
    name = self.unique(self.label(body))
    attrs = f' pos="{fmt(pos)}"' if np.abs(pos).max() > 1e-9 else ''
    if np.abs(rot - np.eye(3)).max() > 1e-9:
      attrs += f' quat="{fmt(quat(rot))}"'
    lines = [f'{ind}<body name="{name}"{attrs}>']
    rigid = moving(body)
    if body in joints:
      for jp, _ in joints[body]:
        j = UsdPhysics.Joint(jp)
        lp = np.array(j.GetLocalPos1Attr().Get(), float)
        q = j.GetLocalRot1Attr().Get()
        jr = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(q)))).T
        scale = np.linalg.norm(matrix(body, self.cache)[:3, :3], axis=0)
        axis = {'X': 0, 'Y': 1, 'Z': 2}[jp.GetAttribute('physics:axis').Get()]
        a = jr[:, axis]
        revolute = jp.IsA(UsdPhysics.RevoluteJoint)
        lo, hi = jp.GetAttribute('physics:lowerLimit').Get(), jp.GetAttribute('physics:upperLimit').Get()
        rng = ''
        if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo < hi:
          rng = f' range="{fmt([math.radians(lo), math.radians(hi)] if revolute else [lo, hi])}"'
        lines.append(f'{ind}  <joint name="{self.unique(name_of(jp.GetPath()))}" type="{"hinge" if revolute else "slide"}"'
                     f' pos="{fmt(lp * scale)}" axis="{fmt(a)}"{rng}/>')
    elif rigid and parent_pose is None:
      lines.append(f'{ind}  <freejoint/>')
    # Geoms: the prims this body owns (not those of rigid bodies under it).
    visual, colliders = [], []
    it = iter(Usd.PrimRange(body))
    for p in it:
      if p != body and (moving(p) or p in self.bodies):
        it.PruneChildren()
        continue
      if collides(p):
        colliders.append(p)
      # A visible mesh can be its own collider too (a table, a wall).
      if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible \
          and UsdGeom.Imageable(p).ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render) \
          and len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []):
        visual.append(p)
    geoms, cols = [], []
    for p in visual:
      tris, uv, normals, face = self.triangles(p, to_body)
      subsets = [s for s in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(p)) if s.GetFamilyNameAttr().Get() == 'materialBind']
      groups, rest = [], np.ones(len(face), bool)
      for s in subsets:
        sel = np.isin(face, np.array(s.GetIndicesAttr().Get()))
        rest &= ~sel
        groups.append((s.GetPrim(), sel))
      if rest.any():
        groups.append((p, rest))
      for k, (target, sel) in enumerate(groups):
        if not sel.any():
          continue
        mat, textured, uvx = self.material(target)
        st = uv[sel] if textured and uv is not None else None
        if st is not None and uvx is not None:
          st = st * uvx[0] + uvx[1]
        mesh = self.write_mesh(self.unique(f'{self.label(body)}_{name_of(p.GetPath())}' + (f'_{k}' if len(groups) > 1 else '')),
                               tris[sel], st,
                               normals[sel] if normals is not None else None)
        geoms.append(f'{ind}  <geom class="visual" type="mesh" mesh="{mesh}" material="{mat}"/>')
      # No collision geom of its own around it: boxes.
      # Collision geoms of its own: itself, UCX_* next to it, shapes under it.
      covered = any(c == p or c.GetParent() == p.GetParent() or c.GetParent() == p for c in colliders)
      if not covered and not (self.no_collision and re.search(self.no_collision, body.GetName())):
        lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
        ext = np.sort(hi - lo)
        mesh_name = f'{self.label(body)}_{name_of(p.GetPath())}'
        if ext[0] < THIN and ext[1] > WIDE:
          cols += [('box', c, h, np.eye(3)) for c, h in self.boxes_covering(tris)]
        elif mesh_name in self.voxelize:
          boxes = self.voxel_boxes(tris, FINE if mesh_name in self.fine else CELL)
          cols += [('box', c, h, np.eye(3), f'{mesh_name}_vox') for c, h in boxes]
        else:
          cols.append(('box', (lo + hi) / 2, np.maximum((hi - lo) / 2, MIN_HALF), np.eye(3), f'{mesh_name}_aabb'))
    for c in colliders:
      cols += self.collision(c, to_body)
    # Mass: physics:mass over the collision geoms by volume.
    mass = body.GetAttribute('physics:mass').Get() if rigid else None
    vol = [8 * np.prod(col[2]) if col[0] == 'box' else
           2 * math.pi * col[2][0] ** 2 * col[2][1] if col[0] in ('cylinder', 'capsule') else None for col in cols]
    known = sum(v for v in vol if v) or 1.0
    for (k, c, h, r, *label), v in zip(cols, vol):
      m = ''
      if rigid:
        share = (v or known / max(len(cols), 1)) / known
        m = f' mass="{(mass if mass else DENSITY * known) * share:.6g}"'
      if k in ('box', 'sphere', 'cylinder', 'capsule'):
        q = '' if np.abs(r - np.eye(3)).max() < 1e-9 else f' quat="{fmt(quat(r))}"'
        named = f' name="{self.unique(label[0])}"' if label else ''
        geoms.append(f'{ind}  <geom{named} class="collision" type="{k}" pos="{fmt(c)}"{q} size="{fmt(h)}"{m}/>')
      else:
        geoms.append(f'{ind}  <geom class="collision" type="mesh" mesh="{h}"{m}/>')
    lines += geoms
    for child in children.get(body, []):
      lines += self.body_xml(child, (world_rot, world_pos), children, joints, ind + '  ')
    lines.append(f'{ind}</body>')
    return lines

  def mjcf(self, model_name):
    bodies, joints, parent = self.tree()
    self.bodies = set(bodies)
    children = {}
    for b in bodies:
      children.setdefault(parent[b], []).append(b)
    worldbody = []
    for b in children.get(None, []):
      worldbody += self.body_xml(b, None, children, joints, '    ')
    assets = []
    for tex in sorted({t for _, t, _, _ in self.materials.values() if t}):
      assets.append(f'    <texture name="{tex}" type="2d" file="{tex}.png"/>')
    for name, tex, rgba, pbr in sorted(self.materials.values()):
      texture = f' texture="{tex}"' if tex else ''
      assets.append(f'    <material name="{name}"{texture} rgba="{rgba}"{pbr}/>')
    assets += [f'    <mesh name="{n}" file="{f}"/>' for n, f in self.meshes]
    return '\n'.join([
      f'<mujoco model="{model_name}">',
      '  <compiler angle="radian" meshdir="assets" texturedir="assets" autolimits="true"/>',
      '  <default>',
      '    <joint armature="0.01" damping="3"/>',
      '    <mesh inertia="shell"/>',
      '    <default class="visual">',
      '      <geom contype="0" conaffinity="0" group="2" density="0"/>',
      '    </default>',
      '    <default class="collision">',
      # Torsional friction: a book on a shelf of small boxes does not creep.
      '      <geom rgba="0.9 0.9 0.0 0.2" group="3" condim="4" friction="1 0.02 0.002"/>',
      '    </default>',
      '  </default>',
      '  <asset>',
      *assets,
      '  </asset>',
      '  <worldbody>',
      *worldbody,
      '  </worldbody>',
      '</mujoco>',
      '',
    ])


def boxes_in_the_way(xml):
  """Meshes with collision boxes on a static body that hold a body that moves
  at rest: (bounding boxes, *_aabb geoms; 5 cm voxel boxes, *_vox geoms)."""
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  aabb, vox = set(), set()
  for c in d.contact[:d.ncon]:
    for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
      name = m.geom(g).name
      if m.body_weldid[m.geom_bodyid[g]] == 0 and m.body_weldid[m.geom_bodyid[other]] != 0 and c.dist < -0.002:
        if re.search(r'_aabb(_\d+)?$', name):
          aabb.add(re.sub(r'_aabb(_\d+)?$', '', name))
        elif re.search(r'_vox(_\d+)?$', name):
          vox.add(re.sub(r'_vox(_\d+)?$', '', name))
  return aabb, vox


def hold_joints(xml):
  """Doors and drawers stay where they are left: a door on a horizontal hinge
  (a dishwasher's, a flap) is gravity compensated, as its springs do on the
  real one; every hinge and slide gets some friction (a hinge 0.5 Nm plus 2 %
  of the most gravity could pull on what it carries, a slide 5 % of its
  weight)."""
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  text = xml.read_text()
  for j in range(m.njnt):
    if m.jnt_type[j] not in (2, 3):
      continue
    b = m.jnt_bodyid[j]
    mass = m.body_subtreemass[b]
    name = m.joint(j).name
    if m.jnt_type[j] == 3:
      axis, anchor = d.xaxis[j], d.xanchor[j]
      r = d.subtree_com[b] - anchor
      lever = np.linalg.norm(r - (r @ axis) * axis)
      tilt = math.sqrt(max(0, 1 - axis[2] ** 2))   # 0 for a vertical axis: gravity does not turn it
      friction = 0.5 + 0.02 * mass * 9.81 * lever * tilt
      if tilt > 0.5:
        body = m.body(b).name
        text = re.sub(rf'(<body name="{re.escape(body)}"(?![^>]*gravcomp)[^>]*?)>', r'\1 gravcomp="1">', text, 1)
    else:
      friction = 0.05 * mass * 9.81
    text = re.sub(rf'(<joint name="{re.escape(name)}"[^>]*?)/>', rf'\1 frictionloss="{friction:.3g}"/>', text, 1)
  xml.write_text(text)


def exclude_resting(xml):
  """Contact excludes for jointed bodies and the bodies they touch at rest."""
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  pairs = set()
  for c in d.contact[:d.ncon]:
    a, b = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
    jointed = lambda x: m.body_jntnum[x] and m.jnt_type[m.body_jntadr[x]] in (2, 3)
    if jointed(a) or jointed(b):
      pairs.add(tuple(sorted((m.body(a).name, m.body(b).name))))
  free = {tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)))
          for c in d.contact[:d.ncon] if c.dist < -0.005} - pairs
  for p in sorted(free):
    print(f'warning: {p[0]} and {p[1]} overlap at rest')
  text = xml.read_text()
  if pairs:
    block = '  <contact>\n' + ''.join(f'    <exclude body1="{a}" body2="{b}"/>\n' for a, b in sorted(pairs)) + '  </contact>\n'
    text = text.replace('</mujoco>', block + '</mujoco>')
  xml.write_text(text)
  return len(pairs)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('usd')
  ap.add_argument('out')
  ap.add_argument('--offset', type=float, nargs=3, default=[0, 0, 0], help='moves the scene (m)')
  ap.add_argument('--max-faces', type=int, default=50000)
  ap.add_argument('--no-collision', metavar='REGEX', help='bodies (by name) that get no generated collision geoms: '
                  'a shell around the scene (ceiling, roof) whose bounding box would hold it all')
  args = ap.parse_args()
  stage = Usd.Stage.Open(args.usd)
  if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-9:
    raise SystemExit('only Z-up stages in meters')
  out = pathlib.Path(args.out)
  shutil.rmtree(out / 'assets', ignore_errors=True)
  (out / 'assets').mkdir(parents=True)
  xml = out / f'{out.name}.xml'
  conv = Converter(stage, out, args.offset, args.max_faces)
  conv.no_collision = args.no_collision
  xml.write_text(conv.mjcf(out.name))
  voxelize, fine = set(), set()
  for _ in range(2):
    # Again, with boxes over voxels for bounding boxes that hold what moves,
    # finer voxels where 5 cm ones do.
    aabb, vox = boxes_in_the_way(xml)
    if not (aabb - voxelize or vox - fine):
      break
    voxelize |= aabb
    fine |= vox
    print(f'boxes over voxels for {len(voxelize)} meshes, finer ones for {len(fine)}: what moves is in their boxes:',
          ' '.join(sorted(voxelize | fine)))
    shutil.rmtree(out / 'assets')
    (out / 'assets').mkdir()
    conv = Converter(stage, out, args.offset, args.max_faces)
    conv.no_collision = args.no_collision
    conv.voxelize, conv.fine = voxelize, fine
    xml.write_text(conv.mjcf(out.name))
  hold_joints(xml)
  n = exclude_resting(xml)
  m = mujoco.MjModel.from_xml_path(str(xml))
  print(f'{xml}: {m.nbody - 1} bodies, {m.njnt} joints, {m.ngeom} geoms, {m.nmeshface} mesh faces, '
        f'{len(conv.materials)} materials, {n} excludes')


if __name__ == '__main__':
  main()
