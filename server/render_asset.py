"""What a browser needs to draw a model file without compiling it: the
visible geoms with their materials, textures and meshes, and the kinematic
tree (to place bodies from qpos itself, public/js/net/net.js forwardKinematics).

Arrays keep the names (and dtypes) of the compiled MjModel's fields, so the
page builds its Filament parts with the same code as from a WASM model
(main.js buildVisual). Only what is drawn goes in: visible geoms (group < 3,
alpha > 0, a supported type), the meshes they use, and the pixels of
MuJoCo's own 2D textures (builtins); image file textures are sent as paths
under public/ instead, the browser decodes them. For the Render panel's
debug overlays (public/js/render/debug.js): the geoms that collide as col_*
tables, each mesh as its convex hull (the vertices it uses and its faces),
and jnt_bodyid.

mesh_facenormal and mesh_facetexcoord are left out when they are mesh_face
itself (as the OBJ files of the importers have them, `f i/i/i`), listed in
the header's `same_as_face`: the page uses mesh_face for them. A mesh
without texcoords gets its faces as facetexcoord, never read (its
texcoordadr is -1), so that it does not keep the array apart.

Packed as: uint32 header length, JSON header ({arrays: {name: [dtype,
offset, count]}, ...}), zero padding to 8 bytes, then the arrays (each at an
8-byte aligned offset of that data block).
"""

import json
import posixpath
import struct

import mujoco
import numpy as np

DRAWN_TYPES = {0, 2, 3, 4, 5, 6, 7}  # plane sphere capsule ellipsoid cylinder box mesh
DTYPES = {np.dtype(np.float32): 'f4', np.dtype(np.float64): 'f8', np.dtype(np.int32): 'i4', np.dtype(np.uint8): 'u1'}


def image_textures(spec, rel, public):
  """texture id -> image path under public/ (like `rel`, the model file),
  for plain 2D file textures."""
  base = posixpath.join(posixpath.dirname(rel), spec.texturedir or '')
  out = {}
  for i, t in enumerate(spec.textures):
    if (t.type != mujoco.mjtTexture.mjTEXTURE_2D or not t.file or tuple(t.gridsize) != (1, 1)
        or ''.join(t.gridlayout).strip('.') or t.hflip or t.vflip):
      continue
    path = posixpath.normpath(posixpath.join(base, t.file))
    if not path.startswith('..') and (public / path).is_file():
      out[i] = path
  return out


def build(spec, model, rel, public):
  m = model
  arrays = {}

  # Geoms that are drawn, with the meshes and materials they use.
  rgba = lambda g: m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
  geoms = [g for g in range(m.ngeom) if m.geom_group[g] < 3 and rgba(g)[3] > 0 and m.geom_type[g] in DRAWN_TYPES]
  meshes = sorted({int(m.geom_dataid[g]) for g in geoms if m.geom_type[g] == 7})
  mesh_index = {k: i for i, k in enumerate(meshes)}
  gi = np.array(geoms, dtype=int)
  for f in ('geom_type', 'geom_size', 'geom_matid', 'geom_rgba', 'geom_group', 'geom_bodyid', 'geom_pos', 'geom_quat'):
    arrays[f] = getattr(m, f)[gi]
  arrays['geom_dataid'] = np.array([mesh_index.get(int(m.geom_dataid[g]), -1) if m.geom_type[g] == 7 else -1 for g in geoms], np.int32)

  # Meshes: each one's slice of the vertex / normal / texcoord / face arrays.
  parts = {k: [] for k in ('mesh_vert', 'mesh_normal', 'mesh_texcoord', 'mesh_face', 'mesh_facenormal', 'mesh_facetexcoord')}
  adr = {k: [] for k in ('mesh_vertadr', 'mesh_normaladr', 'mesh_texcoordadr', 'mesh_faceadr', 'mesh_facenum')}
  at = {'vert': 0, 'normal': 0, 'texcoord': 0, 'face': 0}
  for k in meshes:
    nv, nn, nt, nf = m.mesh_vertnum[k], m.mesh_normalnum[k], m.mesh_texcoordnum[k], m.mesh_facenum[k]
    va, na, ta, fa = m.mesh_vertadr[k], m.mesh_normaladr[k], m.mesh_texcoordadr[k], m.mesh_faceadr[k]
    parts['mesh_vert'].append(m.mesh_vert[va:va + nv])
    parts['mesh_normal'].append(m.mesh_normal[na:na + nn])
    parts['mesh_face'].append(m.mesh_face[fa:fa + nf])
    parts['mesh_facenormal'].append(m.mesh_facenormal[fa:fa + nf])
    adr['mesh_vertadr'].append(at['vert'])
    adr['mesh_normaladr'].append(at['normal'])
    adr['mesh_faceadr'].append(at['face'])
    adr['mesh_facenum'].append(nf)
    if ta >= 0 and nt > 0:
      parts['mesh_texcoord'].append(m.mesh_texcoord[ta:ta + nt])
      parts['mesh_facetexcoord'].append(m.mesh_facetexcoord[fa:fa + nf])
      adr['mesh_texcoordadr'].append(at['texcoord'])
      at['texcoord'] += nt
    else:
      parts['mesh_facetexcoord'].append(m.mesh_face[fa:fa + nf])   # never read (texcoordadr -1)
      adr['mesh_texcoordadr'].append(-1)
    at['vert'] += nv
    at['normal'] += nn
    at['face'] += nf
  empty = {'mesh_vert': np.float32, 'mesh_normal': np.float32, 'mesh_texcoord': np.float32,
           'mesh_face': np.int32, 'mesh_facenormal': np.int32, 'mesh_facetexcoord': np.int32}
  for k, v in parts.items():
    arrays[k] = np.concatenate(v) if v else np.zeros(0, empty[k])
  for k, v in adr.items():
    arrays[k] = np.array(v, np.int32)
  same_as_face = [k for k in ('mesh_facenormal', 'mesh_facetexcoord') if np.array_equal(arrays[k], arrays['mesh_face'])]
  for k in same_as_face:
    del arrays[k]

  # Materials (all; few) and textures: pixels only for 2D textures MuJoCo
  # made itself, image files go by path.
  for f in ('mat_rgba', 'mat_texid', 'mat_texrepeat', 'mat_texuniform',
            'mat_metallic', 'mat_roughness', 'mat_shininess', 'mat_specular', 'mat_reflectance'):
    arrays[f] = getattr(m, f)
  images = image_textures(spec, rel, public) if len(spec.textures) == m.ntex else {}
  data, tex_adr = [], []
  n = 0
  for t in range(m.ntex):
    tex_adr.append(n)
    if m.tex_type[t] == 0 and t not in images:
      size = m.tex_width[t] * m.tex_height[t] * m.tex_nchannel[t]
      data.append(m.tex_data[m.tex_adr[t]:m.tex_adr[t] + size])
      n += size
  for f in ('tex_type', 'tex_width', 'tex_height', 'tex_nchannel'):
    arrays[f] = getattr(m, f)
  arrays['tex_adr'] = np.array(tex_adr, np.int32)
  arrays['tex_data'] = np.concatenate(data) if data else np.zeros(0, np.uint8)

  # The geoms that collide (debug overlays): their meshes as the convex hull
  # MuJoCo collides with (mesh_graph), else all of the mesh.
  cols = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] in DRAWN_TYPES]
  ci = np.array(cols, dtype=int)
  for f in ('geom_type', 'geom_size', 'geom_bodyid', 'geom_pos', 'geom_quat'):
    arrays['col_' + f] = getattr(m, f)[ci] if cols else np.zeros((0,) + getattr(m, f).shape[1:], getattr(m, f).dtype)
  col_meshes = sorted({int(m.geom_dataid[g]) for g in cols if m.geom_type[g] == 7})
  col_index = {k: i for i, k in enumerate(col_meshes)}
  arrays['col_geom_dataid'] = np.array([col_index.get(int(m.geom_dataid[g]), -1) if m.geom_type[g] == 7 else -1 for g in cols], np.int32)
  verts, faces, cadr = [], [], {k: [] for k in ('col_mesh_vertadr', 'col_mesh_vertnum', 'col_mesh_faceadr', 'col_mesh_facenum')}
  nv_at = nf_at = 0
  for k in col_meshes:
    va, nv = m.mesh_vertadr[k], m.mesh_vertnum[k]
    g = m.mesh_graphadr[k]
    if g >= 0:
      numvert, numface = int(m.mesh_graph[g]), int(m.mesh_graph[g + 1])
      at = g + 2 + numvert + numvert + (numvert + 3 * numface)
      face = m.mesh_graph[at:at + 3 * numface].reshape(-1, 3)
    else:
      fa, nf = m.mesh_faceadr[k], m.mesh_facenum[k]
      face = m.mesh_face[fa:fa + nf]
    used, face = np.unique(face, return_inverse=True)  # only the vertices the faces use
    verts.append(m.mesh_vert[va:va + nv][used].astype(np.float32))
    faces.append(face.reshape(-1, 3).astype(np.int32))
    cadr['col_mesh_vertadr'].append(nv_at)
    cadr['col_mesh_vertnum'].append(len(used))
    cadr['col_mesh_faceadr'].append(nf_at)
    cadr['col_mesh_facenum'].append(len(faces[-1]))
    nv_at += len(used)
    nf_at += len(faces[-1])
  arrays['col_mesh_vert'] = np.concatenate(verts) if verts else np.zeros((0, 3), np.float32)
  arrays['col_mesh_face'] = np.concatenate(faces) if faces else np.zeros((0, 3), np.int32)
  for k, v in cadr.items():
    arrays[k] = np.array(v, np.int32)

  # Kinematic tree (mj_kinematics).
  for f in ('body_parentid', 'body_pos', 'body_quat', 'body_jntnum', 'body_jntadr', 'body_mocapid',
            'jnt_type', 'jnt_pos', 'jnt_axis', 'jnt_qposadr', 'jnt_bodyid', 'qpos0'):
    arrays[f] = getattr(m, f)

  # Cameras (public/js/cameras/cameras.js, modelCameras).
  cameras = [{
      'name': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) or f'camera{c}',
      'body': int(m.cam_bodyid[c]), 'mode': int(m.cam_mode[c]), 'fovy': float(m.cam_fovy[c]),
      'pos': m.cam_pos[c].tolist(), 'quat': m.cam_quat[c].tolist(),
      'pos0': m.cam_pos0[c].tolist(), 'mat0': m.cam_mat0[c].tolist(),
      'resolution': [int(v) for v in m.cam_resolution[c]],
  } for c in range(m.ncam)]

  header = {
      'cameras': cameras,
      'ngeom': len(geoms), 'nmesh': len(meshes), 'nmat': m.nmat, 'ntex': m.ntex,
      'nbody': m.nbody, 'njnt': m.njnt,
      'dims': {'nq': m.nq, 'nv': m.nv, 'na': m.na, 'nu': m.nu, 'nmocap': m.nmocap},
      'images': {str(k): v for k, v in images.items()},
      'same_as_face': same_as_face,
      'arrays': {},
  }
  blob = bytearray()
  for name, a in arrays.items():
    a = np.ascontiguousarray(a)
    if a.dtype not in DTYPES:
      a = a.astype(np.int32 if a.dtype.kind in 'iub' else np.float64)
    blob += b'\0' * (-len(blob) % 8)
    header['arrays'][name] = [DTYPES[a.dtype], len(blob), int(a.size)]
    blob += a.tobytes()
  head = json.dumps(header).encode()
  head += b' ' * (-(4 + len(head)) % 8)
  return struct.pack('<I', len(head)) + head + bytes(blob)
