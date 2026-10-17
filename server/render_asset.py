"""What a browser needs to draw a model file without compiling it: the
visible geoms with their materials, textures and meshes, and the kinematic
tree (to place bodies from qpos itself, public/net.js forwardKinematics).

Arrays keep the names (and dtypes) of the compiled MjModel's fields, so the
page builds its Filament parts with the same code as from a WASM model
(main.js buildVisual). Only what is drawn goes in: visible geoms (group < 3,
alpha > 0, a supported type), the meshes they use, and the pixels of
MuJoCo's own 2D textures (builtins); image file textures are sent as paths
under public/ instead, the browser decodes them.

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
      parts['mesh_facetexcoord'].append(np.zeros((nf, 3), np.int32))
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

  # Materials (all; few) and textures: pixels only for 2D textures MuJoCo
  # made itself, image files go by path.
  for f in ('mat_rgba', 'mat_texid', 'mat_texrepeat', 'mat_texuniform'):
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

  # Kinematic tree (mj_kinematics).
  for f in ('body_parentid', 'body_pos', 'body_quat', 'body_jntnum', 'body_jntadr', 'body_mocapid',
            'jnt_type', 'jnt_pos', 'jnt_axis', 'jnt_qposadr', 'qpos0'):
    arrays[f] = getattr(m, f)

  header = {
      'ngeom': len(geoms), 'nmesh': len(meshes), 'nmat': m.nmat, 'ntex': m.ntex,
      'nbody': m.nbody, 'njnt': m.njnt,
      'dims': {'nq': m.nq, 'nv': m.nv, 'na': m.na, 'nu': m.nu, 'nmocap': m.nmocap},
      'images': {str(k): v for k, v in images.items()},
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
