// MuJoCo (official WASM bindings) for physics + Filament.js for rendering.
//
// Rendering and physics use different MuJoCo models. Each robot file is
// compiled once on its own, in a throwaway MuJoCo instance, to bake its
// visible geoms into Filament buffers (vertices from geom_size / mesh data,
// pose relative to the geom's body); all spawned copies share them. The
// physics world (world.js) is a physics-only model rebuilt, in a fresh MuJoCo
// instance, on every spawn. Every frame the body poses (xpos/xmat) of the
// world are composed with each geom's local pose into the Filament transform.
// The scene stays in MuJoCo's Z-up frame.
//
// Fresh instances because the @mujoco/mujoco WASM heap is capped at 2 GB and
// never shrinks, and MjSpec.delete() does not give back what a compile used:
// dropping the whole instance is the only way to get that memory back.

import loadMujoco from '../vendor/mujoco/mujoco.js';
import { loadModelFiles, readFirst, urlSource } from './vfs.js';
import { bodyInertia, compileWorld, firstKey, referencedNames } from './world.js';
import { CONTROL_DT } from './groot.js';
import { defaultPolicy, loadPolicy, loadPolicyIndex, policyById } from './policy.js';
import { Perturb, StagePanel } from './inspect.js';
import { AssetPanel, loadAssets, loadRecent } from './assets.js';
import { AxisGizmo } from './gizmo.js';
import { TeleopPanel, TOUCH, locomotion } from './teleop.js';
import { bodyPose } from './goal.js';
import { BaseController } from './base.js';
import { IkController } from './ik.js';
import { applyOptions, geomOverrides, geomSignature, injectGeomDefaults, loadPhysics, needsRebuild } from './physics.js';
import { WorldPanel } from './worldpanel.js';
import { RenderPanel, loadQuality } from './render.js';
import { ServerPanel } from './serverpanel.js';
import { SharePanel } from './share.js';
import { MARKER } from './marker.js';
import { CHAIN_COLORS } from './teleop.js';
import Stats from '../vendor/stats/Stats.js';
import { NET_MUJOCO, NetClient, loadRender, worldData, worldModel } from './net.js';
import { dataUrl, problem, server } from './endpoint.js';
import { RemoteBase, RemoteIk, RemotePolicy } from './remote.js';

// Filament.init() replaces the global with the loaded module, so re-read it after init.
let Filament = window.Filament;
const MAT_LIT = 'materials/lit.filamat';
const MAT_CHECKER = 'materials/checker.filamat';
const MAT_TEXTURED = 'materials/lit_textured.filamat';
// Alpha-blended variants, for geoms whose rgba alpha is below 1.
const MAT_TRANSPARENT = 'materials/lit_transparent.filamat';
const MAT_TEXTURED_TRANSPARENT = 'materials/lit_textured_transparent.filamat';


// ---------------------------------------------------------------------------
// Procedural meshes in the geom's local frame. Each returns {pos, nrm, idx}.

function grid(rows, cols, vertex) {
  // vertex(i, j) -> [px,py,pz, nx,ny,nz]; builds a (rows+1)x(cols+1) strip grid.
  const pos = [], nrm = [], idx = [];
  for (let i = 0; i <= rows; i++) {
    for (let j = 0; j <= cols; j++) {
      const v = vertex(i, j);
      pos.push(v[0], v[1], v[2]);
      nrm.push(v[3], v[4], v[5]);
    }
  }
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const a = i * (cols + 1) + j, b = a + cols + 1;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { pos, nrm, idx };
}

function merge(...parts) {
  const out = { pos: [], nrm: [], idx: [] };
  for (const p of parts) {
    const base = out.pos.length / 3;
    out.pos.push(...p.pos);
    out.nrm.push(...p.nrm);
    out.idx.push(...p.idx.map((k) => k + base));
  }
  return out;
}

const SEG = 32;

// Ellipsoid with radii (a,b,c); for a capsule, hemispheres are shifted by ±h.
function roundShape(a, b, c, h = 0) {
  const half = SEG / 2;
  return grid(SEG + 1, SEG, (i, j) => {
    // Rows 0..half are the upper hemisphere, half+1..SEG+1 the lower one.
    const upper = i <= half;
    const lat = Math.PI / 2 - (upper ? i : i - 1) * (Math.PI / SEG);
    const lon = (j / SEG) * 2 * Math.PI;
    const x = Math.cos(lat) * Math.cos(lon), y = Math.cos(lat) * Math.sin(lon), z = Math.sin(lat);
    const n = [x / a, y / b, z / c];
    const l = Math.hypot(...n);
    return [a * x, b * y, c * z + (upper ? h : -h), n[0] / l, n[1] / l, n[2] / l];
  });
}

function disk(r, z, nz) {
  const pos = [0, 0, z], nrm = [0, 0, nz], idx = [];
  for (let j = 0; j <= SEG; j++) {
    const t = (j / SEG) * 2 * Math.PI;
    pos.push(r * Math.cos(t), r * Math.sin(t), z);
    nrm.push(0, 0, nz);
    if (j > 0) idx.push(...(nz > 0 ? [0, j, j + 1] : [0, j + 1, j]));
  }
  return { pos, nrm, idx };
}

function cylinder(r, h) {
  const side = grid(1, SEG, (i, j) => {
    const t = (j / SEG) * 2 * Math.PI, c = Math.cos(t), s = Math.sin(t);
    return [r * c, r * s, i === 0 ? h : -h, c, s, 0];
  });
  return merge(side, disk(r, h, 1), disk(r, -h, -1));
}

// Side of a cone frustum along +z: radius r0 at z0 to r1 at z1.
function frustum(r0, r1, z0, z1) {
  const h = z1 - z0, l = Math.hypot(h, r0 - r1);
  return grid(1, SEG, (i, j) => {
    const t = (j / SEG) * 2 * Math.PI, c = Math.cos(t), s = Math.sin(t), r = i ? r0 : r1;
    return [r * c, r * s, i ? z0 : z1, (c * h) / l, (s * h) / l, (r0 - r1) / l];
  });
}

// Ring of radius R around +z, tube radius r.
function torus(R, r, minor = 12) {
  return grid(minor, SEG * 2, (i, j) => {
    const p = (-i / minor) * 2 * Math.PI, t = (j / (SEG * 2)) * 2 * Math.PI;
    const cp = Math.cos(p), sp = Math.sin(p), ct = Math.cos(t), st = Math.sin(t);
    return [(R + r * cp) * ct, (R + r * cp) * st, r * sp, cp * ct, cp * st, sp];
  });
}

// Interactive marker arrows (marker.js MARKER): one along +z and one along -z.
function markerArrows() {
  const [a, b, c] = MARKER.arrow, rs = MARKER.shaft, rh = MARKER.head;
  const up = merge(frustum(rs, rs, a, b), disk(rs, a, -1), frustum(rh, 0, b, c), disk(rh, b, -1));
  const down = { pos: up.pos.map((v, k) => (k % 3 === 0 ? v : -v)), nrm: up.nrm.map((v, k) => (k % 3 === 0 ? v : -v)), idx: up.idx };
  return merge(up, down);  // down: turned half a turn about x, so its winding still faces out
}

function box(sx, sy, sz) {
  const faces = [];
  const s = [sx, sy, sz];
  for (let axis = 0; axis < 3; axis++) {
    for (const sign of [1, -1]) {
      const u = (axis + 1) % 3, v = (axis + 2) % 3;
      faces.push(grid(1, 1, (i, j) => {
        const p = [0, 0, 0], n = [0, 0, 0];
        p[axis] = sign * s[axis];
        p[u] = (sign > 0 ? (i ? 1 : -1) : (j ? 1 : -1)) * s[u];
        p[v] = (sign > 0 ? (j ? 1 : -1) : (i ? 1 : -1)) * s[v];
        n[axis] = sign;
        return [...p, ...n];
      }));
    }
  }
  return merge(...faces);
}

function plane(sx, sy) {
  sx = sx || 10; sy = sy || 10;  // size 0 means infinite in MuJoCo
  return grid(1, 1, (i, j) => [(j ? 1 : -1) * sx, (i ? -1 : 1) * sy, 0, 0, 0, 1]);
}

const SPAWN_MARKER_RADIUS = 0.25;  // m
// IK marker colors: x/y/z arrows and rings, the hovered or dragged part.
const AXIS_COLORS = [[0.9, 0.2, 0.2], [0.25, 0.8, 0.3], [0.25, 0.45, 1]];
const MARKER_HOT = [1, 0.85, 0.2];
// Columns: where a marker part's local x, y, z go, so its +z lies along x, y or z.
const PART_AXES = [[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]];
const ORIGIN_AXIS_LENGTH = 0.3, ORIGIN_AXIS_RADIUS = 0.008;  // m

// The drag-to-spawn marker, flat in the z=0 plane, unit radius: a ring for
// the spawn point and an arrow along +x for the heading (a robot's +x is
// usually its front). Both sides are drawn.
function spawnMarkerShape() {
  const up = (pts) => ({ pos: pts.flat(), nrm: pts.flatMap(() => [0, 0, 1]) });
  const ring = grid(1, SEG, (i, j) => {
    const t = (j / SEG) * 2 * Math.PI, r = i ? 0.85 : 1;
    return [r * Math.cos(t), r * Math.sin(t), 0, 0, 0, 1];
  });
  const shaft = { ...up([[0, -0.06, 0], [0.75, -0.06, 0], [0.75, 0.06, 0], [0, 0.06, 0]]), idx: [0, 1, 2, 0, 2, 3] };
  const head = { ...up([[0.75, -0.22, 0], [1.25, 0, 0], [0.75, 0.22, 0]]), idx: [0, 1, 2] };
  const out = merge(ring, shaft, head);
  for (let k = 0, m = out.idx.length; k < m; k += 3) out.idx.push(out.idx[k], out.idx[k + 2], out.idx[k + 1]);
  return out;
}

// Triangle soup from mesh_vert / mesh_face with MuJoCo's own per-corner
// normals (mesh_facenormal -> mesh_normal): smooth where the mesh is smooth,
// split at the creases MuJoCo detected (or the normals the OBJ file provided).
function mjMesh(model, id) {
  const vadr = model.mesh_vertadr[id], nadr = model.mesh_normaladr[id];
  const fadr = model.mesh_faceadr[id], nf = model.mesh_facenum[id];
  const vert = model.mesh_vert, normal = model.mesh_normal;
  const face = model.mesh_face, facenormal = model.mesh_facenormal;
  const tadr = model.mesh_texcoordadr[id];
  const texcoord = model.mesh_texcoord, facetexcoord = model.mesh_facetexcoord;
  const pos = new Float32Array(9 * nf), nrm = new Float32Array(9 * nf), idx = new Uint32Array(3 * nf);
  const uv = tadr >= 0 ? new Float32Array(6 * nf) : null;
  for (let c = 0; c < 3 * nf; c++) {
    const v = vadr + face[3 * fadr + c], n = nadr + facenormal[3 * fadr + c];
    for (let k = 0; k < 3; k++) {
      pos[3 * c + k] = vert[3 * v + k];
      nrm[3 * c + k] = normal[3 * n + k];
    }
    if (uv) {
      const t = tadr + facetexcoord[3 * fadr + c];
      uv[2 * c] = texcoord[2 * t];
      uv[2 * c + 1] = texcoord[2 * t + 1];
    }
    idx[c] = c;
  }
  return { pos, nrm, idx, uv };
}

// UVs for a textured geom: the mesh's own texcoords times texrepeat, or, for
// primitives and meshes without texcoords, an object-space planar projection
// onto the axis plane the vertex normal faces most (so each box face gets its
// own undistorted mapping). texuniform makes texrepeat "per metre", otherwise
// the repeat spans the geom's size.
function geomUVs(model, g, shape, matid) {
  const [ru, rv] = model.mat_texrepeat.subarray(2 * matid, 2 * matid + 2);
  const n = shape.pos.length / 3, uv = new Float32Array(2 * n);
  if (shape.uv) {
    for (let i = 0; i < n; i++) { uv[2 * i] = shape.uv[2 * i] * ru; uv[2 * i + 1] = shape.uv[2 * i + 1] * rv; }
    return uv;
  }
  let su = ru, sv = rv;
  if (!model.mat_texuniform[matid]) {
    const s = model.geom_size;
    su = ru / (2 * (s[3 * g] || 10));
    sv = rv / (2 * (s[3 * g + 1] || s[3 * g] || 10));
  }
  const { pos, nrm } = shape;
  for (let i = 0; i < n; i++) {
    const ax = Math.abs(nrm[3 * i]), ay = Math.abs(nrm[3 * i + 1]), az = Math.abs(nrm[3 * i + 2]);
    // (u, v) axes: z-facing -> (x, y), x-facing -> (y, z), y-facing -> (x, z)
    const [a, b] = az >= ax && az >= ay ? [0, 1] : ax >= ay ? [1, 2] : [0, 2];
    uv[2 * i] = pos[3 * i + a] * su;
    uv[2 * i + 1] = pos[3 * i + b] * sv;
  }
  return uv;
}

// RGBA8 pixels (row 0 = v 0) -> Filament sRGB texture with mipmaps.
function uploadTexture(engine, w, h, rgba) {
  const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;
  const U = Filament.Texture$Usage;
  const tex = Filament.Texture.Builder()
    .width(w).height(h)
    .levels(levels)
    .sampler(Filament.Texture$Sampler.SAMPLER_2D)
    .format(Filament.Texture$InternalFormat.SRGB8_A8)
    // generateMipmaps() renders into the texture, so it must be an attachment.
    .usage(U.DEFAULT.value | U.COLOR_ATTACHMENT.value)
    .build(engine);
  tex.setImage(engine, 0, Filament.PixelBuffer(rgba, Filament.PixelDataFormat.RGBA, Filament.PixelDataType.UBYTE));
  if (levels > 1) tex.generateMipmaps(engine);
  return tex;
}

// Sky cubemap: a vertical gradient (the scene is Z-up), zenith blue to a pale
// horizon, then a darker haze below it. Faces +X, -X, +Y, -Y, +Z, -Z.
function skyTexture(engine, size = 128) {
  const zenith = [0.22, 0.42, 0.72], horizon = [0.78, 0.85, 0.92], ground = [0.36, 0.39, 0.43];
  const faces = [
    (u, v) => [1, -v, -u], (u, v) => [-1, -v, u],
    (u, v) => [u, 1, v], (u, v) => [u, -1, -v],
    (u, v) => [u, -v, 1], (u, v) => [-u, -v, -1],
  ];
  const srgb = (c) => Math.round(255 * Math.min(1, Math.max(0, c)));
  const px = new Uint8Array(6 * size * size * 4);
  let o = 0;
  for (const dir of faces) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const d = dir((2 * x + 1) / size - 1, (2 * y + 1) / size - 1);
        const z = d[2] / Math.hypot(...d);
        const [a, b, t] = z >= 0 ? [horizon, zenith, Math.pow(z, 0.6)] : [horizon, ground, Math.min(1, -z * 4)];
        for (let k = 0; k < 3; k++) px[o++] = srgb(a[k] + (b[k] - a[k]) * t);
        px[o++] = 255;
      }
    }
  }
  const tex = Filament.Texture.Builder()
    .width(size).height(size).levels(1)
    .sampler(Filament.Texture$Sampler.SAMPLER_CUBEMAP)
    .format(Filament.Texture$InternalFormat.SRGB8_A8)
    .build(engine);
  tex.setImageCube(engine, 0, Filament.PixelBuffer(px, Filament.PixelDataFormat.RGBA, Filament.PixelDataType.UBYTE));
  return tex;
}

// A texture MuJoCo decoded itself (builtins, cube faces, …) from tex_data.
function mjTexture(engine, model, t) {
  const w = model.tex_width[t], h = model.tex_height[t], nc = model.tex_nchannel[t];
  const src = model.tex_data.subarray(model.tex_adr[t], model.tex_adr[t] + w * h * nc);
  const rgba = new Uint8Array(4 * w * h);
  for (let i = 0; i < w * h; i++) {
    const r = src[i * nc], g = nc >= 3 ? src[i * nc + 1] : r, b = nc >= 3 ? src[i * nc + 2] : r;
    rgba[4 * i] = r; rgba[4 * i + 1] = g; rgba[4 * i + 2] = b;
    rgba[4 * i + 3] = nc === 4 ? src[i * nc + 3] : nc === 2 ? src[i * nc + 1] : 255;
  }
  return uploadTexture(engine, w, h, rgba);
}

// An image file decoded by the browser (off the main thread) -> RGBA8 pixels,
// top row first like MuJoCo's own PNG loader, scaled down to maxTexture on
// its longer side (the Render panel's Textures): GPU memory, not looks, is
// what limits a scene (a RoboCasa kitchen has ten 2048² textures; at 4/3 ×
// 16 MB each with mipmaps they and a few more scenes take a laptop's or
// phone's GPU past what it has, and the page loses its WebGL context).
let maxTexture = 1024;
async function decodeImage(bytes) {
  const bmp = await createImageBitmap(new Blob([bytes]), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  const k = Math.min(1, maxTexture / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * k)), h = Math.max(1, Math.round(bmp.height * k));
  const ctx = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  return { w, h, rgba: new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer) };
}

// mjtGeom: PLANE=0 HFIELD=1 SPHERE=2 CAPSULE=3 ELLIPSOID=4 CYLINDER=5 BOX=6 MESH=7
function geomShape(model, g) {
  const s = model.geom_size.subarray(3 * g, 3 * g + 3);
  switch (model.geom_type[g]) {
    case 0: return plane(s[0], s[1]);
    case 2: return roundShape(s[0], s[0], s[0]);
    case 3: return roundShape(s[0], s[0], s[0], s[1]);
    case 4: return roundShape(s[0], s[1], s[2]);
    case 5: return cylinder(s[0], s[1]);
    case 6: return box(s[0], s[1], s[2]);
    case 7: return mjMesh(model, model.geom_dataid[g]);
    default: return null;  // hfield / sdf not handled in this example
  }
}

// ---------------------------------------------------------------------------
// Filament helpers

// GPU buffers for one shape: {vb, ib, box}. `uv` (Float32Array, 2 per
// vertex) is only passed for textured materials.
function makeGeometry(engine, shape, uv = null) {
  const pos = new Float32Array(shape.pos);
  const nrm = new Float32Array(shape.nrm);
  const idx = new Uint32Array(shape.idx);
  const n = pos.length / 3;

  // Lit materials need tangent frames (quaternions), derived here from normals.
  // normals() is a JS shim that does not return the builder (no chaining);
  // build() also frees the builder.
  const sob = new Filament.SurfaceOrientation$Builder();
  sob.vertexCount(n);
  sob.normals(nrm, 0);
  const so = sob.build();
  const quats = so.getQuats(n);
  so.delete();

  const AT = Filament.VertexBuffer$AttributeType, VA = Filament.VertexAttribute;
  const vbb = Filament.VertexBuffer.Builder()
    .vertexCount(n).bufferCount(uv ? 3 : 2)
    .attribute(VA.POSITION, 0, AT.FLOAT3, 0, 12)
    .attribute(VA.TANGENTS, 1, AT.SHORT4, 0, 8)
    .normalized(VA.TANGENTS);
  if (uv) vbb.attribute(VA.UV0, 2, AT.FLOAT2, 0, 8);
  const vb = vbb.build(engine);
  vb.setBufferAt(engine, 0, pos);
  vb.setBufferAt(engine, 1, quats);
  if (uv) vb.setBufferAt(engine, 2, uv);

  const ib = Filament.IndexBuffer.Builder()
    .indexCount(idx.length).bufferType(Filament.IndexBuffer$IndexType.UINT)
    .build(engine);
  ib.setBuffer(engine, idx);

  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i++) {
    lo[i % 3] = Math.min(lo[i % 3], pos[i]);
    hi[i % 3] = Math.max(hi[i % 3], pos[i]);
  }
  const box = {
    center: lo.map((v, k) => (v + hi[k]) / 2),
    halfExtent: lo.map((v, k) => (hi[k] - v) / 2 + 1e-4),
  };
  return { vb, ib, box };
}

// A renderable entity drawing `geo` (makeGeometry) with `matInstance`.
// Several entities may share the same buffers and material instance.
function makeEntity(engine, geo, matInstance, castShadows = true) {
  const entity = Filament.EntityManager.get().create();
  Filament.RenderableManager.Builder(1)
    .boundingBox(geo.box)
    .material(0, matInstance)
    .geometry(0, Filament.RenderableManager$PrimitiveType.TRIANGLES, geo.vb, geo.ib)
    .castShadows(castShadows)
    .receiveShadows(true)
    .build(engine, entity);
  return entity;
}

// Row-major 3x3 rotation of a unit quaternion (w, x, y, z), like mju_quat2Mat.
function quatToMat([w, x, y, z]) {
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

// Simple orbit camera around a Z-up target. Mouse: left drag orbits,
// right/middle drag pans, the wheel zooms. Touch: one finger orbits, two
// pinch to zoom and move together to pan.
class OrbitCamera {
  constructor(canvas, center, dist) {
    this.center = center; this.dist = dist;
    this.azim = -0.6; this.elev = 0.35;
    let last = null, button = 0;
    const touches = new Map();  // pointerId -> [x, y] of the fingers on the canvas
    let pinch = null;           // {d, mid} of the two fingers while pinching
    const two = () => {
      const [a, b] = [...touches.values()];
      return { d: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
    };
    // Drags another handler took over (the IK and goal markers, a pulled body)
    // leave the camera alone, a second finger included.
    this.stop = () => { last = null; pinch = null; };
    canvas.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'touch') {
        touches.set(e.pointerId, [e.clientX, e.clientY]);
        if (this.busy?.()) { this.stop(); return; }
        if (touches.size >= 2) {
          last = null;
          pinch = two();
          canvas.setPointerCapture(e.pointerId);
          return;
        }
      }
      // Shift+drag perturbs (inspect.js); `intercept(e)` claims other drags
      // (the Teleop panel's IK handles).
      if (e.shiftKey || this.intercept?.(e)) { last = null; return; }
      last = [e.clientX, e.clientY]; button = e.button; canvas.setPointerCapture(e.pointerId);
    });
    const up = (e) => {
      touches.delete(e.pointerId);
      if (touches.size < 2) pinch = null;
      last = null;
    };
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointermove', (e) => {
      if (touches.has(e.pointerId)) touches.set(e.pointerId, [e.clientX, e.clientY]);
      if (pinch) {
        if (touches.size < 2 || this.busy?.()) return;
        const now = two();
        this.dist *= pinch.d / now.d;
        this.pan(now.mid[0] - pinch.mid[0], now.mid[1] - pinch.mid[1]);
        pinch = now;
        return;
      }
      if (!last) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      if (button === 0) {
        this.azim -= dx * 0.005;
        this.elev = Math.max(-1.5, Math.min(1.5, this.elev + dy * 0.005));
      } else {  // right/middle drag pans in the view plane
        this.pan(dx, dy);
      }
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.dist *= Math.exp(e.deltaY * 0.001);
    }, { passive: false });
  }

  // Glides the target to point p (over ~0.3 s), keeping the distance and
  // the angles; a pan or another focus on the way takes over.
  focus(p) {
    const from = [...this.center], t0 = performance.now(), id = this.focusing = {};
    const step = () => {
      if (this.focusing !== id) return;
      const k = Math.min(1, (performance.now() - t0) / 300), e = 1 - (1 - k) ** 3;
      for (let i = 0; i < 3; i++) this.center[i] = from[i] + (p[i] - from[i]) * e;
      if (k < 1) requestAnimationFrame(step);
      else this.focusing = null;
    };
    step();
  }

  // Moves the target in the view plane by a pointer motion of (dx, dy) pixels.
  pan(dx, dy) {
    this.focusing = null;
    const k = this.dist * 0.0015;
    const [ca, sa] = [Math.cos(this.azim), Math.sin(this.azim)];
    this.center[0] += (sa * dx - ca * Math.sin(this.elev) * dy) * k;
    this.center[1] += (-ca * dx - sa * Math.sin(this.elev) * dy) * k;
    this.center[2] += Math.cos(this.elev) * dy * k;
  }

  eye() {
    const ce = Math.cos(this.elev);
    return [
      this.center[0] + this.dist * ce * Math.cos(this.azim),
      this.center[1] + this.dist * ce * Math.sin(this.azim),
      this.center[2] + this.dist * Math.sin(this.elev),
    ];
  }
}

// ---------------------------------------------------------------------------

class App {
  constructor(canvas) {
    this.canvas = canvas;
    this.paused = false;
    this.mujoco = this.model = this.data = null;  // the physics world
    this.robots = [];  // spawned, in attach order: {visual, pos, bodyOffset, entities, policy, base, ik, use}
    this.controllers = [];  // the robots' switched-on controllers (a policy, BaseController, IkController), ticked every CONTROL_DT
    this.nextControl = 0;   // sim time of the next policy tick
    this.stepping = null;   // the pending advance(), if any
    this.perturb = new Perturb(this);  // double-click selection, Shift+drag force
    this.panel = null;                 // StagePanel, set by the page
    this.teleop = null;                // TeleopPanel, set by the page
    this.stats = null;                 // stats.js meter, set by the page
    this.net = null;                   // NetClient in network mode (net.js): the server steps the world
    this.netVersion = -1;              // the server scene this.model describes

    // Chrome turns WebGL off for a page for a while after its GPU process
    // crashed or the page lost its context (e.g. out of GPU memory).
    if (!document.createElement('canvas').getContext('webgl2')) {
      throw new Error('WebGL 2 is not available. If the page just lost its GPU context (too much in the scene), '
        + 'the browser has turned WebGL off for it for now: close this tab and open the page in a new one, or restart '
        + 'the browser (chrome://gpu shows whether WebGL is blocked).');
    }
    // A lost context does not come back for Filament: say so (the frame stays frozen).
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      setBusy('The GPU context was lost, most likely out of GPU memory (large scenes and many textures). Reload the page; '
        + 'if WebGL then fails to start, open the page in a new tab or restart the browser.', true);
      contextLost = true;
    });
    // xrCompatible: the same context renders the WebXR session (enterVR).
    const engine = this.engine = Filament.Engine.create(canvas, { xrCompatible: true });
    this.scene = engine.createScene();
    this.swapChain = engine.createSwapChain();
    this.renderer = engine.createRenderer();
    this.camera = engine.createCamera(Filament.EntityManager.get().create());
    this.view = engine.createView();
    this.view.setCamera(this.camera);
    this.view.setScene(this.scene);
    this.quality = loadQuality();      // the Render panel's settings (render.js), applied below
    this.renderer.setClearOptions({ clearColor: [0.1, 0.11, 0.13, 1], clear: true });

    this.matLit = engine.createMaterial(MAT_LIT);
    this.matChecker = engine.createMaterial(MAT_CHECKER);
    this.matTextured = engine.createMaterial(MAT_TEXTURED);
    this.matTransparent = engine.createMaterial(MAT_TRANSPARENT);
    this.matTexturedTransparent = engine.createMaterial(MAT_TEXTURED_TRANSPARENT);
    const F = Filament;
    this.sampler = new F.TextureSampler(F.MinFilter.LINEAR_MIPMAP_LINEAR, F.MagFilter.LINEAR, F.WrapMode.REPEAT);
    this.sampler.setAnisotropy(8);
    // Bound to textured materials until their image has been decoded.
    this.whiteTexture = uploadTexture(engine, 1, 1, new Uint8Array([255, 255, 255, 255]));
    this.images = new Map();  // image file(s) -> shared texture entry (bindTexture)

    // Sun with shadows + flat-ish ambient (SH band 0 only) so nothing is pitch black.
    this.sun = Filament.EntityManager.get().create();
    Filament.LightManager.Builder(Filament.LightManager$Type.SUN)
      .color([1.0, 0.97, 0.92])
      .intensity(90000)
      .direction([0.4, 0.3, -1.0])
      .castShadows(true)
      .sunAngularRadius(1.5)
      .build(engine, this.sun);
    this.scene.addEntity(this.sun);
    const sh = new Float32Array(27);
    sh.set([0.65, 0.7, 0.8], 0);
    this.scene.setSkybox(Filament.Skybox.Builder().environment(skyTexture(engine)).build(engine));
    this.scene.setIndirectLight(
      Filament.IndirectLight.Builder().irradianceSh(3, sh).intensity(25000).build(engine));

    // The floor of world.js's base XML (a size-0, i.e. infinite, plane), drawn
    // 40 m x 40 m: 40 x 40 checker cells.
    const floor = this.matChecker.createInstance();
    floor.setColor3Parameter('color1', Filament.RgbType.sRGB, [0.25, 0.3, 0.36]);
    floor.setColor3Parameter('color2', Filament.RgbType.sRGB, [0.45, 0.5, 0.56]);
    floor.setFloatParameter('scale', 1.0);  // checker cell size (m)
    // Pushed back in depth: an environment's own floor usually lies exactly
    // at z = 0 too, and two coplanar surfaces z-fight (flicker between each
    // other with the view). Offset, the world floor always loses such ties.
    floor.setPolygonOffset(1, 1);
    this.scene.addEntity(makeEntity(engine, makeGeometry(engine, plane(20, 20)), floor));

    // Perturbation markers: the selection point, the drag target and a rope
    // between them. Hidden (scaled to zero) when unused.
    const marker = (rgb, geo) => {
      const mi = this.matLit.createInstance();
      mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [...rgb, 1]);
      mi.setFloatParameter('roughness', 0.4);
      mi.setFloatParameter('metallic', 0.0);
      const e = makeEntity(engine, geo, mi);
      this.scene.addEntity(e);
      return e;
    };
    const sphere = makeGeometry(engine, roundShape(1, 1, 1));
    this.markers = {
      sel: marker([1, 0.45, 0.1], sphere),
      target: marker([1, 0.8, 0.2], sphere),
      rope: marker([1, 0.65, 0.15], makeGeometry(engine, cylinder(1, 0.5))),
    };
    // Teleop arm IK targets: an RViz-style interactive marker per chain
    // (marker.js; TeleopPanel.gizmos()), parts 0-2 arrows, 3-5 rings, 6 the ball.
    const arrows = makeGeometry(engine, markerArrows());
    const ring = makeGeometry(engine, torus(MARKER.ring, MARKER.tube));
    const ball = makeGeometry(engine, roundShape(MARKER.ball, MARKER.ball, MARKER.ball));
    this.gizmoParts = CHAIN_COLORS.map((center) => [arrows, arrows, arrows, ring, ring, ring, ball].map((geo, part) => {
      const mi = this.matLit.createInstance();
      mi.setFloatParameter('roughness', 0.5);
      mi.setFloatParameter('metallic', 0.0);
      const e = makeEntity(engine, geo, mi, false);
      this.scene.addEntity(e);
      const color = part === 6 ? center : AXIS_COLORS[part % 3];
      return { e, mi, color, shown: null };
    }));
    const spawnMi = this.matLit.createInstance();
    spawnMi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [0.2, 0.8, 1, 1]);
    spawnMi.setFloatParameter('roughness', 0.6);
    spawnMi.setFloatParameter('metallic', 0.0);
    this.spawnMarker = makeEntity(engine, makeGeometry(engine, spawnMarkerShape()), spawnMi, false);
    this.scene.addEntity(this.spawnMarker);
    // From the floor up to a raised spawn marker.
    this.spawnPole = makeEntity(engine, makeGeometry(engine, cylinder(1, 0.5)), spawnMi, false);
    this.scene.addEntity(this.spawnPole);
    // The Teleop panel's goal marker (goal.js): the spawn marker's shape in
    // orange, brighter while hovered or dragged. Drawn last and over
    // everything, so a robot's base standing on it does not hide it.
    this.goalMi = this.matLit.createInstance();
    this.goalMi.setFloatParameter('roughness', 0.6);
    this.goalMi.setFloatParameter('metallic', 0.0);
    this.goalMi.setDepthCulling(false);
    this.goalMarker = makeEntity(engine, makeGeometry(engine, spawnMarkerShape()), this.goalMi, false);
    {
      const rcm = engine.getRenderableManager(), inst = rcm.getInstance(this.goalMarker);
      rcm.setPriority(inst, 7);
      inst.delete();
    }
    this.scene.addEntity(this.goalMarker);
    this.goalHot = null;

    // World origin: X/Y/Z axes (red/green/blue, like the gizmo) of a fixed
    // world length, resting on the floor, and a small ball at the origin.
    const cyl = makeGeometry(engine, cylinder(1, 0.5));
    this.originMarker = [
      [marker([0.9, 0.25, 0.25], cyl), (L, r) => [0, r, 0, 0, 0, 0, r, 0, L, 0, 0, 0, L / 2, 0, r, 1]],
      [marker([0.3, 0.7, 0.35], cyl), (L, r) => [0, 0, r, 0, r, 0, 0, 0, 0, L, 0, 0, 0, L / 2, r, 1]],
      [marker([0.25, 0.55, 0.95], cyl), (L, r) => [r, 0, 0, 0, 0, r, 0, 0, 0, 0, L, 0, 0, 0, r + L / 2, 1]],
      [marker([0.92, 0.92, 0.92], sphere), (L, r) => [2.5 * r, 0, 0, 0, 0, 2.5 * r, 0, 0, 0, 0, 2.5 * r, 0, 0, 0, r, 1]],
    ];
    this.setOriginMarker(true);

    this.orbit = new OrbitCamera(canvas, [0, 0, 0.4], 4);
    this.setSpawnMarker(null);
    this.setQuality(this.quality);  // sizes the canvas too
    window.addEventListener('resize', () => this.resize());
  }

  // The Render panel's settings (render.js): `q` as loadQuality returns it.
  setQuality(q) {
    const textures = q.textures !== this.quality.textures;
    this.quality = q;
    maxTexture = q.textures;
    if (textures) for (const e of this.images.values()) e.reload?.();  // the scene's image textures at the new size
    this.viewQuality(this.view);
    for (const { view } of this.xr?.eyes ?? []) this.viewQuality(view);
    const lm = this.engine.getLightManager(), sun = lm.getInstance(this.sun);
    lm.setShadowCaster(sun, q.shadows !== 'off');
    sun.delete();
    this.resize();
  }

  // Anti-aliasing and ambient occlusion of `view` (the window's, or a WebXR eye's).
  viewQuality(view) {
    const q = this.quality, F = Filament, msaa = q.antialiasing === 'msaa';
    view.setAntiAliasing(q.antialiasing === 'fxaa' ? F.View$AntiAliasing.FXAA : F.View$AntiAliasing.NONE);
    view.setMultiSampleAntiAliasingOptions({ enabled: msaa, sampleCount: msaa ? 4 : 1 });
    const high = q.ao === 'high', Q = F.View$QualityLevel;
    view.setAmbientOcclusionOptions({
      enabled: q.ao !== 'off', radius: 0.2,
      resolution: high ? 1 : 0.5, quality: high ? Q.HIGH : Q.LOW, upsampling: high ? Q.HIGH : Q.LOW,
    });
  }

  resize() {
    if (this.xr) return;  // the session sizes the canvas
    const screen = window.devicePixelRatio || 1, pr = this.quality.pixelRatio;
    const dpr = pr === 'screen' ? screen : Math.min(screen, pr);
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    this.canvas.width = w; this.canvas.height = h;
    this.view.setViewport([0, 0, w, h]);
    this.camera.setProjectionFov(45, w / h, 0.01, 200, Filament.Camera$Fov.VERTICAL);
  }

  // Render data for one robot model: one part per visible geom, with GPU
  // buffers, a material instance and the geom's pose in its body. Shared by
  // every spawned copy, so it is built once per robot file. `images` (texture
  // name -> candidate paths, readable with `read`) supplies the image textures
  // the loader replaced with placeholders (see vfs.js).
  // Free it with freeVisual once nothing uses it.
  buildVisual(mujoco, model, images, read) {
    const textures = new Map();  // MuJoCo texture id -> texture entry (bindTexture)
    const parts = [];
    for (let g = 0; g < model.ngeom; g++) {
      const matid = model.geom_matid[g];
      const rgba = matid >= 0 ? model.mat_rgba.slice(4 * matid, 4 * matid + 4)
                              : model.geom_rgba.slice(4 * g, 4 * g + 4);
      if (model.geom_group[g] >= 3 || rgba[3] === 0) continue;  // MuJoCo hides these by default
      const shape = geomShape(model, g);
      if (!shape) { console.warn(`geom ${g}: type ${model.geom_type[g]} not supported`); continue; }

      // mat_texid is nmat x mjNTEXROLE; role 1 (mjTEXROLE_RGB) is the color map.
      // Only 2D textures are drawn; cube textures fall back to the plain color.
      const nrole = model.nmat ? model.mat_texid.length / model.nmat : 0;
      const texid = matid >= 0 ? model.mat_texid[nrole * matid + 1] : -1;
      const textured = texid >= 0 && model.tex_type[texid] === 0;

      // Alpha below 1 is drawn blended (back to front per renderable, so a
      // self-overlapping or interpenetrating transparent mesh can sort wrong).
      const transparent = rgba[3] < 1;
      let mi, uv = null;
      if (textured) {
        mi = (transparent ? this.matTexturedTransparent : this.matTextured).createInstance();
        this.bindTexture(textures, mujoco, model, images, read, texid, mi);
        uv = geomUVs(model, g, shape, matid);
      } else {
        mi = (transparent ? this.matTransparent : this.matLit).createInstance();
      }
      mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [rgba[0], rgba[1], rgba[2], transparent ? rgba[3] : 1]);
      mi.setFloatParameter('roughness', 0.45);
      mi.setFloatParameter('metallic', 0.0);
      parts.push({
        ...makeGeometry(this.engine, shape, uv), mi, transparent,
        body: model.geom_bodyid[g],
        pos: Array.from(model.geom_pos.subarray(3 * g, 3 * g + 3)),
        rot: quatToMat(model.geom_quat.subarray(4 * g, 4 * g + 4)),
      });
    }
    parts.textures = [...new Set(textures.values())];
    for (const e of parts.textures) e.users++;
    return parts;
  }

  // Gives back the GPU memory of buildVisual's `parts` (their entities
  // removed already): buffers, material instances, and the textures no other
  // model shares.
  freeVisual(parts) {
    const mis = new Set(parts.map((p) => p.mi));
    for (const p of parts) {
      this.engine.destroyVertexBuffer(p.vb);
      this.engine.destroyIndexBuffer(p.ib);
      this.engine.destroyMaterialInstance(p.mi);
    }
    for (const e of parts.textures ?? []) {
      e.mis = e.mis.filter((mi) => !mis.has(mi));
      if (--e.users > 0) continue;
      if (e.tex) this.engine.destroyTexture(e.tex);
      e.tex = null;
      e.dead = true;  // an image still decoding is dropped when it arrives
      if (e.key) this.images.delete(e.key);
    }
  }

  // Binds MuJoCo texture `texid` to `mi`, creating it on first use. Image
  // textures are shared by every model that uses the same file (this.images:
  // a RoboCasa kitchen alone names its 19 images 69 times) and decoded
  // asynchronously; until then `mi` shows plain white.
  bindTexture(textures, mujoco, model, images, read, texid, mi) {
    let entry = textures.get(texid);
    if (!entry) {
      const name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE.value, texid);
      const cands = images.get(name);
      const key = cands && cands.join('\n');
      entry = key && this.images.get(key);
      if (!entry) {
        entry = { tex: null, mis: [], users: 0, key, gen: 0 };
        if (cands) {
          this.images.set(key, entry);
          entry.reload = () => this.loadImage(entry, read, cands, name);
          entry.reload();
        } else {
          entry.tex = mjTexture(this.engine, model, texid);
        }
      }
      textures.set(texid, entry);
    }
    entry.mis.push(mi);
    mi.setTextureParameter('baseColorMap', entry.tex ?? this.whiteTexture, this.sampler);
  }

  // Decodes an image texture (again, at the current maxTexture: reload) and
  // puts it on every material instance that uses it.
  async loadImage(entry, read, cands, name) {
    const gen = ++entry.gen;
    try {
      const { w, h, rgba } = await decodeImage(await readFirst(read, cands));
      if (entry.dead || gen !== entry.gen) return;  // freed, or a newer load under way
      const old = entry.tex;
      entry.tex = uploadTexture(this.engine, w, h, rgba);
      for (const mi of entry.mis) mi.setTextureParameter('baseColorMap', entry.tex, this.sampler);
      if (old) this.engine.destroyTexture(old);
    } catch (err) {
      console.warn(`texture ${name}:`, err);
    }
  }

  addEntities(parts) {
    return parts.map((p) => {
      const e = makeEntity(this.engine, p, p.mi, !p.transparent);  // glass casts no shadow
      this.scene.addEntity(e);
      return e;
    });
  }

  removeEntities(entities) {
    for (const e of entities) {
      this.scene.remove(e);
      this.engine.destroyEntity(e);
    }
  }

  // Swaps in a new physics world: `model` belongs to `mujoco` (a fresh
  // instance), `robots` are its robots in attach order with their bodyOffset
  // set. `init(data)` may set the state before the first mj_forward. The old
  // model and data are deleted; the caller drops the old instance.
  setWorld(mujoco, model, robots, init = null) {
    const data = new mujoco.MjData(model);
    this.perturb.clear();
    init?.(data);
    mujoco.mj_forward(model, data);
    this.data?.delete();
    this.model?.delete();
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.robots = robots;
    this.refreshControllers();
    this.nextControl = data.time;
    this.resync();
    this.syncTransforms();
    this.panel?.build();
    this.teleop?.build();
  }

  // Network mode: swaps in the server's world. `model` and `data` are
  // net.js stand-ins (worldModel, worldData): tables for the panels and the
  // state the server sends; nothing is compiled here.
  setNetWorld(model, data, robots) {
    this.perturb.clear();
    this.mujoco = NET_MUJOCO;
    this.model = model;
    this.data = data;
    this.robots = robots;
    this.controllers = [];
    this.panel?.build();
    this.teleop?.build();
  }

  // The controllers to tick: each robot's switched-on ones (robot.use).
  refreshControllers() {
    this.controllers = this.robots.flatMap((r) => {
      if (r.ik) r.ik.enabled = !!r.use?.ik;
      return [r.use?.policy && r.policy, r.use?.base && r.base, r.use?.ik && r.ik].filter(Boolean);
    });
  }

  // Every robot back to its spawn pose (robot.initial), time back to 0.
  reset() {
    if (this.net) { this.net.send('reset'); return; }
    this.mujoco.mj_resetData(this.model, this.data);
    const layout = stateLayout(this.robots);
    for (const r of this.robots) this.writeInitial(r, layout.get(r));
    this.mujoco.mj_forward(this.model, this.data);
    for (const c of this.controllers) c.reset();
    this.teleop?.clearGoals(this.robots);
    this.nextControl = 0;
    this.resync();
    this.syncTransforms();  // frame() only syncs while running
  }

  // One robot back to its spawn pose, at rest; the others and time go on.
  resetRobot(robot) {
    if (this.net) { this.net.send('reset_robot', { id: robot.netId }); return; }
    const { mujoco, model, data } = this;
    const at = stateLayout(this.robots).get(robot);
    if (!at) return;
    this.writeInitial(robot, at);
    data.qvel.fill(0, at.qvel, at.qvel + robot.visual.dims.nv);
    data.act.fill(0, at.act, at.act + robot.visual.dims.na);
    mujoco.mj_forward(model, data);
    for (const c of [robot.policy, robot.base, robot.ik]) c?.reset();
    this.teleop?.clearGoals([robot]);
    this.syncTransforms();
  }

  // The free joint of the robot's root body, or -1 (a fixed base: placed by
  // its attach frame, robot.pos / robot.yaw).
  rootFreeJoint(robot) {
    const m = this.model, b = robot.bodyOffset;
    if (!m || !(m.body_jntnum[b] > 0)) return -1;
    const j = m.body_jntadr[b];
    return m.jnt_type[j] === 0 ? j : -1;
  }

  // Moves a robot on a free joint to `pos` with orientation `quat` (w, x, y,
  // z), at rest, at once (no rebuild); also its spawn pose (Reset, saved
  // stage). False for a fixed base.
  placeFree(robot, pos, quat) {
    const j = this.rootFreeJoint(robot);
    if (j < 0) return false;
    if (this.net) { this.net.send('place', { id: robot.netId, pos, quat }); return true; }
    const { mujoco, model, data } = this, adr = model.jnt_qposadr[j], dof = model.jnt_dofadr[j];
    data.qpos.set(pos, adr);
    data.qpos.set(quat, adr + 3);
    data.qvel.fill(0, dof, dof + 6);
    const at = stateLayout(this.robots).get(robot);
    if (robot.initial && at) robot.initial.qpos.set(data.qpos.subarray(adr, adr + 7), adr - at.qpos);
    robot.pos = [...pos];
    robot.yaw = Math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3]));
    mujoco.mj_forward(model, data);
    this.syncTransforms();
    return true;
  }

  writeInitial(robot, at) {
    if (!robot.initial) return;
    this.data.qpos.set(robot.initial.qpos, at.qpos);
    this.data.ctrl.set(robot.initial.ctrl, at.ctrl);
  }

  // The camera ray under the pointer: {eye, dir (unit), forward (unit view direction)}.
  pointerRay(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = 1 - ((clientY - rect.top) / rect.height) * 2;
    const eye = this.orbit.eye(), c = this.orbit.center;
    const norm = (v) => { const l = Math.hypot(...v); return v.map((k) => k / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const f = norm(c.map((k, i) => k - eye[i]));
    const r = norm(cross(f, [0, 0, 1])), u = cross(r, f);
    const t = Math.tan((45 / 2) * (Math.PI / 180));  // matches setProjectionFov(45, …, VERTICAL)
    const dir = norm([0, 1, 2].map((i) => f[i] + r[i] * x * t * (rect.width / rect.height) + u[i] * y * t));
    return { eye, dir, forward: f };
  }

  // Where the pointer ray hits the floor (or the plane at height z), or null
  // if it misses.
  pickFloor(clientX, clientY, z = 0) {
    const { eye, dir: d } = this.pointerRay(clientX, clientY);
    if (eye[2] - z <= 0 || d[2] > -1e-6) return null;
    const s = (z - eye[2]) / d[2];
    return [eye[0] + s * d[0], eye[1] + s * d[1], z];
  }

  // The first physics geom under the pointer: {body, point}, or null. Body 0
  // is the world (the floor, static geoms). Visual-only geoms are not in the
  // physics world, so this hits collision geoms.
  pickBody(clientX, clientY) {
    const { mujoco, model, data } = this;
    if (!model) return null;
    const { eye, dir } = this.pointerRay(clientX, clientY);
    if (this.net) {  // a promise: the server casts the ray
      return this.net.request('ray', { eye, dir }).then((hit) => (hit.body >= 0 ? hit : null), () => null);
    }
    const geom = new mujoco.IntBuffer(1), normal = new mujoco.DoubleBuffer(3);
    try {
      const dist = mujoco.mj_ray(model, data, eye, dir, null, true, -1, geom, normal);
      const g = geom.GetView()[0];
      if (dist < 0 || g < 0) return null;
      return { body: model.geom_bodyid[g], point: eye.map((v, i) => v + dist * dir[i]) };
    } finally {
      geom.delete();
      normal.delete();
    }
  }

  // Shows or hides the world-origin axes (ORIGIN_AXIS_LENGTH long).
  setOriginMarker(visible) {
    const tcm = this.engine.getTransformManager();
    for (const [entity, matrix] of this.originMarker) {
      const inst = tcm.getInstance(entity);
      tcm.setTransform(inst, visible ? matrix(ORIGIN_AXIS_LENGTH, ORIGIN_AXIS_RADIUS)
                                     : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
      inst.delete();
    }
  }

  // Shows the drag-to-spawn marker at `pos` heading `yaw`, with a pole down
  // to the floor when it is raised, or hides it (null). It has a fixed size
  // in the world (SPAWN_MARKER_RADIUS), so it looks bigger up close and
  // smaller far away, like the models it places.
  setSpawnMarker(pos, yaw = 0) {
    const tcm = this.engine.getTransformManager();
    const put = (entity, m) => {
      const inst = tcm.getInstance(entity);
      tcm.setTransform(inst, m ?? [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
      inst.delete();
    };
    const k = SPAWN_MARKER_RADIUS, c = Math.cos(yaw) * k, s = Math.sin(yaw) * k;
    put(this.spawnMarker, pos && [c, s, 0, 0, -s, c, 0, 0, 0, 0, k, 0, pos[0], pos[1], pos[2] + 0.003, 1]);
    const h = pos?.[2] ?? 0, w = 0.06 * k;
    put(this.spawnPole, Math.abs(h) > 1e-3 ? [w, 0, 0, 0, 0, w, 0, 0, 0, 0, h, 0, pos[0], pos[1], h / 2, 1] : null);
  }

  // Places the perturbation markers (see the constructor).
  syncMarkers() {
    const tcm = this.engine.getTransformManager();
    const put = (entity, m) => {
      const inst = tcm.getInstance(entity);
      tcm.setTransform(inst, m ?? [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
      inst.delete();
    };
    const p = this.model ? this.perturb.point() : null, q = this.perturb.target;
    const r = 0.025;  // m: a 5 cm ball, whatever the zoom
    const ball = (c, k) => [k, 0, 0, 0, 0, k, 0, 0, 0, 0, k, 0, c[0], c[1], c[2], 1];
    put(this.markers.sel, p && ball(p, r));
    put(this.markers.target, p && q && ball(q, 0.7 * r));
    let rope = null;
    if (p && q) {
      const d = q.map((v, i) => v - p[i]), len = Math.hypot(...d);
      if (len > 1e-6) {
        const z = d.map((v) => v / len);
        const a = Math.abs(z[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
        let x = [a[1] * z[2] - a[2] * z[1], a[2] * z[0] - a[0] * z[2], a[0] * z[1] - a[1] * z[0]];
        const xl = Math.hypot(...x);
        x = x.map((v) => v / xl);
        const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
        const w = 0.25 * r, mid = p.map((v, i) => (v + q[i]) / 2);
        rope = [...x.map((v) => v * w), 0, ...y.map((v) => v * w), 0, ...z.map((v) => v * len), 0, ...mid, 1];
      }
    }
    put(this.markers.rope, rope);

    // Goal marker, flat on the floor at its fixed size.
    const goal = this.model ? this.teleop?.goalMarker() : null;
    if (goal) {
      const k = SPAWN_MARKER_RADIUS, c = Math.cos(goal.yaw) * k, s = Math.sin(goal.yaw) * k;
      put(this.goalMarker, [c, s, 0, 0, -s, c, 0, 0, 0, 0, k, 0, goal.pos[0], goal.pos[1], 0.005, 1]);
      if (this.goalHot !== goal.hot) {
        this.goalMi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, goal.hot ? [1, 0.85, 0.2, 1] : [1, 0.5, 0.1, 1]);
        this.goalHot = goal.hot;
      }
    } else {
      put(this.goalMarker, null);
    }

    // IK markers: part frame = marker pose * the part's axis (z -> x/y/z) * scale.
    const gizmos = this.model ? this.teleop?.gizmos() ?? [] : [];
    this.gizmoParts.forEach((parts, i) => {
      const g = gizmos[i];
      parts.forEach((p, part) => {
        if (!g) { put(p.e, null); return; }
        const L = PART_AXES[part % 3], R = g.rot, k = g.scale;
        const W = [0, 1, 2].map((c) => [0, 1, 2].map((r) => (R[3 * r] * L[c][0] + R[3 * r + 1] * L[c][1] + R[3 * r + 2] * L[c][2]) * k));
        put(p.e, [...W[0], 0, ...W[1], 0, ...W[2], 0, ...g.pos, 1]);
        const color = g.hot === part ? MARKER_HOT : p.color;
        if (p.shown !== color) {
          p.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [...color, 1]);
          p.shown = color;
        }
      });
    });
  }

  // Geom pose = body pose (from the world) * geom pose in the body (from the
  // robot's own model). The robot's body k > 0 is world body bodyOffset + k - 1;
  // its static (worldbody) geoms sit at the spawn point, turned by its yaw.
  // Only the parts whose body moved since the last call: in a large scene
  // most bodies are fixed or asleep, and each update is three calls into
  // Filament.
  syncTransforms() {
    const tcm = this.engine.getTransformManager();
    const xpos = this.data.xpos, xmat = this.data.xmat;
    for (const robot of this.robots) {
      const { visual, pos, yaw = 0, bodyOffset, entities } = robot;
      const c = Math.cos(yaw), s = Math.sin(yaw);
      const Rz = [c, -s, 0, s, c, 0, 0, 0, 1];
      // The robot's body poses as last drawn (12 per body: rotation, position).
      const nb = visual.inertia.nbody;
      if (robot.drawn?.entities !== entities || robot.drawn.pose.length !== 12 * nb) {
        robot.drawn = { entities, pose: new Float64Array(12 * nb).fill(NaN) };
      }
      const drawn = robot.drawn.pose, moved = new Uint8Array(nb);
      for (let k = 0; k < nb; k++) {
        const b = bodyOffset + k - 1, o = 12 * k;
        const B = k > 0 ? xmat.subarray(9 * b, 9 * b + 9) : Rz, t = k > 0 ? xpos.subarray(3 * b, 3 * b + 3) : pos;
        for (let j = 0; j < 12; j++) {
          const v = j < 9 ? B[j] : t[j - 9];
          if (drawn[o + j] !== v) { drawn[o + j] = v; moved[k] = 1; }
        }
      }
      for (const [i, part] of visual.parts.entries()) {
        if (!moved[part.body]) continue;
        let B = Rz, t = pos;
        if (part.body > 0) {
          const b = bodyOffset + part.body - 1;
          B = xmat.subarray(9 * b, 9 * b + 9);
          t = xpos.subarray(3 * b, 3 * b + 3);
        }
        const G = part.rot, q = part.pos;
        const R = new Array(9);
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) R[3 * r + c] = B[3 * r] * G[c] + B[3 * r + 1] * G[3 + c] + B[3 * r + 2] * G[6 + c];
        }
        const p = [0, 1, 2].map((r) => t[r] + B[3 * r] * q[0] + B[3 * r + 1] * q[1] + B[3 * r + 2] * q[2]);
        // Row-major 3x3 -> Filament's column-major 4x4.
        const m = [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, p[0], p[1], p[2], 1];
        const inst = tcm.getInstance(entities[i]);
        tcm.setTransform(inst, m);
        inst.delete();
      }
    }
  }

  // Sim time follows wall time at the World panel's speed from now on:
  // target(t) = (t - simStart) · speed.
  resync() {
    this.simStart = performance.now() - (this.data?.time ?? 0) * 1000 / (this.physics?.speed || 1);
  }

  // Steps the world up to wall time × speed (as far as it gets in 20 ms of
  // each frame, so a hidden tab or the "max" speed doesn't freeze the page),
  // ticking the policies every CONTROL_DT of sim time. Inference is async
  // (onnxruntime-web), so this awaits it between steps; a world swapped in
  // meanwhile (spawn, remove) ends the call.
  async advance() {
    const { mujoco, model, data } = this;
    const valid = () => this.data === data;
    const speed = this.physics.speed;
    const target = speed ? (performance.now() - this.simStart) / 1000 * speed : Infinity;
    const t0 = performance.now();
    while (data.time < target && performance.now() - t0 < 20) {
      if (this.controllers.length && data.time >= this.nextControl - 1e-9) {
        await Promise.all(this.controllers.map((c) => c.step(data, valid)));
        if (!valid()) return;
        this.nextControl += CONTROL_DT;
      }
      this.perturb.apply();
      for (const c of this.controllers) c.substep?.(data);  // software servos (ik.js, base.js)
      mujoco.mj_step(model, data);
    }
    if (data.time < target) this.resync();  // slower than asked: let it slip
    this.syncTransforms();
  }

  // Next to the frame meter: the simulation speed (sim time per wall time,
  // steps per second), and in network mode the round trip, state rate and
  // traffic.
  showStats() {
    const now = performance.now();
    if (now - (this.statsShown ?? 0) < 500) return;
    const el = document.getElementById('net'), last = this.simSample, { model, data } = this;
    this.statsShown = now;
    this.simSample = data && { now, time: data.time, data };
    el.hidden = !model && !this.net;
    const sim = el.querySelector('.sim');
    if (model && last?.data === data && now > last.now) {
      const speed = (data.time - last.time) / ((now - last.now) / 1000);
      const dt = model.opt?.timestep ?? this.physics.timestep;  // net.js's stand-in has no opt: the server's setting
      sim.textContent = `sim ${speed.toFixed(2)}× · ${Math.round(speed / dt)} steps/s`;
      sim.className = `sim ${this.paused || speed >= 0.95 ? 'good' : speed >= 0.7 ? 'fair' : 'poor'}`;
    } else if (!model) sim.textContent = '';
    for (const c of ['.rtt', '.rate']) el.querySelector(c).hidden = !this.net;
    if (!this.net) return;
    const { rtt, hz, rate, delay } = this.net.stats();
    const off = !this.net.connected;
    el.querySelector('.rtt').textContent = off ? 'offline' : rtt == null ? 'RTT …' : `RTT ${Math.round(rtt)} ms`;
    el.querySelector('.rtt').className = `rtt ${off ? 'poor' : rtt == null ? '' : rtt < 80 ? 'good' : rtt < 200 ? 'fair' : 'poor'}`;
    el.querySelector('.rate').textContent = `${Math.round(hz)} Hz · ${(rate / 1024).toFixed(rate < 10240 ? 1 : 0)} KB/s · +${Math.round(delay * 1000)} ms`;
  }

  // `xrFrame` is set when called from the WebXR session's animation loop.
  frame(xrFrame = null) {
    this.stats?.begin();
    if (this.net) {
      // The server steps; show its state (interpolated) once this world matches it.
      if (this.model && this.net.apply(this.model, this.data, this.netVersion, this.robots)) this.syncTransforms();
    } else if (this.model && !this.paused && !this.stepping) {
      this.stepping = this.advance()
        .catch((err) => { this.paused = true; console.error('simulation stopped', err); })
        .finally(() => { this.stepping = null; });
    }
    this.showStats();
    this.syncMarkers();
    this.panel?.update(performance.now());
    if (xrFrame) {
      this.renderXR(xrFrame);
    } else {
      this.camera.lookAt(this.orbit.eye(), this.orbit.center, [0, 0, 1]);
      if (this.renderer.beginFrame(this.swapChain)) {
        this.renderer.renderView(this.view);
        this.renderer.endFrame();
      }
      this.engine.execute();
    }
    this.stats?.end();
    const xr = this.xr;
    // A session frame that arrives after the session ended is dropped: 'end'
    // has restarted the window loop.
    if (xr) xr.session.requestAnimationFrame((_t, f) => this.xr === xr && this.frame(f));
    else requestAnimationFrame(() => this.frame());
  }

  // ---- WebXR (view only, no interaction) ----------------------------------
  //
  // Filament.js cannot render into the session's framebuffer, so each eye is
  // rendered by its own View into its viewport of the (canvas) default
  // framebuffer, sized like the XR layer, and the whole image is then blitted
  // onto the layer. The scene is Z-up, WebXR is Y-up: the 'local-floor' space
  // is mapped onto the scene with the viewer starting 2.5 m back on -Y,
  // looking along +Y at the origin.

  async enterVR() {
    const session = await navigator.xr.requestSession('immersive-vr', { requiredFeatures: ['local-floor'] });
    const gl = this.canvas.getContext('webgl2');  // Filament's own context
    await gl.makeXRCompatible();
    const near = 0.05, far = 200;
    session.updateRenderState({ depthNear: near, depthFar: far });
    // No antialias: a multisampled layer cannot be blitted into.
    const layer = new XRWebGLLayer(session, gl, { antialias: false, depth: false });
    session.updateRenderState({ baseLayer: layer });
    const space = await session.requestReferenceSpace('local-floor');
    const eyes = [0, 1].map(() => {
      const entity = Filament.EntityManager.get().create();
      const camera = this.engine.createCamera(entity);
      const view = this.engine.createView();
      view.setCamera(camera);
      view.setScene(this.scene);
      this.viewQuality(view);
      return { entity, camera, view };
    });
    this.canvas.width = layer.framebufferWidth;
    this.canvas.height = layer.framebufferHeight;
    this.xr = { session, gl, layer, space, eyes, near, far, frames: 0, t0: performance.now() };
    session.addEventListener('end', () => {
      this.xr = null;
      for (const { entity, view } of eyes) {
        this.engine.destroyView(view);
        this.engine.destroyCameraComponent(entity);
      }
      this.resize();
      requestAnimationFrame(() => this.frame());  // the session's loop has stopped
    });
    console.log(`WebXR: ${layer.framebufferWidth}x${layer.framebufferHeight} framebuffer`);
    return session;
  }

  renderXR(frame) {
    const { gl, layer, space, eyes, near, far } = this.xr;
    const pose = frame.getViewerPose(space);
    if (!pose) return;
    // scene_from_xr (column-major): x -> x, y -> z, z -> -y, origin at (0, -2.5, 0).
    const M = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, -2.5, 0, 1];
    const mul = (a, b) => {  // column-major 4x4 a * b
      const r = new Array(16);
      for (let c = 0; c < 4; c++) {
        for (let k = 0; k < 4; k++) {
          r[4 * c + k] = a[k] * b[4 * c] + a[4 + k] * b[4 * c + 1] + a[8 + k] * b[4 * c + 2] + a[12 + k] * b[4 * c + 3];
        }
      }
      return r;
    };
    const views = pose.views.slice(0, eyes.length);
    for (const [i, v] of views.entries()) {
      const { camera, view } = eyes[i];
      const vp = layer.getViewport(v);
      view.setViewport([vp.x, vp.y, vp.width, vp.height]);
      camera.setCustomProjection(Array.from(v.projectionMatrix), near, far);
      camera.setModelMatrix(mul(M, v.transform.matrix));
    }
    if (this.renderer.beginFrame(this.swapChain)) {
      for (let i = 0; i < views.length; i++) this.renderer.renderView(eyes[i].view);
      this.renderer.endFrame();
    } else {
      this.xr.skipped = (this.xr.skipped ?? 0) + 1;
    }
    this.engine.execute();

    // Default framebuffer -> layer. Filament caches GL state, so the scissor
    // test and the framebuffer binding are put back as it left them.
    const w = layer.framebufferWidth, h = layer.framebufferHeight;
    const scissor = gl.isEnabled(gl.SCISSOR_TEST);
    if (scissor) gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, layer.framebuffer);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (scissor) gl.enable(gl.SCISSOR_TEST);

    // Frame rate, for checking performance on the headset (remote devtools).
    const x = this.xr;
    x.frames++;
    const dt = performance.now() - x.t0;
    if (dt > 2000) {
      console.log(`WebXR: ${(1000 * x.frames / dt).toFixed(1)} fps, ${x.skipped ?? 0} frames skipped by Filament`);
      x.frames = 0; x.skipped = 0; x.t0 = performance.now();
    }
  }
}

// Robots are attached in order and each one's joints, actuators and mocap
// bodies stay together, so a robot's qpos/qvel/act/ctrl/mocap entries are a
// contiguous range of each array, sized by its own model (visual.dims).
// stateLayout(robots): robot -> {field: start index in data[field]}.
const STATE_FIELDS = [['qpos', 1], ['qvel', 1], ['act', 1], ['ctrl', 1], ['mocap_pos', 3], ['mocap_quat', 4]];
const DIM_OF = { qpos: 'nq', qvel: 'nv', act: 'na', ctrl: 'nu', mocap_pos: 'nmocap', mocap_quat: 'nmocap' };
function stateLayout(robots) {
  const layout = new Map(), at = { nq: 0, nv: 0, na: 0, nu: 0, nmocap: 0 };
  for (const r of robots) {
    const starts = {};
    for (const [field, n] of STATE_FIELDS) starts[field] = n * at[DIM_OF[field]];
    layout.set(r, starts);
    for (const k in at) at[k] += r.visual.dims[k];
  }
  layout.total = at;
  return layout;
}

// ---------------------------------------------------------------------------

const status = document.getElementById('status');

// Full-screen overlay shown while spawning/compiling; it swallows pointer
// input, and keydowns are dropped here (capture phase, registered first) so
// nothing reaches the page meanwhile. Keyups pass, so held keys are released.
const busyEl = document.getElementById('busy');
const busyMsg = busyEl.querySelector('.msg');
let contextLost = false;  // set by App: its message stays
// `stopped`: nothing more will happen (an error), no spinner.
function setBusy(msg, stopped = false) {
  if (contextLost) return;
  busyEl.hidden = msg == null;
  busyEl.classList.toggle('stopped', stopped);
  if (msg == null) return;
  busyMsg.textContent = msg;
  document.activeElement?.blur?.();  // e.g. the asset filter box
}
window.addEventListener('keydown', (e) => {
  if (busyEl.hidden) return;
  e.preventDefault();
  e.stopImmediatePropagation();
}, { capture: true });

// Lets the browser paint the status text before a blocking compile.
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r)));

// Copies the state of the old world into `data` for every robot of `robots`
// (the new world's) that was in the old one too, wherever its entries moved
// (a robot removed before it shifts them).
function copySharedState(old, data, robots) {
  const now = stateLayout(robots);
  for (const r of robots) {
    const from = old.layout.get(r);
    if (!from) continue;
    const to = now.get(r);
    for (const [field, n] of STATE_FIELDS) {
      const len = n * r.visual.dims[DIM_OF[field]];
      data[field].set(old[field].subarray(from[field], from[field] + len), to[field]);
    }
  }
  data.time = old.time;
}

function snapshot(app) {
  const { model: m, data: d } = app;
  if (!m) return null;
  const snap = { time: d.time, layout: stateLayout(app.robots) };
  for (const [field] of STATE_FIELDS) snap[field] = Float64Array.from(d[field]);
  return snap;
}

try {
  await new Promise((resolve) => Filament.init([MAT_LIT, MAT_CHECKER, MAT_TEXTURED, MAT_TRANSPARENT, MAT_TEXTURED_TRANSPARENT], resolve));
  Filament = window.Filament;
  const app = new App(document.getElementById('canvas'));
  app.panel = new StagePanel(app, document.getElementById('stage'), {
    onRemove: (robot) => respawn(robots.filter((r) => r !== robot)),
    onMove: (robot, pos, yaw) => moveFixed(robot, pos, yaw),
    onSave: () => saveStage(),
    onForget: () => forgetStage(),
  });
  app.teleop = new TeleopPanel(app, document.getElementById('teleop'));
  app.physics = loadPhysics();  // World panel settings (physics.js)
  app.world = new WorldPanel(app, document.getElementById('world'));
  app.renderPanel = new RenderPanel(app, document.getElementById('render'));
  // Frame meter (stats.js) in the bottom-right corner; click it to cycle
  // FPS / ms per frame (JS time of App.frame) / MB of JS heap.
  app.stats = new Stats();
  app.stats.dom.id = 'stats';
  app.stats.dom.style.cssText = '';  // placed by #stats in index.html
  document.body.append(app.stats.dom);
  const gizmo = new AxisGizmo(document.getElementById('gizmo'), app.orbit);
  (function drawGizmo() { gizmo.draw(); requestAnimationFrame(drawGizmo); })();
  window.app = app;  // handy for poking at it from devtools

  // The model library index written by scripts/vendor.mjs (empty without
  // mjcf_assets/).
  const [assets, recent] = await Promise.all([loadAssets(), loadRecent(), loadPolicyIndex()]);
  new ServerPanel(app, document.getElementById('server'), { models: assets.length });
  new SharePanel(document.getElementById('share'));

  // Asset paths (`rel`) are under public/:
  // 'mjcf_assets/<category>/<source>/<model>/<file>.xml'.
  // Every rebuild writes the robots' files into a fresh MuJoCo FS, so keep
  // them in memory.
  const fetched = new Map();
  const publicUrl = urlSource(dataUrl);
  const models = (rel) => {
    if (!fetched.has(rel)) {
      const bytes = publicUrl(rel);
      bytes.catch(() => fetched.delete(rel));  // a failed fetch is retried next time
      fetched.set(rel, bytes);
    }
    return fetched.get(rel);
  };

  // Per robot file (asset rel), shared by its spawned copies: render parts,
  // full-model inertia, first keyframe, names referenced in its XML.
  const visuals = new Map();
  // GPU memory back from the render parts of model files no robot uses any
  // more (after Clear, ✕, a new scene from the server); they are built
  // again if spawned again.
  function dropUnused() {
    for (const cache of [visuals, netVisuals]) {
      for (const [rel, pending] of cache) {
        if (!pending.visual || robots.some((r) => r.visual.rel === rel)) continue;
        cache.delete(rel);
        app.freeVisual(pending.visual.parts);
      }
    }
  }
  function robotVisual(rel, label, progress) {
    if (!visuals.has(rel)) {
      const pending = (async () => {
        const mujoco = await loadMujoco();  // throwaway: dropped after this
        const sig = geomSignature(app.physics);
        const { path, images, docs } = await loadModelFiles(mujoco, models, '/models', rel, progress, editDocs);
        progress(`preparing ${label} for rendering…`);
        await nextFrame();
        const model = mujoco.MjModel.from_xml_path(path);
        try {
          return {
            rel, parts: app.buildVisual(mujoco, model, images, models),
            inertia: bodyInertia(model), sig, key: firstKey(docs), refs: referencedNames(docs),
            dims: { nq: model.nq, nv: model.nv, na: model.na, nu: model.nu, nmocap: model.nmocap },
          };
        } finally {
          model.delete();
        }
      })();
      pending.then((v) => { pending.visual = v; }, () => visuals.delete(rel));  // failed: retry on the next click
      visuals.set(rel, pending);
    }
    return visuals.get(rel);
  }

  // The World panel's default geom properties, written into every model's
  // root default class before it is compiled (physics.js).
  const editDocs = (docs) => injectGeomDefaults(docs, geomOverrides(app.physics));

  // `visual` with the body inertias of its full compile under the current
  // default geom properties (they set masses), compiled again if they
  // changed since it was made.
  const inertias = new Map();  // rel + signature -> Promise of bodyInertia
  async function withPhysics(visual, progress = () => {}) {
    const sig = geomSignature(app.physics);
    if (visual.sig === sig) return visual;
    const k = `${visual.rel}\n${sig}`;
    if (!inertias.has(k)) {
      const pending = (async () => {
        const mujoco = await loadMujoco();
        const { path } = await loadModelFiles(mujoco, models, '/models', visual.rel, progress, editDocs);
        progress(`computing masses of ${visual.rel.split('/').at(-1)}…`);
        await nextFrame();
        const model = mujoco.MjModel.from_xml_path(path);
        try { return bodyInertia(model); } finally { model.delete(); }
      })();
      pending.catch(() => inertias.delete(k));
      inertias.set(k, pending);
    }
    return { ...visual, inertia: await inertias.get(k), sig };
  }

  const robots = [];  // spawned, in attach order: {label, visual, prefix, pos, yaw, entities, bodyOffset, policy, base, ik, use}
  let nextId = 1, busy = false;

  const hint = () => {
    const walk = robots.some((r) => r.use) ? ' · drive it from the Teleop panel' : '';
    return `${robots.length} spawned · drag an asset onto the floor` + walk;
  };

  // Which controllers a robot uses ({policy: a POLICIES id or '', base, ik}),
  // saved per robot file from the Teleop panel. Defaults: the policy made
  // for the robot (GR00T on every G1 variant), the base if one is found
  // (decided at bind), IK always.
  const USE_STORE = 'mujoco-filament/controllers/';
  function loadUse(key) {
    try {
      const saved = JSON.parse(localStorage.getItem(USE_STORE + key) ?? 'null');
      if (saved) {
        if ('groot' in saved) { saved.policy = saved.groot ? 'groot-g1' : ''; delete saved.groot; }  // older choice
        return saved;
      }
    } catch { /* no storage */ }
    return { policy: defaultPolicy(key), base: null, ik: true };
  }

  // Binds the controller of `kind` to the current world; false (with
  // robot.errors[kind] set) if it cannot drive this robot.
  function bindController(r, kind, mujoco, model) {
    const bodies = [r.bodyOffset, r.bodyOffset + r.visual.inertia.nbody - 1];
    const byPolicy = (j) => (r.use.policy && r.policy?.owns(j)) || null;
    const owner = (j) => byPolicy(j) || (r.use.base && r.base?.owns(j)) || null;
    try {
      if (kind === 'policy') r.policy.bind(mujoco, model);
      else r[kind].bind(mujoco, model, kind === 'ik' ? owner : byPolicy, bodies);
      delete r.errors[kind];
      return true;
    } catch (err) {
      r.errors[kind] = err.message ?? String(err);
      return false;
    }
  }

  // Makes robot.policy the controller of the policy r.use.policy names
  // (loading its weights); clears r.use.policy if that fails.
  async function makePolicy(r, progress) {
    const p = policyById(r.use.policy);
    if (r.policy && r.policy.policyId !== p?.id) { r.policy.unbind?.(); r.policy = null; }
    if (!p) { if (r.use.policy) r.errors.policy = `no policy ${r.use.policy}`; r.use.policy = ''; return; }
    try {
      const weights = await loadPolicy(p, progress);
      if (!r.policy) r.policy = Object.assign(p.create(weights, r.prefix), { policyId: p.id });
    } catch (err) {
      console.warn(`policy ${p.id} unavailable`, err);
      r.errors.policy = `${p.name}: could not load (${err.message ?? err})`;
      r.use.policy = '';
    }
  }

  // Switches a robot's controller of `kind` ('policy' | 'base' | 'ik') on or
  // off (Teleop panel), remembering the choice for its robot file. For
  // 'policy', `on` is the POLICIES id to run ('' for none).
  app.setUse = async (r, kind, on) => {
    if (app.net) {  // the server runs the controllers; it tells everyone (applyRobot)
      app.net.send('use', { id: r.netId, kind, on: kind === 'policy' ? on || '' : !!on });
      return;
    }
    // An automatic base setup leaves the policy's joints out (a leg joint
    // can look like a wheel once a policy makes its actuator a torque motor):
    // find it again after the switch, giving back what it holds first.
    const rebase = kind === 'policy' && r.base?.config?.auto;
    if (rebase) r.base.release(app.data);
    if (kind === 'policy') {
      r.policy?.unbind?.();
      r.use.policy = on || '';
      delete r.errors.policy;
      await makePolicy(r, (msg) => { status.textContent = msg; });
      if (r.use.policy) {
        r.policy.reset();
        if (!bindController(r, 'policy', app.mujoco, app.model)) r.use.policy = '';
      }
    } else {
      r.use[kind] = on;
    }
    if (rebase) {
      r.base.config = null;
      bindController(r, 'base', app.mujoco, app.model);
      r.use.base = !!r.base.config?.kind && r.use.base !== false;
    }
    if (kind === 'base') { r.base.reset(); if (on) r.base.resolve(); else r.base.release(app.data); }
    if (kind === 'ik' && !on) r.ik?.release(app.data);
    // What the policy or base drives is off limits to IK: find the chains
    // again (an automatic setup may change, e.g. G1 legs without GR00T).
    if (kind !== 'ik' && r.ik?.config?.auto) r.ik.config = null;
    r.ik?.targets.clear();
    if (r.ik) bindController(r, 'ik', app.mujoco, app.model);
    try { localStorage.setItem(USE_STORE + r.key, JSON.stringify(r.use)); } catch { /* no storage */ }
    app.refreshControllers();
  };

  // Compiles the world for `list` in a fresh MuJoCo instance and swaps it in.
  // `fresh` are the robots of `list` that are not in the current world yet:
  // each starts from its own keyframe, if any. Throws (changing nothing) if
  // compiling fails.
  async function rebuild(list, fresh = [], progress = () => {}) {
    const mujoco = await loadMujoco();
    const specs = [];
    for (const r of list) {
      const { path } = await loadModelFiles(mujoco, models, '/models', r.visual.rel, () => {}, editDocs);
      specs.push({ path, prefix: r.prefix, pos: r.pos, yaw: r.yaw, strip: { refs: r.visual.refs, inertia: r.visual.inertia } });
    }
    progress(`compiling ${list.length} robots…`);
    await nextFrame();
    const model = compileWorld(mujoco, specs, app.physics);
    let offset = 1;
    for (const r of list) {
      r.bodyOffset = offset;
      offset += r.visual.inertia.nbody - 1;
    }
    const { total } = stateLayout(list);
    const mismatch = Object.keys(total).find((k) => total[k] !== model[k]);
    if (offset !== model.nbody || mismatch) {
      const msg = mismatch ? `world has ${mismatch} ${model[mismatch]}, expected ${total[mismatch]}`
                           : `world has ${model.nbody} bodies, expected ${offset}`;
      model.delete();
      throw new Error(msg);
    }

    // Controlled robots: find their joints and actuators in the new model
    // (GR00T also sets its actuator gains to the policy's PD gains).
    for (const r of list) {
      if (!r.use) continue;  // environments and objects have no controllers
      if (r.use.policy && r.policy && !bindController(r, 'policy', mujoco, model)) r.use.policy = '';
      if (r.base) bindController(r, 'base', mujoco, model);
      r.use.base ??= !!r.base?.config?.kind;
    }
    // IK chains between the robot's bodies, around the joints the
    // controller drives.
    for (const r of list) {
      if (r.ik) bindController(r, 'ik', mujoco, model);
    }

    const old = snapshot(app);
    const layout = stateLayout(list);
    for (const r of fresh) r.entities = app.addEntities(r.visual.parts);
    try {
      app.setWorld(mujoco, model, list, init);
    } catch (err) {
      for (const r of fresh) app.removeEntities(r.entities);
      throw err;
    }
    function init(data) {
      if (old) copySharedState(old, data, list);
      for (const r of fresh) startPose(data, r);
    }
    // A keyframe made for the robot's own scene can put it under our floor
    // (TIAGo's: z = -0.985, for a floor at -0.99): if its collision geoms
    // reach more than 10 cm below the spawn height, its free joints go up
    // until they clear it (bounding spheres, so it may drop a little).
    function liftAboveFloor(data, fresh, q0, nq) {
      mujoco.mj_kinematics(model, data);
      const first = fresh.bodyOffset, end = first + fresh.visual.inertia.nbody - 1;
      let low = Infinity;
      for (let g = 0; g < model.ngeom; g++) {
        const b = model.geom_bodyid[g];
        if (b < first || b >= end || !(model.geom_contype[g] || model.geom_conaffinity[g])) continue;
        low = Math.min(low, data.geom_xpos[3 * g + 2] - model.geom_rbound[g]);
      }
      const lift = fresh.pos[2] - low;
      if (!(lift > 0.1)) return;
      for (let j = 0; j < model.njnt; j++) {
        const adr = model.jnt_qposadr[j];
        if (adr >= q0 && adr < q0 + nq && model.jnt_type[j] === 0) data.qpos[adr + 2] += lift;
      }
    }
    function startPose(data, fresh) {
      const { qpos: q0, ctrl: u0 } = layout.get(fresh), { nq, nu } = fresh.visual.dims;
      const key = fresh.visual.key;
      if (key?.qpos?.length === nq) {
        data.qpos.set(key.qpos, q0);
        // The keyframe is in the robot's own world frame: move free joints
        // to the spawn point and heading (qpos0 already includes them).
        const c = Math.cos(fresh.yaw), s = Math.sin(fresh.yaw);
        const hw = Math.cos(fresh.yaw / 2), hz = Math.sin(fresh.yaw / 2);
        const q = data.qpos;
        for (let j = 0; j < model.njnt; j++) {
          const adr = model.jnt_qposadr[j];
          if (adr < q0 || adr >= q0 + nq || model.jnt_type[j] !== 0) continue;
          const [x, y] = [q[adr], q[adr + 1]];
          q[adr] = fresh.pos[0] + c * x - s * y;
          q[adr + 1] = fresh.pos[1] + s * x + c * y;
          q[adr + 2] += fresh.pos[2];
          const [w, qx, qy, qz] = q.subarray(adr + 3, adr + 7);  // (hw, 0, 0, hz) * quat
          q.set([hw * w - hz * qz, hw * qx - hz * qy, hw * qy + hz * qx, hw * qz + hz * w], adr + 3);
        }
        liftAboveFloor(data, fresh, q0, nq);
      }
      if (key?.ctrl?.length === nu) data.ctrl.set(key.ctrl, u0);
      if (fresh.use?.policy) fresh.policy?.pose(data);
    }
    // Reset returns a robot to its spawn pose.
    for (const r of fresh) {
      const at = layout.get(r), { nq, nu } = r.visual.dims;
      r.initial = {
        qpos: app.data.qpos.slice(at.qpos, at.qpos + nq),
        ctrl: app.data.ctrl.slice(at.ctrl, at.ctrl + nu),
      };
    }
  }

  // A new robot (not in the world yet) for asset file `rel`.
  async function makeRobot(rel, label, pos, yaw, progress, prefix = `r${nextId++}/`) {
    const visual = await withPhysics(await robotVisual(rel, label, progress), progress);
    const robot = { label, visual, prefix, pos, yaw };
    // Every robot gets the controllers of the Teleop panel, each switched
    // on or off (robot.use): a trained policy (policy.js), a mobile base and
    // IK chains, each saved, built in or found in the model.
    const key = rel.match(/(?:^|\/)mjcf_assets\/robots\/(.+)$/)?.[1];
    if (!key || app.net) return robot;  // network mode: no controllers (yet)
    Object.assign(robot, { key, use: loadUse(key), errors: {} });
    robot.base = new BaseController(key, robot.prefix);
    robot.ik = new IkController(key, robot.prefix);
    await makePolicy(robot, progress);
    return robot;
  }

  async function spawn({ rel, name: label }, pos, yaw) {
    if (app.net) {
      app.net.send('spawn', { rel, label, pos, yaw });
      status.textContent = `spawning ${label} on the server…`;
      return;
    }
    busy = true;
    setBusy(`spawning ${label}…`);
    const t0 = performance.now();
    try {
      const progress = (msg) => { status.textContent = msg; setBusy(msg); };
      const robot = await makeRobot(rel, label, pos, yaw, progress);
      await rebuild([...robots, robot], [robot], progress);
      robots.push(robot);
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      status.textContent = `spawned ${label} (${secs}s) · ${hint()}`;
    } catch (err) {
      status.textContent = `spawn failed: ${err.message ?? err}`;
      console.error(err);
    } finally {
      busy = false;
      setBusy(null);
    }
  }

  // Clear / remove one: recompile with the robots of `list` (a subset
  // of `robots`, in order); the state of the rest is kept.
  async function respawn(list) {
    if (busy) return;
    if (app.net) {
      app.net.send('remove', { ids: robots.filter((r) => !list.includes(r)).map((r) => r.netId) });
      return;
    }
    busy = true;
    setBusy('rebuilding…');
    try {
      const removed = robots.filter((r) => !list.includes(r));
      await rebuild(list, [], (msg) => { status.textContent = msg; setBusy(msg); });
      for (const r of removed) app.removeEntities(r.entities);
      robots.splice(0, robots.length, ...list);
      dropUnused();
      status.textContent = hint();
    } catch (err) {
      status.textContent = `failed: ${err.message ?? err}`;
      console.error(err);
    } finally {
      busy = false;
      setBusy(null);
    }
  }

  // New World panel settings: global options go straight into the live
  // model; changed default geom properties recompile every model's masses
  // and rebuild the world (state kept). Resolves to false if busy.
  app.setPhysics = async (next) => {
    if (app.net) {  // the server's settings: they come back to everyone (onMeta)
      if (geomSignature(next) !== geomSignature(app.physics)) {
        status.textContent = 'default geom properties are not available in server mode yet';
        return false;
      }
      app.net.send('physics', { physics: next });
      return true;
    }
    const rebuildNeeded = needsRebuild(app.physics, next);
    if (rebuildNeeded && busy) return false;
    const speedChanged = next.speed !== app.physics.speed;
    app.physics = next;
    if (speedChanged) app.resync();
    app.world?.render();
    app.teleop?.render();  // a policy's physics note
    if (app.model) applyOptions(app.model, next);
    if (!rebuildNeeded || !app.model) return true;
    busy = true;
    const progress = (msg) => { status.textContent = msg; setBusy(msg); };
    progress('applying physics settings…');
    try {
      for (const r of robots) r.visual = await withPhysics(r.visual, progress);
      await rebuild([...robots], [], progress);
      status.textContent = hint();
    } catch (err) {
      status.textContent = `failed: ${err.message ?? err}`;
      console.error(err);
    } finally {
      busy = false;
      setBusy(null);
    }
    return true;
  };

  // A fixed-base robot to a new attach frame: the world is rebuilt (its
  // state kept). Resolves to false if busy.
  async function moveFixed(robot, pos, yaw) {
    if (busy) return false;
    if (app.net) { app.net.send('move', { id: robot.netId, pos, yaw }); return true; }
    busy = true;
    const old = [robot.pos, robot.yaw];
    [robot.pos, robot.yaw] = [[...pos], yaw];
    const progress = (msg) => { status.textContent = msg; setBusy(msg); };
    progress(`moving ${robot.label}…`);
    try {
      await rebuild([...robots], [], progress);
      // Its mocap bodies are state (kept by the rebuild): carry them along.
      const { model, data, mujoco } = app, [p0, y0] = [old[0], old[1] ?? 0], dy = yaw - y0;
      const c0 = Math.cos(-y0), s0 = Math.sin(-y0), c1 = Math.cos(yaw), s1 = Math.sin(yaw);
      for (let b = robot.bodyOffset; b < robot.bodyOffset + robot.visual.inertia.nbody - 1; b++) {
        const k = model.body_mocapid[b];
        if (k < 0) continue;
        const mp = data.mocap_pos.subarray(3 * k, 3 * k + 3), mq = data.mocap_quat.subarray(4 * k, 4 * k + 4);
        const lx = c0 * (mp[0] - p0[0]) - s0 * (mp[1] - p0[1]), ly = s0 * (mp[0] - p0[0]) + c0 * (mp[1] - p0[1]);
        mp.set([pos[0] + c1 * lx - s1 * ly, pos[1] + s1 * lx + c1 * ly, mp[2] - p0[2] + pos[2]]);
        const hw = Math.cos(dy / 2), hz = Math.sin(dy / 2), [w, x, y, z] = mq;  // (hw, 0, 0, hz) * quat
        mq.set([hw * w - hz * z, hw * x - hz * y, hw * y + hz * x, hw * z + hz * w]);
      }
      mujoco.mj_forward(model, data);
      app.syncTransforms();
      status.textContent = hint();
    } catch (err) {
      [robot.pos, robot.yaw] = old;
      status.textContent = `failed: ${err.message ?? err}`;
      console.error(err);
    } finally {
      busy = false;
      setBusy(null);
    }
    return true;
  }

  // The saved stage: what is spawned, where and heading where (no joint
  // state), in this browser's localStorage. Loaded on the next page load.
  const STAGE_KEY = 'mujoco-filament/stage';
  const savedStage = () => {
    try { return JSON.parse(localStorage.getItem(STAGE_KEY)); } catch { return null; }
  };
  function saveStage() {
    const stage = {
      saved: new Date().toISOString(),
      items: robots.map((r) => ({ rel: r.visual.rel, label: r.label, pos: r.pos.map((v) => +v.toFixed(4)), yaw: +r.yaw.toFixed(4) })),
    };
    try {
      localStorage.setItem(STAGE_KEY, JSON.stringify(stage));
      status.textContent = `stage saved (${stage.items.length} items) · ${hint()}`;
    } catch (err) {
      status.textContent = `could not save the stage: ${err.message ?? err}`;
    }
    app.panel.showSaved(stage);
  }
  function forgetStage() {
    try { localStorage.removeItem(STAGE_KEY); } catch {}
    app.panel.showSaved(null);
    status.textContent = `saved stage forgotten · ${hint()}`;
  }
  // Spawns every item of a saved stage, in one compile. Items whose files
  // fail to load are skipped (and reported).
  async function loadStage(stage) {
    const items = stage?.items ?? [];
    if (!items.length) return;
    const progress = (msg) => { status.textContent = msg; setBusy(msg); };
    const added = [], failed = [];
    for (const [i, it] of items.entries()) {
      progress(`loading stage ${i + 1}/${items.length}: ${it.label}…`);
      try {
        added.push(await makeRobot(it.rel, it.label, it.pos, it.yaw, progress));
      } catch (err) {
        console.error(`stage item ${it.rel}:`, err);
        failed.push(it.label);
      }
    }
    await rebuild([...robots, ...added], added, progress);
    robots.push(...added);
    status.textContent = (failed.length ? `stage loaded, skipped ${failed.join(', ')} · ` : 'stage loaded · ') + hint();
  }

  // Network mode: a robot's controllers as the server runs them (the
  // Teleop panel shows them like local ones, through remote.js).
  function applyRobot(r, it) {
    if (!it.key) return;
    Object.assign(r, { key: it.key, use: it.use, errors: it.errors ?? {} });
    app.teleop.fromServer(r, it.goal ?? null);
    if (r.ik) r.ik.update(it.ik);
    else r.ik = new RemoteIk(app, r, it.ik);
    if (r.base) r.base.update(it.base);
    else r.base = new RemoteBase(app, r, it.base);
    if (!it.policy) r.policy = null;
    else if (r.policy?.policyId === it.use.policy) r.policy.update(it.policy);
    else r.policy = new RemotePolicy(app, r, it.policy, it.use.policy);
  }

  // Network mode: settings and spawn poses of the server's world.
  function applyMeta(msg) {
    app.physics = { ...msg.physics, geom: app.physics.geom };
    app.paused = msg.paused;
    pauseButton.textContent = app.paused ? 'Resume' : 'Pause';
    for (const it of msg.items ?? []) {
      const r = robots.find((x) => x.netId === it.id);
      if (r) [r.pos, r.yaw] = [it.pos, it.yaw];
    }
    app.world?.render();
  }

  // Network mode: render data per model file (net.js loadRender), shared by
  // its copies like robotVisual's.
  const netVisuals = new Map();
  function netVisual(rel, label, progress) {
    if (!netVisuals.has(rel)) {
      const pending = (async () => {
        progress(`loading ${label}…`);
        const { model, images } = await loadRender(rel);
        progress(`preparing ${label} for rendering…`);
        await nextFrame();
        return {
          rel, parts: app.buildVisual(NET_MUJOCO, model, images, models),
          kin: model, dims: model.dims, inertia: { nbody: model.nbody },
        };
      })();
      pending.then((v) => { pending.visual = v; }, () => netVisuals.delete(rel));
      netVisuals.set(rel, pending);
    }
    return netVisuals.get(rel);
  }

  // Network mode: lays out the server's world (its robots, in its order, with
  // its prefixes) from their render data; the state comes from the server.
  async function applyScene(msg) {
    applyMeta(msg);
    const progress = (m) => { status.textContent = m; setBusy(m); };
    try {
      const list = [], fresh = [];
      for (const it of msg.items) {
        let r = robots.find((x) => x.netId === it.id && x.visual.rel === it.rel);  // (a restarted server numbers anew)
        if (!r) {
          const visual = await netVisual(it.rel, it.label, progress);
          r = { label: it.label, visual, prefix: `r${it.id}/`, pos: it.pos, yaw: it.yaw, netId: it.id };
          fresh.push(r);
        }
        list.push(r);
        r.pendingItem = it;  // controllers: once the world below is in place
      }
      const model = worldModel(msg.world);
      let offset = 1;
      for (const r of list) {
        r.bodyOffset = offset;
        offset += r.visual.inertia.nbody - 1;
      }
      const layout = stateLayout(list);
      const mismatch = Object.keys(layout.total).find((k) => layout.total[k] !== model[k]);
      if (offset !== model.nbody || mismatch) {
        throw new Error(mismatch ? `server world has ${mismatch} ${model[mismatch]}, expected ${layout.total[mismatch]}`
                                 : `server world has ${model.nbody} bodies, expected ${offset}`);
      }
      for (const r of list) r.netAt = { qpos: layout.get(r).qpos, mocap: layout.get(r).mocap_pos / 3 };
      for (const r of fresh) r.entities = app.addEntities(r.visual.parts);
      for (const r of robots) if (!list.includes(r)) app.removeEntities(r.entities);
      app.setNetWorld(model, worldData(model), list);
      robots.splice(0, robots.length, ...list);
      dropUnused();
      app.netVersion = msg.version;
      for (const r of list) { applyRobot(r, r.pendingItem); delete r.pendingItem; }
      app.teleop.build();
      status.textContent = `${hint()} · shared world (server)`;
    } catch (err) {
      status.textContent = `could not show the server's world: ${err.message ?? err}`;
      console.error(err);
    }
  }

  // For devtools and scripts/render_previews.mjs.
  window.spawnAsset = (rel, pos = [0, 0, 0], yaw = 0) => spawn(assets.find((a) => a.rel === rel) ?? { rel, name: rel }, pos, yaw);
  window.clearAssets = () => respawn([]);

  // Goal markers from a script (the devtools console, page automation), as
  // the Teleop panel sets them, in local and network mode alike (the server
  // has the same as an HTTP API: /api/goal). `which`: 'all' (every robot
  // whose walking policy or base is on), a name ('r3'), a label, or a list.
  // A robot driven with the keys in the Teleop panel ignores its goal.
  //   goals.move('all', { forward: 1 }); await goals.wait(); goals.move('all', { forward: -1 });
  const pickRobots = (which = 'all') => {
    if (which === 'all') return robots.filter((r) => locomotion(r).length);
    return [which].flat().map((w) => {
      const r = robots.find((x) => w === x.prefix.slice(0, -1) || w === x.label);
      if (!r) throw new Error(`no robot ${w}`);
      return r;
    });
  };
  const goalInfo = (r) => ({
    name: r.prefix.slice(0, -1), label: r.label, drives: locomotion(r).length > 0,
    pose: bodyPose(app.data, r.bodyOffset), goal: r.goal ?? null, arrived: !!r.arrived,
  });
  window.goals = {
    list: () => robots.map(goalInfo),
    // Relative to each robot's pose: forward / left (m, its own frame), turn (rad).
    move(which, { forward = 0, left = 0, turn = 0 } = {}) {
      return pickRobots(which).map((r) => {
        const p = bodyPose(app.data, r.bodyOffset), c = Math.cos(p.yaw), s = Math.sin(p.yaw);
        app.teleop.setGoal(r, { x: p.x + c * forward - s * left, y: p.y + s * forward + c * left, yaw: p.yaw + turn });
        return goalInfo(r);
      });
    },
    // {x, y, yaw} in the world frame.
    set: (which, goal) => pickRobots(which).map((r) => (app.teleop.setGoal(r, { ...goal }), goalInfo(r))),
    clear: (which) => pickRobots(which).map((r) => (app.teleop.setGoal(r, null), goalInfo(r))),
    // Resolves once none of them has a goal left (reached, or cleared).
    wait(which, timeout = 60) {
      const list = pickRobots(which), t0 = performance.now();
      return new Promise((resolve, reject) => {
        const check = () => {
          if (list.every((r) => !r.goal)) resolve(list.map(goalInfo));
          else if (performance.now() - t0 > timeout * 1000) reject(new Error(`goals not reached in ${timeout} s`));
          else setTimeout(check, 100);
        };
        check();
      });
    },
  };

  // Assets are spawned by dragging them from the asset panel onto the floor.
  new AssetPanel(app, document.getElementById('assets'), assets, {
    onDrop: (asset, pos, yaw) => spawn(asset, pos, yaw),
    recent,
    canDrop: () => {
      if (busy) status.textContent = 'busy, try again in a moment';
      return !busy;
    },
  });
  // Panel toggles (top right), apart from the simulation controls.
  // On a phone they start hidden and show one at a time (index.html's
  // narrow layout puts each across the width).
  const narrow = matchMedia('(max-width: 700px)');
  const toggles = [...document.querySelectorAll('#views button')].map((b) => [b, document.getElementById(b.dataset.panel)]);
  const showPanel = (panel, on) => {
    panel.hidden = !on;
    toggles.find(([, p]) => p === panel)[0].setAttribute('aria-pressed', String(on));
    if (on && panel === app.panel.el) app.panel.lastUpdate = 0;
  };
  for (const [b, panel] of toggles) {
    if (narrow.matches) panel.hidden = true;
    b.setAttribute('aria-pressed', String(!panel.hidden));
    b.onclick = () => {
      const on = panel.hidden;
      if (on && narrow.matches) for (const [, p] of toggles) if (p !== panel && !p.hidden) showPanel(p, false);
      showPanel(panel, on);
    };
  }
  if (TOUCH) {  // the hints name mouse gestures
    document.querySelector('#assets .hint').textContent = 'Drag an item sideways out of the list onto the floor to spawn it.';
    document.querySelector('#stage .hint').textContent = 'Double-tap a body to select it, press and hold on it, then drag, to pull it. Joints can be set while paused.';
  }

  // Double-click selects a body, Shift+drag pulls the selection (inspect.js).
  // On a touch screen: double tap, and press and hold on a body, then drag.
  const canvas = document.getElementById('canvas');
  let dragging = false;
  // Dragging an IK marker (Teleop panel) moves it instead of the camera.
  app.orbit.intercept = (e) => e.button === 0 && app.teleop.grab(app.pointerRay(e.clientX, e.clientY), [e.clientX, e.clientY]);
  app.orbit.busy = () => dragging || !!app.teleop.dragging;
  const fingers = new Map();  // pointerId -> [x, y] on the canvas (touch only)
  let hold = null;            // {timer, id, x, y}: a finger that may become a long press
  let goalTurn = null;        // the angle between two fingers while one drags the goal marker
  let lastTap = null;         // {t, x, y} of the last short tap
  const angle = () => {
    const [a, b] = [...fingers.values()];
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  };
  const beginPull = (x, y, id) => {
    if (app.model && app.perturb.begin(app.pointerRay(x, y))) {
      dragging = true;
      app.orbit.stop();
      canvas.setPointerCapture(id);
      return true;
    }
    return false;
  };
  canvas.addEventListener('pointerdown', (e) => {
    if (app.teleop.dragging) canvas.setPointerCapture(e.pointerId);
    if (e.pointerType === 'touch') {
      fingers.set(e.pointerId, [e.clientX, e.clientY]);
      clearTimeout(hold?.timer);
      hold = null;
      if (fingers.size === 2 && app.teleop.dragging?.goal) goalTurn = angle();
      else if (fingers.size === 1 && !app.teleop.dragging) {
        const { pointerId: id, clientX: x, clientY: y } = e;
        // Held still: select the body under the finger and pull it.
        hold = { id, x, y, t: performance.now(), timer: setTimeout(async () => {
          hold = null;
          const hit = app.model && await app.pickBody(x, y);  // a promise in network mode
          if (!hit || !fingers.has(id) || fingers.size !== 1) return;
          app.perturb.select(hit);
          beginPull(x, y, id);
        }, 450) };
      }
      return;
    }
    if (e.shiftKey) beginPull(e.clientX, e.clientY, e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (fingers.has(e.pointerId)) {
      fingers.set(e.pointerId, [e.clientX, e.clientY]);
      if (hold && Math.hypot(e.clientX - hold.x, e.clientY - hold.y) > 10) { clearTimeout(hold.timer); hold = null; }
      if (goalTurn !== null && fingers.size === 2 && app.teleop.dragging?.goal) {
        const a = angle(), d = Math.atan2(Math.sin(a - goalTurn), Math.cos(a - goalTurn));
        goalTurn = a;
        app.teleop.turnGoal(-d);  // screen y points down: a clockwise twist turns it clockwise
        return;
      }
    }
    if (dragging) app.perturb.move(app.pointerRay(e.clientX, e.clientY));
    if (app.teleop.dragging) app.teleop.dragTo(app.pointerRay(e.clientX, e.clientY), [e.clientX, e.clientY]);
    else if (!e.buttons && e.pointerType !== 'touch') canvas.style.cursor = app.teleop.hoverAt(app.pointerRay(e.clientX, e.clientY)) ? 'grab' : '';
  });
  // Double-clicking a controllable robot also makes it the driven one;
  // double-clicking anything else releases control.
  const selectAt = async (x, y) => {
    const hit = await app.pickBody(x, y);  // a promise in network mode
    app.perturb.select(hit);
    if (hit?.body > 0) app.orbit.focus(hit.point);  // the camera turns about the selection
    const robot = hit && app.robots.find((r) => hit.body >= r.bodyOffset && hit.body < r.bodyOffset + r.visual.inertia.nbody - 1);
    app.teleop.drive(robot ?? null);
  };
  let lastTouchEnd = -Infinity;
  const up = (e) => {
    if (e.pointerType === 'touch') lastTouchEnd = performance.now();
    const tap = hold?.id === e.pointerId && performance.now() - hold.t < 300 && fingers.size === 1;
    clearTimeout(hold?.timer);
    hold = null;
    fingers.delete(e.pointerId);
    if (fingers.size < 2) goalTurn = null;
    if (fingers.size) return;  // the drag goes on with the other finger
    app.teleop.release();
    if (dragging) { dragging = false; app.perturb.end(); }
    if (tap && e.type === 'pointerup') {
      const now = performance.now();
      if (lastTap && now - lastTap.t < 350 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 30) {
        lastTap = null;
        selectAt(e.clientX, e.clientY);
      } else {
        lastTap = { t: now, x: e.clientX, y: e.clientY };
      }
    }
  };
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('dblclick', (e) => {
    // Some browsers make a dblclick of a double tap too: handled above.
    if (performance.now() - lastTouchEnd < 700) return;
    selectAt(e.clientX, e.clientY);
  });
  const pauseButton = document.getElementById('pause');
  pauseButton.onclick = (e) => {
    if (app.net) { app.net.send('pause', { paused: !app.paused }); return; }
    app.paused = !app.paused;
    e.target.textContent = app.paused ? 'Resume' : 'Pause';
    if (!app.paused && app.data) app.resync();
  };
  document.getElementById('reset').onclick = () => app.model && app.reset();
  document.getElementById('clear').onclick = () => robots.length && respawn([]);
  document.getElementById('restart').onclick = () => {
    if (app.net && confirm('Restart the shared world for everyone? The server reads the model files again and starts the simulation anew.')) app.net.send('restart');
  };

  // Enter VR: shown only where an immersive-vr session is supported (WebXR
  // needs a secure context: https or localhost).
  const vr = document.getElementById('vr');
  if (await navigator.xr?.isSessionSupported('immersive-vr').catch(() => false)) {
    vr.hidden = false;
    vr.onclick = async () => {
      if (app.xr) { app.xr.session.end(); return; }
      try {
        const session = await app.enterVR();
        vr.textContent = 'Exit VR';
        session.addEventListener('end', () => { vr.textContent = 'Enter VR'; });
      } catch (err) {
        status.textContent = `VR failed: ${err.message ?? err}`;
        console.error(err);
      }
    };
  }

  // Network mode (a server/server.py, endpoint.js): the world is the server's.
  if (server) {
    // Lost the server: a banner counting down to the next try (net.js keeps
    // trying; on reconnecting the server sends its world again).
    const banner = document.getElementById('offline'), bannerMsg = banner.querySelector('.msg');
    let countdown = null;
    banner.querySelector('button').onclick = () => app.net.reconnect();
    const offline = (retryAt) => {
      clearInterval(countdown);
      banner.hidden = retryAt == null;
      if (retryAt == null) return;
      const show = () => {
        const s = Math.ceil((retryAt - performance.now()) / 1000);
        bannerMsg.textContent = s > 0 ? `Disconnected from the server · retrying in ${s} s` : 'Disconnected from the server · reconnecting…';
      };
      show();
      countdown = setInterval(show, 250);
    };
    const scenes = [];  // waiting scene messages; only the newest one is built
    let building = false;
    const buildScenes = async () => {
      if (building) return;
      building = busy = true;
      try {
        while (scenes.length) await applyScene(scenes.splice(0).at(-1));
      } finally {
        building = busy = false;
        setBusy(null);
      }
    };
    document.getElementById('restart').hidden = false;
    app.net = new NetClient({
      scene: (msg) => { scenes.push(msg); buildScenes(); },
      meta: (msg) => applyMeta(msg),
      robot: (msg) => {
        const r = robots.find((x) => x.netId === msg.item.id);
        if (!r || building) return;  // a scene being built brings its own
        applyRobot(r, msg.item);
        app.teleop.render();
      },
      status: (msg) => {
        for (const [id, s] of Object.entries(msg.robots)) {
          const r = robots.find((x) => x.netId === +id);
          if (!r) continue;
          if (s.ik) r.ik?.status(s.ik);
          if (r.base) r.base.measured = s.base?.measured ?? null;
          r.goalResult = s.goal ?? null;
        }
        for (const r of robots) if (!msg.robots[r.netId]) r.goalResult = null;
      },
      // A robot's goal, set by any client (the server follows it).
      goal: (msg) => {
        const r = robots.find((x) => x.netId === msg.id);
        if (r) app.teleop.fromServer(r, msg.goal, msg.arrived);
      },
      // Reset by any client: goals set here would drive the robots off again.
      reset: (msg) => app.teleop.clearGoals(msg.id == null ? robots : robots.filter((r) => r.netId === msg.id)),
      error: (msg) => { status.textContent = `server: ${msg.op} failed: ${msg.msg}`; },
      closed: (retryAt) => offline(retryAt),
      connected: () => offline(null),
    });
    await app.net.opened;
    console.log(`network mode: ${server.info.server}, MuJoCo ${server.info.mujoco}, at ${server.base}`);
    app.world.render();
    app.panel.showSaved(savedStage());
  } else {
    await rebuild([]);
    status.textContent = problem ? `${problem}: running locally (Server panel)` : hint();
    const stage = savedStage();
    app.panel.showSaved(stage);
    if (stage?.items?.length) {
      busy = true;
      try {
        await loadStage(stage);
      } catch (err) {
        status.textContent = `could not load the saved stage: ${err.message ?? err}`;
        console.error(err);
      } finally {
        busy = false;
      }
    }
    setBusy(null);
  }
  app.frame();
} catch (err) {
  status.textContent = `error: ${err.message ?? err}`;
  setBusy(`error: ${err.message ?? err}`, true);
  throw err;
}
