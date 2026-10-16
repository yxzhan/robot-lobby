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

import loadMujoco from './vendor/mujoco/mujoco.js';
import { loadModelFiles, readFirst, urlSource } from './vfs.js';
import { bodyInertia, compileWorld, firstKey, referencedNames } from './world.js';
import { CONTROL_DT, G1_MODEL, GrootController, loadGrootPolicy } from './groot.js';
import { Perturb, StagePanel } from './inspect.js';
import { AssetPanel, loadAssets } from './assets.js';
import { AxisGizmo } from './gizmo.js';
import { ControlPanel } from './control.js';
import Stats from './vendor/stats/Stats.js';

// Filament.init() replaces the global with the loaded module, so re-read it after init.
let Filament = window.Filament;
const MAT_LIT = 'materials/lit.filamat';
const MAT_CHECKER = 'materials/checker.filamat';
const MAT_TEXTURED = 'materials/lit_textured.filamat';


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
// top row first like MuJoCo's own PNG loader.
async function decodeImage(bytes) {
  const bmp = await createImageBitmap(new Blob([bytes]), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  const { width: w, height: h } = bmp;
  const ctx = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
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

// Simple orbit camera around a Z-up target.
class OrbitCamera {
  constructor(canvas, center, dist) {
    this.center = center; this.dist = dist;
    this.azim = -0.6; this.elev = 0.35;
    let last = null, button = 0;
    canvas.addEventListener('pointerdown', (e) => {
      if (e.shiftKey) { last = null; return; }  // Shift+drag perturbs (inspect.js)
      last = [e.clientX, e.clientY]; button = e.button; canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', () => { last = null; });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointermove', (e) => {
      if (!last) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      if (button === 0) {
        this.azim -= dx * 0.005;
        this.elev = Math.max(-1.5, Math.min(1.5, this.elev + dy * 0.005));
      } else {  // right/middle drag pans in the view plane
        const k = this.dist * 0.0015;
        const [ca, sa] = [Math.cos(this.azim), Math.sin(this.azim)];
        this.center[0] += (sa * dx - ca * Math.sin(this.elev) * dy) * k;
        this.center[1] += (-ca * dx - sa * Math.sin(this.elev) * dy) * k;
        this.center[2] += Math.cos(this.elev) * dy * k;
      }
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.dist *= Math.exp(e.deltaY * 0.001);
    }, { passive: false });
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
    this.robots = [];  // spawned, in attach order: {visual, pos, bodyOffset, entities, groot}
    this.controllers = [];  // the robots' policies (GrootController), ticked every CONTROL_DT
    this.nextControl = 0;   // sim time of the next policy tick
    this.stepping = null;   // the pending advance(), if any
    this.perturb = new Perturb(this);  // double-click selection, Shift+drag force
    this.panel = null;                 // StagePanel, set by the page
    this.control = null;               // ControlPanel, set by the page
    this.stats = null;                 // stats.js meter, set by the page

    // xrCompatible: the same context renders the WebXR session (enterVR).
    const engine = this.engine = Filament.Engine.create(canvas, { xrCompatible: true });
    this.scene = engine.createScene();
    this.swapChain = engine.createSwapChain();
    this.renderer = engine.createRenderer();
    this.camera = engine.createCamera(Filament.EntityManager.get().create());
    this.view = engine.createView();
    this.view.setCamera(this.camera);
    this.view.setScene(this.scene);
    this.view.setAmbientOcclusionOptions({ enabled: true, radius: 0.2 });
    this.view.setAntiAliasing(Filament.View$AntiAliasing.FXAA);
    this.renderer.setClearOptions({ clearColor: [0.1, 0.11, 0.13, 1], clear: true });

    this.matLit = engine.createMaterial(MAT_LIT);
    this.matChecker = engine.createMaterial(MAT_CHECKER);
    this.matTextured = engine.createMaterial(MAT_TEXTURED);
    const F = Filament;
    this.sampler = new F.TextureSampler(F.MinFilter.LINEAR_MIPMAP_LINEAR, F.MagFilter.LINEAR, F.WrapMode.REPEAT);
    this.sampler.setAnisotropy(8);
    // Bound to textured materials until their image has been decoded.
    this.whiteTexture = uploadTexture(engine, 1, 1, new Uint8Array([255, 255, 255, 255]));

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

    // The floor of world.js's BASE_XML (a size-0, i.e. infinite, plane), drawn
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
    const spawnMi = this.matLit.createInstance();
    spawnMi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [0.2, 0.8, 1, 1]);
    spawnMi.setFloatParameter('roughness', 0.6);
    spawnMi.setFloatParameter('metallic', 0.0);
    this.spawnMarker = makeEntity(engine, makeGeometry(engine, spawnMarkerShape()), spawnMi, false);
    this.scene.addEntity(this.spawnMarker);
    // From the floor up to a raised spawn marker.
    this.spawnPole = makeEntity(engine, makeGeometry(engine, cylinder(1, 0.5)), spawnMi, false);
    this.scene.addEntity(this.spawnPole);

    this.orbit = new OrbitCamera(canvas, [0, 0, 0.4], 4);
    this.setSpawnMarker(null);
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    if (this.xr) return;  // the session sizes the canvas
    const dpr = window.devicePixelRatio || 1;
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
  buildVisual(mujoco, model, images, read) {
    const textures = new Map();  // MuJoCo texture id -> {tex, waiting: [MaterialInstance]}
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

      let mi, uv = null;
      if (textured) {
        mi = this.matTextured.createInstance();
        mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [rgba[0], rgba[1], rgba[2], 1]);
        this.bindTexture(textures, mujoco, model, images, read, texid, mi);
        uv = geomUVs(model, g, shape, matid);
      } else {
        mi = this.matLit.createInstance();
        mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [rgba[0], rgba[1], rgba[2], 1]);
      }
      mi.setFloatParameter('roughness', 0.45);
      mi.setFloatParameter('metallic', 0.0);
      parts.push({
        ...makeGeometry(this.engine, shape, uv), mi,
        body: model.geom_bodyid[g],
        pos: Array.from(model.geom_pos.subarray(3 * g, 3 * g + 3)),
        rot: quatToMat(model.geom_quat.subarray(4 * g, 4 * g + 4)),
      });
    }
    return parts;
  }

  // Binds MuJoCo texture `texid` to `mi`, creating it on first use. Image
  // textures are decoded asynchronously; until then `mi` shows plain white.
  bindTexture(textures, mujoco, model, images, read, texid, mi) {
    let entry = textures.get(texid);
    if (!entry) {
      entry = { tex: null, waiting: [] };
      textures.set(texid, entry);
      const name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE.value, texid);
      const cands = images.get(name);
      if (cands) this.loadImage(entry, read, cands, name);
      else entry.tex = mjTexture(this.engine, model, texid);
    }
    if (entry.tex) {
      mi.setTextureParameter('baseColorMap', entry.tex, this.sampler);
    } else {
      mi.setTextureParameter('baseColorMap', this.whiteTexture, this.sampler);
      entry.waiting.push(mi);
    }
  }

  async loadImage(entry, read, cands, name) {
    try {
      const { w, h, rgba } = await decodeImage(await readFirst(read, cands));
      entry.tex = uploadTexture(this.engine, w, h, rgba);
      for (const mi of entry.waiting) mi.setTextureParameter('baseColorMap', entry.tex, this.sampler);
      entry.waiting = [];
    } catch (err) {
      console.warn(`texture ${name}:`, err);
    }
  }

  addEntities(parts) {
    return parts.map((p) => {
      const e = makeEntity(this.engine, p, p.mi);
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
    this.controllers = robots.flatMap((r) => (r.groot ? [r.groot] : []));
    this.nextControl = data.time;
    this.simStart = performance.now() - data.time * 1000;
    this.syncTransforms();
    this.panel?.build();
    this.control?.build();
  }

  // Every robot back to its spawn pose (robot.initial), time back to 0.
  reset() {
    this.mujoco.mj_resetData(this.model, this.data);
    const layout = stateLayout(this.robots);
    for (const r of this.robots) this.writeInitial(r, layout.get(r));
    this.mujoco.mj_forward(this.model, this.data);
    for (const c of this.controllers) c.reset();
    this.nextControl = 0;
    this.simStart = performance.now();
    this.syncTransforms();  // frame() only syncs while running
  }

  // One robot back to its spawn pose, at rest; the others and time go on.
  resetRobot(robot) {
    const { mujoco, model, data } = this;
    const at = stateLayout(this.robots).get(robot);
    if (!at) return;
    this.writeInitial(robot, at);
    data.qvel.fill(0, at.qvel, at.qvel + robot.visual.dims.nv);
    data.act.fill(0, at.act, at.act + robot.visual.dims.na);
    mujoco.mj_forward(model, data);
    robot.groot?.reset();
    this.syncTransforms();
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

  // Shows the drag-to-spawn marker at `pos` heading `yaw`, with a pole down
  // to the floor when it is raised, or hides it (null). Its size follows the
  // camera distance.
  setSpawnMarker(pos, yaw = 0) {
    const tcm = this.engine.getTransformManager();
    const put = (entity, m) => {
      const inst = tcm.getInstance(entity);
      tcm.setTransform(inst, m ?? [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
      inst.delete();
    };
    const k = 0.06 * this.orbit.dist, c = Math.cos(yaw) * k, s = Math.sin(yaw) * k;
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
    const r = 0.006 * this.orbit.dist;  // constant on-screen size
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
  }

  // Geom pose = body pose (from the world) * geom pose in the body (from the
  // robot's own model). The robot's body k > 0 is world body bodyOffset + k - 1;
  // its static (worldbody) geoms sit at the spawn point, turned by its yaw.
  syncTransforms() {
    const tcm = this.engine.getTransformManager();
    const xpos = this.data.xpos, xmat = this.data.xmat;
    for (const { visual, pos, yaw = 0, bodyOffset, entities } of this.robots) {
      const c = Math.cos(yaw), s = Math.sin(yaw);
      const Rz = [c, -s, 0, s, c, 0, 0, 0, 1];
      for (const [i, part] of visual.parts.entries()) {
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

  // Steps the world up to wall time (capped so a hidden tab doesn't explode),
  // ticking the policies every CONTROL_DT of sim time. Inference is async
  // (onnxruntime-web), so this awaits it between steps; a world swapped in
  // meanwhile (spawn, undo) ends the call.
  async advance() {
    const { mujoco, model, data } = this;
    const valid = () => this.data === data;
    const target = (performance.now() - this.simStart) / 1000;
    const t0 = performance.now();
    while (data.time < target && performance.now() - t0 < 20) {
      if (this.controllers.length && data.time >= this.nextControl - 1e-9) {
        await Promise.all(this.controllers.map((c) => c.step(data, valid)));
        if (!valid()) return;
        this.nextControl += CONTROL_DT;
      }
      this.perturb.apply();
      mujoco.mj_step(model, data);
    }
    if (data.time < target) this.simStart = performance.now() - data.time * 1000;
    this.syncTransforms();
  }

  // `xrFrame` is set when called from the WebXR session's animation loop.
  frame(xrFrame = null) {
    this.stats?.begin();
    if (this.model && !this.paused && !this.stepping) {
      this.stepping = this.advance()
        .catch((err) => { this.paused = true; console.error('simulation stopped', err); })
        .finally(() => { this.stepping = null; });
    }
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
      view.setAmbientOcclusionOptions({ enabled: true, radius: 0.2 });
      view.setAntiAliasing(Filament.View$AntiAliasing.FXAA);
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
function setBusy(msg) {
  busyEl.hidden = msg == null;
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
  await new Promise((resolve) => Filament.init([MAT_LIT, MAT_CHECKER, MAT_TEXTURED], resolve));
  Filament = window.Filament;
  const app = new App(document.getElementById('canvas'));
  app.panel = new StagePanel(app, document.getElementById('stage'), {
    onRemove: (robot) => respawn(robots.filter((r) => r !== robot)),
    onSave: () => saveStage(),
    onForget: () => forgetStage(),
  });
  app.control = new ControlPanel(app, document.getElementById('control'));
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
  const assets = await loadAssets();

  // Asset paths (`rel`) are under public/:
  // 'mjcf_assets/<category>/<source>/<model>/<file>.xml'.
  // Every rebuild writes the robots' files into a fresh MuJoCo FS, so keep
  // them in memory.
  const fetched = new Map();
  const publicUrl = urlSource('.');
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
  function robotVisual(rel, label, progress) {
    if (!visuals.has(rel)) {
      const pending = (async () => {
        const mujoco = await loadMujoco();  // throwaway: dropped after this
        const { path, images, docs } = await loadModelFiles(mujoco, models, '/models', rel, progress);
        progress(`preparing ${label} for rendering…`);
        await nextFrame();
        const model = mujoco.MjModel.from_xml_path(path);
        try {
          return {
            rel, parts: app.buildVisual(mujoco, model, images, models),
            inertia: bodyInertia(model), key: firstKey(docs), refs: referencedNames(docs),
            dims: { nq: model.nq, nv: model.nv, na: model.na, nu: model.nu, nmocap: model.nmocap },
          };
        } finally {
          model.delete();
        }
      })();
      pending.catch(() => visuals.delete(rel));  // retry on the next click
      visuals.set(rel, pending);
    }
    return visuals.get(rel);
  }

  // The GR00T locomotion policy (groot.js), loaded on the first G1 spawn from
  // public/groot (see scripts/vendor.mjs). Resolves to null if unavailable:
  // the G1 then spawns without a controller.
  let groot = null;
  function grootPolicy(progress) {
    groot ??= (async () => {
      progress('loading GR00T policy…');
      const ort = await import('./vendor/onnxruntime/ort.wasm.min.mjs');
      ort.env.wasm.wasmPaths = new URL('vendor/onnxruntime/', location.href).href;
      ort.env.wasm.numThreads = 1;  // tiny MLPs; threads would need cross-origin isolation
      return loadGrootPolicy(ort, (name) => `groot/${name}`);
    })().catch((err) => {
      console.warn('GR00T policy unavailable, G1 will not walk', err);
      return null;
    });
    return groot;
  }

  const robots = [];  // spawned, in attach order: {label, visual, prefix, pos, yaw, entities, bodyOffset, groot}
  let nextId = 1, busy = false;

  const hint = () => {
    const walk = robots.some((r) => r.groot) ? ' · drive a G1 from the Control panel' : '';
    return `${robots.length} spawned · drag an asset onto the floor` + walk;
  };

  // Compiles the world for `list` in a fresh MuJoCo instance and swaps it in.
  // `fresh` are the robots of `list` that are not in the current world yet:
  // each starts from its own keyframe, if any. Throws (changing nothing) if
  // compiling fails.
  async function rebuild(list, fresh = [], progress = () => {}) {
    const mujoco = await loadMujoco();
    const specs = [];
    for (const r of list) {
      const { path } = await loadModelFiles(mujoco, models, '/models', r.visual.rel);
      specs.push({ path, prefix: r.prefix, pos: r.pos, yaw: r.yaw, strip: { refs: r.visual.refs, inertia: r.visual.inertia } });
    }
    progress(`compiling ${list.length} robots…`);
    await nextFrame();
    const model = compileWorld(mujoco, specs);
    let offset = 1;
    for (const r of list) {
      r.bodyOffset = offset;
      offset += r.visual.inertia.nbody - 1;
    }
    const { total } = stateLayout(list);
    const mismatch = Object.keys(total).find((k) => total[k] !== model[k]);
    if (offset !== model.nbody || mismatch) {
      model.delete();
      throw new Error(mismatch ? `world has ${mismatch} ${model[mismatch]}, expected ${total[mismatch]}`
                               : `world has ${model.nbody} bodies, expected ${offset}`);
    }

    // Policy-driven robots: find their joints in the new model and set its
    // actuator gains to the policy's PD gains.
    for (const r of list) r.groot?.bind(mujoco, model);

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
      }
      if (key?.ctrl?.length === nu) data.ctrl.set(key.ctrl, u0);
      fresh.groot?.pose(data);
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
  async function makeRobot(rel, label, pos, yaw, progress) {
    const visual = await robotVisual(rel, label, progress);
    const robot = { label, visual, prefix: `r${nextId++}/`, pos, yaw };
    if (rel.endsWith(`/mujoco_menagerie/${G1_MODEL}`)) {
      const policy = await grootPolicy(progress);
      if (policy) robot.groot = new GrootController(policy, robot.prefix);
    }
    return robot;
  }

  async function spawn({ rel, name: label }, pos, yaw) {
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

  // Undo / clear / remove one: recompile with the robots of `list` (a subset
  // of `robots`, in order); the state of the rest is kept.
  async function respawn(list) {
    if (busy) return;
    busy = true;
    setBusy('rebuilding…');
    try {
      const removed = robots.filter((r) => !list.includes(r));
      await rebuild(list, [], (msg) => { status.textContent = msg; setBusy(msg); });
      for (const r of removed) app.removeEntities(r.entities);
      robots.splice(0, robots.length, ...list);
      status.textContent = hint();
    } catch (err) {
      status.textContent = `failed: ${err.message ?? err}`;
      console.error(err);
    } finally {
      busy = false;
      setBusy(null);
    }
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

  // For devtools and scripts/render_previews.mjs.
  window.spawnAsset = (rel, pos = [0, 0, 0], yaw = 0) => spawn(assets.find((a) => a.rel === rel) ?? { rel, name: rel }, pos, yaw);
  window.clearAssets = () => respawn([]);

  // Assets are spawned by dragging them from the asset panel onto the floor.
  new AssetPanel(app, document.getElementById('assets'), assets, {
    onDrop: (asset, pos, yaw) => spawn(asset, pos, yaw),
    canDrop: () => {
      if (busy) status.textContent = 'busy, try again in a moment';
      return !busy;
    },
  });
  // Panel toggles (top right), apart from the simulation controls.
  for (const b of document.querySelectorAll('#views button')) {
    const panel = document.getElementById(b.dataset.panel);
    b.setAttribute('aria-pressed', String(!panel.hidden));
    b.onclick = () => {
      panel.hidden = !panel.hidden;
      b.setAttribute('aria-pressed', String(!panel.hidden));
      if (panel === app.panel.el) app.panel.lastUpdate = 0;
    };
  }

  // Double-click selects a body, Shift+drag pulls the selection (inspect.js).
  const canvas = document.getElementById('canvas');
  let dragging = false;
  canvas.addEventListener('pointerdown', (e) => {
    if (!e.shiftKey) return;
    if (app.model && app.perturb.begin(app.pointerRay(e.clientX, e.clientY))) {
      dragging = true;
      canvas.setPointerCapture(e.pointerId);
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (dragging) app.perturb.move(app.pointerRay(e.clientX, e.clientY));
  });
  canvas.addEventListener('pointerup', () => {
    if (dragging) { dragging = false; app.perturb.end(); }
  });
  // Double-clicking a controllable robot also makes it the driven one;
  // double-clicking anything else releases control.
  canvas.addEventListener('dblclick', (e) => {
    const hit = app.pickBody(e.clientX, e.clientY);
    app.perturb.select(hit);
    const robot = hit && app.robots.find((r) => hit.body >= r.bodyOffset && hit.body < r.bodyOffset + r.visual.inertia.nbody - 1);
    app.control.drive(robot ?? null);
  });
  document.getElementById('pause').onclick = (e) => {
    app.paused = !app.paused;
    e.target.textContent = app.paused ? 'Resume' : 'Pause';
    if (!app.paused && app.data) app.simStart = performance.now() - app.data.time * 1000;
  };
  document.getElementById('reset').onclick = () => app.model && app.reset();
  document.getElementById('undo').onclick = () => robots.length && respawn(robots.slice(0, -1));
  document.getElementById('clear').onclick = () => robots.length && respawn([]);

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

  await rebuild([]);
  status.textContent = hint();
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
  app.frame();
} catch (err) {
  status.textContent = `error: ${err.message ?? err}`;
  setBusy(`error: ${err.message ?? err}`);
  throw err;
}
