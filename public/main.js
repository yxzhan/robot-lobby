// MuJoCo (official WASM bindings) for physics + Filament.js for rendering.
//
// Each MuJoCo geom becomes one Filament renderable whose vertices are baked
// from geom_size at load time; every frame we only copy geom_xpos/geom_xmat
// into the Filament transform. The scene stays in MuJoCo's Z-up frame.

import loadMujoco from './vendor/mujoco/mujoco.js';
import { loadModelFiles, readFirst, urlSource } from './vfs.js';
import { compileWorld, firstKey } from './world.js';

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

// `uv` (Float32Array, 2 per vertex) is only passed for textured materials.
function makeRenderable(engine, shape, matInstance, uv = null) {
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
  const entity = Filament.EntityManager.get().create();
  Filament.RenderableManager.Builder(1)
    .boundingBox({
      center: lo.map((v, k) => (v + hi[k]) / 2),
      halfExtent: lo.map((v, k) => (hi[k] - v) / 2 + 1e-4),
    })
    .material(0, matInstance)
    .geometry(0, Filament.RenderableManager$PrimitiveType.TRIANGLES, vb, ib)
    .castShadows(true)
    .receiveShadows(true)
    .build(engine, entity);
  return { entity, vb, ib, matInstance };
}

// Simple orbit camera around a Z-up target.
class OrbitCamera {
  constructor(canvas, center, dist) {
    this.center = center; this.dist = dist;
    this.azim = -0.6; this.elev = 0.35;
    let last = null, button = 0;
    canvas.addEventListener('pointerdown', (e) => { last = [e.clientX, e.clientY]; button = e.button; canvas.setPointerCapture(e.pointerId); });
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
  constructor(mujoco, canvas) {
    this.mujoco = mujoco;
    this.canvas = canvas;
    this.paused = false;
    this.items = [];  // [{geom, entity, vb, ib, matInstance}]
    this.textures = new Map();  // MuJoCo texture id -> {tex, waiting: [MaterialInstance]} (per model)
    this.generation = 0;        // bumped by clear() so late image decodes are dropped

    const engine = this.engine = Filament.Engine.create(canvas);
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
    this.scene.setIndirectLight(
      Filament.IndirectLight.Builder().irradianceSh(3, sh).intensity(25000).build(engine));

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    this.canvas.width = w; this.canvas.height = h;
    this.view.setViewport([0, 0, w, h]);
    this.camera.setProjectionFov(45, w / h, 0.01, 200, Filament.Camera$Fov.VERTICAL);
  }

  // Takes ownership of a compiled MjModel. `images` (texture name ->
  // {read, cands}) supplies the image textures the loader replaced with
  // placeholders (see vfs.js). `init(data)` may set the state before the
  // first mj_forward.
  load(model, images = new Map(), init = null) {
    const { mujoco } = this;
    this.clear();
    this.images = images;
    this.model = model;
    this.data = new mujoco.MjData(model);
    init?.(this.data);
    mujoco.mj_forward(model, this.data);

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
        this.bindTexture(texid, mi);
        mi.setFloatParameter('roughness', 0.45);
        mi.setFloatParameter('metallic', 0.0);
        uv = geomUVs(model, g, shape, matid);
      } else if (model.geom_type[g] === 0 && matid < 0) {  // bare plane: default checker floor
        mi = this.matChecker.createInstance();
        mi.setColor3Parameter('color1', Filament.RgbType.sRGB, [0.25, 0.3, 0.36]);
        mi.setColor3Parameter('color2', Filament.RgbType.sRGB, [0.45, 0.5, 0.56]);
        mi.setFloatParameter('scale', 0.5);
      } else {
        mi = this.matLit.createInstance();
        mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [rgba[0], rgba[1], rgba[2], 1]);
        mi.setFloatParameter('roughness', 0.45);
        mi.setFloatParameter('metallic', 0.0);
      }
      const item = makeRenderable(this.engine, shape, mi, uv);
      item.geom = g;
      this.scene.addEntity(item.entity);
      this.items.push(item);
    }

    // The camera stays put across spawns.
    this.orbit ??= new OrbitCamera(this.canvas, [0, 0, 0.4], 4);
    this.simStart = performance.now() - this.data.time * 1000;
    this.syncTransforms();
  }

  // Binds MuJoCo texture `texid` to `mi`, creating it on first use. Image
  // textures are decoded asynchronously; until then `mi` shows plain white.
  bindTexture(texid, mi) {
    let entry = this.textures.get(texid);
    if (!entry) {
      entry = { tex: null, waiting: [] };
      this.textures.set(texid, entry);
      const name = this.mujoco.mj_id2name(this.model, this.mujoco.mjtObj.mjOBJ_TEXTURE.value, texid);
      const image = this.images.get(name);
      if (image) this.loadImage(entry, image, name);
      else entry.tex = mjTexture(this.engine, this.model, texid);
    }
    if (entry.tex) {
      mi.setTextureParameter('baseColorMap', entry.tex, this.sampler);
    } else {
      mi.setTextureParameter('baseColorMap', this.whiteTexture, this.sampler);
      entry.waiting.push(mi);
    }
  }

  async loadImage(entry, { read, cands }, name) {
    const gen = this.generation;
    try {
      const { w, h, rgba } = await decodeImage(await readFirst(read, cands));
      if (gen !== this.generation) return;  // model was replaced meanwhile
      entry.tex = uploadTexture(this.engine, w, h, rgba);
      for (const mi of entry.waiting) mi.setTextureParameter('baseColorMap', entry.tex, this.sampler);
      entry.waiting = [];
    } catch (err) {
      console.warn(`texture ${name}:`, err);
    }
  }

  clear() {
    this.generation++;
    for (const it of this.items) {
      this.scene.remove(it.entity);
      this.engine.destroyEntity(it.entity);
      this.engine.destroyVertexBuffer(it.vb);
      this.engine.destroyIndexBuffer(it.ib);
      this.engine.destroyMaterialInstance(it.matInstance);
    }
    this.items = [];
    for (const { tex } of this.textures.values()) if (tex) this.engine.destroyTexture(tex);
    this.textures.clear();
    this.data?.delete();
    this.model?.delete();
    this.data = this.model = null;
  }

  // Back to `this.initial` ({qpos, ctrl}: every robot's spawn pose), if set.
  reset() {
    this.mujoco.mj_resetData(this.model, this.data);
    if (this.initial) {
      this.data.qpos.set(this.initial.qpos);
      this.data.ctrl.set(this.initial.ctrl);
    }
    this.mujoco.mj_forward(this.model, this.data);
    this.simStart = performance.now();
    this.syncTransforms();  // frame() only syncs while running
  }

  // Where the pointer ray hits the z=0 floor, or null if it misses.
  pickFloor(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = 1 - ((clientY - rect.top) / rect.height) * 2;
    const eye = this.orbit.eye(), c = this.orbit.center;
    const norm = (v) => { const l = Math.hypot(...v); return v.map((k) => k / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const f = norm(c.map((k, i) => k - eye[i]));
    const r = norm(cross(f, [0, 0, 1])), u = cross(r, f);
    const t = Math.tan((45 / 2) * (Math.PI / 180));  // matches setProjectionFov(45, …, VERTICAL)
    const d = [0, 1, 2].map((i) => f[i] + r[i] * x * t * (rect.width / rect.height) + u[i] * y * t);
    if (d[2] > -1e-6) return null;
    const s = -eye[2] / d[2];
    return [eye[0] + s * d[0], eye[1] + s * d[1], 0];
  }

  syncTransforms() {
    const tcm = this.engine.getTransformManager();
    const xpos = this.data.geom_xpos, xmat = this.data.geom_xmat;
    for (const { geom: g, entity } of this.items) {
      const R = xmat.subarray(9 * g, 9 * g + 9), p = xpos.subarray(3 * g, 3 * g + 3);
      // MuJoCo xmat is row-major 3x3; Filament wants a column-major 4x4.
      const m = [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, p[0], p[1], p[2], 1];
      const inst = tcm.getInstance(entity);
      tcm.setTransform(inst, m);
      inst.delete();
    }
  }

  frame() {
    if (this.model && !this.paused) {
      // Keep simulation time locked to wall time (capped so a hidden tab doesn't explode).
      const target = (performance.now() - this.simStart) / 1000;
      const t0 = performance.now();
      while (this.data.time < target && performance.now() - t0 < 20) {
        this.mujoco.mj_step(this.model, this.data);
      }
      if (this.data.time < target) this.simStart = performance.now() - this.data.time * 1000;
      this.syncTransforms();
    }
    if (this.orbit) this.camera.lookAt(this.orbit.eye(), this.orbit.center, [0, 0, 1]);
    if (this.renderer.beginFrame(this.swapChain)) {
      this.renderer.renderView(this.view);
      this.renderer.endFrame();
    }
    this.engine.execute();
    requestAnimationFrame(() => this.frame());
  }
}

// ---------------------------------------------------------------------------

const status = document.getElementById('status');
const picker = document.getElementById('model');

function addOptions(label, entries) {  // entries: [[text, value]]
  const group = document.createElement('optgroup');
  group.label = label;
  for (const [text, value] of entries) group.append(new Option(text, value));
  picker.append(group);
  return group;
}

// Lets the browser paint the status text before a blocking compile.
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r)));

// Copies the state of the old model into `data` for the robots both models
// share. Robots are attached in order, so their qpos/qvel/act/ctrl/mocap
// entries form a common prefix.
function copySharedState(old, data, model) {
  const prefix = (name, n) => data[name].set(old[name].subarray(0, Math.min(n, data[name].length)));
  prefix('qpos', old.nq);
  prefix('qvel', old.nv);
  prefix('act', old.na);
  prefix('ctrl', old.nu);
  prefix('mocap_pos', 3 * old.nmocap);
  prefix('mocap_quat', 4 * old.nmocap);
  data.time = old.time;
}

function snapshot(app) {
  const { model: m, data: d } = app;
  if (!m) return null;
  const copy = (a) => Float64Array.from(a);
  return {
    nq: m.nq, nv: m.nv, na: m.na, nu: m.nu, nmocap: m.nmocap, time: d.time,
    qpos: copy(d.qpos), qvel: copy(d.qvel), act: copy(d.act), ctrl: copy(d.ctrl),
    mocap_pos: copy(d.mocap_pos), mocap_quat: copy(d.mocap_quat),
  };
}

try {
  const [mujoco] = await Promise.all([
    loadMujoco(),
    new Promise((resolve) => Filament.init([MAT_LIT, MAT_CHECKER, MAT_TEXTURED], resolve)),
  ]);
  Filament = window.Filament;
  const app = new App(mujoco, document.getElementById('canvas'));
  window.app = app;  // handy for poking at it from devtools

  // Menagerie index written by scripts/vendor.mjs; absent if not checked out.
  try {
    const res = await fetch('menagerie.json');
    if (res.ok) {
      const robots = await res.json();
      addOptions('mujoco_menagerie', robots.map((r) => [r.name, `${r.name}/${r.main}`]));
      const panda = robots.find((r) => r.name === 'franka_emika_panda');
      if (panda) picker.value = `${panda.name}/${panda.main}`;
    }
  } catch (err) {
    console.warn('no menagerie index', err);
  }

  // Picker values are menagerie paths: '<robot>/<file>.xml'.
  const menagerie = urlSource('menagerie');
  const robots = [];  // spawned, in attach order: {label, path, prefix, pos, key, images, read}
  let nextId = 1, busy = false;

  const hint = () => picker.value
    ? `${robots.length} robots · click the floor to spawn ${picker.selectedOptions[0].text}`
    : `${robots.length} robots`;

  // Compiles the world for `list` and swaps it in. `fresh` is a robot that is
  // not in the current model yet: it starts from its own keyframe, if any.
  function rebuild(list, fresh = null) {
    const model = compileWorld(mujoco, list);  // throws before anything changes
    const old = snapshot(app);
    const images = new Map();
    for (const r of list) {
      for (const [name, cands] of r.images) images.set(r.prefix + name, { read: r.read, cands });
    }
    app.load(model, images, (data) => {
      if (old) copySharedState(old, data, model);
      if (!fresh) return;
      const q0 = old?.nq ?? 0, u0 = old?.nu ?? 0;
      const key = fresh.key;
      if (key?.qpos?.length === model.nq - q0) {
        data.qpos.set(key.qpos, q0);
        // The keyframe is in the robot's own world frame: shift free joints
        // by the spawn point (qpos0 already includes it).
        for (let j = 0; j < model.njnt; j++) {
          const adr = model.jnt_qposadr[j];
          if (adr < q0 || model.jnt_type[j] !== 0) continue;
          for (let k = 0; k < 3; k++) data.qpos[adr + k] += fresh.pos[k];
        }
      }
      if (key?.ctrl?.length === model.nu - u0) data.ctrl.set(key.ctrl, u0);
    });
    // Reset returns every robot to its spawn pose.
    const prev = app.initial;
    const initial = { qpos: Float64Array.from(app.data.qpos), ctrl: Float64Array.from(app.data.ctrl) };
    if (prev) {
      initial.qpos.set(prev.qpos.subarray(0, Math.min(prev.qpos.length, initial.qpos.length)));
      initial.ctrl.set(prev.ctrl.subarray(0, Math.min(prev.ctrl.length, initial.ctrl.length)));
    }
    app.initial = initial;
  }

  async function spawn(rel, pos) {
    busy = true;
    const t0 = performance.now();
    const label = picker.selectedOptions[0]?.text ?? rel;
    try {
      const progress = (msg) => { status.textContent = msg; };
      const { path, images, docs } = await loadModelFiles(mujoco, menagerie, '/menagerie', rel, progress);
      status.textContent = `compiling ${robots.length + 1} robots…`;
      await nextFrame();
      const robot = { label, path, prefix: `r${nextId}/`, pos, key: firstKey(docs), images, read: menagerie };
      rebuild([...robots, robot], robot);
      robots.push(robot);
      nextId++;
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      status.textContent = `spawned ${label} (${secs}s) · ${hint()}`;
    } catch (err) {
      status.textContent = `spawn failed: ${err.message ?? err}`;
      console.error(err);
    } finally {
      busy = false;
    }
  }

  // Undo / clear: recompile with fewer robots (state of the rest is kept).
  async function respawn(list) {
    if (busy) return;
    busy = true;
    try {
      status.textContent = `compiling ${list.length} robots…`;
      await nextFrame();
      rebuild(list);
      robots.length = 0;
      robots.push(...list);
      status.textContent = hint();
    } catch (err) {
      status.textContent = `failed: ${err.message ?? err}`;
      console.error(err);
    } finally {
      busy = false;
    }
  }

  // A click (not a drag: that orbits/pans) on the floor spawns the picked robot.
  const canvas = document.getElementById('canvas');
  let down = null;
  canvas.addEventListener('pointerdown', (e) => { down = e.button === 0 ? [e.clientX, e.clientY] : null; });
  canvas.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    down = null;
    if (!picker.value) return;
    if (busy) { status.textContent = 'busy, try again in a moment'; return; }
    const hit = app.pickFloor(e.clientX, e.clientY);
    if (hit) spawn(picker.value, hit);
  });

  picker.onchange = () => { if (!busy) status.textContent = hint(); };
  document.getElementById('pause').onclick = (e) => {
    app.paused = !app.paused;
    e.target.textContent = app.paused ? 'Resume' : 'Pause';
    if (!app.paused && app.data) app.simStart = performance.now() - app.data.time * 1000;
  };
  document.getElementById('reset').onclick = () => app.model && app.reset();
  document.getElementById('undo').onclick = () => robots.length && respawn(robots.slice(0, -1));
  document.getElementById('clear').onclick = () => robots.length && respawn([]);

  rebuild([]);
  status.textContent = hint();
  app.frame();
} catch (err) {
  status.textContent = `error: ${err.message ?? err}`;
  throw err;
}
