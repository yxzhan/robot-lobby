// Flex (soft) bodies: cables and rods (1D), cloth and shells (2D), soft
// solids (3D), drawn as MuJoCo 4's own Filament renderer draws them
// (mjrf: src/render/filament/support/mesh_util.cc FillFlexVertices,
// model_renderables.cc):
// - 2D: each element (a triangle) twice, its front offset by the flex radius
//   along the vertex normals and its back by minus that, facing away; and
//   along the boundary (the shell's edges) side faces closing the gap
//   between the two.
// - 3D: the boundary (the shell's triangles), offset by the radius.
// - 1D: each edge a rod (a cylinder of the flex radius) and each vertex a
//   ball, so that the rods join (mjrf draws capsules).
// Vertex normals are the sums of the face normals around each vertex
// (smooth), or each face's own (flat, flex_flatskin). Vertices come from
// mjData's flexvert_xpos (world coordinates) every frame: in network mode
// from the server's states (net.js). One material per flex: the first
// element's, else flex_rgba (mjrf splits a flex with several materials into
// slices; no example does that).
//
// flexTemplates(model) reads a compiled model (WASM, or the server's
// render data with the same field names: server/render_asset.py) once per
// model file; a FlexDraw is one spawned copy's flex on the GPU.

// The static part of drawing each of `model`'s flexes: [{flex, dim,
// vertadr, vertnum, radius, flat, matid, rgba, group, …}], with for 2D / 3D
// the output triangles (`tris`: per output triangle its kind, 0 face / 1
// side, and three local vertex indices; `signs`: +1 front / -1 back), the
// triangles whose normals make the vertex normals (`ntris`), and texture
// coordinates per output vertex (`uv`, or null); for 1D its edges.
export function flexTemplates(model) {
  const out = [];
  for (let f = 0; f < (model.nflex ?? 0); f++) {
    const dim = model.flex_dim[f], elemnum = model.flex_elemnum[f], elemadr = model.flex_elemadr[f];
    const t = {
      flex: f, dim, vertadr: model.flex_vertadr[f], vertnum: model.flex_vertnum[f],
      radius: model.flex_radius[f], flat: !!model.flex_flatskin[f],
      matid: elemnum > 0 ? model.flex_matid[elemadr] : -1,
      rgba: Array.from(model.flex_rgba.subarray(4 * f, 4 * f + 4)),
      group: model.flex_group[f],
    };
    const edata = model.flex_elem.subarray(model.flex_elemdataadr[f], model.flex_elemdataadr[f] + elemnum * (dim + 1));
    const shellnum = model.flex_shellnum[f];
    const sdata = model.flex_shell.subarray(model.flex_shelldataadr[f], model.flex_shelldataadr[f] + shellnum * dim);
    if (dim === 1) {
      const a = model.flex_edgeadr[f], n = model.flex_edgenum[f];
      t.edges = Int32Array.from(model.flex_edge.subarray(2 * a, 2 * (a + n)));
      out.push(t);
      continue;
    }
    const tris = [], signs = [], uvIdx = [];
    const tdata = model.flex_elemtexcoord?.subarray(model.flex_elemdataadr[f], model.flex_elemdataadr[f] + elemnum * (dim + 1));
    if (dim === 2) {
      for (let e = 0; e < elemnum; e++) {
        const [i0, i1, i2] = edata.subarray(3 * e, 3 * e + 3), [u0, u1, u2] = tdata ? tdata.subarray(3 * e, 3 * e + 3) : [i0, i1, i2];
        tris.push(0, i0, i1, i2, 0, i0, i2, i1);
        signs.push(1, -1);
        uvIdx.push(u0, u1, u2, u0, u2, u1);
      }
      for (let s = 0; s < shellnum; s++) {
        const i0 = sdata[2 * s], i1 = sdata[2 * s + 1];
        tris.push(1, i0, i1, i1, 1, i1, i0, i0);
        signs.push(1, -1);
        uvIdx.push(i0, i1, i1, i1, i0, i0);
      }
      t.ntris = edata;
    } else {
      for (let s = 0; s < shellnum; s++) {
        const [i0, i1, i2] = sdata.subarray(3 * s, 3 * s + 3);
        tris.push(0, i0, i1, i2);
        signs.push(1);
        uvIdx.push(i0, i1, i2);
      }
      t.ntris = sdata;
    }
    t.tris = Int32Array.from(tris);
    t.signs = Int8Array.from(signs);
    const tc = model.flex_texcoordadr?.[f] ?? -1;
    if (tc >= 0) {
      const src = model.flex_texcoord;
      t.uv = new Float32Array(2 * uvIdx.length);
      uvIdx.forEach((k, i) => { t.uv[2 * i] = src[2 * (tc + k)]; t.uv[2 * i + 1] = src[2 * (tc + k) + 1]; });
    } else t.uv = null;
    out.push(t);
  }
  return out;
}

// Whether a template is drawn (MuJoCo shows flex groups 0-2 by default).
export const flexVisible = (t) => t.group < 3 && t.rgba[3] > 0;

// One spawned copy's flex on the GPU, in `scene`, drawn with material
// instance `mi`. `h`: helpers from main.js {makeEntity, rodMatrix,
// unitRod, unitBall} (the unit cylinder and sphere geometry, shared).
export class FlexDraw {
  constructor(engine, scene, t, mi, h) {
    this.engine = engine;
    this.scene = scene;
    this.t = t;
    this.h = h;
    this.entities = [];
    if (t.dim === 1) {
      for (let k = 0; k < t.edges.length / 2 + t.vertnum; k++) {
        const e = h.makeEntity(engine, k < t.edges.length / 2 ? h.unitRod : h.unitBall, mi);
        scene.addEntity(e);
        this.entities.push(e);
      }
      return;
    }
    const n = 3 * t.signs.length;  // output vertices
    this.pos = new Float32Array(3 * n);
    this.quat = new Int16Array(4 * n);
    this.normals = new Float32Array(3 * t.vertnum);
    const AT = Filament.VertexBuffer$AttributeType, VA = Filament.VertexAttribute;
    const vbb = Filament.VertexBuffer.Builder()
      .vertexCount(n).bufferCount(t.uv ? 3 : 2)
      .attribute(VA.POSITION, 0, AT.FLOAT3, 0, 12)
      .attribute(VA.TANGENTS, 1, AT.SHORT4, 0, 8)
      .normalized(VA.TANGENTS);
    if (t.uv) vbb.attribute(VA.UV0, 2, AT.FLOAT2, 0, 8);
    this.vb = vbb.build(engine);
    if (t.uv) this.vb.setBufferAt(engine, 2, t.uv);
    this.ib = Filament.IndexBuffer.Builder().indexCount(n).bufferType(Filament.IndexBuffer$IndexType.UINT).build(engine);
    this.ib.setBuffer(engine, Uint32Array.from({ length: n }, (_, i) => i));
    this.vb.setBufferAt(engine, 0, this.pos);
    this.vb.setBufferAt(engine, 1, this.quat);
    const e = h.makeEntity(engine, { vb: this.vb, ib: this.ib, box: { center: [0, 0, 0], halfExtent: [1e-3, 1e-3, 1e-3] } }, mi);
    scene.addEntity(e);
    this.entities.push(e);
  }

  setMaterial(mi) {
    const rcm = this.engine.getRenderableManager();
    for (const e of this.entities) {
      const inst = rcm.getInstance(e);
      rcm.setMaterialInstanceAt(inst, 0, mi);
      inst.delete();
    }
  }

  // Its vertices from `xpos` (mjData flexvert_xpos of the world), its
  // first at world vertex `base`.
  update(xpos, base) {
    const t = this.t, at = 3 * base;
    if (xpos.length < at + 3 * t.vertnum) return;
    const v = (i) => at + 3 * i;
    if (t.dim === 1) return this.updateRods(xpos, v);

    // Vertex normals: the sum of the unit normals of the triangles around.
    const N = this.normals.fill(0), nt = t.ntris, step = t.dim === 2 ? 3 : 3;
    for (let k = 0; k + 2 < nt.length; k += step) {
      const a = v(nt[k]), b = v(nt[k + 1]), c = v(nt[k + 2]);
      const n = unitCross(xpos[b] - xpos[a], xpos[b + 1] - xpos[a + 1], xpos[b + 2] - xpos[a + 2],
        xpos[c] - xpos[a], xpos[c + 1] - xpos[a + 1], xpos[c + 2] - xpos[a + 2]);
      for (const i of [nt[k], nt[k + 1], nt[k + 2]]) { N[3 * i] += n[0]; N[3 * i + 1] += n[1]; N[3 * i + 2] += n[2]; }
    }
    for (let i = 0; i < t.vertnum; i++) {
      const l = Math.hypot(N[3 * i], N[3 * i + 1], N[3 * i + 2]);
      if (l > 1e-16) { N[3 * i] /= l; N[3 * i + 1] /= l; N[3 * i + 2] /= l; } else { N[3 * i] = 1; N[3 * i + 1] = 0; N[3 * i + 2] = 0; }
    }

    const P = this.pos, Q = this.quat, tris = t.tris, signs = t.signs;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    let o = 0;
    const put = (p, n) => {
      for (let j = 0; j < 3; j++) {
        P[3 * o + j] = p[j];
        if (p[j] < lo[j]) lo[j] = p[j];
        if (p[j] > hi[j]) hi[j] = p[j];
      }
      tangentQuat(Q, 4 * o, n);
      o++;
    };
    const X = (i) => [xpos[v(i)], xpos[v(i) + 1], xpos[v(i) + 2]];
    const Nv = (i) => [N[3 * i], N[3 * i + 1], N[3 * i + 2]];
    for (let k = 0; k < signs.length; k++) {
      const kind = tris[4 * k], i0 = tris[4 * k + 1], i1 = tris[4 * k + 2], i2 = tris[4 * k + 3];
      const r = t.radius * signs[k];
      if (kind === 0) {  // a face (AddFlexFace)
        const flat = t.flat ? unitCross(...sub(X(i1), X(i0)), ...sub(X(i2), X(i0))) : null;
        for (const i of [i0, i1, i2]) {
          const n = Nv(i), x = X(i);
          put([x[0] + r * n[0], x[1] + r * n[1], x[2] + r * n[2]], flat ?? (signs[k] > 0 ? n : [-n[0], -n[1], -n[2]]));
        }
      } else {  // a side face along the boundary (AddSideFace)
        const x0 = X(i0), x1 = X(i1), n0 = Nv(i0), n1 = Nv(i1);
        let n = unitCross(...sub(x1, x0), ...n1);
        if (r < 0) n = [-n[0], -n[1], -n[2]];
        put([x0[0] + r * n0[0], x0[1] + r * n0[1], x0[2] + r * n0[2]], n);
        put([x1[0] - r * n1[0], x1[1] - r * n1[1], x1[2] - r * n1[2]], n);
        put([x1[0] + r * n1[0], x1[1] + r * n1[1], x1[2] + r * n1[2]], n);
      }
    }
    this.vb.setBufferAt(this.engine, 0, P);
    this.vb.setBufferAt(this.engine, 1, Q);
    const rcm = this.engine.getRenderableManager(), inst = rcm.getInstance(this.entities[0]);
    rcm.setAxisAlignedBoundingBox(inst, {
      center: lo.map((l, j) => (l + hi[j]) / 2),
      halfExtent: lo.map((l, j) => Math.max(1e-4, (hi[j] - l) / 2)),
    });
    inst.delete();
  }

  updateRods(xpos, v) {
    const t = this.t, w = Math.max(t.radius, 1e-3), tcm = this.engine.getTransformManager();
    const ne = t.edges.length / 2;
    const set = (e, m) => { const inst = tcm.getInstance(e); tcm.setTransform(inst, m); inst.delete(); };
    for (let k = 0; k < ne; k++) {
      const a = v(t.edges[2 * k]), b = v(t.edges[2 * k + 1]);
      const m = this.h.rodMatrix([xpos[a], xpos[a + 1], xpos[a + 2]], [xpos[b], xpos[b + 1], xpos[b + 2]], w);
      set(this.entities[k], m ?? [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    }
    for (let i = 0; i < t.vertnum; i++) {
      const a = v(i);
      set(this.entities[ne + i], [w, 0, 0, 0, 0, w, 0, 0, 0, 0, w, 0, xpos[a], xpos[a + 1], xpos[a + 2], 1]);
    }
  }

  destroy() {
    for (const e of this.entities) {
      this.scene.remove(e);
      this.engine.destroyEntity(e);
    }
    this.entities = [];
    if (this.vb) this.engine.destroyVertexBuffer(this.vb);
    if (this.ib) this.engine.destroyIndexBuffer(this.ib);
    this.vb = this.ib = null;
  }
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

function unitCross(ax, ay, az, bx, by, bz) {
  const x = ay * bz - az * by, y = az * bx - ax * bz, z = ax * by - ay * bx;
  const l = Math.hypot(x, y, z);
  return l > 1e-20 ? [x / l, y / l, z / l] : [0, 0, 1];
}

// The tangent frame quaternion (Filament's TANGENTS, SHORT4 normalized) of a
// vertex with normal n: the rotation of +z onto n (the tangents follow; no
// normal maps on flexes), written at Q[o..o+3] as x, y, z, w with w > 0
// (Filament reads a negative w as a mirrored bitangent).
function tangentQuat(Q, o, n) {
  let x = -n[1], y = n[0], z = 0, w = 1 + n[2];
  if (w < 1e-6) { x = 1; y = 0; w = 0; }
  const l = Math.hypot(x, y, z, w);
  x /= l; y /= l; w /= l;
  if (w < 1 / 32767) w = 1 / 32767;
  Q[o] = Math.round(x * 32767);
  Q[o + 1] = Math.round(y * 32767);
  Q[o + 2] = 0;
  Q[o + 3] = Math.round(w * 32767);
}
