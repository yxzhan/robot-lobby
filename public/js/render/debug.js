// Debug overlays (the Render panel's Show: Collision, Body frames, Joints):
// line segments per body of a robot file, from its model's tables, drawn
// over everything by main.js (App.setOverlays) with the unlit_lines
// material, moved with the bodies like the robot's geoms.
//
//   collision  the geoms that collide (contype or conaffinity set), as
//              wireframes: boxes their edges, spheres, ellipsoids, capsules
//              and cylinders their outlines, meshes the edges of the convex
//              hull MuJoCo collides with (mesh_graph), planes a rectangle
//   frames     each body's x, y and z axes (red, green, blue) at its origin
//   joints     hinges their axis and a ring about it, slides their axis
//              with ticks at the ends, ball joints three rings
//
// The model is a compiled MjModel (local mode) or the server's render data
// (net.js loadRender), which has the colliding geoms as col_* tables and
// their meshes' hulls as col_mesh_* (server/render_asset.py).
//
// debugLines(model) -> {collision, frames, joints}, each a list of
// {body, pos: Float32Array (6 per segment, in the body's frame), col:
// Uint8Array (8 per segment: RGBA at both ends)}, body the robot file's
// body index (0: its world body, placed by the robot's spawn pose).

export const FRAME_LENGTH = 0.08;  // m, a body frame's axes
const JOINT_LENGTH = 0.06;         // m, half a joint's axis
const JOINT_RING = 0.02;           // m, the ring about a hinge
const PLANE_SIZE = 1;              // m, half the side of an infinite plane's rectangle
const CIRCLE = 32;                 // segments of a full circle

const COLORS = {
  collision: [255, 140, 30, 230],
  x: [235, 60, 60, 255], y: [60, 200, 80, 255], z: [70, 130, 255, 255],
  hinge: [255, 220, 40, 255], slide: [40, 220, 230, 255], ball: [230, 80, 230, 255],
};

// Segments collected per body, then packed.
class Lines {
  constructor() { this.byBody = new Map(); }

  add(body, a, b, color) {
    let l = this.byBody.get(body);
    if (!l) this.byBody.set(body, (l = { pos: [], col: [] }));
    l.pos.push(...a, ...b);
    l.col.push(...color, ...color);
  }

  // A closed polyline through the points.
  loop(body, pts, color) {
    for (let i = 0; i < pts.length; i++) this.add(body, pts[i], pts[(i + 1) % pts.length], color);
  }

  packed() {
    return [...this.byBody].map(([body, l]) => ({ body, pos: Float32Array.from(l.pos), col: Uint8Array.from(l.col) }));
  }
}

// Row-major 3x3 of a unit quaternion (w, x, y, z).
function quatMat([w, x, y, z]) {
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}
const apply = (R, t, v) => [0, 1, 2].map((r) => t[r] + R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2]);

// An ellipse in the plane of axes i and j (0 x, 1 y, 2 z) with radii ri, rj,
// at offset `at`, the angle from a0 to a1.
function arc(ri, rj, i, j, at = [0, 0, 0], a0 = 0, a1 = 2 * Math.PI, n = CIRCLE) {
  const steps = Math.max(2, Math.round(n * (a1 - a0) / (2 * Math.PI)));
  const pts = [];
  for (let k = 0; k <= steps; k++) {
    const a = a0 + (a1 - a0) * k / steps, p = [...at];
    p[i] += ri * Math.cos(a);
    p[j] += rj * Math.sin(a);
    pts.push(p);
  }
  return pts;
}
const ring = (ri, rj, i, j, at) => arc(ri, rj, i, j, at).slice(0, -1);  // closed: the last point is the first

// A geom's outline in its own frame: a list of segments [a, b].
function geomSegments(type, s, mesh) {
  const segs = [], strip = (pts) => { for (let k = 0; k + 1 < pts.length; k++) segs.push([pts[k], pts[k + 1]]); };
  const loop = (pts) => { for (let k = 0; k < pts.length; k++) segs.push([pts[k], pts[(k + 1) % pts.length]]); };
  switch (type) {
    case 0: {  // plane: a rectangle (an infinite one PLANE_SIZE across) and its middle cross
      const x = s[0] || PLANE_SIZE, y = s[1] || PLANE_SIZE;
      loop([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]]);
      segs.push([[-x, 0, 0], [x, 0, 0]], [[0, -y, 0], [0, y, 0]]);
      break;
    }
    case 2:  // sphere
      loop(ring(s[0], s[0], 0, 1)); loop(ring(s[0], s[0], 1, 2)); loop(ring(s[0], s[0], 2, 0));
      break;
    case 4:  // ellipsoid
      loop(ring(s[0], s[1], 0, 1)); loop(ring(s[1], s[2], 1, 2)); loop(ring(s[2], s[0], 2, 0));
      break;
    case 3: {  // capsule: the cylinder's circles and sides, the caps' arcs
      const [r, h] = s;
      loop(ring(r, r, 0, 1, [0, 0, h])); loop(ring(r, r, 0, 1, [0, 0, -h]));
      for (const [x, y] of [[r, 0], [-r, 0], [0, r], [0, -r]]) segs.push([[x, y, -h], [x, y, h]]);
      strip(arc(r, r, 0, 2, [0, 0, h], 0, Math.PI)); strip(arc(r, r, 1, 2, [0, 0, h], 0, Math.PI));
      strip(arc(r, r, 0, 2, [0, 0, -h], Math.PI, 2 * Math.PI)); strip(arc(r, r, 1, 2, [0, 0, -h], Math.PI, 2 * Math.PI));
      break;
    }
    case 5: {  // cylinder
      const [r, h] = s;
      loop(ring(r, r, 0, 1, [0, 0, h])); loop(ring(r, r, 0, 1, [0, 0, -h]));
      for (const [x, y] of [[r, 0], [-r, 0], [0, r], [0, -r]]) segs.push([[x, y, -h], [x, y, h]]);
      break;
    }
    case 6: {  // box: its 12 edges
      const c = (i) => [(i & 1 ? 1 : -1) * s[0], (i & 2 ? 1 : -1) * s[1], (i & 4 ? 1 : -1) * s[2]];
      for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) if (!(i & bit)) segs.push([c(i), c(i | bit)]);
      break;
    }
    case 7:  // mesh: its triangles' edges, each once
      if (mesh) {
        const { vert, face } = mesh, seen = new Set();
        for (let f = 0; f < face.length; f += 3) {
          for (let k = 0; k < 3; k++) {
            const a = face[f + k], b = face[f + (k + 1) % 3], key = a < b ? a * 1e7 + b : b * 1e7 + a;
            if (seen.has(key)) continue;
            seen.add(key);
            segs.push([[vert[3 * a], vert[3 * a + 1], vert[3 * a + 2]], [vert[3 * b], vert[3 * b + 1], vert[3 * b + 2]]]);
          }
        }
      }
      break;
    default: break;  // hfield, sdf: not drawn
  }
  return segs;
}

// A compiled model's mesh `id` as {vert, face}: the convex hull MuJoCo
// collides with when it has one (mesh_graph: numvert, numface,
// vert_edgeadr, vert_globalid, edge_localid, face_globalid), else all of it.
function wasmMesh(model, id) {
  const vadr = model.mesh_vertadr[id], nv = model.mesh_vertnum[id];
  const vert = model.mesh_vert.subarray(3 * vadr, 3 * (vadr + nv));
  const g = model.mesh_graphadr?.[id] ?? -1;
  if (g >= 0 && model.mesh_graph) {
    const G = model.mesh_graph, numvert = G[g], numface = G[g + 1];
    const at = g + 2 + numvert + numvert + (numvert + 3 * numface);
    return { vert, face: G.subarray(at, at + 3 * numface) };
  }
  const fadr = model.mesh_faceadr[id], nf = model.mesh_facenum[id];
  return { vert, face: model.mesh_face.subarray(3 * fadr, 3 * (fadr + nf)) };
}

// The colliding geoms as {type, size, body, pos, quat, mesh}.
function colliders(model) {
  const out = [];
  const take = (prefix, g, mesh) => out.push({
    type: model[`${prefix}geom_type`][g], size: model[`${prefix}geom_size`].subarray(3 * g, 3 * g + 3),
    body: model[`${prefix}geom_bodyid`][g], pos: model[`${prefix}geom_pos`].subarray(3 * g, 3 * g + 3),
    quat: model[`${prefix}geom_quat`].subarray(4 * g, 4 * g + 4), mesh,
  });
  if (model.col_geom_type) {  // the server's render data: col_* tables
    const n = model.col_geom_type.length;
    for (let g = 0; g < n; g++) {
      const id = model.col_geom_dataid[g];
      let mesh = null;
      if (id >= 0) {
        const va = model.col_mesh_vertadr[id], vn = model.col_mesh_vertnum[id];
        const fa = model.col_mesh_faceadr[id], fn = model.col_mesh_facenum[id];
        mesh = { vert: model.col_mesh_vert.subarray(3 * va, 3 * (va + vn)), face: model.col_mesh_face.subarray(3 * fa, 3 * (fa + fn)) };
      }
      take('col_', g, mesh);
    }
    return out;
  }
  if (!model.geom_contype) return out;  // render data from an older server: none
  for (let g = 0; g < model.ngeom; g++) {
    if (!model.geom_contype[g] && !model.geom_conaffinity[g]) continue;
    take('', g, model.geom_type[g] === 7 ? wasmMesh(model, model.geom_dataid[g]) : null);
  }
  return out;
}

// Each joint's body: jnt_bodyid, or from body_jntadr / body_jntnum.
function jointBodies(model) {
  if (model.jnt_bodyid) return model.jnt_bodyid;
  const out = new Int32Array(model.njnt);
  for (let b = 0; b < model.nbody; b++) {
    for (let k = 0; k < model.body_jntnum[b]; k++) out[model.body_jntadr[b] + k] = b;
  }
  return out;
}

// Two unit vectors at right angles to unit `a`.
function across(a) {
  const t = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = [a[1] * t[2] - a[2] * t[1], a[2] * t[0] - a[0] * t[2], a[0] * t[1] - a[1] * t[0]];
  const n = Math.hypot(...u);
  const e1 = u.map((v) => v / n);
  return [e1, [a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2], a[0] * e1[1] - a[1] * e1[0]]];
}
const add = (p, v, k = 1) => p.map((x, i) => x + k * v[i]);

export function debugLines(model) {
  const collision = new Lines(), frames = new Lines(), joints = new Lines();

  for (const g of colliders(model)) {
    const R = quatMat(g.quat);
    for (const [a, b] of geomSegments(g.type, g.size, g.mesh)) collision.add(g.body, apply(R, g.pos, a), apply(R, g.pos, b), COLORS.collision);
  }

  for (let b = 1; b < model.nbody; b++) {
    const o = [0, 0, 0];
    frames.add(b, o, [FRAME_LENGTH, 0, 0], COLORS.x);
    frames.add(b, o, [0, FRAME_LENGTH, 0], COLORS.y);
    frames.add(b, o, [0, 0, FRAME_LENGTH], COLORS.z);
  }

  const jb = jointBodies(model);
  for (let j = 0; j < model.njnt; j++) {
    const type = model.jnt_type[j], body = jb[j];
    const p = Array.from(model.jnt_pos.subarray(3 * j, 3 * j + 3));
    const a = Array.from(model.jnt_axis.subarray(3 * j, 3 * j + 3));
    const [e1, e2] = across(a);
    const circle = (c, u, v, r) => {
      const pts = [];
      for (let k = 0; k < CIRCLE; k++) {
        const t = 2 * Math.PI * k / CIRCLE;
        pts.push(add(add(c, u, r * Math.cos(t)), v, r * Math.sin(t)));
      }
      return pts;
    };
    if (type === 3) {  // hinge: the axis and a ring about it
      joints.add(body, add(p, a, -JOINT_LENGTH), add(p, a, JOINT_LENGTH), COLORS.hinge);
      joints.loop(body, circle(p, e1, e2, JOINT_RING), COLORS.hinge);
    } else if (type === 2) {  // slide: the axis, ticks across its ends
      joints.add(body, add(p, a, -JOINT_LENGTH), add(p, a, JOINT_LENGTH), COLORS.slide);
      for (const k of [-1, 1]) {
        const end = add(p, a, k * JOINT_LENGTH);
        joints.add(body, add(end, e1, -JOINT_RING / 2), add(end, e1, JOINT_RING / 2), COLORS.slide);
      }
    } else if (type === 1) {  // ball: three rings
      joints.loop(body, circle(p, [1, 0, 0], [0, 1, 0], JOINT_RING), COLORS.ball);
      joints.loop(body, circle(p, [0, 1, 0], [0, 0, 1], JOINT_RING), COLORS.ball);
      joints.loop(body, circle(p, [0, 0, 1], [1, 0, 0], JOINT_RING), COLORS.ball);
    }  // free: the body's frame shows it
  }

  return { collision: collision.packed(), frames: frames.packed(), joints: joints.packed() };
}
