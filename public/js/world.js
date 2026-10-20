import { geomOverrides, optionXml } from './physics.js';

// The shared world: an empty floor that robots are spawned into.
//
// MuJoCo cannot add bodies to a compiled MjModel, so every spawn rebuilds an
// MjSpec from the base XML, attaches each robot's own spec (mjs_attach) under a
// frame at its spawn point with a unique name prefix, and compiles the lot.
// Rebuilding from scratch (instead of editing one long-lived spec) also makes
// a failed spawn or a removal trivial: just compile a different robot list.
//
// The world model is physics-only: each robot's visual geoms (contype and
// conaffinity 0) and the meshes only they use are removed before attaching.
// Menagerie's visual meshes are dense, and compiling them (normals, BVHs,
// convex hulls) costs hundreds of MB of the 2 GB WASM heap per robot; the
// renderer gets them from a separate per-robot compile instead (main.js).

// The world spec: the global options (physics.js: gravity, wind, …) and the
// default geom properties, which here only reach the floor.
export function baseXml(physics) {
  const geom = Object.entries(geomOverrides(physics)).map(([k, v]) => `${k}="${v}"`).join(' ');
  return `
<mujoco model="world">
  ${optionXml(physics)}
  <default><geom ${geom}/></default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <frame name="spawn_root"/>
  </worldbody>
</mujoco>`;
}

// robots: [{path, prefix, pos, yaw, strip}] where `path` is the robot's MJCF
// in the MuJoCo FS (with its includes and assets already there, see vfs.js)
// and `yaw` (rad, about +z) its heading at the spawn point `pos`.
// `strip` ({refs, inertia}, or null to keep the visual geoms) gives the names
// referenced elsewhere in the robot (referencedNames) and the body inertias
// of its full compile (bodyInertia). `physics` are the world settings
// (physics.js). Returns the compiled MjModel; throws on parse/compile errors.
export function compileWorld(mujoco, robots, physics) {
  const world = mujoco.parseXMLString(baseXml(physics));
  try {
    const worldbody = mujoco.mjs_findBody(world, 'world');
    // The bindings do not accept null for mjs_addFrame's parent frame, so new
    // frames hang off an identity frame declared in the base XML.
    const root = mujoco.mjs_findFrame(world, 'spawn_root');
    for (const r of robots) {
      const dir = r.path.slice(0, r.path.lastIndexOf('/') + 1);
      // <include> is resolved at parse time relative to the working directory;
      // asset files at compile time relative to modelfiledir.
      const cwd = mujoco.FS.cwd();
      mujoco.FS.chdir(dir);
      let child;
      try {
        child = mujoco.parseXMLString(mujoco.FS.readFile(r.path, { encoding: 'utf8' }));
      } finally {
        mujoco.FS.chdir(cwd);
      }
      try {
        child.modelfiledir = dir;
        if (r.strip) {
          pinInertia(mujoco, child, r.strip.inertia);
          stripVisualGeoms(mujoco, child, r.strip.refs);
        }
        const frame = mujoco.mjs_addFrame(worldbody, root);
        frame.pos.set(r.pos);
        frame.quat.set([Math.cos((r.yaw ?? 0) / 2), 0, 0, Math.sin((r.yaw ?? 0) / 2)]);
        if (!mujoco.mjs_attach(frame.element, child.element, r.prefix, '')) {
          throw new Error(`could not attach ${r.path}`);
        }
      } finally {
        child.delete();  // attach copies the child
      }
    }
    return mujoco.mj_compile(world);
  } finally {
    world.delete();
  }
}

// The robot's own first keyframe ({qpos, ctrl}, either may be null), read
// from its XML documents. Keyframes merged by mjs_attach are not used: they
// do not carry the spawn offset for free joints.
export function firstKey(docs) {
  const nums = (s) => (s ? s.trim().split(/\s+/).map(Number) : null);
  for (const doc of docs) {
    for (const key of doc.getElementsByTagName('key')) {
      if (key.parentNode?.nodeName !== 'keyframe') continue;
      return { qpos: nums(key.getAttribute('qpos')), ctrl: nums(key.getAttribute('ctrl')) };
    }
  }
  return null;
}

// Names that something other than a geom or mesh refers to (contact pairs,
// sensors, tendon wrapping surfaces, …), from the robot's XML documents.
// Such geoms are never stripped.
export function referencedNames(docs) {
  const refs = new Set();
  for (const doc of docs) {
    for (const el of doc.getElementsByTagName('*')) {
      if (el.nodeName === 'mesh') continue;
      if (el.nodeName === 'geom') {
        // A spatial tendon's wrap, <geom geom="…" sidesite="…"/>: the wrapped
        // geom is often contype 0 (MS-Human's muscles), yet the tendon needs it.
        if (el.parentNode?.nodeName === 'spatial') refs.add(el.getAttribute('geom'));
        continue;
      }
      for (const a of el.attributes) if (a.name !== 'name') refs.add(a.value);
    }
  }
  return refs;
}

// Per-body inertial properties of a compiled model, indexed by body id.
export function bodyInertia(model) {
  const copy = (a) => Float64Array.from(a);
  return {
    nbody: model.nbody, mass: copy(model.body_mass), ipos: copy(model.body_ipos),
    iquat: copy(model.body_iquat), inertia: copy(model.body_inertia),
  };
}

function allElements(mujoco, spec, type) {
  const out = [];
  for (let e = mujoco.mjs_firstElement(spec, type); e; e = mujoco.mjs_nextElement(spec, e)) out.push(e);
  return out;
}

// Gives every body the explicit inertial of the full model, so that removing
// visual geoms (which a body without <inertial> takes its mass from) leaves
// the dynamics unchanged. Spec bodies are listed in compiled body order.
function pinInertia(mujoco, spec, inertia) {
  const bodies = allElements(mujoco, spec, mujoco.mjtObj.mjOBJ_BODY);
  const first = inertia.nbody - bodies.length;  // 1 if the world body is not listed
  if (first !== 0 && first !== 1) throw new Error(`body count mismatch: ${bodies.length} vs ${inertia.nbody}`);
  for (const [i, e] of bodies.entries()) {
    const k = first + i;
    if (k === 0) continue;
    const b = mujoco.mjs_asBody(e);
    b.explicitinertial = true;
    b.mass = inertia.mass[k];
    b.ipos.set(inertia.ipos.subarray(3 * k, 3 * k + 3));
    b.iquat.set(inertia.iquat.subarray(4 * k, 4 * k + 4));
    b.inertia.set(inertia.inertia.subarray(3 * k, 3 * k + 3));
    b.fullinertia[0] = NaN;  // use inertia + iquat
  }
  spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE;
}

// Deletes geoms that cannot collide and nothing refers to, then the meshes no
// remaining geom uses.
function stripVisualGeoms(mujoco, spec, refs) {
  const T = mujoco.mjtObj;
  const all = (type) => allElements(mujoco, spec, type);
  const usedMeshes = new Set();
  for (const e of all(T.mjOBJ_GEOM)) {
    const g = mujoco.mjs_asGeom(e);
    const name = mujoco.mjs_getName(e);
    if (g.contype === 0 && g.conaffinity === 0 && !(name && refs.has(name))) mujoco.mjs_delete(spec, e);
    else if (g.meshname) usedMeshes.add(g.meshname);
  }
  for (const e of all(T.mjOBJ_MESH)) {
    // An unnamed mesh is named after its file stem.
    const name = mujoco.mjs_getName(e) || mujoco.mjs_asMesh(e).file.split('/').pop().replace(/\.[^.]*$/, '');
    if (!usedMeshes.has(name) && !refs.has(name)) mujoco.mjs_delete(spec, e);
  }
}
