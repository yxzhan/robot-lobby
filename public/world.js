// The shared world: an empty floor that robots are spawned into.
//
// MuJoCo cannot add bodies to a compiled MjModel, so every spawn rebuilds an
// MjSpec from BASE_XML, attaches each robot's own spec (mjs_attach) under a
// frame at its spawn point with a unique name prefix, and compiles the lot.
// Rebuilding from scratch (instead of editing one long-lived spec) also makes
// a failed spawn or an undo trivial: just compile a different robot list.

export const BASE_XML = `
<mujoco model="world">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <frame name="spawn_root"/>
  </worldbody>
</mujoco>`;

// robots: [{path, prefix, pos}] where `path` is the robot's MJCF in the
// MuJoCo FS (with its includes and assets already there, see vfs.js).
// Returns the compiled MjModel; throws on parse/compile errors.
export function compileWorld(mujoco, robots) {
  const world = mujoco.parseXMLString(BASE_XML);
  try {
    const worldbody = mujoco.mjs_findBody(world, 'world');
    // The bindings do not accept null for mjs_addFrame's parent frame, so new
    // frames hang off an identity frame declared in BASE_XML.
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
        const frame = mujoco.mjs_addFrame(worldbody, root);
        frame.pos.set(r.pos);
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
