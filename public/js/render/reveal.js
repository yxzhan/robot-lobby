// Spawn and reset effect: a robot materialises from its feet up, after the
// Microduck simulator's (its fx-wireframe.js). A scan line rises through it
// in world Z, an orange hologram of its triangles' edges below it; a solid
// line trails 0.3 s behind, the robot's own surfaces below that with a hot
// edge at the line. ~0.9 s for a Microduck, longer for a taller robot.
//
// While a robot reveals, its parts get their own material instances
// (lit_reveal / lit_textured_reveal, masked: cut above the solid line,
// shadow included) in place of their usual ones, which every spawned copy of
// the robot file shares, and a wireframe entity each (unlit_reveal_wire,
// additive), a child of the part's entity so it follows it. Its transparent
// parts are hidden until the end. Afterwards everything is put back as it
// was. The lines ride with the robot's root body, so one that drops or walks
// while it reveals keeps its scan.
//
//   const reveal = new Reveal(engine, scene, mats, makeEntity, white, sampler)
//   reveal.start(robot, layer)          (again: starts over)
//   reveal.update(now, robots, data)    each frame, before rendering
//   reveal.forget(entities)             before a robot's entities go

const SCAN_S = 0.6;        // scan line: feet -> top
const SOLID_DELAY_S = 0.3; // solid line starts this long after the scan
const SOLID_S = 0.6;       // solid line: feet -> top
const TOTAL_S = SOLID_DELAY_S + SOLID_S;
const BOTTOM_PAD = 0.025;  // m below the robot's lowest point
// Band widths and the timeline are for a Microduck (~0.35 m tall); a taller
// robot scales the widths with its height and the time with its square root.
const REF_SPAN = 0.35;

const clamp01 = (x) => Math.min(Math.max(x, 0), 1);
// Ease-out: full speed the moment it starts, landing softly.
const ease = (x) => 1 - (1 - x) * (1 - x);
const hash = (x) => {
  const s = Math.sin(x * 127.1) * 43758.5453;
  return s - Math.floor(s);
};
const flickerAt = (t) => {
  let f = 0.82 + 0.18 * hash(Math.floor(t * 60) + 0.5);
  if (hash(Math.floor(t * 24) + 7.7) < 0.12) f *= 0.5;  // dropouts
  return f;
};

export class Reveal {
  // mats: {solid, solidTextured, wire} (Filament materials); makeEntity as
  // main.js's (engine, geo, mi, castShadows, layer, primitive, receiveShadows);
  // white and sampler what a textured part is bound with until its image is in.
  constructor(engine, scene, mats, makeEntity, white, sampler) {
    Object.assign(this, { engine, scene, mats, makeEntity, white, sampler });
    this.active = new Map();  // robot -> its reveal
  }

  // Starts robot's reveal (its entities as they are now) on `layer`, the
  // one its parts are drawn on.
  start(robot, layer) {
    const old = this.active.get(robot);
    if (old?.entities === robot.entities) {
      old.t0 = null;  // starts over, on the next update
      return;
    }
    if (old) this.end(robot, old);
    const { engine } = this, F = Filament, P = F.RenderableManager$PrimitiveType;
    const rcm = engine.getRenderableManager(), tcm = engine.getTransformManager();
    const wireMi = this.mats.wire.createInstance();
    const st = { entities: robot.entities, t0: null, wireMi, mis: [], wires: [], hidden: [], layer };
    for (const [i, p] of robot.visual.parts.entries()) {
      const e = robot.entities[i], inst = rcm.getInstance(e);
      if (p.transparent) {
        rcm.setLayerMask(inst, 0xff, 0);
        st.hidden.push(e);
      } else {
        const { rgba, roughness, metallic, tex, doubleSided } = p.look;
        const mi = (tex ? this.mats.solidTextured : this.mats.solid).createInstance();
        mi.setDoubleSided(doubleSided);
        mi.setColor4Parameter('baseColor', F.RgbaType.sRGB, rgba);
        mi.setFloatParameter('roughness', roughness);
        mi.setFloatParameter('metallic', metallic);
        if (tex) mi.setTextureParameter('baseColorMap', tex.tex ?? this.white, this.sampler);
        mi.setFloatParameter('solidZ', -1e3);  // nothing until the first update
        mi.setFloatParameter('spanK', 1);
        rcm.setMaterialInstanceAt(inst, 0, mi);
        st.mis.push(mi);
      }
      inst.delete();
      // Triangle edges, built on first use.
      p.wireIb ??= edgeIndices(engine, p.tri);
      if (!p.wireIb) continue;
      const w = this.makeEntity(engine, { vb: p.vb, ib: p.wireIb, box: p.box }, wireMi, false, layer, P.LINES, false);
      const wi = tcm.getInstance(w), pi = tcm.getInstance(e);
      tcm.setParent(wi, pi);
      wi.delete();
      pi.delete();
      this.scene.addEntity(w);
      st.wires.push(w);
    }
    wireMi.setFloatParameter('scanZ', -1e3);
    wireMi.setFloatParameter('solidZ', -1e3);
    this.active.set(robot, st);
  }

  // Moves every reveal on to `now` (ms); `data` the world's MjData. One
  // whose robot has gone or been rebuilt is dropped.
  update(now, robots, data) {
    for (const [robot, st] of this.active) {
      if (!robots.includes(robot) || robot.entities !== st.entities) {
        this.end(robot, st, false);
        continue;
      }
      const root = () => (robot.visual.inertia.nbody > 1 ? data.xpos[3 * robot.bodyOffset + 2] : 0);
      if (st.t0 === null) {
        // The robot's height range from its parts as drawn, the lines from
        // there relative to its root body.
        const [lo, hi] = this.zRange(st.entities, robot.visual.parts);
        if (!(hi > lo)) continue;  // not placed yet
        st.t0 = now;
        st.root0 = root();
        st.lo = Math.min(lo, 0) - BOTTOM_PAD;
        st.span = Math.max(hi - st.lo, 0.04) * 1.06;
        st.k = Math.max(1, st.span / REF_SPAN);
        st.durK = Math.min(Math.sqrt(st.k), 2.5);
        for (const mi of st.mis) mi.setFloatParameter('spanK', st.k);
        st.wireMi.setFloatParameter('spanK', st.k);
      }
      const t = (now - st.t0) / 1000;
      if (t >= TOTAL_S * st.durK) {
        this.end(robot, st);
        continue;
      }
      const scanP = ease(clamp01(t / (SCAN_S * st.durK)));
      const solidP = ease(clamp01((t - SOLID_DELAY_S * st.durK) / (SOLID_S * st.durK)));
      const jitter = scanP > 0 && scanP < 1 ? (hash(t * 41.3) - 0.5) * 0.02 * st.span : 0;
      const base = st.lo + root() - st.root0;
      const solidZ = base + st.span * solidP;
      for (const mi of st.mis) mi.setFloatParameter('solidZ', solidZ);
      st.wireMi.setFloatParameter('scanZ', base + st.span * scanP + jitter);
      st.wireMi.setFloatParameter('solidZ', solidZ);
      st.wireMi.setFloatParameter('time', t);
      st.wireMi.setFloatParameter('flicker', flickerAt(t));
    }
  }

  // Drops the reveal drawn with `entities` (a robot's, about to be removed).
  forget(entities) {
    for (const [robot, st] of this.active) if (st.entities === entities) this.end(robot, st, false);
  }

  // Lowest and highest world Z of the parts' bounding boxes, as drawn.
  zRange(entities, parts) {
    const tcm = this.engine.getTransformManager();
    let lo = Infinity, hi = -Infinity;
    for (const [i, p] of parts.entries()) {
      const inst = tcm.getInstance(entities[i]);
      const m = tcm.getWorldTransform(inst);  // column-major
      inst.delete();
      const { center: c, halfExtent: h } = p.box;
      const z = m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14];
      const r = Math.abs(m[2]) * h[0] + Math.abs(m[6]) * h[1] + Math.abs(m[10]) * h[2];
      lo = Math.min(lo, z - r);
      hi = Math.max(hi, z + r);
    }
    return [lo, hi];
  }

  // Puts robot's parts back as they were (unless `restore` is false: they
  // are gone) and frees the reveal's own instances and entities.
  end(robot, st, restore = true) {
    const { engine } = this, rcm = engine.getRenderableManager();
    if (restore) {
      for (const [i, p] of robot.visual.parts.entries()) {
        const inst = rcm.getInstance(st.entities[i]);
        if (p.transparent) rcm.setLayerMask(inst, 0xff, st.layer);
        else rcm.setMaterialInstanceAt(inst, 0, p.mi);
        inst.delete();
      }
    }
    for (const w of st.wires) {
      this.scene.remove(w);
      engine.destroyEntity(w);
    }
    for (const mi of st.mis) engine.destroyMaterialInstance(mi);
    engine.destroyMaterialInstance(st.wireMi);
    this.active.delete(robot);
  }
}

// An index buffer of the edges of triangles `tri` (each one's three, shared
// ones twice), drawn as LINES over the triangles' own vertex buffer; null
// without triangles.
function edgeIndices(engine, tri) {
  if (!tri?.length) return null;
  const idx = new Uint32Array(2 * tri.length);
  for (let i = 0; i < tri.length; i += 3) {
    const [a, b, c] = [tri[i], tri[i + 1], tri[i + 2]];
    idx.set([a, b, b, c, c, a], 2 * i);
  }
  const ib = Filament.IndexBuffer.Builder()
    .indexCount(idx.length).bufferType(Filament.IndexBuffer$IndexType.UINT)
    .build(engine);
  ib.setBuffer(engine, idx);
  return ib;
}
