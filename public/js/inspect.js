// Simulate-style inspection of the physics world (like MuJoCo's simulate /
// MuJoCo Live): double-click selects a body, Shift+drag pulls it with a spring
// (Perturb), and the Stage panel (StagePanel) lists what is spawned, each with
// its joint positions and actuator controls (joints can be set while paused,
// controls at any time) and buttons to reset or remove it.

const sub = (a, b) => a.map((v, i) => v - b[i]);
const DEG = 180 / Math.PI;

// Unit quaternion (w, x, y, z) <-> roll, pitch, yaw (rad): R = Rz(yaw) Ry(pitch) Rx(roll).
function quatToRpy([w, x, y, z]) {
  return [
    Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
    Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))),
    Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
  ];
}
function rpyToQuat([roll, pitch, yaw]) {
  const [cr, sr] = [Math.cos(roll / 2), Math.sin(roll / 2)], [cp, sp] = [Math.cos(pitch / 2), Math.sin(pitch / 2)];
  const [cy, sy] = [Math.cos(yaw / 2), Math.sin(yaw / 2)];
  return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy];
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// Pulls the selected body's selection point towards a target that follows the
// pointer in the plane through the grab point facing the camera. The spring is
// scaled by the body's subtree mass and critically damped, like simulate's
// mjv_applyPerturbForce, and applied through xfrc_applied before every step.
export class Perturb {
  static OMEGA = 15;      // spring natural frequency (rad/s)
  static MAX_ACC = 100;   // force cap, as an acceleration of the subtree mass (m/s²)

  constructor(app) {
    this.app = app;
    this.clear();
  }

  // Forgets the selection (a rebuilt world may number its bodies differently).
  clear() {
    this.body = -1;
    this.local = null;   // selection point in the body frame
    this.target = null;  // world point being dragged towards, while dragging
    this.normal = null;  // drag plane normal
    this.prev = null;    // [point, time] at the last step, for its velocity
  }

  get name() {
    const { mujoco, model } = this.app;
    return this.body > 0 ? mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY.value, this.body) || `body ${this.body}` : '';
  }

  select(hit) {
    this.end();
    this.clear();
    if (!hit || hit.body <= 0) return;
    const { data } = this.app;
    const b = hit.body;
    const R = data.xmat.subarray(9 * b, 9 * b + 9), d = sub(hit.point, data.xpos.subarray(3 * b, 3 * b + 3));
    this.body = b;
    this.local = [0, 1, 2].map((c) => R[c] * d[0] + R[3 + c] * d[1] + R[6 + c] * d[2]);  // Rᵀ d
  }

  // The selection point in world coordinates, or null.
  point() {
    if (this.body <= 0) return null;
    const { data } = this.app, b = this.body, l = this.local;
    const R = data.xmat.subarray(9 * b, 9 * b + 9), p = data.xpos.subarray(3 * b, 3 * b + 3);
    return [0, 1, 2].map((r) => p[r] + R[3 * r] * l[0] + R[3 * r + 1] * l[1] + R[3 * r + 2] * l[2]);
  }

  // Starts dragging (ray = App.pointerRay); false if nothing is selected.
  begin(ray) {
    const p = this.point();
    if (!p) return false;
    this.normal = ray.forward;
    this.target = p;
    this.move(ray);
    return true;
  }

  move({ eye, dir }) {
    if (!this.target) return;
    const den = dot(dir, this.normal);
    if (Math.abs(den) < 1e-6) return;
    const s = dot(sub(this.target, eye), this.normal) / den;
    if (s > 0) this.target = eye.map((v, i) => v + s * dir[i]);
    this.sync();
  }

  // Network mode: the server applies the spring (net.js).
  sync() {
    this.app.net?.send('perturb', this.target ? { body: this.body, local: this.local, target: this.target } : { body: -1 });
  }

  end() {
    if (!this.target) return;
    this.target = null;
    this.prev = null;
    this.sync();
    const { data } = this.app;
    if (this.body > 0 && this.body < this.app.model.nbody) data.xfrc_applied.fill(0, 6 * this.body, 6 * this.body + 6);
  }

  // Before each mj_step: the spring force (and its torque about the body's
  // center of mass) on the selected body.
  apply() {
    if (!this.target) return;
    const { model, data } = this.app, b = this.body;
    const p = this.point();
    const v = this.prev && data.time > this.prev[1]
      ? sub(p, this.prev[0]).map((k) => k / (data.time - this.prev[1])) : [0, 0, 0];
    this.prev = [p, data.time];
    const m = model.body_subtreemass[b], w = Perturb.OMEGA;
    let f = [0, 1, 2].map((i) => m * (w * w * (this.target[i] - p[i]) - 2 * w * v[i]));
    const n = Math.hypot(...f), cap = m * Perturb.MAX_ACC;
    if (n > cap) f = f.map((k) => (k * cap) / n);
    const t = cross(sub(p, data.xipos.subarray(3 * b, 3 * b + 3)), f);
    data.xfrc_applied.set([...f, ...t], 6 * b);
  }
}

// Side panel: per spawned robot, Reset (back to its spawn pose) and Remove
// buttons, a slider per hinge/slide joint (qpos) and per actuator (ctrl).
// Rebuilt for every world, refreshed ~10 times a second for the expanded
// robots. `onRemove(robot)` removes a robot (the page rebuilds the world);
// the Save / Import / Forget buttons call `onSave()` / `onImport()` /
// `onForget()` (the saved stages' list is stage.js's StageList). Each robot's
// Pose section places it: on a free joint at once (App.placeFree), a fixed
// base through `onMove(robot, pos, yaw)` (a rebuild).
export class StagePanel {
  constructor(app, el, { onRemove = () => {}, onSave = () => {}, onImport = () => {}, onForget = () => {}, onMove = async () => false } = {}) {
    this.app = app;
    this.el = el;
    this.onRemove = onRemove;
    this.onMove = onMove;
    this.poses = [];        // {inputs, read(), sub} per robot's Pose section
    el.querySelector('.save').onclick = onSave;
    el.querySelector('.import').onclick = onImport;
    el.querySelector('.forget').onclick = onForget;
    this.savedEl = el.querySelector('.saved');
    this.empty = el.querySelector('.empty');
    this.open = new Set();  // robot prefixes whose section is expanded
    this.closed = new Set();  // "<prefix>Joints"/"<prefix>Controls" subsections collapsed
    this.rows = [];         // {input, out, read(), write(v), joint}
    this.lastUpdate = 0;
  }

  build() {
    const { mujoco, model, robots } = this.app;
    const T = mujoco.mjtObj;
    const keep = (set, d, on) => (on ? set.add(d.dataset.key) : set.delete(d.dataset.key));
    this.el.querySelectorAll('details details').forEach((d) => keep(this.closed, d, !d.open));
    this.el.querySelectorAll(':scope > details').forEach((d) => { keep(this.open, d, d.open); d.remove(); });
    this.rows = [];
    this.poses = [];
    const robotOfBody = (b) => robots.find((r) => b >= r.bodyOffset && b < r.bodyOffset + r.visual.inertia.nbody - 1);
    const groups = new Map(robots.map((r) => [r, { joints: [], ctrls: [] }]));

    for (let j = 0; j < model.njnt; j++) {
      const type = model.jnt_type[j];
      if (type !== 2 && type !== 3) continue;  // slide, hinge
      const r = robotOfBody(model.jnt_bodyid[j]);
      if (!r) continue;
      const adr = model.jnt_qposadr[j], dof = model.jnt_dofadr[j];
      const range = model.jnt_limited[j] ? [model.jnt_range[2 * j], model.jnt_range[2 * j + 1]]
                                         : type === 3 ? [-Math.PI, Math.PI] : [-1, 1];
      groups.get(r).joints.push({
        name: mujoco.mj_id2name(model, T.mjOBJ_JOINT.value, j) || `joint ${j}`, range, joint: true,
        read: () => this.app.data.qpos[adr],
        write: (v) => {
          if (this.app.net) { this.app.net.send('qpos', { adr, dof, v }); return; }
          const { mujoco: mj, model: m, data } = this.app;
          data.qpos[adr] = v;
          data.qvel[dof] = 0;
          mj.mj_forward(m, data);
          this.app.syncTransforms();
        },
      });
    }
    for (let a = 0; a < model.nu; a++) {
      const name = mujoco.mj_id2name(model, T.mjOBJ_ACTUATOR.value, a) || `actuator ${a}`;
      let r = robots.find((x) => name.startsWith(x.prefix));
      // Unnamed: the robot of the joint it drives (trntype 0 = joint).
      if (!r && model.actuator_trntype[a] === 0) r = robotOfBody(model.jnt_bodyid[model.actuator_trnid[2 * a]]);
      if (!r) continue;
      const range = model.actuator_ctrllimited[a] ? [model.actuator_ctrlrange[2 * a], model.actuator_ctrlrange[2 * a + 1]] : [-1, 1];
      groups.get(r).ctrls.push({
        name, range, joint: false,
        read: () => this.app.data.ctrl[a],
        write: (v) => { if (this.app.net) this.app.net.send('ctrl', { a, v }); else this.app.data.ctrl[a] = v; },
      });
    }

    for (const [r, { joints, ctrls }] of groups) {
      const det = document.createElement('details');
      det.dataset.key = r.prefix;
      det.open = this.open.has(r.prefix);
      det.innerHTML = '<summary><span class="title"></span><span class="actions">'
        + '<button class="reset" title="Back to its spawn pose">Reset</button>'
        + '<button class="remove" title="Remove from the scene">✕</button></span></summary>';
      det.querySelector('.title').textContent = `${r.prefix.slice(0, -1)} · ${r.label}`;
      // Buttons inside <summary> would also fold the section.
      const action = (sel, fn) => {
        det.querySelector(sel).onclick = (e) => { e.preventDefault(); e.stopPropagation(); fn(); };
      };
      action('.reset', () => { this.app.resetRobot(r); this.lastUpdate = 0; });
      action('.remove', () => this.onRemove(r));
      if (!joints.length && !ctrls.length) {
        const none = document.createElement('div');
        none.className = 'hint';
        none.textContent = 'No joints or actuators to set.';
        det.append(none);
      }
      det.ontoggle = () => { this.lastUpdate = 0; };
      det.append(this.poseSection(r));
      for (const [title, list] of [['Joints', joints], ['Controls', ctrls]]) {
        if (!list.length) continue;
        const sub = document.createElement('details');
        sub.dataset.key = r.prefix + title;
        sub.open = !this.closed.has(sub.dataset.key);
        sub.innerHTML = `<summary>${title} <span class="count">${list.length}</span></summary>`;
        sub.ontoggle = () => { this.lastUpdate = 0; };
        for (const item of list) sub.append(this.row(r, item));
        det.append(sub);
      }
      this.el.append(det);
    }
    if (this.empty) this.empty.hidden = robots.length > 0;
    this.lastUpdate = 0;
  }

  // Position (m) and roll / pitch / yaw (deg, about x, y, z: yaw first) of
  // the robot's root: its free joint, live; or its attach frame (heading
  // only), which rebuilds the world.
  poseSection(r) {
    const sub = document.createElement('details');
    sub.dataset.key = r.prefix + 'Pose';
    sub.open = !this.closed.has(sub.dataset.key);
    const free = this.app.rootFreeJoint(r) >= 0;
    sub.innerHTML = `<summary>Pose <span class="count">${free ? 'free joint · live' : 'fixed base · rebuilds the world'}</span></summary>`;
    sub.ontoggle = () => { this.lastUpdate = 0; };
    const read = () => {
      const j = this.app.rootFreeJoint(r);
      if (j < 0) return [...r.pos, 0, 0, (r.yaw ?? 0) * DEG];
      const q = this.app.data.qpos, a = this.app.model.jnt_qposadr[j];
      return [q[a], q[a + 1], q[a + 2], ...quatToRpy(q.subarray(a + 3, a + 7)).map((v) => v * DEG)];
    };
    const inputs = [];
    const apply = async () => {
      const v = inputs.map((inp) => Number(inp.value));
      if (v.some((x) => !Number.isFinite(x))) return;
      const pos = v.slice(0, 3);
      if (this.app.rootFreeJoint(r) >= 0) this.app.placeFree(r, pos, rpyToQuat(v.slice(3).map((x) => x / DEG)));
      else if (!(await this.onMove(r, pos, v[5] / DEG))) inputs.forEach((inp, i) => { inp.value = +read()[i].toFixed(i < 3 ? 3 : 1); });
    };
    for (const [label, names, step] of [['Position', ['x', 'y', 'z'], 0.01], ['Rotation', ['roll', 'pitch', 'yaw'], 1]]) {
      const row = document.createElement('div');
      row.className = 'pose';
      row.append(Object.assign(document.createElement('span'), { textContent: label }));
      for (const n of names) {
        const inp = Object.assign(document.createElement('input'), { type: 'number', step, title: `${n} (${step === 1 ? 'deg' : 'm'})` });
        inp.disabled = !free && (n === 'roll' || n === 'pitch');  // an attach frame takes a heading only
        inp.onchange = apply;
        inputs.push(inp);
        row.append(inp);
      }
      sub.append(row);
    }
    const v = read();
    inputs.forEach((inp, i) => { inp.value = +v[i].toFixed(i < 3 ? 3 : 1); });
    this.poses.push({ inputs, read, sub });
    return sub;
  }

  // Shows the name of the stage loaded at the next page load (null: none;
  // undefined: network mode, where the server keeps its world).
  showStart(name) {
    const forget = this.el.querySelector('.forget');
    forget.hidden = name === undefined;
    forget.disabled = !name;
    this.savedEl.textContent = name === undefined ? '' : name ? `at start: ${name}` : 'empty at start';
    this.savedEl.title = name ? 'Loaded on the next page load (the stage last saved or loaded)' : '';
  }

  row(robot, item) {
    const div = document.createElement('label');
    div.className = 'row';
    const name = document.createElement('span');
    name.textContent = item.name.startsWith(robot.prefix) ? item.name.slice(robot.prefix.length) : item.name;
    name.title = item.name;
    const input = document.createElement('input');
    input.type = 'range';
    [input.min, input.max] = item.range;
    input.step = (item.range[1] - item.range[0]) / 1000 || 0.001;
    const out = document.createElement('output');
    input.oninput = () => {
      item.write(Number(input.value));
      out.textContent = Number(input.value).toFixed(3);
    };
    div.append(name, input, out);
    this.rows.push({ ...item, input, out });
    return div;
  }

  // The selected body's mass, and with what hangs from it if that is more.
  massText() {
    const { model } = this.app, b = this.app.perturb.body;
    const kg = (v) => `${v < 10 ? v.toPrecision(3) : v.toFixed(1)} kg`;
    const own = model.body_mass?.[b], all = model.body_subtreemass?.[b];
    if (own == null) return '';
    return all != null && all > own * 1.001 + 1e-6 ? `${kg(own)} (${kg(all)} with its children)` : kg(own);
  }

  // Called every frame; refreshes the expanded robots' rows at ~10 Hz.
  update(now) {
    if (this.el.hidden || now - this.lastUpdate < 100 || !this.app.data) return;
    this.lastUpdate = now;
    const sel = this.el.querySelector('#selected');
    if (sel) sel.textContent = this.app.perturb.name ? `${this.app.perturb.name} · ${this.massText()}` : 'none (double-click a body)';
    const paused = this.app.paused;
    for (const row of this.rows) {
      const sub = row.input.closest('details');
      if (!sub.open || !sub.parentElement.closest('details').open) continue;  // collapsed
      if (row.joint) {
        row.input.disabled = !paused;
        row.input.title = paused ? '' : 'pause to set joint positions';
      }
      if (document.activeElement === row.input && row.input.matches(':active')) continue;  // being dragged
      const v = row.read();
      row.input.value = v;
      row.out.textContent = v.toFixed(3);
    }
    for (const p of this.poses) {
      if (!p.sub.open || !p.sub.parentElement.open || p.inputs.includes(document.activeElement)) continue;
      const v = p.read();
      p.inputs.forEach((inp, i) => { inp.value = +v[i].toFixed(i < 3 ? 3 : 1); });
    }
  }
}
