// Cameras of the spawned robots (environments and objects too), rendered in
// this browser, in local and network mode alike.
//
// Each switched-on camera has a Filament View of its own, rendered after the
// window's frame into a framebuffer of its own: App.fbTarget makes Filament
// bind it where it binds the default framebuffer (the trick WebXR uses, see
// main.js xrGl), so the main canvas is untouched. Its pixels are read back
// asynchronously (into a pixel buffer, collected a frame or so later once
// its fence has passed: a plain readPixels would wait for the GPU to finish
// everything queued, the window's frame too, every time) for its
// picture-in-picture window and for ROS (ros.js, through onFrame). A
// camera renders at its own rate, at most two per window frame.
//
// A camera, like an IK chain's TCP, rides a link of its robot: {name, link
// (a body name, unprefixed; '' the robot's own frame), follow, pos, rpy,
// fovy, width, height, hz}. `pos` is an offset in the link's frame, `rpy` a
// roll / pitch / yaw (degrees, about the link's x, y, z, applied in that
// order) from looking along the link's +x with +z up (so pitch > 0 looks
// down); follow 'link' turns with the link, 'position' only follows where it
// is, keeping its turn in the robot's frame (MJCF track / trackcom cameras).
// The view looks along the camera's -z with y up, as MuJoCo's and Filament's
// cameras do. Poses come from the world's body poses (data.xpos / xmat:
// MuJoCo in local mode, net.js forwardKinematics in network mode).
//
// A robot file's cameras (as IK's chains): saved in this browser (CAM_STORE,
// the panel's Save), else its built-in setup in
// public/json/camera_defaults.json (CAMERA_DEFAULTS, keyed by the file under
// mjcf_assets/robots/ like ik_defaults.json, other models by theirs under
// mjcf_assets/; the panel's Export copies a setup for it), else the MJCF
// <camera>s of its model (autoCameras).

const CAM_STORE = 'robot-lobby/cameras/';
export const CAMERA_DEFAULTS = await fetch(new URL('../json/camera_defaults.json', import.meta.url)).then((r) => r.json()).catch(() => ({}));
// A model file's key in CAMERA_DEFAULTS.
export const cameraKey = (rel) => rel.replace(/^(.*\/)?mjcf_assets\/(robots\/)?/, '');
const SIZE = 256, HZ = 10, FOVY = 60;  // new cameras'
const MAX_SIZE = 2048;
const RATES = [5, 10, 15, 30];
const NEAR = 0.01, FAR = 100;
const PER_FRAME = 2;  // cameras rendered per window frame, at most
const MARK = { depth: 0.12, radius: 0.0025, frame: [1, 0.75, 0.15], up: [0.3, 0.9, 0.4] };  // the pose marker

// The cameras of a compiled model (MuJoCo's MjModel, or the server's render
// data, render_asset.py, which has them as a list already).
export function modelCameras(mujoco, model) {
  if (model.cameras) return model.cameras;
  const out = [];
  for (let c = 0; c < (model.ncam ?? 0); c++) {
    const slice = (a, n) => Array.from(a.subarray(n * c, n * c + n));
    out.push({
      name: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA.value, c) || `camera${c}`,
      body: model.cam_bodyid[c], mode: model.cam_mode[c], fovy: model.cam_fovy[c],
      pos: slice(model.cam_pos, 3), quat: slice(model.cam_quat, 4),
      pos0: slice(model.cam_pos0, 3), mat0: slice(model.cam_mat0, 9),
      resolution: model.cam_resolution ? slice(model.cam_resolution, 2) : [1, 1],
    });
  }
  return out;
}

const quatMat = ([w, x, y, z]) => [
  1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
const mulMM = (A, B) => [0, 3, 6].flatMap((r) => [0, 1, 2].map((c) => A[r] * B[c] + A[r + 1] * B[3 + c] + A[r + 2] * B[6 + c]));
const mulMV = (A, v) => [0, 3, 6].map((r) => A[r] * v[0] + A[r + 1] * v[1] + A[r + 2] * v[2]);
const transpose = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
const yawMat = (t) => [Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t), 0, 0, 0, 1];
const DEG = Math.PI / 180;
// Roll, pitch, yaw (degrees) about x, y, z: Rz · Ry · Rx.
function rpyMat([r, p, y]) {
  const [a, b, c] = [r, p, y].map((d) => d * DEG);
  const Rx = [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
  const Ry = [Math.cos(b), 0, Math.sin(b), 0, 1, 0, -Math.sin(b), 0, Math.cos(b)];
  return mulMM(yawMat(c), mulMM(Ry, Rx));
}
// ... and back (rounded to 0.01°).
function matRpy(R) {
  const round = (v) => Math.round(v / DEG * 100) / 100 + 0;  // (+0: no -0)
  return [round(Math.atan2(R[7], R[8])), round(-Math.asin(Math.max(-1, Math.min(1, R[6])))), round(Math.atan2(R[3], R[0]))];
}
// Camera axes in a link frame for looking along its +x with +z up: the
// camera's x (right) = -y, y (up) = +z, z (back) = -x.
const LOOK_X = [0, 0, -1, -1, 0, 0, 0, 1, 0];
// The transform putting a unit rod (radius 1, z from -0.5 to 0.5) from p to q, radius w (as xr.js).
function rodMatrix(p, q, w) {
  const d = p.map((v, i) => q[i] - v), len = Math.hypot(...d);
  if (len < 1e-6) return null;
  const z = d.map((v) => v / len), a = Math.abs(z[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let x = [a[1] * z[2] - a[2] * z[1], a[2] * z[0] - a[0] * z[2], a[0] * z[1] - a[1] * z[0]];
  const xl = Math.hypot(...x);
  x = x.map((v) => v / xl);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return [...x.map((v) => v * w), 0, ...y.map((v) => v * w), 0, ...d, 0, ...p.map((v, i) => (v + q[i]) / 2), 1];
}
const HIDDEN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

export class Cameras {
  constructor(app) {
    this.app = app;
    this.configs = new Map();  // robot file -> {cameras, source ('saved' | 'built-in' | 'automatic'), edited}
    this.on = new Map();   // key -> switched-on camera (see switchOn)
    this.marked = new Set();  // keys of the cameras whose pose marker shows
    this.parts = [];       // marker rods, {e, mi, color} (App.xrPart), reused each frame
    this.look = null;      // key of the camera the window looks through, or null
    this.frameDt = 1000 / 60;  // ms between window frames (render)
    this.pips = el('div', { id: 'pips' });
    document.body.append(this.pips);
    this.onChange = () => {};  // the panel re-renders
    this.onFrame = () => {};   // a camera's new frame has arrived (ROS publishes it)
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && this.look) this.lookThrough(null); });
  }

  // A robot file's setup (see the top); null while the robot's bodies are
  // not in the world yet (its MJCF cameras need their names).
  config(robot) {
    const rel = robot.visual.rel;
    if (!this.configs.has(rel)) {
      let saved = null;
      try { saved = JSON.parse(localStorage.getItem(CAM_STORE + rel) ?? 'null'); } catch { /* no storage */ }
      if (!Array.isArray(saved?.cameras)) saved = null;  // (an older format)
      const builtIn = CAMERA_DEFAULTS[cameraKey(rel)];
      if (!saved && !builtIn && !this.bodies(robot).length) return null;
      const cameras = structuredClone(saved?.cameras ?? builtIn?.cameras ?? this.autoCameras(robot));
      this.configs.set(rel, { cameras: cameras.map((c) => this.complete(c)), source: saved ? 'saved' : builtIn ? 'built-in' : 'automatic', edited: false });
    }
    return this.configs.get(rel);
  }

  // Missing fields filled in.
  complete(c) {
    return { name: 'camera', link: '', follow: 'link', pos: [0, 0, 0], rpy: [0, 0, 0], fovy: FOVY, width: SIZE, height: SIZE, hz: HZ, ...c };
  }

  // The MJCF <camera>s of a robot's model, as cameras (targetbody cameras
  // as fixed ones).
  autoCameras(robot) {
    const names = this.bodies(robot).map(([, n]) => n);
    return (robot.visual.cameras ?? []).map((cam) => {
      const track = cam.mode === 1 || cam.mode === 2;
      const R = track ? cam.mat0 : quatMat(cam.quat);
      const [w, h] = cam.resolution ?? [1, 1];
      return this.complete({
        name: cam.name, link: cam.body > 0 ? names[cam.body - 1] ?? `#${cam.body}` : '', follow: track ? 'position' : 'link',
        pos: (track ? cam.pos0 : cam.pos).map((v) => +v.toFixed(5)), rpy: matRpy(mulMM(R, transpose(LOOK_X))),
        fovy: +cam.fovy.toFixed(2), ...(w > 1 && h > 1 ? { width: w, height: h } : {}),
      });
    });
  }

  // Saves a robot file's setup in this browser.
  save(robot) {
    const cfg = this.config(robot);
    if (!cfg) return;
    try { localStorage.setItem(CAM_STORE + robot.visual.rel, JSON.stringify({ cameras: cfg.cameras })); } catch { /* no storage */ }
    Object.assign(cfg, { source: 'saved', edited: false });
    this.onChange();
  }

  // Back to the built-in setup, else the model's cameras (this browser's
  // forgotten); every robot of the file follows.
  restoreDefaults(robot) {
    const rel = robot.visual.rel;
    try { localStorage.removeItem(CAM_STORE + rel); } catch { /* no storage */ }
    this.configs.delete(rel);
    this.build();
  }

  hasDefaults(robot) { return !!CAMERA_DEFAULTS[cameraKey(robot.visual.rel)]; }

  // The setup as JSON for camera_defaults.json: {key: {cameras}}, number
  // lists and flat objects on one line each (as ik_defaults.json has them).
  exportJson(robot) {
    return JSON.stringify({ [cameraKey(robot.visual.rel)]: { cameras: this.config(robot)?.cameras ?? [] } }, null, 2)
      .replace(/\[\s+([^[\]{}]*?)\s+\]/g, (_, x) => `[${x.split(/,\s*/).join(', ')}]`)
      .replace(/\{\s+([^{}]*?)\s+\}/g, (_, x) => `{ ${x.split(/,\n\s*/).join(', ')} }`);
  }

  // Every camera of the world: [{key, robot, cam}].
  list() {
    return (this.app.robots ?? []).flatMap((robot) => (this.config(robot)?.cameras ?? []).map((cam) => ({ key: robot.prefix + cam.name, robot, cam })));
  }

  find(key) { return this.list().find((c) => c.key === key) ?? null; }

  // A new camera on `link` of the robot; returns its key.
  add(robot, link) {
    const cfg = this.config(robot);
    if (!cfg) return null;
    const names = new Set(cfg.cameras.map((c) => c.name));
    let n = 1;
    while (names.has(`camera${n}`)) n++;
    cfg.cameras.push(this.complete({ name: `camera${n}`, link }));
    cfg.edited = true;
    this.build();
    return robot.prefix + `camera${n}`;
  }

  // Changes a camera (any of its fields); a new name must be free.
  edit(key, change) {
    const found = this.find(key);
    if (!found) return;
    const cfg = this.config(found.robot), i = cfg.cameras.indexOf(found.cam);
    if ('name' in change) {
      const name = String(change.name).trim();
      if (!name || (name !== found.cam.name && cfg.cameras.some((c) => c.name === name))) delete change.name;
      else change.name = name;
    }
    const cam = cfg.cameras[i] = { ...found.cam, ...change };
    cfg.edited = true;
    const next = found.robot.prefix + cam.name;
    for (const r of this.app.robots ?? []) {  // renamed: windows and markers go with it (every robot of the file)
      if (r.visual.rel !== found.robot.visual.rel || next === key) continue;
      const [from, to] = [r.prefix + found.cam.name, r.prefix + cam.name];
      const c = this.on.get(from);
      if (c) {
        this.on.delete(from);
        c.key = to;
        this.on.set(to, c);
        c.title.textContent = `${r.prefix.slice(0, -1)} · ${cam.name}`;
      }
      if (this.marked.delete(from)) this.marked.add(to);
      if (this.look === from) this.look = to;
    }
    this.build();
  }

  remove(key) {
    const found = this.find(key);
    if (!found) return;
    for (const r of this.app.robots ?? []) {
      if (r.visual.rel !== found.robot.visual.rel) continue;
      this.switchOff(r.prefix + found.cam.name);
      this.marked.delete(r.prefix + found.cam.name);
    }
    const cfg = this.config(found.robot);
    cfg.cameras = cfg.cameras.filter((c) => c !== found.cam);
    cfg.edited = true;
    this.build();
  }

  // A robot's bodies as [world body id, name (unprefixed)].
  bodies(robot) {
    const { app } = this, mj = app.mujoco, out = [];
    if (!app.model || !mj || robot.bodyOffset == null) return out;
    for (let b = robot.bodyOffset; b < robot.bodyOffset + robot.visual.inertia.nbody - 1; b++) {
      const n = mj.mj_id2name(app.model, mj.mjtObj.mjOBJ_BODY.value, b) ?? '';
      out.push([b, n.startsWith(robot.prefix) ? n.slice(robot.prefix.length) : n || `#${b - robot.bodyOffset + 1}`]);
    }
    return out;
  }

  // The world body of a camera's link, or -1 (the robot's own frame).
  linkBody(robot, link) {
    if (!link) return -1;
    const m = /^#(\d+)$/.exec(link);
    if (m) return robot.bodyOffset + Number(m[1]) - 1;
    return this.bodies(robot).find(([, n]) => n === link)?.[0] ?? -1;  // (network mode's names have no name2id)
  }

  // After a new world or a changed setup: switched-on cameras follow their
  // robot and setup (by key), the others go.
  build() {
    for (const [key, c] of this.on) {
      const found = this.find(key);
      if (found) this.switchOn(key, false);
      else this.switchOff(key);
    }
    for (const key of this.marked) if (!this.find(key)) this.marked.delete(key);
    if (this.look && !this.find(this.look)) this.lookThrough(null);
    this.onChange();
  }

  // World pose of a camera: {pos, rot (row-major 3x3)}.
  pose({ robot, cam }) {
    const d = this.app.data, R0 = yawMat(robot.yaw ?? 0);
    const b = this.linkBody(robot, cam.link);
    const at = b >= 0 ? Array.from(d.xpos.subarray(3 * b, 3 * b + 3)) : robot.pos;
    const Rb = b >= 0 && cam.follow !== 'position' ? Array.from(d.xmat.subarray(9 * b, 9 * b + 9)) : R0;
    const off = mulMV(Rb, cam.pos);
    return { pos: at.map((v, i) => v + off[i]), rot: mulMM(Rb, mulMM(rpyMat(cam.rpy), LOOK_X)) };
  }

  // A Filament model matrix (column-major) for a pose.
  static matrix({ pos, rot: R }) {
    return [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, pos[0], pos[1], pos[2], 1];
  }

  // Switches a camera on, shown in a window, or takes up its changed
  // setup (a new size makes it anew, in the same place).
  switchOn(key, notify = true) {
    const found = this.find(key);
    if (!found) return;
    let c = this.on.get(key);
    const { width, height } = found.cam;
    let place = null;
    if (c && (c.width !== width || c.height !== height)) {
      place = c.pip.style.position === 'fixed' ? { left: c.pip.style.left, top: c.pip.style.top } : null;
      this.switchOff(key, true, false);
      c = null;
    }
    if (!c) {
      c = { key, ...found, width, height, next: -Infinity };
      this.create(c);
      if (place) { Object.assign(c.pip.style, { position: 'fixed', ...place }); document.body.append(c.pip); }
      this.on.set(key, c);
    }
    Object.assign(c, found, { hz: found.cam.hz });
    if (notify) this.onChange();
  }

  switchOff(key, keepLook = false, notify = true) {
    const c = this.on.get(key);
    if (!c) return;
    this.on.delete(key);
    const { engine } = this.app, gl = this.gl();
    engine.destroyView(c.view);
    engine.destroyCameraComponent(c.entity);
    Filament.EntityManager.get().destroy(c.entity);
    gl.deleteFramebuffer(c.fb);
    gl.deleteRenderbuffer(c.rb);
    gl.deleteBuffer(c.pbo);
    if (c.fence) gl.deleteSync(c.fence);
    c.pip.remove();
    if (!keepLook && this.look === key) this.lookThrough(null);
    if (notify) this.onChange();
  }

  // Shows or hides a camera's pose marker: a frustum (its field of view, and
  // its aspect) with a tick on its top edge.
  mark(key, on = !this.marked.has(key)) {
    if (on) this.marked.add(key);
    else this.marked.delete(key);
    this.onChange();
  }

  drawMarkers() {
    const segs = [];
    if (this.app.data) {
      for (const key of this.marked) {
        const found = this.find(key);
        if (!found) continue;
        const { pos, rot: R } = this.pose(found), cam = found.cam;
        const h = MARK.depth * Math.tan(cam.fovy * DEG / 2), w = h * cam.width / cam.height;
        const at = (x, y, z) => pos.map((v, i) => v + R[3 * i] * x + R[3 * i + 1] * y + R[3 * i + 2] * z);
        const c = [at(-w, -h, -MARK.depth), at(w, -h, -MARK.depth), at(w, h, -MARK.depth), at(-w, h, -MARK.depth)];
        for (let k = 0; k < 4; k++) segs.push([pos, c[k], MARK.frame], [c[k], c[(k + 1) % 4], MARK.frame]);
        segs.push([at(0, h, -MARK.depth), at(0, h * 1.5, -MARK.depth), MARK.up]);
      }
    }
    const tcm = this.app.engine.getTransformManager();
    while (this.parts.length < segs.length) this.parts.push({ ...this.app.xrPart(true), color: null });
    this.parts.forEach((part, k) => {
      const s = segs[k], inst = tcm.getInstance(part.e);
      tcm.setTransform(inst, (s && rodMatrix(s[0], s[1], MARK.radius)) ?? HIDDEN);
      inst.delete();
      if (s && part.color !== s[2]) {
        part.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, [...s[2], 1]);
        part.color = s[2];
      }
    });
  }

  gl() { return this.app.canvas.getContext('webgl2'); }

  // Its view, framebuffer, pixel buffer and window.
  create(c) {
    const { app } = this, F = Filament, gl = this.gl();
    c.entity = F.EntityManager.get().create();
    c.camera = app.engine.createCamera(c.entity);
    c.view = app.engine.createView();
    c.view.setCamera(c.camera);
    c.view.setScene(app.scene);
    c.view.setVisibleLayers(0xff, 0x1);  // the world only, not the viewer's markers (main.js SCENE)
    c.view.setViewport([0, 0, c.width, c.height]);
    app.viewQuality(c.view);  // anti-aliasing, SSAO and reflections as the Render panel says (App.setQuality keeps it so)
    c.rb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, c.rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, c.width, c.height);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    c.fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, c.fb);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, c.rb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    c.pixels = new Uint8Array(4 * c.width * c.height);  // the last frame, bottom row first (as read)
    c.pbo = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, c.pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, c.pixels.length, gl.STREAM_READ);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    c.fence = null;  // a read in flight
    c.frames = 0;
    // The window: title bar (drag it), look-through and close buttons, the picture.
    const canvas = el('canvas', { width: c.width, height: c.height });
    const look = el('button', { textContent: '◉', title: 'Look through this camera (Esc: back)' });
    const close = el('button', { textContent: '✕', title: 'Switch this camera off' });
    const title = el('span', { textContent: `${c.robot.prefix.slice(0, -1)} · ${c.cam.name}`, title: `${c.robot.label} › ${c.cam.name}` });
    c.title = title;
    const bar = el('div', { className: 'bar' }, title, look, close);
    c.pip = el('div', { className: 'pip' }, bar, canvas);
    c.image = canvas.getContext('2d').createImageData(c.width, c.height);
    c.canvas = canvas;
    look.onclick = () => this.lookThrough(this.look === c.key ? null : c.key);
    close.onclick = () => this.switchOff(c.key);
    this.draggable(c.pip, bar);
    this.pips.append(c.pip);
  }

  // Drags a window by its bar; from then on it stays where it is put. The
  // pointer is followed on the whole window (moving the window to <body>
  // would drop a pointer capture), however fast it leaves the bar.
  draggable(win, bar) {
    bar.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || (e.target !== bar && e.target.tagName !== 'SPAN')) return;
      e.preventDefault();
      const r = win.getBoundingClientRect(), from = [e.clientX - r.left, e.clientY - r.top];
      Object.assign(win.style, { position: 'fixed', left: `${r.left}px`, top: `${r.top}px` });
      document.body.append(win);
      const move = (ev) => {
        if (ev.pointerId !== e.pointerId) return;
        win.style.left = `${Math.max(0, Math.min(innerWidth - 40, ev.clientX - from[0]))}px`;
        win.style.top = `${Math.max(0, Math.min(innerHeight - 24, ev.clientY - from[1]))}px`;
      };
      const up = (ev) => {
        if (ev.pointerId !== e.pointerId) return;
        removeEventListener('pointermove', move, true);
        removeEventListener('pointerup', up, true);
        removeEventListener('pointercancel', up, true);
      };
      addEventListener('pointermove', move, true);
      addEventListener('pointerup', up, true);
      addEventListener('pointercancel', up, true);
    });
  }

  // The window looks through camera `key` (null: back to the orbit camera).
  lookThrough(key) {
    this.look = key;
    for (const c of this.on.values()) c.pip.classList.toggle('looking', c.key === key);
    if (!key) this.app.resize();  // the orbit camera's projection
    this.onChange();
  }

  // For App.frame: the camera to look through, {matrix, fovy}, or null.
  lookPose() {
    if (!this.look || !this.app.data) return null;
    const found = this.on.get(this.look) ?? this.find(this.look);
    return found ? { matrix: Cameras.matrix(this.pose(found)), fovy: found.cam.fovy } : null;
  }

  // After the window's frame: the reads that have arrived collected, then
  // the cameras that are due (and have no read in flight) rendered. A
  // camera keeps a timetable of its own (c.next: when its next frame is
  // due, a period after the last one was due), taken in the window frame
  // nearest to it, i.e. up to half a window frame early: going by when the
  // last frame was actually taken instead, a frame that came late in its
  // window frame (the main thread's work varies) made the next one wait a
  // window frame more, a hitch.
  render(now = performance.now()) {
    if (this.marked.size || this.parts.length) this.drawMarkers();
    const dt = now - (this.lastRender ?? now);
    this.lastRender = now;
    if (dt > 0 && dt < 100) this.frameDt += 0.1 * (dt - this.frameDt);  // the window's frame interval, smoothed
    if (!this.on.size || !this.app.data) return;
    for (const c of this.on.values()) {
      if (c.fence && this.collect(c)) {
        this.show(c);
        this.onFrame(c);
      }
    }
    const early = this.frameDt / 2;
    const due = [...this.on.values()].filter((c) => !c.fence && now >= c.next - early).sort((a, b) => a.next - b.next);
    for (const c of due.slice(0, PER_FRAME)) {
      if (!this.capture(c)) continue;  // skipped by Filament: due still
      const period = 1000 / c.hz;
      if (now - c.next > period) c.next = now;  // fell behind (or the first frame): a new timetable
      c.next += period;
    }
  }

  // A finished read into c.pixels; false while it is still in flight.
  collect(c) {
    const gl = this.gl();
    if (gl.getSyncParameter(c.fence, gl.SYNC_STATUS) !== gl.SIGNALED) return false;
    gl.deleteSync(c.fence);
    c.fence = null;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, c.pbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, c.pixels);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    c.frames++;
    c.stamp = c.shot;
    return true;
  }

  capture(c) {
    const { app } = this, gl = this.gl();
    c.camera.setModelMatrix(Cameras.matrix(this.pose(c)));
    c.camera.setProjectionFov(c.cam.fovy, c.width / c.height, NEAR, FAR, Filament.Camera$Fov.VERTICAL);
    // Filament renders into c.fb where it would the canvas (App.fbTarget);
    // its state cache believes the default framebuffer is bound throughout.
    app.offscreen = c.fb;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // Filament skips a frame while the GPU is behind: then nothing is read
    // (the framebuffer still holds the last frame), and it is tried again.
    const drawn = app.renderer.beginFrame(app.swapChain);
    if (drawn) {
      app.renderer.renderView(c.view);
      app.renderer.endFrame();
    }
    app.engine.execute();
    if (!drawn) {
      app.offscreen = null;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return false;
    }
    // Into the pixel buffer, collected once the fence has passed.
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, c.fb);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, c.pbo);
    gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    c.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    c.shot = Date.now();  // when it was taken (stamp: of the frame in c.pixels)
    app.offscreen = null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return true;
  }

  // One picture of the world, size x size, from a camera `aim(camera)`
  // places (App.capture: the window's view, for a stage's thumbnail):
  // rendered like a camera's frame, read at once. Its pixels (bottom row
  // first), or null if Filament skipped the frame.
  snapshot(size, aim) {
    const { app } = this, F = Filament, gl = this.gl();
    const entity = F.EntityManager.get().create(), camera = app.engine.createCamera(entity), view = app.engine.createView();
    view.setCamera(camera);
    view.setScene(app.scene);
    view.setVisibleLayers(0xff, 0x1);  // the world only
    view.setViewport([0, 0, size, size]);
    app.viewQuality(view);
    aim(camera);
    const rb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, size, size);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
    app.offscreen = fb;  // as capture(c)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const drawn = app.renderer.beginFrame(app.swapChain);
    if (drawn) {
      app.renderer.renderView(view);
      app.renderer.endFrame();
    }
    app.engine.execute();
    let pixels = null;
    if (drawn) {
      pixels = new Uint8Array(4 * size * size);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
      gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    }
    app.offscreen = null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    app.engine.destroyView(view);
    app.engine.destroyCameraComponent(entity);
    F.EntityManager.get().destroy(entity);
    gl.deleteFramebuffer(fb);
    gl.deleteRenderbuffer(rb);
    return pixels;
  }

  // Into its window: rows flipped (GL reads bottom up), opaque.
  show(c) {
    const { width: w, height: h, pixels, image } = c, row = 4 * w, out = image.data;
    for (let y = 0; y < h; y++) out.set(pixels.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    for (let i = 3; i < out.length; i += 4) out[i] = 255;
    c.canvas.getContext('2d').putImageData(image, 0, 0);
  }
}

// The Cameras panel, like the Teleop panel's IK chains: one robot at a time
// (picked, or double-clicked; the Teleop panel's at first, a newly spawned one), its cameras each in a
// foldable box (window switch, pose marker, delete; name, link, follow,
// offset, turn, size, rate, field of view), + Camera (on the selected,
// double-clicked, body if it is the robot's), Save (this browser), Defaults
// / Auto, Export (for camera_defaults.json).
export class CamerasPanel {
  constructor(cameras, root) {
    this.cameras = cameras;
    this.body = root.querySelector('.body');
    this.prefix = null;     // the robot shown
    this.known = new Set();  // prefixes of the robots seen (a new one gets shown)
    this.closed = new Set();  // keys of folded cameras
    cameras.onChange = () => this.render();
    this.render();
  }

  // Shows the robot's cameras (a double-clicked one, as the Teleop panel
  // drives it); null (nothing, or the floor) keeps the one shown.
  show(robot) {
    if (!robot || robot.prefix === this.prefix) return;
    this.prefix = robot.prefix;
    this.render();
  }

  num(value, { id, ...props }, onchange) {
    const inp = el('input', { type: 'number', value: +(+value).toFixed(5), ...props });
    inp.dataset.id = id;
    inp.onchange = () => { if (inp.value !== '' && Number.isFinite(+inp.value)) onchange(+inp.value); };
    inp.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') inp.blur(); };  // keys do not drive the robot
    return inp;
  }

  select(options, value, onchange, title = '') {
    const s = el('select', { title });
    for (const [v, text] of options) s.append(new Option(text, v, false, v === value));
    s.onchange = () => { onchange(s.value); s.blur(); };
    return s;
  }

  button(text, title, onclick, className = '') {
    return el('button', { type: 'button', textContent: text, title, className, onclick: (e) => { e.preventDefault(); onclick(); } });
  }

  field(label, ...ctl) {
    return el('div', { className: 'field' }, el('span', { textContent: label }), el('div', { className: 'ctl' }, ...ctl));
  }

  // The robot's body selected by a double-click: [world id, name], or null.
  selected(robot) {
    const b = this.cameras.app.perturb?.body ?? -1;
    return this.cameras.bodies(robot).find(([id]) => id === b) ?? null;
  }

  render() {
    const { cameras } = this, robots = cameras.app.robots ?? [];
    // The field in focus (e.g. tabbed into after a change) gets it back.
    const focus = this.body.contains(document.activeElement) ? document.activeElement.dataset.id : null;
    this.body.replaceChildren();
    if (!robots.length) {
      this.body.append(el('div', { className: 'hint', textContent: 'Nothing spawned yet. A robot\'s MJCF <camera>s show here, and + Camera adds one on any link.' }));
      return;
    }
    const fresh = robots.filter((r) => !this.known.has(r.prefix));
    this.known = new Set(robots.map((r) => r.prefix));
    if (fresh.length && this.prefix !== null) this.prefix = fresh.at(-1).prefix;
    const robot = robots.find((r) => r.prefix === this.prefix) ?? robots.find((r) => r === cameras.app.teleop?.active) ?? robots[0];
    this.prefix = robot.prefix;
    this.body.append(this.field('Robot', this.select(robots.map((r) => [r.prefix, `${r.prefix.slice(0, -1)} · ${r.label}`]), robot.prefix,
      (v) => { this.prefix = v; this.render(); })));
    const cfg = cameras.config(robot);
    if (!cfg) {
      this.body.append(el('div', { className: 'hint', textContent: 'Loading…' }));
      return;
    }
    if (!cfg.cameras.length) this.body.append(el('div', { className: 'hint', textContent: 'No cameras on this robot: double-click the body to put one on (e.g. a hand or the head), then + Camera.' }));
    for (const cam of cfg.cameras) this.body.append(this.renderCamera(robot, cam));

    const note = el('span', { className: 'note', textContent: cfg.source + (cfg.edited ? ' · changed' : '') });
    const hasDefaults = cameras.hasDefaults(robot);
    this.body.append(el('div', { className: 'bar' },
      this.button('+ Camera', 'A new camera on the selected (double-clicked) body of this robot, else its base', () => {
        const link = (this.selected(robot) ?? cameras.bodies(robot)[0])?.[1] ?? '';
        const key = cameras.add(robot, link);
        if (key) cameras.switchOn(key);
      }),
      this.button('Save', 'Use these cameras for every spawn of this robot (this browser)', () => cameras.save(robot)),
      this.button(hasDefaults ? 'Defaults' : 'Auto', hasDefaults ? 'Back to the built-in cameras (camera_defaults.json)' : 'Back to the cameras in the model', () => cameras.restoreDefaults(robot)),
      this.button('Export', 'Copy the cameras as JSON (for camera_defaults.json)', () => {
        const json = cameras.exportJson(robot);
        navigator.clipboard?.writeText(json).catch(() => {});
        console.log(json);
        note.textContent = 'copied (also in the console)';
      }),
      note));
    this.body.append(el('div', { className: 'hint', textContent: 'Rendered in this browser, at most two cameras per frame. ⌖ shows where a camera is and what it sees; ◉ on its window looks through it (Esc: back). Turn: roll / pitch / yaw about the link\'s x, y, z from looking along its +x (pitch > 0 looks down).' }));
    if (focus) this.body.querySelector(`[data-id="${CSS.escape(focus)}"]`)?.focus();
  }

  renderCamera(robot, cam) {
    const { cameras } = this, key = robot.prefix + cam.name, id = (what) => `${key}/${what}`;
    const det = el('details', { className: 'camera', open: !this.closed.has(key) });
    det.ontoggle = () => (det.open ? this.closed.delete(key) : this.closed.add(key));
    const box = el('input', { type: 'checkbox', checked: cameras.on.has(key), title: 'Show this camera in a window' });
    box.onchange = () => (box.checked ? cameras.switchOn(key) : cameras.switchOff(key));
    det.append(el('summary', {}, box,
      el('span', { className: 'title', textContent: cam.name + (cam.link ? ` › ${cam.link}` : '') }),
      this.button('⌖', 'Show where this camera is and what it sees', () => cameras.mark(key), cameras.marked.has(key) ? 'on' : ''),
      this.button('✕', 'Delete this camera', () => cameras.remove(key))));
    const edit = (change) => cameras.edit(key, change);

    const name = el('input', { type: 'text', value: cam.name, title: 'Name (also of its ROS topics, later)' });
    name.dataset.id = id('name');
    name.onchange = () => edit({ name: name.value });
    name.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') name.blur(); };
    det.append(this.field('Name', name));

    const links = cameras.bodies(robot).map(([, n]) => [n, n]);
    if (cam.link && !links.some(([n]) => n === cam.link)) links.unshift([cam.link, `${cam.link} (missing)`]);
    det.append(this.field('Link', this.select([['', '(robot frame)'], ...links], cam.link, (v) => edit({ link: v }), 'The link it rides'),
      this.button('◎', 'Link = the selected (double-clicked) body', () => {
        const hit = this.selected(robot);
        if (!hit) { alert('Double-click a body of this robot first.'); return; }
        edit({ link: hit[1] });
      })));
    det.append(this.field('Follow', this.select([['link', 'turns with the link'], ['position', 'position only']], cam.follow,
      (v) => edit({ follow: v }), 'position only: follows where the link is, keeping its turn in the robot frame (MJCF track cameras)')));
    const vec = (k, step, title) => cam[k].map((v, i) => this.num(v, { step, id: id(k + i), title: title(i) },
      (x) => { const next = [...cam[k]]; next[i] = x; edit({ [k]: next }); }));
    det.append(this.field('Offset', ...vec('pos', 0.01, (i) => `${'xyz'[i]} in the link frame (m)`)));
    det.append(this.field('Turn', ...vec('rpy', 5, (i) => `${['roll', 'pitch', 'yaw'][i]}: degrees about the link's ${'xyz'[i]}`)));
    const size = (k) => this.num(cam[k], { min: 16, max: MAX_SIZE, step: 1, id: id(k), title: `${k} (px)` },
      (v) => edit({ [k]: Math.round(Math.min(MAX_SIZE, Math.max(16, v))) }));
    det.append(this.field('Size', size('width'), el('span', { className: 'unit', textContent: '×' }), size('height'),
      this.select(RATES.map((hz) => [String(hz), `${hz} Hz`]), String(cam.hz), (v) => edit({ hz: Number(v) }), 'Frames a second')));
    det.append(this.field('FOV', this.num(cam.fovy, { min: 5, max: 170, step: 1, id: id('fovy'), title: 'Vertical field of view (degrees)' },
      (v) => edit({ fovy: Math.min(170, Math.max(5, v)) })), el('span', { className: 'unit', textContent: '° vertical' })));
    return det;
  }
}
