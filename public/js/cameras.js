// Cameras: the MJCF <camera>s of what is spawned (robots, environments,
// objects), rendered in this browser, in local and network mode alike.
//
// Each switched-on camera has a Filament View of its own, rendered after the
// window's frame into a framebuffer of its own: App.fbTarget makes Filament
// bind it where it binds the default framebuffer (the trick WebXR uses, see
// main.js xrGl), so the main canvas is untouched. Its pixels are read back
// asynchronously (into a pixel buffer, collected a frame or so later once
// its fence has passed: a plain readPixels would wait for the GPU to finish
// everything queued, the window's frame too, every time) for its
// picture-in-picture window, and later for ROS. A camera renders at its own
// rate (CamerasPanel), at most two per window frame.
//
// A camera's pose comes from the world's body poses (data.xpos / xmat: MuJoCo
// in local mode, net.js forwardKinematics in network mode) and its own model
// (modelCameras): fixed cameras ride their body; track / trackcom keep the
// world orientation and offset they have in the model's initial pose,
// following the body (trackcom approximated by the body). The view looks
// along the camera's -z with y up, as MuJoCo's and Filament's cameras do.

const RESOLUTIONS = [[160, 120], [320, 240], [640, 480], [1280, 720]];
const RATES = [5, 10, 15, 30];
const NEAR = 0.01, FAR = 100;
const PER_FRAME = 2;  // cameras rendered per window frame, at most

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
const yawMat = (t) => [Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t), 0, 0, 0, 1];

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

export class Cameras {
  constructor(app) {
    this.app = app;
    this.on = new Map();   // key -> switched-on camera (see switchOn)
    this.look = null;      // key of the camera the window looks through, or null
    this.pips = el('div', { id: 'pips' });
    document.body.append(this.pips);
    this.onChange = () => {};  // the panel re-renders
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && this.look) this.lookThrough(null); });
  }

  // Every camera of the world: [{key, robot, cam, label}].
  list() {
    return (this.app.robots ?? []).flatMap((robot) => (robot.visual.cameras ?? []).map((cam) => ({
      key: robot.prefix + cam.name, robot, cam, label: cam.name,
    })));
  }

  find(key) { return this.list().find((c) => c.key === key) ?? null; }

  // After a new world: switched-on cameras follow their robot (by key),
  // the others go.
  build() {
    for (const [key, c] of this.on) {
      const found = this.find(key);
      if (found) Object.assign(c, { robot: found.robot, cam: found.cam });
      else this.switchOff(key);
    }
    if (this.look && !this.find(this.look)) this.lookThrough(null);
    this.onChange();
  }

  // World pose of a camera: {pos, rot (row-major 3x3)}.
  pose({ robot, cam }) {
    const d = this.app.data, R0 = yawMat(robot.yaw ?? 0);
    const w = cam.body > 0 ? robot.bodyOffset + cam.body - 1 : -1;
    const bodyPos = w >= 0 ? Array.from(d.xpos.subarray(3 * w, 3 * w + 3)) : robot.pos;
    if (cam.mode === 1 || cam.mode === 2) {  // track, trackcom
      const off = mulMV(R0, cam.pos0);
      return { pos: bodyPos.map((v, i) => v + off[i]), rot: mulMM(R0, cam.mat0) };
    }
    const Rb = w >= 0 ? Array.from(d.xmat.subarray(9 * w, 9 * w + 9)) : R0;
    const off = mulMV(Rb, cam.pos);
    return { pos: bodyPos.map((v, i) => v + off[i]), rot: mulMM(Rb, quatMat(cam.quat)) };
  }

  // A Filament model matrix (column-major) for a pose.
  static matrix({ pos, rot: R }) {
    return [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, pos[0], pos[1], pos[2], 1];
  }

  defaults(cam) {
    const [w, h] = cam.resolution ?? [1, 1];
    return { width: w > 1 ? w : 320, height: h > 1 ? h : 240, hz: 10 };
  }

  // Switches a camera on, shown in a window (or settings changed: width,
  // height, hz).
  switchOn(key, settings = {}) {
    const found = this.find(key);
    if (!found) return;
    let c = this.on.get(key);
    const next = { ...(c ? { width: c.width, height: c.height, hz: c.hz } : this.defaults(found.cam)), ...settings };
    if (c && (c.width !== next.width || c.height !== next.height)) { this.switchOff(key, true); c = null; }
    if (!c) {
      c = { key, ...found, ...next, last: -Infinity };
      this.create(c);
      this.on.set(key, c);
    }
    c.hz = next.hz;
    this.onChange();
  }

  switchOff(key, keepLook = false) {
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
    this.onChange();
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
    c.view.setAntiAliasing(F.View$AntiAliasing.NONE);
    c.view.setMultiSampleAntiAliasingOptions({ enabled: false, sampleCount: 1 });
    c.view.setAmbientOcclusionOptions({ enabled: false });
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
    const bar = el('div', { className: 'bar' }, title, look, close);
    c.pip = el('div', { className: 'pip' }, bar, canvas);
    c.image = canvas.getContext('2d').createImageData(c.width, c.height);
    c.canvas = canvas;
    look.onclick = () => this.lookThrough(this.look === c.key ? null : c.key);
    close.onclick = () => this.switchOff(c.key);
    this.draggable(c.pip, bar);
    this.pips.append(c.pip);
  }

  // Drags a window by its bar; from then on it stays where it is put.
  draggable(win, bar) {
    let from = null;
    bar.addEventListener('pointerdown', (e) => {
      if (e.target !== bar && e.target.tagName !== 'SPAN') return;
      const r = win.getBoundingClientRect();
      from = [e.clientX - r.left, e.clientY - r.top];
      bar.setPointerCapture(e.pointerId);
      Object.assign(win.style, { position: 'fixed', left: `${r.left}px`, top: `${r.top}px` });
      document.body.append(win);
    });
    bar.addEventListener('pointermove', (e) => {
      if (!from) return;
      win.style.left = `${Math.max(0, Math.min(innerWidth - 40, e.clientX - from[0]))}px`;
      win.style.top = `${Math.max(0, Math.min(innerHeight - 24, e.clientY - from[1]))}px`;
    });
    bar.addEventListener('pointerup', () => { from = null; });
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
  // the cameras that are due (and have no read in flight) rendered.
  render(now = performance.now()) {
    if (!this.on.size || !this.app.data) return;
    for (const c of this.on.values()) if (c.fence && this.collect(c)) this.show(c);
    const due = [...this.on.values()].filter((c) => !c.fence && now - c.last >= 1000 / c.hz - 4).sort((a, b) => a.last - b.last);
    for (const c of due.slice(0, PER_FRAME)) {
      c.last = now;
      this.capture(c);
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
    if (app.renderer.beginFrame(app.swapChain)) {
      app.renderer.renderView(c.view);
      app.renderer.endFrame();
    }
    app.engine.execute();
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
  }

  // Into its window: rows flipped (GL reads bottom up), opaque.
  show(c) {
    const { width: w, height: h, pixels, image } = c, row = 4 * w, out = image.data;
    for (let y = 0; y < h; y++) out.set(pixels.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    for (let i = 3; i < out.length; i += 4) out[i] = 255;
    c.canvas.getContext('2d').putImageData(image, 0, 0);
  }
}

// The Cameras panel: every camera of the world, per robot, with its window
// switch, resolution, rate and look-through button.
export class CamerasPanel {
  constructor(cameras, root) {
    this.cameras = cameras;
    this.body = root.querySelector('.body');
    cameras.onChange = () => this.render();
    this.render();
  }

  render() {
    const cams = this.cameras.list();
    this.body.replaceChildren();
    if (!cams.length) {
      this.body.append(el('div', { className: 'hint', textContent: 'No cameras in what is spawned. Robots with MJCF <camera>s (wrist or head cameras) list them here.' }));
      return;
    }
    let group = null;
    for (const c of cams) {
      if (c.robot !== group) {
        group = c.robot;
        this.body.append(el('div', { className: 'sub', textContent: `${c.robot.prefix.slice(0, -1)} · ${c.robot.label}` }));
      }
      const on = this.cameras.on.get(c.key);
      const box = el('input', { type: 'checkbox', checked: !!on, title: 'Show this camera in a window' });
      box.onchange = () => (box.checked ? this.cameras.switchOn(c.key) : this.cameras.switchOff(c.key));
      const s = on ?? { ...this.cameras.defaults(c.cam) };
      const res = el('select', { title: 'Resolution (px)' });
      const sizes = RESOLUTIONS.some(([w, h]) => w === s.width && h === s.height) ? RESOLUTIONS : [[s.width, s.height], ...RESOLUTIONS];
      for (const [w, h] of sizes) res.append(new Option(`${w}×${h}`, `${w}x${h}`, false, w === s.width && h === s.height));
      res.onchange = () => {
        const [width, height] = res.value.split('x').map(Number);
        this.cameras.switchOn(c.key, { width, height });
      };
      const rate = el('select', { title: 'Frames a second' });
      for (const hz of RATES) rate.append(new Option(`${hz} Hz`, String(hz), false, hz === s.hz));
      rate.onchange = () => this.cameras.switchOn(c.key, { hz: Number(rate.value) });
      const look = el('button', { textContent: '◉', title: 'Look through this camera (Esc: back)', className: this.cameras.look === c.key ? 'on' : '' });
      look.onclick = () => this.cameras.lookThrough(this.cameras.look === c.key ? null : c.key);
      const name = el('label', { className: 'cam', title: `${c.cam.name}: fovy ${c.cam.fovy.toFixed(0)}°, ${['fixed', 'track', 'trackcom', 'targetbody', 'targetbodycom'][c.cam.mode] ?? ''}` },
        box, el('span', { textContent: c.cam.name }));
      this.body.append(el('div', { className: 'camrow' }, name, res, rate, look));
    }
    this.body.append(el('div', { className: 'hint', textContent: 'Rendered in this browser, at most two cameras per frame; ◉ looks through one in the main view (Esc: back).' }));
  }
}
