// Network mode: the physics world lives on a server (server/server.py), shared
// by every browser connected to it. The page loads no MuJoCo: it draws each
// model from the render data the server compiled for it (loadRender), lays
// the world out from the server's scene, and places the bodies from the
// qpos / mocap state it broadcasts (forwardKinematics). States arrive ~30
// times a second; frames in between are interpolated, drawn a little in the
// past so there is usually a snapshot on either side: one state interval
// plus a margin for how unevenly they arrive (jitter), between MIN_DELAY and
// MAX_DELAY.
//
// Requests (spawn, remove, reset, drag, picking, …) go to the server, which
// applies them and sends everyone the result. See server/server.py for the
// protocol. Which server, if any, endpoint.js decides.

import { serverUrl, wsUrl } from './endpoint.js';

const MIN_DELAY = 0.04, MAX_DELAY = 0.25;  // s of server time the view runs behind the newest snapshot
const KEEP = 40;    // snapshots kept
const PING_EVERY = 1000;  // ms
const SILENT_MAX = 5000;  // ms without a word from the server: the connection is gone
const RETRY_MAX = 10000;  // ms between reconnect attempts, at most


// Stands in for the MuJoCo module where the page's code asks it for names
// (inspect.js, teleop.js, main.js buildVisual): models made here carry them.
export const NET_MUJOCO = {
  mjtObj: Object.fromEntries(['BODY', 'JOINT', 'ACTUATOR', 'TEXTURE'].map((k) => [`mjOBJ_${k}`, { value: k.toLowerCase() }])),
  mj_id2name: (model, type, id) => model.names?.[type]?.[id] ?? '',
};

export class NetClient {
  // handlers: {scene(msg), meta(msg), error(msg), connected(), closed(retryAt)}.
  // Lost connections are retried until they come back (the server sends its
  // scene again on each); `opened` resolves on the first.
  constructor(handlers) {
    this.handlers = handlers;
    this.snaps = [];       // {version, epoch, time, qpos, mocap_pos, mocap_quat, ctrl, recv}
    this.renderTime = null;
    this.lastFrame = null;
    this.replies = new Map();  // rid -> {resolve, reject}
    this.nextRid = 1;
    this.rtt = null;           // ms, smoothed
    this.interval = 1 / 30;    // s between states as they arrive, smoothed
    this.jitter = 0.01;        // s, smoothed deviation from that
    this.delay = 0.1;          // s, the view's current lag behind the newest state
    this.traffic = { bytes: 0, states: 0, t0: performance.now(), rate: 0, hz: 0 };
    this.backoff = 0;          // reconnect attempts since the last open
    let first;
    this.opened = new Promise((resolve) => { first = resolve; });  // the first connection
    this.onOpen = first;
    // A phone waking up or a network coming back: try at once.
    const now = () => { if (this.retry) this.reconnect(); };
    window.addEventListener('online', now);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) now(); });
    this.connect();
  }

  connect() {
    clearTimeout(this.retry);
    this.retry = null;
    const ws = this.ws = new WebSocket(wsUrl());
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      if (ws !== this.ws) return;
      this.backoff = 0;
      this.waiting = null;
      this.snaps = [];  // the server sends its scene and state anew (it may have restarted)
      this.renderTime = null;
      this.ctrl = null;
      const ping = () => {
        this.checkAlive();
        this.waiting ??= performance.now();  // cleared by any message
        this.send('ping', { t: performance.now() });
      };
      ping();
      this.pinger = setInterval(ping, PING_EVERY);
      this.onOpen();
      this.handlers.connected?.();
    };
    ws.onclose = () => {
      if (ws !== this.ws) return;
      clearInterval(this.pinger);
      for (const { reject } of this.replies.values()) reject(new Error('disconnected'));
      this.replies.clear();
      // Try again after 1, 2, 4, 8, then every 10 s.
      const wait = Math.min(RETRY_MAX, 1000 * 2 ** this.backoff++);
      this.retryAt = performance.now() + wait;
      this.retry = setTimeout(() => this.connect(), wait);
      this.handlers.closed?.(this.retryAt);
    };
    ws.onmessage = (e) => {
      if (ws !== this.ws) return;
      this.waiting = null;
      this.traffic.bytes += typeof e.data === 'string' ? e.data.length : e.data.byteLength;
      if (typeof e.data !== 'string') { this.push(e.data); return; }
      const msg = JSON.parse(e.data);
      if (msg.type === 'pong') {
        const rtt = performance.now() - msg.t;
        this.rtt = this.rtt == null ? rtt : this.rtt + (rtt - this.rtt) * 0.3;
        return;
      }
      const waiting = msg.rid != null && this.replies.get(msg.rid);
      if (waiting) {
        this.replies.delete(msg.rid);
        if (msg.type === 'error') waiting.reject(new Error(msg.msg));
        else waiting.resolve(msg);
        return;
      }
      this.handlers[msg.type]?.(msg);
    };
  }

  get connected() {
    return this.ws.readyState === WebSocket.OPEN;
  }

  // Now rather than when the wait is over (the banner's button, main.js).
  reconnect() {
    if (this.retry) this.connect();
  }

  // An open socket whose pings have gone unanswered for a while is dead
  // without knowing it (e.g. after a phone slept, or Wi-Fi changed): drop it,
  // which starts the reconnecting.
  checkAlive() {
    if (this.connected && this.waiting != null && performance.now() - this.waiting > SILENT_MAX) {
      const ws = this.ws;
      ws.onclose();  // at once: a dead socket's close can take a long time
      ws.onclose = ws.onmessage = ws.onopen = null;
      ws.close();
    }
  }

  send(op, args = {}) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ op, ...args }));
  }

  // A request with a reply (e.g. 'ray'): resolves to the server's answer.
  request(op, args = {}) {
    const rid = this.nextRid++;
    return new Promise((resolve, reject) => {
      if (this.ws.readyState !== WebSocket.OPEN) { reject(new Error('not connected')); return; }
      this.replies.set(rid, { resolve, reject });
      this.ws.send(JSON.stringify({ op, rid, ...args }));
    });
  }

  // {rtt (ms), hz (states/s), rate (bytes/s)}, updated about once a second.
  stats() {
    const t = this.traffic, now = performance.now(), dt = (now - t.t0) / 1000;
    if (dt >= 1) {
      t.rate = t.bytes / dt;
      t.hz = t.states / dt;
      t.bytes = t.states = 0;
      t.t0 = now;
    }
    return { rtt: this.rtt, hz: t.hz, rate: t.rate, delay: this.delay };
  }

  push(buf) {
    const [version, epoch, nq, nmocap, nu] = new Int32Array(buf, 0, 5);
    const time = new Float64Array(buf, 24, 1)[0];
    const f = new Float32Array(buf, 32);
    let o = 0;
    const take = (n) => f.subarray(o, (o += n));
    const s = {
      version, epoch, time, recv: performance.now(),
      qpos: take(nq), mocap_pos: take(3 * nmocap), mocap_quat: take(4 * nmocap), ctrl: nu ? take(nu) : null,
    };
    this.traffic.states++;
    const prev = this.snaps.at(-1);
    if (prev) {  // arrival statistics (gaps over 0.5 s: paused, not jitter)
      const dt = (s.recv - prev.recv) / 1000;
      if (dt < 0.5) {
        this.interval += (dt - this.interval) * 0.05;
        this.jitter += (Math.abs(dt - this.interval) - this.jitter) * 0.05;
      }
    }
    if (s.ctrl) this.ctrl = { version, ctrl: s.ctrl };
    const last = this.snaps.at(-1);
    if (last && (last.version !== version || last.epoch !== epoch || s.time < last.time)) {
      this.snaps = [];  // a jump: no interpolating across it
      this.renderTime = null;
    }
    this.snaps.push(s);
    if (this.snaps.length > KEEP) this.snaps.shift();
  }

  // Writes the state at the current view time into `data` (worldData() of
  // `model`, the world of scene `version`) and places the bodies of `robots`
  // (forwardKinematics). False if there is no state for that world yet.
  apply(model, data, version, robots) {
    const snaps = this.snaps;
    const last = snaps.at(-1);
    if (!last || last.version !== version || last.qpos.length !== model.nq || last.mocap_pos.length !== 3 * model.nmocap) return false;
    const now = performance.now();
    // Running: trail the newest snapshot by DELAY. Paused (nothing new for a
    // while): catch up with it.
    const want = Math.min(MAX_DELAY, Math.max(MIN_DELAY, this.interval + 3 * this.jitter));
    this.delay += (want - this.delay) * 0.02;
    const fresh = now - last.recv < 200;
    const target = last.time - (fresh ? this.delay : 0);
    if (this.renderTime == null || Math.abs(target - this.renderTime) > 0.5) this.renderTime = target;
    else {
      this.renderTime += (now - (this.lastFrame ?? now)) / 1000;
      this.renderTime += (target - this.renderTime) * 0.05;  // drift back towards the target
    }
    this.lastFrame = now;
    const t = this.renderTime = Math.min(Math.max(this.renderTime, snaps[0].time), last.time);
    let i = snaps.length - 1;
    while (i > 0 && snaps[i - 1].time > t) i--;
    const b = snaps[i], a = snaps[Math.max(0, i - 1)];
    const alpha = b.time > a.time ? Math.min(1, Math.max(0, (t - a.time) / (b.time - a.time))) : 1;

    lerp(data.qpos, a.qpos, b.qpos, alpha);
    for (const k of quatSlots(model)) blendQuat(data.qpos, k, a.qpos, b.qpos, alpha);
    lerp(data.mocap_pos, a.mocap_pos, b.mocap_pos, alpha);
    for (let k = 0; k < model.nmocap; k++) blendQuat(data.mocap_quat, 4 * k, a.mocap_quat, b.mocap_quat, alpha);
    if (this.ctrl?.version === version && this.ctrl.ctrl.length === data.ctrl.length) data.ctrl.set(this.ctrl.ctrl);
    data.time = a.time + alpha * (b.time - a.time);
    forwardKinematics(robots, data);
    return true;
  }
}

// ---------------------------------------------------------------------------
// Models without MuJoCo

const ARRAY_TYPES = { f4: Float32Array, f8: Float64Array, i4: Int32Array, u1: Uint8Array };

// Render data of a model file (server/render_asset.py): a stand-in model with
// the fields main.js buildVisual reads, plus the kinematic tree; `images`
// maps texture names to image paths under public/ (like vfs.js).
export async function loadRender(rel) {
  const res = await fetch(serverUrl('render', { rel }));
  if (!res.ok) throw new Error(`${rel}: ${(await res.text()) || `HTTP ${res.status}`}`);
  const buf = await res.arrayBuffer();
  const n = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  const base = 4 + n;
  const model = { ...head, names: { texture: Array.from({ length: head.ntex }, (_, t) => `#${t}`) } };
  for (const [name, [dtype, offset, count]] of Object.entries(head.arrays)) {
    model[name] = new ARRAY_TYPES[dtype](buf, base + offset, count);
  }
  delete model.arrays;
  const images = new Map(Object.entries(head.images).map(([t, path]) => [`#${t}`, [path]]));
  return { model, images };
}

// The world's tables the panels read (the scene's `world`), as a model.
export function worldModel(w) {
  const model = { names: w.names };
  for (const [k, v] of Object.entries(w)) {
    if (k === 'names') continue;
    model[k] = Array.isArray(v) ? (v.every(Number.isInteger) ? Int32Array.from(v) : Float64Array.from(v)) : v;
  }
  return model;
}

// The state the page keeps of the world: what arrives from the server plus
// the body poses forwardKinematics computes.
export function worldData(model) {
  const xmat = new Float64Array(9 * model.nbody), xquat = new Float64Array(4 * model.nbody);
  xmat.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);  // the world body
  xquat[0] = 1;
  return {
    time: 0,
    qpos: new Float64Array(model.nq), qvel: new Float64Array(model.nv), act: new Float64Array(model.na),
    ctrl: new Float64Array(model.nu),
    mocap_pos: new Float64Array(3 * model.nmocap), mocap_quat: new Float64Array(4 * model.nmocap),
    xpos: new Float64Array(3 * model.nbody), xquat, xmat,
    xfrc_applied: new Float64Array(6 * model.nbody),
  };
}

// MuJoCo's mj_kinematics for the bodies of `robots`, each from its own
// model's tree (robot.visual.kin): body k of a robot is world body
// bodyOffset + k - 1, its root bodies hang off its attach frame (pos, yaw),
// its joints read qpos from robot.netAt.qpos and its mocap bodies
// mocap_pos / mocap_quat from robot.netAt.mocap on.
export function forwardKinematics(robots, data) {
  const { qpos, mocap_pos, mocap_quat, xpos, xquat, xmat } = data;
  const v = [0, 0, 0], qloc = [0, 0, 0, 0], tmp = [0, 0, 0, 0];
  for (const r of robots) {
    const k = r.visual.kin, off = r.bodyOffset - 1, q0 = r.netAt.qpos, m0 = r.netAt.mocap;
    const yaw = r.yaw ?? 0, fp = r.pos, fq = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
    for (let b = 1; b < k.nbody; b++) {
      const w = off + b, p = k.body_parentid[b];
      const P = p === 0 ? fp : xpos.subarray(3 * (off + p), 3 * (off + p) + 3);
      const Q = p === 0 ? fq : xquat.subarray(4 * (off + p), 4 * (off + p) + 4);
      const pos = xpos.subarray(3 * w, 3 * w + 3), quat = xquat.subarray(4 * w, 4 * w + 4);
      const mid = k.body_mocapid[b];
      if (mid >= 0) {
        pos.set(mocap_pos.subarray(3 * (m0 + mid), 3 * (m0 + mid) + 3));
        quat.set(mocap_quat.subarray(4 * (m0 + mid), 4 * (m0 + mid) + 4));
      } else {
        rotVec(Q, k.body_pos, 3 * b, v);
        for (let c = 0; c < 3; c++) pos[c] = P[c] + v[c];
        mulQuat(Q, 0, k.body_quat, 4 * b, quat);
        for (let j = k.body_jntadr[b], end = j + k.body_jntnum[b]; j < end; j++) {
          const local = k.jnt_qposadr[j], adr = q0 + local, type = k.jnt_type[j];
          if (type === 0) {  // free: the pose is the qpos
            pos.set(qpos.subarray(adr, adr + 3));
            quat.set(qpos.subarray(adr + 3, adr + 7));
            normalize4(quat);
            continue;
          }
          rotVec(quat, k.jnt_pos, 3 * j, v);
          const anchor = [pos[0] + v[0], pos[1] + v[1], pos[2] + v[2]];
          if (type === 2) {  // slide
            rotVec(quat, k.jnt_axis, 3 * j, v);
            const d = qpos[adr] - k.qpos0[local];
            for (let c = 0; c < 3; c++) pos[c] += v[c] * d;
            continue;
          }
          if (type === 1) {  // ball
            for (let c = 0; c < 4; c++) qloc[c] = qpos[adr + c];
            normalize4(qloc);
          } else {  // hinge
            const half = (qpos[adr] - k.qpos0[local]) / 2, s = Math.sin(half);
            qloc[0] = Math.cos(half);
            for (let c = 0; c < 3; c++) qloc[c + 1] = k.jnt_axis[3 * j + c] * s;
          }
          mulQuat(quat, 0, qloc, 0, tmp);
          quat.set(tmp);
          rotVec(quat, k.jnt_pos, 3 * j, v);  // turn about the anchor, not the body origin
          for (let c = 0; c < 3; c++) pos[c] = anchor[c] - v[c];
        }
      }
      normalize4(quat);
      quatToMat(quat, xmat, 9 * w);
    }
  }
}

function rotVec(q, a, i, out) {  // out = q * a[i..i+3] * q⁻¹
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]], vx = a[i], vy = a[i + 1], vz = a[i + 2];
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  out[0] = vx + w * tx + (y * tz - z * ty);
  out[1] = vy + w * ty + (z * tx - x * tz);
  out[2] = vz + w * tz + (x * ty - y * tx);
}

function mulQuat(a, i, b, j, out) {
  const aw = a[i], ax = a[i + 1], ay = a[i + 2], az = a[i + 3];
  const bw = b[j], bx = b[j + 1], by = b[j + 2], bz = b[j + 3];
  out[0] = aw * bw - ax * bx - ay * by - az * bz;
  out[1] = aw * bx + ax * bw + ay * bz - az * by;
  out[2] = aw * by - ax * bz + ay * bw + az * bx;
  out[3] = aw * bz + ax * by - ay * bx + az * bw;
}

function normalize4(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (n < 1e-15) { q[0] = 1; q[1] = q[2] = q[3] = 0; return; }
  for (let c = 0; c < 4; c++) q[c] /= n;
}

function quatToMat(q, out, o) {  // row-major, like mju_quat2Mat
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  out[o] = 1 - 2 * (y * y + z * z); out[o + 1] = 2 * (x * y - w * z); out[o + 2] = 2 * (x * z + w * y);
  out[o + 3] = 2 * (x * y + w * z); out[o + 4] = 1 - 2 * (x * x + z * z); out[o + 5] = 2 * (y * z - w * x);
  out[o + 6] = 2 * (x * z - w * y); out[o + 7] = 2 * (y * z + w * x); out[o + 8] = 1 - 2 * (x * x + y * y);
}

function lerp(out, a, b, t) {
  for (let i = 0; i < out.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
}

// Shortest-path quaternion blend at `k` (a and b may sit on opposite
// hemispheres), normalized.
function blendQuat(out, k, a, b, t) {
  const sign = a[k] * b[k] + a[k + 1] * b[k + 1] + a[k + 2] * b[k + 2] + a[k + 3] * b[k + 3] < 0 ? -1 : 1;
  let n = 0;
  for (let c = 0; c < 4; c++) {
    out[k + c] = a[k + c] + (sign * b[k + c] - a[k + c]) * t;
    n += out[k + c] * out[k + c];
  }
  n = Math.sqrt(n) || 1;
  for (let c = 0; c < 4; c++) out[k + c] /= n;
}

// qpos indices where a quaternion starts: free joints (after x, y, z) and
// ball joints. Cached per model.
const slots = new WeakMap();
function quatSlots(model) {
  let s = slots.get(model);
  if (!s) {
    s = [];
    for (let j = 0; j < model.njnt; j++) {
      const type = model.jnt_type[j], adr = model.jnt_qposadr[j];
      if (type === 0) s.push(adr + 3);
      else if (type === 1) s.push(adr);
    }
    slots.set(model, s);
  }
  return s;
}
