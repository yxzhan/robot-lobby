// ROS from the Teleop panel: each robot can have a rosbridge connection of
// its own (rosbridge_suite's WebSocket JSON protocol, v2: advertise /
// publish / subscribe), made from this browser, in local and network mode
// alike. Per robot (RosLink):
//   publishes   joint states (sensor_msgs/JointState: the robot's hinge and
//               slide joints, names without the robot's prefix; velocities
//               only in local mode, the server sends positions) and TF
//               (tf2_msgs/TFMessage: each body in its parent's frame, the
//               root bodies in "world"), RATE times a second;
//   subscribes  joint commands (sensor_msgs/JointState: position targets for
//               the actuators that drive those joints) and cmd_vel
//               (geometry_msgs/Twist: linear x/y and angular z go to the
//               robot's walking policy or base, as the keys would; a twist
//               older than TWIST_TIMEOUT counts as zero).
//   publishes   the robot's cameras switched on in the Cameras panel
//               (cameras.js), each frame they render (<ns>: the robot's
//               id, r1, unless set): <ns>/<camera>/
//               image_raw/compressed (sensor_msgs/CompressedImage, JPEG) or
//               <ns>/<camera>/image_raw (sensor_msgs/Image, rgb8), and
//               <ns>/<camera>/camera_info (sensor_msgs/CameraInfo: a pinhole
//               of the camera's vertical field of view, square pixels, no
//               distortion); with TF on, each camera's <camera>_optical_frame
//               (z forward, x right, y down) in its link's frame.
// Stamps are wall time, ROS 2 style ({sec, nanosec}). Settings are kept per
// robot file in this browser; the connection itself is switched on in the
// panel each time.

const STORE = 'robot-lobby/ros';
const TWIST_TIMEOUT = 0.5;   // s
const RETRY_MS = 2000;

export const ROS_DEFAULTS = {
  url: 'localhost:9090',
  rate: 30,
  jointStates: { on: true, topic: 'joint_states' },  // what robot_state_publisher and RViz listen to
  tf: { on: true, topic: '/tf' },
  jointCommand: { on: true, topic: 'joint_command' },
  cmdVel: { on: true, topic: '/cmd_vel' },
  camera: { on: true, topic: '', format: 'jpeg', quality: 80 },  // topic: the namespace of the camera topics ('': the robot's id, r1; '/': none)
};
const MAX_BUFFERED = 8e6;  // bytes waiting in the WebSocket: camera frames are dropped above

const TYPES = {
  jointStates: 'sensor_msgs/JointState', tf: 'tf2_msgs/TFMessage',
  jointCommand: 'sensor_msgs/JointState', cmdVel: 'geometry_msgs/Twist',
  image: 'sensor_msgs/Image', compressed: 'sensor_msgs/CompressedImage', info: 'sensor_msgs/CameraInfo',
};
const PUBLISHED = ['jointStates', 'tf'];
const SUBSCRIBED = ['jointCommand', 'cmdVel'];

function loadSettings(key) {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? '{}')[key];
    if (saved?.jointStates?.topic === 'joint_state') saved.jointStates.topic = 'joint_states';  // the old default
    if (saved?.camera?.topic === 'camera/image_raw') delete saved.camera;  // the old placeholder
    if (saved) return { ...structuredClone(ROS_DEFAULTS), ...saved };
  } catch { /* no storage */ }
  return structuredClone(ROS_DEFAULTS);
}

function saveSettings(key, settings) {
  try {
    const all = JSON.parse(localStorage.getItem(STORE) ?? '{}');
    all[key] = settings;
    localStorage.setItem(STORE, JSON.stringify(all));
  } catch { /* no storage */ }
}

// 'localhost:9090' -> 'ws://localhost:9090'; a ws:// or wss:// URL as it is.
export function bridgeUrl(text) {
  const t = text.trim();
  return /^wss?:\/\//.test(t) ? t : `ws://${t}`;
}

function stamp(ms = Date.now()) {
  return { sec: Math.floor(ms / 1000), nanosec: Math.round(ms % 1000) * 1e6 };
}

// Bytes as base64 (rosbridge's JSON for uint8[]).
function base64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

// A rotation matrix (row-major 3x3) as a quaternion {x, y, z, w}.
function matQuat(R) {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = R, tr = m00 + m11 + m22;
  let w, x, y, z;
  if (tr > 0) {
    const S = Math.sqrt(tr + 1) * 2;
    [w, x, y, z] = [S / 4, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S];
  } else if (m00 > m11 && m00 > m22) {
    const S = Math.sqrt(1 + m00 - m11 - m22) * 2;
    [w, x, y, z] = [(m21 - m12) / S, S / 4, (m01 + m10) / S, (m02 + m20) / S];
  } else if (m11 > m22) {
    const S = Math.sqrt(1 + m11 - m00 - m22) * 2;
    [w, x, y, z] = [(m02 - m20) / S, (m01 + m10) / S, S / 4, (m12 + m21) / S];
  } else {
    const S = Math.sqrt(1 + m22 - m00 - m11) * 2;
    [w, x, y, z] = [(m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, S / 4];
  }
  return { x, y, z, w };
}

// One robot's rosbridge connection. `app` gives the world (model, data,
// mujoco names, net); `robot` is updated by the panel when the world is
// rebuilt (robot objects and body numbers change then).
export class RosLink {
  constructor(app, robot) {
    this.app = app;
    this.robot = robot;
    this.key = robot.key ?? robot.label;
    this.settings = loadSettings(this.key);
    this.on = false;
    this.ws = null;
    this.status = 'off';        // off | connecting | connected | error
    this.error = '';
    this.sent = 0;
    this.received = 0;
    this.twist = null;          // [vx, vy, wz] from cmd_vel, and when
    this.twistAt = -Infinity;
    this.layout = null;         // joint and body tables of the current model
    this.onChange = () => {};   // the panel re-renders its status
    this.timer = null;
    this.camTopics = new Map();  // advertised camera topics -> type
    this.encoding = new Set();   // keys of cameras with a frame being encoded
    this.dropped = 0;            // camera frames dropped (the WebSocket was behind)
  }

  save() { saveSettings(this.key, this.settings); }

  setOn(on) {
    this.on = on;
    clearTimeout(this.retry);
    if (on) this.connect();
    else this.close('off');
  }

  connect() {
    this.close('connecting');
    let ws;
    try {
      ws = this.ws = new WebSocket(bridgeUrl(this.settings.url));
    } catch (err) {
      this.fail(err.message ?? String(err));
      return;
    }
    ws.onopen = () => {
      if (ws !== this.ws) return;
      this.status = 'connected';
      this.error = '';
      this.topics = {};
      this.sync();
      this.restartTimer();
      this.onChange();
    };
    ws.onmessage = (e) => { if (ws === this.ws) this.receive(e.data); };
    ws.onerror = () => { if (ws === this.ws) this.error = `no rosbridge at ${bridgeUrl(this.settings.url)}`; };
    ws.onclose = () => { if (ws === this.ws) this.fail(this.error || 'connection closed'); };
    this.onChange();
  }

  fail(message) {
    this.close('error');
    this.error = message;
    if (this.on) this.retry = setTimeout(() => this.on && this.connect(), RETRY_MS);
    this.onChange();
  }

  close(status) {
    clearInterval(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
      try { ws.close(); } catch { /* closing */ }
    }
    this.topics = {};
    this.camTopics = new Map();
    this.twist = null;
    this.status = status;
  }

  send(msg) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify(msg));
  }

  // (Re)advertises and (re)subscribes to match the settings: called on
  // connecting and after every change in the panel.
  sync() {
    if (this.status !== 'connected') return;
    for (const k of [...PUBLISHED, ...SUBSCRIBED]) {
      const want = this.settings[k].on ? this.settings[k].topic.trim() : '';
      const have = this.topics[k] ?? '';
      if (want === have) continue;
      const pub = PUBLISHED.includes(k);
      if (have) this.send({ op: pub ? 'unadvertise' : 'unsubscribe', topic: have });
      if (want) this.send(pub ? { op: 'advertise', topic: want, type: TYPES[k] }
        : { op: 'subscribe', topic: want, type: TYPES[k], queue_length: 1 });
      this.topics[k] = want;
    }
    if (!this.topics.cmdVel) this.twist = null;
    // Camera topics: advertised again as frames come (publishCamera).
    for (const topic of this.camTopics.keys()) this.send({ op: 'unadvertise', topic });
    this.camTopics.clear();
  }

  restartTimer() {
    clearInterval(this.timer);
    if (this.status === 'connected') this.timer = setInterval(() => this.publish(), 1000 / this.settings.rate);
  }

  // ---- the world as ROS sees it ----

  // The robot's joints and bodies in the current model, recomputed when the
  // model changes.
  tables() {
    const { model } = this.app, r = this.robot;
    if (this.layout?.model === model && this.layout.robot === r) return this.layout;
    const mj = this.app.mujoco, T = (k) => mj.mjtObj[k].value;
    const name = (type, i) => mj.mj_id2name(model, T(type), i) ?? '';
    const p = r.prefix, own = (n) => n.startsWith(p);
    const joints = [];
    for (let j = 0; j < model.njnt; j++) {
      const n = name('mjOBJ_JOINT', j), type = model.jnt_type[j];
      if (own(n) && (type === 2 || type === 3)) joints.push({ j, name: n.slice(p.length), q: model.jnt_qposadr[j], v: model.jnt_dofadr[j] });
    }
    const bodies = [], frame = new Map();
    for (let b = 1; b < model.nbody; b++) {
      const n = name('mjOBJ_BODY', b);
      if (own(n)) { frame.set(b, n.slice(p.length)); bodies.push(b); }
    }
    const acts = new Map();  // joint id -> actuators driving it
    for (let a = 0; a < model.nu; a++) {
      if (model.actuator_trntype[a] !== 0) continue;  // joint transmissions only
      const j = model.actuator_trnid[2 * a];
      if (!acts.has(j)) acts.set(j, []);
      acts.get(j).push(a);
    }
    const byName = new Map(joints.map((jt) => [jt.name, jt]));
    this.layout = { model, robot: r, joints, bodies, frame, acts, byName };
    return this.layout;
  }

  publish() {
    const { model, data } = this.app;
    if (!model || !data || this.status !== 'connected') return;
    const t = this.tables(), header = (frame_id = '') => ({ stamp: stamp(), frame_id });
    if (this.topics.jointStates) {
      const net = !!this.app.net;
      this.send({ op: 'publish', topic: this.topics.jointStates, msg: {
        header: header(),
        name: t.joints.map((jt) => jt.name),
        position: t.joints.map((jt) => data.qpos[jt.q]),
        velocity: net ? [] : t.joints.map((jt) => data.qvel[jt.v]),
        effort: [],
      } });
      this.sent++;
    }
    if (this.topics.tf) {
      const transforms = t.bodies.map((b) => {
        const parent = model.body_parentid[b];
        const { translation: [x, y, z], rotation } = relative(data, parent, b);
        return {
          header: header(t.frame.get(parent) ?? 'world'),
          child_frame_id: t.frame.get(b),
          transform: { translation: { x, y, z }, rotation },
        };
      });
      if (this.settings.camera.on) transforms.push(...this.cameraFrames(t, header));
      this.send({ op: 'publish', topic: this.topics.tf, msg: { transforms } });
      this.sent++;
    }
  }

  // ---- cameras ----

  // TF of the robot's cameras: <camera>_optical_frame in its link's frame
  // (the world's for the robot's own frame).
  cameraFrames(t, header) {
    const cams = this.app.cameras, d = this.app.data;
    if (!cams) return [];
    return cams.list().filter((c) => c.robot.prefix === this.robot.prefix).map((c) => {
      const { pos, rot: R } = cams.pose(c), b = cams.linkBody(c.robot, c.cam.link);
      const Ro = [R[0], -R[1], -R[2], R[3], -R[4], -R[5], R[6], -R[7], -R[8]];  // optical: y down, z forward
      const pp = b >= 0 ? Array.from(d.xpos.subarray(3 * b, 3 * b + 3)) : [0, 0, 0];
      const Rp = b >= 0 ? Array.from(d.xmat.subarray(9 * b, 9 * b + 9)) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
      const dp = pos.map((v, i) => v - pp[i]);
      const [x, y, z] = [0, 1, 2].map((k) => Rp[k] * dp[0] + Rp[3 + k] * dp[1] + Rp[6 + k] * dp[2]);  // Rpᵀ·dp
      const rel = [0, 1, 2].flatMap((r) => [0, 1, 2].map((k) => Rp[r] * Ro[k] + Rp[3 + r] * Ro[3 + k] + Rp[6 + r] * Ro[6 + k]));  // Rpᵀ·Ro
      return {
        header: header(b >= 0 ? t.frame.get(b) ?? 'world' : 'world'),
        child_frame_id: `${c.cam.name}_optical_frame`,
        transform: { translation: { x, y, z }, rotation: matQuat(rel) },
      };
    });
  }

  advertise(topic, type) {
    if (this.camTopics.get(topic) === type) return;
    this.send({ op: 'advertise', topic, type });
    this.camTopics.set(topic, type);
  }

  // A camera frame that has just arrived (Cameras.onFrame: `c` a
  // switched-on camera, its picture in c.canvas and c.pixels, bottom row
  // first), with its camera_info.
  publishCamera(c) {
    const cs = this.settings.camera;
    if (this.status !== 'connected' || !cs.on || c.robot.prefix !== this.robot.prefix || this.encoding.has(c.key)) return;
    if (this.ws.bufferedAmount > MAX_BUFFERED) { this.dropped++; return; }
    const ns = (cs.topic.trim() || this.robot.prefix).replace(/\/+$/, ''), base = ns ? `${ns}/${c.cam.name}` : c.cam.name;
    const { width: w, height: h } = c, fovy = c.cam.fovy, header = { stamp: stamp(c.stamp), frame_id: `${c.cam.name}_optical_frame` };
    const fy = (h / 2) / Math.tan(fovy * Math.PI / 360), cx = w / 2, cy = h / 2;
    const info = {
      header, height: h, width: w, distortion_model: 'plumb_bob', d: [0, 0, 0, 0, 0],
      k: [fy, 0, cx, 0, fy, cy, 0, 0, 1], r: [1, 0, 0, 0, 1, 0, 0, 0, 1], p: [fy, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
      binning_x: 0, binning_y: 0, roi: { x_offset: 0, y_offset: 0, height: 0, width: 0, do_rectify: false },
    };
    const send = (topic, type, msg) => {
      if (this.status !== 'connected') return;
      this.advertise(topic, type);
      this.send({ op: 'publish', topic, msg });
      this.sent++;
    };
    if (cs.format === 'raw') {
      const rgb = new Uint8Array(3 * w * h), px = c.pixels;
      for (let y = 0; y < h; y++) {  // top row first, alpha dropped
        const src = 4 * w * (h - 1 - y), dst = 3 * w * y;
        for (let x = 0; x < w; x++) {
          rgb[dst + 3 * x] = px[src + 4 * x];
          rgb[dst + 3 * x + 1] = px[src + 4 * x + 1];
          rgb[dst + 3 * x + 2] = px[src + 4 * x + 2];
        }
      }
      send(`${base}/image_raw`, TYPES.image, { header, height: h, width: w, encoding: 'rgb8', is_bigendian: 0, step: 3 * w, data: base64(rgb) });
      send(`${base}/camera_info`, TYPES.info, info);
      return;
    }
    this.encoding.add(c.key);
    c.canvas.toBlob(async (blob) => {
      try {
        if (!blob) return;
        const jpeg = new Uint8Array(await blob.arrayBuffer());
        send(`${base}/image_raw/compressed`, TYPES.compressed, { header, format: 'jpeg', data: base64(jpeg) });
        send(`${base}/camera_info`, TYPES.info, info);
      } finally {
        this.encoding.delete(c.key);
      }
    }, 'image/jpeg', (cs.quality ?? 80) / 100);
  }

  receive(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    if (msg.op !== 'publish') return;
    this.received++;
    if (msg.topic === this.topics.cmdVel) {
      const l = msg.msg?.linear ?? {}, a = msg.msg?.angular ?? {};
      this.twist = [+l.x || 0, +l.y || 0, +a.z || 0];
      this.twistAt = performance.now();
    } else if (msg.topic === this.topics.jointCommand) {
      this.command(msg.msg ?? {});
    }
  }

  // Position targets by joint name: into the ctrl of the actuators on
  // those joints (clamped to their control range).
  command({ name = [], position = [] }) {
    const { model, data, net } = this.app;
    if (!model || !data) return;
    const t = this.tables();
    name.forEach((n, i) => {
      const jt = t.byName.get(n), v = position[i];
      if (!jt || !Number.isFinite(v)) return;
      for (const a of t.acts.get(jt.j) ?? []) {
        let u = v;
        if (model.actuator_ctrllimited[a]) u = Math.min(model.actuator_ctrlrange[2 * a + 1], Math.max(model.actuator_ctrlrange[2 * a], u));
        if (net) net.send('ctrl', { a, v: u });
        else data.ctrl[a] = u;
      }
    });
  }

  // The twist from cmd_vel while it is fresh, else null.
  freshTwist() {
    return this.twist && (performance.now() - this.twistAt) / 1000 < TWIST_TIMEOUT ? this.twist : null;
  }
}

// Body b's pose in body `parent`'s frame: {translation: [x, y, z], rotation: {x, y, z, w}}.
function relative(data, parent, b) {
  const p = (i) => [data.xpos[3 * i], data.xpos[3 * i + 1], data.xpos[3 * i + 2]];
  const q = (i) => [data.xquat[4 * i], data.xquat[4 * i + 1], data.xquat[4 * i + 2], data.xquat[4 * i + 3]];
  const [pw, px, py, pz] = q(parent), [cw, cx, cy, cz] = q(b);
  const d = p(b).map((v, k) => v - p(parent)[k]);
  // rotate d by the parent's inverse rotation
  const rot = (w, x, y, z, v) => {
    const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
  };
  const t = rot(pw, -px, -py, -pz, d);
  // conj(parent) * child
  const w = pw * cw + px * cx + py * cy + pz * cz;
  const x = pw * cx - px * cw - py * cz + pz * cy;
  const y = pw * cy + px * cz - py * cw - pz * cx;
  const z = pw * cz - px * cy + py * cx - pz * cw;
  return { translation: t, rotation: { x, y, z, w } };
}
