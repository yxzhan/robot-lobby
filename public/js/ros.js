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
//               (cameras.js), each frame they render: <camera>/
//               image_raw/compressed (sensor_msgs/CompressedImage, JPEG) or
//               <camera>/image_raw (sensor_msgs/Image, rgb8), and
//               <camera>/camera_info (sensor_msgs/CameraInfo: a pinhole
//               of the camera's vertical field of view, square pixels, no
//               distortion); with TF on, each camera's <camera>_optical_frame
//               (z forward, x right, y down) in its link's frame.
// One prefix per robot (the robot's id, r1, unless set; / for none) goes
// before every name, so robots of one model on one bridge do not overwrite
// each other: the topics (r1/joint_states, r1/wrist_cam/image_raw/compressed;
// absolute ones, /tf, as they are) and the TF frames (r1/base_link; the
// roots in "world"). Joint names in the messages stay plain.
// Stamps are wall time, ROS 2 style ({sec, nanosec}), or simulated time
// (simTime, for nodes on use_sim_time), with /clock (rosgraph_msgs/Clock)
// published if asked. Body TF can be left out (tf.bodies off: when
// robot_state_publisher computes it from the URDF); the cameras' frames
// still go. Settings are kept per robot file in this browser, the prefix per
// robot (it names one robot, not its model); the connection itself is
// switched on in the panel each time.
//
// A robot file can have a ROS profile (public/json/ros_defaults.json, keyed
// like ik_defaults.json): how the real robot's ROS side names things, so
// programs written for it (MoveIt, ros2_control with
// topic_based_ros2_control) work unchanged:
//   settings  defaults for the settings above (topics, simTime, tf.bodies…)
//   prefix    the prefix to start with ('/' for none)
//   joints    {model joint: ROS name}, in joint states and commands (null:
//             left out)
//   frames    {model body: TF frame}; world: the frame of the root bodies'
//             parent ('world')
//   commands  {ROS joint: {actuator, scale, offset}}: a commanded joint
//             driven by that actuator (ctrl = position * scale + offset;
//             e.g. a gripper on a tendon), beside those with actuators on
//             their joints
//   cameras   {camera: {topic, frame}}: its topic namespace (…/image_raw,
//             …/camera_info) and optical frame, as the real driver names
//             them (a RealSense: camera/camera/color, camera_color_optical_frame)
// Effort in joint states is qfrc_actuator (the actuators' force on each
// joint); in network mode velocities and efforts come with the server's
// states while a bridge is connected (NetClient.wantDynamics).

const STORE = 'robot-lobby/ros';
export const ROS_PROFILES = await fetch(new URL('../json/ros_defaults.json', import.meta.url)).then((r) => r.json()).catch(() => ({}));
const TWIST_TIMEOUT = 0.5;   // s
const RETRY_MS = 2000;

export const ROS_DEFAULTS = {
  url: 'localhost:9090',
  rate: 30,
  jointStates: { on: true, topic: 'joint_states' },  // what robot_state_publisher and RViz listen to
  tf: { on: true, topic: '/tf', bodies: true },
  clock: { on: false, topic: '/clock' },
  jointCommand: { on: true, topic: 'joint_command' },
  cmdVel: { on: true, topic: 'cmd_vel' },
  camera: { on: true, format: 'jpeg', quality: 80 },
  simTime: false,  // stamps in simulated time (else wall time)
};
const MAX_BUFFERED = 8e6;  // bytes waiting in the WebSocket: camera frames are dropped above

const TYPES = {
  jointStates: 'sensor_msgs/JointState', tf: 'tf2_msgs/TFMessage',
  jointCommand: 'sensor_msgs/JointState', cmdVel: 'geometry_msgs/Twist', clock: 'rosgraph_msgs/Clock',
  image: 'sensor_msgs/Image', compressed: 'sensor_msgs/CompressedImage', info: 'sensor_msgs/CameraInfo',
};
const PUBLISHED = ['jointStates', 'tf', 'clock'];
const SUBSCRIBED = ['jointCommand', 'cmdVel'];

// `over` on `base`, a level deep (a topic's {on, topic} merged).
function merged(base, over = {}) {
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(over)) out[k] = v && typeof v === 'object' && !Array.isArray(v) && out[k] && typeof out[k] === 'object' ? { ...out[k], ...v } : v;
  return out;
}

// The defaults, the robot file's profile's, then what was saved here.
function loadSettings(key) {
  const base = merged(ROS_DEFAULTS, ROS_PROFILES[key]?.settings);
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? '{}')[key];
    if (saved?.jointStates?.topic === 'joint_state') saved.jointStates.topic = 'joint_states';  // the old default
    if (saved?.camera) delete saved.camera.topic;  // the old camera namespace: the prefix now
    if (saved?.cmdVel?.topic === '/cmd_vel') saved.cmdVel.topic = 'cmd_vel';  // the old default: every robot on one topic
    if (saved) return merged(base, saved);
  } catch { /* no storage */ }
  return base;
}

// Settings back to the defaults (the profile's): the panel's Defaults.
export function resetSettings(key) {
  try {
    const all = JSON.parse(localStorage.getItem(STORE) ?? '{}');
    delete all[key];
    localStorage.setItem(STORE, JSON.stringify(all));
  } catch { /* no storage */ }
  return loadSettings(key);
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

// Seconds (simulated time) as a ROS 2 stamp.
function simStamp(t) {
  const sec = Math.floor(t);
  return { sec, nanosec: Math.min(999999999, Math.round((t - sec) * 1e9)) };
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
    this.profile = ROS_PROFILES[this.key] ?? {};  // how the real robot names things (ros_defaults.json)
    this.settings = loadSettings(this.key);
    this.prefix = this.profile.prefix ?? '';  // before every name ('': the robot's id; '/': none)
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

  // Back to the defaults (the profile's), connected as it was.
  restoreDefaults() {
    this.settings = resetSettings(this.key);
    this.prefix = this.profile.prefix ?? '';
    this.sync();
    this.restartTimer();
  }

  setOn(on) {
    this.on = on;
    clearTimeout(this.retry);
    this.app.net?.wantDynamics(this, on);  // velocities and efforts in the server's states while on
    if (on) this.connect();
    else this.close('off');
  }

  // The stamp of now: wall time, or the simulation's (simTime).
  now() {
    return this.settings.simTime ? simStamp(this.app.data?.time ?? 0) : stamp();
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

  // The prefix of this robot's names, without slashes ('' for none).
  ns() {
    const p = this.prefix.trim();
    return p === '/' ? '' : (p || this.robot.prefix).replace(/^\/+|\/+$/g, '');
  }

  // A topic under the prefix (an absolute one as it is), a frame under it.
  topicName(topic, ns = this.ns()) { return ns && topic && !topic.startsWith('/') ? `${ns}/${topic}` : topic; }
  frameName(frame, ns = this.ns()) { return ns ? `${ns}/${frame}` : frame; }

  send(msg) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify(msg));
  }

  // (Re)advertises and (re)subscribes to match the settings: called on
  // connecting and after every change in the panel.
  sync() {
    if (this.status !== 'connected') return;
    const ns = this.synced = this.ns();
    for (const k of [...PUBLISHED, ...SUBSCRIBED]) {
      const want = this.settings[k].on ? this.topicName(this.settings[k].topic.trim(), ns) : '';
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
    const p = r.prefix, own = (n) => n.startsWith(p), prof = this.profile;
    const rename = (map, n) => (map && n in map ? map[n] : n);  // (null: left out)
    const joints = [];
    for (let j = 0; j < model.njnt; j++) {
      const n = name('mjOBJ_JOINT', j), type = model.jnt_type[j];
      if (!own(n) || (type !== 2 && type !== 3)) continue;
      const ros = rename(prof.joints, n.slice(p.length));
      if (ros) joints.push({ j, name: ros, q: model.jnt_qposadr[j], v: model.jnt_dofadr[j] });
    }
    const bodies = [], frame = new Map();
    for (let b = 1; b < model.nbody; b++) {
      const n = name('mjOBJ_BODY', b);
      if (own(n)) { frame.set(b, rename(prof.frames, n.slice(p.length)) || n.slice(p.length)); bodies.push(b); }
    }
    const commands = new Map();  // ROS joint -> [{a, scale, offset}] of the profile's commands
    for (const [joint, c] of Object.entries(prof.commands ?? {})) {
      let a = -1;
      for (let i = 0; i < model.nu && a < 0; i++) if (name('mjOBJ_ACTUATOR', i) === p + c.actuator) a = i;
      if (a >= 0) commands.set(joint, [{ a, scale: c.scale ?? 1, offset: c.offset ?? 0 }]);
    }
    const acts = new Map();  // joint id -> actuators driving it
    for (let a = 0; a < model.nu; a++) {
      if (model.actuator_trntype[a] !== 0) continue;  // joint transmissions only
      const j = model.actuator_trnid[2 * a];
      if (!acts.has(j)) acts.set(j, []);
      acts.get(j).push(a);
    }
    const byName = new Map(joints.map((jt) => [jt.name, jt]));
    this.layout = { model, robot: r, joints, bodies, frame, acts, byName, commands };
    return this.layout;
  }

  publish() {
    const { model, data } = this.app;
    if (!model || !data || this.status !== 'connected') return;
    const t = this.tables(), now = this.now(), header = (frame_id = '') => ({ stamp: now, frame_id });
    const ns = this.synced, world = this.profile.world ?? 'world';
    const frame = (b) => t.frame.has(b) ? this.frameName(t.frame.get(b), ns) : world;
    if (this.topics.clock) {
      this.send({ op: 'publish', topic: this.topics.clock, msg: { clock: simStamp(data.time) } });
      this.sent++;
    }
    if (this.topics.jointStates) {
      const dyn = !this.app.net || data.dynamics;  // (in network mode while the server sends them)
      this.send({ op: 'publish', topic: this.topics.jointStates, msg: {
        header: header(),
        name: t.joints.map((jt) => jt.name),
        position: t.joints.map((jt) => data.qpos[jt.q]),
        velocity: dyn ? t.joints.map((jt) => data.qvel[jt.v]) : [],
        effort: dyn ? t.joints.map((jt) => data.qfrc_actuator[jt.v]) : [],
      } });
      this.sent++;
    }
    if (this.topics.tf) {
      const transforms = (this.settings.tf.bodies === false ? [] : t.bodies).map((b) => {
        const parent = model.body_parentid[b];
        const { translation: [x, y, z], rotation } = relative(data, parent, b);
        return {
          header: header(frame(parent)),
          child_frame_id: frame(b),
          transform: { translation: { x, y, z }, rotation },
        };
      });
      if (this.settings.camera.on) transforms.push(...this.cameraFrames(frame, header));
      if (transforms.length) {
        this.send({ op: 'publish', topic: this.topics.tf, msg: { transforms } });
        this.sent++;
      }
    }
  }

  // ---- cameras ----

  // TF of the robot's cameras: <camera>_optical_frame in its link's frame
  // (the world's for the robot's own frame); `frame(b)` names body b's.
  cameraFrames(frame, header) {
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
        header: header(b >= 0 ? frame(b) : this.profile.world ?? 'world'),
        child_frame_id: this.cameraFrame(c),
        transform: { translation: { x, y, z }, rotation: matQuat(rel) },
      };
    });
  }

  // A camera's optical frame and topic namespace: the profile's (as the
  // real driver names them), else <camera>_optical_frame and <camera>.
  cameraFrame(c) { return this.frameName(this.profile.cameras?.[c.cam.name]?.frame ?? `${c.cam.name}_optical_frame`, this.synced); }
  cameraTopic(c) { return this.topicName(this.profile.cameras?.[c.cam.name]?.topic ?? c.cam.name, this.synced); }

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
    const base = this.cameraTopic(c);
    const { width: w, height: h } = c, fovy = c.cam.fovy;
    const header = { stamp: this.settings.simTime ? simStamp(c.simStamp ?? 0) : stamp(c.stamp), frame_id: this.cameraFrame(c) };
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
      if (!Number.isFinite(v)) return;
      const drives = t.commands.get(n) ?? (jt ? (t.acts.get(jt.j) ?? []).map((a) => ({ a, scale: 1, offset: 0 })) : []);
      for (const { a, scale, offset } of drives) {
        let u = v * scale + offset;
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
