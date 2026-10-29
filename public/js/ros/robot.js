// ROS, from this browser, in local and network mode alike: the world's one
// ROS setup (RosWorld, app.ros) and per spawned thing its own connection
// (RosRobot), each switched on in the ROS panel (rospanel.js).
//
// The world (RosWorld): the default bridge's address (localhost:9090 until
// changed), how often states go out (rate), /clock (rosgraph_msgs/Clock,
// the simulated time, on the default bridge only) and the stamp mode
// (wall time, ROS 2 style {sec, nanosec}, or simulated time for nodes on
// use_sim_time). Every robot's states go out on one timer.
//
// Per robot (RosRobot), through its bridge (bridge.js: the default one, or
// an address of its own; one WebSocket per address, shared):
//   publishes   joint states (sensor_msgs/JointState: the robot's hinge and
//               slide joints, names without the robot's prefix; positions,
//               velocities and efforts, the actuators' force on each joint)
//               and TF (tf2_msgs/TFMessage: each body in its parent's frame,
//               the root bodies in "world"; the bodies can be left out,
//               tf.bodies off, when robot_state_publisher computes them
//               from the URDF), `rate` times a second;
//   subscribes  joint commands (sensor_msgs/JointState: position targets for
//               the actuators that drive those joints) and cmd_vel
//               (geometry_msgs/Twist: linear x/y and angular z), both into
//               the drive layer (drive.js): cmd_vel drives the walking
//               policy or base before the keys and the goal while fresh;
//               a joint a policy or IK drives is theirs, unless the
//               robot's jointCommand.takeover;
//   publishes   the robot's cameras switched on in the Cameras panel
//               (cameras.js; but those in camera.off), each frame they
//               render: <camera>/image_raw/compressed (sensor_msgs/
//               CompressedImage, JPEG) or <camera>/image_raw
//               (sensor_msgs/Image, rgb8), and <camera>/camera_info
//               (sensor_msgs/CameraInfo: a pinhole of the camera's vertical
//               field of view, square pixels, no distortion); with TF on,
//               each camera's <camera>_optical_frame (z forward, x right,
//               y down) in its link's frame.
// One prefix per robot (the robot's id, r1, unless set; / for none) goes
// before every name, so robots of one model on one bridge do not overwrite
// each other: the topics (r1/joint_states, r1/wrist_cam/image_raw/compressed;
// absolute ones, /tf, as they are) and the TF frames (r1/base_link; the
// roots in "world"). Joint names in the messages stay plain.
//
// Settings stay in this browser: the world's, and per robot file its
// topics and its own address if it has one; the prefix per robot (it names
// one robot, not its model). Switched on in the panel each time.
//
// A robot file can have a ROS profile (profiles in public/json/
// ros/profiles.json, keyed like controllers/ik/defaults.json): how the real robot's ROS
// side names things, so programs written for it (MoveIt, ros2_control with
// topic_based_ros2_control) work unchanged:
//   settings  defaults for the robot's settings above (topics, tf.bodies…)
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
//
// Network mode: one browser publishes a robot (else every browser with ROS
// on would publish its joint states again). The server keeps, per robot,
// which browser holds its ROS: the one that switched it on (op ros_hold),
// until it switches it off or leaves; /clock likewise. Another browser
// sees the robot as held (holds, from the server's `ros` messages) and
// cannot switch it on meanwhile. Velocities and efforts come with the
// server's states while a robot is on here (NetClient.wantDynamics).

import { RosBridge } from './bridge.js';

const WORLD_STORE = 'robot-lobby/ros-world';
const STORE = 'robot-lobby/ros';  // per robot file
export const ROS_PROFILES = await fetch(new URL('./profiles.json', import.meta.url)).then((r) => r.json()).catch(() => ({}));
const MAX_BUFFERED = 8e6;  // bytes waiting in the WebSocket: camera frames are dropped above

export const WORLD_DEFAULTS = {
  url: 'localhost:9090',
  rate: 30,
  clock: { on: false, topic: '/clock' },
  simTime: false,  // stamps in simulated time (else wall time)
};
export const ROBOT_DEFAULTS = {
  url: '',  // its own bridge ('': the default one)
  jointStates: { on: true, topic: 'joint_states' },  // what robot_state_publisher and RViz listen to
  tf: { on: true, topic: '/tf', bodies: true },
  jointCommand: { on: true, topic: 'joint_command', takeover: false },  // takeover: over a policy's or IK's joints too
  cmdVel: { on: true, topic: 'cmd_vel' },
  camera: { on: true, format: 'jpeg', quality: 80, off: [] },  // off: cameras not published
};
const WORLD_KEYS = Object.keys(WORLD_DEFAULTS).filter((k) => k !== 'url');

const TYPES = {
  jointStates: 'sensor_msgs/JointState', tf: 'tf2_msgs/TFMessage',
  jointCommand: 'sensor_msgs/JointState', cmdVel: 'geometry_msgs/Twist', clock: 'rosgraph_msgs/Clock',
  image: 'sensor_msgs/Image', compressed: 'sensor_msgs/CompressedImage', info: 'sensor_msgs/CameraInfo',
};
const PUBLISHED = ['jointStates', 'tf'];
const SUBSCRIBED = ['jointCommand', 'cmdVel'];

// `over` on `base`, a level deep (a topic's {on, topic} merged).
function merged(base, over = {}) {
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(over)) out[k] = v && typeof v === 'object' && !Array.isArray(v) && out[k] && typeof out[k] === 'object' ? { ...out[k], ...v } : v;
  return out;
}

const read = (key) => { try { return JSON.parse(localStorage.getItem(key) ?? 'null'); } catch { return null; } };
const write = (key, v) => { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* no storage */ } };

// Once: the settings of before, all per robot file, split into the world's
// and the robots'. A bridge address saved the same for every file becomes
// the default; differing ones stay as those files' own.
function migrate() {
  if (read(WORLD_STORE)) return;
  const all = read(STORE) ?? {}, entries = Object.values(all).filter((s) => s && typeof s === 'object');
  const world = structuredClone(WORLD_DEFAULTS);
  const urls = [...new Set(entries.map((s) => s.url?.trim()).filter(Boolean))];
  if (urls.length === 1) world.url = urls[0];
  for (const k of WORLD_KEYS) {
    const s = entries.find((e) => k in e);
    if (s) world[k] = k === 'clock' ? { ...world.clock, ...s.clock } : s[k];
  }
  for (const s of entries) {
    for (const k of WORLD_KEYS) delete s[k];
    if (urls.length === 1 || s.url?.trim() === world.url) delete s.url;
  }
  write(STORE, all);
  write(WORLD_STORE, world);
}

function loadWorld() {
  return merged(WORLD_DEFAULTS, read(WORLD_STORE) ?? {});
}

// The defaults, the robot file's profile's, then what was saved here.
function loadRobot(key) {
  const prof = { ...ROS_PROFILES[key]?.settings };
  for (const k of WORLD_KEYS) delete prof[k];
  const base = merged(ROBOT_DEFAULTS, prof);
  const saved = read(STORE)?.[key];
  if (!saved) return base;
  if (saved.jointStates?.topic === 'joint_state') saved.jointStates.topic = 'joint_states';  // the old default
  if (saved.camera) delete saved.camera.topic;  // the old camera namespace: the prefix now
  if (saved.cmdVel?.topic === '/cmd_vel') saved.cmdVel.topic = 'cmd_vel';  // the old default: every robot on one topic
  return merged(base, saved);
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

// The world's ROS: its settings, the bridges, a RosRobot per spawned thing
// that has been switched on (or looked at in the panel).
export class RosWorld {
  constructor(app) {
    this.app = app;
    migrate();
    this.settings = loadWorld();
    this.bridges = new Map();  // address -> RosBridge
    this.robots = new Map();   // robot prefix -> RosRobot
    this.holds = {};           // network mode: robot id or 'clock' -> 'you' | 'other' (the server's)
    this.onChange = new Set(); // () => …: something the panel shows changed
    this.clockOn = false;      // /clock advertised on the default bridge
    this.timer = null;
    this.restartTimer();
  }

  changed() { for (const f of this.onChange) f(); }

  save() { write(WORLD_STORE, this.settings); }

  // The bridge at `url` (made on first use; it forgets itself with its last user).
  bridge(url) {
    const key = url.trim() || this.settings.url;
    let b = this.bridges.get(key);
    if (!b) {
      b = new RosBridge(key, () => { if (this.bridges.get(key) === b) this.bridges.delete(key); this.changed(); });
      b.onChange.add(() => this.changed());
      this.bridges.set(key, b);
    }
    return b;
  }

  // The default bridge if anything uses it now, else null.
  get defaultBridge() { return this.bridges.get(this.settings.url) ?? null; }

  // A new default address: the robots on the default bridge move there.
  setUrl(url) {
    const next = url.trim() || WORLD_DEFAULTS.url;
    if (next === this.settings.url) return;
    this.settings.url = next;
    this.save();
    this.reconnect();
  }

  // Every robot on its bridge again (an address changed, or ↻).
  reconnect(fresh = false) {
    if (fresh) for (const b of this.bridges.values()) if (b.users.size) b.connect();
    for (const rr of this.robots.values()) if (rr.on) rr.attach();
    this.syncClock();
    this.changed();
  }

  // Robot r's ROS (made the first time; `create` false: null then).
  robot(r, create = true) {
    let rr = this.robots.get(r.prefix);
    if (!rr && create) this.robots.set(r.prefix, rr = new RosRobot(this, r));
    return rr ?? null;
  }

  // Called for every new world (App.setWorld): the robots' ROS follow them
  // into it (by prefix: in network mode every scene brings new robot
  // objects); a removed robot's closes.
  build() {
    const all = this.app.robots ?? [];
    for (const [prefix, rr] of this.robots) {
      const r = all.find((x) => x.prefix === prefix);
      if (r && (r.key ?? r.label) === rr.key) { rr.robot = r; rr.layout = null; continue; }
      rr.setOn(false);
      this.robots.delete(prefix);
    }
    this.syncClock();
    this.changed();
  }

  // The stamp of now: wall time, or the simulation's (simTime).
  now() {
    return this.settings.simTime ? simStamp(this.app.data?.time ?? 0) : stamp();
  }

  restartTimer() {
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 1000 / this.settings.rate);
  }

  // `rate` times a second: every robot's states, and /clock.
  tick() {
    const { model, data } = this.app;
    if (!model || !data) return;
    const now = this.now();
    for (const rr of this.robots.values()) if (rr.on && rr.bridge?.connected) rr.publish(now);
    const b = this.clockOn && this.defaultBridge;
    if (b?.connected) b.publish(this.settings.clock.topic, { clock: simStamp(data.time) });
  }

  // ---- /clock: on the default bridge, while on and a robot of this
  // browser is (in network mode: held by this browser) ----

  syncClock() {
    const c = this.settings.clock, net = this.app.net;
    const wanted = c.on && [...this.robots.values()].some((rr) => rr.on);
    if (net) {
      const mine = this.holds.clock === 'you';
      if (wanted && !mine && this.holds.clock !== 'other') net.send('ros_hold', { key: 'clock', on: true });
      if (!wanted && mine) net.send('ros_hold', { key: 'clock', on: false });
    }
    const on = wanted && (!net || this.holds.clock === 'you');
    const topic = c.topic.trim() || WORLD_DEFAULTS.clock.topic;
    if (this.clockOn && (!on || this.clockBridge !== this.settings.url || this.clockTopic !== topic)) {
      const b = this.bridges.get(this.clockBridge);
      b?.unadvertise(this.clockTopic, this);
      b?.release(this);
      this.clockOn = false;
    }
    if (on && !this.clockOn) {
      const b = this.bridge(this.settings.url);
      b.acquire(this);
      b.advertise(topic, TYPES.clock, this);
      Object.assign(this, { clockOn: true, clockBridge: this.settings.url, clockTopic: topic });
    }
  }

  // ---- network mode: who holds which robot's ROS ----

  // Robot r's ROS is held by another browser.
  heldElsewhere(r) {
    return !!this.app.net && this.holds[r.netId] === 'other';
  }

  // The server's holds ({key: 'you' | 'other'}): a robot of ours another
  // browser took meanwhile goes off here; one the server forgot (it
  // restarted) is claimed again.
  setHolds(holds) {
    this.holds = holds ?? {};
    for (const rr of this.robots.values()) {
      if (!rr.on) continue;
      const h = this.holds[rr.robot.netId];
      if (h === 'other') { rr.setOn(false); rr.lost = true; } else if (!h) this.app.net?.send('ros_hold', { key: rr.robot.netId, on: true });
    }
    this.syncClock();
    this.changed();
  }

  // ---- cameras ----

  // A camera frame that has just arrived (Cameras.onFrame): its robot's.
  publishCamera(c) {
    const rr = this.robots.get(c.robot.prefix);
    if (rr?.on) rr.publishCamera(c);
  }
}

// One robot's ROS (as RosWorld describes). `robot` is updated by the world
// when the world is rebuilt (robot objects and body numbers change then).
export class RosRobot {
  constructor(world, robot) {
    this.world = world;
    this.app = world.app;
    this.robot = robot;
    this.key = robot.key ?? robot.label;
    this.profile = ROS_PROFILES[this.key] ?? {};  // how the real robot names things (ros/profiles.json)
    this.settings = loadRobot(this.key);
    this.prefix = this.profile.prefix ?? '';  // before every name ('': the robot's id; '/': none)
    this.on = false;
    this.lost = false;          // switched off because another browser took it (network mode)
    this.bridge = null;
    this.sent = 0;
    this.received = 0;
    this.dropped = 0;           // camera frames dropped (the WebSocket was behind)
    this.layout = null;         // joint and body tables of the current model
    this.topics = {};           // setting key -> the topic advertised or subscribed
    this.unsub = {};            // setting key -> its unsubscribe
    this.camTopics = new Set(); // camera topics advertised
    this.encoding = new Set();  // keys of cameras with a frame being encoded
  }

  // The bridge address it uses: its own, else the world's default.
  get url() { return this.settings.url.trim() || this.world.settings.url; }

  // off | held (by another browser) | connecting | connected | error
  get status() {
    if (!this.on) return this.world.heldElsewhere(this.robot) ? 'held' : 'off';
    return this.bridge?.status ?? 'connecting';
  }

  get error() { return this.bridge?.error ?? ''; }

  save() {
    const all = read(STORE) ?? {};
    all[this.key] = this.settings;
    write(STORE, all);
  }

  // Back to the defaults (the profile's), connected as it was.
  restoreDefaults() {
    const all = read(STORE) ?? {};
    delete all[this.key];
    write(STORE, all);
    this.settings = loadRobot(this.key);
    this.prefix = this.profile.prefix ?? '';
    if (this.on) this.attach();
    this.world.changed();
  }

  // On or off; false if another browser holds it (network mode).
  setOn(on) {
    if (on && this.world.heldElsewhere(this.robot)) return false;
    if (on === this.on) return true;
    this.on = on;
    this.lost = false;
    const net = this.app.net;
    net?.wantDynamics(this, on);  // velocities and efforts in the server's states while on
    net?.send('ros_hold', { key: this.robot.netId, on });
    if (on) this.attach();
    else this.detach();
    this.world.syncClock();
    this.world.changed();
    return true;
  }

  // On its bridge (again: the address may have changed), topics as set.
  attach() {
    const b = this.world.bridge(this.url);
    if (b !== this.bridge) {
      this.detach();
      this.bridge = b;
      b.acquire(this);
    }
    this.sync();
  }

  detach() {
    const b = this.bridge;
    if (!b) return;
    for (const off of Object.values(this.unsub)) off();
    this.unsub = {};
    this.topics = {};
    this.camTopics.clear();
    this.bridge = null;
    b.release(this);  // (also its topics)
  }

  // The prefix of this robot's names, without slashes ('' for none).
  ns() {
    const p = this.prefix.trim();
    return p === '/' ? '' : (p || this.robot.prefix).replace(/^\/+|\/+$/g, '');
  }

  // A topic under the prefix (an absolute one as it is), a frame under it.
  topicName(topic, ns = this.ns()) { return ns && topic && !topic.startsWith('/') ? `${ns}/${topic}` : topic; }
  frameName(frame, ns = this.ns()) { return ns ? `${ns}/${frame}` : frame; }

  // Advertises and subscribes to match the settings: on attaching and
  // after every change in the panel.
  sync() {
    const b = this.bridge;
    if (!b) return;
    const ns = this.synced = this.ns();
    for (const k of [...PUBLISHED, ...SUBSCRIBED]) {
      const want = this.settings[k].on ? this.topicName(this.settings[k].topic.trim(), ns) : '';
      const have = this.topics[k] ?? '';
      if (want === have) continue;
      const pub = PUBLISHED.includes(k);
      if (have) { if (pub) b.unadvertise(have, this); else { this.unsub[k]?.(); delete this.unsub[k]; } }
      if (want) {
        if (pub) b.advertise(want, TYPES[k], this);
        else this.unsub[k] = b.subscribe(want, TYPES[k], (msg) => this.receive(k, msg));
      }
      this.topics[k] = want;
    }
    // Camera topics: advertised again as frames come (publishCamera).
    for (const topic of this.camTopics) b.unadvertise(topic, this);
    this.camTopics.clear();
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

  send(topic, msg) {
    this.bridge.publish(topic, msg);
    this.sent++;
  }

  // Joint states and TF, stamped `now` (RosWorld.tick).
  publish(now) {
    const { model, data } = this.app;
    const t = this.tables(), header = (frame_id = '') => ({ stamp: now, frame_id });
    const ns = this.synced, world = this.profile.world ?? 'world';
    const frame = (b) => t.frame.has(b) ? this.frameName(t.frame.get(b), ns) : world;
    if (this.topics.jointStates) {
      const dyn = !this.app.net || data.dynamics;  // (in network mode while the server sends them)
      this.send(this.topics.jointStates, {
        header: header(),
        name: t.joints.map((jt) => jt.name),
        position: t.joints.map((jt) => data.qpos[jt.q]),
        velocity: dyn ? t.joints.map((jt) => data.qvel[jt.v]) : [],
        effort: dyn ? t.joints.map((jt) => data.qfrc_actuator[jt.v]) : [],
      });
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
      if (transforms.length) this.send(this.topics.tf, { transforms });
    }
  }

  // ---- cameras ----

  // The robot's cameras switched on in the Cameras panel that go out (not in camera.off).
  cameras() {
    const cams = this.app.cameras, off = new Set(this.settings.camera.off ?? []);
    return (cams?.list() ?? []).filter((c) => c.robot.prefix === this.robot.prefix && !off.has(c.cam.name));
  }

  // TF of the robot's cameras: <camera>_optical_frame in its link's frame
  // (the world's for the robot's own frame); `frame(b)` names body b's.
  cameraFrames(frame, header) {
    const cams = this.app.cameras, d = this.app.data;
    return this.cameras().map((c) => {
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

  // A camera frame that has just arrived (Cameras.onFrame: `c` a
  // switched-on camera, its picture in c.canvas and c.pixels, bottom row
  // first), with its camera_info.
  publishCamera(c) {
    const cs = this.settings.camera, b = this.bridge;
    if (!b?.connected || !cs.on || (cs.off ?? []).includes(c.cam.name) || this.encoding.has(c.key)) return;
    if (b.buffered > MAX_BUFFERED) { this.dropped++; return; }
    const base = this.cameraTopic(c);
    const { width: w, height: h } = c, fovy = c.cam.fovy;
    const header = { stamp: this.world.settings.simTime ? simStamp(c.simStamp ?? 0) : stamp(c.stamp), frame_id: this.cameraFrame(c) };
    const fy = (h / 2) / Math.tan(fovy * Math.PI / 360), cx = w / 2, cy = h / 2;
    const info = {
      header, height: h, width: w, distortion_model: 'plumb_bob', d: [0, 0, 0, 0, 0],
      k: [fy, 0, cx, 0, fy, cy, 0, 0, 1], r: [1, 0, 0, 0, 1, 0, 0, 0, 1], p: [fy, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
      binning_x: 0, binning_y: 0, roi: { x_offset: 0, y_offset: 0, height: 0, width: 0, do_rectify: false },
    };
    const send = (topic, type, msg) => {
      if (this.bridge !== b || !b.connected) return;
      b.advertise(topic, type, this);
      this.camTopics.add(topic);
      this.send(topic, msg);
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

  // ---- commands, into the drive layer ----

  receive(k, msg) {
    this.received++;
    if (k === 'cmdVel') {
      const l = msg.linear ?? {}, a = msg.angular ?? {};
      this.app.drive.push(this.robot, 'cmdVel', [+l.x || 0, +l.y || 0, +a.z || 0]);
    } else {
      this.command(msg);
    }
  }

  // Position targets by joint name: the ctrl of the actuators on those
  // joints (clamped to their control range), into the drive layer.
  command({ name = [], position = [] }) {
    const { model, data } = this.app;
    if (!model || !data) return;
    const t = this.tables(), pairs = [];
    name.forEach((n, i) => {
      const jt = t.byName.get(n), v = position[i];
      if (!Number.isFinite(v)) return;
      const drives = t.commands.get(n) ?? (jt ? (t.acts.get(jt.j) ?? []).map((a) => ({ a, scale: 1, offset: 0 })) : []);
      for (const { a, scale, offset } of drives) {
        let u = v * scale + offset;
        if (model.actuator_ctrllimited[a]) u = Math.min(model.actuator_ctrlrange[2 * a + 1], Math.max(model.actuator_ctrlrange[2 * a], u));
        pairs.push([a, u]);
      }
    });
    this.app.drive.jointCommand(this.robot, pairs, this.settings.jointCommand.takeover);
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
