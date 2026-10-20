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
// Camera images are a placeholder until there are cameras.
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
  camera: { on: false, topic: 'camera/image_raw' },
};

const TYPES = {
  jointStates: 'sensor_msgs/JointState', tf: 'tf2_msgs/TFMessage',
  jointCommand: 'sensor_msgs/JointState', cmdVel: 'geometry_msgs/Twist', camera: 'sensor_msgs/Image',
};
const PUBLISHED = ['jointStates', 'tf'];
const SUBSCRIBED = ['jointCommand', 'cmdVel'];

function loadSettings(key) {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? '{}')[key];
    if (saved?.jointStates?.topic === 'joint_state') saved.jointStates.topic = 'joint_states';  // the old default
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

function stamp() {
  const ms = Date.now();
  return { sec: Math.floor(ms / 1000), nanosec: (ms % 1000) * 1e6 };
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
      this.send({ op: 'publish', topic: this.topics.tf, msg: { transforms } });
      this.sent++;
    }
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
