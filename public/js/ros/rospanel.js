// The ROS panel: shows and edits the world's one ROS setup (robot.js,
// app.ros). On top the default bridge (address, state, connect again) and
// the world's options (rate, /clock, stamps); then everything spawned with
// its switch and state (in network mode also when another browser holds
// it) and, unfolded, its bridge (the default or an address of its own),
// prefix, profile, topics, cmd_vel, joint commands (who writes which
// joints: drive.js) and cameras (which of the Cameras panel's go out).

import { bridgeUrl } from './bridge.js';
import { WORLD_DEFAULTS } from './robot.js';
import { locomotion } from '../drive/drive.js';

const FOLDED = 'robot-lobby/ros-open';  // the robots unfolded, by prefix
const UPDATE_MS = 250;

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

const STATES = {
  off: () => 'off', held: () => 'held by another browser',
  connecting: (url) => `connecting to ${bridgeUrl(url)}…`, connected: (url) => `connected to ${bridgeUrl(url)}`,
};

export class RosPanel {
  constructor(app, root) {
    this.app = app;
    this.ros = app.ros;
    this.el = root;
    this.body = root.querySelector('.body');
    try { this.open = new Set(JSON.parse(localStorage.getItem(FOLDED) ?? '[]')); } catch { this.open = new Set(); }
    this.views = [];     // live parts: {update()} each
    this.shape = '';     // what the panel was built for (render again when it changes)
    this.shown = 0;
    this.ros.onChange.add(() => { this.shown = 0; });  // (shown at the next frame)
    this.render();
  }

  // What is listed and how: a change renders the panel again.
  shapeKey() {
    const robots = this.app.robots ?? [];
    return JSON.stringify([robots.map((r) => [r.prefix, r.label, this.ros.robot(r, false)?.on ?? false, this.ros.heldElsewhere(r),
      locomotion(r).length > 0, !!r.use?.ik, !!r.use?.policy]), !!this.app.net]);
  }

  // Every frame (App.frame), a few times a second while shown.
  update(now = performance.now()) {
    if (this.el.hidden || now - this.shown < UPDATE_MS) return;
    this.shown = now;
    const shape = this.shapeKey();
    if (shape !== this.shape && !this.el.contains(document.activeElement)) this.render();
    else for (const v of this.views) v.update();
  }

  // Robot r unfolded (the Teleop panel's link).
  reveal(r) {
    if (!r) return;
    this.open.add(r.prefix);
    this.saveOpen();
    this.render();
    this.body.querySelector(`[data-prefix="${CSS.escape(r.prefix)}"]`)?.scrollIntoView({ block: 'nearest' });
  }

  saveOpen() {
    try { localStorage.setItem(FOLDED, JSON.stringify([...this.open])); } catch { /* no storage */ }
  }

  render() {
    this.shape = this.shapeKey();
    this.views = [];
    this.body.replaceChildren();
    this.renderWorld();
    const robots = this.app.robots ?? [];
    this.body.append(el('div', { className: 'sub', textContent: 'Spawned' }));
    if (!robots.length) {
      this.body.append(el('div', { className: 'hint', textContent: 'Nothing spawned yet. Each robot (or environment, or object) gets its own switch here: its joint states and TF go out, joint commands and cmd_vel come in.' }));
    }
    for (const r of robots) this.body.append(this.renderRobot(r));
    this.body.append(el('div', { className: 'hint', textContent:
      'Settings stay in this browser: the bridge address and the options above, per robot file its topics and its own address.'
      + (this.app.net ? ' In a shared world one browser publishes a robot: the one that switched it on, until it switches it off or leaves.' : '') }));
    for (const v of this.views) v.update();
  }

  // The default bridge and the world's options.
  renderWorld() {
    const ros = this.ros, s = ros.settings;
    const url = el('input', { type: 'text', value: s.url, placeholder: WORLD_DEFAULTS.url, spellcheck: false });
    url.onchange = () => ros.setUrl(url.value);
    url.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') url.blur(); };
    this.body.append(this.field('Bridge', 'rosbridge_server\'s WebSocket (host:port, or a ws:// / wss:// URL): every robot\'s, unless it has an address of its own; /clock goes here',
      url, this.button('↻', 'Connect again', () => ros.reconnect(true))));
    const state = el('span', { className: 'ros-state' });
    this.body.append(el('div', { className: 'bar' }, state));
    this.views.push({ update: () => {
      const b = ros.defaultBridge;
      this.setState(state, b ? b.status : 'off', b ? (b.status === 'error' ? `${b.error} · retrying` : STATES[b.status]?.(b.url)) : 'not in use: no robot on it is switched on');
    } });
    const hz = el('span', { className: 'unit', textContent: `${s.rate} Hz` });
    const rate = el('input', { type: 'range', min: 1, max: 60, step: 1, value: s.rate });
    rate.oninput = () => { s.rate = +rate.value; hz.textContent = `${s.rate} Hz`; };
    rate.onchange = () => { ros.save(); ros.restartTimer(); rate.blur(); };
    this.body.append(this.field('Rate', 'How often every robot\'s joint states and TF go out (and /clock)', rate, hz));
    const clock = el('input', { type: 'checkbox', checked: s.clock.on });
    const clockTopic = el('input', { type: 'text', value: s.clock.topic, spellcheck: false });
    const clockChanged = () => { ros.save(); ros.syncClock(); };
    clock.onchange = () => { s.clock.on = clock.checked; clockChanged(); };
    clockTopic.onchange = () => { s.clock.topic = clockTopic.value.trim() || WORLD_DEFAULTS.clock.topic; clockChanged(); };
    clockTopic.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') clockTopic.blur(); };
    this.body.append(this.field('Clock', 'rosgraph_msgs/Clock: the simulated time, for nodes with use_sim_time; on the default bridge, while a robot is on'
      + (this.app.net ? ' (one browser publishes it: the first with it on)' : ''), clock, clockTopic));
    if (this.app.net) {
      const note = el('div', { className: 'note' });
      this.body.append(note);
      this.views.push({ update: () => {
        const h = ros.holds.clock, t = !s.clock.on ? '' : h === 'you' ? '/clock: from this browser' : h === 'other' ? '/clock: from another browser' : '';
        if (note.textContent !== t) note.textContent = t;
        note.hidden = !t;
      } });
    }
    const sim = el('input', { type: 'checkbox', checked: !!s.simTime });
    sim.onchange = () => { s.simTime = sim.checked; ros.save(); };
    this.body.append(this.field('Stamps', 'Message stamps in simulated time (with Clock: what use_sim_time nodes expect), else wall time',
      el('label', { className: 'inline' }, sim, el('span', { textContent: 'simulated time' }))));
  }

  // One spawned thing: its switch and state; unfolded, its settings.
  renderRobot(r) {
    const ros = this.ros, rr = ros.robot(r), s = rr.settings, held = ros.heldElsewhere(r);
    const det = el('details', { className: 'ros-robot', open: this.open.has(r.prefix) });
    det.dataset.prefix = r.prefix;
    det.ontoggle = () => {
      if (det.open) this.open.add(r.prefix); else this.open.delete(r.prefix);
      this.saveOpen();
    };
    const on = el('input', { type: 'checkbox', checked: rr.on, disabled: held && !rr.on, title: held ? 'Another browser publishes this robot' : 'Its ROS on: joint states and TF out, commands in' });
    on.onclick = (e) => e.stopPropagation();  // (not the fold)
    on.onchange = () => { if (!rr.setOn(on.checked)) on.checked = false; this.render(); };
    const state = el('span', { className: 'ros-state' });
    det.append(el('summary', {}, on, el('span', { className: 'title', textContent: `${r.prefix.slice(0, -1)} · ${r.label}` }), state));
    const counts = el('div', { className: 'readout' });
    this.views.push({ update: () => {
      const st = rr.status;
      this.setState(state, st, st === 'held' ? 'held elsewhere' : st);
      state.title = st === 'error' ? `${rr.error} · retrying` : STATES[st]?.(rr.url) ?? st;
      const tw = this.app.drive.twist(r);
      const text = !rr.on ? '' : `sent ${rr.sent} · received ${rr.received}` + (rr.dropped ? ` · camera frames dropped ${rr.dropped}` : '')
        + (tw ? `\ncmd_vel vx ${tw[0].toFixed(2)} vy ${tw[1].toFixed(2)} ωz ${tw[2].toFixed(2)}` : '');
      if (counts.textContent !== text) counts.textContent = text;
      counts.hidden = !text;
    } });
    const error = el('div', { className: 'note warn' });
    det.append(error);
    this.views.push({ update: () => {
      const text = rr.status === 'error' ? `${rr.error} · retrying` : '';
      if (error.textContent !== text) error.textContent = text;
      error.hidden = !text;
    } });
    if (rr.lost) det.append(el('div', { className: 'note warn', textContent: 'Switched off here: another browser has switched it on.' }));
    const changed = () => { rr.save(); rr.sync(); };
    // Its bridge: the default one or an address of its own.
    const own = el('input', { type: 'text', value: s.url, placeholder: `default: ${ros.settings.url}`, spellcheck: false });
    own.onchange = () => { s.url = own.value.trim(); rr.save(); if (rr.on) { rr.attach(); ros.syncClock(); } };
    own.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') own.blur(); };
    det.append(this.field('Bridge', 'An address of this robot\'s own (empty: the default bridge above)', own));
    const prefix = el('input', { type: 'text', value: rr.prefix, placeholder: `${r.prefix.slice(0, -1)} (robot id)`, spellcheck: false });
    prefix.onchange = () => { rr.prefix = prefix.value.trim(); rr.sync(); };
    prefix.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') prefix.blur(); };
    det.append(this.field('Prefix', 'Before every topic not starting with / (r1/joint_states, r1/<camera>/…) and every TF frame (r1/base_link): empty, the robot\'s id; / for none', prefix));
    const prof = rr.profile, named = [prof.joints && 'joints', prof.frames && 'frames', prof.commands && 'commands', prof.cameras && 'cameras'].filter(Boolean);
    det.append(el('div', { className: 'bar' },
      el('span', { className: 'note', title: prof.about ?? '', textContent: Object.keys(prof).length ? `ROS profile of ${rr.key} (ros/profiles.json)${named.length ? `: ${named.join(', ')} named as on the real robot` : ''}` : 'no ROS profile for this model' }),
      this.button('Defaults', 'This robot\'s settings back to the defaults (the profile\'s, if it has one)', () => { rr.restoreDefaults(); this.render(); })));
    const topic = (k, label, title) => {
      const box = el('input', { type: 'checkbox', checked: s[k].on });
      const name = el('input', { type: 'text', value: s[k].topic, spellcheck: false });
      box.onchange = () => { s[k].on = box.checked; changed(); this.render(); };
      name.onchange = () => { s[k].topic = name.value.trim(); changed(); };
      name.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') name.blur(); };
      det.append(this.field(label, title, box, name));
    };
    const check = (label, title, text, get, set) => {
      const box = el('input', { type: 'checkbox', checked: !!get() });
      box.onchange = () => { set(box.checked); changed(); };
      det.append(this.field(label, title, el('label', { className: 'inline' }, box, el('span', { textContent: text }))));
    };
    det.append(el('div', { className: 'minor', textContent: 'Publish' }));
    topic('jointStates', 'Joints', 'sensor_msgs/JointState: the robot\'s joint positions, velocities and efforts (the actuators\' force on each joint)');
    topic('tf', 'TF', 'tf2_msgs/TFMessage: every body in its parent\'s frame, the root in "world"; the cameras\' optical frames');
    check('TF bodies', 'The bodies in TF; off when robot_state_publisher computes them from the URDF (the cameras\' frames still go)', 'every body',
      () => s.tf.bodies !== false, (v) => { s.tf.bodies = v; });
    this.renderCameras(det, r, rr, changed);
    det.append(el('div', { className: 'minor', textContent: 'Subscribe' }));
    topic('cmdVel', 'cmd_vel', 'geometry_msgs/Twist: linear x, y and angular z drive the walking policy or the base, before the keys and the goal while fresh');
    if (s.cmdVel.on && !locomotion(r).length) {
      det.append(el('div', { className: 'note warn' }, document.createTextNode('cmd_vel has nothing to drive: no walking policy or base is on. '),
        this.link('Teleop', 'Switch one on in the Teleop panel', () => this.app.openPanel?.('teleop', r))));
    }
    topic('jointCommand', 'Joint cmd', 'sensor_msgs/JointState: position targets for the actuators on those joints');
    check('Take over', 'Joint commands also move the joints a policy or IK drives, while they come (those give them up meanwhile, as the goal does for cmd_vel)',
      'over policy and IK', () => s.jointCommand.takeover, (v) => { s.jointCommand.takeover = v; });
    const joints = el('div', { className: 'note joints-owner' });
    det.append(joints);
    this.views.push({ update: () => {
      const text = this.app.drive.jointsText(r, s.jointCommand.on, s.jointCommand.takeover) || 'no joints with actuators';
      if (joints.textContent !== text) joints.textContent = text;
    } });
    det.append(counts);
    return det;
  }

  // Cameras: JPEG or raw, and which of the robot's go out (those switched
  // on in the Cameras panel, but the unticked).
  renderCameras(det, r, rr, changed) {
    const c = rr.settings.camera, cams = this.app.cameras;
    const on = el('input', { type: 'checkbox', checked: c.on });
    on.onchange = () => { c.on = on.checked; changed(); };
    const format = this.select([['jpeg', 'JPEG'], ['raw', 'raw']], c.format, (v) => { c.format = v; changed(); });
    det.append(this.field('Cameras', 'The cameras switched on in the Cameras panel, each frame: <camera>/image_raw/compressed (sensor_msgs/CompressedImage, JPEG) '
      + 'or …/image_raw (sensor_msgs/Image, rgb8), and …/camera_info; with TF, <camera>_optical_frame in its link\'s frame', on, format));
    const list = cams?.config(r)?.cameras ?? [];
    if (!list.length) return;
    const boxes = el('div', { className: 'cams' });
    for (const cam of list) {
      const off = new Set(c.off ?? []), box = el('input', { type: 'checkbox', checked: !off.has(cam.name) });
      box.onchange = () => {
        const set = new Set(c.off ?? []);
        if (box.checked) set.delete(cam.name); else set.add(cam.name);
        c.off = [...set];
        changed();
      };
      const label = el('span', { textContent: cam.name });
      boxes.append(el('label', { title: 'Published while switched on in the Cameras panel' }, box, label));
      this.views.push({ update: () => { label.className = cams.on?.has(r.prefix + cam.name) ? 'live' : ''; } });
    }
    det.append(this.field('', 'Which cameras go out (bright: switched on in the Cameras panel)', boxes));
  }

  setState(span, status, text) {
    if (span.textContent !== text) span.textContent = text;
    const cls = `ros-state ${status}`;
    if (span.className !== cls) span.className = cls;
  }

  field(label, title, ...controls) {
    return el('div', { className: 'field', title }, el('span', { textContent: label }), el('div', { className: 'ctl' }, ...controls));
  }

  select(options, value, onchange) {
    const s = el('select');
    for (const [v, text] of options) s.append(new Option(text, v, false, v === value));
    s.onchange = () => { onchange(s.value); s.blur(); };
    return s;
  }

  button(text, title, onclick) {
    return el('button', { type: 'button', textContent: text, title, onclick: (e) => { e.preventDefault(); onclick(); } });
  }

  link(text, title, onclick) {
    return el('a', { href: '#', textContent: text, title, onclick: (e) => { e.preventDefault(); onclick(); } });
  }
}
