// Recording the simulation's state and replaying it (the toolbar's Record,
// the replay bar under the view): a clip of this tab only, in memory, gone
// on reload or when the world changes (a spawn or removal rebuilds it).
//
// Recording keeps, every REC_DT of simulated time, what places the bodies:
// qpos and the mocap poses (float32), and the time. A Reset during it (time
// back to 0) goes on in the same clip: the clip's own clock counts the time
// recorded, not data.time. At MAX_SECONDS the oldest samples go.
//
// Replaying shows the clip at a point of its clock: the two samples around
// it blended (positions linearly, quaternions on the short arc) and the
// bodies placed from them (mj_forward here, forwardKinematics in network
// mode). The local simulation is paused meanwhile and its state put back on
// Live; in network mode the shared world goes on for everyone and Live
// shows it again (nothing is sent to the server).

import { blendQuat, forwardKinematics, lerp, quatSlots } from '../net/net.js';

const REC_DT = 1 / 30;     // s of simulated time between samples
const MAX_SECONDS = 300;   // s of recording kept (the oldest dropped)
const MAX_GAP = 0.5;       // s: a longer step between samples (a Reset, a jump) counts as this
const SPEEDS = [0.1, 0.25, 0.5, 1, 2];

export class Recorder {
  // app: the viewer (model, data, robots, net, paused, syncTransforms,
  // showPaused); ui: {button, bar} (index.html #record, #replay).
  constructor(app, { button, bar }) {
    this.app = app;
    this.button = button;
    this.bar = bar;
    this.clip = null;        // {model, data, samples: [{t, time, qpos, mocap_pos, mocap_quat}]}
    this.recording = false;
    this.replay = null;      // while replaying: {t, playing, speed, last (wall ms), saved (the live state)}
    this.speed = 1;
    const $ = (s) => bar.querySelector(s);
    this.ui = { play: $('.play'), seek: $('.seek'), time: $('.time'), speed: $('.speed'), live: $('.live'), discard: $('.discard'), note: $('.note') };
    this.ui.speed.replaceChildren(...SPEEDS.map((v) => new Option(`${v}×`, v, v === 1, v === 1)));
    button.onclick = () => (this.recording ? this.stop() : this.start());
    this.ui.play.onclick = () => this.togglePlay();
    this.ui.seek.oninput = () => this.seek(+this.ui.seek.value);
    this.ui.speed.onchange = () => { this.speed = +this.ui.speed.value; };
    this.ui.live.onclick = () => this.live();
    this.ui.discard.onclick = () => this.discard();
    this.show();
  }

  get length() {
    return this.clip?.samples.at(-1)?.t ?? 0;
  }

  get replaying() {
    return !!this.replay;
  }

  start() {
    const { model, data } = this.app;
    if (!model) return;
    if (this.replay) this.live();
    this.clip = { model, data, samples: [] };
    this.recording = true;
    this.note = '';
    this.sample(true);
    this.show();
  }

  stop() {
    this.recording = false;
    if (this.clip && this.clip.samples.length < 2) this.clip = null;
    this.show();
  }

  discard() {
    if (this.replay) this.live();
    this.recording = false;
    this.clip = null;
    this.note = '';
    this.show();
  }

  // The world was rebuilt (or is gone): the clip no longer fits it.
  dropped(why) {
    this.replay = null;  // (the state to restore went with the old world)
    this.recording = false;
    this.clip = null;
    this.note = why;
    this.show();
  }

  // Every frame (main.js frame): records, or replays, and keeps the bar current.
  frame(now) {
    const { model, data } = this.app;
    if (this.clip && (model !== this.clip.model || data !== this.clip.data)) {
      this.dropped('the world changed: recording dropped');
      return false;
    }
    if (this.replay) {
      const r = this.replay;
      if (r.playing) {
        r.t += (now - r.last) / 1000 * this.speed;
        if (r.t >= this.length) { r.t = this.length; r.playing = false; }
      }
      r.last = now;
      this.place(r.t);
      this.showTime();
      return true;
    }
    if (this.recording) this.sample();
    if (this.recording && now - (this.shownAt ?? 0) > 250) { this.shownAt = now; this.showTime(); }
    return false;
  }

  sample(first = false) {
    const { data } = this.app, s = this.clip.samples, prev = s.at(-1);
    if (!first && prev && data.time >= prev.time && data.time - prev.time < REC_DT - 1e-9) return;
    const dt = prev ? Math.min(MAX_GAP, Math.max(0, data.time - prev.time)) : 0;
    s.push({
      t: (prev?.t ?? 0) + (data.time < (prev?.time ?? 0) ? REC_DT : dt), time: data.time,
      qpos: Float32Array.from(data.qpos), mocap_pos: Float32Array.from(data.mocap_pos), mocap_quat: Float32Array.from(data.mocap_quat),
    });
    while (s.length > 2 && s.at(-1).t - s[0].t > MAX_SECONDS) s.shift();
  }

  // Replay from where it is (or the start, at the end); pause it.
  togglePlay() {
    if (!this.clip) return;
    this.enter();
    const r = this.replay;
    if (!r.playing && r.t >= this.length - 1e-9) r.t = this.clip.samples[0].t;
    r.playing = !r.playing;
    r.last = performance.now();
    this.show();
  }

  seek(t) {
    if (!this.clip) return;
    this.enter();
    this.replay.t = t;
  }

  // Into replay: the live simulation paused, its state kept for Live.
  enter() {
    if (this.replay) return;
    if (this.recording) this.stop();
    const app = this.app, d = app.data;
    const saved = app.net ? null : {
      paused: app.paused,
      ...Object.fromEntries(['qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat', 'qacc_warmstart']
        .map((k) => [k, d[k] && Float64Array.from(d[k])])),
      time: d.time,
    };
    if (saved) { app.paused = true; app.showPaused?.(); }
    this.replay = { t: this.clip.samples[0].t, playing: false, last: performance.now(), saved };
    this.show();
  }

  // Back to the live world (the local one as it was, paused or not).
  live() {
    const r = this.replay, app = this.app;
    if (!r) return;
    this.replay = null;
    if (r.saved) {
      const d = app.data;
      for (const [k, v] of Object.entries(r.saved)) if (v instanceof Float64Array) d[k].set(v);
      d.time = r.saved.time;
      app.mujoco.mj_forward(app.model, d);
      app.syncTransforms();
      app.paused = r.saved.paused;
      app.showPaused?.();
      if (!app.paused) app.resync();
    }
    this.show();
  }

  // The bodies as recorded at clip time `t`.
  place(t) {
    const { app } = this, s = this.clip.samples, d = app.data, model = app.model;
    let i = 1;
    while (i < s.length - 1 && s[i].t < t) i++;
    const a = s[i - 1], b = s[i];
    const alpha = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 1;
    lerp(d.qpos, a.qpos, b.qpos, alpha);
    for (const k of quatSlots(model)) blendQuat(d.qpos, k, a.qpos, b.qpos, alpha);
    lerp(d.mocap_pos, a.mocap_pos, b.mocap_pos, alpha);
    for (let k = 0; k < model.nmocap; k++) blendQuat(d.mocap_quat, 4 * k, a.mocap_quat, b.mocap_quat, alpha);
    d.time = a.time + alpha * (b.time - a.time);
    if (app.net) forwardKinematics(app.robots, d);
    else app.mujoco.mj_forward(model, d);
    app.syncTransforms();
  }

  show() {
    const { button, bar, ui } = this, clip = this.clip, r = this.replay;
    button.toggleAttribute('data-recording', this.recording);
    button.querySelector('.label').textContent = this.recording ? 'Stop' : 'Record';
    button.title = this.recording ? 'Stop recording (then replay it in the bar below)'
      : 'Record the simulation for replay (this tab only, in memory: gone on reload or when the world changes)';
    bar.hidden = !clip && !this.note;
    bar.toggleAttribute('data-replaying', !!r);
    for (const k of ['play', 'seek', 'speed', 'live', 'time']) ui[k].hidden = !clip;
    ui.discard.hidden = !clip && !this.note;
    ui.play.disabled = ui.seek.disabled = this.recording;
    ui.play.textContent = r?.playing ? '❚❚' : '▶';
    ui.play.title = r?.playing ? 'Pause the replay' : 'Replay';
    ui.live.hidden = !r;
    ui.note.textContent = this.note ?? '';
    ui.note.hidden = !this.note || !!clip;
    if (clip) {
      ui.seek.min = clip.samples[0].t;
      ui.seek.step = REC_DT;
    }
    this.showTime();
  }

  showTime() {
    const { ui, clip, replay: r } = this;
    if (!clip) return;
    const end = this.length, from = clip.samples[0].t;
    ui.seek.max = end;
    if (r) {
      ui.seek.value = r.t;
      ui.time.textContent = `${(r.t - from).toFixed(1)} / ${(end - from).toFixed(1)} s`;
      if (!r.playing && ui.play.textContent !== '▶') ui.play.textContent = '▶';
    } else {
      ui.seek.value = end;
      ui.time.textContent = this.recording ? `● ${(end - from).toFixed(1)} s` : `${(end - from).toFixed(1)} s recorded`;
    }
  }
}
