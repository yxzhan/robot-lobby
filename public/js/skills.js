// Policies of more than one model, the same for every policy controller
// (GrootController, MicroduckController) and on the server (Skills in
// server/policies.py): the policy directory's policy.json lists the models
// as "skills", sharing the controller's observation and actions, and Skills
// picks the one to run every control tick:
//
//   "skills": {
//     "walk":    {"onnx": "walk.onnx", "label": "Walk"},
//     "balance": {"onnx": "balance.onnx", "when": {"command_below": 0.05}},
//     "kick":    {"onnx": "kick.onnx", "label": "Kick", "duration": 0.3, "key": "k", "commands": false}
//   }
//
// - The first is the default.
// - One with `when` is automatic: it runs instead of the default while its
//   condition holds (the first of them that does); no one picks it.
//   Conditions: command_below: v, the drive twist's norm |(vx, vy, yaw
//   rate)| under v.
// - The others, the default too, are manual: picked with setSkill (on the
//   controller; on the server a `skill` request), the Teleop panel's skill
//   controls, given `skillMenu` when there are two or more: a button for
//   each one with a duration (a kick), a select of the others, keys
//   (`key`, a skill's default key; the panel can change it). Skills are not
//   commands (commands.js): those are inputs of the model running; a skill
//   picks the model.
//   One with a `duration` runs that long (simulation time: the same physics
//   steps at any speed; a pick may ask for another, as the panel does) and
//   then the default again, one
//   without until another is picked; automatic skills only stand in for the
//   default.
// - reset_action: true: trained from episodes whose action history starts
//   at zero; the controller's last action is zeroed when it starts (`started`),
//   or it reads the previous model's last action as its own (the kick then
//   stood still after the walking policy's stand).
// - commands: false: trained with zeros for the twist and the commands (the
//   controller observes it so).
// - twist: [vx, vy, yaw rate], what the model sees in the twist slot while it
//   runs instead of the drive command (e.g. Pollen's sitstand: [1, 0, 0] is
//   its posture flag, sit); phase: {period} (s), the twist slot as a phase
//   clock instead, [cos 2πφ, sin 2πφ, 0] with φ = time since it started /
//   period (Pollen's ground pick). With commands: false the other commands
//   are zeros.
// - label, about: for the panel.
// A controller with skills has `skills` (a Skills) and, for the panel and
// the network: skillMenu (menu), skill (the manual one picked, `want`),
// setSkill(name, duration?) (request), activeSkill (the one running when
// there is a choice, `shown`).

// The durations a pick may ask for (s).
export const DURATION_RANGE = [0.1, 60];

const CONDITIONS = {
  command_below: (v, c) => Math.hypot(...c.cmd) < v,
};

// {name: session} of `skills`; `source(name)` is a file's URL (or bytes).
export async function loadSessions(ort, source, skills) {
  const entries = await Promise.all(Object.entries(skills).map(
    async ([name, s]) => [name, await ort.InferenceSession.create(await source(s.onnx))]));
  return Object.fromEntries(entries);
}

// A policy directory's policy.json (or another spec file of it), its skills checked.
export async function loadSpec(source, file = 'policy.json') {
  const res = await fetch(await source(file));
  if (!res.ok) throw new Error(`${file}: ${res.status} ${res.statusText}`);
  const spec = await res.json();
  if (!spec.skills || !Object.keys(spec.skills).length) throw new Error(`${file}: no skills`);
  return spec;
}

export class Skills {
  constructor(skills) {
    this.specs = skills;
    const names = Object.keys(skills);
    this.default = names[0];
    for (const [name, s] of Object.entries(skills)) {
      for (const k of Object.keys(s.when ?? {})) {
        if (!CONDITIONS[k]) throw new Error(`skill ${name}: no condition ${k} (${Object.keys(CONDITIONS).join(', ')})`);
      }
    }
    if (skills[this.default].when) throw new Error(`skill ${this.default}: the default cannot be automatic`);
    this.manual = names.filter((n) => !skills[n].when);
    this.auto = names.filter((n) => skills[n].when);
    this.reset();
  }

  // What the panel needs to pick (also sent to browsers in network mode):
  // {default, manual: [names], skills: {name: {label, about?, duration?,
  // key?}}}, or null without a choice.
  get menu() {
    const m = this.manual;
    if (m.length < 2) return null;
    const skills = Object.fromEntries(m.map((n) => {
      const { label, about, duration, key } = this.specs[n];
      return [n, { label: label ?? n, ...(about && { about }), ...(duration && { duration }), ...(key && { key }) }];
    }));
    return { default: this.default, manual: m, skills };
  }

  // Picks manual skill `name`; it runs from the next tick. A timed one runs
  // `duration` seconds if given (within DURATION_RANGE), else its own.
  request(name, duration = null) {
    if (!this.manual.includes(name)) throw new Error(`no skill ${name} to pick (${this.manual.join(', ')})`);
    this.want = name;
    this.wantFor = this.specs[name].duration && Number.isFinite(duration) && duration > 0
      ? Math.min(DURATION_RANGE[1], Math.max(DURATION_RANGE[0], duration)) : null;
    if (this.picked?.[0] === name) this.picked = null;  // picked again: from now
  }

  // Forget what ran (robot reset): the default again.
  reset() {
    this.want = this.default;
    this.wantFor = null;     // the duration asked for with it, if any
    this.picked = null;  // the manual skill running, and since when: [name, data.time]
    this.active = this.default;
  }

  // The skill to run at `time` for controller `c` (its cmd, for the
  // conditions). `time` is simulation time (data.time): a duration lasts as
  // many physics steps at any speed of the Physics panel.
  pick(c, time) {
    const want = this.want;
    this.started = null;     // the manual skill that starts this tick, if one does
    if (want !== this.picked?.[0] || time < this.picked[1]) {  // time went back: anew
      if (want !== this.picked?.[0] && want !== this.default) this.started = want;
      this.picked = [want, time];
    }
    const d = this.specs[want]?.duration && (this.wantFor ?? this.specs[want].duration);
    if (d && time - this.picked[1] >= d - 1e-9) {
      this.want = this.default;
      this.wantFor = null;
      this.picked = [this.default, time];
    }
    let run = this.picked[0];
    if (run === this.default) {
      run = this.auto.find((n) => Object.entries(this.specs[n].when).every(([k, v]) => CONDITIONS[k](v, c))) ?? run;
    }
    this.active = run;
    return run;
  }

  get shown() {
    return Object.keys(this.specs).length > 1 ? this.active : null;
  }

  // Whether the skill starting this tick starts from a zero action history.
  get freshStart() {
    return !!(this.started && this.specs[this.started].reset_action);
  }

  blind(name) {
    return this.specs[name]?.commands === false;
  }

  // The twist skill `name` is shown at `time` (twist or phase), or null: the drive command.
  twistFor(name, time) {
    const s = this.specs[name];
    if (s?.twist) return s.twist;
    if (!s?.phase) return null;
    const since = this.picked?.[0] === name ? time - this.picked[1] : 0;
    const a = 2 * Math.PI * since / s.phase.period;
    return [Math.cos(a), Math.sin(a), 0];
  }
}
