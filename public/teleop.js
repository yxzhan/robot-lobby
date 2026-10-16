// Teleoperation: the Teleop panel picks which robot is driven, by which
// controller and from which input; every other controllable robot gets a
// zero command (it stands). For now the only controller is the GR00T
// locomotion policy (groot.js, on robot.groot) and the only input the
// keyboard; the selects are where other policies, end-effector control, a
// ROS bridge or a gamepad would plug in.

// Velocity command per held key: [command index (vx, vy, yaw rate), value].
const DRIVE = { w: [0, 0.5], s: [0, -0.3], a: [1, 0.3], d: [1, -0.3], q: [2, 0.6], e: [2, -0.6] };
const HEIGHT_RATE = 0.15;  // m/s while R/F is held
const HEIGHT_RANGE = [0.5, 1.0];

// Keys typed into a text field are not commands.
function isTyping(target) {
  if (!(target instanceof Element)) return false;
  if (target.closest('textarea, select, [contenteditable]:not([contenteditable=false])')) return true;
  return target.tagName === 'INPUT' && !['range', 'checkbox', 'radio', 'button', 'submit'].includes(target.type);
}

// WASD walk, QE turn, held R/F raise/lower the commanded base height.
export class KeyboardInput {
  constructor() {
    this.held = new Set();
    const onKey = (e) => {
      const k = e.key.toLowerCase();
      if (!(k in DRIVE) && k !== 'r' && k !== 'f') return;
      if (e.type === 'keyup') { this.held.delete(k); return; }
      if (!this.enabled || e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return;
      e.preventDefault();
      this.held.add(k);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    window.addEventListener('blur', () => this.held.clear());
    this.enabled = false;  // only while a robot is being driven
  }

  // {cmd: [vx, vy, yaw rate], dh: height change} for the last `dt` seconds.
  read(dt) {
    const cmd = [0, 0, 0];
    for (const k of this.held) if (DRIVE[k]) cmd[DRIVE[k][0]] += DRIVE[k][1];
    return { cmd, dh: (this.held.has('r') - this.held.has('f')) * HEIGHT_RATE * dt };
  }
}

export class TeleopPanel {
  constructor(app, el) {
    this.app = app;
    this.el = el;
    this.body = el.querySelector('.body');
    this.active = null;      // the driven robot
    this.known = new Set();  // controllable robots seen so far; a new one gets selected
    this.input = new KeyboardInput();
    this.readout = null;
    let last = performance.now();
    const loop = () => {
      const now = performance.now();
      this.tick((now - last) / 1000);
      last = now;
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  controllable() {
    return (this.app.robots ?? []).filter((r) => r.groot);
  }

  // Called for every new world (App.setWorld).
  build() {
    const robots = this.controllable();
    const added = robots.filter((r) => !this.known.has(r));
    this.known = new Set(robots);
    if (added.length) this.active = added.at(-1);
    else if (!robots.includes(this.active)) this.active = robots.at(-1) ?? null;
    this.render();
  }

  // Drives `robot` (e.g. double-clicked) if it is controllable; anything
  // else (null, a robot without a controller) releases control.
  drive(robot) {
    const next = robot?.groot ? robot : null;
    if (next === this.active) return;
    this.active = next;
    this.render();
  }

  render() {
    const robots = this.controllable();
    this.body.replaceChildren();
    this.readout = null;
    if (!robots.length) {
      this.body.innerHTML = '<div class="hint">Nothing to drive: spawn a Unitree G1 (Robots › Humanoids), which walks under the GR00T policy.</div>';
      return;
    }
    const field = (label, control) => {
      const row = document.createElement('label');
      row.className = 'field';
      row.append(Object.assign(document.createElement('span'), { textContent: label }), control);
      this.body.append(row);
      return control;
    };
    const select = (options, value, onchange) => {
      const s = document.createElement('select');
      for (const [v, text] of options) s.append(new Option(text, v, false, v === value));
      s.onchange = () => { onchange?.(s.value); s.blur(); };  // blur: keys drive again
      s.disabled = options.length < 2;
      return s;
    };
    field('Robot', select(
      [['', 'none'], ...robots.map((r) => [r.prefix, `${r.prefix.slice(0, -1)} · ${r.label}`])],
      this.active?.prefix ?? '',
      (v) => { this.active = robots.find((r) => r.prefix === v) ?? null; this.render(); },
    ));
    if (!this.active) return;
    field('Controller', select([['groot', 'GR00T locomotion policy']], 'groot'));
    field('Input', select([['keyboard', 'Keyboard']], 'keyboard'));
    const keys = document.createElement('div');
    keys.className = 'hint';
    keys.innerHTML = '<kbd>W</kbd><kbd>S</kbd> forward/back · <kbd>A</kbd><kbd>D</kbd> sideways · '
      + '<kbd>Q</kbd><kbd>E</kbd> turn · hold <kbd>R</kbd><kbd>F</kbd> height';
    this.body.append(keys);
    this.readout = document.createElement('div');
    this.readout.className = 'readout';
    this.body.append(this.readout);
  }

  // Every frame: the active robot gets the input's command, the others stand.
  tick(dt) {
    const robots = this.controllable();
    this.input.enabled = !!this.active;
    const { cmd, dh } = this.input.read(dt);
    for (const r of robots) {
      const c = r.groot;
      if (r === this.active) {
        c.cmd = cmd;
        c.height = Math.min(HEIGHT_RANGE[1], Math.max(HEIGHT_RANGE[0], c.height + dh));
      } else {
        c.cmd = [0, 0, 0];
      }
    }
    if (this.readout && this.active) {
      const c = this.active.groot;
      const text = `vx ${c.cmd[0].toFixed(1)}  vy ${c.cmd[1].toFixed(1)}  ωz ${c.cmd[2].toFixed(1)}  height ${c.height.toFixed(2)} m`;
      if (this.readout.textContent !== text) this.readout.textContent = text;
    }
  }
}
