// A gamepad (the Gamepad API's standard mapping: Xbox / PlayStation layout)
// read through bindings of actions to stick axes and buttons:
// public/json/gamepad_defaults.json, changed in the Teleop panel (Input:
// Gamepad) and kept in this browser. A binding is 'axisN' (stick axis N,
// 'axis0'/'axis1' the left stick's x/y, 'axis2'/'axis3' the right one's, y
// growing downward), '-axisN' (reversed) or 'buttonN'; '' is none.

const STORE = 'mujoco-filament/gamepad';
const FALLBACK = {
  deadzone: 0.15, forward: '-axis1', left: '-axis0', turn: '-axis2',
  height_up: 'button12', height_down: 'button13', gripper1: 'button6', gripper2: 'button7',
};
const FILE = await fetch(new URL('../json/gamepad_defaults.json', import.meta.url)).then((r) => r.json()).catch(() => FALLBACK);
const { _comment: COMMENT, ...defaults } = FILE;
export const GAMEPAD_DEFAULTS = defaults;

// [action, label, title]; walk and turn take -1..1, the grippers 0..1, the
// height actions count while held.
export const ACTIONS = [
  ['forward', 'Walk forward', 'forward/back: -1 (back) .. 1 (forward)'],
  ['left', 'Walk left', 'sideways: -1 (right) .. 1 (left)'],
  ['turn', 'Turn left', 'yaw: -1 (right) .. 1 (left)'],
  ['height_up', 'Height up', 'raises the commanded base height while held'],
  ['height_down', 'Height down', 'lowers the commanded base height while held'],
  ['gripper1', 'Gripper 1', 'closes the first IK chain\'s gripper as far as pressed'],
  ['gripper2', 'Gripper 2', 'closes the second IK chain\'s gripper as far as pressed'],
];

const BUTTONS = ['A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT', 'Back', 'Start', 'Left stick press', 'Right stick press',
  'D-pad up', 'D-pad down', 'D-pad left', 'D-pad right', 'Home'];
const AXES = [['Left stick →', 'Left stick ←'], ['Left stick ↓', 'Left stick ↑'], ['Right stick →', 'Right stick ←'], ['Right stick ↓', 'Right stick ↑']];

// [binding, label] for a select.
export const BINDINGS = [['', 'none'],
  ...AXES.flatMap(([pos, neg], i) => [[`-axis${i}`, neg], [`axis${i}`, pos]]),
  ...BUTTONS.map((name, i) => [`button${i}`, name])];
export const bindingName = (b) => BINDINGS.find(([v]) => v === b)?.[1] ?? b;

// The first connected gamepad with the standard mapping, or null.
export function gamepad() {
  return [...(navigator.getGamepads?.() ?? [])].find((g) => g?.connected && g.mapping === 'standard') ?? null;
}

export class GamepadMap {
  constructor() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(STORE)); } catch { /* no storage */ }
    this.map = { ...GAMEPAD_DEFAULTS, ...(saved ?? {}) };
    this.saved = !!saved;
  }

  set(action, binding) {
    this.map[action] = binding;
    this.save();
  }

  save() {
    try { localStorage.setItem(STORE, JSON.stringify(this.map)); } catch { /* no storage */ }
    this.saved = true;
  }

  reset() {
    this.map = { ...GAMEPAD_DEFAULTS };
    try { localStorage.removeItem(STORE); } catch { /* no storage */ }
    this.saved = false;
  }

  // Action `action`'s value on gamepad `gp`: an axis -1..1 (0 within the
  // dead zone of its stick), a button 0..1.
  value(gp, action) {
    const b = this.map[action];
    const m = b && b.match(/^(-?)(axis|button)(\d+)$/);
    if (!m) return 0;
    const i = +m[3];
    if (m[2] === 'button') return gp.buttons[i]?.value ?? +!!gp.buttons[i]?.pressed;
    const pair = i ^ 1;  // the stick's other axis: a round dead zone
    if (Math.hypot(gp.axes[i] ?? 0, gp.axes[pair] ?? 0) < (this.map.deadzone ?? 0)) return 0;
    return (m[1] ? -1 : 1) * (gp.axes[i] ?? 0);
  }

  // The binding of whatever on `gp` is pressed or pushed now that was not in
  // `before` (a snapshot()), or null.
  detect(gp, before) {
    const b = gp.buttons.findIndex((x, i) => (x.value ?? +x.pressed) > 0.5 && !before.buttons[i]);
    if (b >= 0) return `button${b}`;
    const a = gp.axes.findIndex((v, i) => Math.abs(v) > 0.6 && Math.abs(before.axes[i] ?? 0) < 0.3);
    return a >= 0 ? `${gp.axes[a] < 0 ? '-' : ''}axis${a}` : null;
  }

  static snapshot(gp) {
    return { buttons: gp ? gp.buttons.map((x) => (x.value ?? +x.pressed) > 0.5) : [], axes: gp ? [...gp.axes] : [] };
  }

  // The bindings as JSON for gamepad_defaults.json.
  toJSON() {
    return { _comment: COMMENT, ...this.map };
  }
}
