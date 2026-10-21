// The WebXR menu (xr.js opens it): a panel floating in front of the viewer,
// drawn on a 2D canvas and shown as a texture (App.xrPanel), pointed at with
// a controller's ray and chosen with its trigger.
//
// A menu is {title, items} (items: an array, or a function returning one,
// read every time the panel is drawn). An item is {label, value?, checked?,
// disabled?, action?, menu?}: label and value are strings or functions
// returning one (value shows on the right), checked and disabled functions
// returning a boolean, info a row that only shows its value; `action()` runs when it is chosen (returning true
// closes the menu), `menu` (a menu, or a function returning one) opens a
// submenu, which gets a "‹ Back" row of its own.

const PX = 512, PY = 768;           // canvas pixels
const WIDTH = 0.34;                 // m: panel width (height in proportion)
const HEIGHT = WIDTH * PY / PX;
const DIST = 0.55, DROP = 0.12;     // m: in front of the head, and below it
const HEAD = 76, ROW = 60, PAD = 14;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const text = (v) => (typeof v === 'function' ? v() : v) ?? '';

export class XrMenu {
  // `panel(px, py, width, height)`: a hidden panel entity, {put(matrix),
  // upload(canvas)} (App.xrPanel), made the first time the menu opens.
  constructor(panel) {
    this.makePanel = panel;
    this.panel = null;
    this.stack = [];   // open menus, the root first
    this.hot = -1;     // row under a ray
    this.canvas = document.createElement('canvas');
    this.canvas.width = PX;
    this.canvas.height = PY;
  }

  get open() { return this.stack.length > 0; }

  // Opens `menu` in front of the head (`head`: {pos, rot} in the scene).
  show(menu, head) {
    this.panel ??= this.makePanel(PX, PY, WIDTH, HEIGHT);
    this.stack = [menu];
    this.hot = -1;
    // Facing the head, upright, along where it looks (level).
    const f0 = [-head.rot[2], -head.rot[5], 0], n = Math.hypot(f0[0], f0[1]) || 1;
    const f = [f0[0] / n, f0[1] / n, 0], up = [0, 0, 1], right = [f[1], -f[0], 0];
    this.pos = [head.pos[0] + f[0] * DIST, head.pos[1] + f[1] * DIST, head.pos[2] - DROP];
    this.axes = { right, up, normal: [-f[0], -f[1], 0] };
    this.panel.put([...right, 0, ...up, 0, ...this.axes.normal, 0, ...this.pos, 1]);
    this.draw();
  }

  close() {
    this.stack = [];
    this.panel?.put(null);
  }

  menu() { return this.stack.at(-1); }

  // The current menu's rows: its items, then Back in a submenu.
  rows() {
    const m = this.menu();
    const items = [...(typeof m.items === 'function' ? m.items() : m.items)];
    if (this.stack.length > 1) items.push({ label: '‹ Back', back: true });
    return items;
  }

  // The row a ray {eye, dir} points at: {row (-1 off the rows), depth}, or
  // null if it misses the panel.
  hit({ eye, dir }) {
    if (!this.open) return null;
    const { right, up, normal } = this.axes, den = dot(dir, normal);
    if (den > -1e-6) return null;  // from behind, or along it
    const depth = dot(sub(this.pos, eye), normal) / den;
    if (depth <= 0) return null;
    const d = sub(eye.map((v, i) => v + depth * dir[i]), this.pos);
    const u = dot(d, right), v = dot(d, up);
    if (Math.abs(u) > WIDTH / 2 || Math.abs(v) > HEIGHT / 2) return null;
    const y = (HEIGHT / 2 - v) / HEIGHT * PY;
    const row = Math.floor((y - HEAD) / ROW), rows = this.rows();
    return { row: y >= HEAD && row < rows.length ? row : -1, depth };
  }

  // Redraws once a second while open, for values that change by themselves.
  tick() {
    if (this.open && performance.now() - (this.drawn ?? 0) > 1000) this.draw();
  }

  hover(row) {
    if (!this.open) { this.hot = -1; return; }
    if (this.rows()[row]?.info) row = -1;
    if (row === this.hot) return;
    this.hot = row;
    this.draw();
  }

  // Chooses a row.
  select(row) {
    const item = this.rows()[row];
    if (!item || item.info || item.disabled?.()) return;
    if (item.back) this.stack.pop();
    else if (item.menu) this.stack.push(typeof item.menu === 'function' ? item.menu() : item.menu);
    else if (item.action?.()) { this.close(); return; }
    this.hot = -1;
    this.draw();
  }

  draw() {
    const g = this.canvas.getContext('2d'), rows = this.rows();
    g.clearRect(0, 0, PX, PY);
    const h = HEAD + rows.length * ROW + PAD;
    g.fillStyle = 'rgba(22, 24, 29, 0.94)';
    g.beginPath();
    g.roundRect(2, 2, PX - 4, h - 4, 22);
    g.fill();
    g.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    g.lineWidth = 2;
    g.stroke();
    g.textBaseline = 'middle';
    g.fillStyle = '#9aa4b2';
    g.font = '600 26px system-ui, sans-serif';
    g.fillText(this.stack.map((m) => m.title).join(' › '), 24, HEAD / 2 + 4, PX - 48);
    rows.forEach((item, i) => {
      const y = HEAD + i * ROW, off = item.disabled?.();
      if (i === this.hot && !off) {
        g.fillStyle = 'rgba(80, 160, 255, 0.35)';
        g.beginPath();
        g.roundRect(10, y + 3, PX - 20, ROW - 6, 12);
        g.fill();
      }
      g.fillStyle = off ? '#5c636e' : '#e8ecf2';
      g.font = '500 30px system-ui, sans-serif';
      const mark = item.checked ? (item.checked() ? '✓' : '') : null;
      const x = mark !== null ? 64 : 24;
      if (mark) g.fillText(mark, 26, y + ROW / 2);
      g.fillText(text(item.label), x, y + ROW / 2, PX - x - 120);
      const value = item.menu ? `${text(item.value)}  ›`.trim() : text(item.value);
      if (value) {
        g.fillStyle = off ? '#5c636e' : '#9aa4b2';
        g.textAlign = 'right';
        g.fillText(value, PX - 26, y + ROW / 2);
        g.textAlign = 'left';
      }
    });
    this.panel.upload(this.canvas);
    this.drawn = performance.now();
  }
}
