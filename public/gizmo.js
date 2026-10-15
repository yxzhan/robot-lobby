// World-axes gizmo: a small 2D canvas in a corner showing the world X (red),
// Y (green) and Z (blue, up) axes as seen by the orbit camera. Clicking an
// axis end looks along it (from the +/- side clicked), keeping the target.

const AXES = [
  ['X', [1, 0, 0], '#e5484d'],
  ['Y', [0, 1, 0], '#46a758'],
  ['Z', [0, 0, 1], '#3e8ef7'],
];

export class AxisGizmo {
  constructor(canvas, orbit) {
    this.canvas = canvas;
    this.orbit = orbit;
    this.ends = [];  // screen positions of the axis ends, for clicks
    this.last = '';
    canvas.addEventListener('click', (e) => this.click(e));
    canvas.addEventListener('pointermove', (e) => {
      canvas.style.cursor = this.hit(e) ? 'pointer' : 'default';
    });
  }

  // Camera basis like App.pointerRay: forward, right, up (world frame).
  basis() {
    const { azim, elev } = this.orbit;
    const ce = Math.cos(elev), se = Math.sin(elev), ca = Math.cos(azim), sa = Math.sin(azim);
    const f = [-ce * ca, -ce * sa, -se];
    const r = [-sa, ca, 0];  // f × z, normalized
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    return { f, r, u };
  }

  // Redraws when the view direction changed; cheap to call every frame.
  draw() {
    const { azim, elev } = this.orbit;
    const dpr = window.devicePixelRatio || 1, size = this.canvas.clientWidth;
    const key = `${azim.toFixed(4)},${elev.toFixed(4)},${size},${dpr}`;
    if (key === this.last) return;
    this.last = key;
    const c = this.canvas;
    c.width = c.height = Math.round(size * dpr);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);
    const mid = size / 2, len = size * 0.34, rad = size * 0.085;
    g.fillStyle = 'rgba(0,0,0,.35)';
    g.beginPath();
    g.arc(mid, mid, size / 2 - 1, 0, 2 * Math.PI);
    g.fill();

    const { f, r, u } = this.basis();
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    this.ends = [];
    for (const [name, v, color] of AXES) {
      for (const sign of [1, -1]) {
        const w = v.map((k) => k * sign);
        this.ends.push({
          name, sign, color, v: w,
          x: mid + dot(w, r) * len, y: mid - dot(w, u) * len,
          depth: dot(w, f),  // > 0: pointing away from the camera
        });
      }
    }
    // Back to front, so nearer axes are drawn over farther ones.
    for (const e of [...this.ends].sort((a, b) => b.depth - a.depth)) {
      if (e.sign > 0) {
        g.strokeStyle = e.color;
        g.lineWidth = 2.5;
        g.beginPath();
        g.moveTo(mid, mid);
        g.lineTo(e.x, e.y);
        g.stroke();
      }
      g.globalAlpha = e.sign > 0 ? 1 : 0.45;
      g.fillStyle = e.color;
      g.beginPath();
      g.arc(e.x, e.y, e.sign > 0 ? rad : rad * 0.7, 0, 2 * Math.PI);
      g.fill();
      g.globalAlpha = 1;
      if (e.sign > 0) {
        g.fillStyle = '#fff';
        g.font = `600 ${Math.round(rad * 1.3)}px system-ui, sans-serif`;
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillText(e.name, e.x, e.y + 0.5);
      }
    }
  }

  // The axis end under the pointer, nearest to the camera first.
  hit(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top, rad = rect.width * 0.1;
    return [...this.ends].sort((a, b) => a.depth - b.depth)
      .find((p) => Math.hypot(p.x - x, p.y - y) <= rad) ?? null;
  }

  // Puts the camera on the clicked side of the axis, looking at the target.
  // The orbit camera keeps |elev| <= 1.5, so Z views are almost top-down.
  click(e) {
    const p = this.hit(e);
    if (!p) return;
    const o = this.orbit;
    if (p.v[2]) {
      o.elev = 1.5 * p.v[2];
    } else {
      o.elev = 0;
      o.azim = Math.atan2(p.v[1], p.v[0]);
    }
  }
}
