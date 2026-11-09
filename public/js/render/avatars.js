// The avatars (world/avatar.js) drawn where their bodies are, as they collide:
// a head (a ball with a dark visor the way its owner looks) and two hands (the
// palm's prism, a cylinder per finger bone), in the owner's color (a shared
// world: its client's, net/control.js). A slot whose bodies are all parked is
// hidden, and so is this page's own avatar while it is in VR (it sees its
// hands as their skeleton, xr.js, and has no use for its own head). Shapes are
// made the first time a slot is used (App.xrPart: a unit ball / rod, or the
// palm's shape, on OVERLAY).

import { HAND_BONES, HAND_PARTS, HEAD_RADIUS, PALM, PALM_HALF, PARTS, SIDES, parked } from '../world/avatar.js';

const ZERO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
const VISOR = [0.075, 0.045, 0.03];  // m: the visor's half axes (sideways, up, depth)
const VISOR_COLOR = [0.08, 0.09, 0.12, 1];
const OWN_COLOR = '#4fb3ff';
const hexRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const BONE = new Map(HAND_BONES.map(([part, , , r, length]) => [part, { r, half: length / 2 }]));

// The palm's prism as a flat-shaded shape {pos, nrm, idx} in its body's frame.
const PALM_SHAPE = (() => {
  const v = (k, z) => [PALM[k][0], PALM[k][1], z], lo = -PALM_HALF, hi = PALM_HALF, pos = [], nrm = [], idx = [];
  const face = (corners) => {  // a flat polygon, its normal from its first three corners
    const [a, b, c] = corners, u = b.map((x, i) => x - a[i]), w = c.map((x, i) => x - a[i]);
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]], l = Math.hypot(...n) || 1;
    const base = pos.length / 3;
    for (const p of corners) { pos.push(...p); nrm.push(...n.map((x) => x / l)); }
    for (let i = 1; i < corners.length - 1; i++) idx.push(base, base + i, base + i + 1);
  };
  // PALM goes clockwise seen from +z (wrist, index side, little finger side): the top face reversed.
  face([v(0, hi), v(2, hi), v(1, hi)]);
  face([v(0, lo), v(1, lo), v(2, lo)]);
  for (let k = 0; k < 3; k++) {
    const j = (k + 1) % 3;
    face([v(k, hi), v(j, hi), v(j, lo), v(k, lo)]);  // (outward: PALM runs clockwise)
  }
  return { pos, nrm, idx };
})();

export class AvatarView {
  constructor(app) {
    this.app = app;
    this.slots = [];  // slot -> {head, visor, hands: {side: {palm, bones: [part]}}, color, shown}
  }

  // Every time the bodies moved (App.syncTransforms).
  sync() {
    const app = this.app, ids = app.avatarSlots(), d = app.data;
    if (!d?.mocap_pos) return;
    const owners = app.avatarOwners(), own = app.xr ? app.ownAvatarSlot() : null;
    ids.forEach((mocap, s) => {
      const pos = (i) => d.mocap_pos.subarray(3 * mocap[i], 3 * mocap[i] + 3);
      const active = s !== own && owners[s] != null && mocap.some((_, i) => !parked(pos(i)));
      const v = this.slots[s];
      if (!active) {
        if (v?.shown) this.hide(v);
        return;
      }
      const view = v ?? (this.slots[s] = this.make());
      this.color(view, owners[s]);
      this.place(view, mocap, d);
    });
  }

  make() {
    const part = (shape) => this.app.xrPart(shape);
    const visor = part(false);
    visor.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, VISOR_COLOR);
    visor.mi.setFloatParameter('roughness', 0.15);
    const hands = Object.fromEntries(SIDES.map((side) => [side, { palm: part(PALM_SHAPE), bones: HAND_BONES.map(() => part(true)) }]));
    return { head: part(false), visor, hands, color: null, shown: false };
  }

  all(v) {
    return [v.head, ...SIDES.flatMap((s) => [v.hands[s].palm, ...v.hands[s].bones])];
  }

  color(v, owner) {
    const client = this.app.net ? this.app.control?.clients.get(owner) : null;
    const hex = client?.color ?? OWN_COLOR;
    if (v.color === hex) return;
    v.color = hex;
    const rgba = [...hexRgb(hex), 1];
    for (const p of this.all(v)) p.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, rgba);
  }

  put(p, m) {
    const tcm = this.app.engine.getTransformManager(), inst = tcm.getInstance(p.e);
    tcm.setTransform(inst, m ?? ZERO);
    inst.delete();
  }

  hide(v) {
    for (const p of [v.visor, ...this.all(v)]) this.put(p, null);
    v.shown = false;
  }

  place(v, mocap, d) {
    v.shown = true;
    const index = new Map(PARTS.map((p, i) => [p, i]));
    const at = (i) => Array.from(d.mocap_pos.subarray(3 * mocap[i], 3 * mocap[i] + 3));
    const cols = (i) => quatColumns(d.mocap_quat.subarray(4 * mocap[i], 4 * mocap[i] + 4));
    const pose = (c, o, s = [1, 1, 1]) => [...c[0].map((a) => a * s[0]), 0, ...c[1].map((a) => a * s[1]), 0, ...c[2].map((a) => a * s[2]), 0, ...o, 1];
    // The head, and its visor in front (the view's -z).
    const h = index.get('head'), head = at(h);
    if (parked(head)) { this.put(v.head, null); this.put(v.visor, null); } else {
      const c = cols(h);
      this.put(v.head, pose([[1, 0, 0], [0, 1, 0], [0, 0, 1]], head, [HEAD_RADIUS, HEAD_RADIUS, HEAD_RADIUS]));
      this.put(v.visor, pose(c, head.map((x, i) => x - 0.8 * HEAD_RADIUS * c[2][i]), VISOR));
    }
    // The hands: the palm's prism, each bone's cylinder along its body's z.
    for (const side of SIDES) {
      const parts = v.hands[side];
      HAND_PARTS.forEach((part, k) => {
        const i = index.get(`${side}_${part}`), o = at(i), p = k === 0 ? parts.palm : parts.bones[k - 1];
        if (parked(o)) { this.put(p, null); return; }
        const c = cols(i);
        if (k === 0) { this.put(p, pose(c, o)); return; }
        const { r, half } = BONE.get(part);
        this.put(p, pose(c, o, [r, r, 2 * half]));  // (the unit rod: radius 1, length 1 along z)
      });
    }
  }
}

// A quaternion [w, x, y, z]'s rotation as its three columns.
function quatColumns([w, x, y, z]) {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
    [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
  ];
}
