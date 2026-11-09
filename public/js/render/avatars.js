// The avatars (world/avatar.js) drawn where their bodies are: a head (a ball
// with a dark visor the way its owner looks), a torso and legs (the humanoid's
// capsules) and two hands (a ball per joint, rods along the fingers), in the
// owner's color (a shared world: its client's, net/control.js). A slot whose
// bodies are all parked is hidden; this page's own avatar in VR shows only its
// torso and legs (its hands are their skeleton, xr.js; its head would be in
// the way). Shapes are made the
// first time a slot is used (App.xrPart: a unit ball / rod on OVERLAY).

import { BONES, CAPSULES, HEAD_RADIUS, PARTS, SIDES, parked, radius } from '../world/avatar.js';

const HANDS = PARTS.filter((p) => p.includes('_'));  // the hand joints' parts
const BODY_PARTS = Object.keys(CAPSULES);           // torso, legs

const ZERO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
const VISOR = [0.075, 0.045, 0.03];  // m: the visor's half axes (sideways, up, depth)
const VISOR_COLOR = [0.08, 0.09, 0.12, 1];
const BONE_RADIUS = 0.005;
const OWN_COLOR = '#4fb3ff';
const hexRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);

export class AvatarView {
  constructor(app) {
    this.app = app;
    this.slots = [];  // slot -> {head, visor, balls: [part], bones: [part], color, shown}
  }

  // Every time the bodies moved (App.syncTransforms).
  sync() {
    const app = this.app, ids = app.avatarSlots(), d = app.data;
    if (!d?.mocap_pos) return;
    const owners = app.avatarOwners(), own = app.xr ? app.ownAvatarSlot() : null;
    ids.forEach((mocap, s) => {
      const pos = (i) => d.mocap_pos.subarray(3 * mocap[i], 3 * mocap[i] + 3);
      const active = owners[s] != null && mocap.some((_, i) => !parked(pos(i)));
      const v = this.slots[s];
      if (!active) {
        if (v?.shown) this.hide(v);
        return;
      }
      const view = v ?? (this.slots[s] = this.make());
      this.color(view, owners[s]);
      this.place(view, mocap, d, s === own);
    });
  }

  make() {
    const part = (rod) => this.app.xrPart(rod);
    const visor = part(false);
    visor.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, VISOR_COLOR);
    visor.mi.setFloatParameter('roughness', 0.15);
    // A capsule: a rod and a ball at each end.
    const capsules = Object.fromEntries(BODY_PARTS.map((b) => [b, CAPSULES[b].map(() => ({ rod: part(true), ends: [part(false), part(false)] }))]));
    return { head: part(false), visor, balls: HANDS.map(() => part(false)), bones: SIDES.flatMap(() => BONES.map(() => part(true))),
             capsules, color: null, shown: false };
  }

  color(v, owner) {
    const client = this.app.net ? this.app.control?.clients.get(owner) : null;
    const hex = client?.color ?? OWN_COLOR;
    if (v.color === hex) return;
    v.color = hex;
    const rgba = [...hexRgb(hex), 1];
    for (const p of this.colored(v)) p.mi.setColor4Parameter('baseColor', Filament.RgbaType.sRGB, rgba);
  }

  put(p, m) {
    const tcm = this.app.engine.getTransformManager(), inst = tcm.getInstance(p.e);
    tcm.setTransform(inst, m ?? ZERO);
    inst.delete();
  }

  colored(v) {
    return [v.head, ...v.balls, ...v.bones, ...Object.values(v.capsules).flat().flatMap((c) => [c.rod, ...c.ends])];
  }

  hide(v) {
    for (const p of [v.visor, ...this.colored(v)]) this.put(p, null);
    v.shown = false;
  }

  // `own`: this page's own avatar in VR: its torso and legs only.
  place(v, mocap, d, own = false) {
    v.shown = true;
    const index = new Map(PARTS.map((p, i) => [p, i]));
    const at = (i) => Array.from(d.mocap_pos.subarray(3 * mocap[i], 3 * mocap[i] + 3));
    const cols = (i) => quatColumns(d.mocap_quat.subarray(4 * mocap[i], 4 * mocap[i] + 4));
    const ball = (p, c, r) => this.put(p, c && !parked(c) ? [r, 0, 0, 0, 0, r, 0, 0, 0, 0, r, 0, ...c, 1] : null);
    // The head, and its visor in front (the view's -z).
    const head = own ? null : at(index.get('head'));
    ball(v.head, head, HEAD_RADIUS);
    if (!head || parked(head)) this.put(v.visor, null);
    else {
      const col = cols(index.get('head'));
      const c = head.map((h, i) => h - 0.8 * HEAD_RADIUS * col[2][i]);
      this.put(v.visor, [...col[0].map((a) => a * VISOR[0]), 0, ...col[1].map((a) => a * VISOR[1]), 0,
        ...col[2].map((a) => a * VISOR[2]), 0, ...c, 1]);
    }
    // The torso and legs: their capsules, where their body is and turned as it is.
    for (const b of BODY_PARTS) {
      const o = at(index.get(b)), col = cols(index.get(b)), off = parked(o);
      const world = (p) => o.map((x, i) => x + col[0][i] * p[0] + col[1][i] * p[1] + col[2][i] * p[2]);
      CAPSULES[b].forEach((c, k) => {
        const parts = v.capsules[b][k], a = world(c.from), z = world(c.to);
        this.put(parts.rod, off ? null : rod(a, z, c.r));
        ball(parts.ends[0], off ? null : a, c.r);
        ball(parts.ends[1], off ? null : z, c.r);
      });
    }
    // The hands: a ball per joint, rods along the bones.
    HANDS.forEach((part, k) => ball(v.balls[k], own ? null : at(index.get(part)), radius(part)));
    SIDES.forEach((side, si) => BONES.forEach(([a, b], bi) => {
      const p = at(index.get(`${side}_${a}`)), q = at(index.get(`${side}_${b}`));
      this.put(v.bones[si * BONES.length + bi], own || parked(p) || parked(q) ? null : rod(p, q, BONE_RADIUS));
    }));
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

// The unit rod (App.xrPart(true): radius 1, z from -0.5 to 0.5) from p to q, w wide.
function rod(p, q, w) {
  const dv = q.map((v, i) => v - p[i]), len = Math.hypot(...dv);
  if (len < 1e-6) return null;
  const z = dv.map((v) => v / len);
  const a = Math.abs(z[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let x = [a[1] * z[2] - a[2] * z[1], a[2] * z[0] - a[0] * z[2], a[0] * z[1] - a[1] * z[0]];
  const xl = Math.hypot(...x);
  x = x.map((v) => v / xl);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const mid = p.map((v, i) => (v + q[i]) / 2);
  return [...x.map((v) => v * w), 0, ...y.map((v) => v * w), 0, ...z.map((v) => v * len), 0, ...mid, 1];
}

