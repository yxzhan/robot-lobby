// The leash between an IK chain's tip and its target, for targets set by
// hand (a marker dragged with the mouse, held in VR): a soft one. Up to KNEE
// of its length the target goes where the hand puts it; beyond, the hand's
// motion counts less and less, the target never getting further than
// LEASH_POS / LEASH_ROT from where the tip is commanded to be. A target out
// of reach so stays near the arm, moving along as the hand goes on, and
// comes back the moment the hand does (no stretch to take up first); the arm
// is never pulled taut towards a singular, fully stretched pose by a target
// far away. Let go, the target goes back onto the tip (snapToTip). Targets
// set by programs (behaviour trees) are not leashed.
//
// strain(): how taut the leash is, 0 (slack, within KNEE) .. 1 (at its end),
// for the marker's color and the controller's rumble.

export const LEASH_POS = 0.15;              // m
export const LEASH_ROT = 35 * Math.PI / 180; // rad
const KNEE = 0.5;                           // of the leash: free up to here, then softer and softer

// An offset d along a leash of length L, softened past the knee.
function soft(d, L) {
  const k = KNEE * L;
  return d <= k ? d : k + (L - k) * Math.tanh((d - k) / (L - k));
}

// Rotation matrix (row-major 3x3) helpers.
const mulMM = (A, B) => [0, 3, 6].flatMap((r) => [0, 1, 2].map((c) => A[r] * B[c] + A[r + 1] * B[3 + c] + A[r + 2] * B[6 + c]));
const mulMT = (A, B) => [0, 3, 6].flatMap((r) => [0, 3, 6].map((c) => A[r] * B[c] + A[r + 1] * B[c + 1] + A[r + 2] * B[c + 2]));  // A Bᵀ

// R as axis * angle.
function rotVec(R) {
  const v = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
  const s = Math.hypot(...v) / 2, c = (R[0] + R[4] + R[8] - 1) / 2;
  const angle = Math.atan2(s, c);
  if (s < 1e-9) {
    if (c > 0) return [0, 0, 0];
    // 180°: the axis from the diagonal.
    const d = [R[0], R[4], R[8]], i = d.indexOf(Math.max(...d));
    const a = [0, 1, 2].map((k) => (k === i ? Math.sqrt(Math.max(0, (R[4 * i] + 1) / 2)) : (R[3 * i + k] + R[3 * k + i]) / 4 / Math.sqrt(Math.max(1e-12, (R[4 * i] + 1) / 2))));
    return a.map((x) => x * Math.PI);
  }
  return v.map((x) => (x / (2 * s)) * angle);
}

// Rotation of angle |w| about w (Rodrigues).
function expRot(w) {
  const t = Math.hypot(...w);
  if (t < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const [x, y, z] = w.map((v) => v / t), c = Math.cos(t), s = Math.sin(t), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}

// R made orthonormal again (rotations composed frame after frame drift).
function orthonormal(R) {
  const n = (v) => { const l = Math.hypot(...v) || 1; return v.map((k) => k / l); };
  const x = n([R[0], R[3], R[6]]);
  let y = [R[1], R[4], R[7]];
  const d = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  y = n(y.map((v, i) => v - d * x[i]));
  const z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  return [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]];
}

// How far `pose` is from `tip`: [m, rad].
export function offset(tip, pose) {
  const dp = Math.hypot(...pose.pos.map((v, i) => v - tip.pos[i]));
  const dr = Math.hypot(...rotVec(mulMT(pose.rot, tip.rot)));
  return [dp, dr];
}

// `pose` ({pos, rot}, world) on the soft leash from `tip`.
export function leashed(tip, pose) {
  if (!tip) return pose;
  const d = pose.pos.map((v, i) => v - tip.pos[i]), dp = Math.hypot(...d), sp = soft(dp, LEASH_POS);
  const pos = sp < dp ? tip.pos.map((v, i) => v + (d[i] * sp) / dp) : pose.pos;
  const rot0 = orthonormal(pose.rot);
  const w = rotVec(mulMT(rot0, tip.rot)), dr = Math.hypot(...w), sr = soft(dr, LEASH_ROT);  // R_pose R_tipᵀ
  const rot = sr < dr ? mulMM(expRot(w.map((v) => (v * sr) / dr)), tip.rot) : rot0;
  return { pos, rot };
}

// 0 (slack) .. 1 (taut) for `pose` against `tip`.
export function strain(tip, pose) {
  if (!tip || !pose) return 0;
  const [dp, dr] = offset(tip, pose);
  const k = Math.max(dp / LEASH_POS, dr / LEASH_ROT);
  return Math.min(1, Math.max(0, (k - KNEE) / (0.95 - KNEE)));  // (the soft end is only ever approached)
}

// The chain's target back onto its tip (a marker let go).
export function snapToTip(ik, chain, data) {
  const tip = data && ik?.leashTip?.(chain, data);
  if (tip) ik.setTargetPose(chain, data, tip);
}
