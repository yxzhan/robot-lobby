// The rope between an IK chain's tip and its marker, like Shift+drag's: a
// marker dragged with the mouse or held in VR goes wherever the hand puts
// it, and the arm follows as far as it reaches. The rope only shows how far
// the tip is behind: drawn from the tip (measured: the real TCP) to the
// marker (App.syncMarkers), redder and the controller rumbling the further
// (strain). Let go, the marker goes back onto the tip (snapToTip): the arm
// stays where it got to, and is not left reaching for a target out of reach.

export const ROPE_SLACK = 0.05;              // m: closer than this, no strain (tracking lag)
export const ROPE_TAUT = 0.2;                // m: this far, full strain
const ROT_SLACK = 15 * Math.PI / 180, ROT_TAUT = 60 * Math.PI / 180;  // rad: the same for orientation

const mulMT = (A, B) => [0, 3, 6].flatMap((r) => [0, 3, 6].map((c) => A[r] * B[c] + A[r + 1] * B[c + 1] + A[r + 2] * B[c + 2]));  // A Bᵀ

// The angle of rotation R.
function angle(R) {
  const s = Math.hypot(R[7] - R[5], R[2] - R[6], R[3] - R[1]) / 2, c = (R[0] + R[4] + R[8] - 1) / 2;
  return Math.atan2(s, c);
}

// 0 (slack) .. 1 (taut) for marker `pose` against `tip`.
export function strain(tip, pose) {
  if (!tip || !pose) return 0;
  const dp = Math.hypot(...pose.pos.map((v, i) => v - tip.pos[i]));
  const dr = angle(mulMT(pose.rot, tip.rot));
  const k = Math.max((dp - ROPE_SLACK) / (ROPE_TAUT - ROPE_SLACK), (dr - ROT_SLACK) / (ROT_TAUT - ROT_SLACK));
  return Math.min(1, Math.max(0, k));
}

// A marker let go: the arm stops where it is, the target on its tip
// (IkController.snapTarget, RemoteIk's on the server).
export function snapToTip(ik, chain, data) {
  if (data && !chain.error) ik?.snapTarget?.(chain, data);
}
