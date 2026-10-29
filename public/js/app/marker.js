// A 6-DoF interactive marker like RViz's: three axis arrows (drag to move
// along the axis), three rings (drag to turn about the axis) and a center
// ball (drag to move in the plane facing the camera). Axes are those of the
// pose's rotation (red x, green y, blue z). This module is the geometry of
// picking and dragging; App (main.js) draws it with the same dimensions.

// Dimensions in units of the marker's scale, MARKER.size m: a fixed size in
// the world, like the objects it moves.
export const MARKER = {
  size: 0.15,
  arrow: [0.62, 0.86, 1.0],  // shaft start, head start, tip, along each axis (both sides)
  shaft: 0.035, head: 0.09,  // radii
  ring: 0.5, tube: 0.018,    // ring radius, tube radius
  ball: 0.09,
};
// Part ids: 0-2 arrows along x/y/z, 3-5 rings about x/y/z, 6 the ball.
export const BALL = 6;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const axis = (R, k) => [R[k], R[3 + k], R[6 + k]];

export function markerScale() {
  return MARKER.size;
}

// Where the ray {eye, dir} meets the plane through c with normal n: [point, depth] or null.
function planeHit({ eye, dir }, c, n) {
  const den = dot(dir, n);
  if (Math.abs(den) < 1e-6) return null;
  const s = dot(sub(c, eye), n) / den;
  return s > 0 ? [add(eye, mul(dir, s)), s] : null;
}

// Parameter t of the point on the line c + t a closest to the ray, and the
// ray depth and distance there; null if they are (nearly) parallel.
function lineClosest({ eye, dir }, c, a) {
  const w = sub(eye, c), b = dot(dir, a), den = 1 - b * b;
  if (den < 1e-4) return null;
  const t = (dot(w, a) - b * dot(w, dir)) / den;
  const s = (b * dot(w, a) - dot(w, dir)) / den;
  const gap = Math.hypot(...sub(add(eye, mul(dir, s)), add(c, mul(a, t))));
  return { t, s, gap };
}

// The part under the ray, nearest first: {part, depth} or null.
// `pose` = {pos, rot (row-major 3x3, world)}.
export function pickMarker(ray, pose, scale) {
  const c = pose.pos, hits = [];
  const d = sub(c, ray.eye), s = dot(d, ray.dir);
  if (s > 0 && Math.hypot(...sub(d, mul(ray.dir, s))) < MARKER.ball * 1.4 * scale) hits.push({ part: BALL, depth: s });
  for (let k = 0; k < 3; k++) {
    const a = axis(pose.rot, k);
    const l = lineClosest(ray, c, a);
    const t = l && Math.abs(l.t) / scale;
    if (l && l.s > 0 && t > MARKER.arrow[0] - 0.05 && t < MARKER.arrow[2] + 0.05 && l.gap < MARKER.head * 1.2 * scale) {
      hits.push({ part: k, depth: l.s });
    }
    const h = planeHit(ray, c, a);
    if (h && Math.abs(Math.hypot(...sub(h[0], c)) / scale - MARKER.ring) < MARKER.tube * 4) hits.push({ part: 3 + k, depth: h[1] });
  }
  hits.sort((x, y) => x.depth - y.depth);
  return hits[0] ?? null;
}

// Row-major rotation by `angle` about the unit axis n.
function axisAngle(n, angle) {
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c, [x, y, z] = n;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ];
}
const mulMM = (A, B) => [0, 3, 6].flatMap((r) => [0, 1, 2].map((k) => A[r] * B[k] + A[r + 1] * B[3 + k] + A[r + 2] * B[6 + k]));

// Starts dragging `part` of the marker at `pose`; `forward` is the camera's
// view direction (the ball's drag plane). Returns the drag state, or null.
export function startDrag(part, ray, pose, scale, forward) {
  const start = { pos: [...pose.pos], rot: [...pose.rot] };
  if (part === BALL) return { part, start, normal: forward, from: pose.pos };
  const a = axis(pose.rot, part % 3);
  if (part < 3) {
    const l = lineClosest(ray, pose.pos, a);
    return l && { part, start, a, t0: l.t };
  }
  const h = planeHit(ray, pose.pos, a);
  return h && { part, start, a, v0: sub(h[0], pose.pos) };
}

// The marker pose for the drag state `g` and the current ray; null if the
// ray gives no answer (e.g. parallel to the drag plane).
export function dragPose(g, ray) {
  const { start } = g;
  if (g.part === BALL) {
    const h = planeHit(ray, g.from, g.normal);
    return h && { pos: h[0], rot: start.rot };
  }
  if (g.part < 3) {
    const l = lineClosest(ray, start.pos, g.a);
    return l && { pos: add(start.pos, mul(g.a, l.t - g.t0)), rot: start.rot };
  }
  const h = planeHit(ray, start.pos, g.a);
  if (!h) return null;
  const v = sub(h[0], start.pos);
  const angle = Math.atan2(dot(g.a, cross(g.v0, v)), dot(g.v0, v));
  return { pos: start.pos, rot: mulMM(axisAngle(g.a, angle), start.rot) };
}
