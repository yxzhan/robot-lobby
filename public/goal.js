// Driving to a goal pose: the Teleop panel's "Goal marker" input. The goal
// (x, y, yaw on the floor) is a marker dragged like a spawn marker
// (assets.js); every frame goalCommand turns the robot's base pose and the
// goal into the body twist command [vx, vy, yaw rate] the keys would give,
// within the controller's key speeds (`drive`), so it drives a walking
// policy and a wheeled base alike (in network mode the server does this:
// server/goal.py is a port).
//
// Far from the goal the robot turns towards it and goes forward (slower the
// more it is turned away), backwards for a goal behind a robot that cannot
// strafe; near it (NEAR), a robot that can strafe closes the rest sideways,
// one that cannot keeps aiming at the point. At the point it turns to the
// goal heading and stops. Tolerances have hysteresis (a step out of the
// point by more than twice TOL_POS starts over), and commands do not drop
// below what still moves a robot (a walking policy stands below ~0.05; one
// with a larger dead band says so in drive.minSpeed / drive.minTurn, and
// the tolerances grow with it so it does not overshoot back and forth).

const NEAR = 0.4;          // m: from here on, fine positioning
const TOL_POS = 0.06;      // m
const TOL_YAW = 0.06;      // rad
const K_POS = 1.0;         // 1/s: speed per metre of distance
const K_YAW = 1.5;         // 1/s: turn rate per radian of heading error
const MIN_SPEED = 0.08;    // m/s
const MIN_TURN = 0.15;     // rad/s

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (v, m) => Math.max(-m, Math.min(m, v));
const atLeast = (v, m) => (v === 0 ? 0 : Math.sign(v) * Math.max(Math.abs(v), m));

// The planar pose {x, y, yaw} of world body `b` in MjData-like `data`.
export function bodyPose(data, b) {
  const R = data.xmat.subarray(9 * b, 9 * b + 9);
  return { x: data.xpos[3 * b], y: data.xpos[3 * b + 1], yaw: Math.atan2(R[3], R[0]) };
}

// {cmd, phase ('drive' | 'turn' | 'arrived'), dist, dyaw} for the robot at
// `pose` and `goal` ({x, y, yaw}); `state` ({phase}) persists between calls.
export function goalCommand(pose, goal, drive, state) {
  const minSpeed = drive.minSpeed ?? MIN_SPEED, minTurn = drive.minTurn ?? MIN_TURN;
  const tolPos = Math.max(TOL_POS, 0.25 * minSpeed), tolYaw = Math.max(TOL_YAW, 0.25 * minTurn);
  const dxw = goal.x - pose.x, dyw = goal.y - pose.y;
  const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
  const dx = c * dxw + s * dyw, dy = -s * dxw + c * dyw;  // the goal in the robot's frame
  const dist = Math.hypot(dx, dy), dyaw = wrap(goal.yaw - pose.yaw);
  if (dist > 2 * tolPos) state.phase = 'drive';
  else if (state.phase === 'drive' && dist < tolPos) state.phase = 'turn';
  if (state.phase === 'turn' && Math.abs(dyaw) < tolYaw) state.phase = 'arrived';
  if (state.phase === 'arrived' && Math.abs(dyaw) > 2 * tolYaw) state.phase = 'turn';
  const out = { cmd: [0, 0, 0], phase: state.phase ?? 'drive', dist, dyaw };
  if (out.phase === 'arrived') return out;
  if (out.phase === 'turn') {
    out.cmd[2] = atLeast(clamp(K_YAW * dyaw, drive.turn), minTurn);
    return out;
  }
  const strafe = drive.side > 0;
  let bearing = Math.atan2(dy, dx), dir = 1;
  // A goal behind a robot that cannot strafe: back up to it rather than turn around.
  if (!strafe && Math.abs(bearing) > Math.PI / 2 && dist < 3 * NEAR && drive.back > 0) {
    bearing = wrap(bearing - Math.PI);
    dir = -1;
  }
  if (dist > NEAR || !strafe) {
    const align = Math.max(0, Math.cos(bearing));
    const top = dir > 0 ? drive.forward : drive.back;
    out.cmd[0] = dir * atLeast(Math.min(top, K_POS * dist) * align * align, align > 0.7 ? minSpeed : 0);
    out.cmd[2] = clamp(K_YAW * bearing, drive.turn);
    // Near and roughly facing the point, the heading can start turning already.
    if (dist < NEAR && align > 0.9) out.cmd[2] = clamp(K_YAW * bearing + 0.3 * dyaw, drive.turn);
  } else {
    // Near: close the rest straight, forward/back and sideways, turning to the goal heading.
    const v = [clamp(K_POS * dx, dx > 0 ? drive.forward : drive.back), clamp(K_POS * dy, drive.side)];
    const n = Math.hypot(...v);
    if (n > 0 && n < minSpeed) v.forEach((k, i) => { v[i] = (k * minSpeed) / n; });
    out.cmd = [v[0], v[1], clamp(K_YAW * dyaw, drive.turn)];
  }
  return out;
}
