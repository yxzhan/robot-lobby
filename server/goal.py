"""Driving to a goal pose on the server: a port of public/js/goal.js (see there).

In network mode a robot's goal is shared: any browser sets it (op `goal`),
the server follows it and tells everyone, so two browsers never drive the
same robot to two different goals.
"""

import math

NEAR = 0.4         # m: from here on, fine positioning
TOL_POS = 0.06     # m
TOL_YAW = 0.06     # rad
K_POS = 1.0        # 1/s: speed per metre of distance
K_YAW = 1.5        # 1/s: turn rate per radian of heading error
MIN_SPEED = 0.08   # m/s
MIN_TURN = 0.15    # rad/s


def wrap(a):
  return math.atan2(math.sin(a), math.cos(a))


def clamp(v, m):
  return max(-m, min(m, v))


def at_least(v, m):
  return 0.0 if v == 0 else math.copysign(max(abs(v), m), v)


def body_pose(data, b):
  """The planar pose {x, y, yaw} of world body `b`."""
  R = data.xmat[b]
  return {'x': float(data.xpos[b, 0]), 'y': float(data.xpos[b, 1]), 'yaw': math.atan2(R[3], R[0])}


def goal_command(pose, goal, drive, state):
  """{cmd, phase, dist, dyaw} as goal.js goalCommand; `state` ({phase}) persists between calls."""
  min_speed = drive.get('minSpeed') or MIN_SPEED
  min_turn = drive.get('minTurn') or MIN_TURN
  near = drive.get('near') or NEAR  # (a policy.json's goal.near)
  tol_pos, tol_yaw = max(TOL_POS, 0.25 * min_speed), max(TOL_YAW, 0.25 * min_turn)
  dxw, dyw = goal['x'] - pose['x'], goal['y'] - pose['y']
  c, s = math.cos(pose['yaw']), math.sin(pose['yaw'])
  dx, dy = c * dxw + s * dyw, -s * dxw + c * dyw  # the goal in the robot's frame
  dist, dyaw = math.hypot(dx, dy), wrap(goal['yaw'] - pose['yaw'])
  # A new goal starts driving (also one at the robot's feet: a turn in place).
  if dist > 2 * tol_pos or not state.get('phase'):
    state['phase'] = 'drive'
  elif state.get('phase') == 'drive' and dist < tol_pos:
    state['phase'] = 'turn'
  if state.get('phase') == 'turn' and abs(dyaw) < tol_yaw:
    state['phase'] = 'arrived'
  if state.get('phase') == 'arrived' and abs(dyaw) > 2 * tol_yaw:
    state['phase'] = 'turn'
  out = {'cmd': [0.0, 0.0, 0.0], 'phase': state.get('phase', 'drive'), 'dist': dist, 'dyaw': dyaw}
  if out['phase'] == 'arrived':
    return out
  if out['phase'] == 'turn':
    out['cmd'][2] = at_least(clamp(K_YAW * dyaw, drive['turn']), min_turn)
    return out
  strafe = drive.get('side', 0) > 0
  bearing, direction = math.atan2(dy, dx), 1
  # A goal behind a robot that cannot strafe: back up to it rather than turn around.
  if not strafe and abs(bearing) > math.pi / 2 and dist < 3 * near and drive.get('back', 0) > 0:
    bearing, direction = wrap(bearing - math.pi), -1
  if dist > near or not strafe:
    align = max(0.0, math.cos(bearing))
    top = drive['forward'] if direction > 0 else drive['back']
    out['cmd'][0] = direction * at_least(min(top, K_POS * dist) * align * align, min_speed if align > 0.7 else 0)
    out['cmd'][2] = clamp(K_YAW * bearing, drive['turn'])
    # Near and roughly facing the point, the heading can start turning already.
    if dist < near and align > 0.9:
      out['cmd'][2] = clamp(K_YAW * bearing + 0.3 * dyaw, drive['turn'])
  else:
    # Near: close the rest straight, forward/back and sideways, turning to the goal heading.
    v = [clamp(K_POS * dx, drive['forward'] if dx > 0 else drive['back']), clamp(K_POS * dy, drive['side'])]
    n = math.hypot(*v)
    if 0 < n < min_speed:
      v = [k * min_speed / n for k in v]
    out['cmd'] = [v[0], v[1], clamp(K_YAW * dyaw, drive['turn'])]
  return out
