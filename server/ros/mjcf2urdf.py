"""A URDF of a lobby robot's MJCF, for ros2_control with the lobby as its
hardware (topic_based_ros2_control): the kinematic tree only (links with
no geometry: robot_state_publisher computes TF from it, the controllers
take the joints from it), hinge joints as revolute, slide joints as
prismatic, the rest fixed; a joint following another through an equality
constraint (a gripper's second finger) as its mimic. The <ros2_control>
block has every joint that an actuator drives with a position command and
position / velocity state interfaces (topic_based_ros2_control/
TopicBasedSystem on the lobby's ROS bridge topics: joint states in, joint
commands out), plus the joints named with --command (driven through the ROS
profile's `commands`, e.g. a gripper on a tendon), mimics left out.

    .venv/bin/python server/ros/mjcf2urdf.py \\
        mjcf_assets/robots/custom/franka_fr3_hand/franka_fr3_hand.xml \\
        --command fr3_finger_joint1 -o server/ros/fr3/fr3.urdf

Needs a joint's anchor at its body's origin (MJCF joint pos 0), as
URDF-made models have it; others are reported.
"""

import argparse
import math
import pathlib
import sys
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np

COMMANDS_TOPIC, STATES_TOPIC = '/robot_joint_commands', '/robot_joint_states'  # topic_based_ros2_control's defaults


def rpy(quat):
  """URDF roll, pitch, yaw (fixed axes x, y, z) of a MuJoCo quaternion (w, x, y, z)."""
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, np.asarray(quat, float))
  R = m.reshape(3, 3)
  pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
  if abs(R[2, 0]) < 1 - 1e-9:
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])
  return math.atan2(-R[1, 2], R[1, 1]), pitch, 0.0


def fmt(v):
  return ' '.join(f'{x:.9g}' if abs(x) > 1e-12 else '0' for x in v)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('mjcf')
  ap.add_argument('-o', '--out', required=True)
  ap.add_argument('--name', help='the robot name (the model\'s by default)')
  ap.add_argument('--command', action='append', default=[], help='a joint commanded through the ROS profile (repeatable)')
  args = ap.parse_args()
  m = mujoco.MjModel.from_xml_path(args.mjcf)
  name = lambda obj, i: mujoco.mj_id2name(m, obj, i) or ''
  body = lambda b: name(mujoco.mjtObj.mjOBJ_BODY, b)
  joint = lambda j: name(mujoco.mjtObj.mjOBJ_JOINT, j)
  driven = set()
  for a in range(m.nu):
    if m.actuator_trntype[a] == int(mujoco.mjtTrn.mjTRN_JOINT):
      driven.add(int(m.actuator_trnid[a, 0]))
  lead = lambda j: j in driven or joint(j) in args.command
  # Mimics: equality joint constraints j1 = c0 + c1 j2 (linear): the joint
  # not driven or commanded follows the other.
  mimic = {}
  for e in range(m.neq):
    if m.eq_type[e] == int(mujoco.mjtEq.mjEQ_JOINT) and m.eq_obj2id[e] >= 0:
      j1, j2, (c0, c1) = int(m.eq_obj1id[e]), int(m.eq_obj2id[e]), m.eq_data[e][:2]
      c1 = c1 or 1.0
      if lead(j1) and not lead(j2):
        mimic[j2] = (j1, 1 / c1, -c0 / c1)
      else:
        mimic[j1] = (j2, c1, c0)
  lines = [f'<?xml version="1.0"?>',
           f'<!-- Made by server/ros/mjcf2urdf.py from {pathlib.Path(args.mjcf).name}: its kinematic tree for ros2_control '
           f'with the lobby as the hardware (topic_based_ros2_control). -->',
           f'<robot name={quoteattr(args.name or m.names[:m.names.index(0)].decode() or "robot")}>',
           '  <link name="world"/>']
  problems = []
  joints = []  # (name, kind) of the URDF's movable joints
  for b in range(1, m.nbody):
    lines.append(f'  <link name={quoteattr(body(b))}/>')
  for b in range(1, m.nbody):
    parent = 'world' if m.body_parentid[b] == 0 else body(m.body_parentid[b])
    pos, quat = m.body_pos[b], m.body_quat[b]
    origin = f'<origin xyz="{fmt(pos)}" rpy="{fmt(rpy(quat))}"/>'
    n = m.body_jntnum[b]
    j = m.body_jntadr[b] if n else -1
    kinds = {int(mujoco.mjtJoint.mjJNT_HINGE): 'revolute', int(mujoco.mjtJoint.mjJNT_SLIDE): 'prismatic'}
    if n == 1 and m.jnt_type[j] in kinds:
      if np.linalg.norm(m.jnt_pos[j]) > 1e-9:
        problems.append(f'{joint(j)}: its anchor is off its body\'s origin ({fmt(m.jnt_pos[j])})')
      kind = kinds[m.jnt_type[j]]
      lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi)
      lines += [f'  <joint name={quoteattr(joint(j))} type="{kind}">', f'    <parent link={quoteattr(parent)}/>',
                f'    <child link={quoteattr(body(b))}/>', f'    {origin}', f'    <axis xyz="{fmt(m.jnt_axis[j])}"/>',
                f'    <limit lower="{lo:.9g}" upper="{hi:.9g}" effort="1000" velocity="10"/>']
      if j in mimic:
        lead, mult, off = mimic[j]
        lines.append(f'    <mimic joint={quoteattr(joint(lead))} multiplier="{mult:.9g}" offset="{off:.9g}"/>')
      lines.append('  </joint>')
      if j not in mimic and lead(j):
        joints.append(joint(j))
    else:
      if n:
        problems.append(f'{body(b)}: {n} joint(s) of a kind URDF has not here: fixed')
      lines += [f'  <joint name={quoteattr(body(b) + "_fixed")} type="fixed">', f'    <parent link={quoteattr(parent)}/>',
                f'    <child link={quoteattr(body(b))}/>', f'    {origin}', '  </joint>']
  lines += ['  <ros2_control name="lobby" type="system">', '    <hardware>',
            '      <plugin>topic_based_ros2_control/TopicBasedSystem</plugin>',
            f'      <param name="joint_commands_topic">{COMMANDS_TOPIC}</param>',
            f'      <param name="joint_states_topic">{STATES_TOPIC}</param>', '    </hardware>']
  for jn in joints:
    lines += [f'    <joint name={quoteattr(jn)}>', '      <command_interface name="position"/>',
              '      <state_interface name="position"/>', '      <state_interface name="velocity"/>', '    </joint>']
  lines += ['  </ros2_control>', '</robot>', '']
  pathlib.Path(args.out).write_text('\n'.join(lines))
  print(f'{args.out}: {m.nbody - 1} links, ros2_control joints: {", ".join(joints)}')
  for p in problems:
    print('  note:', p, file=sys.stderr)


if __name__ == '__main__':
  main()
