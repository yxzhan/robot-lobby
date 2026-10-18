"""Unitree Go2 joystick task: Playground's Go1 joystick task on a Go2.

Playground has no Go2 task, so this builds one from the robot the viewer
spawns (mjcf_assets/.../unitree_go2/go2.xml, the same bodies, inertias,
joints and ranges) the way Playground models its Go1 for training
(go1_mjx_feetonly.xml): only the feet collide (spheres, with the floor),
position servos instead of go2.xml's torque motors (the viewer's controller
closes the same PD loop in software on the motors), Euler at 4 ms with one
solver iteration, and the sensors the task reads. The task itself (rewards,
observation, commands, termination) and its PPO config and domain
randomization are Go1's.
"""

import pathlib

import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

HERE = pathlib.Path(__file__).resolve().parent
ROBOT = HERE.parent.parent / 'mjcf_assets' / 'robots' / 'mujoco_menagerie' / 'unitree_go2' / 'go2.xml'
BUILD = HERE / 'build'

FEET = ['FR', 'FL', 'RR', 'RL']  # geoms in go2.xml; sites of the same names are added
ROOT_BODY = 'base'
# go2_mjx.xml's servo ranges (a bit inside the joint ranges) and go2.xml's torque limits.
CTRL_RANGE = {'hip': (-0.9472, 0.9472), 'thigh': (-1.4, 2.5), 'calf': (-2.6227, -0.84776)}
FORCE = {'hip': 23.7, 'thigh': 23.7, 'calf': 45.43}
HOME = [0, 0.9, -1.8] * 4

SCENE = """<mujoco model="go2 feetonly flat terrain">
  <include file="go2_feetonly.xml"/>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
  </worldbody>
  <sensor>
    <gyro site="imu" name="gyro"/>
    <velocimeter site="imu" name="local_linvel"/>
    <accelerometer site="imu" name="accelerometer"/>
    <framepos objtype="site" objname="imu" name="position"/>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
    <framexaxis objtype="site" objname="imu" name="forwardvector"/>
    <framelinvel objtype="site" objname="imu" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu" name="global_angvel"/>
    <framequat objtype="site" objname="imu" name="orientation"/>
{feet_sensors}
  </sensor>
  <keyframe>
    <key name="home" qpos="0 0 0.27 1 0 0 0 {home}" ctrl="{home}"/>
  </keyframe>
</mujoco>
"""


def build_scene():
  """Writes build/go2_feetonly.xml and build/scene_flat.xml; returns the scene's path."""
  spec = mujoco.MjSpec.from_file(str(ROBOT))
  opt = spec.option
  opt.timestep = 0.004
  opt.iterations = 1
  opt.ls_iterations = 5
  opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
  opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  opt.impratio = 1
  opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  feet = {}
  for g in list(spec.geoms):
    if g.name in FEET:
      feet[g.name] = g
    else:
      spec.delete(g)  # visual meshes and collision primitives: only the feet collide
  for m in list(spec.meshes):
    spec.delete(m)
  for k in list(spec.keys):
    spec.delete(k)  # the scene's home key (with ctrl for the servos) replaces go2.xml's
  for name, g in feet.items():
    g.contype, g.conaffinity, g.priority, g.condim = 0, 1, 0, 1
    g.solimp[:3] = [0.9, 0.95, 0.023]
    g.parent.add_site(name=name, pos=g.pos, size=[0.022, 0, 0], group=5)
  for a in list(spec.actuators):
    joint, name = a.target, a.name
    spec.delete(a)
    part = name.split('_')[1]
    s = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    s.set_to_position(kp=35)
    s.ctrllimited, s.ctrlrange = 1, CTRL_RANGE[part]
    s.forcelimited, s.forcerange = 1, (-FORCE[part], FORCE[part])
  BUILD.mkdir(exist_ok=True)
  (BUILD / 'go2_feetonly.xml').write_text(spec.to_xml())
  lines = []
  for f in FEET:
    lines.append(f'    <framelinvel objtype="site" objname="{f}" name="{f}_global_linvel"/>')
    lines.append(f'    <framepos objtype="site" objname="{f}" name="{f}_pos" reftype="site" refname="imu"/>')
    lines.append(f'    <contact name="{f}_floor_found" geom1="{f}" geom2="floor" reduce="mindist" num="1" data="found"/>')
  scene = BUILD / 'scene_flat.xml'
  scene.write_text(SCENE.format(feet_sensors='\n'.join(lines), home=' '.join(map(str, HOME))))
  return scene


def default_config():
  cfg = go1_joystick.default_config()
  return cfg


class Joystick(go1_joystick.Joystick):
  """Go1's joystick task on the Go2 model above."""

  def __init__(self, task='flat_terrain', config=None, config_overrides=None):
    assert task == 'flat_terrain', task
    config = config or default_config()
    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    # As Go1Env.__init__, with this model.
    self._xml_path = build_scene().as_posix()
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.opt.ccd_iterations = 20
    self._mj_model.dof_damping[6:] = config.Kd
    self._mj_model.actuator_gainprm[:, 0] = config.Kp
    self._mj_model.actuator_biasprm[:, 1] = -config.Kp
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._imu_site_id = self._mj_model.site('imu').id
    self._feet_floor_found_sensor = [self._mj_model.sensor(f'{g}_floor_found').id for g in FEET]
    assert self._mj_model.geom('floor').id == 0 and self._mj_model.body(ROOT_BODY).id == 1, \
        "Go1's domain randomization expects the floor as geom 0 and the trunk as body 1"
    self._post_init()

  def _post_init(self):
    # Go1's, with this robot's root body.
    m = self._mj_model
    self._init_q = jp.array(m.keyframe('home').qpos)
    self._default_pose = jp.array(m.keyframe('home').qpos[7:])
    self._lowers, self._uppers = m.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor
    self._torso_body_id = m.body(ROOT_BODY).id
    self._torso_mass = m.body_subtreemass[self._torso_body_id]
    self._feet_site_id = np.array([m.site(n).id for n in FEET])
    self._floor_geom_id = m.geom('floor').id
    self._feet_geom_id = np.array([m.geom(n).id for n in FEET])
    adr = []
    for site in FEET:
      s = m.sensor(f'{site}_global_linvel').id
      adr.append(list(range(m.sensor_adr[s], m.sensor_adr[s] + m.sensor_dim[s])))
    self._foot_linvel_sensor_adr = jp.array(adr)
    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)
