"""ANYbotics ANYmal C joystick task: Playground's Spot joystick task on the
menagerie's anymal_c.xml (the model the viewer spawns).

Playground has no ANYmal task. The robot is built from anymal_c.xml (the
same bodies, inertias, joints and ranges) the way Playground models its
Spot for training (spot_mjx_feetonly.xml): only the feet collide (spheres,
with the floor), Euler at 4 ms with one solver iteration, and the sensors
the task reads. anymal_c.xml has no home pose: HOME stands with the feet
under the HFE joints, knees in (front knees back, hind knees forward), as
ANYmal stands.

The task (rewards, observation, termination) and PPO config are
Playground's SpotFlatTerrainJoystick: a 45 kg robot, much as Spot (Go1's
task only ever learnt to stand on Spot, see envs/spot_arm.py). As for
spot_arm: Go1's domain randomization, random pushes, and new commands every
1-4 s. Gains: Kp 250, Kd 4, action scale 0.3 (near Spot's 300 / 1 / 0.3),
within the ANYdrives' 80 Nm. With Kp 120 (it sagged 0.15 rad at the knees
standing) it only learnt to stand: no foot ever left the ground.
"""

import pathlib

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick

HERE = pathlib.Path(__file__).resolve().parent
ROBOT = HERE.parent.parent / 'mjcf_assets' / 'robots' / 'mujoco_menagerie' / 'anybotics_anymal_c' / 'anymal_c.xml'
BUILD = HERE / 'build'

LEGS = ['LF', 'RF', 'LH', 'RH']  # Spot's FL, FR, HL, HR order
FEET = [f'{leg}_FOOT' for leg in LEGS]  # foot geoms (unnamed in anymal_c.xml) and sites, named here
ROOT_BODY = 'base'
# HAA, HFE, KFE per leg: front legs (HFE, KFE) = (0.7, -1.0), hind legs mirrored.
HOME = {'LF': [0, 0.7, -1.0], 'RF': [0, 0.7, -1.0], 'LH': [0, -0.7, 1.0], 'RH': [0, -0.7, 1.0]}
HOME_Z = 0.53  # base height in HOME with the feet (r 0.03) just on the floor
# Servo ranges: the HAA joint ranges; HFE and KFE (unlimited joints) home +- KNEE_SPAN.
KNEE_SPAN = 1.6
FORCE = 80.0
ARMATURE = 0.05  # anymal_c.xml has none; keeps Euler at 4 ms stable on the light shanks

SCENE = """<mujoco model="anymal c feetonly flat terrain">
  <include file="anymal_c_feetonly.xml"/>
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
    <key name="home" qpos="0 0 {z} 1 0 0 0 {home}" ctrl="{home}"/>
  </keyframe>
</mujoco>
"""


def home_pose():
  return [v for leg in LEGS for v in HOME[leg]]


def build_scene():
  """Writes build/anymal_c_feetonly.xml and build/scene_anymal_c.xml; returns the scene's path."""
  spec = mujoco.MjSpec.from_file(str(ROBOT))
  opt = spec.option
  opt.timestep = 0.004
  opt.iterations = 1
  opt.ls_iterations = 5
  opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
  opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  opt.impratio = 1
  opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  for light in list(spec.lights):
    spec.delete(light)
  for g in list(spec.geoms):
    body = g.parent.name
    if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and body.endswith('_SHANK'):
      g.name = body[:2] + '_FOOT'
      g.contype, g.conaffinity, g.priority, g.condim = 0, 1, 0, 1
      g.solimp[:3] = [0.9, 0.95, 0.03]
      g.parent.add_site(name=g.name, pos=g.pos, size=[0.03, 0, 0], group=5)
    else:
      spec.delete(g)  # visual meshes and collision primitives: only the feet collide
  for m in list(spec.meshes):
    spec.delete(m)
  for t in list(spec.textures):
    spec.delete(t)
  for m in list(spec.materials):
    spec.delete(m)
  base = spec.body(ROOT_BODY)
  base.quat = [1, 0, 0, 0]
  base.add_site(name='imu')
  for j in spec.joints:
    if j.name:
      j.armature = ARMATURE
  joints = {j.name: j for j in spec.joints if j.name}
  home = dict(zip([f'{leg}_{k}' for leg in LEGS for k in ('HAA', 'HFE', 'KFE')], home_pose()))
  for a in spec.actuators:
    j = joints[a.target]
    a.ctrlrange = j.range if a.target.endswith('HAA') else (home[a.target] - KNEE_SPAN, home[a.target] + KNEE_SPAN)
    a.forcelimited, a.forcerange = 1, (-FORCE, FORCE)
  BUILD.mkdir(exist_ok=True)
  (BUILD / 'anymal_c_feetonly.xml').write_text(spec.to_xml())
  lines = []
  for f in FEET:
    lines.append(f'    <framelinvel objtype="site" objname="{f}" name="{f}_global_linvel"/>')
    lines.append(f'    <framepos objtype="site" objname="{f}" name="{f}_pos" reftype="site" refname="imu"/>')
    lines.append(f'    <contact name="{f}_floor_found" geom1="{f}" geom2="floor" reduce="mindist" num="1" data="found"/>')
  scene = BUILD / 'scene_anymal_c.xml'
  scene.write_text(SCENE.format(feet_sensors='\n'.join(lines), home=' '.join(map(str, home_pose())), z=HOME_Z))
  return scene


def default_config():
  cfg = spot_joystick.default_config()
  cfg.Kp = 250.0
  cfg.Kd = 4.0
  cfg.action_scale = 0.3
  cfg.reward_config.max_foot_height = 0.15
  cfg.pert_config.enable = True  # random pushes
  cfg.cmd_min_steps = 50  # a command lasts this many control steps (of 0.02 s) at least, 200 at most
  return cfg


class Joystick(spot_joystick.Joystick):
  """Playground's Spot joystick task on the ANYmal C model above."""

  def __init__(self, task='flat_terrain', config=None, config_overrides=None):
    assert task == 'flat_terrain', task
    config = config or default_config()
    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    # As SpotEnv.__init__ and Joystick.__init__, with this model.
    self._xml_path = build_scene().as_posix()
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = config.sim_dt
    self._mj_model.dof_damping[6:] = config.Kd
    self._mj_model.actuator_gainprm[:, 0] = config.Kp
    self._mj_model.actuator_biasprm[:, 1] = -config.Kp
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._feet_floor_found_sensor = [self._mj_model.sensor(f'{g}_floor_found').id for g in FEET]
    assert self._mj_model.geom('floor').id == 0 and self._mj_model.body(ROOT_BODY).id == 1, \
        'domain_randomize expects the floor as geom 0 and the trunk as body 1'
    assert self._mj_model.nu == 12 and self._mj_model.nq == 19
    self._post_init()
    self._pert_func = self._maybe_apply_perturbation if config.pert_config.enable else lambda state, _: state

  def _post_init(self):
    # Spot's, with this robot's names.
    m = self._mj_model
    self._init_q = jp.array(m.keyframe('home').qpos)
    self._default_pose = m.keyframe('home').qpos[7:]
    self._lowers, self._uppers = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
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
    self._weights = jp.array([1.0, 1.0, 1.0] * 4)

  def get_feet_pos(self, data):
    return jp.vstack([mjx_env.get_sensor_data(self.mj_model, data, f'{f}_pos') for f in FEET])

  def step(self, state, action):
    # As envs/spot_arm.py: the next command after cmd_min_steps to 200 steps.
    state = super().step(state, action)
    rng, key = jax.random.split(state.info['rng'])
    start = jax.random.randint(key, (), 0, 201 - self._config.cmd_min_steps)
    state.info['step'] = jp.where((state.info['step'] == 0) & (state.done == 0), start, state.info['step'])
    state.info['rng'] = rng
    return state


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Go1's domain randomization (floor friction, joint friction loss and
  armature, torso center of mass and mass, link masses, qpos0)."""
  floor, torso = 0, 1

  @jax.vmap
  def rand(rng):
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[floor, 0].set(jax.random.uniform(key, minval=0.4, maxval=1.0))
    rng, key = jax.random.split(rng)
    dof_frictionloss = model.dof_frictionloss.at[6:].set(
        model.dof_frictionloss[6:] * jax.random.uniform(key, (12,), minval=0.9, maxval=1.1))
    rng, key = jax.random.split(rng)
    dof_armature = model.dof_armature.at[6:].set(
        model.dof_armature[6:] * jax.random.uniform(key, (12,), minval=1.0, maxval=1.05))
    rng, key = jax.random.split(rng)
    body_ipos = model.body_ipos.at[torso].add(jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05))
    rng, key = jax.random.split(rng)
    body_mass = model.body_mass * jax.random.uniform(key, (model.nbody,), minval=0.9, maxval=1.1)
    rng, key = jax.random.split(rng)
    body_mass = body_mass.at[torso].add(jax.random.uniform(key, minval=-2.0, maxval=2.0))
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0.at[7:].add(jax.random.uniform(key, (12,), minval=-0.05, maxval=0.05))
    return geom_friction, body_ipos, body_mass, qpos0, dof_frictionloss, dof_armature

  friction, body_ipos, body_mass, qpos0, frictionloss, armature = rand(rng)
  fields = {'geom_friction': friction, 'body_ipos': body_ipos, 'body_mass': body_mass, 'qpos0': qpos0,
            'dof_frictionloss': frictionloss, 'dof_armature': armature}
  in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({k: 0 for k in fields})
  return model.tree_replace(fields), in_axes
