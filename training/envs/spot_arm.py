"""Spot with its arm, joystick task: Playground's Spot joystick task on the
menagerie's spot_arm.xml (the model the viewer spawns).

The robot is built from spot_arm.xml (the same bodies, inertias, joints and
ranges) the way Playground models its Spot for training
(spot_mjx_feetonly.xml): only the feet collide (spheres, with the floor),
Euler at 4 ms with one solver iteration, Playground's Spot joint armature
and friction loss, and the sensors the task reads. The arm is part of the
body: its seven joints are fixed (their bodies keep their masses and
inertias), so the policy drives the 12 leg joints only. To walk whatever the
arm does in the viewer (IK, a held object), every training environment
gets an arm pose of its own (domain_randomize): stowed, or anywhere in front.

The task (rewards, observation, commands every 4 s, termination), its PD
gains and PPO config are Playground's SpotFlatTerrainJoystick. Go1's task
(envs/go2.py), tried first, only ever learnt to stand on this 56 kg robot.
Added: domain randomization (Go1's, plus the arm pose; Playground's Spot
task has none), random pushes (its pert_config, off by default) and new
commands every 1-4 s instead of every 4 s (it tipped over when the keys or
the goal marker changed the command quickly), so it copes with what the
viewer does to it.
"""

import pathlib

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick
from mujoco_playground._src.locomotion.spot import spot_constants as spot_consts

HERE = pathlib.Path(__file__).resolve().parent
ROBOT = HERE.parent.parent / 'mjcf_assets' / 'robots' / 'mujoco_menagerie' / 'boston_dynamics_spot' / 'spot_arm.xml'
BUILD = HERE / 'build'

FEET = ['FL', 'FR', 'HL', 'HR']  # foot geoms in spot_arm.xml; sites of the same names are added
ROOT_BODY = 'body'
LEGS = ['fl', 'fr', 'hl', 'hr']
HOME_LEG = [0, 1.04, -1.8]
# spot_arm.xml's home key: the arm stowed.
ARM_HOME = {'arm_sh0': 0, 'arm_sh1': -3.14, 'arm_el0': 3.06, 'arm_el1': 0, 'arm_wr0': 0, 'arm_wr1': 0, 'arm_f1x': 0}
# Arm poses the training environments get (domain_randomize): uniform in these, or stowed.
ARM_RANGE = {'arm_sh0': (-1.2, 1.2), 'arm_sh1': (-3.14, 0.3), 'arm_el0': (0.0, 3.1), 'arm_el1': (-1.5, 1.5),
             'arm_wr0': (-1.5, 1.5), 'arm_wr1': (-1.5, 1.5), 'arm_f1x': (-1.57, 0.0)}

SCENE = """<mujoco model="spot arm feetonly flat terrain">
  <include file="spot_arm_feetonly.xml"/>
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
    <key name="home" qpos="0 0 0.46 1 0 0 0 {home}" ctrl="{home}"/>
  </keyframe>
</mujoco>
"""


def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def axis_angle(axis, angle):
  s = np.sin(angle / 2) if isinstance(angle, (int, float)) else jp.sin(angle / 2)
  c = np.cos(angle / 2) if isinstance(angle, (int, float)) else jp.cos(angle / 2)
  return [c, axis[0] * s, axis[1] * s, axis[2] * s]


def build_scene():
  """Writes build/spot_arm_feetonly.xml and build/scene_spot_arm.xml; returns
  the scene's path and the arm joints, [(body name, its quat with the joint
  at zero, axis, name)]."""
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
      spec.delete(g)  # visual meshes and collision meshes: only the feet collide
  for m in list(spec.meshes):
    spec.delete(m)
  for k in list(spec.keys):
    spec.delete(k)  # the scene's home key replaces spot_arm.xml's
  for name, g in feet.items():
    g.contype, g.conaffinity, g.priority, g.condim = 0, 1, 0, 1
    g.solimp[:3] = [0.9, 0.95, 0.036]  # spot_mjx_feetonly.xml's
    g.parent.add_site(name=name, pos=g.pos, size=[0.036, 0, 0], group=5)
  spec.body(ROOT_BODY).add_site(name='imu')
  # The arm: no actuators, its joints fixed in the stowed pose.
  arm = []
  for a in list(spec.actuators):
    if a.target.startswith('arm_'):
      spec.delete(a)
  for j in list(spec.joints):
    if not j.name.startswith('arm_'):
      continue
    body = j.parent
    q0 = [float(v) for v in body.quat]
    axis = [float(v) for v in j.axis]
    arm.append((body.name, q0, axis, j.name))
    body.quat = quat_mul(q0, axis_angle(axis, ARM_HOME[j.name]))
    spec.delete(j)
  # Leg joints: Playground's Spot armature and friction loss (damping: the task's Kd).
  for j in spec.joints:
    if j.name[:2] in LEGS:
      j.armature, j.frictionloss = 0.1, 0.2
  BUILD.mkdir(exist_ok=True)
  (BUILD / 'spot_arm_feetonly.xml').write_text(spec.to_xml())
  lines = []
  for f in FEET:
    lines.append(f'    <framelinvel objtype="site" objname="{f}" name="{f}_global_linvel"/>')
    lines.append(f'    <framepos objtype="site" objname="{f}" name="{f}_pos" reftype="site" refname="imu"/>')
    lines.append(f'    <contact name="{f}_floor_found" geom1="{f}" geom2="floor" reduce="mindist" num="1" data="found"/>')
  scene = BUILD / 'scene_spot_arm.xml'
  scene.write_text(SCENE.format(feet_sensors='\n'.join(lines), home=' '.join(map(str, HOME_LEG * 4))))
  return scene, arm


def default_config():
  cfg = spot_joystick.default_config()
  cfg.pert_config.enable = True  # random pushes
  cfg.cmd_min_steps = 50  # a command lasts this many control steps (of 0.02 s) at least, 200 at most
  cfg.arm_stowed = 0.3  # share of training environments with the arm stowed (domain_randomize)
  cfg.arm_reach = 1.0   # how much of ARM_RANGE the others' arm poses span, about the stowed pose
  return cfg


class Joystick(spot_joystick.Joystick):
  """Playground's Spot joystick task on the Spot-with-arm model above."""

  def __init__(self, task='flat_terrain', config=None, config_overrides=None):
    assert task == 'flat_terrain', task
    config = config or default_config()
    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    # As SpotEnv.__init__ and Joystick.__init__, with this model.
    scene, self.arm = build_scene()
    self._xml_path = scene.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = config.sim_dt
    self._mj_model.dof_damping[6:] = config.Kd
    self._mj_model.actuator_gainprm[:, 0] = config.Kp
    self._mj_model.actuator_biasprm[:, 1] = -config.Kp
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._feet_floor_found_sensor = [self._mj_model.sensor(f'{g}_floor_found').id for g in spot_consts.FEET_GEOMS]
    assert self._mj_model.geom('floor').id == 0 and self._mj_model.body(ROOT_BODY).id == 1, \
        'domain_randomize expects the floor as geom 0 and the trunk as body 1'
    assert self._mj_model.nu == 12 and self._mj_model.nq == 19, 'the legs only should move'
    self._post_init()
    self._pert_func = self._maybe_apply_perturbation if config.pert_config.enable else lambda state, _: state

  def step(self, state, action):
    # Spot's step samples a new command once its step count passes 200 and
    # restarts the count at 0; starting it anywhere up to 200 - cmd_min_steps
    # instead makes the next one come after cmd_min_steps to 200 steps.
    state = super().step(state, action)
    rng, key = jax.random.split(state.info['rng'])
    start = jax.random.randint(key, (), 0, 201 - self._config.cmd_min_steps)
    state.info['step'] = jp.where((state.info['step'] == 0) & (state.done == 0), start, state.info['step'])
    state.info['rng'] = rng
    return state


def make_randomizer(cfg):
  """Go1's domain randomization (floor friction, joint friction loss and
  armature, torso center of mass and mass, link masses, qpos0) plus an arm
  pose per environment: the fixed arm bodies turned as their joints would."""
  _, arm = build_scene()
  m = mujoco.MjModel.from_xml_path(str(BUILD / 'scene_spot_arm.xml'))
  bodies = jp.array([m.body(b).id for b, _, _, _ in arm])
  q0 = jp.array([q for _, q, _, _ in arm])
  axes = jp.array([a for _, _, a, _ in arm])
  home = jp.array([ARM_HOME[n] for _, _, _, n in arm])
  k = cfg.arm_reach
  lo = home + k * (jp.array([ARM_RANGE[n][0] for _, _, _, n in arm]) - home)
  hi = home + k * (jp.array([ARM_RANGE[n][1] for _, _, _, n in arm]) - home)

  def domain_randomize(model: mjx.Model, rng: jax.Array):
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
      body_mass = body_mass.at[torso].add(jax.random.uniform(key, minval=-1.0, maxval=1.0))
      rng, key = jax.random.split(rng)
      qpos0 = model.qpos0.at[7:].add(jax.random.uniform(key, (12,), minval=-0.05, maxval=0.05))
      # The arm: stowed, or a pose in ARM_RANGE.
      rng, k1, k2 = jax.random.split(rng, 3)
      angles = jp.where(jax.random.uniform(k1) < cfg.arm_stowed, home, jax.random.uniform(k2, home.shape, minval=lo, maxval=hi))
      quats = jax.vmap(lambda q, ax, t: jp.array(quat_mul(q, axis_angle(ax, t))))(q0, axes, angles)
      body_quat = model.body_quat.at[bodies].set(quats)
      return geom_friction, body_ipos, body_mass, qpos0, dof_frictionloss, dof_armature, body_quat

    friction, body_ipos, body_mass, qpos0, frictionloss, armature, body_quat = rand(rng)
    fields = {'geom_friction': friction, 'body_ipos': body_ipos, 'body_mass': body_mass, 'qpos0': qpos0,
              'dof_frictionloss': frictionloss, 'dof_armature': armature, 'body_quat': body_quat}
    in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({k: 0 for k in fields})
    return model.tree_replace(fields), in_axes

  return domain_randomize
