# Mobile bases

Every robot gets a `BaseController` (`public/js/controllers/base/base.js`), driven
from the Teleop panel like the G1 (**WASD**, **QE**), with a PI loop on the
floating base's measured twist (ground truth as odometry) and an
acceleration limit on the command (a step to full speed tips a tall robot
over). With no command the wheels brake. The panel shows command and
measurement and edits the setup; **Save** keeps it per robot file
(`mujoco-filament/base/<path>`), **Auto** / **Defaults** goes back to the
found / built-in one, **Export** copies JSON for `BASE_DEFAULTS`.

Kinds:
- **wheels**: a free root joint on driven wheels (velocity servos, or
  motors with a P loop on wheel speed). Wheel positions and spin axes come
  from the model; each wheel turns at `ω = g·v / (r g·d)`, `v` the wheel
  center's velocity for the commanded twist, `d` its rolling direction and
  `g` the direction it pushes: `d` for a plain wheel (differential, skid
  steer; no sideways motion), `d ± axis` for mecanum rollers at ±45°.
  A wheel given as a joint instead of an actuator runs on a software
  velocity servo (every physics step, a joint force with feedforward
  against its damping and friction loss), and an actuator on that joint is
  switched off meanwhile; when no wheel has an actuator, `autoConfig()`
  takes the unactuated level hinges hanging right off the base this way.
  Configurable: base joint, wheels (actuator or joint, roller), radius, key
  speeds, feedforward / P / I / integral limit / acceleration per axis.
- **planar**: x / y / yaw position servos on the root body (TidyBot): the
  pose target is integrated from the command.

`autoConfig()` finds them: wheels are hinge joints with a velocity servo
or motor, a level axis and a low axle; a wheel carrying unactuated roller
hinges is mecanum, the side its lowest roller leans to giving ±45°; the
radius is the axle height in the default pose. Garmi has a built-in setup
(its mecanum base: forward and sideways track open-loop; its rollers give
yaw a lever of only |x − y| = 4 cm, hence a yaw feedforward of 3). Checked:
Garmi, TIAGo (position / velocity / motor variants: forward to 0.01 m/s),
TidyBot (planar). Model limits: RB-Y1's base mesh rests on the floor (no
casters), so its wheels cannot move it; TIAGo++'s omni base has no rollers
(no sideways motion, a skid-steer box). Stretch 3's wheel velocity servos
chatter (the left one flips ±2.3 rad/s every step, in plain MuJoCo too), so
its built-in setup drives the wheel joints with software servos (0.3 m/s
and 1 rad/s tracked).
