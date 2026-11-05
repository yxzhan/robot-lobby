"""SO-101 (SO-ARM101, Feetech STS3215 servos on one serial bus): the spec,
ready for the lobby's robotstudio_so101 model; the driver still to be
written. Its plan:

  connect   open the bus (feetech-servo-sdk's scservo_sdk: PortHandler at
            1 Mbaud, PacketHandler protocol 0), servo ids 1..6, torque on,
            and the calibration (`calibration`, a JSON file: per joint the
            servo's ticks at the lobby's zero and its direction; LeRobot's
            calibration files hold the same, `homing_offset` and
            `drive_mode`)
  read      GroupSyncRead of Present_Position (address 56, 2 bytes) ->
            ticks -> rad: (ticks - zero) * direction * 2*pi/4096
  command   rad -> ticks, GroupSyncWrite of Goal_Position (address 42)
  gripper   the sixth servo, a joint like the others (spec.gripper)
  hold      Goal_Position = Present_Position

The servos follow their goal on their own PID at their own speed: the
bridge's velocity limit (spec.max_vel) keeps the steps small.
"""

from __future__ import annotations

from .base import GripperSpec, RobotDriver, RobotSpec

SO101 = RobotSpec(
    name='so101',
    lobby_key='mujoco_menagerie/robotstudio_so101/so101.xml',
    joints=['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'],
    # The menagerie model's joint ranges.
    lower=[-1.91986, -1.74533, -1.69, -1.65806, -2.74385],
    upper=[1.91986, 1.74533, 1.69, 1.65806, 2.84121],
    max_vel=[3.0] * 5,
    gripper=GripperSpec(joint='gripper', open=1.74533, closed=-0.17453),
    limit_margin=0.05,
    start_tolerance=0.25,
)


class SO101Driver(RobotDriver):
  def __init__(self, spec=SO101, port='/dev/ttyACM0', calibration=None, **options):
    super().__init__(spec, **options)
    self.port = port
    self.calibration = calibration

  def connect(self):
    raise NotImplementedError('SO-101: the driver is not written yet (real_robot/drivers/so101.py has its plan)')

  def read(self):
    raise NotImplementedError

  def command(self, q):
    raise NotImplementedError

  def hold(self):
    raise NotImplementedError
