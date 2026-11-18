"""The CRAM side of the lobby's CRAM link (public/js/cram/cram.js): run in a Jupyter
kernel of a cramera Binder session (CRAM's venv, ROS sourced), once as the kernel's
first cell. The Behavior panel's trees then call pick_and_place(...) in cells of their
own (lobby.CramRun).

start() builds CRAM's world (semantic digital twin) from what the lobby has (a robot
of ROBOTS at its pose in the lobby, an object at its pose). act() then plans and runs
one action (park, pick, place) in CRAM's kinematic simulation, from the lobby's state
of that moment (joints, the object's pose, whether the lobby's robot holds it), and
gives back what the lobby's robot has to do: its actuators' targets every `dt` s. The
lobby plays them back in its physics, watches how it goes and asks for the next action
(or the same again): the loop is closed per action. pick_and_place() is the open loop
in one go (park, pick, place, park from one state).

The robot's CRAM class is made from ROBOTS' data (arm root and tip, TCP, the two
fingers' joints with their open and closed values, approach and closing axes in the
TCP frame, park pose), not hand written. Models are read from /shared/mjcf_assets,
the lobby's library, mounted in every session of the hub."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.datastructures.definitions import GripperState, StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm, HasTwoFingers
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm, EndEffector, Finger
from semantic_digital_twin.semantic_annotations.semantic_annotations import ShapeSortingPiece
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3, Pose, Vector3
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart

ASSETS = "/shared/mjcf_assets/"
DT = 0.05  # s: the kinematic executor's control step, if it does not say

# The robots CRAM can drive here, by the lobby's model path. tcp: a body made for the
# tool frame (CRAM's is a body), `offset` from `hand`. ctrl: the lobby's actuators,
# each the target of one CRAM joint, times `scale` (the Panda's gripper actuator: 0..255
# for 0..0.04 m of each finger).
ROBOTS = {
    "mjcf_assets/robots/mujoco_menagerie/franka_emika_panda/panda.xml": {
        "root": "link0", "arm_root": "link0", "arm_tip": "link7", "hand": "hand",
        "tcp": "lobby_tcp", "offset": [0, 0, 0.1034],
        "fingers": [["hand", "left_finger", "finger_joint1"], ["hand", "right_finger", "finger_joint2"]],
        "open": [0.04, 0.04], "closed": [0.0, 0.0], "approach": "z", "closing": "y",
        "park": {f"joint{i + 1}": v for i, v in enumerate([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])},
        "ctrl": {**{f"actuator{i}": [f"joint{i}", 1.0] for i in range(1, 8)},
                 "actuator8": ["finger_joint1", 255 / 0.04]},
    },
}

# Objects that look the same turned by any of these rotations of their own frame (a
# cube: the 24 that put its faces on its faces): not held, CRAM is told the one of
# those poses that stands most upright and is turned least about the vertical, so its
# one grasp of them fits however it lies.
_CUBE_TURNS = [P for P in (np.array([[sx if c == r else 0 for c in range(3)] for r, sx in zip(perm, signs)], float).T
                           for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
                           for signs in ((a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)))
               if round(np.linalg.det(P)) == 1]
SYMMETRY = {"mjcf_assets/objects/custom/cube/cube.xml": _CUBE_TURNS}


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _mat_to_quat(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    return [float(v) for v in q]


def _canonical(q, turns):
    """Of the poses `q` turned by `turns` (in its own frame), the most upright, then
    the one turned least about the vertical."""
    R = _quat_to_mat(q)

    def score(P):
        M = R @ P
        return (round(M[2, 2], 3), -abs(math.atan2(M[1, 0], M[0, 0])))
    return _mat_to_quat(R @ max(turns, key=score))


FLOOR = """<mujoco model="lobby_floor"><worldbody>
  <body name="lobby_floor" pos="0 0 -0.05"><geom name="lobby_floor" type="box" size="3 3 0.05"/></body>
</worldbody></mujoco>"""

AXES = {"x": Vector3.X, "y": Vector3.Y, "z": Vector3.Z}


class Reply(dict):
    """A cell's value the lobby reads (as application/json)."""

    def _repr_json_(self):
        return dict(self)


def _tf(x, y, z, yaw):
    return HomogeneousTransformationMatrix.from_xyz_rpy(float(x), float(y), float(z), 0.0, 0.0, float(yaw))


def add_tcp_body(world, parent, name, offset):
    """The TCP as a body fixed to `parent` at `offset`: CRAM's tool frame is a body."""
    with world.modify_world():
        b = Body(name=PrefixedName(name))
        world.add_kinematic_structure_entity(b)
        world.add_connection(FixedConnection(parent=world.get_body_by_name(parent), child=b,
                                             parent_T_connection_expression=_tf(*offset, 0.0)))
    return b


def make_robot_class(S: dict) -> type[AbstractRobot]:
    """A CRAM robot class from ROBOTS' data: one arm, a two-finger gripper."""

    def body(r, n):
        return r._world.get_body_in_branch_by_name(r, n)

    def finger_cls(i):
        @dataclass(eq=False)
        class F(Finger):
            def setup_hardware_interfaces(self):
                pass

            def setup_joint_states(self):
                return []

            @classmethod
            def setup_default_configuration_in_world_below_robot_root(cls, r):
                return cls(root=body(r, S["fingers"][i][0]), tip=body(r, S["fingers"][i][1]))
        F.__name__ = F.__qualname__ = f"LobbyFinger{i}"
        return F
    F0, F1 = finger_cls(0), finger_cls(1)

    @dataclass(eq=False)
    class LobbyGripper(EndEffector, HasTwoFingers[F0, F1]):
        def setup_hardware_interfaces(self):
            self._setup_hardware_interfaces_for_active_connections()

        def setup_joint_states(self):
            joints = [self._world.get_connection_by_name(f[2]) for f in S["fingers"]]
            return [JointState.from_mapping(name=PrefixedName("open", prefix=self.name.name), mapping=dict(zip(joints, S["open"])),
                                            state_type=GripperState.OPEN),
                    JointState.from_mapping(name=PrefixedName("close", prefix=self.name.name), mapping=dict(zip(joints, S["closed"])),
                                            state_type=GripperState.CLOSE)]

        @property
        def approach_axis(self):
            return AXES[S["approach"]](reference_frame=self.tool_frame)

        @property
        def closing_axis(self):
            return AXES[S["closing"]](reference_frame=self.tool_frame)

        @classmethod
        def setup_default_configuration_in_world_below_robot_root(cls, r):
            return cls(root=body(r, S["hand"]), tool_frame=body(r, S["tcp"]))

    @dataclass(eq=False)
    class LobbyArm(Arm[LobbyGripper]):
        def setup_hardware_interfaces(self):
            self._setup_hardware_interfaces_for_active_connections()

        def setup_joint_states(self):
            return [JointState.from_mapping(name=PrefixedName("park", prefix=self.name.name),
                                            mapping={self._world.get_connection_by_name(j): v for j, v in S["park"].items()},
                                            state_type=StaticJointState.PARK)]

        @classmethod
        def setup_default_configuration_in_world_below_robot_root(cls, r):
            return cls(root=body(r, S["arm_root"]), tip=body(r, S["arm_tip"]))

    @dataclass(eq=False)
    class LobbyRobot(AbstractRobot, HasOneArm[LobbyArm]):
        @classmethod
        def get_ros_file_path(cls):
            return ""

        @classmethod
        def _get_root_body_name(cls):
            return S["root"]

        def _setup_collision_rules(self):
            pass

        def _setup_velocity_limits(self):
            self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=defaultdict(lambda: S.get("max_velocity", 0.5)))
    return LobbyRobot


_classes = {}


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


class Task:
    """One robot and one object in a CRAM world built from the lobby's: the robot's
    model with its base at the world's origin, the object's model; poses from the
    lobby are turned into the base's frame. The world stays between actions (what
    CRAM holds stays held); each action first takes the lobby's joints and the
    object's pose."""

    def __init__(self, robot, base, obj, grasp_height=0.025):
        if robot not in ROBOTS:
            raise KeyError(f"CRAM has no robot class for {robot} (lobby_cram.ROBOTS: {', '.join(ROBOTS)})")
        self.robot, self.base, self.obj, self.grasp_height = robot, [float(v) for v in base], obj, grasp_height
        self.S = ROBOTS[robot]

    def build(self, joints, obj_pose):
        S = self.S
        world = MJCFParser.from_file(ASSETS + self.robot.removeprefix("mjcf_assets/")).create_world()
        add_tcp_body(world, S["hand"], S["tcp"], S["offset"])
        if self.robot not in _classes:
            _classes[self.robot] = make_robot_class(S)
        self.world, self.r = world, _classes[self.robot].from_world(world)
        # the lobby's floor (z 0), for CRAM's collision avoidance: without it, a plan may
        # put the hand where the floor is
        floor = MJCFParser.from_xml_string(FLOOR).create_world()
        world.merge_world_at_pose(floor, _tf(0, 0, -self.base[2], 0))
        ow = MJCFParser.from_file(ASSETS + self.obj.removeprefix("mjcf_assets/")).create_world()
        name = next(b for b in ow.bodies if b is not ow.root).name.name  # (e.g. cube.xml's "cube")
        world.merge_world_at_pose(ow, _tf(0, 0, 0, 0))
        self.body = next(b for b in world.bodies if b.name.name == name)
        self.free_parent = self.body.parent_connection.parent  # (its model's world body, on CRAM's world)
        with world.modify_world():
            self.piece = ShapeSortingPiece(root=self.body, grasp_height=self.grasp_height)
            world.add_semantic_annotations([self.piece])
        self.r.arm.end_effector.get_joint_state_by_type(GripperState.OPEN).apply_to(world)  # (nothing held)
        self.sync(joints, obj_pose)

    def sync(self, joints, obj_pose):
        """The lobby's joints {name: position} (not the fingers') and the object's pose [x, y, z, yaw] or
        [x, y, z, qw, qx, qy, qz] (its body's, in the lobby's world)."""
        w = self.world
        fingers = {f[2] for f in self.S["fingers"]}
        with w.modify_world():
            for name, q in (joints or {}).items():
                if name in fingers:
                    continue  # CRAM's are the gripper's command (open, closed); the lobby's, how far a held object lets it close
                try:
                    w.state[w.get_connection_by_name(name).raw_dof.id].position = float(q)
                except Exception:
                    pass  # a joint CRAM does not know (or one without a single DoF)
        w.notify_state_change()
        if obj_pose is not None:
            bx, by, bz, byaw = self.base
            x, y, z, *rest = [float(v) for v in obj_pose]
            q = [math.cos(rest[0] / 2), 0, 0, math.sin(rest[0] / 2)] if len(rest) == 1 else rest
            turns = SYMMETRY.get(self.obj)
            if turns and not self.attached():
                q = _canonical(_qmul([math.cos(byaw / 2), 0, 0, -math.sin(byaw / 2)], q), turns)  # (in the base's frame)
                q = _qmul([math.cos(byaw / 2), 0, 0, math.sin(byaw / 2)], q)
            c, s = math.cos(byaw), math.sin(byaw)
            dx, dy = x - bx, y - by
            qb = _qmul([math.cos(byaw / 2), 0, 0, -math.sin(byaw / 2)], q)
            with w.modify_world():  # (a pose in the world's frame: below the gripper too, once held)
                self.body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
                    c * dx + s * dy, -s * dx + c * dy, z - bz, qb[1], qb[2], qb[3], qb[0], reference_frame=w.root)

    def attached(self):
        """Whether CRAM's world has the object held (not where it was put)."""
        return self.body.parent_connection.parent not in (self.free_parent, self.world.root)  # (a place puts it on the root)

    def joints(self):
        """{name: position} of the robot's 1-DoF joints in CRAM's world."""
        out = {}
        for c in self.world.connections:
            try:
                out[c.name.name] = float(self.world.state[c.raw_dof.id].position)
            except Exception:
                pass
        return out

    def obj_in_lobby(self):
        p = self.body.global_pose.to_np()[:3, 3]
        bx, by, bz, byaw = self.base
        c, s = math.cos(byaw), math.sin(byaw)
        return [round(float(v), 4) for v in (bx + c * p[0] - s * p[1], by + s * p[0] + c * p[1], bz + p[2])]

    def node(self, action, target=None):
        r = self.r
        if action == "park":
            return ParkArmsAction(arms=r.all_arms)
        if action == "pick":
            [g] = self.piece.grasp_candidates()
            return PickUpAction(grasp=g, arm=r.arm)
        if action == "place":
            bx, by, bz, byaw = self.base
            c, s = math.cos(byaw), math.sin(byaw)
            dx, dy = target[0] - bx, target[1] - by
            p = Point3(c * dx + s * dy, -s * dx + c * dy, target[2] - bz, reference_frame=self.world.root)
            return PlaceAction(object_designator=self.piece, target_location=Pose(position=p, reference_frame=self.world.root))
        raise ValueError(f"no action {action} (park, pick, place)")

    def run(self, nodes):
        """Runs `nodes` in sequence in CRAM's kinematic simulation: the lobby's
        actuators and their targets after every control step (the world's state
        changes in between too, as the plan tries things), and the step (s)."""
        S, world = self.S, self.world
        acts = list(S["ctrl"])
        dofs = [world.get_connection_by_name(S["ctrl"][a][0]).raw_dof.id for a in acts]
        scale = [S["ctrl"][a][1] for a in acts]
        ctrl = []

        def sample():
            ctrl.append([round(float(world.state[d].position) * k, 5) for d, k in zip(dofs, scale)])
        sample()
        ex = KinematicallySimulatedPlanExecutor(StatechartContext.with_extensions(world, [RobotAccess(self.r)]))
        tick = ex.tick

        def ticked(*a, **kw):
            out = tick(*a, **kw)
            sample()
            return out
        ex.tick = ticked
        try:
            sc = Statechart(context=ex.context)
            sc.add_node(Sequence(nodes=nodes))
            ex.compile(sc)
            ex.execute()
            dt = float(ex.time) / max(1, len(ctrl) - 1) if getattr(ex, "time", None) else DT
        finally:
            ex.close()
        sample()  # (where it came to rest)
        return acts, ctrl, round(dt, 4)


TASK = None  # start()'s, for act()


def _failed(e):
    import traceback
    return Reply(ok=False, error=f"{type(e).__name__}: {e}"[:600], where=traceback.format_exc()[-1500:])


def start(robot, base, joints, obj, obj_pose, grasp_height=0.025):
    """CRAM's world for act(): `robot` (a model of ROBOTS) with its base at `base`
    [x, y, z, yaw] in the lobby, its joints {name: position}, the object `obj` (a
    model path) at `obj_pose` ([x, y, z, yaw] or [x, y, z, qw, qx, qy, qz]: its
    body's)."""
    global TASK
    t0 = time.time()
    try:
        TASK = Task(robot, base, obj, grasp_height)
        TASK.build(joints, obj_pose)
        return Reply(ok=True, secs=round(time.time() - t0, 2), obj=TASK.obj_in_lobby())
    except Exception as e:
        TASK = None
        return _failed(e)


def act(action, joints, obj_pose, held=None, target=None):
    """One action, 'park', 'pick' or 'place' (at `target` [x, y, z]: where the
    object's body goes, in the lobby's world), planned from the lobby's state now:
    its joints {name: position}, the object's pose (as start's) and whether the
    lobby sees it `held` (None: as CRAM has it). CRAM's world holding the object
    while the lobby's robot does not (a grasp that missed, the object dropped): the
    world is built anew from the lobby's state. Returns Reply {ok, actuators, joints
    ([joint, scale] each actuator's target is of), dt, ctrl: [[target per actuator]
    every dt s], held (CRAM's, after it), obj (where
    CRAM has the object after it, in the lobby's world), rebuilt, secs} or {ok:
    False, error}."""
    t0 = time.time()
    if TASK is None:
        return Reply(ok=False, error="no task: start(...) first")
    try:
        rebuilt = held is False and TASK.attached()
        if rebuilt:
            if obj_pose is None:
                return Reply(ok=False, error="the object's pose is needed (CRAM holds it, the lobby does not)")
            TASK.build({**TASK.joints(), **(joints or {})}, obj_pose)
        else:
            TASK.sync(joints, obj_pose)
        acts, ctrl, dt = TASK.run([TASK.node(action, target)])
        return Reply(ok=True, actuators=acts, joints=[TASK.S["ctrl"][a] for a in acts], dt=dt, ctrl=ctrl, held=TASK.attached(), obj=TASK.obj_in_lobby(),
                     rebuilt=rebuilt, secs=round(time.time() - t0, 2))
    except Exception as e:
        return _failed(e)


def pick_and_place(robot, base, joints, obj, obj_pose, target, grasp_height=0.025):
    """Open loop, in one go: park, pick up `obj`, place it at `target`, park, all
    planned from the state given (as start's and act's). Returns Reply {ok,
    actuators, dt, ctrl, secs, end (the object's body at the end, in the lobby's
    world)} or {ok: False, error}."""
    t0 = time.time()
    try:
        t = Task(robot, base, obj, grasp_height)
        t.build(joints, obj_pose)
        acts, ctrl, dt = t.run([t.node("park"), t.node("pick"), t.node("place", target), t.node("park")])
        return Reply(ok=True, actuators=acts, dt=dt, ctrl=ctrl, secs=round(time.time() - t0, 2), end=t.obj_in_lobby())
    except Exception as e:
        return _failed(e)
