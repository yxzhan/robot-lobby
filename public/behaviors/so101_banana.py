# The SO-ARM101 puts a banana on a plate, by its IK (the lobby's ik_chains /
# set_ik_target / set_gripper). The tree sees that the world has a table, the
# arm on it and a plate and a banana before it (lobby.Spawn: what is already
# there is reset, not spawned again, and nothing else is touched; so Restart
# resets the scene), then takes the banana from above, across its middle,
# lifts it and carries it over the plate (from the lift on, the gripper's
# orientation left free: held down, the tip reaches little) and lets go there.
#
# The gripper comes straight down: its tip frame's x (the approach) pointing
# down, its z (the way the jaws close) across the banana. Each move walks the
# target along a straight line at SPEED (turning at TURN_SPEED) and is done
# once the tip has caught up; a move with no orientation is a free target
# (set_ik_target(..., free=True)): the IK only minds the position.
import math
import py_trees
from py_trees.common import Status
import lobby

TABLE, ARM, PLATE, BANANA = "ikea_table", "robotstudio_so101", "plate_1", "banana_1"
TOP = 0.75                    # the table top (m)
ARM_AT = (-0.45, 0.0)         # the arm's base on the table, reaching towards +x
PLATE_AT = (-0.15, -0.2)
BANANA_AT, BANANA_YAW = (-0.20, 0.18), 0.3
THINGS = [(TABLE, 0, 0, 0, 0), (ARM, *ARM_AT, 0, TOP), (PLATE, *PLATE_AT, 0, TOP), (BANANA, *BANANA_AT, BANANA_YAW, TOP)]
CHAIN = "gripper"
# The banana in its body frame (banana_1.xml): its middle, and its long axis's angle.
BANANA_MID, BANANA_AXIS = (0.018, 0.012), math.radians(116)
GRASP_Z = -0.004             # the tip (gripperframe, at the fingertips) over the banana's body origin (its middle height) when gripping
ABOVE = 0.08                 # m: the approach, over the grasp
LIFT = 0.2                  # m: the banana lifted this far over the grasp
DROP_Z = 0.10                # m: the tip over the plate's base when letting go
SPEED, TURN_SPEED = 0.12, 1.5  # m/s, rad/s of the target
NOSLIP = 10                  # the noslip post-pass's iterations (a held banana slips less)
CONE = "pyramidal"           # the friction cone
REACHED, SETTLE = 0.008, 1.5   # m: a move is done this near; or this long (s) after its end
plan = {}  # noted by Ready: arm, plate, banana (names); grasp (pos, yaw)


# -- rotations: row-major 3x3 as lists; quaternions (w, x, y, z) ------------

def matmul(a, b):
    return [sum(a[3 * i + k] * b[3 * k + j] for k in range(3)) for i in range(3) for j in range(3)]


def rot(axis, rad):
    c, s = math.cos(rad), math.sin(rad)
    return {"y": [c, 0, s, 0, 1, 0, -s, 0, c], "z": [c, -s, 0, s, c, 0, 0, 0, 1]}[axis]


def to_quat(m):
    a, b, c, d, e, f, g, h, i = m
    t = a + e + i
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = [s / 4, (h - f) / s, (c - g) / s, (d - b) / s]
    elif a > e and a > i:
        s = 2 * math.sqrt(1 + a - e - i)
        q = [(h - f) / s, s / 4, (b + d) / s, (c + g) / s]
    elif e > i:
        s = 2 * math.sqrt(1 + e - a - i)
        q = [(c - g) / s, (b + d) / s, s / 4, (f + h) / s]
    else:
        s = 2 * math.sqrt(1 + i - a - e)
        q = [(d - b) / s, (c + g) / s, (f + h) / s, s / 4]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def slerp(p, q, t):
    dot = sum(a * b for a, b in zip(p, q))
    if dot < 0:
        q, dot = [-v for v in q], -dot
    if dot > 0.9995:
        r = [a + (b - a) * t for a, b in zip(p, q)]
    else:
        th = math.acos(dot)
        r = [(math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th) for a, b in zip(p, q)]
    n = math.sqrt(sum(v * v for v in r))
    return [v / n for v in r]


def angle(p, q):
    return 2 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(p, q)))))


def down(yaw):
    """The tip straight down (its x to -z), its z (the jaws' closing) along `yaw` in the world."""
    return to_quat(matmul(rot("z", yaw), rot("y", math.pi / 2)))


def across(yaw):
    """The jaws' closing direction, of the two equal ones (yaw, yaw + pi), in
    [-pi/2, pi/2): the one the wrist rolls to without a half turn."""
    return (yaw + math.pi / 2) % math.pi - math.pi / 2


def chain():
    return next(c for c in lobby.ik_chains(plan["arm"]) if c["name"] == CHAIN)


# -- behaviours --------------------------------------------------------------

class Ready(py_trees.behaviour.Behaviour):
    """Noslip and the cone set, the arm's IK on and its target known; where the banana is to be taken."""

    def __init__(self, spawn):
        super().__init__("IK on, find the banana")
        self.spawn = spawn

    def update(self):
        plan["arm"], plan["plate"], plan["banana"] = self.spawn.names[1:]
        ph = lobby.physics()
        if ph.get("noslip_iterations") != NOSLIP or ph.get("cone") != CONE:
            lobby.physics(noslip_iterations=NOSLIP, cone=CONE)
        if not lobby.controllers(plan["arm"])["ik"]:
            lobby.set_controller(plan["arm"], "ik", True)
            return Status.RUNNING
        c = next((c for c in lobby.ik_chains(plan["arm"]) if c["name"] == CHAIN), None)
        if c is None or c.get("target") is None or c.get("tip_pose") is None:
            self.feedback_message = "waiting for the IK chain"
            return Status.RUNNING
        b = lobby.body(f'{plan["banana"]}/object')
        if b is None:
            return Status.RUNNING
        cy, sy = math.cos(b["yaw"]), math.sin(b["yaw"])
        mx, my = BANANA_MID
        plan["grasp"] = ([b["x"] + cy * mx - sy * my, b["y"] + sy * mx + cy * my, b["z"] + GRASP_Z],
                         across(b["yaw"] + BANANA_AXIS + math.pi / 2))
        self.feedback_message = "banana at ({:.2f}, {:.2f})".format(*plan["grasp"][0][:2])
        return Status.SUCCESS


class Move(py_trees.behaviour.Behaviour):
    """The target along a straight line to where(plan) ((pos, quat)), at
    SPEED and TURN_SPEED; quat None: the position only, the IK turning the
    tip as it likes (set_ik_target's free). Done when the tip is within REACHED of its end, or
    SETTLE after the walk ended."""

    def __init__(self, name, where):
        super().__init__(name)
        self.where = where

    def initialise(self):
        t = chain()["target"]
        self.start = (t["pos"], t["quat"])
        self.end = self.where(plan)
        turn = 0.0 if self.end[1] is None else angle(self.start[1], self.end[1]) / TURN_SPEED
        self.secs = max(math.dist(self.start[0], self.end[0]) / SPEED, turn, 0.1)
        self.t0 = lobby.time()

    def update(self):
        elapsed = lobby.time() - self.t0
        k = min(1.0, elapsed / self.secs)
        (p0, q0), (p1, q1) = self.start, self.end
        tip = chain().get("tip_pose")
        pos = [a + (b - a) * k for a, b in zip(p0, p1)]
        if q1 is None:
            lobby.set_ik_target(plan["arm"], CHAIN, pos, free=True)
        else:
            lobby.set_ik_target(plan["arm"], CHAIN, pos, slerp(q0, q1, k))
        far = math.dist(tip["pos"], p1) if tip else 1.0
        self.feedback_message = f"{far * 100:.1f} cm to go"
        return Status.SUCCESS if k >= 1 and (far < REACHED or elapsed > self.secs + SETTLE) else Status.RUNNING


class Grip(py_trees.behaviour.Behaviour):
    """The gripper to `amount` (0 open, 1 closed), then `seconds` (simulated) for it to get there."""

    def __init__(self, name, amount, seconds=1.0):
        super().__init__(name)
        self.amount, self.seconds = amount, seconds

    def initialise(self):
        lobby.set_gripper(plan["arm"], CHAIN, self.amount)
        self.t0 = lobby.time()

    def update(self):
        return Status.SUCCESS if lobby.time() - self.t0 >= self.seconds else Status.RUNNING


class Check(py_trees.behaviour.Behaviour):
    """Whether the banana is on the plate (within its rim, not under it)."""

    def __init__(self):
        super().__init__("on the plate?")

    def update(self):
        b, p = lobby.body(f'{plan["banana"]}/object'), lobby.body(f'{plan["plate"]}/object')
        off = math.hypot(b["x"] - p["x"], b["y"] - p["y"])
        self.feedback_message = f"{off * 100:.1f} cm from the plate's middle"
        return Status.SUCCESS if off < 0.08 and b["z"] > p["z"] else Status.FAILURE


# -- where the tip goes -------------------------------------------------------

def over_banana(dz, free=False):
    """Over the grasp; gripper down across the banana, or (free) any way the IK likes."""
    return lambda p: ([p["grasp"][0][0], p["grasp"][0][1], p["grasp"][0][2] + dz], None if free else down(p["grasp"][1]))


def over_plate(dz):
    def where(p):
        b = lobby.body(f'{p["plate"]}/object')
        return [b["x"], b["y"], b["z"] + GRASP_Z + dz], None
    return where


spawn = lobby.Spawn("table, arm, plate, banana", THINGS, near=None)
root = py_trees.composites.Sequence("soarm101 pick&place", memory=True, children=[
    spawn,
    Ready(spawn),
    Grip("open", 0.0, 0.8),
    Move("over the banana", over_banana(ABOVE)),
    Move("down to it", over_banana(0.0)),
    Grip("close", 1.0, 1.2),
    Move("lift", over_banana(LIFT, free=True)),
    Move("over the plate", over_plate(DROP_Z)),
    Grip("let go", 0.0, 1.0),
    Check(),
])
