# Tracy picks up a cube and puts it down again, by its IK (the lobby's
# ik_chains / set_ik_target / set_gripper): the tree spawns Tracy if there
# is none, turns its IK on and the noslip post-pass up (100 iterations: a
# held cube does not creep out of the fingers), drops a cube on the table
# right under the left gripper, then opens, goes down, closes, lifts, carries
# it across, lowers it, opens and goes back up. Each move walks the target
# along a straight line at SPEED (a target set far away at once would swing
# the arm through anything on the way), and is done once the tip has caught
# up with it.
import math
import py_trees
from py_trees.common import Status
import lobby

ROBOT_MODEL = "tracy"
CUBE_MODEL = "mjcf_assets/objects/custom/cube/cube.xml"
CHAIN = "left_wrist_3"     # the left arm (its target: between the fingertips)
TABLE_Z = 0.88             # Tracy's table top (m, in Tracy's frame: spawned at the origin)
CUBE_HALF = 0.025          # the cube's half size (m)
GRASP_DOWN = 0.01          # m: the grasp point this much under the cube's centre (fingers deep enough)
LIFT = 0.15                # m over the table to carry it
CARRY = (0.0, -0.15)       # where it is put down: this far (x, y) from where it was picked up
SPEED = 0.08               # m/s of the target along a move
REACHED = 0.01             # m: the tip this near the end of a move: done
NOSLIP = 100
plan = {}                  # where things are, noted by Setup: {robot, start (the target then), cube}


def tracy():
    return next((r for r in lobby.robots() if r["label"] == ROBOT_MODEL), None)


def chain():
    return next((c for c in lobby.ik_chains(plan["robot"]) if c["name"] == CHAIN), None)


class Setup(py_trees.behaviour.Behaviour):
    """Tracy (spawned if missing), its IK on, noslip up, and a cube under the left gripper."""

    def __init__(self):
        super().__init__("Tracy, IK, noslip, cube")
        self.step = "tracy"

    def update(self):
        r = tracy()
        if self.step == "tracy":
            if r is None:
                lobby.spawn(ROBOT_MODEL, 0, 0, 0)
                self.feedback_message = "spawning Tracy"
                self.step = "wait tracy"
                return Status.RUNNING
            self.step = "ik"
        if self.step == "wait tracy":
            if r is None:
                return Status.RUNNING
            self.step = "ik"
        plan["robot"] = r["name"]
        if self.step == "ik":
            lobby.physics(noslip_iterations=NOSLIP)
            if not lobby.controllers(r["name"])["ik"]:
                lobby.set_controller(r["name"], "ik", True)
            self.step = "target"
            return Status.RUNNING
        if self.step == "target":
            c = chain()
            if c is None or c.get("target") is None:  # (IK coming on; in a shared world the server's target arrives)
                self.feedback_message = "waiting for the IK chain"
                return Status.RUNNING
            plan["start"] = c["target"]
            x, y, _ = c["target"]["pos"]
            lobby.spawn(CUBE_MODEL, x, y, 0, TABLE_Z)
            plan["cube"] = (x, y)
            self.feedback_message = f"cube at {x:.2f}, {y:.2f}"
            self.step = "wait cube"
            return Status.RUNNING
        if self.step == "wait cube":
            if lobby.body("cube") is None:
                return Status.RUNNING
            return Status.SUCCESS
        return Status.RUNNING


class Tip(py_trees.behaviour.Behaviour):
    """The left gripper's target along a straight line to where(plan) ([x, y, z]), at SPEED;
    its orientation as at the start. Done when the tip is within REACHED of the end."""

    def __init__(self, name, where):
        super().__init__(name)
        self.where = where

    def initialise(self):
        self.t0 = lobby.time()
        self.a = chain()["target"]["pos"]
        self.b = self.where(plan)
        self.length = math.dist(self.a, self.b)

    def update(self):
        k = min(1.0, (lobby.time() - self.t0) * SPEED / max(self.length, 1e-6))
        lobby.set_ik_target(plan["robot"], CHAIN, [a + (b - a) * k for a, b in zip(self.a, self.b)], plan["start"]["quat"])
        tip = chain()["tip_pose"]["pos"]
        off = math.dist(tip, self.b)
        self.feedback_message = f"{off * 100:.1f} cm to go"
        return Status.SUCCESS if k >= 1 and off < REACHED else Status.RUNNING


class Grip(py_trees.behaviour.Behaviour):
    """The left gripper to `amount` (0 open, 1 closed), then `seconds` (simulated) for it to get there."""

    def __init__(self, name, amount, seconds=1.0):
        super().__init__(name)
        self.amount, self.seconds = amount, seconds

    def initialise(self):
        lobby.set_gripper(plan["robot"], CHAIN, self.amount)
        self.t0 = lobby.time()

    def update(self):
        return Status.SUCCESS if lobby.time() - self.t0 >= self.seconds else Status.RUNNING


class Report(py_trees.behaviour.Behaviour):
    """How high the cube is (the feedback), checking it is held: over the table by at least `above`."""

    def __init__(self, name, above):
        super().__init__(name)
        self.above = above

    def update(self):
        z = lobby.body("cube")["z"] - CUBE_HALF - TABLE_Z
        self.feedback_message = f"cube {z * 100:.1f} cm over the table"
        return Status.SUCCESS if z >= self.above else Status.FAILURE


def at(dx=0.0, dy=0.0, z=0.0):
    """A point over the cube's spot (moved by dx, dy), z over the table top."""
    return lambda p: [p["cube"][0] + dx, p["cube"][1] + dy, TABLE_Z + z]


grasp = CUBE_HALF - GRASP_DOWN
root = py_trees.composites.Sequence("Tracy pick and place", memory=True, children=[
    Setup(),
    Grip("open", 0.0),
    Tip("down to the cube", at(z=grasp)),
    Grip("close", 1.0, 1.5),
    Tip("lift", at(z=LIFT)),
    Report("holding it?", LIFT - 0.05),
    Tip("carry", at(*CARRY, z=LIFT)),
    Tip("lower", at(*CARRY, z=grasp + 0.005)),
    Grip("open", 0.0),
    Tip("up", at(*CARRY, z=LIFT)),
    Report("put down?", -0.01),
])
