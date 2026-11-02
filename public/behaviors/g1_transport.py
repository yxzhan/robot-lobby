# A Unitree G1 with hands sorts four bottles, two of water and two of a
# drink, from the kitchen counter of IAI's apartment onto two tables in it,
# the water onto the dining table and the drinks onto the standing table,
# two at a time, one in each hand (the lobby's goals, IK targets and
# grippers). The tree clears the world, spawns the apartment (at 5, -5,
# turned 180 degrees), the G1 and the bottles on the counter mixed (each pair
# of them one of each), one at a time (a spawn while another one is still
# being added can lose it), and turns its IK on and the noslip post-pass up
# (a held bottle creeps less).
#
# Both hands go up first: the G1's default pose holds its forearms out at
# table height, and walking up to the table they would sweep a bottle over.
# Each pair: the G1 walks to where it stands with the two bottles before it,
# one a little to its left, one to its right. Each hand, together, opens
# beside its bottle on the outside (palm facing it, fingers forward), moves
# in, closes, and lifts. A bottle that did not come up is tried again
# (RETRIES times); one that has fallen over is left, and that hand comes up
# empty. Then it goes to each table a bottle in its hands belongs on (by the
# bottle's model, DEST) and puts that one down there: lowers it until its
# base is just on its spot (allowing for where it sits in the fingers),
# opens, and takes the hand out sideways before raising it (straight up,
# the fingers would catch the bottle and tip it); then it steps BACK from
# the table (walking on along its edge, its legs catch on it). It comes to a
# table to put down from APPROACH further out first, then up to it. Every
# position is worked out from where the bottles and the robot are then, not
# where they were meant to be: walking ends a few cm off. A walk that does
# not get there in WALK_TIMEOUT fails the tree. At the end it checks every
# bottle stands on its table, and fails if one does not.
import math
import py_trees
from py_trees.common import Status
import lobby

ROBOT_MODEL = "unitree_g1_with_hands"
APARTMENT = "mjcf_assets/environments/aicor/iai_apartment_usd/iai_apartment_usd.xml"
APARTMENT_POSE = (5.0, -5.0, math.pi)  # x, y, yaw
WATER, DRINK = "milk_0", "bottled_drink_1"
HALF = {WATER: 0.08, DRINK: 0.0825}  # half a bottle's height: its body is at its middle
GRASP_DZ = {WATER: 0.02, DRINK: 0.005}  # where it is taken, from its middle (the drink's narrows higher up)
LEFT, RIGHT = "left_wrist_yaw", "right_wrist_yaw"  # the IK chains
SIGN = {LEFT: 1, RIGHT: -1}  # the side of the robot each hand works on (+1: its left)
NAME = {LEFT: "left", RIGHT: "right"}
# The apartment's tables, with it where APARTMENT_POSE puts it: 0 the kitchen
# counter (along y, its front at x -1.3), 1 the dining table, 2 the standing table.
TOP = {0: 0.932, 1: 0.755, 2: 1.05}    # their tops (m)
BOUNDS = {1: ((0.24, 1.64), (1.12, 2.02)), 2: ((1.49, 2.19), (-2.15, -1.45))}  # x, y extents
START = (0.0, 0.0, math.pi)            # where the G1 is spawned and ends: x, y, yaw
PICKS = [(-1.4, 0.25), (-1.4, -0.25)]  # the pairs' middles on the counter, along y
# The bottles of each pair as spawned, from the left hand's side, then the right's:
# one of each, the other way round in the second pair.
KINDS = [(DRINK, WATER), (WATER, DRINK)]
DEST = {WATER: 1, DRINK: 2}            # the table each goes on
# Where the pairs' bottles go on each table (the middle the robot stands before,
# per pair), along x near its edge the robot comes from: the dining table's -y
# one, the standing table's +y one (only 0.7 m wide, x 1.49 to 2.19). Facing -y
# there, the robot's left is +x: the first drink (left hand) goes at 1.77, the
# second (right hand) at 2.05, 0.28 m apart.
PLACES = {1: [(0.75, 1.3), (1.2, 1.3)], 2: [(1.85, -1.55), (1.8, -1.55)]}
PICK_YAW = math.pi                                # facing the counter
PLACE_YAW = {1: math.pi / 2, 2: -math.pi / 2}     # facing each table
# m: it steps back this much after putting down, and comes to a table to put
# down from APPROACH further out first (walking along a table's edge, its legs catch on it)
BACK, APPROACH = 0.35, 0.40
WALK_TIMEOUT = 40.0         # s: a walk not there by then fails (stuck on something)
STILL = 1.0                 # s: after a walk, it stands this long to settle before its hands move
REACH = (0.30, 0.12)        # each bottle from the robot when it works: forward, to its hand's side (m)
SIDE = 0.03                 # m: a hand opens this far outside its bottle
PALM = 0.02                # m: and closes this far (the tip is on the palm: at the middle, it pushes the bottle away)
LIFT = 0.12                 # m: a bottle lifted this much
PLACE_DZ = 0.08             # m: a bottle let go this much over the table (lowered onto it, it may be pushed into it and tip)
# Its default pose holds both forearms out at table height: walking up to the
# table, a hand sweeps a bottle over. So they are up out of the way first.
# A carried bottle's base is ~0.1 m below the tip: over the standing table's top (1.05 m).
CARRY = (0.25, 0.20, 1.22)  # where a hand is carried with a bottle: forward, to its side, height (m)
READY_Z = 1.15              # m: and an empty one (the ready pose), there but lower
SPEED = 0.25                # m/s of a hand's target
REACHED, SETTLE = 0.015, 1.5  # m: a move is done this near; or this long (s) after its end
RETRIES = 2
NEAR = 0.12                 # m: a bottle further than this from where it was spawned is not taken
NOSLIP = 10
plan = {"grip": {}, "has": {}, "held": {}}  # robot, bottles [(name, model)] noted by Setup; per hand: gripper, bottle in it, where it sits


def first(label):
    return next((r for r in lobby.robots() if r["label"] == label), None)


def bottle(i):
    """{x, y, z, yaw, up} of bottle i."""
    return lobby.body(f'{plan["bottles"][i][0]}/root')


def kind(i):
    return plan["bottles"][i][1]


def middle_z(i, t=0):
    """The height of bottle i's middle standing on table t."""
    return TOP[t] + HALF[kind(i)]


def which(k, c):
    """The bottle hand c takes of pair k (spawned left, right, pair by pair)."""
    return 2 * k + (0 if c == LEFT else 1)


def robot_pose():
    return lobby.pose(plan["robot"])


def ahead(pose, forward, left):
    """A point `forward` and `left` (m) of `pose` ({x, y, yaw}) on the floor."""
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    return pose["x"] + c * forward - s * left, pose["y"] + s * forward + c * left


def spot(middle, yaw, c):
    """Hand c's bottle of the pair at `middle`, for the robot facing yaw."""
    return middle[0] - math.sin(yaw) * SIGN[c] * REACH[1], middle[1] + math.cos(yaw) * SIGN[c] * REACH[1]


def hand_quat(yaw):
    """A hand level, fingers forward, palm facing in (each hand's bottle is
    on its palm's side): the robot's frame, turned to yaw, for both."""
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def tip(c):
    ch = next(x for x in lobby.ik_chains(plan["robot"]) if x["name"] == c)
    return ch.get("tip_pose"), ch.get("target")


# -- behaviours --------------------------------------------------------------

class Setup(py_trees.behaviour.Behaviour):
    """An empty world, then the apartment, the G1 and the bottles, one at a time;
    the G1's IK on, noslip up."""

    def __init__(self):
        super().__init__("Apartment, G1, bottles")

    def initialise(self):
        self.spots = [(model, *spot(m, PICK_YAW, c)) for m, kinds in zip(PICKS, KINDS) for c, model in zip((LEFT, RIGHT), kinds)]
        self.todo = ([(APARTMENT, *APARTMENT_POSE, 0.0), (ROBOT_MODEL, *START, 0.0)]
                     + [(model, x, y, 0.0, TOP[0]) for model, x, y in self.spots])
        self.step, self.count, self.known, self.names = "clear", 0, set(), []
        lobby.clear_world()

    def update(self):
        listed = lobby.robots()
        if self.step == "clear":
            if listed:
                self.feedback_message = "clearing the world"
                return Status.RUNNING
            self.step = "spawn"
        if self.step == "spawn":
            if len(listed) < self.count:  # (the one spawned last not in yet)
                return Status.RUNNING
            # The one spawned last: the name robots() did not list before.
            new = [r["name"] for r in listed if r["name"] not in self.known]
            if self.count and len(new) != 1:
                self.feedback_message = f"spawned {len(new)} at once: {new}"
                return Status.FAILURE
            self.names += new
            self.known |= set(new)
            if self.todo:
                model, x, y, yaw, z = self.todo.pop(0)
                lobby.spawn(model, x, y, yaw, z)
                self.count = len(listed) + 1
                self.feedback_message = f"spawning {model.split('/')[-1]}"
                return Status.RUNNING
            plan["robot"] = first(ROBOT_MODEL)["name"]
            # Each spot's bottle: the one spawned there, by spawn order (the
            # nearest one of its model could be the same for two spots).
            plan["bottles"] = [(name, model) for name, (model, x, y) in zip(self.names[-len(self.spots):], self.spots)]
            plan["spots"] = [(x, y) for _, x, y in self.spots]
            lobby.physics(noslip_iterations=NOSLIP)
            if not lobby.controllers(plan["robot"])["ik"]:
                lobby.set_controller(plan["robot"], "ik", True)
            self.step = "ik"
        if self.step == "ik":
            if (any(None in tip(c) for c in (LEFT, RIGHT))
                    or any(bottle(i) is None for i in range(2 * len(PICKS)))):
                self.feedback_message = "waiting for the IK chains and the bottles"
                return Status.RUNNING
            return Status.SUCCESS
        return Status.RUNNING


class Walk(py_trees.behaviour.Behaviour):
    """The robot to where(), {x, y, yaw}, by its goal; done once it has arrived
    there and stood STILL. (Right after the goal is set, robots() can still say
    it has arrived, at the last one: that counts only once the new goal has
    shown, or near it.)"""

    def __init__(self, name, where):
        super().__init__(name)
        self.where = where

    def initialise(self):
        self.goal = self.where()
        self.t0, self.seen, self.there = lobby.time(), False, None
        lobby.set_goal(plan["robot"], self.goal["x"], self.goal["y"], self.goal["yaw"])

    def update(self):
        r = next((r for r in lobby.robots() if r["name"] == plan["robot"]), None)
        if r is None:
            return Status.FAILURE
        p = r["pose"]
        far = math.hypot(p["x"] - self.goal["x"], p["y"] - self.goal["y"])
        self.feedback_message = f'{far:.2f} m to go'
        self.seen = self.seen or r.get("goal") is not None
        if self.there is not None or (r.get("arrived") and (self.seen or far < 0.05)):
            self.there = self.there if self.there is not None else lobby.time()
            self.feedback_message = "there, settling"
            return Status.SUCCESS if lobby.time() - self.there >= STILL else Status.RUNNING
        if lobby.time() - self.t0 > WALK_TIMEOUT:
            self.feedback_message += f": stuck, not there after {WALK_TIMEOUT:.0f} s"
            lobby.stop(plan["robot"])
            return Status.FAILURE
        return Status.RUNNING

    def terminate(self, new_status):
        if new_status == Status.INVALID:
            lobby.stop(plan["robot"])


class Reach(py_trees.behaviour.Behaviour):
    """Hand c's target along a straight line to where() ([x, y, z]) at SPEED,
    level and facing the robot's way. Done when the tip is within REACHED,
    or SETTLE after the line's end (a bottle in the way, a table)."""

    def __init__(self, name, c, where):
        super().__init__(name)
        self.c, self.where = c, where

    def initialise(self):
        self.start = None

    def update(self):
        if self.start is None:
            _, target = tip(self.c)
            if target is None:
                self.feedback_message = "waiting for the arm's target"
                return Status.RUNNING
            self.start, self.end = target["pos"], self.where()
            self.quat = hand_quat(robot_pose()["yaw"])
            self.t0 = lobby.time()
            self.secs = math.dist(self.start, self.end) / SPEED
        elapsed = lobby.time() - self.t0
        k = min(1.0, elapsed / max(self.secs, 1e-6))
        lobby.set_ik_target(plan["robot"], self.c, [a + (b - a) * k for a, b in zip(self.start, self.end)], self.quat)
        pose, _ = tip(self.c)
        far = math.dist(pose["pos"], self.end) if pose else 1.0
        self.feedback_message = f"{far * 100:.1f} cm to go"
        return Status.SUCCESS if k >= 1 and (far < REACHED or elapsed > self.secs + SETTLE) else Status.RUNNING


class Grip(py_trees.behaviour.Behaviour):
    """Hand c from where it is to `amount` (0 open, 1 closed) over `seconds`
    (simulated) for the whole way, less for part of it (none: already there):
    closing slowly, the fingers do not knock the bottle away."""

    def __init__(self, name, c, amount, seconds=1.5):
        super().__init__(name)
        self.c, self.amount, self.seconds = c, amount, seconds

    def initialise(self):
        self.t0, self.from_ = lobby.time(), plan["grip"].get(self.c, 0.0)
        self.secs = self.seconds * abs(self.amount - self.from_)

    def update(self):
        k = min(1.0, (lobby.time() - self.t0) / self.secs) if self.secs > 0 else 1.0
        plan["grip"][self.c] = self.from_ + (self.amount - self.from_) * k
        lobby.set_gripper(plan["robot"], self.c, plan["grip"][self.c])
        return Status.SUCCESS if k >= 1 else Status.RUNNING


class Upright(py_trees.behaviour.Behaviour):
    """Whether bottle i stands on the counter by its spot (a fallen one is left: it
    cannot be taken from the side; one gone elsewhere is not reached for)."""

    def __init__(self, i):
        super().__init__(f"bottle {i + 1} standing at its spot?")
        self.i = i

    def update(self):
        b, (x, y) = bottle(self.i), plan["spots"][self.i]
        off = math.hypot(b["x"] - x, b["y"] - y)
        if off > NEAR or abs(b["z"] - middle_z(self.i)) > 0.05:
            self.feedback_message = f'{off * 100:.0f} cm from its spot, {b["z"]:.2f} m high: not there, left'
            return Status.FAILURE
        up = b["up"]
        self.feedback_message = "standing" if up > 0.9 else "fallen over: left where it is"
        return Status.SUCCESS if up > 0.9 else Status.FAILURE


class Holding(py_trees.behaviour.Behaviour):
    """Whether bottle i came up in hand c (its middle LIFT / 2 above the table's
    top); noted in plan["has"]."""

    def __init__(self, i, c):
        super().__init__(f"bottle {i + 1} in the {NAME[c]} hand?")
        self.i, self.c = i, c

    def update(self):
        b = bottle(self.i)
        plan["has"][self.c] = b["z"] > middle_z(self.i) + LIFT / 2
        self.feedback_message = f'its middle {b["z"]:.2f} m high'
        return Status.SUCCESS if plan["has"][self.c] else Status.FAILURE


class Has(py_trees.behaviour.Behaviour):
    """Whether hand c carries bottle i and it belongs on table t; if so, notes
    where it sits in the hand (from the tip), so it is put down on its spot
    and its base just onto the table, not the tip (it may have slipped in the
    fingers, down too)."""

    def __init__(self, i, c, t):
        super().__init__(f"bottle {i + 1} in the {NAME[c]} hand, for this table?")
        self.i, self.c, self.t = i, c, t

    def update(self):
        if not plan["has"].get(self.c) or DEST[kind(self.i)] != self.t:
            self.feedback_message = "no"
            return Status.FAILURE
        b, (pose, _) = bottle(self.i), tip(self.c)
        plan["held"][self.c] = (b["x"] - pose["pos"][0], b["y"] - pose["pos"][1], b["z"] - pose["pos"][2])
        self.feedback_message = "its middle {:.1f}, {:.1f}, {:.1f} cm off the tip".format(*(v * 100 for v in plan["held"][self.c]))
        return Status.SUCCESS


class Empty(py_trees.behaviour.Behaviour):
    """Hand c carries nothing (it could not take its bottle)."""

    def __init__(self, c):
        super().__init__(f"the {NAME[c]} hand empty")
        self.c = c

    def update(self):
        plan["has"][self.c] = False
        return Status.SUCCESS


class For(py_trees.behaviour.Behaviour):
    """Whether either hand carries a bottle of pair k that belongs on table t."""

    def __init__(self, k, t):
        super().__init__(f"a bottle for table {t + 1}?")
        self.k, self.t = k, t

    def update(self):
        yes = any(plan["has"].get(c) and DEST[kind(which(self.k, c))] == self.t for c in (LEFT, RIGHT))
        return Status.SUCCESS if yes else Status.FAILURE


class Check(py_trees.behaviour.Behaviour):
    """Every bottle upright on its table (the feedback says which are not)."""

    def __init__(self):
        super().__init__("all on their tables, upright?")

    def update(self):
        bad = []
        for i in range(2 * len(PICKS)):
            t = DEST[kind(i)]
            b, ((x0, x1), (y0, y1)) = bottle(i), BOUNDS[t]
            on = x0 < b["x"] < x1 and y0 < b["y"] < y1 and b["z"] > TOP[t]
            if not on or b["up"] < 0.9:
                bad.append(f'{i + 1} ({"lying" if on else "not there"})')
        self.feedback_message = "all there" if not bad else "bottle " + ", ".join(bad)
        return Status.FAILURE if bad else Status.SUCCESS


# -- where the hands go --------------------------------------------------------

def outside(c, x, y, z, by):
    """(x, y, z) moved `by` outwards: to the robot's left for the left hand, its right for the right one."""
    yaw = robot_pose()["yaw"]
    return [x - math.sin(yaw) * SIGN[c] * by, y + math.cos(yaw) * SIGN[c] * by, z]


def carry(c, z=CARRY[2]):
    def where():
        x, y = ahead(robot_pose(), CARRY[0], SIGN[c] * CARRY[1])
        return [x, y, z]
    return where


def at_bottle(i, c, dz=0.0, by=0.0):
    """Where hand c's tip goes for bottle i: by its middle, `by` outside it, `dz` up."""
    return lambda: outside(c, bottle(i)["x"], bottle(i)["y"], middle_z(i) + GRASP_DZ[kind(i)] + dz, by)


def at_spot(i, c, t, x, y, dz=0.0, by=0.0):
    """Where hand c's tip goes to put bottle i, which it holds, down on (x, y) on table t."""
    def where():
        hx, hy, hz = plan["held"][c]
        return outside(c, x - hx, y - hy, middle_z(i, t) - hz + dz, by)
    return where


def back():
    """Walk's goal: BACK behind where the robot is, facing the same way."""
    p = robot_pose()
    x, y = ahead(p, -BACK, 0.0)
    return {"x": x, "y": y, "yaw": p["yaw"]}


def stand(middle, yaw, out=0.0):
    """Walk's goal: where the robot stands, facing yaw, with the pair at `middle`
    before it (`out` further back)."""
    x, y = ahead({"x": middle[0], "y": middle[1], "yaw": yaw}, -REACH[0] - out, 0.0)
    return lambda: {"x": x, "y": y, "yaw": yaw}


def take(k, c):
    """Hand c takes its bottle of pair k and carries it, or comes up empty."""
    i, who = which(k, c), NAME[c]
    grasp = py_trees.composites.Sequence(f"grasp bottle {i + 1}", memory=True, children=[
        Upright(i),
        Grip("open", c, 0.0, 0.8),
        Reach("beside it", c, at_bottle(i, c, 0.03, SIDE)),
        Reach("down", c, at_bottle(i, c, 0.0, SIDE)),
        Reach("around it", c, at_bottle(i, c, 0.0, PALM)),
        Grip("close", c, 1.0),
        Reach("lift", c, at_bottle(i, c, LIFT, PALM)),
        Holding(i, c),
    ])
    start_over = py_trees.composites.Sequence("start over", memory=True, children=[
        Grip("open", c, 0.0, 0.8),
        Reach("out to the side", c, lambda: outside(c, *tip(c)[0]["pos"], SIDE)),
        py_trees.behaviours.Failure("again"),
    ])
    return py_trees.composites.Selector(f"the {who} hand takes bottle {i + 1}, else none", memory=True, children=[
        py_trees.composites.Sequence(f"the {who} hand takes bottle {i + 1}", memory=True, children=[
            py_trees.decorators.Retry(f"up to {RETRIES + 1} tries", py_trees.composites.Selector(
                "grasp, else start over", memory=True, children=[grasp, start_over]), num_failures=RETRIES + 1),
            Reach("carry it", c, carry(c)),
        ]),
        py_trees.composites.Sequence(f"the {who} hand up, empty", memory=True, children=[
            Empty(c), Grip("open", c, 0.0, 0.8), Reach("up", c, carry(c, READY_Z)),
        ]),
    ])


def give(k, c, t):
    """Hand c puts its bottle of pair k down on its spot on table t, if it has it and it goes there."""
    i, who = which(k, c), NAME[c]
    x, y = spot(PLACES[t][k], PLACE_YAW[t], c)
    return py_trees.composites.Selector(f"the {who} hand puts bottle {i + 1} down, if it goes here", memory=True, children=[
        py_trees.composites.Sequence(f"the {who} hand puts bottle {i + 1} down", memory=True, children=[
            Has(i, c, t),
            Reach("over the spot", c, at_spot(i, c, t, x, y, PLACE_DZ + 0.02)),
            Reach("down", c, at_spot(i, c, t, x, y, PLACE_DZ)),
            Grip("open", c, 0.0),
            Reach("out to the side", c, at_spot(i, c, t, x, y, PLACE_DZ, SIDE)),
            Reach("up", c, carry(c, READY_Z)),
        ]),
        py_trees.behaviours.Success("nothing for here"),
    ])


def both(name, f):
    return py_trees.composites.Parallel(name, py_trees.common.ParallelPolicy.SuccessOnAll(),
                                        children=[f(LEFT), f(RIGHT)])


def pair(k):
    return py_trees.composites.Sequence(f"pair {k + 1}", memory=True, children=[
        Walk(f"to pair {k + 1}", stand(PICKS[k], PICK_YAW)),
        both("both hands take one", lambda c: take(k, c)),
        *[py_trees.composites.Selector(f"to table {t + 1}, if a bottle goes there", memory=True, children=[
            py_trees.composites.Sequence(f"to table {t + 1}", memory=True, children=[
                For(k, t),
                Walk(f"to {APPROACH * 100:.0f} cm before table {t + 1}", stand(PLACES[t][k], PLACE_YAW[t], APPROACH)),
                Walk(f"up to table {t + 1}", stand(PLACES[t][k], PLACE_YAW[t])),
                both("put down what goes here", lambda c, t=t: give(k, c, t)),
                Walk(f"back {BACK * 100:.0f} cm", back),
            ]),
            py_trees.behaviours.Success("none for it"),
        ]) for t in sorted(PLACES)],
    ])


root = py_trees.composites.Sequence("G1 sorts four bottles onto two tables", memory=True, children=[
    Setup(),
    # The hands open on the way up: open already, they need not at each bottle.
    both("both hands up and open", lambda c: py_trees.composites.Parallel(
        f"the {NAME[c]} hand up and open", py_trees.common.ParallelPolicy.SuccessOnAll(), children=[
            Reach("up", c, carry(c, READY_Z)), Grip("open", c, 0.0, 0.8)])),
    *[pair(k) for k in range(len(PICKS))],
    Walk("step back", lambda: dict(zip("xy", START[:2]), yaw=START[2])),
    Check(),
])
