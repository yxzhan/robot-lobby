# Tries every skill of Pollen's Microduck policy set (the Teleop panel's
# "Walk + get up, tricks · Microduck (Pollen)") on one Microduck, one after
# the other, and checks what each did: walking and turning, both kicks (a
# ball put at the toe: spawn a microduck_ball too), sitting and standing up,
# touching the ground, the forward roll, getting up from the face and the
# back, the stand-only and the older walking policies. Each check's node
# says what it measured; the last node sums them up.
import math
import py_trees
from py_trees.common import Status
import lobby
from lobby import Move, Wait

POLICY = "microduck-official"
results = []   # (what, passed, detail)


def duck():
    return next((r for r in lobby.robots() if r["label"].startswith("microduck")
                 and "ball" not in r["label"] and "football" not in r["label"]), None)


def ball():
    return next((r for r in lobby.robots() if "ball" in r["label"] and "football" not in r["label"]), None)


def standing(name):
    up = lobby.upright(name)
    trunk = lobby.body(name + "/trunk_base")
    return up is not None and up > 0.9 and trunk["z"] > 0.09


class Do(py_trees.behaviour.Behaviour):
    """Calls fn(duck name) once and succeeds (fails if it raises)."""

    def __init__(self, name, fn):
        super().__init__(name)
        self.fn = fn

    def update(self):
        d = duck()
        if d is None:
            self.feedback_message = "no Microduck"
            return Status.FAILURE
        try:
            self.feedback_message = str(self.fn(d["name"]) or "")
        except Exception as err:
            self.feedback_message = str(err)
            return Status.FAILURE
        return Status.SUCCESS


class Until(py_trees.behaviour.Behaviour):
    """Runs until cond(duck name) holds (success) or `seconds` of simulated time pass (failure)."""

    def __init__(self, name, cond, seconds):
        super().__init__(name)
        self.cond, self.seconds = cond, seconds

    def initialise(self):
        self.t0 = lobby.time()

    def update(self):
        d = duck()
        if d and self.cond(d["name"]):
            return Status.SUCCESS
        t = lobby.time()
        if t < self.t0:
            self.t0 = t
        return Status.FAILURE if t - self.t0 > self.seconds else Status.RUNNING


class Check(py_trees.behaviour.Behaviour):
    """Notes whether test(duck name) -> (passed, detail) holds; always succeeds (the next test runs)."""

    def __init__(self, name, test):
        super().__init__(name)
        self.test = test

    def update(self):
        d = duck()
        ok, detail = self.test(d["name"]) if d else (False, "no Microduck")
        results.append((self.name, ok, detail))
        self.feedback_message = ("PASS  " if ok else "FAIL  ") + detail
        return Status.SUCCESS


class Watch(py_trees.behaviour.Behaviour):
    """Runs alongside a test: keeps the lowest of measure(duck name) in watched[key]; never ends."""

    def __init__(self, name, key, measure):
        super().__init__(name)
        self.key, self.measure = key, measure

    def initialise(self):
        watched[self.key] = math.inf

    def update(self):
        d = duck()
        if d:
            watched[self.key] = min(watched[self.key], self.measure(d["name"]))
            self.feedback_message = f"lowest {watched[self.key]:.3f}"
        return Status.RUNNING


watched = {}
marks = {}  # where things were when a test started


def mark(key):
    def fn(name):
        marks[key] = (dict(lobby.pose(name)), dict(lobby.pose(ball()["name"])) if ball() else None)
        return ""
    return fn


def skill(name_, duration=None):
    return lambda name: lobby.use_skill(name, name_, duration)


def put_ball(side):
    """The ball just in front of the right (side -1) or left (+1) toe: (0.095, +-0.042) m in the duck's frame."""
    def fn(name):
        b = ball()
        if b is None:
            raise RuntimeError("no ball: spawn a microduck_ball")
        p = lobby.pose(name)
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        fx, fy = 0.095, side * 0.042
        lobby.place(b["name"], p["x"] + c * fx - s * fy, p["y"] + s * fx + c * fy, 0.0)
        mark("kick")(name)
        return "ball at the toe"
    return fn


def lie(pitch):
    def fn(name):
        p = lobby.pose(name)
        lobby.place(name, p["x"], p["y"], p["yaw"], pitch=pitch, z=0.08)
        return "on its face" if pitch > 0 else "on its back"
    return fn


def stood(what):
    return Check(what, lambda name: (standing(name), f"up {lobby.upright(name):.2f}, trunk {lobby.body(name + '/trunk_base')['z']:.3f} m"))


def moved(what, want):
    def test(name):
        p0, p = marks["walk"][0], lobby.pose(name)
        d = math.hypot(p["x"] - p0["x"], p["y"] - p0["y"])
        return d >= want and standing(name), f"{d:.2f} m"
    return Check(what, test)


def turned(what, want_deg):
    def test(name):
        dy = math.degrees((lobby.pose(name)["yaw"] - marks["turn"][0]["yaw"] + math.pi) % (2 * math.pi) - math.pi)
        return abs(dy - want_deg) < 40 and standing(name), f"{dy:+.0f} deg (asked {want_deg:+.0f})"
    return Check(what, test)


def kicked(what):
    def test(name):
        b0, b = marks["kick"][1], lobby.pose(ball()["name"])
        d = math.hypot(b["x"] - b0["x"], b["y"] - b0["y"])
        return d > 0.3 and standing(name), f"ball {d:.2f} m"
    return Check(what, test)


def low_check(what, key, below, unit="m", then_standing=True):
    """watched[key] went below `below` (a number, or a function of the duck's name: measured before),
    and (then_standing) the duck stands at the end."""
    def test(name):
        limit = below(name) if callable(below) else below
        ok = watched[key] < limit and (standing(name) or not then_standing)
        return ok, f"lowest {watched[key]:.3f} {unit} (below {limit:.3f}?)" + (f", then standing {standing(name)}" if then_standing else "")
    return Check(what, test)


def step(name, *children):
    return py_trees.composites.Sequence(name, memory=True, children=list(children))


def during(name, seconds, watch):
    """Waits `seconds` (simulated) while `watch` keeps its lowest value."""
    return py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
                                        children=[Wait(seconds), watch])


class Report(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("Report")

    def update(self):
        failed = [w for w, ok, _ in results if not ok]
        self.feedback_message = (f"{len(results) - len(failed)}/{len(results)} passed"
                                 + (f"; failed: {', '.join(failed)}" if failed else ""))
        return Status.SUCCESS


trunk_z = lambda name: lobby.body(name + "/trunk_base")["z"]
head_z = lambda name: lobby.body(name + "/jaw_soft")["z"]

root = step("Microduck skills", *[
    step("Policy", Do("switch to Pollen's set", lambda name: lobby.policy(name) == POLICY or lobby.set_policy(name, POLICY)),
         Until("loaded", lambda name: lobby.policy(name) == POLICY and lobby.active_skill(name) is not None, 30.0),
         Check("policy on", lambda name: (lobby.policy(name) == POLICY, str(lobby.policy(name))))),
    step("Stand", Wait(2.0), stood("stands")),
    step("Walk", Do("mark", mark("walk")),
         py_trees.decorators.Timeout("20 s at most", duration=20.0, child=Move("all", forward=0.5)),
         moved("walked 0.5 m", 0.35)),
    step("Turn", Do("mark", mark("turn")),
         py_trees.decorators.Timeout("20 s at most", duration=20.0, child=Move("all", turn=math.pi / 2)),
         turned("turned 90 deg", 90)),
    step("Kick right", Wait(1.0), Do("ball to the right toe", put_ball(-1)), Do("kick", skill("kick_right")),
         Wait(3.0), kicked("kicked right")),
    step("Kick left", Wait(1.0), Do("ball to the left toe", put_ball(+1)), Do("kick", skill("kick_left")),
         Wait(3.0), kicked("kicked left")),
    step("Sit", Do("sit", skill("sit")), during("seated 4 s", 4.0, Watch("trunk", "sit", trunk_z)),
         low_check("sat down", "sit", 0.08, "m (trunk)", then_standing=False)),
    step("Stand up", Do("walk", skill("walk")), Wait(3.0), stood("stood up from sitting")),
    step("Touch the ground", Do("jaw height standing", lambda name: marks.__setitem__("jaw", head_z(name))),
         Do("bow", skill("ground_pick")), during("3.5 s", 3.5, Watch("jaw", "pick", head_z)),
         low_check("touched the ground", "pick", lambda name: marks["jaw"] - 0.06, "m (the jaw body; the mouth tip is lower)")),
    step("Forward roll", Wait(1.0), Do("roll", skill("roulade")), during("4 s", 4.0, Watch("upright", "roll", lobby.upright)),
         low_check("rolled over and stood again", "roll", -0.5, "(upright: < 0 is upside down)")),
    step("Get up from the face", Do("lay it on its face", lie(math.pi / 2)), Wait(5.0), stood("up from the face")),
    step("Get up from the back", Do("lay it on its back", lie(-math.pi / 2)), Wait(5.0), stood("up from the back")),
    step("Stand only", Do("stand", skill("stand")), Do("mark", mark("walk")), Wait(1.0),
         Do("lay it on its back", lie(-math.pi / 2)), Wait(5.0), stood("stand policy gets up")),
    step("Older walk", Do("walk_alpha", skill("walk_alpha")), Wait(1.0), Do("mark", mark("walk")),
         py_trees.decorators.Timeout("20 s at most", duration=20.0, child=Move("all", forward=0.4)),
         moved("walk_alpha walked", 0.25), Do("back to walk", skill("walk"))),
    Report(),
])
