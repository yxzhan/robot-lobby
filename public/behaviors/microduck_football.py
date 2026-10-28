# Duck football 3v3 (on the microduck_football_1v1 stage, or anywhere: the
# tree spawns the pitch if there is none and the ducks each team lacks, all
# on Pollen's policy set, microduck-official): the graphite Microducks attack the +x goal, the white
# ones the -x goal (in the pitch's own frame, however it is placed). In each
# team the duck nearest the ball attacks: it lines up behind the ball for a
# kick with the nearer foot (the ball in front of that toe, where the kick was
# trained), coming at it from behind, stops, and kicks it on (from a stand:
# a kick started mid-stride does not lift the foot), or walks through it when it cannot get close
# enough; of the other two the one nearer its own goal defends (between that
# goal and the ball) and the other supports (behind the ball, to one side).
# A duck that falls gets back up by itself (velstand). A referee scores goals
# and restarts (Reset: the kick-off again) when the ball goes out. Each
# kick-off (the first one too: the tree starts with a Reset) puts the ball at
# a random spot of the pitch, clear of the ducks, which start on their spots
# (Reset itself puts the ball back on the centre spot).
import math
import random
import time
import py_trees
from py_trees.common import Status
import lobby

TEAMS = [("microduck_graphite", +1, "graphite"), ("microduck", -1, "white")]  # label (= model), attacks +x / -x, name
PER_TEAM = 3
PITCH = "microduck_football"       # the model spawned when there is no pitch,
PITCH_AT = (0.0, 2.0, 0.0)         # with its centre spot here (x, y, z; on the floor)
# Where ducks are spawned (pitch frame, for the team attacking +x; the other's are mirrored through the
# centre spot): the first one clear of every duck. The first is the stage's spot.
SPOTS = [(-0.6, 0.2), (-1.2, 0.6), (-1.2, -0.6), (-0.3, 0.9), (-1.5, 0.0)]
SPOT_CLEAR = 0.35                  # m: a spot with a duck closer than this is taken
SPAWN_WAIT = 60.0                  # s (wall time): a spawn not seen by then is asked for again
SPAWN_SETTLE = 1.0                 # s (wall time) after a spawn shows before the next: in a shared world the
                                   # poses come a moment after the new scene (all at the origin until then)
SWITCH_MARGIN = 0.2                # m: a teammate takes over the attack once this much nearer the ball
DEFEND_OUT = 0.6                   # m out from its goal, on the line to the ball (at most half way)
SUPPORT_BACK, SUPPORT_SIDE = 0.7, 0.5  # m behind the ball (toward its own goal) and to its side
KEEP_OFF = 0.45                    # m: who does not attack stays this far from the ball
GOAL_X, GOAL_HALF_W = 2.02, 0.40   # goal mouths at x = +-2.02 (pitch frame)
TOUCH_Y, END_X = 1.3, 2.05        # ball wholly over the touchline / the goal line (pitch frame, m: the lines'
                                   # outer edges at 1.3125 / 2.0125, the ball's radius 0.04)
BEHIND, PUSH = 0.22, 0.45          # going round the ball: this far behind it; walking through it: this far on
# The kicks (skill: ball centre in the duck's frame, m; how far left of the
# duck's heading the ball then runs, rad): trained with the ball 9 cm ahead of
# that toe; measured in the viewer, the right kick sends it ~7 deg right, the
# left one ~9 deg left (1.6-3 m).
KICKS = {
    "kick_right": ((0.095, -0.042), math.radians(-7)),
    "kick_left": ((0.095, 0.042), math.radians(9)),
}
KICK_TOL = (0.03, 0.015, 0.015)    # how far from the ball spot it still kicks (m: the ball nearer / farther
                                   # than its spot, sideways); 2.5 cm farther, a kick after walking misses
KICK_YAW_TOL = 0.2                 # rad off the kick heading
KICK_REST = 1.5                    # s (simulated) after a kick (0.5 s) before the next
KICK_SET = 0.3                     # s (simulated) stopped on the spot before kicking: a kick started while
                                   # walking does not lift the foot (0.1 s does not do, 0.3 s does)
KICK_LEAD = 0.06                   # m: the goal set past the kick spot (a goal counts as reached 6 cm short)
LINE_UP_NEAR = 0.35                # m from the kick spot: closer, it lines up on the spot itself (back out past
LINE_UP_FAR = 0.50                 # this); farther, it first walks to a point APPROACH_BACK behind the spot
APPROACH_BACK = 0.25               # on the kick line, so it comes at the ball from behind
LINE_UP_YAW = 0.5                  # rad: it lines up only facing the kick heading this nearly, from behind the
                                   # spot (within 45 deg of the kick line), else it goes to the approach point
SETTLE = 2.0                       # s lining up close to the ball (within STUCK_NEAR) without a kick: walk
STUCK_NEAR = 0.20                  # through the ball instead (m; at the kick spot the ball is ~0.10 m off)
OUT_TIME = 0.3                     # s the ball must stay out (or in the goal)
COOLDOWN = 5.0                     # s after a restart before the referee looks again
BALL_SPREAD = (1.2, 0.9)           # random kick-off: the ball within this of the centre spot (m, along / across the pitch)
DUCK_CLEAR = 0.35                  # and at least this far from every duck (m)
score = {team: 0 for _, _, team in TEAMS}
kickoff = {"placing": False}   # the referee placing the ball after a Reset: the players wait
attackers = {}                 # label -> name of the team's duck that attacks


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def pitch():
    return next((r for r in lobby.robots() if "football" in r["label"]), None)


def team(label):
    """The team's ducks (lobby.robots() entries), in spawn order."""
    num = lambda r: int(r["name"][1:]) if r["name"][1:].isdigit() else 0
    return sorted((r for r in lobby.robots() if r["label"] == label), key=num)


def dist(r, x, y):
    return math.hypot(r["pose"]["x"] - x, r["pose"]["y"] - y)


def roles(label, sign, fr, bx, by):
    """{name: 'attack' | 'defend' | 'support'} of the team, the ball at bx, by."""
    ducks = team(label)
    if not ducks:
        return {}
    near = {r["name"]: dist(r, bx, by) for r in ducks}
    cur, best = attackers.get(label), min(near, key=near.get)
    if cur not in near or near[best] < near[cur] - SWITCH_MARGIN:
        cur = attackers[label] = best
    own = fr.world(-sign * GOAL_X, 0)
    rest = sorted((r for r in ducks if r["name"] != cur), key=lambda r: dist(r, *own))
    return {cur: "attack", **{r["name"]: "defend" if i == 0 else "support" for i, r in enumerate(rest)}}


class Frame:
    """The pitch's frame on the floor (its pose: x, y, yaw)."""

    def __init__(self, pose):
        self.x, self.y = pose["x"], pose["y"]
        self.c, self.s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        self.yaw = pose["yaw"]

    def local(self, x, y):
        dx, dy = x - self.x, y - self.y
        return self.c * dx + self.s * dy, -self.s * dx + self.c * dy

    def world(self, u, v):
        return self.x + self.c * u - self.s * v, self.y + self.s * u + self.c * v


class Player(py_trees.behaviour.Behaviour):
    def __init__(self, label, attack_sign, team_name, index):
        super().__init__(f"{team_name} {index + 1} plays")
        self.label, self.sign, self.index = label, attack_sign, index
        self.last = None        # the goal last given: (x, y, mode)
        self.kicked = None      # simulated time of the last kick
        self.at_spot = None     # since when it has been lining up close to the ball
        self.foot = None        # the kick it lines up for (KICKS), kept until it kicks or goes round
        self.set_at = None      # simulated time it stopped on the spot to kick (self.foot)

    def update(self):
        f, ducks = pitch(), team(self.label)
        if not f or len(ducks) <= self.index:
            self.feedback_message = "no pitch or no " + self.label
            return Status.FAILURE
        me = ducks[self.index]
        name, t = me["name"], lobby.time()
        if kickoff["placing"]:
            self.last = self.foot = self.set_at = None
            self.feedback_message = "kick-off"
            return Status.RUNNING
        for k in ("kicked", "at_spot", "set_at"):  # time went back (Reset): forget
            if getattr(self, k) is not None and t < getattr(self, k):
                setattr(self, k, None)
        if self.kicked is not None and t - self.kicked < KICK_REST:
            self.feedback_message = "kicked"
            return Status.RUNNING
        fr, b = Frame(f["pose"]), lobby.body(f["name"] + "/ball")
        bx, by = b["x"], b["y"]
        mx, my, myaw = me["pose"]["x"], me["pose"]["y"], me["pose"]["yaw"]
        gx, gy = fr.world(self.sign * GOAL_X, 0)        # the goal to attack
        ux, uy = gx - bx, gy - by
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        attack_yaw = math.atan2(uy, ux)
        role = roles(self.label, self.sign, fr, bx, by).get(name, "support")
        if role != "attack":
            self.set_at = None
            return self.position(name, role, fr, me, bx, by, ux, uy, attack_yaw)
        # Each kick's spot: facing so that the ball (with that kick's veer) runs
        # at the goal, the ball at its spot in front of the toe.
        spots = {}
        for foot, ((ax, ay), veer) in KICKS.items():
            face = attack_yaw - veer
            cf, sf = math.cos(face), math.sin(face)
            spots[foot] = (bx - (cf * ax - sf * ay), by - (sf * ax + cf * ay), face, ax, ay)
        # The ball in the duck's own frame: kick when it is where a kick wants it.
        c, s = math.cos(myaw), math.sin(myaw)
        fwd, side = c * (bx - mx) + s * (by - my), -s * (bx - mx) + c * (by - my)
        on_spot = lambda foot: (-KICK_TOL[0] < fwd - spots[foot][3] < KICK_TOL[1] and abs(side - spots[foot][4]) < KICK_TOL[2]
                                and abs(wrap(myaw - spots[foot][2])) < KICK_YAW_TOL)
        if self.set_at is not None:  # stopped on the spot: kick once it stands, if still there
            if t - self.set_at < KICK_SET:
                self.feedback_message = "set " + self.foot
                return Status.RUNNING
            foot, self.set_at = self.foot, None
            if on_spot(foot):
                lobby.use_skill(name, foot)
                self.kicked, self.at_spot, self.last, self.foot = t, None, None, None
                self.feedback_message = "KICK " + foot
                return Status.RUNNING
        else:
            foot = next((f for f in spots if on_spot(f)), None)
            if foot is not None:
                lobby.stop(name)
                self.set_at, self.foot, self.last = t, foot, None
                self.feedback_message = "set " + foot
                return Status.RUNNING
        if self.foot is None:  # line up for the foot whose spot is nearer
            self.foot = min(spots, key=lambda f: math.hypot(mx - spots[f][0], my - spots[f][1]))
        sx, sy, face, _, _ = spots[self.foot]
        cf, sf = math.cos(face), math.sin(face)
        # Where the duck stands relative to the ball, along / across the attack line.
        dx, dy = mx - bx, my - by
        along, across = dx * ux + dy * uy, -dx * uy + dy * ux
        spot = math.hypot(mx - sx, my - sy)
        pushing = self.last is not None and self.last[2] == "push"
        if pushing and along < 0.12 and abs(across) < 0.16:   # keep pushing while still on the ball
            tx, ty, mode, yaw = bx + ux * PUSH, by + uy * PUSH, "push", attack_yaw
        elif along > 0.0:                       # in front of the ball: go round it
            self.foot = None
            side_sign = 1.0 if across >= 0 else -1.0
            tx = bx - ux * BEHIND - uy * side_sign * 0.30
            ty = by - uy * BEHIND + ux * side_sign * 0.30
            mode, yaw = "around", attack_yaw
        elif self.at_spot is not None and t - self.at_spot > SETTLE and math.hypot(dx, dy) < STUCK_NEAR:
            tx, ty, mode, yaw = bx + ux * PUSH, by + uy * PUSH, "push", attack_yaw  # not close enough to kick: walk through
        else:                                   # behind the ball: to the kick line, then the kick spot
            lined = self.last is not None and self.last[2].startswith("line up")
            back = -((mx - sx) * cf + (my - sy) * sf)        # how far behind the spot, along the kick heading
            lat = abs(-(mx - sx) * sf + (my - sy) * cf)      # and off the kick line
            behind = lat < max(0.08, back) and abs(wrap(myaw - face)) < (1.6 if lined else 1.0) * LINE_UP_YAW
            if spot > (LINE_UP_FAR if lined else LINE_UP_NEAR) or not behind:
                tx, ty, mode, yaw = sx - cf * APPROACH_BACK, sy - sf * APPROACH_BACK, "approach", face
            else:
                tx, ty, mode, yaw = sx + cf * KICK_LEAD, sy + sf * KICK_LEAD, "line up " + self.foot.split("_")[1], face
        if mode.startswith("line up") and math.hypot(dx, dy) < STUCK_NEAR:  # (by the ball: another duck may keep it off the spot)
            self.at_spot = t if self.at_spot is None else self.at_spot
        elif mode != "push":
            self.at_spot = None
        return self.go(name, fr, tx, ty, yaw, mode, math.hypot(dx, dy))

    def go(self, name, fr, tx, ty, yaw, mode, ball, moved=0.03):
        """The goal marker to tx, ty facing yaw (kept on the pitch), unless it is there already."""
        u, v = fr.local(tx, ty)
        tx, ty = fr.world(max(-2.2, min(2.2, u)), max(-1.5, min(1.5, v)))
        if self.last is None or math.hypot(tx - self.last[0], ty - self.last[1]) > moved or mode != self.last[2]:
            lobby.set_goal(name, tx, ty, yaw)
            self.last = (tx, ty, mode)
        self.feedback_message = f"{mode}  ball {ball:.2f} m"
        return Status.RUNNING

    def position(self, name, role, fr, me, bx, by, ux, uy, attack_yaw):
        """Defend (between its goal and the ball) or support (behind the ball, to one side)."""
        self.foot = self.at_spot = None
        mx, my = me["pose"]["x"], me["pose"]["y"]
        if role == "defend":
            gx, gy = fr.world(-self.sign * GOAL_X, 0)
            vx, vy = bx - gx, by - gy
            n = math.hypot(vx, vy) or 1.0
            out = min(DEFEND_OUT, 0.5 * n)
            tx, ty = gx + vx / n * out, gy + vy / n * out
        else:
            side = 1.0 if -(mx - bx) * uy + (my - by) * ux >= 0 else -1.0  # (it keeps to its side of the attack line)
            tx = bx - ux * SUPPORT_BACK - uy * side * SUPPORT_SIDE
            ty = by - uy * SUPPORT_BACK + ux * side * SUPPORT_SIDE
        d = math.hypot(tx - bx, ty - by)
        if d < KEEP_OFF:  # off the ball, out along the line from it
            k = KEEP_OFF / (d or 1e-6)
            tx, ty = bx + (tx - bx) * k, by + (ty - by) * k
        return self.go(name, fr, tx, ty, math.atan2(by - ty, bx - tx), role, math.hypot(mx - bx, my - by), moved=0.08)


class Teams(py_trees.behaviour.Behaviour):
    """Spawns the pitch if there is none, then the ducks the teams lack, one at a time (each spawn rebuilds
    the world), each on the first free spot of SPOTS; done when every team has PER_TEAM."""

    def __init__(self):
        super().__init__(f"Teams of {PER_TEAM}")
        self.pending = None  # (what: a label or PITCH, how many there were, wall time): a spawn on its way
        self.asked = []      # the spots spawned on: taken (a new duck's pose shows a moment after it does)
        self.settle = 0.0    # wall time until which the last spawn settles (its poses arriving)
        self.frame = None    # the pitch's frame, noted once settled (it does not move)

    def initialise(self):
        self.settle = time.monotonic() + SPAWN_SETTLE  # (the world may have just changed: its poses first)

    def count(self, what):
        return (1 if pitch() else 0) if what == PITCH else len(team(what))

    def update(self):
        if self.pending is not None:  # one spawn at a time: until it shows (wall time: a rebuild may set the simulated one back)
            what, had, t0 = self.pending
            if self.count(what) <= had and time.monotonic() - t0 < SPAWN_WAIT:
                self.feedback_message = f"spawning {what}"
                return Status.RUNNING
            self.pending = None
            self.settle = time.monotonic() + SPAWN_SETTLE
        if time.monotonic() < self.settle:
            return Status.RUNNING
        f = pitch()
        if f is None:
            lobby.spawn(PITCH, PITCH_AT[0], PITCH_AT[1], 0.0)
            self.pending = (PITCH, 0, time.monotonic())
            self.feedback_message = f"spawning {PITCH}"
            return Status.RUNNING
        counts = {label: len(team(label)) for label, _, _ in TEAMS}
        self.feedback_message = ", ".join(f"{name} {counts[label]}" for label, _, name in TEAMS)
        missing = next(((label, sign) for label, sign, _ in TEAMS if counts[label] < PER_TEAM), None)
        if missing is None:
            return Status.SUCCESS
        label, sign = missing
        fr = self.frame = self.frame or Frame(f["pose"])
        ducks = [r for l, _, _ in TEAMS for r in team(l)]
        spots = [fr.world(sign * u, sign * v) for u, v in SPOTS]
        clear = lambda p: min([dist(r, *p) for r in ducks] + [math.hypot(p[0] - ax, p[1] - ay) for ax, ay in self.asked],
                              default=math.inf)
        x, y = next((p for p in spots if clear(p) >= SPOT_CLEAR), max(spots, key=clear))  # (else the least crowded)
        lobby.spawn(label, x, y, fr.yaw + (0.0 if sign > 0 else math.pi))
        self.asked.append((x, y))
        self.pending = (label, counts[label], time.monotonic())
        self.feedback_message += f"; spawning {label}"
        return Status.RUNNING


class Referee(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("Referee")
        self.last = "kick-off"
        self.quiet_until = None       # no calls before this (simulated) time: a restart landing
        self.out_since = None
        self.prev_t = None
        self.landing = None           # after a Reset: (the time before it, ticks since), until the ducks are placed

    def restart(self, why, t):
        self.last = why
        self.quiet_until = t + COOLDOWN
        self.out_since = None
        lobby.reset_world()                     # the kick-off again
        self.landing = (t, 0)
        kickoff["placing"] = True

    def kick_off(self, fr, ball):
        """The ball at a random spot of the pitch, clear of every duck (on its spot after the Reset)."""
        ducks = [fr.local(p["pose"]["x"], p["pose"]["y"]) for label, _, _ in TEAMS for p in team(label)]
        for _ in range(50):  # (the last try stands if none is clear)
            u = random.uniform(-BALL_SPREAD[0], BALL_SPREAD[0])
            v = random.uniform(-BALL_SPREAD[1], BALL_SPREAD[1])
            if all(math.hypot(u - du, v - dv) >= DUCK_CLEAR for du, dv in ducks):
                break
        lobby.place_body(ball["name"], *fr.world(u, v))

    def update(self):
        f = pitch()
        if not f:
            return Status.FAILURE
        t = lobby.time()
        if self.prev_t is None:  # the first tick: the kick-off
            self.restart("kick-off", t)
            self.prev_t = t
            return Status.RUNNING
        if t < self.prev_t:  # time went back (the Reset): the cool-down from here
            self.quiet_until = t + COOLDOWN
            self.out_since = None
        self.prev_t = t
        fr, b = Frame(f["pose"]), lobby.body(f["name"] + "/ball")
        if self.landing is not None:  # the Reset landed (time went back, or a few ticks on): place the ducks
            before, ticks = self.landing
            if t < before or ticks >= 5:
                self.landing = None
                self.kick_off(fr, b)
                kickoff["placing"] = False
            else:
                self.landing = (before, ticks + 1)
        rx, ry = fr.local(b["x"], b["y"])
        is_goal = abs(ry) < GOAL_HALF_W and abs(rx) > END_X
        is_out = not is_goal and (abs(ry) > TOUCH_Y or abs(rx) > END_X)
        self.out_since = (self.out_since if self.out_since is not None else t) if (is_goal or is_out) else None
        if self.quiet_until is None or t >= self.quiet_until:
            if self.out_since is not None and t - self.out_since >= OUT_TIME:
                if is_goal:
                    team = next(team for _, sign, team in TEAMS if sign * rx > 0)
                    score[team] += 1
                    self.restart(f"GOAL {team}", t)
                else:
                    self.restart("ball out", t)
        a, b2 = TEAMS[0][2], TEAMS[1][2]
        self.feedback_message = f"SCORE  {a} {score[a]} : {score[b2]} {b2}   (last: {self.last})"
        return Status.RUNNING


root = py_trees.composites.Sequence("Duck football", memory=True, children=[
    Teams(),
    py_trees.composites.Parallel(
        "Play",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
        children=[Referee()] + [Player(label, sign, name, i) for label, sign, name in TEAMS for i in range(PER_TEAM)],
    ),
]
)
