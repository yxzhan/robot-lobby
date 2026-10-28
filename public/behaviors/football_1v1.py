# Duck football 1v1 (the microduck_football_1v1 stage): the graphite Microduck
# attacks the +x goal, the white one the -x goal (in the pitch's own frame,
# however it is placed). Each duck lines up behind the ball for its kick (the
# ball in front of its right foot, where the kick was trained) and kicks it
# on; lined up but not close enough to kick, it walks through the ball
# instead. A referee scores goals and restarts (Reset: the kick-off again)
# when the ball goes out, both ducks are down, or the ball has not moved for
# a while (the two pushing it from both sides). Each kick-off (the first one
# too: the tree starts with a Reset) puts the ducks at random spots near the
# stage's, in their own halves (Reset itself still goes back to the stage's).
import math
import random
import py_trees
from py_trees.common import Status
import lobby

TEAMS = [("microduck_graphite", +1, "graphite"), ("microduck", -1, "white")]  # label, attacks +x / -x, name
GOAL_X, GOAL_HALF_W = 2.02, 0.40   # goal mouths at x = +-2.02 (pitch frame)
TOUCH_Y = 1.40                     # ball wholly over the touchline (pitch frame, m)
BEHIND, PUSH = 0.22, 0.45          # going round the ball: this far behind it; walking through it: this far on
KICK_AT = (0.095, -0.042)          # ball centre in the duck's frame for kick_right (trained: (0.09, -0.042) +-0.015, a 35 mm ball)
KICK_TOL = (0.03, 0.025)           # how far from KICK_AT it still kicks (m, forward / sideways)
KICK_YAW_TOL = 0.2                 # rad off the kick heading
KICK_VEER = math.radians(18)       # the kicked ball runs this much left of the duck's heading
KICK_REST = 1.5                    # s (simulated) after a kick before the next
KICK_LEAD = 0.035                  # m: the goal set past the kick spot (a goal counts as reached 6 cm short)
SETTLE = 2.0                       # s at the kick spot without a kick: walk through the ball instead
FALL_UP = 0.5                      # trunk up-axis z below this = fallen
FALL_TIME = 1.5                    # s on the ground before it counts
OUT_TIME = 0.3                     # s the ball must stay out (or in the goal)
STALL_MOVE, STALL_TIME = 0.05, 8.0  # the ball moving less than this (m) for this long (s): a stalemate
COOLDOWN = 5.0                     # s after a restart before the referee looks again
SPREAD_X, SPREAD_Y = 0.35, 0.50    # random kick-off: this far (m) from the stage's spot, along / across the pitch
SPREAD_YAW = math.radians(40)      # and turned this much at most
OWN_HALF = 0.25                    # m: at least this far into its own half
BALL_CLEAR, DUCK_CLEAR = 0.35, 0.40  # m from the ball, and between the two, at least
score = {team: 0 for _, _, team in TEAMS}
kickoff = {"placing": False}   # the referee placing the ducks after a Reset: the players wait


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def pitch():
    return next((r for r in lobby.robots() if "football" in r["label"]), None)


def player(label):
    return next((r for r in lobby.robots() if r["label"] == label), None)


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
    def __init__(self, label, attack_sign, team):
        super().__init__(f"{team} plays")
        self.label, self.sign = label, attack_sign
        self.last = None        # the goal last given: (x, y, mode)
        self.kicked = None      # simulated time of the last kick
        self.at_spot = None     # since when it has stood at the kick spot

    def update(self):
        f, me = pitch(), player(self.label)
        if not f or not me:
            self.feedback_message = "no pitch or no " + self.label
            return Status.FAILURE
        name, t = me["name"], lobby.time()
        if kickoff["placing"]:
            self.last = None
            self.feedback_message = "kick-off"
            return Status.RUNNING
        for k in ("kicked", "at_spot"):  # time went back (Reset): forget
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
        # The kick spot: facing so that the ball (veering left) runs at the goal,
        # the ball at KICK_AT in front of the right foot.
        face = attack_yaw - KICK_VEER
        cf, sf = math.cos(face), math.sin(face)
        sx, sy = bx - (cf * KICK_AT[0] - sf * KICK_AT[1]), by - (sf * KICK_AT[0] + cf * KICK_AT[1])
        # The ball in the duck's own frame: kick when it is where the kick wants it.
        c, s = math.cos(myaw), math.sin(myaw)
        fwd, side = c * (bx - mx) + s * (by - my), -s * (bx - mx) + c * (by - my)
        if abs(fwd - KICK_AT[0]) < KICK_TOL[0] and abs(side - KICK_AT[1]) < KICK_TOL[1] and abs(wrap(myaw - face)) < KICK_YAW_TOL:
            lobby.stop(name)
            lobby.use_skill(name, "kick_right")
            self.kicked, self.at_spot, self.last = t, None, None
            self.feedback_message = "KICK"
            return Status.RUNNING
        # Where the duck stands relative to the ball, along / across the attack line.
        dx, dy = mx - bx, my - by
        along, across = dx * ux + dy * uy, -dx * uy + dy * ux
        spot = math.hypot(mx - sx, my - sy)
        pushing = self.last is not None and self.last[2] == "push"
        if pushing and along < 0.12 and abs(across) < 0.16:   # keep pushing while still on the ball
            tx, ty, mode, yaw = bx + ux * PUSH, by + uy * PUSH, "push", attack_yaw
        elif along > 0.0:                       # in front of the ball: go round it
            side_sign = 1.0 if across >= 0 else -1.0
            tx = bx - ux * BEHIND - uy * side_sign * 0.30
            ty = by - uy * BEHIND + ux * side_sign * 0.30
            mode, yaw = "around", attack_yaw
        elif spot < 0.08 and self.at_spot is not None and t - self.at_spot > SETTLE:
            tx, ty, mode, yaw = bx + ux * PUSH, by + uy * PUSH, "push", attack_yaw  # not close enough to kick: walk through
        else:                                   # behind the ball: to the kick spot
            tx, ty, mode, yaw = sx + cf * KICK_LEAD, sy + sf * KICK_LEAD, "line up", face
        if mode == "line up" and spot < 0.08:
            self.at_spot = t if self.at_spot is None else self.at_spot
        elif mode != "push":
            self.at_spot = None
        # stay on the pitch
        u, v = fr.local(tx, ty)
        tx, ty = fr.world(max(-2.2, min(2.2, u)), max(-1.5, min(1.5, v)))
        if self.last is None or math.hypot(tx - self.last[0], ty - self.last[1]) > 0.03 or mode != self.last[2]:
            lobby.set_goal(name, tx, ty, yaw)
            self.last = (tx, ty, mode)
        self.feedback_message = f"{mode}  ball {math.hypot(dx, dy):.2f} m"
        return Status.RUNNING


class Referee(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("Referee")
        self.last = "kick-off"
        self.quiet_until = None       # no calls before this (simulated) time: a restart landing
        self.out_since = None
        self.down_since = {label: None for label, _, _ in TEAMS}
        self.still = None             # (x, y, since): where the ball has stayed
        self.prev_t = None
        self.home = None              # {label: (u, v, yaw)}: the stage's kick-off spots (pitch frame)
        self.landing = None           # after a Reset: (the time before it, ticks since), until the ducks are placed

    def restart(self, why, t):
        self.last = why
        self.quiet_until = t + COOLDOWN
        self.out_since, self.still = None, None
        self.down_since = {label: None for label in self.down_since}
        lobby.reset_world()                     # the kick-off again
        self.landing = (t, 0)
        kickoff["placing"] = True

    def kick_off(self, fr, ball):
        """The ducks at random spots near their stage spots (the first time: noted)."""
        if self.home is None:
            self.home = {}
            for label, _, _ in TEAMS:
                p = player(label)["pose"]
                self.home[label] = (*fr.local(p["x"], p["y"]), wrap(p["yaw"] - fr.yaw))
        bu, bv = fr.local(ball["x"], ball["y"])
        placed = []
        for label, sign, _ in TEAMS:
            hu, hv, hyaw = self.home[label]
            for _ in range(50):  # (the last try stands if none is clear)
                u = hu + random.uniform(-SPREAD_X, SPREAD_X)
                v = hv + random.uniform(-SPREAD_Y, SPREAD_Y)
                u = sign * min(sign * u, -OWN_HALF)
                v = max(-TOUCH_Y + 0.3, min(TOUCH_Y - 0.3, v))
                if math.hypot(u - bu, v - bv) >= BALL_CLEAR and all(math.hypot(u - pu, v - pv) >= DUCK_CLEAR for pu, pv in placed):
                    break
            placed.append((u, v))
            x, y = fr.world(u, v)
            lobby.place(player(label)["name"], x, y, fr.yaw + hyaw + random.uniform(-SPREAD_YAW, SPREAD_YAW))


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
            self.out_since, self.still = None, None
        self.prev_t = t
        fr, b = Frame(f["pose"]), lobby.body(f["name"] + "/ball")
        if self.landing is not None:  # the Reset landed (time went back, or a few ticks on): place the ducks
            before, ticks = self.landing
            if t < before or ticks >= 5:
                self.landing = None
                self.kick_off(fr, b)
                kickoff["placing"] = False
                self.still = None
            else:
                self.landing = (before, ticks + 1)
        rx, ry = fr.local(b["x"], b["y"])
        for label in self.down_since:
            up = lobby.upright(player(label)["name"]) if player(label) else None
            fallen = up is not None and up < FALL_UP
            self.down_since[label] = (self.down_since[label] if self.down_since[label] is not None else t) if fallen else None
        is_goal = abs(ry) < GOAL_HALF_W and abs(rx) > GOAL_X + 0.03
        is_out = not is_goal and (abs(ry) > TOUCH_Y or abs(rx) > GOAL_X + 0.05)
        self.out_since = (self.out_since if self.out_since is not None else t) if (is_goal or is_out) else None
        if self.still is None or math.hypot(b["x"] - self.still[0], b["y"] - self.still[1]) > STALL_MOVE:
            self.still = (b["x"], b["y"], t)
        if self.quiet_until is None or t >= self.quiet_until:
            if self.out_since is not None and t - self.out_since >= OUT_TIME:
                if is_goal:
                    team = next(team for _, sign, team in TEAMS if sign * rx > 0)
                    score[team] += 1
                    self.restart(f"GOAL {team}", t)
                else:
                    self.restart("ball out", t)
            elif all(d is not None and t - d >= FALL_TIME for d in self.down_since.values()):
                self.restart("both ducks down", t)
            elif t - self.still[2] >= STALL_TIME:
                self.restart("stalemate", t)
        a, b2 = TEAMS[0][2], TEAMS[1][2]
        self.feedback_message = f"SCORE  {a} {score[a]} : {score[b2]} {b2}   (last: {self.last})"
        return Status.RUNNING


root = py_trees.composites.Parallel(
    "Duck football 1v1",
    policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
    children=[Referee()] + [Player(label, sign, team) for label, sign, team in TEAMS],
)
