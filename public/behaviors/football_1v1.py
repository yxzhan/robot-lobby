# Duck football 1v1: r2 (white) attacks the +x goal, r3 (graphite) the -x goal.
# Each duck gets behind the ball (on the line ball -> opponent goal), then walks
# through it to push it on. A goal scores a point and resets the kick-off.
import math
import py_trees
from py_trees.common import Status
import js
from pyodide.ffi import to_js

GOAL_X, GOAL_HALF_W = 2.02, 0.40   # goal mouths at x = +-2.02 (field frame)
BEHIND, PUSH = 0.22, 0.45           # set-up distance behind the ball, push-through distance
app = js.window.app
score = {"r2": 0, "r3": 0}


def body_xy(i):
    return app.data.xpos[3 * i], app.data.xpos[3 * i + 1]


def find(prefix=None, label=None):
    for r in app.robots:
        if (prefix and r.prefix == prefix) or (label and label in r.label):
            return r
    return None


def set_goal(name, x, y, yaw):
    js.window.goals.set(name, to_js({"x": x, "y": y, "yaw": yaw}, dict_converter=js.Object.fromEntries))


class Player(py_trees.behaviour.Behaviour):
    def __init__(self, name, attack_sign):
        super().__init__(f"{name} plays")
        self.robot, self.sign, self.last = name, attack_sign, None

    def update(self):
        field, me = find(label="football"), find(prefix=self.robot + "/")
        if not field or not me:
            return Status.FAILURE
        fx, fy = body_xy(field.bodyOffset)
        bx, by = body_xy(field.bodyOffset + 1)          # the ball
        mx, my = body_xy(me.bodyOffset)
        gx, gy = fx + self.sign * GOAL_X, fy            # goal to attack
        ux, uy = gx - bx, gy - by
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        attack_yaw = math.atan2(uy, ux)
        # where the duck stands relative to the ball, along/across the attack line
        dx, dy = mx - bx, my - by
        along, across = dx * ux + dy * uy, -dx * uy + dy * ux
        pushing = self.last is not None and self.last[2] == "push"
        if pushing and along < 0.12 and abs(across) < 0.16:   # keep pushing while still on the ball
            tx, ty = bx + ux * PUSH, by + uy * PUSH
            mode, yaw = "push", attack_yaw
        elif along > -0.08:                     # in front of / beside the ball: go round it
            side = 1.0 if across >= 0 else -1.0
            tx = bx - ux * BEHIND - uy * side * 0.30
            ty = by - uy * BEHIND + ux * side * 0.30
            mode, yaw = "around", attack_yaw
        elif abs(across) > 0.12 or along < -0.40:  # behind, but not lined up: line up
            tx, ty = bx - ux * BEHIND, by - uy * BEHIND
            mode, yaw = "line up", attack_yaw
        else:                                   # lined up: walk through the ball
            tx, ty = bx + ux * PUSH, by + uy * PUSH
            mode, yaw = "push", attack_yaw
        # stay on the pitch
        tx = max(fx - 2.2, min(fx + 2.2, tx))
        ty = max(fy - 1.5, min(fy + 1.5, ty))
        if self.last is None or math.hypot(tx - self.last[0], ty - self.last[1]) > 0.03 or mode != self.last[2]:
            set_goal(self.robot, tx, ty, yaw)
            self.last = (tx, ty, mode)
        self.feedback_message = f"{mode}  ball {math.hypot(dx, dy):.2f} m"
        return Status.RUNNING


def upright(i):
    """z of the body's up axis (1 = standing, 0 = lying on its side)."""
    q = app.data.xquat
    qw, qx, qy, qz = q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]
    return 1 - 2 * (qx * qx + qy * qy)


TOUCH_Y = 1.40      # ball wholly over the touchline (field frame, m)
FALL_UP = 0.5       # up-axis z below this = fallen
FALL_TICKS = 15     # 1.5 s on the ground before it counts


class Referee(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("Referee")
        self.cooldown = 0
        self.down = {"r2": 0, "r3": 0}
        self.last = "kick-off"
        self.out = 0

    def restart(self, why):
        self.last = why
        self.cooldown = 50                      # 5 s at 10 Hz for the reset to land
        self.out = 0
        self.down = {"r2": 0, "r3": 0}
        try:
            app.reset()                         # kick-off again
        except Exception:
            pass

    def update(self):
        field = find(label="football")
        if not field:
            return Status.FAILURE
        fx, fy = body_xy(field.bodyOffset)
        bx, by = body_xy(field.bodyOffset + 1)
        rx, ry = bx - fx, by - fy
        for name in ("r2", "r3"):
            r = find(prefix=name + "/")
            fallen = r is not None and upright(r.bodyOffset) < FALL_UP
            self.down[name] = self.down[name] + 1 if fallen else 0
        is_goal = abs(ry) < GOAL_HALF_W and abs(rx) > GOAL_X + 0.03
        is_out = not is_goal and (abs(ry) > TOUCH_Y or abs(rx) > GOAL_X + 0.05)
        self.out = self.out + 1 if (is_goal or is_out) else 0
        if self.cooldown > 0:
            self.cooldown -= 1
        elif self.out < 3:                      # must hold for 0.3 s
            if all(n >= FALL_TICKS for n in self.down.values()):
                self.restart("both ducks down")
        elif is_goal:
            scorer = "r2" if rx > 0 else "r3"
            score[scorer] += 1
            self.restart(f"GOAL {scorer}")
        else:
            self.restart("ball out")
        self.feedback_message = f"SCORE  white r2 {score['r2']} : {score['r3']} r3 graphite   (last: {self.last})"
        return Status.RUNNING


root = py_trees.composites.Parallel(
    "Duck football 1v1",
    policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
    children=[Referee(), Player("r2", +1), Player("r3", -1)],
)
