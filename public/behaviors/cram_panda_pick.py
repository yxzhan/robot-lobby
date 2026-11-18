# CRAM plans, the lobby moves, closed loop: the Panda picks up a cube and puts
# it down elsewhere, each action (park, pick, place) planned by CRAM from what
# the lobby's world is like at that moment, in a Jupyter kernel of a cramera
# Binder session (lobby.Cram starts it, about a minute; loading CRAM there
# some 12 s more).
#
# CRAM's world is built once from the lobby's (lobby_cram.start: the Panda's
# model at its base's pose, the cube's model) and kept. Before each action
# the tree waits for the cube to be at rest and sends the lobby's joint
# positions, the cube's pose and whether the Panda holds it
# (lobby_cram.act); CRAM plans and runs the action in its own kinematic
# simulation and gives back the Panda's actuator targets every 0.05 s, which
# the tree plays back here (the IK off), in physics, ramping the targets
# from tick to tick (lobby.set_actuators(..., ramp)). Where the gripper opens
# or closes, the playback waits for the arm to have got where CRAM has it,
# and then for the fingers to stop; anywhere else for an arm far behind.
# While it plays it watches the cube: moved before the fingers close, or
# slipped out of them while carried, the action stops and fails. After it,
# it checks the outcome: the cube held after a pick, on the target after a
# place. A failed pick is tried again (parked first, planned from where the
# cube is now); a failed place (dropped, or set down too far off) starts
# again from the pick.
import math
import py_trees
from py_trees.common import Status
import lobby

PANDA = "mjcf_assets/robots/mujoco_menagerie/franka_emika_panda/panda.xml"
CUBE = "mjcf_assets/objects/custom/cube/cube.xml"
THINGS = [(PANDA, 0.0, 0.0, 0.0), (CUBE, 0.5, 0.0, 0.0)]
TARGET = (0.45, 0.2, 0.025)  # where the cube's centre goes (m, in the world)
GRASP_HEIGHT = 0.045         # m over the cube's bottom (CRAM's ShapeSortingPiece; it brings the TCP 1.8 cm lower
                             # still: the fingers' reach), so the Panda's fingers close at the cube's middle
NEAR = 0.02                  # m: a place is done with the cube this near TARGET
HELD = 0.04                  # m: the cube this near the TCP (between the fingers): held
MOVED = 0.02                 # m: the cube moved this far during a pick, before the fingers close: start again
REST = 0.002                 # m: the cube moving less than this in a tick (0.1 s) is at rest
ARRIVED = 0.008              # rad: the arm's joints this near CRAM's before the gripper moves
WAIT = 2.0                   # s (simulated): at most that long, for the cube's rest or the arm
TICK = 0.1                   # s (simulated): how often the Behavior panel ticks the tree
LAG = 0.05                   # rad: the playback waits whenever the arm is this far behind CRAM's
GRIP = 0.3                   # s (simulated): after the gripper moved, the playback waits at least this long,
FINGERS_STILL = 0.0003       # m: and until its fingers move less than this in a tick (the Panda's close slowly)
SETTLE = 10                  # steps after the last for the arm to get there
TRIES = 3                    # picks, and picks-and-places, at most
NOSLIP = 10                  # the noslip post-pass's iterations (a held cube slips less)
CONE = "pyramidal"           # the friction cone
plan = {}                    # the arm's and the cube's names


def cube():
    b = lobby.body(f'{plan["cube"]}/cube')
    return [b["x"], b["y"], b["z"], *b["quat"]]


def tcp():
    return lobby.ik_chains(plan["arm"])[0]["tip_pose"]["pos"]


def held():
    return math.dist(cube()[:3], tcp()) < HELD


def joints():
    return {j["name"].split("/")[-1]: j["qpos"] for bd in lobby.inspect(plan["arm"])["bodies"]
            for j in bd.get("joints", []) if isinstance(j.get("qpos"), (int, float))}  # (1-DoF joints)


class Ready(py_trees.behaviour.Behaviour):
    """Notes the Panda and the cube, sets noslip and the cone, and switches the
    Panda's IK off (its joints are CRAM's)."""

    def update(self):
        plan["arm"], plan["cube"] = spawn.names
        ph = lobby.physics()
        if ph.get("noslip_iterations") != NOSLIP or ph.get("cone") != CONE:
            lobby.physics(noslip_iterations=NOSLIP, cone=CONE)
        if lobby.controllers(plan["arm"])["ik"]:
            lobby.set_controller(plan["arm"], "ik", False)
            self.feedback_message = "IK off…"
            return Status.RUNNING
        return Status.SUCCESS


def start_cell():
    """CRAM's world from the lobby's: the Panda at its base, the cube."""
    p = lobby.pose(plan["arm"])
    return lobby.cram_call("start", PANDA, [p["x"], p["y"], 0.0, p["yaw"]], joints(), CUBE, cube(), grasp_height=GRASP_HEIGHT)


class Act(py_trees.behaviour.Behaviour):
    """One action by CRAM: planned from the lobby's state once the cube is at
    rest, played back, watched while it plays, its outcome checked."""

    def __init__(self, action, target=None):
        super().__init__(action)
        self.action, self.target = action, target

    def initialise(self):
        self.job = self.traj = None
        self.last, self.t_rest = cube(), lobby.time()
        self.feedback_message = "waiting for the cube to rest"

    def fail(self, why):
        self.feedback_message = why
        k = int(getattr(self, "k", 0) or 0)
        print(f"{self.action}: {why} (step {k}; the cube at {[round(v, 3) for v in cube()[:3]]}, the TCP at {[round(v, 3) for v in tcp()]})")
        return Status.FAILURE

    def update(self):
        if self.job is None:  # the cube at rest first: then planned from the lobby's state
            c = cube()
            moving = math.dist(c[:3], self.last[:3]) > REST
            self.last = c
            if moving and lobby.time() - self.t_rest < WAIT:
                return Status.RUNNING
            self.held, self.cube0 = held(), c
            self.job = lobby.cram_run(lobby.cram_call("act", self.action, joints(), c, held=self.held,
                                                      target=self.target and list(self.target)))
            self.feedback_message = "planning"
            return Status.RUNNING
        if self.traj is None:  # waiting for CRAM's plan
            r = lobby.cram_result(self.job)
            if r is None:
                return Status.RUNNING
            v = r.get("value") if isinstance(r.get("value"), dict) else {}
            if not (r.get("ok") and v.get("ok")):
                return self.fail("CRAM: " + str(v.get("error") or r.get("error")).splitlines()[0][:150])
            self.traj, self.k, self.t, self.since, self.hold, self.fingers = v, 0.0, lobby.time(), lobby.time(), 0.0, None
            note = " (CRAM's world built anew: the lobby's Panda does not hold the cube)" if v["rebuilt"] else ""
            print(f'{self.action}: {len(v["ctrl"])} steps, planned in {v["secs"]} s{note}')
        t, now = self.traj, lobby.time()
        ctrl, n = t["ctrl"], len(t["ctrl"])
        k = min(int(self.k), n - 1)
        step, done = ctrl[k], int(self.k) >= n + SETTLE  # (the last step's targets held a while for the arm)

        def send(i):  # the targets of step i, reached by the next tick (smoothly: ramped)
            lobby.set_actuators(plan["arm"], dict(zip(t["actuators"], ctrl[i])), ramp=TICK)
        # the next step: once the time for it has come and the arm is not far
        # behind; where the gripper changes, once the arm has got where CRAM has it
        gate = 0 < k < n - 1 and abs(ctrl[k + 1][-1] - step[-1]) > 1
        if now - self.since < WAIT:
            q = joints()
            gap = max(abs(q[j] - u / s) for (j, s), u in zip(t["joints"], step) if s == 1.0 and j in q)
            if gap > (ARRIVED if gate else LAG):
                send(k)
                self.t = now
                self.feedback_message = f"step {k} of {n}: the arm {gap:.3f} rad from CRAM's"
                return Status.RUNNING
        if self.fingers is not None:  # the gripper moved: on once its fingers stopped
            f = joints()[t["joints"][-1][0]]
            moving, self.fingers = abs(f - self.fingers) > FINGERS_STILL, f
            if (moving or now - self.hold < GRIP) and now - self.hold < WAIT:
                send(k)
                self.t = now
                self.feedback_message = f"step {k} of {n}: the fingers"
                return Status.RUNNING
            self.fingers = None
        self.k += (now - self.t) / t["dt"]
        self.t = now
        if int(self.k) != k:
            self.since = now
            if gate:
                self.hold, self.fingers = now, joints()[t["joints"][-1][0]]
        # one tick ahead, but not past the next change of the gripper (waited for there)
        i, last = min(k + 1, n - 1) if gate else k, min(int(self.k) + round(TICK / t["dt"]), n - 1)
        while i < last and abs(ctrl[i + 1][-1] - ctrl[i][-1]) <= 1:
            i += 1
        send(i)
        # watching it
        if self.action == "pick" and step[-1] > 128 and math.dist(cube()[:3], self.cube0[:3]) > MOVED:
            return self.fail("the cube moved before the grasp")  # (fingers still open: the gripper actuator > half)
        if (self.held and self.action == "place" and step[-1] < 128 and not held()
                and math.dist(cube()[:3], self.target) > NEAR):  # (before the release; down on the target already: as good)
            return self.fail("the cube slipped out of the fingers")
        if not done:
            self.feedback_message = f"step {min(k + 1, n)} of {n}"
            return Status.RUNNING
        # its outcome
        if self.action == "pick":
            c, p, q = cube(), tcp(), joints()
            print(f"pick: the cube {[round(a - b, 3) for a, b in zip(c[:3], p)]} from the TCP, tilted "
                  f"{math.degrees(math.acos(min(1, 1 - 2 * (c[4] ** 2 + c[5] ** 2)))):.0f} deg, fingers "
                  f"{q.get('finger_joint1', 0):.3f} {q.get('finger_joint2', 0):.3f}")
            if not held():
                return self.fail("the grasp missed")
        if self.action == "place":
            off = math.dist(cube()[:3], self.target)
            if held() or off > NEAR:
                return self.fail(f"the cube {off * 100:.1f} cm from the target")
            print(f"place: the cube {off * 100:.1f} cm from the target")
        self.feedback_message = "done"
        return Status.SUCCESS


spawn = lobby.Spawn("Panda and cube", THINGS)
pick = py_trees.composites.Sequence("park, pick", memory=True, children=[Act("park"), Act("pick")])
root = py_trees.composites.Sequence("Pick and place by CRAM", memory=True, children=[
    spawn,
    lobby.Cram(),
    Ready("Ready"),
    lobby.CramRun("CRAM's world", start_cell),
    py_trees.decorators.Retry("pick and place", py_trees.composites.Sequence("transport", memory=True, children=[
        py_trees.decorators.Retry("pick, again if it fails", pick, TRIES),
        Act("place", TARGET),
    ]), TRIES),
    Act("park"),
])
