# A condition of one's own: r1 walks to the ball unless it is near already,
# then waits; over and over (move the ball with Shift+drag).
import py_trees
from py_trees.common import Status
import lobby
from lobby import Approach, Wait


class Near(py_trees.behaviour.Behaviour):
    """SUCCESS while `robot` is within `distance` of `target`."""

    def __init__(self, robot, target, distance):
        super().__init__(f"{robot} near {target}?")
        self.robot, self.target, self.distance = robot, target, distance

    def update(self):
        d = lobby.distance(self.robot, self.target)
        self.feedback_message = f"{d:.2f} m"
        return Status.SUCCESS if d < self.distance else Status.FAILURE


root = py_trees.decorators.Repeat("Forever", num_success=-1, child=py_trees.composites.Sequence("Round", memory=True, children=[
    py_trees.composites.Selector("Get there", memory=False, children=[
        Near("r1", "ball", 0.5),
        Approach("r1", "ball", distance=0.3),
    ]),
    Wait(2.0),
]))
