# A py_trees tree (Python, run by Pyodide): r1 patrols between two points
# while r2 follows it, until r1 has done three rounds.
import py_trees
from lobby import Follow, GoTo, Wait

patrol = py_trees.composites.Sequence("Patrol", memory=True, children=[
    GoTo("r1", x=1.0, y=0.0),
    Wait(1.0),
    GoTo("r1", x=-1.0, y=0.0),
    Wait(1.0),
])

rounds = py_trees.decorators.Repeat("Three rounds", child=patrol, num_success=3)
root = py_trees.composites.Parallel(
    "Patrol and follow",
    policy=py_trees.common.ParallelPolicy.SuccessOnSelected(children=[rounds]),
    children=[rounds, Follow("r2", "r1", distance=0.4)],
)
