import py_trees
from lobby import Follow, GoTo

GAP = 1.0  # m each follower keeps from the robot ahead of it

patrol = py_trees.composites.Sequence("Patrol", memory=True, children=[
    GoTo("r2", x=140.0, y=-48.8),
    GoTo("r2", x=140.0, y=46.3),
    GoTo("r2", x=-132.0, y=46.3),
    GoTo("r2", x=-148.0, y=-48.8),
])

root = py_trees.composites.Parallel(
    "Convoy patrol",
    policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
    children=[
        py_trees.decorators.Repeat("For ever", child=patrol, num_success=-1),
        Follow("r3", "r2", distance=GAP),
        Follow("r4", "r3", distance=GAP),
        Follow("r5", "r4", distance=GAP),
    ],
)
